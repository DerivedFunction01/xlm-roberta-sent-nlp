from __future__ import annotations

import glob
import json
import os
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from datasets import enable_progress_bar

from source_config import RUN
from paths import PATHS

MAX_LENGTH = RUN["len"]
USE_TOKENIZED_CACHE = RUN["tok_cache"]
FORCE_REBUILD_TOKENIZED_CACHE = RUN["tok_rebuild"]
SKIP_TOKENIZED_CACHE_VALIDATION = RUN["tok_skip_check"]
TOKENIZED_CACHE_VERSION = PATHS["versions"]["tokenized"]
SYNTHETIC_CACHE_META = PATHS["synthetic"]["cache_meta"]

def _load_dataset_dict_from_cache_dir(cache_dir: str):
    """Load a DatasetDict from a saved cache directory or its Arrow shards."""
    try:
        return load_from_disk(cache_dir)
    except Exception:
        split_names = ["train", "eval"]
        loaded_splits = {}
        for split_name in split_names:
            split_dir = os.path.join(cache_dir, split_name)
            arrow_files = sorted(glob.glob(os.path.join(split_dir, "*.arrow")))
            if not arrow_files:
                return None
            split_parts = [Dataset.from_file(path) for path in arrow_files]
            loaded_splits[split_name] = (
                split_parts[0] if len(split_parts) == 1 else concatenate_datasets(split_parts)
            )
        return DatasetDict(loaded_splits)


def tokenize_and_align(
    example: dict,
    *,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    max_length: int = MAX_LENGTH,
) -> dict:
    """Tokenize mutated tokens and align labels.
    
    The training data has mutated tokens (swapped, with code/noise injected, 
    punctuation stripped, etc.), NOT the original text. We reconstruct text
    from these mutated tokens to ensure training matches inference.
    
    This avoids double-tokenization that occurs when passing subword tokens
    with is_split_into_words=True.
    """
    # Reconstruct text from the MUTATED subword tokens
    # This preserves all mutations (swaps, code injection, noise, etc.)
    mutated_tokens = example["tokens"]
    reconstructed_text = tokenizer.convert_tokens_to_string(mutated_tokens)
    
    # Now tokenize WITH special tokens for the model input
    encoding = tokenizer(
        reconstructed_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,  # Include [CLS] and [SEP]
    )
    
    # Build label list: -100 for special tokens, then the ner_tags
    num_special_before = tokenizer.num_special_tokens_to_add(pair=False) // 2
    labels = [-100] * num_special_before  # [CLS] token(s)
    
    # Add labels for content tokens (truncate if necessary due to max_length)
    ner_tags = example["ner_tags"]
    num_content_tokens = len(encoding["input_ids"]) - num_special_before - 1  # -1 for [SEP]
    labels.extend(ner_tags[:num_content_tokens])
    
    # Add [SEP] and padding labels
    labels.extend([-100] * (len(encoding["input_ids"]) - len(labels)))
    
    encoding["labels"] = labels
    # Store reconstructed text for reference (this is what was actually used)
    encoding["original_text"] = reconstructed_text
    return encoding


def save_tokenized_dataset_cache(
    train_dataset,
    eval_dataset,
    *,
    seed: int,
    model_checkpoint: str,
    max_length: int = MAX_LENGTH,
) -> None:
    """Persist the tokenized train/eval split to disk."""
    synthetic_meta = None
    if os.path.exists(SYNTHETIC_CACHE_META):
        with open(SYNTHETIC_CACHE_META, encoding="utf-8") as f:
            synthetic_meta = json.load(f)

    cache_dir = PATHS["tokenized"]["cache_dir"]
    cache_meta = PATHS["tokenized"]["cache_meta"]
    DatasetDict({"train": train_dataset, "eval": eval_dataset}).save_to_disk(cache_dir)
    with open(cache_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_version": TOKENIZED_CACHE_VERSION,
                "model_checkpoint": model_checkpoint,
                "max_length": max_length,
                "seed": seed,
                "synthetic_cache_meta": synthetic_meta,
            },
            f,
            indent=2,
        )


def load_tokenized_dataset_cache(
    *,
    seed: int,
    model_checkpoint: str,
    max_length: int = MAX_LENGTH,
) -> Any | None:
    """Load tokenized train/eval split if the metadata matches the current config."""
    cache_dir = PATHS["tokenized"]["cache_dir"]
    cache_meta = PATHS["tokenized"]["cache_meta"]
    if not (os.path.exists(cache_dir) and os.path.exists(cache_meta)):
        return None

    if SKIP_TOKENIZED_CACHE_VALIDATION:
        return _load_dataset_dict_from_cache_dir(cache_dir)

    with open(cache_meta, encoding="utf-8") as f:
        meta = json.load(f)

    synthetic_meta = None
    if os.path.exists(SYNTHETIC_CACHE_META):
        with open(SYNTHETIC_CACHE_META, encoding="utf-8") as f:
            synthetic_meta = json.load(f)

    expected_meta = {
        "cache_version": TOKENIZED_CACHE_VERSION,
        "model_checkpoint": model_checkpoint,
        "max_length": max_length,
        "seed": seed,
        "synthetic_cache_meta": synthetic_meta,
    }
    if meta != expected_meta:
        return None

    return _load_dataset_dict_from_cache_dir(cache_dir)


def load_tokenized_dataset_splits() -> Any | None:
    """Load the cached tokenized train/eval split without validating metadata."""
    cache_dir = PATHS["tokenized"]["cache_dir"]
    if not os.path.exists(cache_dir):
        return None
    return _load_dataset_dict_from_cache_dir(cache_dir)


def build_tokenized_dataset(
    synthetic_dataset,
    *,
    seed: int = 42,
    model_checkpoint: str,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    max_length: int = MAX_LENGTH,
    num_samples: int | None = None,
):
    """Load or build the tokenized train/eval split."""
    enable_progress_bar()
    cached_tokenized = (
        load_tokenized_dataset_cache(seed=seed, model_checkpoint=model_checkpoint, max_length=max_length)
        if (USE_TOKENIZED_CACHE and not FORCE_REBUILD_TOKENIZED_CACHE)
        else None
    )
    if cached_tokenized is None and USE_TOKENIZED_CACHE and not FORCE_REBUILD_TOKENIZED_CACHE:
        cached_tokenized = load_tokenized_dataset_splits()
    if cached_tokenized is not None:
        print(f"Loaded tokenized dataset cache from {PATHS['tokenized']['cache_dir']}")
        return cached_tokenized["train"], cached_tokenized["eval"]

    if synthetic_dataset is None:
        raise RuntimeError(
            "Synthetic examples are not available in memory or cache. "
            "Run the generation cell first, or enable USE_SYNTHETIC_CACHE."
        )

    def _decode_synthetic_example(example: dict) -> dict:
        return {
            "original_text": example.get("original_text", ""),
            "tokens": json.loads(example["tokens"]),
            "ner_tags": json.loads(example["ner_tags"]),
        }

    coverage_dataset = synthetic_dataset.filter(  # type: ignore
        lambda ex: ex["kind"] in {"coverage", "pure", "homogeneous"},
        desc="Filtering coverage split",
    ).map(
        _decode_synthetic_example,
        batched=False,
        remove_columns=["kind"],
        desc="Decoding coverage split",
    )
    random_dataset = synthetic_dataset.filter(  # type: ignore
        lambda ex: ex["kind"] in {"random", "mixed"},
        desc="Filtering random split",
    ).map(
        _decode_synthetic_example,
        batched=False,
        remove_columns=["kind"],
        desc="Decoding random split",
    )

    # Limit dataset size for testing/quick iteration if specified
    if num_samples is not None:
        coverage_size = int(num_samples * 0.7)  # ~70% from coverage
        random_size = num_samples - coverage_size  # ~30% from random
        coverage_dataset = coverage_dataset.select(range(min(coverage_size, len(coverage_dataset))))
        random_dataset = random_dataset.select(range(min(random_size, len(random_dataset))))

    coverage_dataset = coverage_dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer=tokenizer, label2id=label2id, id2label=id2label),
        batched=False,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenizing coverage split",
    )
    random_dataset = random_dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer=tokenizer, label2id=label2id, id2label=id2label),
        batched=False,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenizing random split",
    )

    split = random_dataset.train_test_split(test_size=0.05, seed=seed)
    train_dataset = concatenate_datasets([coverage_dataset, split["train"]])
    eval_dataset = split["test"]
    if USE_TOKENIZED_CACHE:
        save_tokenized_dataset_cache(
            train_dataset,
            eval_dataset,
            seed=seed,
            model_checkpoint=model_checkpoint,
            max_length=max_length,
        )

    return train_dataset, eval_dataset
