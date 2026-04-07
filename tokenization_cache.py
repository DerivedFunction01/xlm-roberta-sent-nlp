from __future__ import annotations

import glob
import json
import os
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from source_config import RUN
from synthetic_cache import CACHE_DIR, CACHE_META, SYNTHETIC_CACHE_META, TOKENIZED_CACHE_VERSION

MAX_LENGTH = RUN["len"]
USE_TOKENIZED_CACHE = RUN["tok_cache"]
FORCE_REBUILD_TOKENIZED_CACHE = RUN["tok_rebuild"]
SKIP_TOKENIZED_CACHE_VALIDATION = RUN["tok_skip_check"]


def tokenize_and_align(
    example: dict,
    *,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    max_length: int = MAX_LENGTH,
) -> dict:
    """Re-encode the pre-tokenized token list and propagate labels."""
    encoding = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    word_ids = encoding.word_ids()
    labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(example["ner_tags"][word_id])
        else:
            orig_label = example["ner_tags"][word_id]
            lang_tag = id2label[orig_label]
            if lang_tag.startswith("B-"):
                i_tag = "I-" + lang_tag[2:]
                labels.append(label2id.get(i_tag, orig_label))
            else:
                labels.append(orig_label)
        prev_word_id = word_id

    encoding["labels"] = labels
    encoding["original_text"] = example.get("original_text", " ".join(example["tokens"]))
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

    DatasetDict({"train": train_dataset, "eval": eval_dataset}).save_to_disk(CACHE_DIR)
    with open(CACHE_META, "w", encoding="utf-8") as f:
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
    if not (os.path.exists(CACHE_DIR) and os.path.exists(CACHE_META)):
        return None

    if SKIP_TOKENIZED_CACHE_VALIDATION:
        try:
            return load_from_disk(CACHE_DIR)
        except Exception:
            split_names = ["train", "eval"]
            loaded_splits = {}
            for split_name in split_names:
                split_dir = os.path.join(CACHE_DIR, split_name)
                arrow_files = sorted(glob.glob(os.path.join(split_dir, "*.arrow")))
                if not arrow_files:
                    return None
                split_parts = [Dataset.from_file(path) for path in arrow_files]
                loaded_splits[split_name] = (
                    split_parts[0] if len(split_parts) == 1 else concatenate_datasets(split_parts)
                )
            return DatasetDict(loaded_splits)

    with open(CACHE_META, encoding="utf-8") as f:
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

    try:
        return load_from_disk(CACHE_DIR)
    except Exception:
        split_names = ["train", "eval"]
        loaded_splits = {}
        for split_name in split_names:
            split_dir = os.path.join(CACHE_DIR, split_name)
            arrow_files = sorted(glob.glob(os.path.join(split_dir, "*.arrow")))
            if not arrow_files:
                return None
            split_parts = [Dataset.from_file(path) for path in arrow_files]
            loaded_splits[split_name] = (
                split_parts[0] if len(split_parts) == 1 else concatenate_datasets(split_parts)
            )
        return DatasetDict(loaded_splits)


def build_tokenized_dataset(
    synthetic_dataset,
    *,
    seed: int,
    model_checkpoint: str,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    max_length: int = MAX_LENGTH,
):
    """Load or build the tokenized train/eval split."""
    cached_tokenized = (
        load_tokenized_dataset_cache(seed=seed, model_checkpoint=model_checkpoint, max_length=max_length)
        if (USE_TOKENIZED_CACHE and not FORCE_REBUILD_TOKENIZED_CACHE)
        else None
    )
    if cached_tokenized is not None:
        print(f"Loaded tokenized dataset cache from {CACHE_DIR}")
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

    coverage_dataset = synthetic_dataset.filter(lambda ex: ex["kind"] == "coverage").map(  # type: ignore
        _decode_synthetic_example,
        batched=False,
        remove_columns=["kind"],
    )
    random_dataset = synthetic_dataset.filter(lambda ex: ex["kind"] == "random").map(  # type: ignore
        _decode_synthetic_example,
        batched=False,
        remove_columns=["kind"],
    )

    coverage_dataset = coverage_dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer=tokenizer, label2id=label2id, id2label=id2label),
        batched=False,
        remove_columns=["tokens", "ner_tags"],
    )
    random_dataset = random_dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer=tokenizer, label2id=label2id, id2label=id2label),
        batched=False,
        remove_columns=["tokens", "ner_tags"],
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
