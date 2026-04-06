# %% [markdown]
# # Multilingual Language Detection via Sentence-NER (Token Classification)
# Fine-tunes XLM-RoBERTa to tag each token with its source language (BIO scheme),
# enabling transparent, evidence-based language identification.

# %%
# --- Environment Setup ---
# pip install evaluate pysbd faker seqeval
# %%
import random
import codecs
import re
import json
import gc
import multiprocessing as mp
import unicodedata
import traceback
from collections import defaultdict, deque
import string
from faker import Faker
import torch
import numpy as np
import evaluate
from datasets import (
    load_dataset,
    get_dataset_config_names,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)

import os
import glob
import pandas as pd
from pathlib import Path
from huggingface_hub import login
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_CHECKPOINT = "xlm-roberta-base"
MAX_LENGTH = 512
EXAMPLES_TARGET = 2_000_000  # synthetic mixed-language training examples to generate
MIN_COVERAGE_DOCS_PER_LANG = 2
MAX_COVERAGE_DOCS_PER_LANG = 5
USE_SYNTHETIC_CACHE = True
FORCE_REBUILD_SYNTHETIC_CACHE = False
USE_TOKENIZED_CACHE = True
FORCE_REBUILD_TOKENIZED_CACHE = False
SKIP_TOKENIZED_CACHE_VALIDATION = False
GENERATION_WORKERS = mp.cpu_count() // 4

# Optional notebook-state placeholders.
# These let later cells run even if the generation cell was skipped.
lang_sentences: dict[str, list[str]] | None = None
smol_sentences: dict[str, list[str]] | None = None
ft_sentences: dict[str, list[str]] | None = None
reserved_sentence_pools: dict[str, deque[str]] | None = None
main_sentence_pools: dict[str, deque[str]] | None = None
coverage_plan: list[str] | None = None
reserved_worker_pools: list[dict[str, deque[str]]] | None = None
main_worker_pools: list[dict[str, deque[str]]] | None = None
# %%
# --- Project Imports ---
from paths import SENTENCES_DIR
from source_config import (
    FT_FORCE_REBUILD,
    FT_INCLUDE_TRANSLATED_ENGLISH,
    FT_MAX_RESERVED_SENTENCES,
    FT_MAX_SENTENCES_PER_LANG,
    FT_MIN_RESERVED_SENTENCES,
    FT_RESERVE_FRACTION,
    USE_FINETRANS_AUGMENTATION,
    MAX_RESERVED_SENTENCES,
    MIN_RESERVED_SENTENCES,
    RESERVE_FRACTION,
    SMOL_FORCE_REBUILD,
    SMOL_MAX_RESERVED_SENTENCES,
    SMOL_MIN_RESERVED_SENTENCES,
    SMOL_RESERVE_FRACTION,
    USE_SMOL_AUGMENTATION,
)
from language import ALL_LANGS, LANG_TO_GROUP, LANGUAGE_GROUPS, LANGUAGE_GROUP_WEIGHTS
from wiki_sources import ARTICLES_PER_LANG
from source_pools import (
    build_source_sentence_pools,
    chunk_list,
    draw_sentence,
    partition_sentence_pools,
    remaining_sentence_count,
)
from wiki_sources import finalize_wiki_sentence_cache, load_wiki_sentences
from smol_sources import load_smol_sentences
from finetranslations_sources import load_finetranslations_sentences
from io_utils import write_json_atomic
from neutral_sources import build_neutral_sources
from synthetic_cache import (
    CACHE_DIR,
    CACHE_META,
    CACHE_VERSION,
    SYNTHETIC_CACHE,
    SYNTHETIC_CACHE_META,
    SYNTHETIC_TEMP_DIR,
    TOKENIZED_CACHE_VERSION,
    _append_synthetic_rows,
    _clear_synthetic_cache_dir,
    _load_synthetic_examples_dataset,
    _move_synthetic_shard,
    _synthetic_example_to_row,
    _synthetic_row_to_example,
    _synthetic_rows_to_table,
    _synthetic_worker_temp_path,
)
# %%
# Build BIO label map  (O=0, B-XX=odd, I-XX=even starting at 2)
label2id = {"O": 0}
id2label = {0: "O"}
for idx, lang in enumerate(ALL_LANGS):
    b_id = 2 * idx + 1
    i_id = 2 * idx + 2
    label2id[f"B-{lang.upper()}"] = b_id
    label2id[f"I-{lang.upper()}"] = i_id
    id2label[b_id] = f"B-{lang.upper()}"
    id2label[i_id] = f"I-{lang.upper()}"

NUM_LABELS = len(label2id)
print(f"Total labels: {NUM_LABELS}")
print(f"Total languages: {len(ALL_LANGS)}")
print("Sample:", dict(list(id2label.items())[:7]))
# %%
if Path("hf_token").exists():
    with open("hf_token") as f:
        token = f.read().strip()
    login(token=token)
    print("Logged in to Hugging Face Hub")
# %%
# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
# %%
# --- Data Loading ---
MAX_WIKI_WORKERS = max(1, mp.cpu_count() // 2)
lang_sentences = load_wiki_sentences(
    ALL_LANGS,
    lang_to_group=LANG_TO_GROUP,
    seed=SEED,
    sentences_dir=SENTENCES_DIR,
    articles_per_lang=ARTICLES_PER_LANG,
    max_workers=MAX_WIKI_WORKERS,
)
lang_sentences = finalize_wiki_sentence_cache(lang_sentences, lang_to_group=LANG_TO_GROUP)

smol_sentences = None
if USE_SMOL_AUGMENTATION:
    try:
        smol_sentences = load_smol_sentences(
            sentences_dir=SENTENCES_DIR,
            lang_to_group=LANG_TO_GROUP,
            force_rebuild=SMOL_FORCE_REBUILD,
            seed=SEED,
        )
        total_smol_sentences = sum(len(v) for v in smol_sentences.values())
        print(
            f"\nSMOL kept separate for pool split: "
            f"{len(smol_sentences)} languages | {total_smol_sentences} sentences"
        )
    except Exception as exc:
        print(f"\nSMOL augmentation skipped: {exc}")

ft_sentences = None
if USE_FINETRANS_AUGMENTATION:
    try:
        ft_sentences = load_finetranslations_sentences(
            sentences_dir=SENTENCES_DIR,
            lang_to_group=LANG_TO_GROUP,
            force_rebuild=FT_FORCE_REBUILD,
            seed=SEED,
            max_sentences_per_lang=FT_MAX_SENTENCES_PER_LANG,
            include_translated_english=FT_INCLUDE_TRANSLATED_ENGLISH,
            max_workers=MAX_WIKI_WORKERS,
        )
        total_ft_sentences = sum(len(v) for v in ft_sentences.values())
        print(
            f"\nFineTranslations kept separate for pool split: "
            f"{len(ft_sentences)} languages | {total_ft_sentences} sentences"
        )
    except Exception as exc:
        print(f"\nFineTranslations augmentation skipped: {exc}")

neutral_sources = build_neutral_sources(
    sentences_dir=SENTENCES_DIR,
    english_seed_sentences=(
        lang_sentences.get("en", [])
        + (ft_sentences.get("en", []) if ft_sentences else [])
    ),
    seed=SEED,
)
latex_formulas = neutral_sources.latex_formulas
synth_math_pool = neutral_sources.synth_math_pool
html_noise_pool = neutral_sources.html_noise_pool
css_noise_pool = neutral_sources.css_noise_pool
code_noise_pool = neutral_sources.code_noise_pool
noise_pool = neutral_sources.noise_pool
gibberish_pool = neutral_sources.gibberish_pool
sample_o_span = neutral_sources.sample_o_span
sample_code_span = neutral_sources.sample_code_span

# %%
# --- Synthetic Document Mixer ---


def bio_label_tokens(tokens: list[str], lang: str, is_first: bool) -> list[int]:
    """Assign BIO labels to a token sequence for a given language."""
    labels = []
    for j, _ in enumerate(tokens):
        if j == 0 and is_first:
            labels.append(label2id[f"B-{lang.upper()}"])
        elif j == 0:
            labels.append(label2id[f"B-{lang.upper()}"])
        else:
            labels.append(label2id[f"I-{lang.upper()}"])
    return labels


def augment_boundary(tokens: list[str], strip_punct: bool) -> list[str]:
    """Optionally remove sentence-final punctuation to simulate no-boundary code-switching."""
    if strip_punct and tokens:
        tokens = [t for t in tokens if t not in [".", "!", "?", "▁.", "▁!", "▁?"]]
    return tokens


def swap_random_tokens(tokens: list[str], labels: list[int], swap_rate: float = 0.02):
    """Randomly swap tokens between positions to simulate within-sentence code-switching."""
    n = len(tokens)
    if n < 2:
        return tokens, labels
    n_swaps = max(1, int(n * swap_rate))
    for _ in range(n_swaps):
        i, j = random.sample(range(n), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
        labels[i], labels[j] = labels[j], labels[i]
    return tokens, labels


def generate_synthetic_examples_chunk(
    worker_idx: int,
    coverage_langs: list[str],
    random_count: int,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None,
    synthetic_temp_dir: str,
) -> str:
    """
    Generate a chunk of synthetic examples in one worker.

    Each worker gets a deterministic seed offset and its own pool shard so that
    sentence reuse stays low until the local shard is exhausted.
    """
    seed = SEED + (worker_idx * 10_000)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    worker_desc = f"Worker {worker_idx}"
    temp_path = _synthetic_worker_temp_path(synthetic_temp_dir, worker_idx)
    batch_rows: list[dict] = []
    coverage_count = 0
    random_written = 0

    if pq is None:
        raise RuntimeError("pyarrow is required for streaming synthetic shard writes")

    writer = pq.ParquetWriter(
        temp_path,
        _synthetic_rows_to_table(
            [
                {
                    "kind": "coverage",
                    "original_text": "",
                    "tokens": "",
                    "ner_tags": "",
                }
            ]
        ).schema,
    )
    try:
        total_jobs = len(coverage_langs) + random_count
        with tqdm(total=total_jobs, desc=worker_desc, position=worker_idx, leave=False) as pbar:
            for lang in coverage_langs:
                example = build_synthetic_doc_with_retry(
                    primary_pool,
                    fallback_pool=fallback_pool,
                    required_langs=[lang],
                    worker_idx=worker_idx,
                )
                batch_rows.append(_synthetic_example_to_row("coverage", example))
                coverage_count += 1
                pbar.update(1)
                pbar.set_postfix_str(f"coverage={coverage_count} random={random_written}")
                if len(batch_rows) >= 64:
                    _append_synthetic_rows(writer, batch_rows)
                    batch_rows.clear()

            for _ in range(random_count):
                example = build_synthetic_doc_with_retry(
                    primary_pool,
                    fallback_pool=fallback_pool,
                    worker_idx=worker_idx,
                )
                batch_rows.append(_synthetic_example_to_row("random", example))
                random_written += 1
                pbar.update(1)
                pbar.set_postfix_str(f"coverage={coverage_count} random={random_written}")
                if len(batch_rows) >= 64:
                    _append_synthetic_rows(writer, batch_rows)
                    batch_rows.clear()

        _append_synthetic_rows(writer, batch_rows)
    finally:
        writer.close()

    write_json_atomic(
        temp_path.replace(".parquet", ".meta.json"),
        {
            "worker_idx": worker_idx,
            "seed": seed,
            "job_count": len(coverage_langs) + random_count,
            "coverage_count": coverage_count,
            "random_count": random_written,
        },
    )
    return temp_path


def save_synthetic_examples_cache(
    shard_paths: list[str],
    total_examples: int,
) -> None:
    """Persist synthetic shard metadata after the worker parquet files are published."""
    with open(SYNTHETIC_CACHE_META, "w") as f:
        json.dump(
            {
                "cache_version": CACHE_VERSION,
                "examples_target": EXAMPLES_TARGET,
                "reserve_fraction": RESERVE_FRACTION,
                "min_reserved_sentences": MIN_RESERVED_SENTENCES,
                "max_reserved_sentences": MAX_RESERVED_SENTENCES,
                "min_coverage_docs_per_lang": MIN_COVERAGE_DOCS_PER_LANG,
                "max_coverage_docs_per_lang": MAX_COVERAGE_DOCS_PER_LANG,
                "total_examples": total_examples,
                "shards": [os.path.basename(p) for p in shard_paths],
            },
            f,
            indent=2,
        )


def load_synthetic_examples_cache():
    """Load cached synthetic examples as a parquet-backed dataset if metadata matches."""
    if not os.path.exists(SYNTHETIC_CACHE_META):
        return None

    with open(SYNTHETIC_CACHE_META) as f:
        meta = json.load(f)

    expected_fields = {
        "cache_version": CACHE_VERSION,
        "examples_target": EXAMPLES_TARGET,
        "reserve_fraction": RESERVE_FRACTION,
        "min_reserved_sentences": MIN_RESERVED_SENTENCES,
        "max_reserved_sentences": MAX_RESERVED_SENTENCES,
        "min_coverage_docs_per_lang": MIN_COVERAGE_DOCS_PER_LANG,
        "max_coverage_docs_per_lang": MAX_COVERAGE_DOCS_PER_LANG,
    }
    for key, expected_value in expected_fields.items():
        if meta.get(key) != expected_value:
            return None

    shard_names = meta.get("shards")
    if not isinstance(shard_names, list) or not shard_names:
        return None

    shard_paths = [os.path.join(SYNTHETIC_CACHE, shard_name) for shard_name in shard_names]
    if any(not os.path.exists(path) for path in shard_paths):
        return None

    return _load_synthetic_examples_dataset(SYNTHETIC_CACHE)


def create_synthetic_doc(
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
    latex_pool: list[str] | None = None,
    required_langs: list[str] | None = None,
    o_inject_prob: float = 0.4,   # P(inserting at least one O-label span)
    n_segments: int = 4,
    strip_punct_prob: float = 0.5,
    swap_prob: float = 0.3,
) -> dict:
    """
    Sampling strategy:
      - Groups are weighted by their "global footprint" so major language buckets
        appear more often than tail groups.
      - Languages are chosen from what is still available in the pools, so
        depleted languages naturally stop appearing.
      - `required_langs` can force coverage so every language appears at least
        once in the overall synthetic corpus.
    """
    GROUP_WEIGHTS = LANGUAGE_GROUP_WEIGHTS
    chosen_langs: list[str] = []
    seen_langs: set[str] = set()

    def _candidate_langs() -> list[str]:
        candidates = [
            lang for lang in ALL_LANGS
            if remaining_sentence_count(lang, primary_pool, fallback_pool) > 0
            and lang not in seen_langs
        ]
        return candidates

    def _sample_language(candidates: list[str]) -> str | None:
        if not candidates:
            return None
        weights = [
            GROUP_WEIGHTS.get(LANG_TO_GROUP.get(lang, ""), 1.0)
            * remaining_sentence_count(lang, primary_pool, fallback_pool)
            for lang in candidates
        ]
        return random.choices(candidates, weights=weights, k=1)[0]

    # Always keep any requested languages. This is the coverage guarantee.
    for lang in required_langs or []:
        if remaining_sentence_count(lang, primary_pool, fallback_pool) > 0 and lang not in seen_langs:
            chosen_langs.append(lang)
            seen_langs.add(lang)

    # Sample remaining segments from the languages that still have usable sentence supply.
    n_remaining_segments = max(0, n_segments - len(chosen_langs))
    for _ in range(n_remaining_segments):
        lang = _sample_language(_candidate_langs())
        if lang is None:
            break
        if lang not in seen_langs:
            chosen_langs.append(lang)
            seen_langs.add(lang)

    all_tokens, all_labels = [], []
    original_text_parts: list[str] = []
    total_tokens = 0

    for lang in chosen_langs:
        if total_tokens >= MAX_LENGTH - 20:
            break
        sent = draw_sentence(lang, primary_pool, fallback_pool)
        if sent is None:
            continue
        original_text_parts.append(sent)
        tokens = tokenizer.tokenize(sent)
        if not tokens:
            continue

        strip = random.random() < strip_punct_prob
        tokens = augment_boundary(tokens, strip_punct=strip)

        labels = bio_label_tokens(tokens, lang, is_first=(len(all_tokens) == 0))

        if random.random() < swap_prob:
            tokens, labels = swap_random_tokens(tokens[:], labels[:])

        # Trim to fit within MAX_LENGTH
        remaining = MAX_LENGTH - 2 - total_tokens  # reserve [CLS] and [SEP]
        tokens = tokens[:remaining]
        labels = labels[:remaining]

        all_tokens.extend(tokens)
        all_labels.extend(labels)
        total_tokens += len(tokens)

    # --- O-label span injection ---
    # Sample from the combined pool (LaTeX / synthetic math / symbol noise).
    # Allow 1-3 injections per doc so the model sees O spans in varied positions.
    if len(chosen_langs) == 1 and total_tokens < MAX_LENGTH - 20:
        span = sample_code_span()
        original_text_parts.append(span)
        code_tokens = tokenizer.tokenize(span)
        remaining = MAX_LENGTH - 2 - len(all_tokens)
        code_tokens = code_tokens[:min(remaining, 120)]  # code snippets can be longer than other O spans
        if code_tokens:
            insert_pos = random.randint(0, len(all_tokens))
            all_tokens = all_tokens[:insert_pos] + code_tokens + all_tokens[insert_pos:]
            all_labels = all_labels[:insert_pos] + [0] * len(code_tokens) + all_labels[insert_pos:]
            total_tokens += len(code_tokens)

    if random.random() < o_inject_prob and total_tokens < MAX_LENGTH - 20:
        n_injections = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(n_injections):
            if total_tokens >= MAX_LENGTH - 10:
                break
            span = sample_o_span()
            original_text_parts.append(span)
            o_tokens = tokenizer.tokenize(span)
            remaining = MAX_LENGTH - 2 - len(all_tokens)
            o_tokens = o_tokens[:min(remaining, 40)]  # cap single span at 40 tokens
            if o_tokens:
                insert_pos = random.randint(0, len(all_tokens))
                all_tokens = all_tokens[:insert_pos] + o_tokens + all_tokens[insert_pos:]
                all_labels = all_labels[:insert_pos] + [0] * len(o_tokens) + all_labels[insert_pos:]
                total_tokens += len(o_tokens)

    return {"original_text": " ".join(original_text_parts).strip(), "tokens": all_tokens, "ner_tags": all_labels}


SYNTHETIC_DOC_RETRY_LIMIT = 8

def build_synthetic_doc_with_retry(
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
    required_langs: list[str] | None = None,
    worker_idx: int = 0,
    max_retries: int = SYNTHETIC_DOC_RETRY_LIMIT,
) -> dict:
    """Build a synthetic doc and retry if it is malformed or too long for the model."""
    for attempt in range(1, max_retries + 1):
        example = create_synthetic_doc(
            primary_pool,
            fallback_pool=fallback_pool,
            required_langs=required_langs,
        )
        if len(example.get("tokens", ())) != len(example.get("ner_tags", ())):
            continue
        try:
            encoded = tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
                return_overflowing_tokens=True,
            )
        except Exception:
            continue
        if encoded.get("overflowing_tokens"):
            continue
        return example

    raise RuntimeError(
        f"Worker {worker_idx}: failed to build a valid synthetic doc after "
        f"{max_retries} attempts"
    )


print("Generating synthetic mixed-language documents …")
synthetic_dataset = load_synthetic_examples_cache() if (USE_SYNTHETIC_CACHE and not FORCE_REBUILD_SYNTHETIC_CACHE) else None
if synthetic_dataset is not None:
    print(f"Loaded cached synthetic examples from {SYNTHETIC_CACHE}")
else:
    source_specs = [
        {
            "name": "wiki",
            "sentence_map": lang_sentences,
            "reserve_fraction": RESERVE_FRACTION,
            "min_reserved": MIN_RESERVED_SENTENCES,
            "max_reserved": MAX_RESERVED_SENTENCES,
        },
        {
            "name": "smol",
            "sentence_map": smol_sentences,
            "reserve_fraction": SMOL_RESERVE_FRACTION,
            "min_reserved": SMOL_MIN_RESERVED_SENTENCES,
            "max_reserved": SMOL_MAX_RESERVED_SENTENCES,
        },
        {
            "name": "finetranslations",
            "sentence_map": ft_sentences,
            "reserve_fraction": FT_RESERVE_FRACTION,
            "min_reserved": FT_MIN_RESERVED_SENTENCES,
            "max_reserved": FT_MAX_RESERVED_SENTENCES,
        },
    ]
    reserved_sentence_pools, main_sentence_pools, source_summaries = build_source_sentence_pools(source_specs)
    for summary in source_summaries:
        if summary["skipped"]:
            continue
        print(
            f"{summary['name'].upper()} split -> "
            f"reserved: {summary['reserved']} | main: {summary['main']}"
        )

    reserved_total = sum(len(pool) for pool in reserved_sentence_pools.values())
    main_total = sum(len(pool) for pool in main_sentence_pools.values())
    print(
        f"Reserved sentence bags: {reserved_total} total | "
        f"Main sentence bags: {main_total} total"
    )

    missing_coverage_langs = [lang for lang in ALL_LANGS if not lang_sentences.get(lang)]
    if missing_coverage_langs:
        print("WARNING: no extracted sentences for:", ", ".join(missing_coverage_langs))

    coverage_plan = []
    for lang in ALL_LANGS:
        if not lang_sentences.get(lang):
            continue
        reserved_n = len(reserved_sentence_pools.get(lang, []))
        coverage_docs_for_lang = max(
            1,
            min(MAX_COVERAGE_DOCS_PER_LANG, max(MIN_COVERAGE_DOCS_PER_LANG, (reserved_n + 3) // 4)),
        )
        coverage_plan.extend([lang] * coverage_docs_for_lang)

    # The raw sentence maps are no longer needed once the pools and coverage
    # plan are built, so drop them before the worker fan-out.
    lang_sentences = None
    smol_sentences = None

    _clear_synthetic_cache_dir(SYNTHETIC_CACHE)
    random_job_count = max(0, EXAMPLES_TARGET - len(coverage_plan))
    generation_workers = min(GENERATION_WORKERS, max(1, EXAMPLES_TARGET))
    shard_paths: list[str] = []
    synthetic_total_examples = 0
    coverage_chunks = chunk_list(coverage_plan, generation_workers)
    random_counts = [
        random_job_count // generation_workers + (1 if i < (random_job_count % generation_workers) else 0)
        for i in range(generation_workers)
    ]

    if generation_workers == 1:
        temp_path = generate_synthetic_examples_chunk(
            0,
            coverage_plan,
            random_job_count,
            reserved_sentence_pools,
            main_sentence_pools,
            SYNTHETIC_TEMP_DIR,
        )
        final_path = _move_synthetic_shard(temp_path, 0, synthetic_cache_dir=SYNTHETIC_CACHE)
        shard_paths.append(final_path)
        with open(final_path.replace(".parquet", ".meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        synthetic_total_examples += int(meta.get("coverage_count", 0)) + int(meta.get("random_count", 0))
    else:
        reserved_worker_pools = partition_sentence_pools(reserved_sentence_pools, generation_workers)
        main_worker_pools = partition_sentence_pools(main_sentence_pools, generation_workers)

        with ProcessPoolExecutor(max_workers=generation_workers) as pool:
            future_to_worker = {}
            for worker_idx in range(generation_workers):
                coverage_langs = coverage_chunks[worker_idx] if worker_idx < len(coverage_chunks) else []
                future = pool.submit(
                    generate_synthetic_examples_chunk,
                    worker_idx,
                    coverage_langs,
                    random_counts[worker_idx],
                    reserved_worker_pools[worker_idx],
                    main_worker_pools[worker_idx],
                    SYNTHETIC_TEMP_DIR,
                )
                future_to_worker[future] = worker_idx

            for future in tqdm(as_completed(future_to_worker), total=len(future_to_worker), desc="Synthetic docs"):
                temp_path = future.result()
                worker_idx = future_to_worker[future]
                final_path = _move_synthetic_shard(temp_path, worker_idx, synthetic_cache_dir=SYNTHETIC_CACHE)
                shard_paths.append(final_path)
                with open(final_path.replace(".parquet", ".meta.json"), encoding="utf-8") as f:
                    meta = json.load(f)
                synthetic_total_examples += int(meta.get("coverage_count", 0)) + int(meta.get("random_count", 0))

    if USE_SYNTHETIC_CACHE:
        save_synthetic_examples_cache(shard_paths, synthetic_total_examples)

    synthetic_dataset = _load_synthetic_examples_dataset(SYNTHETIC_CACHE)

if synthetic_dataset is None:
    raise RuntimeError("Synthetic dataset cache could not be loaded.")

print(f"Generated {len(synthetic_dataset)} examples")
sample_example = _synthetic_row_to_example(synthetic_dataset[0])
print("Sample tokens:", sample_example["tokens"][:12])
print("Sample labels:", [id2label[l] for l in sample_example["ner_tags"][:12]])


def print_sampling_stats(examples: list[dict], top_n: int = 12) -> None:
    """Print per-language and per-group coverage stats for the synthetic corpus."""
    example_lang_counts = defaultdict(int)
    token_lang_counts = defaultdict(int)
    group_example_counts = defaultdict(int)
    group_token_counts = defaultdict(int)

    for example in examples:
        langs_in_example: set[str] = set()
        for tag_id in example["ner_tags"]:
            if tag_id == 0:
                continue
            lang = id2label[tag_id][2:].lower()
            token_lang_counts[lang] += 1
            group = LANG_TO_GROUP.get(lang, "Unknown")
            group_token_counts[group] += 1
            langs_in_example.add(lang)

        for lang in langs_in_example:
            example_lang_counts[lang] += 1
            group = LANG_TO_GROUP.get(lang, "Unknown")
            group_example_counts[group] += 1

    missing_langs = [lang for lang in ALL_LANGS if example_lang_counts[lang] == 0]

    print("\nSampling stats")
    print("-" * 72)
    print(f"Examples: {len(examples)}")
    print(f"Languages covered: {len(ALL_LANGS) - len(missing_langs)}/{len(ALL_LANGS)}")
    if missing_langs:
        print("Missing languages:", ", ".join(missing_langs))
    else:
        print("Missing languages: none")

    print("\nPer-language coverage (examples containing the language):")
    for lang, count in sorted(example_lang_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]:
        print(f"  {lang:<3}  {count:>5}")

    print("\nPer-group coverage (examples / tokens):")
    for group in LANGUAGE_GROUPS:
        print(
            f"  {group:<12} "
            f"{group_example_counts[group]:>5} examples | "
            f"{group_token_counts[group]:>7} tokens"
        )

    top_tokens = sorted(token_lang_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    print("\nTop token languages:")
    for lang, count in top_tokens:
        print(f"  {lang:<3}  {count:>7}")


preview_n = min(2_000, len(synthetic_dataset))
preview_examples = [_synthetic_row_to_example(row) for row in synthetic_dataset.select(range(preview_n))]
if preview_examples:
    print_sampling_stats(preview_examples)


def release_wikipedia_generation_memory() -> None:
    """Drop the Wikipedia extraction and sentence-pool artifacts early."""
    for name in [
        "reserved_sentence_pools",
        "main_sentence_pools",
        "lang_sentences",
        "smol_sentences",
        "ft_sentences",
        "neutral_sources",
        "coverage_plan",
        "reserved_worker_pools",
        "main_worker_pools",
        "latex_formulas",
        "synth_math_pool",
        "html_noise_pool",
        "css_noise_pool",
        "code_noise_pool",
        "noise_pool",
        "gibberish_pool",
    ]:
        globals()[name] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


release_wikipedia_generation_memory()

# %%
# --- Label Alignment (sub-token → word-level) ---
# XLM-R uses SentencePiece; the tokenizer produces sub-tokens.
# We already work at the tokenizer sub-token level above, so alignment is 1:1.
# Below we convert token lists → input IDs and add special-token labels (-100).

def tokenize_and_align(example: dict) -> dict:
    """
    Re-encode the pre-tokenized token list and propagate labels.
    Special tokens ([CLS], [SEP]) receive label -100 (ignored by loss).
    """
    encoding = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
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
            # Continuation sub-token → use I- variant
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


def save_tokenized_dataset_cache(train_dataset, eval_dataset) -> None:
    """Persist the tokenized train/eval split to disk."""
    synthetic_meta = None
    if os.path.exists(SYNTHETIC_CACHE_META):
        with open(SYNTHETIC_CACHE_META) as f:
            synthetic_meta = json.load(f)

    DatasetDict({"train": train_dataset, "eval": eval_dataset}).save_to_disk(CACHE_DIR)
    with open(CACHE_META, "w") as f:
        json.dump(
            {
                "cache_version": TOKENIZED_CACHE_VERSION,
                "model_checkpoint": MODEL_CHECKPOINT,
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "synthetic_cache_meta": synthetic_meta,
            },
            f,
            indent=2,
        )


def load_tokenized_dataset_cache():
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

    with open(CACHE_META) as f:
        meta = json.load(f)

    synthetic_meta = None
    if os.path.exists(SYNTHETIC_CACHE_META):
        with open(SYNTHETIC_CACHE_META) as f:
            synthetic_meta = json.load(f)

    expected_meta = {
        "cache_version": TOKENIZED_CACHE_VERSION,
        "model_checkpoint": MODEL_CHECKPOINT,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "synthetic_cache_meta": synthetic_meta,
    }
    if meta != expected_meta:
        return None

    try:
        return load_from_disk(CACHE_DIR)
    except Exception:
        # Some Colab zips/extractions are happier if we reconstruct the split
        # datasets directly from their Arrow shards.
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


cached_tokenized = load_tokenized_dataset_cache() if (USE_TOKENIZED_CACHE and not FORCE_REBUILD_TOKENIZED_CACHE) else None
if cached_tokenized is not None:
    train_dataset = cached_tokenized["train"]
    eval_dataset = cached_tokenized["eval"]
    print(f"Loaded tokenized dataset cache from {CACHE_DIR}")
else:
    if synthetic_dataset is None:
        synthetic_dataset = load_synthetic_examples_cache() if (USE_SYNTHETIC_CACHE and not FORCE_REBUILD_SYNTHETIC_CACHE) else None
        if synthetic_dataset is not None:
            print(f"Loaded cached synthetic examples from {SYNTHETIC_CACHE}")
        else:
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
        tokenize_and_align,
        batched=False,
        remove_columns=["tokens", "ner_tags"],
    )
    random_dataset = random_dataset.map(
        tokenize_and_align,
        batched=False,
        remove_columns=["tokens", "ner_tags"],
    )

    # Train / validation split (90 / 10).
    # Keep the coverage set in train so every language is guaranteed to appear there.
    split = random_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = concatenate_datasets([coverage_dataset, split["train"]])
    eval_dataset = split["test"]
    if USE_TOKENIZED_CACHE:
        save_tokenized_dataset_cache(train_dataset, eval_dataset)

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")


def release_generation_memory() -> None:
    """Drop synthetic-example artifacts that are no longer needed after tokenization."""
    for name in [
        "synthetic_dataset",
        "preview_examples",
        "coverage_dataset",
        "random_dataset",
    ]:
        globals()[name] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


release_generation_memory()

# %%
# --- Model ---
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

# %%
# --- Training ---
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_preds = [
        [id2label[pred] for pred, lbl in zip(preds, lbls) if lbl != -100]
        for preds, lbls in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lbl] for lbl in lbls if lbl != -100]
        for lbls in labels
    ]
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"], # type: ignore
        "recall":    results["overall_recall"], # type: ignore
        "f1":        results["overall_f1"], # type: ignore
        "accuracy":  results["overall_accuracy"], # type: ignore
    }


data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./lang-ner-xlmr",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effectively batch size 32
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=2500,  # Evaluate less frequently for speed
    save_steps=2500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    logging_steps=100,  # Less noise in the console
    save_total_limit=2,  # Essential for 500k runs
    report_to="tensorboard",
    dataloader_num_workers=mp.cpu_count() // 2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # type: ignore
    eval_dataset=eval_dataset,  # type: ignore
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning …")
trainer.train()
trainer.save_model("./lang-ner-xlmr-final")
tokenizer.save_pretrained("./lang-ner-xlmr-final")
trainer.push_to_hub()
print("Model saved to ./lang-ner-xlmr-final")

# %%
# --- Transparency Validation ---
# Feed a mixed-language sentence to the NER pipeline and visualise the evidence.

ner_pipeline = pipeline(
    "ner",
    model="./lang-ner-xlmr-final",
    tokenizer="./lang-ner-xlmr-final",
    aggregation_strategy="simple",   # merges consecutive same-label tokens
    device=0 if torch.cuda.is_available() else -1,
)

DEMO_SENTENCES = [
    # English + French
    "The committee approved the proposal. Le comité a approuvé la proposition avec quelques modifications.",
    # English + Spanish
    "I really enjoyed the conference yesterday. Fue una experiencia increíble para todos los participantes.",
    # English + German + Russian
    "Hello, my name is Anna. Ich komme aus Deutschland. Я живу в Берлине уже пять лет.",
]

def display_transparency(text: str):
    """Print a token-level language attribution report."""
    results = ner_pipeline(text)
    print(f"\nInput : {text}")
    print("-" * 70)
    print(f"{'Span':<35} {'Label':<12} {'Confidence':>10}")
    print("-" * 70)
    for entity in results:
        word  = entity["word"].replace("▁", " ").strip()
        label = entity["entity_group"]
        score = entity["score"]
        bar   = "█" * int(score * 20)
        print(f"{word:<35} {label:<12} {score:>6.2%}  {bar}")
    print()


print("\n=== TRANSPARENCY VALIDATION ===")
for sentence in DEMO_SENTENCES:
    display_transparency(sentence)

# %%
# --- Save Label Map for Later Use ---
with open("./lang-ner-xlmr-final/label_map.json", "w") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
print("Label map saved.")
