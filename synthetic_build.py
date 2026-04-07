from __future__ import annotations

import gc
import json
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic
from source_config import FT, POOL, RUN, SMOL

MAX_LENGTH = RUN["len"]
EXAMPLES_TARGET = RUN["target"]
MIN_COVERAGE_DOCS_PER_LANG = RUN["cov_min"]
MAX_COVERAGE_DOCS_PER_LANG = RUN["cov_max"]
USE_SYNTHETIC_CACHE = RUN["syn_cache"]
FORCE_REBUILD_SYNTHETIC_CACHE = RUN["syn_rebuild"]
SYNTHETIC_DOC_RETRY_LIMIT = RUN["retry"]
SYNTHETIC_PREVIEW_ROWS = RUN["preview"]

RESERVE_FRACTION = POOL["wiki"]["reserve"]
MIN_RESERVED_SENTENCES = POOL["wiki"]["min"]
MAX_RESERVED_SENTENCES = POOL["wiki"]["max"]
SMOL_RESERVE_FRACTION = POOL["smol"]["reserve"]
SMOL_MIN_RESERVED_SENTENCES = POOL["smol"]["min"]
SMOL_MAX_RESERVED_SENTENCES = POOL["smol"]["max"]
FT_RESERVE_FRACTION = POOL["ft"]["reserve"]
FT_MIN_RESERVED_SENTENCES = POOL["ft"]["min"]
FT_MAX_RESERVED_SENTENCES = POOL["ft"]["max"]

USE_SMOL_AUGMENTATION = SMOL["use"]
USE_FINETRANS_AUGMENTATION = FT["use"]
from source_pools import (
    build_source_sentence_pools,
    chunk_list,
    draw_sentence,
    partition_sentence_pools,
    remaining_sentence_count,
)
from synthetic_cache import (
    CACHE_VERSION,
    SYNTHETIC_CACHE,
    SYNTHETIC_CACHE_META,
    SYNTHETIC_TEMP_DIR,
    _append_synthetic_rows,
    _clear_synthetic_cache_dir,
    _load_synthetic_examples_dataset,
    _move_synthetic_shard,
    _synthetic_example_to_row,
    _synthetic_row_to_example,
    _synthetic_rows_to_table,
    _synthetic_worker_temp_path,
)

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def bio_label_tokens(tokens: list[str], lang: str, is_first: bool, label2id: dict[str, int]) -> list[int]:
    """Assign BIO labels to a token sequence for a given language."""
    labels: list[int] = []
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


def swap_random_tokens(tokens: list[str], labels: list[int], swap_rate: float = 0.02) -> tuple[list[str], list[int]]:
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
    *,
    seed: int,
    worker_idx: int,
    coverage_langs: list[str],
    random_count: int,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None,
    synthetic_temp_dir: str,
    tokenizer,
    all_langs: list[str],
    lang_to_group: dict[str, str],
    language_group_weights: dict[str, float],
    max_length: int,
    label2id: dict[str, int],
    sample_o_span: Callable[[], str],
    sample_code_span: Callable[[], str],
) -> str:
    """Generate a chunk of synthetic examples in one worker."""
    worker_seed = seed + (worker_idx * 10_000)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))

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
                    primary_pool=primary_pool,
                    fallback_pool=fallback_pool,
                    required_langs=[lang],
                    worker_idx=worker_idx,
                    tokenizer=tokenizer,
                    all_langs=all_langs,
                    lang_to_group=lang_to_group,
                    language_group_weights=language_group_weights,
                    max_length=max_length,
                    label2id=label2id,
                    sample_o_span=sample_o_span,
                    sample_code_span=sample_code_span,
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
                    primary_pool=primary_pool,
                    fallback_pool=fallback_pool,
                    worker_idx=worker_idx,
                    tokenizer=tokenizer,
                    all_langs=all_langs,
                    lang_to_group=lang_to_group,
                    language_group_weights=language_group_weights,
                    max_length=max_length,
                    label2id=label2id,
                    sample_o_span=sample_o_span,
                    sample_code_span=sample_code_span,
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
            "seed": worker_seed,
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
    with open(SYNTHETIC_CACHE_META, "w", encoding="utf-8") as f:
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

    with open(SYNTHETIC_CACHE_META, encoding="utf-8") as f:
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
    *,
    tokenizer,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
    required_langs: list[str] | None = None,
    o_inject_prob: float = 0.4,
    n_segments: int = 4,
    strip_punct_prob: float = 0.5,
    swap_prob: float = 0.3,
    all_langs: list[str],
    lang_to_group: dict[str, str],
    language_group_weights: dict[str, float],
    max_length: int,
    label2id: dict[str, int],
    sample_o_span: Callable[[], str],
    sample_code_span: Callable[[], str],
) -> dict:
    """Build one synthetic mixed-language training example."""
    chosen_langs: list[str] = []
    seen_langs: set[str] = set()

    def _candidate_langs() -> list[str]:
        candidates = [
            lang
            for lang in all_langs
            if remaining_sentence_count(lang, primary_pool, fallback_pool) > 0 and lang not in seen_langs
        ]
        return candidates

    def _sample_language(candidates: list[str]) -> str | None:
        if not candidates:
            return None
        weights = [
            language_group_weights.get(lang_to_group.get(lang, ""), 1.0)
            * remaining_sentence_count(lang, primary_pool, fallback_pool)
            for lang in candidates
        ]
        return random.choices(candidates, weights=weights, k=1)[0]

    for lang in required_langs or []:
        if remaining_sentence_count(lang, primary_pool, fallback_pool) > 0 and lang not in seen_langs:
            chosen_langs.append(lang)
            seen_langs.add(lang)

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
        if total_tokens >= max_length - 20:
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

        labels = bio_label_tokens(tokens, lang, is_first=(len(all_tokens) == 0), label2id=label2id)

        if random.random() < swap_prob:
            tokens, labels = swap_random_tokens(tokens[:], labels[:])

        remaining = max_length - 2 - total_tokens
        tokens = tokens[:remaining]
        labels = labels[:remaining]

        all_tokens.extend(tokens)
        all_labels.extend(labels)
        total_tokens += len(tokens)

    if len(chosen_langs) == 1 and total_tokens < max_length - 20:
        span = sample_code_span()
        original_text_parts.append(span)
        code_tokens = tokenizer.tokenize(span)
        remaining = max_length - 2 - len(all_tokens)
        code_tokens = code_tokens[:min(remaining, 120)]
        if code_tokens:
            insert_pos = random.randint(0, len(all_tokens))
            all_tokens = all_tokens[:insert_pos] + code_tokens + all_tokens[insert_pos:]
            all_labels = all_labels[:insert_pos] + [0] * len(code_tokens) + all_labels[insert_pos:]
            total_tokens += len(code_tokens)

    if random.random() < o_inject_prob and total_tokens < max_length - 20:
        n_injections = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(n_injections):
            if total_tokens >= max_length - 10:
                break
            span = sample_o_span()
            original_text_parts.append(span)
            o_tokens = tokenizer.tokenize(span)
            remaining = max_length - 2 - len(all_tokens)
            o_tokens = o_tokens[:min(remaining, 40)]
            if o_tokens:
                insert_pos = random.randint(0, len(all_tokens))
                all_tokens = all_tokens[:insert_pos] + o_tokens + all_tokens[insert_pos:]
                all_labels = all_labels[:insert_pos] + [0] * len(o_tokens) + all_labels[insert_pos:]
                total_tokens += len(o_tokens)

    return {"original_text": " ".join(original_text_parts).strip(), "tokens": all_tokens, "ner_tags": all_labels}


def build_synthetic_doc_with_retry(
    *,
    tokenizer,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
    required_langs: list[str] | None = None,
    worker_idx: int = 0,
    max_retries: int = SYNTHETIC_DOC_RETRY_LIMIT,
    all_langs: list[str],
    lang_to_group: dict[str, str],
    language_group_weights: dict[str, float],
    max_length: int,
    label2id: dict[str, int],
    sample_o_span: Callable[[], str],
    sample_code_span: Callable[[], str],
) -> dict:
    """Build a synthetic doc and retry if it is malformed or too long for the model."""
    for _ in range(1, max_retries + 1):
        example = create_synthetic_doc(
            tokenizer=tokenizer,
            primary_pool=primary_pool,
            fallback_pool=fallback_pool,
            required_langs=required_langs,
            all_langs=all_langs,
            lang_to_group=lang_to_group,
            language_group_weights=language_group_weights,
            max_length=max_length,
            label2id=label2id,
            sample_o_span=sample_o_span,
            sample_code_span=sample_code_span,
        )
        if len(example.get("tokens", ())) != len(example.get("ner_tags", ())):
            continue
        try:
            encoded = tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_overflowing_tokens=True,
            )
        except Exception:
            continue
        if encoded.get("overflowing_tokens"):
            continue
        return example

    raise RuntimeError(
        f"Worker {worker_idx}: failed to build a valid synthetic doc after {max_retries} attempts"
    )


def print_sampling_stats(
    examples: list[dict],
    *,
    id2label: dict[int, str],
    all_langs: list[str],
    language_groups: list[str],
    lang_to_group: dict[str, str],
    top_n: int = 12,
) -> None:
    """Print per-language and per-group coverage stats for the synthetic corpus."""
    example_lang_counts = {lang: 0 for lang in all_langs}
    token_lang_counts = {lang: 0 for lang in all_langs}
    group_example_counts = {group: 0 for group in language_groups}
    group_token_counts = {group: 0 for group in language_groups}

    for example in examples:
        langs_in_example: set[str] = set()
        for tag_id in example["ner_tags"]:
            if tag_id == 0:
                continue
            lang = id2label[tag_id][2:].lower()
            token_lang_counts[lang] += 1
            group = lang_to_group.get(lang, "Unknown")
            group_token_counts[group] = group_token_counts.get(group, 0) + 1
            langs_in_example.add(lang)

        for lang in langs_in_example:
            example_lang_counts[lang] += 1
            group = lang_to_group.get(lang, "Unknown")
            group_example_counts[group] = group_example_counts.get(group, 0) + 1

    missing_langs = [lang for lang in all_langs if example_lang_counts[lang] == 0]

    print("\nSampling stats")
    print("-" * 72)
    print(f"Examples: {len(examples)}")
    print(f"Languages covered: {len(all_langs) - len(missing_langs)}/{len(all_langs)}")
    if missing_langs:
        print("Missing languages:", ", ".join(missing_langs))
    else:
        print("Missing languages: none")

    print("\nPer-language coverage (examples containing the language):")
    for lang, count in sorted(example_lang_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]:
        print(f"  {lang:<3}  {count:>5}")

    print("\nPer-group coverage (examples / tokens):")
    for group in language_groups:
        print(
            f"  {group:<12} "
            f"{group_example_counts.get(group, 0):>5} examples | "
            f"{group_token_counts.get(group, 0):>7} tokens"
        )

    top_tokens = sorted(token_lang_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    print("\nTop token languages:")
    for lang, count in top_tokens:
        print(f"  {lang:<3}  {count:>7}")


def release_generation_memory() -> None:
    """Drop synthetic-example artifacts that are no longer needed after tokenization."""
    for name in [
        "synthetic_dataset",
        "preview_examples",
        "coverage_dataset",
        "random_dataset",
        "reserved_sentence_pools",
        "main_sentence_pools",
        "reserved_worker_pools",
        "main_worker_pools",
        "coverage_plan",
    ]:
        globals()[name] = None

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_synthetic_dataset(
    *,
    seed: int,
    tokenizer,
    coverage_sentence_map: dict[str, list[str]],
    smol_sentence_map: dict[str, list[str]] | None,
    ft_sentence_map: dict[str, list[str]] | None,
    all_langs: list[str],
    lang_to_group: dict[str, str],
    language_groups: list[str],
    language_group_weights: dict[str, float],
    label2id: dict[str, int],
    id2label: dict[int, str],
    sample_o_span: Callable[[], str],
    sample_code_span: Callable[[], str],
    generation_workers: int,
) -> Dataset:
    """Build or load the synthetic mixed-language dataset."""
    synthetic_dataset = load_synthetic_examples_cache() if (USE_SYNTHETIC_CACHE and not FORCE_REBUILD_SYNTHETIC_CACHE) else None
    if synthetic_dataset is not None:
        print(f"Loaded cached synthetic examples from {SYNTHETIC_CACHE}")
    if synthetic_dataset is None:
        source_specs: list[dict[str, Any]] = [
            {
                "name": "wiki",
                "sentence_map": coverage_sentence_map,
                "reserve_fraction": RESERVE_FRACTION,
                "min_reserved": MIN_RESERVED_SENTENCES,
                "max_reserved": MAX_RESERVED_SENTENCES,
            },
        ]
        if USE_SMOL_AUGMENTATION:
            source_specs.append(
                {
                    "name": "smol",
                    "sentence_map": smol_sentence_map,
                    "reserve_fraction": SMOL_RESERVE_FRACTION,
                    "min_reserved": SMOL_MIN_RESERVED_SENTENCES,
                    "max_reserved": SMOL_MAX_RESERVED_SENTENCES,
                }
            )
        if USE_FINETRANS_AUGMENTATION:
            source_specs.append(
                {
                    "name": "finetranslations",
                    "sentence_map": ft_sentence_map,
                    "reserve_fraction": FT_RESERVE_FRACTION,
                    "min_reserved": FT_MIN_RESERVED_SENTENCES,
                    "max_reserved": FT_MAX_RESERVED_SENTENCES,
                }
            )

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

        missing_coverage_langs = [lang for lang in all_langs if not coverage_sentence_map.get(lang)]
        if missing_coverage_langs:
            print("WARNING: no extracted sentences for:", ", ".join(missing_coverage_langs))

        coverage_plan: list[str] = []
        for lang in all_langs:
            if not coverage_sentence_map.get(lang):
                continue
            reserved_n = len(reserved_sentence_pools.get(lang, []))
            coverage_docs_for_lang = max(
                1,
                min(MAX_COVERAGE_DOCS_PER_LANG, max(MIN_COVERAGE_DOCS_PER_LANG, (reserved_n + 3) // 4)),
            )
            coverage_plan.extend([lang] * coverage_docs_for_lang)

        _clear_synthetic_cache_dir(SYNTHETIC_CACHE)
        random_job_count = max(0, EXAMPLES_TARGET - len(coverage_plan))
        shard_paths: list[str] = []
        synthetic_total_examples = 0
        coverage_chunks = chunk_list(coverage_plan, generation_workers)
        random_counts = [
            random_job_count // generation_workers + (1 if i < (random_job_count % generation_workers) else 0)
            for i in range(generation_workers)
        ]

        if generation_workers == 1:
            temp_path = generate_synthetic_examples_chunk(
                seed=seed,
                worker_idx=0,
                coverage_langs=coverage_plan,
                random_count=random_job_count,
                primary_pool=reserved_sentence_pools,
                fallback_pool=main_sentence_pools,
                synthetic_temp_dir=SYNTHETIC_TEMP_DIR,
                tokenizer=tokenizer,
                all_langs=all_langs,
                lang_to_group=lang_to_group,
                language_group_weights=language_group_weights,
                max_length=MAX_LENGTH,
                label2id=label2id,
                sample_o_span=sample_o_span,
                sample_code_span=sample_code_span,
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
                        seed=seed,
                        worker_idx=worker_idx,
                        coverage_langs=coverage_langs,
                        random_count=random_counts[worker_idx],
                        primary_pool=reserved_worker_pools[worker_idx],
                        fallback_pool=main_worker_pools[worker_idx],
                        synthetic_temp_dir=SYNTHETIC_TEMP_DIR,
                        tokenizer=tokenizer,
                        all_langs=all_langs,
                        lang_to_group=lang_to_group,
                        language_group_weights=language_group_weights,
                        max_length=MAX_LENGTH,
                        label2id=label2id,
                        sample_o_span=sample_o_span,
                        sample_code_span=sample_code_span,
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

    preview_n = min(SYNTHETIC_PREVIEW_ROWS, len(synthetic_dataset))
    preview_examples = [_synthetic_row_to_example(row) for row in synthetic_dataset.select(range(preview_n))]
    if preview_examples:
        print_sampling_stats(
            preview_examples,
            id2label=id2label,
            all_langs=all_langs,
            language_groups=language_groups,
            lang_to_group=lang_to_group,
        )

    return synthetic_dataset
