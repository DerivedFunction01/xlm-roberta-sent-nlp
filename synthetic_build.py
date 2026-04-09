from __future__ import annotations

import gc
import json
import os
import random
import re
import unicodedata
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic
from paths import PATHS
from source_config import DOC_MIX, FT, INSTRUCT, POOL, RUN, SMOL
from language import ALL_LANGS, LANG_TO_GROUP, LANGUAGE_GROUPS, LANGUAGE_GROUP_WEIGHTS, LATIN_GROUPS

MAX_LENGTH = RUN["len"]
EXAMPLES_TARGET = RUN["target"]
USE_SYNTHETIC_CACHE = RUN["syn_cache"]
FORCE_REBUILD_SYNTHETIC_CACHE = RUN["syn_rebuild"]
SYNTHETIC_DOC_RETRY_LIMIT = RUN["retry"]
SYNTHETIC_PREVIEW_ROWS = RUN["preview"]

CACHE_VERSION = PATHS["versions"]["cache"]
SYNTHETIC_CACHE = PATHS["synthetic"]["cache_dir"]
SYNTHETIC_CACHE_META = PATHS["synthetic"]["cache_meta"]
SYNTHETIC_TEMP_DIR = PATHS["synthetic"]["temp_dir"]

RESERVE_FRACTION = POOL["wiki"]["reserve"]
MIN_RESERVED_SENTENCES = POOL["wiki"]["min"]
MAX_RESERVED_SENTENCES = POOL["wiki"]["max"]
SMOL_RESERVE_FRACTION = POOL["smol"]["reserve"]
SMOL_MIN_RESERVED_SENTENCES = POOL["smol"]["min"]
SMOL_MAX_RESERVED_SENTENCES = POOL["smol"]["max"]
INSTRUCT_RESERVE_FRACTION = POOL["instruct"]["reserve"]
INSTRUCT_MIN_RESERVED_SENTENCES = POOL["instruct"]["min"]
INSTRUCT_MAX_RESERVED_SENTENCES = POOL["instruct"]["max"]
FT_RESERVE_FRACTION = POOL["ft"]["reserve"]
FT_MIN_RESERVED_SENTENCES = POOL["ft"]["min"]
FT_MAX_RESERVED_SENTENCES = POOL["ft"]["max"]

USE_SMOL_AUGMENTATION = SMOL["use"]
USE_INSTRUCTION_AUGMENTATION = INSTRUCT["use"]
USE_FINETRANS_AUGMENTATION = FT["use"]
PURE_DOC_MIX = DOC_MIX["pure"]
HOMOGENEOUS_DOC_MIX = DOC_MIX["homogeneous"]
MIXED_DOC_MIX = DOC_MIX["mixed"]
from source_pools import (
    build_disk_sentence_pool_shards,
    chunk_list,
    draw_sentence,
    load_worker_sentence_pool,
    remaining_sentence_count,
)
from synthetic_cache import (
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


def _build_language_doc_plan(
    language_stats: dict[str, dict[str, int]],
    *,
    source_key: str,
    target_docs: int,
    docs_per_sentence_estimate: int,
    seed: int,
) -> list[str]:
    """Build a deterministic per-language doc plan bounded by available sentence supply."""
    capacities: dict[str, int] = {}
    for lang, stats in language_stats.items():
        available = int(stats.get(source_key, 0))
        if available <= 0:
            continue
        capacities[lang] = max(1, available // max(1, docs_per_sentence_estimate))

    if not capacities or target_docs <= 0:
        return []

    rng = random.Random(seed)
    plan: list[str] = []

    for lang in sorted(capacities):
        if len(plan) >= target_docs:
            break
        plan.append(lang)
        capacities[lang] -= 1
        if capacities[lang] <= 0:
            del capacities[lang]

    while len(plan) < target_docs and capacities:
        candidates = list(capacities)
        weights = [capacities[lang] for lang in candidates]
        lang = rng.choices(candidates, weights=weights, k=1)[0]
        plan.append(lang)
        capacities[lang] -= 1
        if capacities[lang] <= 0:
            del capacities[lang]

    rng.shuffle(plan)
    return plan


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


@lru_cache(maxsize=16_384)
def _is_punctuation_token(token: str) -> bool:
    """Check if a token (possibly with ▁ prefix) is pure punctuation in any script.
    
    Cached to avoid redundant Unicode category checks for repeated tokens.
    """
    # Remove the space marker (▁) if present
    text = token[1:] if token.startswith("▁") else token
    # Check if all characters are Unicode punctuation (category P*)
    return len(text) > 0 and all(unicodedata.category(c).startswith("P") for c in text)


def augment_boundary(tokens: list[str], strip_punct: bool) -> list[str]:
    """Optionally remove sentence-final punctuation to simulate no-boundary code-switching.
    
    Handles punctuation from all Unicode scripts (ASCII, Chinese, Arabic, etc.).
    """
    if strip_punct and tokens:
        tokens = [t for t in tokens if not _is_punctuation_token(t)]
    return tokens


def _inject_formatting_artifact(
    tokens: list[str],
    labels: list[int],
    *,
    tokenizer,
    prefix: str = "",
    suffix: str = "",
    insert_at: int | None = None,
) -> tuple[list[str], list[int], list[str]]:
    """Inject formatting-only tokens that should stay labeled as O."""
    artifact_text: list[str] = []
    if prefix:
        prefix_tokens = tokenizer.tokenize(prefix)
        if prefix_tokens:
            tokens = prefix_tokens + tokens
            labels = [0] * len(prefix_tokens) + labels
            artifact_text.append(prefix)
    if suffix:
        suffix_tokens = tokenizer.tokenize(suffix)
        if suffix_tokens:
            tokens = tokens + suffix_tokens
            labels = labels + [0] * len(suffix_tokens)
            artifact_text.append(suffix)
    if insert_at is not None and 0 < insert_at < len(tokens):
        # Keep infix insertion available for future expansions.
        pass
    return tokens, labels, artifact_text


def _add_formatting_noise(
    tokens: list[str],
    labels: list[int],
    *,
    tokenizer,
    lang: str,
    artifact_prob: float,
) -> tuple[list[str], list[int], list[str]]:
    """Add light formatting noise to pure/homogeneous rows."""
    if not tokens or artifact_prob <= 0 or random.random() >= artifact_prob:
        return tokens, labels, []

    pattern = random.choice(["wrap", "bullet", "trail", "tag"])
    if pattern == "wrap":
        prefix, suffix = random.choice([
            ("(", ")"),
            ("[", "]"),
            ("\"", "\""),
            ("“", "”"),
            ("«", "»"),
        ])
        return _inject_formatting_artifact(tokens, labels, tokenizer=tokenizer, prefix=prefix, suffix=suffix)
    if pattern == "bullet":
        prefix = random.choice(["-", "•", "*", "1.", "i.", "##", "###"])
        return _inject_formatting_artifact(tokens, labels, tokenizer=tokenizer, prefix=prefix)
    if pattern == "trail":
        suffix = random.choice([":", ";", "...", "?!", " |", " | |"])
        return _inject_formatting_artifact(tokens, labels, tokenizer=tokenizer, suffix=suffix)
    prefix, suffix = random.choice([
        ("<p>", "</p>"),
        ("<div>", "</div>"),
        ("<blockquote>", "</blockquote>"),
        ("<table>", "</table>"),
        ("<tr>", "</tr>"),
        ("<td>", "</td>"),
        ("<span>", "</span>"),
        ("**", "**"),
    ])
    return _inject_formatting_artifact(tokens, labels, tokenizer=tokenizer, prefix=prefix, suffix=suffix)


def _render_original_text(parts: list[str], paragraph_break_prob: float = 0.0) -> str:
    """Join generated sentence parts into a lightly paragraphized preview string."""
    if not parts:
        return ""
    if paragraph_break_prob <= 0 or len(parts) == 1:
        return " ".join(parts).strip()

    rendered: list[str] = [parts[0]]
    for part in parts[1:]:
        if random.random() < paragraph_break_prob:
            rendered.append("\n\n")
        else:
            rendered.append(" ")
        rendered.append(part)
    return "".join(rendered).strip()


_WORD_RE = re.compile(r"\b[^\W\d_]{2,}\b", flags=re.UNICODE)


def _apply_random_word_casing(
    sentence: str,
    *,
    lang: str,
    uppercase_prob: float,
    lowercase_prob: float,
    titlecase_prob: float,
) -> str:
    """Apply one random casing transform to a word in a mostly Latin sentence."""
    total_prob = max(0.0, uppercase_prob) + max(0.0, lowercase_prob) + max(0.0, titlecase_prob)
    if total_prob <= 0 or random.random() >= total_prob:
        return sentence
    if LANG_TO_GROUP.get(lang) not in LATIN_GROUPS:
        return sentence

    matches = list(_WORD_RE.finditer(sentence))
    if not matches:
        return sentence
    match = random.choice(matches)
    word = match.group(0)
    if len(word) < 3:
        return sentence
    roll = random.random() * total_prob
    if roll < uppercase_prob:
        replacement = word.upper()
    elif roll < uppercase_prob + lowercase_prob:
        replacement = word.lower()
    else:
        replacement = word[:1].upper() + word[1:].lower()
    if replacement == word:
        return sentence
    return f"{sentence[:match.start()]}{replacement}{sentence[match.end():]}"


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
    pure_langs: list[str],
    homogeneous_langs: list[str],
    mixed_count: int,
    primary_pool_path: str,
    fallback_pool_path: str | None,
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

    primary_pool = load_worker_sentence_pool(primary_pool_path)
    fallback_pool = load_worker_sentence_pool(fallback_pool_path) if fallback_pool_path else None

    worker_desc = f"Worker {worker_idx}"
    temp_path = _synthetic_worker_temp_path(synthetic_temp_dir, worker_idx)
    batch_rows: list[dict] = []
    pure_written = 0
    homogeneous_written = 0
    mixed_written = 0

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
        total_jobs = len(pure_langs) + len(homogeneous_langs) + mixed_count
        with tqdm(total=total_jobs, desc=worker_desc, position=worker_idx, leave=False) as pbar:
            for lang in pure_langs:
                example = build_synthetic_doc_with_retry(
                    primary_pool=primary_pool,
                    fallback_pool=None,
                    required_langs=[lang],
                    pure=True,
                    pure_lang=lang,
                    min_sentences=PURE_DOC_MIX["min_sentences"],
                    max_sentences=PURE_DOC_MIX["max_sentences"],
                    strip_punct_prob=PURE_DOC_MIX["strip_punct_prob"],
                    format_noise_prob=PURE_DOC_MIX.get("format_noise_prob", 0.0),
                    paragraph_break_prob=PURE_DOC_MIX.get("paragraph_break_prob", 0.0),
                    uppercase_word_prob=PURE_DOC_MIX.get("uppercase_word_prob", 0.0),
                    lowercase_word_prob=PURE_DOC_MIX.get("lowercase_word_prob", 0.0),
                    titlecase_word_prob=PURE_DOC_MIX.get("titlecase_word_prob", 0.0),
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
                batch_rows.append(_synthetic_example_to_row("pure", example))
                pure_written += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"pure={pure_written} homogeneous={homogeneous_written} mixed={mixed_written}"
                )
                if len(batch_rows) >= 64:
                    _append_synthetic_rows(writer, batch_rows)
                    batch_rows.clear()

            for lang in homogeneous_langs:
                example = build_synthetic_doc_with_retry(
                    primary_pool=fallback_pool or primary_pool,
                    fallback_pool=None,
                    required_langs=[lang],
                    pure=True,
                    pure_lang=lang,
                    min_sentences=HOMOGENEOUS_DOC_MIX["min_sentences"],
                    max_sentences=HOMOGENEOUS_DOC_MIX["max_sentences"],
                    strip_punct_prob=HOMOGENEOUS_DOC_MIX["strip_punct_prob"],
                    format_noise_prob=HOMOGENEOUS_DOC_MIX.get("format_noise_prob", 0.0),
                    paragraph_break_prob=HOMOGENEOUS_DOC_MIX.get("paragraph_break_prob", 0.0),
                    uppercase_word_prob=HOMOGENEOUS_DOC_MIX.get("uppercase_word_prob", 0.0),
                    lowercase_word_prob=HOMOGENEOUS_DOC_MIX.get("lowercase_word_prob", 0.0),
                    titlecase_word_prob=HOMOGENEOUS_DOC_MIX.get("titlecase_word_prob", 0.0),
                    worker_idx=worker_idx,
                    tokenizer=tokenizer,
                    all_langs=all_langs,
                    lang_to_group=lang_to_group,
                    language_group_weights=language_group_weights,
                    max_length=max_length,
                    label2id=label2id,
                    sample_o_span=sample_o_span,
                    sample_code_span=sample_code_span,
                    n_segments=1,
                )
                batch_rows.append(_synthetic_example_to_row("homogeneous", example))
                homogeneous_written += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"pure={pure_written} homogeneous={homogeneous_written} mixed={mixed_written}"
                )
                if len(batch_rows) >= 64:
                    _append_synthetic_rows(writer, batch_rows)
                    batch_rows.clear()

            for _ in range(mixed_count):
                example = build_synthetic_doc_with_retry(
                    primary_pool=fallback_pool or primary_pool,
                    fallback_pool=None,
                    worker_idx=worker_idx,
                    n_segments=random.randint(MIXED_DOC_MIX["min_segments"], MIXED_DOC_MIX["max_segments"]),
                    strip_punct_prob=MIXED_DOC_MIX["strip_punct_prob"],
                    swap_prob=MIXED_DOC_MIX["swap_prob"],
                    o_inject_prob=MIXED_DOC_MIX["o_inject_prob"],
                    allow_repeated_langs=MIXED_DOC_MIX["allow_repeated_langs"],
                    tokenizer=tokenizer,
                    all_langs=all_langs,
                    lang_to_group=lang_to_group,
                    language_group_weights=language_group_weights,
                    max_length=max_length,
                    label2id=label2id,
                    sample_o_span=sample_o_span,
                    sample_code_span=sample_code_span,
                )
                batch_rows.append(_synthetic_example_to_row("mixed", example))
                mixed_written += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"pure={pure_written} homogeneous={homogeneous_written} mixed={mixed_written}"
                )
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
                "job_count": total_jobs,
                "pure_count": len(pure_langs),
                "homogeneous_count": len(homogeneous_langs),
                "mixed_count": mixed_count,
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
                "doc_mix": DOC_MIX,
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
        "doc_mix": DOC_MIX,
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
    n_segments: int = 4,
    strip_punct_prob: float = 0.35,
    swap_prob: float = 0.12,
    o_inject_prob: float = 0.12,
    allow_repeated_langs: bool = False,
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
            if remaining_sentence_count(lang, primary_pool, fallback_pool) > 0
        ]
        if not allow_repeated_langs:
            candidates = [lang for lang in candidates if lang not in seen_langs]
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
        if allow_repeated_langs or lang not in seen_langs:
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

        if swap_prob > 0 and random.random() < swap_prob:
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

    if o_inject_prob > 0 and random.random() < o_inject_prob and total_tokens < max_length - 20:
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

    return {"original_text": _render_original_text(original_text_parts), "tokens": all_tokens, "ner_tags": all_labels}


def create_pure_synthetic_doc(
    *,
    tokenizer,
    primary_pool: dict[str, deque[str]],
    lang: str,
    label2id: dict[str, int],
    min_sentences: int = 1,
    max_sentences: int = 4,
    strip_punct_prob: float = 0.15,
    format_noise_prob: float = 0.0,
    paragraph_break_prob: float = 0.0,
    uppercase_word_prob: float = 0.0,
    lowercase_word_prob: float = 0.0,
    titlecase_word_prob: float = 0.0,
) -> dict:
    """Build one mostly homogeneous single-language synthetic example."""
    sent_count = random.randint(min_sentences, max_sentences)
    all_tokens, all_labels = [], []
    original_text_parts: list[str] = []
    total_tokens = 0

    for _ in range(sent_count):
        if total_tokens >= MAX_LENGTH - 20:
            break
        sent = draw_sentence(lang, primary_pool, None)
        if sent is None:
            break
        sent = _apply_random_word_casing(
            sent,
            lang=lang,
            uppercase_prob=uppercase_word_prob,
            lowercase_prob=lowercase_word_prob,
            titlecase_prob=titlecase_word_prob,
        )
        original_text_parts.append(sent)
        tokens = tokenizer.tokenize(sent)
        if not tokens:
            continue
        if strip_punct_prob > 0 and random.random() < strip_punct_prob:
            tokens = augment_boundary(tokens, strip_punct=True)
        labels = bio_label_tokens(tokens, lang, is_first=(len(all_tokens) == 0), label2id=label2id)
        tokens, labels, artifact_parts = _add_formatting_noise(
            tokens,
            labels,
            tokenizer=tokenizer,
            lang=lang,
            artifact_prob=format_noise_prob,
        )
        if artifact_parts:
            original_text_parts.extend(artifact_parts)
        remaining = MAX_LENGTH - 2 - total_tokens
        tokens = tokens[:remaining]
        labels = labels[:remaining]
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        total_tokens += len(tokens)

    return {
        "original_text": _render_original_text(original_text_parts, paragraph_break_prob=paragraph_break_prob),
        "tokens": all_tokens,
        "ner_tags": all_labels,
    }


def build_synthetic_doc_with_retry(
    *,
    tokenizer,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
    required_langs: list[str] | None = None,
    pure: bool = False,
    pure_lang: str | None = None,
    min_sentences: int = 1,
    max_sentences: int = 4,
    n_segments: int = 4,
    strip_punct_prob: float = 0.35,
    format_noise_prob: float = 0.0,
    paragraph_break_prob: float = 0.0,
    uppercase_word_prob: float = 0.0,
    lowercase_word_prob: float = 0.0,
    titlecase_word_prob: float = 0.0,
    swap_prob: float = 0.12,
    o_inject_prob: float = 0.12,
    allow_repeated_langs: bool = False,
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
        if pure:
            lang = pure_lang or (required_langs[0] if required_langs else None)
            if lang is None:
                raise ValueError("pure synthetic docs require pure_lang or required_langs")
            example = create_pure_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                lang=lang,
                label2id=label2id,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                strip_punct_prob=strip_punct_prob,
                format_noise_prob=format_noise_prob,
                paragraph_break_prob=paragraph_break_prob,
                uppercase_word_prob=uppercase_word_prob,
                lowercase_word_prob=lowercase_word_prob,
                titlecase_word_prob=titlecase_word_prob,
            )
        else:
            example = create_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                fallback_pool=fallback_pool,
                required_langs=required_langs,
                n_segments=n_segments,
                strip_punct_prob=strip_punct_prob,
                swap_prob=swap_prob,
                o_inject_prob=o_inject_prob,
                allow_repeated_langs=allow_repeated_langs,
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
        "pure_dataset",
        "homogeneous_dataset",
        "mixed_dataset",
        "reserved_sentence_pools",
        "main_sentence_pools",
        "reserved_worker_pools",
        "main_worker_pools",
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
    seed: int = 42,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    sample_o_span: Callable[[], str],
    sample_code_span: Callable[[], str],
) -> Dataset:
    """Build or load the synthetic mixed-language dataset."""
    all_langs = ALL_LANGS
    lang_to_group = LANG_TO_GROUP
    language_groups = LANGUAGE_GROUPS
    language_group_weights = LANGUAGE_GROUP_WEIGHTS
    generation_workers = max(1, mp.cpu_count() // 4)

    synthetic_dataset = load_synthetic_examples_cache() if (USE_SYNTHETIC_CACHE and not FORCE_REBUILD_SYNTHETIC_CACHE) else None
    if synthetic_dataset is not None:
        print(f"Loaded cached synthetic examples from {SYNTHETIC_CACHE}")
    if synthetic_dataset is None:
        source_specs: list[dict[str, Any]] = [
            {
                "name": "wiki",
                "cache_dir": PATHS["wiki"]["cache_dir"],
                "reserve_fraction": RESERVE_FRACTION,
                "min_reserved": MIN_RESERVED_SENTENCES,
                "max_reserved": MAX_RESERVED_SENTENCES,
            },
        ]
        if USE_SMOL_AUGMENTATION:
            source_specs.append(
                {
                    "name": "smol",
                    "cache_dir": PATHS["smol"]["cache_dir"],
                    "reserve_fraction": SMOL_RESERVE_FRACTION,
                    "min_reserved": SMOL_MIN_RESERVED_SENTENCES,
                    "max_reserved": SMOL_MAX_RESERVED_SENTENCES,
                }
            )
        if USE_INSTRUCTION_AUGMENTATION:
            source_specs.append(
                {
                    "name": "instruction",
                    "cache_dir": PATHS["instruction"]["cache_dir"],
                    "reserve_fraction": INSTRUCT_RESERVE_FRACTION,
                    "min_reserved": INSTRUCT_MIN_RESERVED_SENTENCES,
                    "max_reserved": INSTRUCT_MAX_RESERVED_SENTENCES,
                }
            )
        if USE_FINETRANS_AUGMENTATION:
            source_specs.append(
                {
                    "name": "finetranslations",
                    "cache_dir": PATHS["finetrans"]["cache_dir"],
                    "reserve_fraction": FT_RESERVE_FRACTION,
                    "min_reserved": FT_MIN_RESERVED_SENTENCES,
                    "max_reserved": FT_MAX_RESERVED_SENTENCES,
                }
            )

        source_pool_manifest = build_disk_sentence_pool_shards(
            source_specs,
            n_workers=generation_workers,
            seed=seed,
            pool_cache_dir=PATHS["source_pools"]["cache_dir"],
            force_rebuild=FORCE_REBUILD_SYNTHETIC_CACHE,
        )
        source_summaries = source_pool_manifest.get("source_summaries", [])
        for summary in source_summaries:
            print(
                f"{summary['name'].upper()} split -> "
                f"reserved: {summary['reserved']} | main: {summary['main']}"
            )

        reserved_total = int(source_pool_manifest.get("total_reserved", 0))
        main_total = int(source_pool_manifest.get("total_main", 0))
        print(
            f"Reserved sentence bags: {reserved_total} total | "
            f"Main sentence bags: {main_total} total"
        )

        language_stats = source_pool_manifest.get("language_stats", {})
        missing_coverage_langs = [lang for lang in all_langs if int(language_stats.get(lang, {}).get("total", 0)) == 0]
        if missing_coverage_langs:
            print("WARNING: no extracted sentences for:", ", ".join(missing_coverage_langs))

        pure_target = int(round(EXAMPLES_TARGET * PURE_DOC_MIX["fraction"]))
        homogeneous_target = int(round(EXAMPLES_TARGET * HOMOGENEOUS_DOC_MIX["fraction"]))
        mixed_target = max(0, EXAMPLES_TARGET - pure_target - homogeneous_target)

        pure_plan = _build_language_doc_plan(
            language_stats,
            source_key="reserved",
            target_docs=pure_target,
            docs_per_sentence_estimate=3,
            seed=seed + 101,
        )
        homogeneous_plan = _build_language_doc_plan(
            language_stats,
            source_key="main",
            target_docs=homogeneous_target,
            docs_per_sentence_estimate=4,
            seed=seed + 202,
        )
        mixed_target = max(0, EXAMPLES_TARGET - len(pure_plan) - len(homogeneous_plan))

        _clear_synthetic_cache_dir(SYNTHETIC_CACHE)
        shard_paths: list[str] = []
        synthetic_total_examples = 0
        pure_chunks = chunk_list(pure_plan, generation_workers)
        homogeneous_chunks = chunk_list(homogeneous_plan, generation_workers)
        mixed_counts = [
            mixed_target // generation_workers + (1 if i < (mixed_target % generation_workers) else 0)
            for i in range(generation_workers)
        ]

        if generation_workers == 1:
            temp_path = generate_synthetic_examples_chunk(
                seed=seed,
                worker_idx=0,
                pure_langs=pure_plan,
                homogeneous_langs=homogeneous_plan,
                mixed_count=mixed_target,
                primary_pool_path=source_pool_manifest["reserved_shards"][0],
                fallback_pool_path=source_pool_manifest["main_shards"][0],
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
            synthetic_total_examples += int(meta.get("pure_count", 0)) + int(meta.get("homogeneous_count", 0)) + int(meta.get("mixed_count", 0))
        else:
            with ProcessPoolExecutor(max_workers=generation_workers) as pool:
                future_to_worker = {}
                for worker_idx in range(generation_workers):
                    pure_langs = pure_chunks[worker_idx] if worker_idx < len(pure_chunks) else []
                    homogeneous_langs = homogeneous_chunks[worker_idx] if worker_idx < len(homogeneous_chunks) else []
                    future = pool.submit(
                        generate_synthetic_examples_chunk,
                        seed=seed,
                        worker_idx=worker_idx,
                        pure_langs=pure_langs,
                        homogeneous_langs=homogeneous_langs,
                        mixed_count=mixed_counts[worker_idx],
                        primary_pool_path=source_pool_manifest["reserved_shards"][worker_idx],
                        fallback_pool_path=source_pool_manifest["main_shards"][worker_idx],
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
                    synthetic_total_examples += int(meta.get("pure_count", 0)) + int(meta.get("homogeneous_count", 0)) + int(meta.get("mixed_count", 0))

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
