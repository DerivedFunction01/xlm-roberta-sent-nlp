from __future__ import annotations

import json
import os
import random
import re
import time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from datasets.utils.logging import disable_progress_bar
from tqdm.auto import tqdm

from io_utils import write_json_atomic, write_records_parquet, write_sentence_parquet
from paths import (
    FINETRANS_CACHE_DIR,
    FINETRANS_CACHE_DIR_META,
    FINETRANS_CACHE_FILE,
    FINETRANS_CACHE_META,
    SENTENCES_DIR,
)
from language import ENGLISH_STOP_WORDS, LANG_ISO2_TO_ISO3
from source_config import (
    FT_ENGLISH_ACCEPT_EVERY,
    FT_INCLUDE_TRANSLATED_ENGLISH,
    FT_MAX_MISS_STREAK,
    FT_MAX_ROW_INDEX,
    FT_MAX_SENTENCES_PER_LANG,
    FT_OVERFLOW_SENTENCES_PER_LANG,
    FT_TRANSLATED_ENGLISH_LANGS,
)
from text_utils import (
    LATIN_GROUPS,
    SENT_SPLIT,
    _get_segmenter,
    post_clean_sentences,
    sanitize_paragraph_for_pysbd,
)


FINETRANS_DATASET = "HuggingFaceFW/finetranslations"
FINETRANS_MIN_LANGUAGE_SCORE = 0.85
FINETRANS_LATIN_STRICT_LANGUAGE_SCORE_CUTOFF = 0.95
FINETRANS_LATIN_MAX_ENGLISH_RATIO = 0.35
FINETRANS_LATIN_MIN_TOKENS = 4
FINETRANS_MIN_TOKEN_COUNT = 20
FINETRANS_LATIN_LONGEST_CHUNKS = 2
FINETRANS_MIN_QUALITY_SCORE = 0.80
FINETRANS_CHECKPOINT_EVERY_ROWS = 250
FINETRANS_WORKER_COLUMNS = [
    "lang",
    "sentence",
    "sentence_token_length",
    "og_language_score",
    "og_token_count",
    "og_quality_score",
    "edu_score_raw",
    "source_language",
    "is_wikipedia",
    "score",
]
FINETRANS_FINAL_COLUMNS = ["lang", "sentence"]
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WIKIPEDIA_URL_RE = re.compile(r"wikipedia(?:\.org)?", flags=re.IGNORECASE)
FT_ISO3_TO_LANG = {iso3: lang for lang, iso3 in LANG_ISO2_TO_ISO3.items()}

disable_progress_bar()


def _finetrans_cache_path(sentences_dir: str) -> str:
    return FINETRANS_CACHE_FILE if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "finetranslations_sentences.parquet",
    )


def _finetrans_meta_path(sentences_dir: str) -> str:
    return FINETRANS_CACHE_DIR_META if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "finetranslations",
        "finetranslations.meta.json",
    )


def _finetrans_cache_dir(sentences_dir: str) -> str:
    return FINETRANS_CACHE_DIR if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "finetranslations",
    )


def _finetrans_lang_path(sentences_dir: str, lang: str) -> str:
    return os.path.join(_finetrans_cache_dir(sentences_dir), f"{lang}.parquet")


def _finetrans_config_dir(sentences_dir: str, config_idx: int) -> str:
    return os.path.join(sentences_dir, "_finetrans_tmp", f"config_{config_idx:05d}")


def _finetrans_config_meta_path(sentences_dir: str, config_idx: int) -> str:
    return os.path.join(_finetrans_config_dir(sentences_dir, config_idx), "checkpoint.meta.json")


def _finetrans_english_config_dir(sentences_dir: str, config_idx: int) -> str:
    return os.path.join(_finetrans_config_dir(sentences_dir, config_idx), "en")


def _finetrans_config_records_path(sentences_dir: str, config_idx: int) -> str:
    return os.path.join(_finetrans_config_dir(sentences_dir, config_idx), "source.parquet")


def _finetrans_english_records_path(sentences_dir: str, config_idx: int) -> str:
    return os.path.join(_finetrans_english_config_dir(sentences_dir, config_idx), "en.parquet")


def _config_name_to_lang(
    config_name: str,
    lang_to_group: dict[str, str],
) -> str | None:
    config_name = config_name.rsplit("/", 1)[-1]
    if config_name == "all":
        return None
    base = config_name.split("_", 1)[0]
    if len(base) == 2 and base in lang_to_group:
        if lang_to_group.get(base) not in LATIN_GROUPS and config_name.endswith("_Latn"):
            return None
        return base
    lang = FT_ISO3_TO_LANG.get(base)
    if lang in lang_to_group:
        if lang_to_group.get(lang) not in LATIN_GROUPS and config_name.endswith("_Latn"):
            return None
        return lang
    return None


def _matching_configs(
    lang_to_group: dict[str, str],
) -> list[tuple[str, str]]:
    matched: list[tuple[str, str]] = []
    for config in get_dataset_config_names(FINETRANS_DATASET):
        lang = _config_name_to_lang(config, lang_to_group)
        if lang is not None:
            matched.append((config, lang))
    matched.sort(key=lambda item: item[0])
    return matched


def _row_chunks(row: dict[str, Any], chunk_key: str, text_key: str) -> list[str]:
    chunks = row.get(chunk_key)
    if isinstance(chunks, list):
        result = [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
        if result:
            return result
    text = row.get(text_key)
    if isinstance(text, str) and text.strip():
        return [text.strip()]
    return []


def _row_language_score(row: dict[str, Any]) -> float | None:
    score = row.get("og_language_score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _row_token_count(row: dict[str, Any]) -> int | None:
    count = row.get("og_token_count")
    if isinstance(count, int):
        return count
    if isinstance(count, float):
        return int(count)
    return None


def _row_quality_score(row: dict[str, Any]) -> float | None:
    score = row.get("og_quality_score")
    if isinstance(score, (int, float)):
        score = float(score)
        if score >= 0.0:
            return score
    return None


def _row_edu_score(row: dict[str, Any]) -> float | None:
    score = row.get("edu_score_raw")
    if isinstance(score, (int, float)):
        return float(score)
    score = row.get("edu_score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _row_source_language(row: dict[str, Any]) -> str | None:
    lang = row.get("og_language")
    if isinstance(lang, str) and lang.strip():
        base = lang.strip().split("_", 1)[0]
        if len(base) == 2:
            return base
        return FT_ISO3_TO_LANG.get(base)
    return None


def _row_is_wikipedia(row: dict[str, Any]) -> bool:
    url = row.get("url")
    if not isinstance(url, str):
        return False
    return bool(WIKIPEDIA_URL_RE.search(url))


def _longest_chunks(chunks: list[str], limit: int = FINETRANS_LATIN_LONGEST_CHUNKS) -> list[str]:
    ordered = sorted(
        enumerate(chunks),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return [chunk for _, chunk in ordered[:limit]]


@lru_cache(maxsize=1)
def _english_stopwords() -> set[str]:
    return {word.lower() for word in ENGLISH_STOP_WORDS}


@lru_cache(maxsize=32_768)
def _is_english_word(token: str) -> bool:
    word = token.lower().strip()
    if not word:
        return False
    sw = _english_stopwords()
    if word in sw:
        return True
    if word.endswith("s") and word[:-1] in sw:
        return True
    return False


def _latin_tokens(text: str) -> list[str]:
    return LATIN_TOKEN_RE.findall(text.lower())


def _sentence_token_length(sentence: str) -> int:
    return len(_latin_tokens(sentence))


def _looks_english_heavy(text: str) -> bool:
    tokens = _latin_tokens(text)
    if len(tokens) < FINETRANS_LATIN_MIN_TOKENS:
        return any(_is_english_word(token) for token in tokens)
    stopword_hits = sum(1 for token in tokens if _is_english_word(token))
    english_ratio = stopword_hits / max(1, len(tokens))
    return stopword_hits >= 3 and english_ratio >= FINETRANS_LATIN_MAX_ENGLISH_RATIO


def _row_base_score(row: dict[str, Any], lang: str) -> float:
    score = 0.0
    quality = _row_quality_score(row)
    if quality is not None:
        score += quality * 10.0
    token_count = _row_token_count(row)
    if token_count is not None:
        score += min(token_count / 50.0, 4.0)
    if lang == "en":
        edu = _row_edu_score(row)
        if edu is not None:
            score += edu * 2.0
    return score


def _sentence_base_score(row: dict[str, Any], lang: str, sentence_token_length: int) -> float:
    return _row_base_score(row, lang) + min(sentence_token_length / 8.0, 4.0)


def _latin_source_lines(
    chunks: list[str],
    *,
    lang_score: float | None = None,
) -> list[str]:
    selected_chunks = _longest_chunks(chunks)
    lines: list[str] = []
    seen: set[str] = set()
    apply_english_filter = (
        lang_score is None or lang_score < FINETRANS_LATIN_STRICT_LANGUAGE_SCORE_CUTOFF
    )
    for chunk in selected_chunks:
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if apply_english_filter and _looks_english_heavy(line):
                continue
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
    return lines


def _sentence_records_from_row(
    row: dict[str, Any],
    *,
    lang: str,
    lang_to_group: dict[str, str],
    translated: bool = False,
) -> list[dict[str, Any]]:
    if translated:
        chunks = _row_chunks(row, "translated_chunks", "translated_text")
    else:
        chunks = _row_chunks(row, "og_chunks", "og_full_text")
    if not chunks:
        return []

    raw_sentences: list[str]
    if translated:
        raw_sentences = []
        for chunk in chunks:
            raw_sentences.extend(_segment_text(chunk, lang))
        cleaned_sentences = post_clean_sentences(raw_sentences, lang, lang_to_group)
    elif lang_to_group.get(lang) in LATIN_GROUPS:
        cleaned_sentences = post_clean_sentences(
            _latin_source_lines(chunks, lang_score=_row_language_score(row)),
            lang,
            lang_to_group,
        )
    else:
        raw_sentences = []
        for chunk in chunks:
            raw_sentences.extend(_segment_text(chunk, lang))
        cleaned_sentences = post_clean_sentences(raw_sentences, lang, lang_to_group)

    record_sentences: list[dict[str, Any]] = []
    for sentence in cleaned_sentences:
        sentence_len = _sentence_token_length(sentence)
        record_sentences.append(
            {
                "sentence": sentence,
                "sentence_token_length": sentence_len,
                "og_language_score": _row_language_score(row),
                "og_token_count": _row_token_count(row),
                "og_quality_score": _row_quality_score(row),
                "edu_score_raw": _row_edu_score(row) if translated else None,
                "source_language": _row_source_language(row),
                "is_wikipedia": _row_is_wikipedia(row),
                "score": _sentence_base_score(row, lang, sentence_len),
            }
        )
    return record_sentences


def _segment_text(text: str, lang: str) -> list[str]:
    safe_text = sanitize_paragraph_for_pysbd(text)
    segmenter = _get_segmenter(lang)
    try:
        segments = segmenter.segment(safe_text) if segmenter else SENT_SPLIT.split(safe_text)
    except re.error:
        segments = SENT_SPLIT.split(safe_text)
    return [segment for segment in segments if isinstance(segment, str) and segment.strip()]


def _row_is_translated_en_acceptable(row: dict[str, Any]) -> bool:
    edu = _row_edu_score(row)
    if edu is None:
        return True
    return edu >= 1.0


def _should_keep_english_sentence(english_sentence_idx: int, accept_every: int) -> bool:
    if accept_every <= 1:
        return True
    return english_sentence_idx % accept_every == 0


def _include_translated_english_for_lang(lang: str, include_translated_english: bool) -> bool:
    return include_translated_english and lang in FT_TRANSLATED_ENGLISH_LANGS


def _compute_length_thresholds(records: list[dict[str, Any]]) -> tuple[int, int]:
    lengths = sorted(
        record["sentence_token_length"]
        for record in records
        if isinstance(record.get("sentence_token_length"), int)
    )
    if not lengths:
        return (0, 0)
    q25_idx = max(0, min(len(lengths) - 1, int(round((len(lengths) - 1) * 0.25))))
    q75_idx = max(0, min(len(lengths) - 1, int(round((len(lengths) - 1) * 0.75))))
    return lengths[q25_idx], lengths[q75_idx]


def _select_bucketed_records(
    records: list[dict[str, Any]],
    max_sentences: int,
) -> list[dict[str, Any]]:
    short_max, medium_max = _compute_length_thresholds(records)
    buckets: dict[str, list[dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for record in records:
        length = int(record.get("sentence_token_length") or 0)
        if length <= short_max:
            bucket = "short"
        elif length <= medium_max:
            bucket = "medium"
        else:
            bucket = "long"
        buckets.setdefault(bucket, []).append(record)

    per_bucket_cap = max(1, max_sentences // 3)
    selected: list[dict[str, Any]] = []
    for bucket_name in ("short", "medium", "long"):
        bucket_records = buckets.get(bucket_name, [])
        bucket_records.sort(
            key=lambda rec: (
                -float(rec["score"]),
                -(rec["sentence_token_length"]),
                -(rec["og_quality_score"] or 0.0),
                -(rec["og_language_score"] or 0.0),
                -(rec["og_token_count"] or 0),
            )
        )
        selected.extend(bucket_records[:per_bucket_cap])

    if len(selected) < max_sentences:
        leftovers = [
            rec
            for bucket_name in ("short", "medium", "long")
            for rec in buckets.get(bucket_name, [])[per_bucket_cap:]
        ]
        leftovers.sort(
            key=lambda rec: (
                -float(rec["score"]),
                -(rec["sentence_token_length"]),
                -(rec["og_quality_score"] or 0.0),
                -(rec["og_language_score"] or 0.0),
                -(rec["og_token_count"] or 0),
            )
        )
        selected.extend(leftovers[: max(0, max_sentences - len(selected))])

    return selected[:max_sentences]


def _annotate_record(record: dict[str, Any], lang: str) -> dict[str, Any]:
    annotated = dict(record)
    annotated["lang"] = lang
    return annotated


def _record_signature(record: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return a stable fingerprint for a cached record."""
    return tuple(sorted(record.items()))


def _records_to_rows(records_by_lang: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lang, records in records_by_lang.items():
        for record in records:
            row = dict(record)
            row["lang"] = lang
            rows.append(row)
    return rows


def _rows_to_sentence_map(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        lang = row.get("lang")
        sentence = row.get("sentence")
        if isinstance(lang, str) and isinstance(sentence, str) and sentence.strip():
            result.setdefault(lang, []).append(sentence)
    return result


def _load_finetrans_cache_map(cache_dir: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not os.path.isdir(cache_dir):
        return result
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".parquet"):
            continue
        lang = name[:-8]
        path = os.path.join(cache_dir, name)
        rows = _read_records_parquet(path)
        sentences = [
            row.get("sentence")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("sentence"), str) and row["sentence"].strip()
        ]
        if sentences:
            result[lang] = sentences
        else:
            result[lang] = []
    return result


def _write_finetrans_cache_map(cache_dir: str, sentence_map: dict[str, list[str]]) -> dict[str, int]:
    os.makedirs(cache_dir, exist_ok=True)
    lang_counts: dict[str, int] = {}
    for lang, sentences in sorted(sentence_map.items()):
        path = os.path.join(cache_dir, f"{lang}.parquet")
        write_sentence_parquet(path, sentences)
        lang_counts[lang] = len(sentences)
    return lang_counts


def _dedupe_sentence_list(sentences: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    deduped: list[str] = []
    removed = 0
    for sentence in sentences:
        if not isinstance(sentence, str):
            removed += 1
            continue
        cleaned = sentence.strip()
        if not cleaned:
            removed += 1
            continue
        if cleaned in seen:
            removed += 1
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped, removed


def _read_records_parquet(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return []
    records = frame.to_dict(orient="records")
    return [record for record in records if isinstance(record, dict)]


def _write_record_batch(writer, pa_module, batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    table = pa_module.Table.from_pylist(batch)
    writer.write_table(table)


def _finetrans_cache_meta(
    *,
    seed: int,
    max_sentences_per_lang: int,
    overflow_sentences_per_lang: int,
    max_row_index: int,
    max_miss_streak: int,
    english_accept_every: int,
    include_translated_english: bool,
    configs: list[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "dataset": FINETRANS_DATASET,
        "cache_layout": "per_language_parquet_v1",
        "seed": seed,
        "max_sentences_per_lang": max_sentences_per_lang,
        "overflow_sentences_per_lang": overflow_sentences_per_lang,
        "max_row_index": max_row_index,
        "max_miss_streak": max_miss_streak,
        "english_accept_every": english_accept_every,
        "include_translated_english": include_translated_english,
        "configs": [[config, lang] for config, lang in configs],
        "cache_format": "parquet_records_v1",
    }


def _load_finetrans_meta(
    *,
    cache_file: str,
    cache_meta: str,
    expected_meta: dict[str, Any],
) -> dict[str, Any] | None:
    if not os.path.exists(cache_file) or not os.path.exists(cache_meta):
        return None
    try:
        with open(cache_meta, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    def _normalize_configs(value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized: list[list[Any]] = []
        for item in value:
            if isinstance(item, list) and len(item) >= 2:
                normalized.append([item[0], item[1]])
            else:
                normalized.append(item)
        return normalized
    if _normalize_configs(meta.get("configs")) != _normalize_configs(expected_meta.get("configs")):
        return None
    for key, value in expected_meta.items():
        if key == "configs":
            continue
        if meta.get(key) != value:
            return None
    return meta


def _load_finetrans_records_from_configs(config_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(config_root):
        return rows
    for root, _, files in os.walk(config_root):
        for name in files:
            if name in {"source.parquet", "en.parquet"}:
                rows.extend(_read_records_parquet(os.path.join(root, name)))
    return rows


def _has_finetrans_temp_shards(config_root: str) -> bool:
    if not os.path.isdir(config_root):
        return False
    for root, _, files in os.walk(config_root):
        if "source.parquet" in files or "en.parquet" in files:
            return True
    return False


def _clear_finetrans_config_dir(config_dir: str) -> None:
    if not os.path.isdir(config_dir):
        return
    for root, _, files in os.walk(config_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                os.remove(path)
            except OSError:
                pass


def _load_config_checkpoint_meta(config_meta_path: str, expected_meta: dict[str, Any]) -> dict[str, Any] | None:
    if not os.path.exists(config_meta_path):
        return None
    try:
        with open(config_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if meta and meta.get("status") == "checkpoint" and meta.get("dataset") == expected_meta["dataset"]:
        if meta.get("config") == expected_meta.get("config") and meta.get("lang") == expected_meta.get("lang"):
            return meta
    return None


def _load_config_complete_meta(config_meta_path: str, expected_meta: dict[str, Any]) -> dict[str, Any] | None:
    if not os.path.exists(config_meta_path):
        return None
    try:
        with open(config_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if meta and meta.get("status") == "complete" and meta.get("dataset") == expected_meta["dataset"]:
        if meta.get("config") == expected_meta.get("config") and meta.get("lang") == expected_meta.get("lang"):
            return meta
    return None


def _load_config_meta_raw(config_meta_path: str) -> dict[str, Any] | None:
    if not os.path.exists(config_meta_path):
        return None
    try:
        with open(config_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _process_finetrans_config(
    *,
    config_idx: int,
    config: str,
    lang: str,
    sentences_dir: str,
    lang_to_group: dict[str, str],
    seed: int,
    max_row_index: int,
    max_miss_streak: int,
    overflow_sentences_per_lang: int,
    english_accept_every: int,
    include_translated_english: bool,
    expected_meta: dict[str, Any],
    force_rebuild: bool = False,
) -> dict[str, Any]:
    config_dir = _finetrans_config_dir(sentences_dir, config_idx)
    english_dir = _finetrans_english_config_dir(sentences_dir, config_idx)
    config_records_path = _finetrans_config_records_path(sentences_dir, config_idx)
    english_records_path = _finetrans_english_records_path(sentences_dir, config_idx)
    config_meta_path = _finetrans_config_meta_path(sentences_dir, config_idx)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(english_dir, exist_ok=True)

    complete_meta = None if force_rebuild else _load_config_complete_meta(config_meta_path, expected_meta)
    if complete_meta is not None:
        return {
            "config_idx": config_idx,
            "config": config,
            "lang": lang,
            "status": "complete",
            "config_dir": config_dir,
            "accepted_rows": int(complete_meta.get("accepted_rows", 0)),
        }

    checkpoint_meta = None if force_rebuild else _load_config_checkpoint_meta(config_meta_path, expected_meta)
    raw_meta = None if force_rebuild else _load_config_meta_raw(config_meta_path)
    source_buffer: list[dict[str, Any]] = _read_records_parquet(config_records_path)
    english_buffer: list[dict[str, Any]] = _read_records_parquet(english_records_path)

    if force_rebuild or (checkpoint_meta is None and not source_buffer and not english_buffer):
        _clear_finetrans_config_dir(config_dir)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(english_dir, exist_ok=True)
    if checkpoint_meta is None and raw_meta is not None:
        if raw_meta.get("status") in {"checkpoint", "complete"}:
            raw_config = raw_meta.get("config")
            raw_lang = raw_meta.get("lang")
            if raw_config == config and raw_lang == lang:
                checkpoint_meta = raw_meta
                print(f"  Recovering {config} ({lang}) from raw checkpoint metadata")

    if checkpoint_meta is None and (source_buffer or english_buffer):
        checkpoint_meta = {
            **expected_meta,
            "status": "checkpoint",
            "config_idx": config_idx,
            "config": config,
            "lang": lang,
            "accepted_rows": 0,
            "accepted_sentences": len(source_buffer),
            "accepted_english_sentences": len(english_buffer),
            "miss_streak": 0,
            "next_row_idx": 0,
            "english_seen_sentences": len(english_buffer),
        }
        write_json_atomic(config_meta_path, checkpoint_meta)

    accepted_rows = int(checkpoint_meta.get("accepted_rows", 0)) if checkpoint_meta else 0
    accepted_sentences = int(checkpoint_meta.get("accepted_sentences", 0)) if checkpoint_meta else len(source_buffer)
    accepted_english_sentences = int(checkpoint_meta.get("accepted_english_sentences", 0)) if checkpoint_meta else len(english_buffer)
    miss_streak = int(checkpoint_meta.get("miss_streak", 0)) if checkpoint_meta else 0
    english_seen_sentences = int(checkpoint_meta.get("english_seen_sentences", 0)) if checkpoint_meta else len(english_buffer)
    next_row_idx = int(checkpoint_meta.get("next_row_idx", 0)) if checkpoint_meta else 0

    source_seen = {_record_signature(record) for record in source_buffer}
    english_seen = {_record_signature(record) for record in english_buffer}

    if checkpoint_meta is not None:
        print(
            f"  Resuming {config} ({lang}) from row {next_row_idx}"
        )
    elif source_buffer or english_buffer:
        print(f"  Recovering partial {config} ({lang}) from cached temp files")

    def _flush_pending(*, row_idx: int, status: str) -> None:
        nonlocal source_buffer, english_buffer
        meta = dict(expected_meta)
        meta.update(
            {
                "status": status,
                "config_idx": config_idx,
                "config": config,
                "lang": lang,
                "accepted_rows": accepted_rows,
                "accepted_sentences": accepted_sentences,
                "accepted_english_sentences": accepted_english_sentences,
                "miss_streak": miss_streak,
                "next_row_idx": row_idx,
                "english_seen_sentences": english_seen_sentences,
            }
        )
        write_records_parquet(config_records_path, source_buffer, columns=FINETRANS_WORKER_COLUMNS)
        write_records_parquet(english_records_path, english_buffer, columns=FINETRANS_WORKER_COLUMNS)
        write_json_atomic(config_meta_path, meta)

    try:
        ds = load_dataset(FINETRANS_DATASET, config, split="train", streaming=True)
        ds = ds.shuffle(buffer_size=1000, seed=seed)
    except Exception as exc:
        return {
            "config_idx": config_idx,
            "config": config,
            "lang": lang,
            "status": "skipped",
            "error": str(exc),
            "config_dir": config_dir,
        }

    for row_idx, row in enumerate(ds):
        if row_idx < next_row_idx:
            continue
        if row_idx >= max_row_index:
            print(
                f"  Stopping {config} ({lang}) at row index {row_idx} "
                f"(max_row_index={max_row_index})"
            )
            break
        if accepted_sentences >= overflow_sentences_per_lang:
            print(
                f"  Stopping {config} ({lang}) after reaching overflow "
                f"sentence cap={overflow_sentences_per_lang}"
            )
            break
        if not isinstance(row, dict):
            continue
        if _row_is_wikipedia(row):
            continue
        source_records = _sentence_records_from_row(
            row,
            lang=lang,
            lang_to_group=lang_to_group,
        )
        _lang_score = _row_language_score(row) or 0
        if _lang_score < FINETRANS_MIN_LANGUAGE_SCORE:
            continue
        _count_row = _row_token_count(row) or 0
        if _count_row < FINETRANS_MIN_TOKEN_COUNT:
            continue
        if _row_source_language(row) and _row_source_language(row) != lang:
            continue
        if _row_base_score(row, lang) < FINETRANS_MIN_QUALITY_SCORE:
            continue

        if source_records:
            new_source_records: list[dict[str, Any]] = []
            for record in source_records:
                annotated = _annotate_record(record, lang)
                signature = _record_signature(annotated)
                if signature in source_seen:
                    continue
                source_seen.add(signature)
                new_source_records.append(annotated)
            if new_source_records:
                source_buffer.extend(new_source_records)
                accepted_rows += 1
                accepted_sentences += len(new_source_records)
                miss_streak = 0
        else:
            miss_streak += 1
            if miss_streak >= max_miss_streak:
                print(
                    f"  Stopping {config} ({lang}) after miss_streak={miss_streak} "
                    f"(max_miss_streak={max_miss_streak})"
                )
                break

        if _include_translated_english_for_lang(lang, include_translated_english):
            english_records = _sentence_records_from_row(
                row,
                lang="en",
                lang_to_group=lang_to_group,
                translated=True,
            )
            new_english_records: list[dict[str, Any]] = []
            for record in english_records:
                english_seen_sentences += 1
                if _row_is_translated_en_acceptable(row) and _should_keep_english_sentence(
                    english_seen_sentences,
                    english_accept_every,
                ):
                    annotated = _annotate_record(record, "en")
                    signature = _record_signature(annotated)
                    if signature in english_seen:
                        continue
                    english_seen.add(signature)
                    new_english_records.append(annotated)
            if new_english_records:
                english_buffer.extend(new_english_records)
                accepted_english_sentences += len(new_english_records)

        if (
            len(source_buffer) >= FINETRANS_CHECKPOINT_EVERY_ROWS
            or len(english_buffer) >= FINETRANS_CHECKPOINT_EVERY_ROWS
        ):
            _flush_pending(row_idx=row_idx + 1, status="checkpoint")

    _flush_pending(row_idx=0, status="complete")
    meta = dict(expected_meta)
    meta.update(
        {
            "status": "complete",
            "config_idx": config_idx,
            "config": config,
            "lang": lang,
            "accepted_rows": accepted_rows,
            "accepted_sentences": accepted_sentences,
            "accepted_english_sentences": accepted_english_sentences,
            "miss_streak": miss_streak,
            "next_row_idx": 0,
            "english_seen_sentences": english_seen_sentences,
        }
    )
    write_json_atomic(config_meta_path, meta)
    return {
        "config_idx": config_idx,
        "config": config,
        "lang": lang,
        "status": "complete",
        "config_dir": config_dir,
        "accepted_rows": accepted_rows,
        "accepted_english_sentences": accepted_english_sentences,
    }


def load_finetranslations_sentences(
    *,
    sentences_dir: str,
    lang_to_group: dict[str, str],
    force_rebuild: bool = False,
    seed: int = 42,
    max_sentences_per_lang: int = FT_MAX_SENTENCES_PER_LANG,
    overflow_sentences_per_lang: int = FT_OVERFLOW_SENTENCES_PER_LANG,
    max_row_index: int = FT_MAX_ROW_INDEX,
    max_miss_streak: int = FT_MAX_MISS_STREAK,
    include_translated_english: bool = FT_INCLUDE_TRANSLATED_ENGLISH,
    english_accept_every: int = FT_ENGLISH_ACCEPT_EVERY,
    max_workers: int | None = None,
) -> dict[str, list[str]]:
    configs = _matching_configs(lang_to_group)
    if not configs:
        print("No FineTranslations subsets matched the current language set.")
        return {}

    cache_dir = _finetrans_cache_dir(sentences_dir)
    cache_meta = _finetrans_meta_path(sentences_dir)
    expected_meta = _finetrans_cache_meta(
        seed=seed,
        max_sentences_per_lang=max_sentences_per_lang,
        overflow_sentences_per_lang=overflow_sentences_per_lang,
        max_row_index=max_row_index,
        max_miss_streak=max_miss_streak,
        english_accept_every=english_accept_every,
        include_translated_english=include_translated_english,
        configs=configs,
    )

    final_meta = _load_finetrans_meta(
        cache_file=cache_dir,
        cache_meta=cache_meta,
        expected_meta=expected_meta,
    )
    if not force_rebuild and final_meta is not None and final_meta.get("status") == "complete":
        print(f"Loading FineTranslations cache from {cache_dir}")
        result = _load_finetrans_cache_map(cache_dir)
        if result:
            total = sum(len(v) for v in result.values())
            print(f"  {len(result)} languages | {total:,} sentences total")
            return result

    config_root = os.path.join(sentences_dir, "_finetrans_tmp")
    os.makedirs(config_root, exist_ok=True)
    recover_from_temp = _has_finetrans_temp_shards(config_root)

    if not force_rebuild and (
        os.path.exists(cache_dir) or os.path.exists(cache_meta)
    ) and not recover_from_temp:
        raise RuntimeError(
            "FineTranslations cache metadata does not match the current config. "
            "Refusing to rebuild over the existing cache. "
            "Delete the cache files or set FT_FORCE_REBUILD=True to regenerate them."
        )
    if force_rebuild:
        _clear_finetrans_config_dir(config_root)

    max_workers = min(len(configs), max(1, max_workers or 1))
    if recover_from_temp and not force_rebuild:
        print("Recovering FineTranslations cache from existing temp shards ...")
    else:
        print(f"Loading FineTranslations ({len(configs)} subsets) ...")
        print(f"(Workers: {max_workers} processes | matched configs only)\n")

        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for config_idx, (config, lang) in enumerate(configs):
                futures[pool.submit(
                    _process_finetrans_config,
                    config_idx=config_idx,
                    config=config,
                    lang=lang,
                    sentences_dir=sentences_dir,
                    lang_to_group=lang_to_group,
                    seed=seed,
                    max_row_index=max_row_index,
                    max_miss_streak=max_miss_streak,
                    overflow_sentences_per_lang=overflow_sentences_per_lang,
                    english_accept_every=english_accept_every,
                    include_translated_english=include_translated_english,
                    expected_meta=expected_meta,
                    force_rebuild=force_rebuild,
                )] = (config_idx, config, lang)

            bars: dict[int, Any] = {}
            pending_futures = dict(futures)
            try:
                for future, (config_idx, config, lang) in futures.items():
                    bars[config_idx] = tqdm(
                        total=max_row_index,
                        desc=f"{config:<16} -> {lang}",
                        position=config_idx,
                        leave=False,
                        dynamic_ncols=True,
                    )

                while pending_futures:
                    done_futures = [future for future in pending_futures if future.done()]
                    for future in done_futures:
                        config_idx, config, lang = pending_futures.pop(future)
                        try:
                            outcome = future.result()
                        except Exception as exc:
                            tqdm.write(f"  Skipping {config}: {exc}")
                            continue
                        status = outcome.get("status", "unknown")
                        if status == "complete":
                            tqdm.write(
                                f"  {config:<16} -> {lang}: +{int(outcome.get('accepted_rows', 0)):,} rows accepted"
                            )
                        elif status == "skipped":
                            tqdm.write(f"  Skipping {config}: {outcome.get('error', 'unknown error')}")
                        bar = bars.get(config_idx)
                        if bar is not None:
                            bar.n = max(bar.n, bar.total or 0)
                            bar.refresh()
                            bar.close()
                            bars.pop(config_idx, None)

                    for future, (config_idx, config, lang) in pending_futures.items():
                        bar = bars.get(config_idx)
                        if bar is None:
                            continue
                        config_meta_path = _finetrans_config_meta_path(sentences_dir, config_idx)
                        if not os.path.exists(config_meta_path):
                            continue
                        try:
                            with open(config_meta_path, encoding="utf-8") as f:
                                meta = json.load(f)
                        except Exception:
                            continue
                        next_row_idx = int(meta.get("next_row_idx", 0))
                        accepted_rows = int(meta.get("accepted_rows", 0))
                        accepted_sentences = int(meta.get("accepted_sentences", 0))
                        accepted_english_sentences = int(meta.get("accepted_english_sentences", 0))
                        bar.total = max(bar.total or 0, max_row_index)
                        bar.n = min(next_row_idx, bar.total or next_row_idx)
                        bar.set_postfix_str(
                            f"rows {accepted_rows:,} | sent {accepted_sentences:,} | en {accepted_english_sentences:,}"
                        )
                        bar.refresh()

                    time.sleep(1.0)
            finally:
                for bar in bars.values():
                    try:
                        bar.close()
                    except Exception:
                        pass

    rows = _load_finetrans_records_from_configs(config_root)
    records_by_lang: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        lang = row.get("lang")
        if not isinstance(lang, str):
            continue
        records_by_lang.setdefault(lang, []).append({k: v for k, v in row.items() if k != "lang"})

    selected_records: dict[str, list[dict[str, Any]]] = {}
    result: dict[str, list[str]] = {}
    dedup_summary: dict[str, dict[str, int]] = {}
    total_before = 0
    total_after = 0
    total_removed = 0
    rng = random.Random(seed)
    for lang, records in sorted(records_by_lang.items()):
        kept_records = _select_bucketed_records(records, max_sentences_per_lang)
        rng.shuffle(kept_records)
        selected_records[lang] = kept_records
        raw_sentences = [record["sentence"] for record in kept_records]
        deduped_sentences, removed = _dedupe_sentence_list(raw_sentences)
        result[lang] = deduped_sentences
        dedup_summary[lang] = {
            "before": len(raw_sentences),
            "after": len(deduped_sentences),
            "removed": removed,
        }
        total_before += len(raw_sentences)
        total_after += len(deduped_sentences)
        total_removed += removed

    # Rebuilds are written atomically, so we keep any previous live cache in
    # place until the new parquet and metadata are fully ready.
    _write_finetrans_cache_map(cache_dir, result)
    write_json_atomic(
        cache_meta,
        {
            **expected_meta,
            "status": "complete",
            "deduped": True,
            "dedup_summary": dedup_summary,
            "lang_counts": {lang: len(sentences) for lang, sentences in result.items()},
            "total_before": total_before,
            "total_after": total_after,
            "total_removed": total_removed,
            "next_config_idx": len(configs),
            "next_row_idx": 0,
            "next_shard_idx": 0,
        },
    )
    _clear_finetrans_config_dir(config_root)

    total = sum(len(v) for v in result.values())
    print(f"\nFineTranslations sentences cached -> {cache_dir}/")
    print(f"  {len(result)} languages | {total:,} sentences total")
    for lang in sorted(result):
        print(f"  {lang:<6}  {len(result[lang]):>5} sentences")
    return result
