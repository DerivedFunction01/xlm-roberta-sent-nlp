from __future__ import annotations

import json
import os
import random
import re
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic, write_records_parquet
from paths import (
    FINETRANS_CACHE_FILE,
    FINETRANS_CACHE_META,
    FINETRANS_TEMP_FILE,
    SENTENCES_DIR,
)
from language import ENGLISH_STOP_WORDS, LANG_ISO2_TO_ISO3
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
FINETRANS_LATIN_SHORT_TOKEN_LIMIT = 4
FINETRANS_MIN_TOKEN_COUNT = 20
FINETRANS_LATIN_LONGEST_CHUNKS = 2
FINETRANS_MIN_QUALITY_SCORE = 0.80
FINETRANS_CHECKPOINT_EVERY_ROWS = 2_500
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WIKIPEDIA_URL_RE = re.compile(r"wikipedia(?:\.org)?", flags=re.IGNORECASE)
FT_ISO3_TO_LANG = {iso3: lang for lang, iso3 in LANG_ISO2_TO_ISO3.items()}


def _finetrans_cache_path(sentences_dir: str) -> str:
    return FINETRANS_CACHE_FILE if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "finetranslations_sentences.parquet",
    )


def _finetrans_meta_path(sentences_dir: str) -> str:
    return FINETRANS_CACHE_META if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "finetranslations_sentences.meta.json",
    )


def _finetrans_temp_path(sentences_dir: str) -> str:
    return FINETRANS_TEMP_FILE if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "_finetrans_tmp",
        "finetranslations_sentences.parquet",
    )


def _finetrans_shard_dir(sentences_dir: str) -> str:
    return os.path.join(sentences_dir, "_finetrans_tmp", "shards")


def _finetrans_shard_path(sentences_dir: str, shard_idx: int) -> str:
    return os.path.join(_finetrans_shard_dir(sentences_dir), f"part-{shard_idx:05d}.parquet")


def _finetrans_config_dir(sentences_dir: str, config_idx: int) -> str:
    return os.path.join(sentences_dir, "_finetrans_tmp", f"config_{config_idx:05d}")


def _finetrans_config_meta_path(sentences_dir: str, config_idx: int) -> str:
    return os.path.join(_finetrans_config_dir(sentences_dir, config_idx), "checkpoint.meta.json")


def _finetrans_config_shard_path(sentences_dir: str, config_idx: int, shard_idx: int) -> str:
    return os.path.join(
        _finetrans_config_dir(sentences_dir, config_idx),
        f"part-{shard_idx:05d}.parquet",
    )


def _config_name_to_lang(config_name: str, lang_to_group: dict[str, str]) -> str | None:
    config_name = config_name.rsplit("/", 1)[-1]
    if config_name == "all":
        return None
    base = config_name.split("_", 1)[0]
    if len(base) == 2 and base in lang_to_group:
        return base
    return FT_ISO3_TO_LANG.get(base) if FT_ISO3_TO_LANG.get(base) in lang_to_group else None


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
        return float(score)
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
    if len(tokens) <= FINETRANS_LATIN_SHORT_TOKEN_LIMIT:
        return sum(1 for token in tokens if _is_english_word(token)) >= 1
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


def _read_records_parquet(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return []
    records = frame.to_dict(orient="records")
    return [record for record in records if isinstance(record, dict)]


def _finetrans_cache_meta(
    *,
    seed: int,
    max_sentences_per_lang: int,
    include_translated_english: bool,
    configs: list[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "dataset": FINETRANS_DATASET,
        "seed": seed,
        "max_sentences_per_lang": max_sentences_per_lang,
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
    return meta if meta == expected_meta else None


def _load_finetrans_checkpoint_meta(
    *,
    cache_meta: str,
    expected_meta: dict[str, Any],
) -> dict[str, Any] | None:
    if not os.path.exists(cache_meta):
        return None
    try:
        with open(cache_meta, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if meta == expected_meta and meta.get("status") == "checkpoint":
        return meta
    return None


def _load_finetrans_records_from_shards(shard_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(shard_dir):
        return rows
    parquet_paths: list[str] = []
    for root, _, files in os.walk(shard_dir):
        for name in files:
            if name.endswith(".parquet"):
                parquet_paths.append(os.path.join(root, name))
    for path in sorted(parquet_paths):
        rows.extend(_read_records_parquet(path))
    return rows


def _clear_finetrans_shards(shard_dir: str) -> None:
    if not os.path.isdir(shard_dir):
        return
    for root, dirs, files in os.walk(shard_dir, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            try:
                os.remove(path)
            except OSError:
                pass
        for name in dirs:
            path = os.path.join(root, name)
            try:
                os.rmdir(path)
            except OSError:
                pass


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


def _process_finetrans_config(
    *,
    config_idx: int,
    config: str,
    lang: str,
    sentences_dir: str,
    lang_to_group: dict[str, str],
    seed: int,
    include_translated_english: bool,
    expected_meta: dict[str, Any],
    force_rebuild: bool = False,
) -> dict[str, Any]:
    config_dir = _finetrans_config_dir(sentences_dir, config_idx)
    config_meta_path = _finetrans_config_meta_path(sentences_dir, config_idx)
    os.makedirs(config_dir, exist_ok=True)

    if force_rebuild:
        _clear_finetrans_config_dir(config_dir)

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
    next_row_idx = int(checkpoint_meta.get("next_row_idx", 0)) if checkpoint_meta else 0
    next_shard_idx = int(checkpoint_meta.get("next_shard_idx", 0)) if checkpoint_meta else 0

    pending_records: list[dict[str, Any]] = []
    accepted_rows = int(checkpoint_meta.get("accepted_rows", 0)) if checkpoint_meta else 0

    if checkpoint_meta is not None:
        print(
            f"  Resuming {config} ({lang}) from row {next_row_idx} shard {next_shard_idx}"
        )

    def _flush_pending(*, row_idx: int, shard_idx: int, status: str) -> int:
        nonlocal pending_records, accepted_rows
        if pending_records:
            shard_path = _finetrans_config_shard_path(sentences_dir, config_idx, shard_idx)
            write_records_parquet(shard_path, pending_records)
            shard_idx += 1
            pending_records = []
        meta = dict(expected_meta)
        meta.update(
            {
                "status": status,
                "config_idx": config_idx,
                "config": config,
                "lang": lang,
                "accepted_rows": accepted_rows,
                "next_row_idx": row_idx,
                "next_shard_idx": shard_idx,
            }
        )
        write_json_atomic(config_meta_path, meta)
        return shard_idx

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
        if not isinstance(row, dict):
            continue
        if _row_is_wikipedia(row):
            continue
        source_records = _sentence_records_from_row(row, lang=lang, lang_to_group=lang_to_group)
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
            pending_records.extend(_annotate_record(record, lang) for record in source_records)
            accepted_rows += 1

        if include_translated_english:
            english_records = _sentence_records_from_row(
                row,
                lang="en",
                lang_to_group=lang_to_group,
                translated=True,
            )
            for record in english_records:
                if _row_is_translated_en_acceptable(row):
                    pending_records.append(_annotate_record(record, "en"))

        if len(pending_records) >= FINETRANS_CHECKPOINT_EVERY_ROWS:
            next_shard_idx = _flush_pending(
                row_idx=row_idx + 1,
                shard_idx=next_shard_idx,
                status="checkpoint",
            )

    next_shard_idx = _flush_pending(row_idx=0, shard_idx=next_shard_idx, status="complete")
    meta = dict(expected_meta)
    meta.update(
        {
            "status": "complete",
            "config_idx": config_idx,
            "config": config,
            "lang": lang,
            "accepted_rows": accepted_rows,
            "next_row_idx": 0,
            "next_shard_idx": next_shard_idx,
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
        "shards": next_shard_idx,
    }


def load_finetranslations_sentences(
    *,
    sentences_dir: str,
    lang_to_group: dict[str, str],
    force_rebuild: bool = False,
    seed: int = 42,
    max_sentences_per_lang: int = 5_000,
    include_translated_english: bool = False,
    max_workers: int | None = None,
) -> dict[str, list[str]]:
    configs = _matching_configs(lang_to_group)
    if not configs:
        print("No FineTranslations subsets matched the current language set.")
        return {}

    cache_file = _finetrans_cache_path(sentences_dir)
    cache_meta = _finetrans_meta_path(sentences_dir)
    temp_file = _finetrans_temp_path(sentences_dir)
    expected_meta = _finetrans_cache_meta(
        seed=seed,
        max_sentences_per_lang=max_sentences_per_lang,
        include_translated_english=include_translated_english,
        configs=configs,
    )

    final_meta = _load_finetrans_meta(
        cache_file=cache_file,
        cache_meta=cache_meta,
        expected_meta=expected_meta,
    )
    if not force_rebuild and final_meta is not None and final_meta.get("status") == "complete":
        print(f"Loading FineTranslations cache from {cache_file}")
        rows = _read_records_parquet(cache_file)
        result = _rows_to_sentence_map(rows)
        total = sum(len(v) for v in result.values())
        print(f"  {len(result)} languages | {total:,} sentences total")
        return result

    if force_rebuild:
        for path in (cache_file, cache_meta):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    config_root = os.path.join(sentences_dir, "_finetrans_tmp", "configs")
    os.makedirs(config_root, exist_ok=True)
    if force_rebuild:
        _clear_finetrans_shards(config_root)

    max_workers = min(len(configs), max(1, max_workers or 1))
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
                include_translated_english=include_translated_english,
                expected_meta=expected_meta,
                force_rebuild=force_rebuild,
            )] = (config_idx, config, lang)

        for future in tqdm(as_completed(futures), total=len(futures), desc="FineTranslations configs"):
            config_idx, config, lang = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                print(f"  Skipping {config}: {exc}")
                continue
            status = outcome.get("status", "unknown")
            if status == "complete":
                print(
                    f"  {config:<16} -> {lang}: +{int(outcome.get('accepted_rows', 0)):,} rows accepted"
                )
            elif status == "skipped":
                print(f"  Skipping {config}: {outcome.get('error', 'unknown error')}")

    rows = _load_finetrans_records_from_shards(config_root)
    records_by_lang: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        lang = row.get("lang")
        if not isinstance(lang, str):
            continue
        records_by_lang.setdefault(lang, []).append({k: v for k, v in row.items() if k != "lang"})

    selected_records: dict[str, list[dict[str, Any]]] = {}
    result: dict[str, list[str]] = {}
    rng = random.Random(seed)
    for lang, records in sorted(records_by_lang.items()):
        kept_records = _select_bucketed_records(records, max_sentences_per_lang)
        rng.shuffle(kept_records)
        selected_records[lang] = kept_records
        result[lang] = [record["sentence"] for record in kept_records]

    write_records_parquet(cache_file, _records_to_rows(selected_records))
    write_json_atomic(
        cache_meta,
        {
            **expected_meta,
            "status": "complete",
            "next_config_idx": len(configs),
            "next_row_idx": 0,
            "next_shard_idx": 0,
        },
    )
    _clear_finetrans_shards(config_root)

    total = sum(len(v) for v in result.values())
    print(f"\nFineTranslations sentences cached -> {cache_file}")
    print(f"  {len(result)} languages | {total:,} sentences total")
    for lang in sorted(result):
        print(f"  {lang:<6}  {len(result[lang]):>5} sentences")
    return result
