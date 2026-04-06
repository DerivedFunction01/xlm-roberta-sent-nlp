from __future__ import annotations

import json
import os
import random
import re
from typing import Any

from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic
from paths import FINETRANS_CACHE_FILE, FINETRANS_CACHE_META, SENTENCES_DIR
from language import LANG_ISO2_TO_ISO3
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
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WIKIPEDIA_URL_RE = re.compile(r"wikipedia(?:\.org)?", flags=re.IGNORECASE)
FT_ISO3_TO_LANG = {iso3: lang for lang, iso3 in LANG_ISO2_TO_ISO3.items()}


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


def _english_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords  # type: ignore

        return set(stopwords.words("english"))
    except Exception:
        return {
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
            "for", "from", "have", "has", "he", "her", "his", "i", "in", "is",
            "it", "its", "of", "on", "or", "our", "she", "that", "the", "their",
            "there", "these", "they", "this", "to", "was", "we", "were", "with",
            "you", "your",
        }


def _latin_tokens(text: str) -> list[str]:
    return LATIN_TOKEN_RE.findall(text.lower())


def _sentence_token_length(sentence: str) -> int:
    return len(_latin_tokens(sentence))


def _looks_english_heavy(text: str) -> bool:
    tokens = _latin_tokens(text)
    if len(tokens) < FINETRANS_LATIN_MIN_TOKENS:
        return False
    sw = _english_stopwords()
    stopword_hits = sum(1 for token in tokens if token in sw)
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


def _serialize_records(records_by_lang: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        lang: [
            {
                "sentence": record["sentence"],
                "sentence_token_length": record["sentence_token_length"],
                "og_language_score": record["og_language_score"],
                "og_token_count": record["og_token_count"],
                "og_quality_score": record["og_quality_score"],
                "edu_score_raw": record["edu_score_raw"],
                "source_language": record["source_language"],
                "is_wikipedia": record["is_wikipedia"],
                "score": record["score"],
            }
            for record in records
        ]
        for lang, records in records_by_lang.items()
    }


def _normalize_cached_records(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, list[str]] = {}
    for lang, items in payload.items():
        if not isinstance(items, list):
            continue
        sentences: list[str] = []
        for item in items:
            if isinstance(item, str):
                sentences.append(item)
            elif isinstance(item, dict):
                sentence = item.get("sentence")
                if isinstance(sentence, str) and sentence.strip():
                    sentences.append(sentence)
        result[lang] = sentences
    return result


def load_finetranslations_sentences(
    *,
    sentences_dir: str,
    lang_to_group: dict[str, str],
    force_rebuild: bool = False,
    seed: int = 42,
    max_sentences_per_lang: int = 5_000,
    include_translated_english: bool = False,
) -> dict[str, list[str]]:
    cache_file = FINETRANS_CACHE_FILE if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "finetranslations_sentences.json",
    )
    cache_meta = FINETRANS_CACHE_META if sentences_dir == SENTENCES_DIR else os.path.join(
        sentences_dir,
        "finetranslations_sentences.meta.json",
    )

    if not force_rebuild and os.path.exists(cache_file) and os.path.exists(cache_meta):
        try:
            with open(cache_meta, encoding="utf-8") as f:
                meta = json.load(f)
            expected_meta = {
                "dataset": FINETRANS_DATASET,
                "seed": seed,
                "max_sentences_per_lang": max_sentences_per_lang,
                "include_translated_english": include_translated_english,
                "cache_format": "records_v1",
            }
            if meta == expected_meta:
                print(f"Loading FineTranslations cache from {cache_file}")
                with open(cache_file, encoding="utf-8") as f:
                    cached = _normalize_cached_records(json.load(f))
                print(
                    f"  {len(cached)} languages | "
                    f"{sum(len(v) for v in cached.values()):,} sentences total"
                )
                return cached
        except Exception:
            pass

    configs = _matching_configs(lang_to_group)
    if not configs:
        print("No FineTranslations subsets matched the current language set.")
        return {}

    candidate_records: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(seed)

    print(f"Loading FineTranslations ({len(configs)} subsets) ...")
    for config, lang in tqdm(configs, desc="FineTranslations subsets"):

        try:
            ds = load_dataset(FINETRANS_DATASET, config, split="train", streaming=True)
        except Exception as exc:
            print(f"  Skipping {config}: {exc}")
            continue
        ds = ds.shuffle(buffer_size=1000, seed=seed)

        records = candidate_records.setdefault(lang, [])

        for row in ds:
            if not isinstance(row, dict):
                continue
            if _row_is_wikipedia(row):
                continue
            source_records = _sentence_records_from_row(row, lang=lang, lang_to_group=lang_to_group)
            _lang_score = _row_language_score(row) or 0
            if  _lang_score < FINETRANS_MIN_LANGUAGE_SCORE:
                continue
            _count_row = _row_token_count(row) or 0
            if  _count_row < FINETRANS_MIN_TOKEN_COUNT:
                continue
            if _row_source_language(row) and _row_source_language(row) != lang:
                continue
            if _row_base_score(row, lang) < FINETRANS_MIN_QUALITY_SCORE:
                continue
            records.extend(source_records)

            if include_translated_english:
                english_records = _sentence_records_from_row(
                    row,
                    lang="en",
                    lang_to_group=lang_to_group,
                    translated=True,
                )
                for record in english_records:
                    if not _row_is_translated_en_acceptable(row):
                        continue
                    candidate_records.setdefault("en", []).append(record)

        kept_records = _select_bucketed_records(records, max_sentences_per_lang)
        print(f"  {config:<16} -> {lang}: +{len(kept_records):,} sentences")
        candidate_records[lang] = kept_records

    result: dict[str, list[str]] = {}
    for lang, records in sorted(candidate_records.items()):
        sentences = [record["sentence"] for record in records]
        rng.shuffle(sentences)
        if len(sentences) > max_sentences_per_lang:
            sentences = sentences[:max_sentences_per_lang]
        result[lang] = sentences

    write_json_atomic(cache_file, _serialize_records(candidate_records))
    write_json_atomic(
        cache_meta,
        {
            "dataset": FINETRANS_DATASET,
            "seed": seed,
            "max_sentences_per_lang": max_sentences_per_lang,
            "include_translated_english": include_translated_english,
            "cache_format": "records_v1",
        },
    )

    total = sum(len(v) for v in result.values())
    print(f"\nFineTranslations sentences cached -> {cache_file}")
    print(f"  {len(result)} languages | {total:,} sentences total")
    for lang in sorted(result):
        print(f"  {lang:<6}  {len(result[lang]):>5} sentences")
    return result
