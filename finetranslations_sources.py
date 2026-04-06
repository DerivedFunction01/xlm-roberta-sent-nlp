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
FINETRANS_LATIN_MAX_ENGLISH_RATIO = 0.35
FINETRANS_LATIN_MIN_TOKENS = 4
FINETRANS_MIN_TOKEN_COUNT = 20
FINETRANS_LATIN_LONGEST_CHUNKS = 2
FINETRANS_MIN_QUALITY_SCORE = 0.40
FINETRANS_KEEP_TOP_PER_LANG = 50_000
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


def _looks_english_heavy(text: str) -> bool:
    tokens = _latin_tokens(text)
    if len(tokens) < FINETRANS_LATIN_MIN_TOKENS:
        return False
    sw = _english_stopwords()
    stopword_hits = sum(1 for token in tokens if token in sw)
    english_ratio = stopword_hits / max(1, len(tokens))
    return stopword_hits >= 3 and english_ratio >= FINETRANS_LATIN_MAX_ENGLISH_RATIO


def _latin_source_lines(chunks: list[str]) -> list[str]:
    selected_chunks = _longest_chunks(chunks)
    lines: list[str] = []
    seen: set[str] = set()
    for chunk in selected_chunks:
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _looks_english_heavy(line):
                continue
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
    return lines


def _row_sentence_score(row: dict[str, Any], lang: str) -> float:
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


def _segment_text(text: str, lang: str) -> list[str]:
    safe_text = sanitize_paragraph_for_pysbd(text)
    segmenter = _get_segmenter(lang)
    try:
        segments = segmenter.segment(safe_text) if segmenter else SENT_SPLIT.split(safe_text)
    except re.error:
        segments = SENT_SPLIT.split(safe_text)
    return [segment for segment in segments if isinstance(segment, str) and segment.strip()]


def _prepare_source_sentences(
    row: dict[str, Any],
    lang: str,
    lang_to_group: dict[str, str],
) -> list[str]:
    chunks = _row_chunks(row, "og_chunks", "og_full_text")
    if not chunks:
        return []
    if lang_to_group.get(lang) in LATIN_GROUPS:
        lines = _latin_source_lines(chunks)
        return post_clean_sentences(lines, lang, lang_to_group)
    sentences: list[str] = []
    for chunk in chunks:
        sentences.extend(_segment_text(chunk, lang))
    return post_clean_sentences(sentences, lang, lang_to_group)


def _extend_bucket(
    bucket: list[str],
    seen: set[str],
    scored_rows: list[tuple[float, dict[str, Any]]],
    *,
    lang: str,
    lang_to_group: dict[str, str],
    max_sentences: int,
) -> None:
    for _, row in scored_rows:
        if len(bucket) >= max_sentences:
            return
        score = _row_language_score(row)
        if score is not None and score < FINETRANS_MIN_LANGUAGE_SCORE:
            continue
        token_count = _row_token_count(row)
        if token_count is not None and token_count < FINETRANS_MIN_TOKEN_COUNT:
            continue
        row_lang = _row_source_language(row)
        if row_lang and row_lang != lang:
            continue
        cleaned = _prepare_source_sentences(row, lang, lang_to_group)
        for sentence in cleaned:
            if sentence in seen:
                continue
            seen.add(sentence)
            bucket.append(sentence)
            if len(bucket) >= max_sentences:
                return


def _sort_scored_rows(rows: list[dict[str, Any]], lang: str) -> list[tuple[float, dict[str, Any]]]:
    scored = [(_row_sentence_score(row, lang), row) for row in rows]
    scored.sort(key=lambda item: (-item[0], _row_token_count(item[1]) or 0))
    return scored


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
            }
            if meta == expected_meta:
                print(f"Loading FineTranslations cache from {cache_file}")
                with open(cache_file, encoding="utf-8") as f:
                    cached: dict[str, list[str]] = json.load(f)
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

    accumulator: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(seed)

    print(f"Loading FineTranslations ({len(configs)} subsets) ...")
    for config, lang in tqdm(configs, desc="FineTranslations subsets"):

        try:
            ds = load_dataset(FINETRANS_DATASET, config, split="train", streaming=True)
        except Exception as exc:
            print(f"  Skipping {config}: {exc}")
            continue
        ds = ds.shuffle(buffer_size=1000, seed=seed)

        bucket = accumulator.setdefault(lang, [])
        lang_seen = seen.setdefault(lang, set())
        rows = candidate_rows.setdefault(lang, [])

        for row in ds:
            if len(bucket) >= max_sentences_per_lang:
                break

            if not isinstance(row, dict):
                continue
            if _row_is_wikipedia(row):
                continue
            score = _row_sentence_score(row, lang)
            if score < FINETRANS_MIN_QUALITY_SCORE:
                continue
            rows.append(row)

            if include_translated_english:
                english_chunks = _row_chunks(row, "translated_chunks", "translated_text")
                if english_chunks:
                    english_bucket = accumulator.setdefault("en", [])
                    english_seen = seen.setdefault("en", set())
                    _extend_bucket(
                        english_bucket,
                        english_seen,
                        english_chunks,
                        lang="en",
                        lang_to_group=lang_to_group,
                        max_sentences=max_sentences_per_lang,
                    )

        scored_rows = _sort_scored_rows(rows, lang)
        _extend_bucket(
            bucket,
            lang_seen,
            scored_rows[:FINETRANS_KEEP_TOP_PER_LANG],
            lang=lang,
            lang_to_group=lang_to_group,
            max_sentences=max_sentences_per_lang,
        )
        added = len(bucket)
        print(f"  {config:<16} -> {lang}: +{added:,} sentences")

    result: dict[str, list[str]] = {}
    for lang, sentences in sorted(accumulator.items()):
        deduped = list(sentences)
        rng.shuffle(deduped)
        if len(deduped) > max_sentences_per_lang:
            deduped = deduped[:max_sentences_per_lang]
        result[lang] = deduped

    write_json_atomic(cache_file, result)
    write_json_atomic(
        cache_meta,
        {
            "dataset": FINETRANS_DATASET,
            "seed": seed,
            "max_sentences_per_lang": max_sentences_per_lang,
            "include_translated_english": include_translated_english,
        },
    )

    total = sum(len(v) for v in result.values())
    print(f"\nFineTranslations sentences cached -> {cache_file}")
    print(f"  {len(result)} languages | {total:,} sentences total")
    for lang in sorted(result):
        print(f"  {lang:<6}  {len(result[lang]):>5} sentences")
    return result
