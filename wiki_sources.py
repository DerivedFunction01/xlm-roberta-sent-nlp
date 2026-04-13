from __future__ import annotations

from collections import deque
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic as _write_json_atomic, write_sentence_parquet as _write_sentence_parquet
from language import ALL_LANGS, LANG_TO_GROUP, LANGUAGE_GROUP_MIN_CHARS, LATIN_GROUPS, canonical_lang
from paths import PATHS
from text_utils import (
    _article_min_chars,
    _get_segmenter,
    _is_valid_sentence,
    _split_paragraphs,
    clean_sentence,
    log_segmentation_failure,
    _looks_like_english_text,
    post_clean_sentences,
    sanitize_paragraph_for_pysbd,
    SENT_SPLIT,
)



MAX_WIKI_INDEX = 100_000
ARTICLES_PER_LANG = 10_000
MAX_WIKI_SENTENCES = 150_000
WIKI_PARAGRAPH_FRACTION_DEFAULT = 0.60
WIKI_PARAGRAPH_FRACTIONS_BY_GROUP: dict[str, float] = {}
WIKI_LONG_PARAGRAPH_STREAK = 2
WIKI_LONG_PARAGRAPH_BONUS = 0.10
WIKI_CAP_MULTIPLIERS = {
    "en": 1.50,
    "de": 1.25,
    "fr": 1.25,
    "es": 1.25,
    "ru": 1.25,
    "zh": 1.25,
    "ja": 1.25,
    "pt": 1.25,
    "it": 1.25,
    "hi": 1.25,
    "ko": 1.25,
    "ar": 1.25,
    "no": 0.5,
}
WIKI_SOURCE_LANGS = {
    "no": ("no", "nn"),
}
WIKI_LATIN_SECONDARY_GROUPS = {
    "AfricanLatin",
    "AdriaticLatin",
    "BalticLatin",
    "CelticLatin",
    "KurdishLatin",
    "PeripheralLatin",
    "WesternLatin",
}
WIKI_ROLLING_STATS_WINDOW = 250
LENGTH_PRIORITY_SCAN_LIMIT = int(MAX_WIKI_INDEX // 1.5)
LENGTH_PRIORITY_SENTENCE_CAP_BY_LANG = {
    "vi": 50_000,
    "sv": 50_000,
    "lo": 20_000,
    "sd": 20_000,
    "am": 20_000,
    "km": 20_000,
    "ug": 20_000,
    "my": 20_000,
}
LENGTH_PRIORITY_LANGS = set(LENGTH_PRIORITY_SENTENCE_CAP_BY_LANG)
LENGTH_PRIORITY_SENTENCE_CAP = 25_000


def _is_missing_wiki_config_error(exc: Exception, lang: str) -> bool:
    message = str(exc).lower()
    config_name = f"20231101.{lang}".lower()
    return (
        config_name in message
        and (
            "config" in message
            or "configuration" in message
            or "available configs" in message
            or "not found" in message
            or "does not exist" in message
        )
    )


def _wiki_source_langs(lang: str) -> tuple[str, ...]:
    lang = canonical_lang(lang)
    return WIKI_SOURCE_LANGS.get(lang, (lang,))


def _wiki_cache_meta_path(sentences_dir: str, lang: str) -> str:
    return os.path.join(sentences_dir, f"{lang}.meta.json")


def _write_wiki_cache_meta(cache_meta_path: str, lang: str, source_langs: list[str]) -> None:
    _write_json_atomic(
        cache_meta_path,
        {
            "lang": canonical_lang(lang),
            "source_langs": source_langs,
            "status": "complete",
            "cache_origin": "external_parquet",
        },
    )


def _assign_wiki_paragraph_fraction(fraction: float, *groups: str) -> None:
    for group in groups:
        WIKI_PARAGRAPH_FRACTIONS_BY_GROUP[group] = fraction


_assign_wiki_paragraph_fraction(0.75, "English")


def _wiki_paragraph_fraction(lang: str, lang_to_group: dict[str, str]) -> float:
    group = lang_to_group.get(canonical_lang(lang))
    return WIKI_PARAGRAPH_FRACTIONS_BY_GROUP.get(group, WIKI_PARAGRAPH_FRACTION_DEFAULT)


def _wiki_long_paragraph_chars(lang: str, lang_to_group: dict[str, str]) -> int:
    group = lang_to_group.get(canonical_lang(lang))
    group_min_chars = LANGUAGE_GROUP_MIN_CHARS.get(group or "", 2_000)
    return max(250, group_min_chars // 4)


def _wiki_use_nltk_secondary(lang: str, lang_to_group: dict[str, str]) -> bool:
    group = lang_to_group.get(canonical_lang(lang))
    if group is None:
        return True
    if group not in LATIN_GROUPS:
        return True
    return group in WIKI_LATIN_SECONDARY_GROUPS


def _load_wiki_dataset(source_lang: str, *, seed: int):
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{source_lang}",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        if _is_missing_wiki_config_error(exc, source_lang):
            print(f"  Skipping {source_lang}: Wikipedia config 20231101.{source_lang} was not found")
            return None
        raise
    return dataset.shuffle(buffer_size=1000, seed=seed)

def parquet_path(sentences_dir: str, lang: str) -> str:
    return os.path.join(sentences_dir, f"{lang}.parquet")


def prepare_wiki_paragraphs(text: str, lang: str, lang_to_group: dict[str, str]) -> list[str] | None:
    paragraphs = _split_paragraphs(text, lang, lang_to_group)
    if paragraphs is None:
        return None

    fraction = _wiki_paragraph_fraction(lang, lang_to_group)
    long_paragraph_chars = _wiki_long_paragraph_chars(lang, lang_to_group)
    long_streak = 0
    for paragraph in paragraphs:
        if len(paragraph) >= long_paragraph_chars:
            long_streak += 1
            if long_streak >= WIKI_LONG_PARAGRAPH_STREAK:
                fraction = min(0.90, fraction + WIKI_LONG_PARAGRAPH_BONUS)
                break
        else:
            long_streak = 0

    cutoff = max(1, int(len(paragraphs) * fraction))
    selected = paragraphs[:cutoff]
    selected.sort(key=len, reverse=True)
    return selected


def _extract_article_sentences(
    article_text: str,
    lang: str,
    segmenter,
    article_idx: int,
    *,
    lang_to_group: dict[str, str],
    article_title: str = "",
) -> list[str]:
    paragraphs = prepare_wiki_paragraphs(article_text, lang, lang_to_group)
    if paragraphs is None:
        return []
    use_nltk_secondary = _wiki_use_nltk_secondary(lang, lang_to_group)

    article_batch: list[str] = []
    for paragraph_idx, paragraph in enumerate(paragraphs):
        safe_paragraph = sanitize_paragraph_for_pysbd(paragraph)
        try:
            sents = segmenter.segment(safe_paragraph) if segmenter else SENT_SPLIT.split(safe_paragraph)  # type: ignore[attr-defined]
        except re.error as exc:
            log_segmentation_failure(
                os.path.join(PATHS["wiki"]["seg_debug_dir"], f"{lang}.log"),
                lang=lang,
                article_idx=article_idx,
                paragraph_idx=paragraph_idx,
                paragraph=paragraph,
                exc=exc,
                article_title=article_title,
            )
            sents = SENT_SPLIT.split(safe_paragraph)
        for s in sents:
            s = clean_sentence(
                s,
                lang,
                lang_to_group,
                use_nltk_secondary=use_nltk_secondary,
            )
            if _is_valid_sentence(s, lang, lang_to_group):
                article_batch.append(s)

    return article_batch


def _collect_priority_articles(dataset, lang: str, scan_limit: int, *, lang_to_group: dict[str, str]) -> list[tuple[int, int, dict]]:
    min_chars = _article_min_chars(lang, lang_to_group)
    candidates: list[tuple[int, int, dict]] = []
    for article_idx, article in enumerate(dataset.take(scan_limit)):
        article_text = article.get("text", "")
        if len(article_text) < min_chars:
            continue
        candidates.append((len(article_text), article_idx, article))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def temp_parquet_path(lang: str) -> str:
    return os.path.join(PATHS["wiki"]["temp_dir"], f"{lang}.parquet")


def temp_meta_path(lang: str) -> str:
    return os.path.join(PATHS["wiki"]["temp_dir"], f"{lang}.meta.json")


def _wiki_cap_multiplier(source_langs: tuple[str, ...]) -> float:
    return sum(WIKI_CAP_MULTIPLIERS.get(canonical_lang(source_lang), 1.0) for source_lang in source_langs)


def max_wiki_sentences_for_lang(lang: str, *, source_langs: tuple[str, ...] | None = None) -> int:
    source_langs = source_langs or (lang,)
    multiplier = _wiki_cap_multiplier(source_langs)
    return int(round(MAX_WIKI_SENTENCES * multiplier))


def max_length_priority_sentences_for_lang(lang: str, *, source_langs: tuple[str, ...] | None = None) -> int:
    return min(
        max_wiki_sentences_for_lang(lang, source_langs=source_langs),
        LENGTH_PRIORITY_SENTENCE_CAP_BY_LANG.get(lang, LENGTH_PRIORITY_SENTENCE_CAP),
    )


def _rolling_avg(values: deque[int]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def _wiki_checkpoint_payload(
    lang: str,
    next_article_idx: int,
    accepted_articles: int,
    miss_streak: int,
    n_articles: int,
    committed_sentences: list[str],
    article_lengths_window: deque[int],
    sentence_lengths_window: deque[int],
    seed: int,
) -> dict:
    return {
        "lang": lang,
        "next_article_idx": next_article_idx,
        "accepted_articles": accepted_articles,
        "miss_streak": miss_streak,
        "final_target_articles": n_articles,
        "committed_sentence_count": len(committed_sentences),
        "rolling_avg_article_chars": _rolling_avg(article_lengths_window),
        "rolling_avg_sentence_chars": _rolling_avg(sentence_lengths_window),
        "rolling_article_lengths": list(article_lengths_window),
        "seed": seed,
    }


def _write_sentence_batch(writer, pa_module, batch: list[str]) -> None:
    if not batch:
        return
    table = pa_module.table({"sentence": pa_module.array(batch, type=pa_module.string())})
    writer.write_table(table)


def extract_sentences_from_wiki(
    lang: str,
    n_articles: int = 10_000,
    *,
    lang_to_group: dict[str, str],
    seed: int = 42,
    sentences_dir: str = PATHS["wiki"]["cache_dir"],
) -> str:
    lang = canonical_lang(lang)
    source_langs = _wiki_source_langs(lang)
    segmenter = _get_segmenter(lang)
    fetch_target = n_articles * 20
    final_path = parquet_path(sentences_dir, lang)
    temp_path = temp_parquet_path(lang)
    meta_path = temp_meta_path(lang)
    cache_meta_path = _wiki_cache_meta_path(sentences_dir, lang)
    datasets: list[tuple[str, Any]] = []
    for source_lang in source_langs:
        dataset = _load_wiki_dataset(source_lang, seed=seed)
        if dataset is not None:
            datasets.append((source_lang, dataset))
    if not datasets:
        return ""

    if lang in LENGTH_PRIORITY_LANGS:
        scan_limit = min(fetch_target, LENGTH_PRIORITY_SCAN_LIMIT)
        sentence_cap = max_length_priority_sentences_for_lang(lang, source_langs=source_langs)
        use_nltk_secondary = _wiki_use_nltk_secondary(lang, lang_to_group)
        print(
            f"  Length-priority mode enabled for {lang} "
            f"(scan_limit={scan_limit}, sentence_cap={sentence_cap}, "
            f"min_chars={_article_min_chars(lang, lang_to_group)})"
        )
        dataset = datasets[0][1]
        priority_articles = _collect_priority_articles(
            dataset,
            lang,
            scan_limit,
            lang_to_group=lang_to_group,
        )

        committed_sentences: list[str] = []
        sentence_lengths_window: deque[int] = deque(maxlen=WIKI_ROLLING_STATS_WINDOW)
        article_lengths_window: deque[int] = deque(maxlen=WIKI_ROLLING_STATS_WINDOW)
        accepted_articles = 0

        with tqdm(
            total=sentence_cap,
            desc=f"{lang} sentences",
            unit="sentence",
            leave=False,
            dynamic_ncols=True,
        ) as bar:
            for article_len, article_idx, article in priority_articles:
                if len(committed_sentences) >= sentence_cap or accepted_articles >= n_articles:
                    break

                article_text = article.get("text", "")
                article_title = article.get("title", "")
                article_lengths_window.append(article_len)

                article_batch = _extract_article_sentences(
                    article_text,
                    lang,
                    segmenter,
                    article_idx,
                    lang_to_group=lang_to_group,
                    article_title=article_title,
                )
                if not article_batch:
                    continue

                remaining_sentences = sentence_cap - len(committed_sentences)
                if remaining_sentences <= 0:
                    break
                if len(article_batch) > remaining_sentences:
                    article_batch = article_batch[:remaining_sentences]

                committed_sentences.extend(article_batch)
                sentence_lengths_window.extend(len(s) for s in article_batch)
                accepted_articles += 1
                bar.update(len(article_batch))
                bar.set_postfix_str(
                    f"acc {accepted_articles}/{n_articles} | "
                    f"sent {len(committed_sentences)} | "
                    f"avgA {_rolling_avg(article_lengths_window):.0f} | "
                    f"avgS {_rolling_avg(sentence_lengths_window):.0f}"
                )

        committed_sentences = post_clean_sentences(
            committed_sentences,
            lang,
            lang_to_group,
            use_nltk_secondary=use_nltk_secondary,
        )
        _write_sentence_parquet(final_path, committed_sentences)
        _write_json_atomic(
            cache_meta_path,
            {
                "lang": lang,
                "source_langs": list(source_langs),
            },
        )
        return final_path

    committed_sentences: list[str] = []
    sentence_cap = max_wiki_sentences_for_lang(lang, source_langs=source_langs)
    use_nltk_secondary = _wiki_use_nltk_secondary(lang, lang_to_group)
    sentence_lengths_window: deque[int] = deque(
        (len(s) for s in committed_sentences[-WIKI_ROLLING_STATS_WINDOW:]),
        maxlen=WIKI_ROLLING_STATS_WINDOW,
    )
    article_lengths_window: deque[int] = deque(maxlen=WIKI_ROLLING_STATS_WINDOW)
    next_article_idx = 0
    accepted_articles = 0
    miss_streak = 0
    completed_cleanly = False
    if os.path.exists(temp_path) and os.path.exists(meta_path):
        try:
            committed_sentences = pd.read_parquet(temp_path)["sentence"].tolist()
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if "accepted_articles" not in meta or "next_article_idx" not in meta:
                raise ValueError("stale checkpoint metadata")
            next_article_idx = int(meta["next_article_idx"])
            accepted_articles = int(meta["accepted_articles"])
            miss_streak = int(meta.get("miss_streak", 0))
            sentence_lengths_window.extend(
                len(s) for s in committed_sentences[-WIKI_ROLLING_STATS_WINDOW:]
            )
            article_lengths_window.extend(int(v) for v in meta.get("rolling_article_lengths", []))
            print(
                f"  Resuming {lang} from stream article {next_article_idx} "
                f"with {accepted_articles} accepted articles"
            )
        except Exception:
            committed_sentences = []
            next_article_idx = 0
            accepted_articles = 0
            miss_streak = 0
            for path in (temp_path, meta_path):
                if os.path.exists(path):
                    os.remove(path)
    elif os.path.exists(temp_path) or os.path.exists(meta_path):
        for path in (temp_path, meta_path):
            if os.path.exists(path):
                os.remove(path)

    def _update_progress(bar, scanned_articles: int) -> None:
        bar.set_postfix_str(
            f"acc {accepted_articles}/{n_articles} | "
            f"sent {len(committed_sentences)} | "
            f"yield {accepted_articles / max(1, scanned_articles):.1%} | "
            f"avgA {_rolling_avg(article_lengths_window):.0f} | "
            f"avgS {_rolling_avg(sentence_lengths_window):.0f} | "
            f"miss {miss_streak}"
        )

    try:
        with tqdm(
            total=sentence_cap,
            initial=len(committed_sentences),
            desc=f"{lang} sentences",
            unit="sentence",
            leave=False,
            dynamic_ncols=True,
        ) as bar:
            global_article_idx = 0
            stop_extracting = False
            for source_lang, dataset in datasets:
                if stop_extracting:
                    break
                for article_idx, article in enumerate(dataset.take(fetch_target)):
                    if global_article_idx < next_article_idx:
                        global_article_idx += 1
                        _update_progress(bar, global_article_idx)
                        continue
                    if global_article_idx >= MAX_WIKI_INDEX:
                        print(
                            f"  Stopping {lang} at article index {global_article_idx} "
                            f"(MAX_WIKI_INDEX={MAX_WIKI_INDEX})"
                        )
                        stop_extracting = True
                        break

                    article_text = article.get("text", "")
                    article_title = article.get("title", "")
                    article_lengths_window.append(len(article_text))

                    article_batch = _extract_article_sentences(
                        article_text,
                        lang,
                        segmenter,
                        global_article_idx,
                        lang_to_group=lang_to_group,
                        article_title=article_title,
                    )
                    if not article_batch:
                        next_article_idx = global_article_idx + 1
                        miss_streak += 1
                        _write_sentence_parquet(temp_path, committed_sentences)
                        _write_json_atomic(
                            meta_path,
                            _wiki_checkpoint_payload(
                                lang,
                                next_article_idx,
                                accepted_articles,
                                miss_streak,
                                n_articles,
                                committed_sentences,
                                article_lengths_window,
                                sentence_lengths_window,
                                seed,
                            ),
                        )
                        global_article_idx += 1
                        _update_progress(bar, global_article_idx)
                        continue

                    remaining_sentences = sentence_cap - len(committed_sentences)
                    if remaining_sentences <= 0:
                        print(
                            f"  Stopping {lang} after reaching sentence cap="
                            f"{sentence_cap}"
                        )
                        stop_extracting = True
                        break

                    if len(article_batch) > remaining_sentences:
                        article_batch = article_batch[:remaining_sentences]

                    committed_sentences.extend(article_batch)
                    sentence_lengths_window.extend(len(s) for s in article_batch)
                    accepted_articles += 1
                    miss_streak = 0
                    _write_sentence_parquet(temp_path, committed_sentences)
                    _write_json_atomic(
                        meta_path,
                        _wiki_checkpoint_payload(
                            lang,
                            global_article_idx + 1,
                            accepted_articles,
                            miss_streak,
                            n_articles,
                            committed_sentences,
                            article_lengths_window,
                            sentence_lengths_window,
                            seed,
                        ),
                    )
                    next_article_idx = global_article_idx + 1
                    bar.update(len(article_batch))
                    _update_progress(bar, global_article_idx + 1)
                    if len(committed_sentences) >= sentence_cap:
                        print(
                            f"  Stopping {lang} after reaching sentence cap="
                            f"{sentence_cap}"
                        )
                        stop_extracting = True
                        break
                    if accepted_articles >= n_articles:
                        stop_extracting = True
                        break
                    global_article_idx += 1
                if stop_extracting:
                    break

        committed_sentences = post_clean_sentences(
            committed_sentences,
            lang,
            lang_to_group,
            use_nltk_secondary=use_nltk_secondary,
        )
        _write_sentence_parquet(final_path, committed_sentences)
        _write_json_atomic(
            cache_meta_path,
            {
                "lang": lang,
                "source_langs": list(source_langs),
            },
        )
        completed_cleanly = True
    finally:
        if completed_cleanly:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)

    return final_path


def load_or_extract(
    lang: str,
    *,
    lang_to_group: dict[str, str],
    seed: int = 42,
    sentences_dir: str = PATHS["sentences_dir"],
    articles_per_lang: int = ARTICLES_PER_LANG,
) -> tuple[str, str | None]:
    lang = canonical_lang(lang)
    path = parquet_path(sentences_dir, lang)
    cache_meta_path = _wiki_cache_meta_path(sentences_dir, lang)
    source_langs = list(_wiki_source_langs(lang))
    if os.path.exists(path):
        meta_missing = not os.path.exists(cache_meta_path)
        meta_valid = True
        if meta_missing:
            _write_wiki_cache_meta(cache_meta_path, lang, source_langs)
        else:
            try:
                with open(cache_meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None
            meta_valid = bool(
                isinstance(meta, dict)
                and meta.get("lang") == lang
                and meta.get("source_langs") == source_langs
            )
        if meta_valid:
            cached = pd.read_parquet(path)["sentence"].tolist()
            cleaned = post_clean_sentences(
                cached,
                lang,
                lang_to_group,
                use_nltk_secondary=_wiki_use_nltk_secondary(lang, lang_to_group),
            )
            if cleaned != cached:
                _write_sentence_parquet(path, cleaned)
            if meta_missing and not os.path.exists(cache_meta_path):
                _write_wiki_cache_meta(cache_meta_path, lang, source_langs)
            return lang, path
        stale_temp_path = temp_parquet_path(lang)
        stale_meta_path = temp_meta_path(lang)
        for stale_path in (path, cache_meta_path, stale_temp_path, stale_meta_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)
    extracted_path = extract_sentences_from_wiki(
        lang,
        n_articles=articles_per_lang,
        lang_to_group=lang_to_group,
        seed=seed,
        sentences_dir=sentences_dir,
    )
    if not extracted_path:
        return lang, None
    return lang, extracted_path


def load_wiki_sentences(
    langs: list[str] = ALL_LANGS,
    *,
    lang_to_group: dict[str, str] = LANG_TO_GROUP,
    seed: int = 42,
    articles_per_lang: int = ARTICLES_PER_LANG,
    max_workers: int,
    sentences_dir: str = PATHS["wiki"]["cache_dir"],
) -> dict[str, list[str]]:
    os.makedirs(sentences_dir, exist_ok=True)
    result: dict[str, list[str]] = {}
    print(f"Extracting sentences -> cached under '{sentences_dir}/'")
    print(f"(Workers: {max_workers} processes | cached languages skip extraction)\n")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                load_or_extract,
                lang,
                lang_to_group=lang_to_group,
                seed=seed,
                sentences_dir=sentences_dir,
                articles_per_lang=articles_per_lang,
            ): lang
            for lang in langs
        }
        for future in tqdm(as_completed(futures), total=len(langs), desc="Languages"):
            lang, cache_path = future.result()
            if not cache_path:
                tqdm.write(f"  {lang}: skipped (Wikipedia config missing)")
                continue
            sentences = pd.read_parquet(cache_path)["sentence"].tolist()
            result[lang] = sentences
            tqdm.write(f"  {lang}: {len(sentences)} sentences -> {cache_path}")
    return result


def refilter_cached_wiki_sentences(
    langs: list[str] = ALL_LANGS,
    *,
    lang_to_group: dict[str, str] = LANG_TO_GROUP,
    sentences_dir: str = PATHS["wiki"]["cache_dir"],
    latin_only: bool = True,
) -> dict[str, int]:
    updated_counts: dict[str, int] = {}
    for raw_lang in langs:
        lang = canonical_lang(raw_lang)
        if latin_only and lang_to_group.get(lang) not in LATIN_GROUPS:
            continue
        path = parquet_path(sentences_dir, lang)
        if not os.path.exists(path):
            continue
        cached = pd.read_parquet(path)["sentence"].tolist()
        cleaned = post_clean_sentences(
            cached,
            lang,
            lang_to_group,
            use_nltk_secondary=_wiki_use_nltk_secondary(lang, lang_to_group),
        )
        if cleaned != cached:
            _write_sentence_parquet(path, cleaned)
        updated_counts[lang] = len(cleaned)
    return updated_counts


def collect_rejected_english_sentences(
    lang: str,
    *,
    lang_to_group: dict[str, str] = LANG_TO_GROUP,
    sentences_dir: str = PATHS["wiki"]["cache_dir"],
) -> list[dict[str, Any]]:
    lang = canonical_lang(lang)
    path = parquet_path(sentences_dir, lang)
    if not os.path.exists(path):
        return []
    cached = pd.read_parquet(path)["sentence"].tolist()
    use_nltk_secondary = _wiki_use_nltk_secondary(lang, lang_to_group)
    rejected: list[dict[str, Any]] = []
    for idx, sentence in enumerate(cached):
        if _looks_like_english_text(
            sentence,
            lang,
            lang_to_group,
            use_nltk_secondary=use_nltk_secondary,
        ):
            rejected.append(
                {
                    "lang": lang,
                    "source_index": idx,
                    "sentence": sentence,
                    "use_nltk_secondary": use_nltk_secondary,
                }
            )
    return rejected
