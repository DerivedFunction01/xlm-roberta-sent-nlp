from __future__ import annotations

from collections import deque
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic as _write_json_atomic, write_sentence_parquet as _write_sentence_parquet
from paths import SENTENCES_DIR, WIKI_CLEANUP_META, WIKI_SEGMENTATION_DEBUG_DIR, WIKI_TEMP_DIR
from text_utils import (
    _article_min_chars,
    _get_segmenter,
    _is_valid_sentence,
    _split_paragraphs,
    clean_sentence,
    log_segmentation_failure,
    post_clean_sentences,
    sanitize_paragraph_for_pysbd,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


MAX_WIKI_INDEX = 100_000
ARTICLES_PER_LANG = 10_000
MAX_WIKI_SENTENCES = 200_000
MAX_WIKI_SENTENCES_BY_LANG = {
    "en": 300_000,
}
WIKI_ROLLING_STATS_WINDOW = 250
WIKI_MARKUP = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==.*?==", flags=re.DOTALL)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
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

def parquet_path(sentences_dir: str, lang: str) -> str:
    return os.path.join(sentences_dir, f"{lang}.parquet")


def _load_cleanup_meta() -> dict[str, dict]:
    if not os.path.exists(WIKI_CLEANUP_META):
        return {}
    try:
        with open(WIKI_CLEANUP_META, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_cleanup_meta(meta: dict[str, dict]) -> None:
    _write_json_atomic(WIKI_CLEANUP_META, meta)


def _cleanup_fingerprint(path: str) -> dict[str, int]:
    stat = os.stat(path)
    return {
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def finalize_wiki_sentence_cache(
    sentence_map: dict[str, list[str]],
    *,
    lang_to_group: dict[str, str],
) -> dict[str, list[str]]:
    cleaned_map: dict[str, list[str]] = {}
    total_before = 0
    total_after = 0
    changed_langs = 0
    cleanup_meta = _load_cleanup_meta()

    langs = sorted(sentence_map.keys())

    with tqdm(total=len(langs), desc="Languages", unit="lang") as pbar_langs:
        for lang in langs:
            sentences = sentence_map[lang]
            path = parquet_path(SENTENCES_DIR, lang)
            fingerprint = _cleanup_fingerprint(path) if os.path.exists(path) else {}
            meta_entry = cleanup_meta.get(lang, {})
            already_cleaned = (
                bool(meta_entry.get("cleaned"))
                and meta_entry.get("path") == path
                and meta_entry.get("input_fingerprint") == fingerprint
            )

            with tqdm(
                total=len(sentences),
                desc=f"{lang} cleanup",
                unit="sent",
                leave=False,
            ) as pbar_sent:
                if already_cleaned:
                    cleaned = sentences
                else:
                    cleaned = post_clean_sentences(sentences, lang, lang_to_group)
                pbar_sent.update(len(sentences))

            before = len(sentences)
            after = len(cleaned)
            total_before += before
            total_after += after

            if cleaned != sentences:
                changed_langs += 1
                _write_sentence_parquet(path, cleaned)

            cleaned_map[lang] = cleaned
            cleanup_meta[lang] = {
                "path": path,
                "cleaned": True,
                "input_fingerprint": fingerprint,
                "input_sentence_count": before,
                "cleaned_sentence_count": after,
                "updated": cleaned != sentences,
            }
            pbar_langs.update(1)

    _write_cleanup_meta(cleanup_meta)
    print(
        f"\nWiki post-clean: {changed_langs} languages updated | "
        f"{total_before:,} -> {total_after:,} sentences"
    )
    return cleaned_map


def prepare_wiki_paragraphs(text: str, lang: str, lang_to_group: dict[str, str]) -> list[str] | None:
    paragraphs = _split_paragraphs(text, lang, lang_to_group)
    if paragraphs is None:
        return None

    fraction = 0.75 if lang == "en" else 0.50
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

    article_batch: list[str] = []
    for paragraph_idx, paragraph in enumerate(paragraphs):
        safe_paragraph = sanitize_paragraph_for_pysbd(paragraph)
        try:
            sents = segmenter.segment(safe_paragraph) if segmenter else SENT_SPLIT.split(safe_paragraph)  # type: ignore[attr-defined]
        except re.error as exc:
            log_segmentation_failure(
                os.path.join(WIKI_SEGMENTATION_DEBUG_DIR, f"{lang}.log"),
                lang=lang,
                article_idx=article_idx,
                paragraph_idx=paragraph_idx,
                paragraph=paragraph,
                exc=exc,
                article_title=article_title,
            )
            sents = SENT_SPLIT.split(safe_paragraph)
        for s in sents:
            s = clean_sentence(s, lang, lang_to_group)
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


def temp_parquet_path(sentences_dir: str, lang: str) -> str:
    return os.path.join(sentences_dir, "_wiki_tmp", f"{lang}.parquet")


def temp_meta_path(sentences_dir: str, lang: str) -> str:
    return os.path.join(sentences_dir, "_wiki_tmp", f"{lang}.meta.json")


def max_wiki_sentences_for_lang(lang: str) -> int:
    return MAX_WIKI_SENTENCES_BY_LANG.get(lang, MAX_WIKI_SENTENCES)


def max_length_priority_sentences_for_lang(lang: str) -> int:
    return min(
        max_wiki_sentences_for_lang(lang),
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
    sentences_dir: str = SENTENCES_DIR,
) -> str:
    segmenter = _get_segmenter(lang)
    fetch_target = n_articles * 20
    final_path = parquet_path(sentences_dir, lang)
    temp_path = temp_parquet_path(sentences_dir, lang)
    meta_path = temp_meta_path(sentences_dir, lang)

    if lang in LENGTH_PRIORITY_LANGS:
        scan_limit = min(fetch_target, LENGTH_PRIORITY_SCAN_LIMIT)
        sentence_cap = max_length_priority_sentences_for_lang(lang)
        print(
            f"  Length-priority mode enabled for {lang} "
            f"(scan_limit={scan_limit}, sentence_cap={sentence_cap}, "
            f"min_chars={_article_min_chars(lang, lang_to_group)})"
        )
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            split="train",
            streaming=True,
        )
        dataset = dataset.shuffle(buffer_size=1000, seed=seed)
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

        committed_sentences = post_clean_sentences(committed_sentences, lang, lang_to_group)
        _write_sentence_parquet(final_path, committed_sentences)
        return final_path

    committed_sentences: list[str] = []
    sentence_cap = max_wiki_sentences_for_lang(lang)
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

    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(buffer_size=1000, seed=seed)

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
            for article_idx, article in enumerate(dataset.take(fetch_target)):
                if article_idx < next_article_idx:
                    _update_progress(bar, article_idx + 1)
                    continue
                if article_idx >= MAX_WIKI_INDEX:
                    print(
                        f"  Stopping {lang} at article index {article_idx} "
                        f"(MAX_WIKI_INDEX={MAX_WIKI_INDEX})"
                    )
                    break

                article_text = article.get("text", "")
                article_title = article.get("title", "")
                article_lengths_window.append(len(article_text))

                article_batch = _extract_article_sentences(
                    article_text,
                    lang,
                    segmenter,
                    article_idx,
                    lang_to_group=lang_to_group,
                    article_title=article_title,
                )
                if not article_batch:
                    next_article_idx = article_idx + 1
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
                    _update_progress(bar, article_idx + 1)
                    continue

                remaining_sentences = sentence_cap - len(committed_sentences)
                if remaining_sentences <= 0:
                    print(
                        f"  Stopping {lang} after reaching sentence cap="
                        f"{sentence_cap}"
                    )
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
                        article_idx + 1,
                        accepted_articles,
                        miss_streak,
                        n_articles,
                        committed_sentences,
                        article_lengths_window,
                        sentence_lengths_window,
                        seed,
                    ),
                )
                next_article_idx = article_idx + 1
                bar.update(len(article_batch))
                _update_progress(bar, article_idx + 1)
                if len(committed_sentences) >= sentence_cap:
                    print(
                        f"  Stopping {lang} after reaching sentence cap="
                        f"{sentence_cap}"
                    )
                    break
                if accepted_articles >= n_articles:
                    break

        committed_sentences = post_clean_sentences(committed_sentences, lang, lang_to_group)
        _write_sentence_parquet(final_path, committed_sentences)
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
    seed: int,
    sentences_dir: str,
    articles_per_lang: int,
) -> tuple[str, str]:
    path = parquet_path(sentences_dir, lang)
    if os.path.exists(path):
        cached = pd.read_parquet(path)["sentence"].tolist()
        cleaned = post_clean_sentences(cached, lang, lang_to_group)
        if cleaned != cached:
            _write_sentence_parquet(path, cleaned)
        return lang, path
    extracted_path = extract_sentences_from_wiki(
        lang,
        n_articles=articles_per_lang,
        lang_to_group=lang_to_group,
        seed=seed,
        sentences_dir=sentences_dir,
    )
    return lang, extracted_path


def load_wiki_sentences(
    langs: list[str],
    *,
    lang_to_group: dict[str, str],
    seed: int,
    sentences_dir: str,
    articles_per_lang: int,
    max_workers: int,
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
            sentences = pd.read_parquet(cache_path)["sentence"].tolist()
            result[lang] = sentences
            tqdm.write(f"  {lang}: {len(sentences)} sentences -> {cache_path}")
    return result
