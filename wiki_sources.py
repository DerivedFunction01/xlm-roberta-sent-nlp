from __future__ import annotations

import json
import os
import random
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Callable

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


LATIN_GROUPS = {"English", "LatinCore", "LatinTier2"}
WIKI_MARKUP = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==.*?==", flags=re.DOTALL)
WIKI_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
BRACKET_NOTES = re.compile(r"\s*[\(\[【（][^\)\]】）]{0,60}[\)\]】）]\s*")
WIKI_ASCII_WORDS = re.compile(r"[A-Za-z]+")
WIKI_SPACES = re.compile(r"\s{2,}")
WIKI_PUNCT_REPEAT = re.compile(r"([,.;:!?…،。！？])\1+")
WIKI_TRAILING_ORPHAN_LETTER = re.compile(r"[\s,.;:!?…،。！？]+([^\W\d_])$")
WIKI_LEADING_ORPHAN_LETTER = re.compile(r"^[\"'“”‘’«»‹›\s,.;:!?…،。！？]+([^\W\d_])\s+")
WIKI_BLOCKED_MARKERS = ("http",)
WIKI_BLOCKED_CHARS = {"=", "<", ">", "|"}
WIKI_OPENING_QUOTES = {"\"", "'", "“", "”", "‘", "’", "«", "»", "‹", "›"}

PYSBD_SUPPORTED = {
    "en", "hi", "mr", "bg", "es", "ru", "ar", "am", "hy", "fa",
    "ur", "pl", "zh", "nl", "da", "fr", "it", "el", "my", "ja", "de", "kk",
}
PYSBD_FALLBACKS = {
    "uk": "ru", "be": "ru", "sr": "ru", "mk": "ru", "mn": "ru",
    "pt": "es", "ro": "fr", "la": "it", "sq": "it",
    "sv": "da", "no": "da", "is": "da",
    "fi": "en", "hu": "en", "cs": "pl",
    "vi": "en", "id": "en", "ms": "en", "af": "nl", "tr": "en",
    "he": "ar", "ps": "fa", "ug": "ar",
    "bn": "hi", "ta": "hi", "te": "hi", "gu": "hi", "kn": "hi",
    "ml": "hi", "pa": "hi", "as": "hi", "or": "hi", "sd": "hi",
    "ka": "en", "km": "zh", "ko": "zh", "lo": "zh", "th": "zh",
}

_SENT_BOUNDS: dict[str, tuple[int, int]] = {
    "zh": (8, 180), "ja": (10, 180),
    "ko": (15, 220), "th": (15, 250), "km": (15, 250), "lo": (15, 250), "my": (15, 250),
    "ar": (25, 450), "fa": (25, 450), "he": (25, 400), "ur": (25, 450),
    "hi": (30, 500), "bn": (30, 500), "ta": (30, 500), "te": (30, 500), "am": (25, 400),
    "fi": (20, 450), "hu": (20, 450), "tr": (20, 450), "vi": (15, 300),
    "de": (40, 600), "ru": (35, 650), "uk": (35, 650), "el": (35, 650),
    "hy": (30, 500), "ka": (25, 450), "en": (24, 600),
}
_DEFAULT_BOUNDS = (30, 600)
_PROC_SEGMENTERS: dict[str, object] = {}


def parquet_path(sentences_dir: str, lang: str) -> str:
    return os.path.join(sentences_dir, f"{lang}.parquet")


def _non_punct_char_count(s: str) -> int:
    return len(re.sub(r"[\W_]+", "", s, flags=re.UNICODE))


def _digit_count(s: str) -> int:
    return len(re.findall(r"\d", s))


def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))


def _strip_bracket_notes(text: str) -> str:
    return BRACKET_NOTES.sub(" ", text)


def _collapse_spaces(text: str) -> str:
    return WIKI_SPACES.sub(" ", text)


def _strip_leading_punct(sentence: str) -> str:
    sentence = sentence.lstrip()
    if not sentence or sentence[0] in WIKI_OPENING_QUOTES:
        return sentence
    idx = 0
    while idx < len(sentence):
        ch = sentence[idx]
        if ch.isspace():
            idx += 1
            continue
        if re.match(r"\W", ch, flags=re.UNICODE):
            idx += 1
            continue
        break
    return sentence[idx:].lstrip()


def _collapse_repeated_punct(sentence: str) -> str:
    return WIKI_PUNCT_REPEAT.sub(r"\1", sentence)


def _strip_trailing_orphan_letter(sentence: str, lang_to_group: dict[str, str], lang: str) -> str:
    if lang_to_group.get(lang) in LATIN_GROUPS:
        return sentence
    return WIKI_TRAILING_ORPHAN_LETTER.sub("", sentence).rstrip()


def _strip_leading_orphan_letter(sentence: str) -> str:
    return WIKI_LEADING_ORPHAN_LETTER.sub("", sentence).lstrip()


def _has_blocked_artifact(sentence: str) -> bool:
    lower = sentence.lower()
    return any(marker in lower for marker in WIKI_BLOCKED_MARKERS) or any(ch in sentence for ch in WIKI_BLOCKED_CHARS)


def _get_min_article_chars(lang: str, lang_to_group: dict[str, str]) -> int:
    group = lang_to_group.get(lang, "OtherScripts")
    return {
        "English": 2_000,
        "LatinCore": 2_000,
        "LatinTier2": 2_000,
        "Cyrillic": 2_000,
        "EastAsian": 1_200,
        "Indic": 2_000,
        "ArabicScript": 2_000,
        "OtherScripts": 2_000,
    }.get(group, 3_000)


def _is_valid_sentence(s: str, lang: str) -> bool:
    mn, mx = _SENT_BOUNDS.get(lang, _DEFAULT_BOUNDS)
    return mn < len(s) < mx


def clean_wiki_sentence(sentence: str, lang: str, lang_to_group: dict[str, str]) -> str:
    sentence = _strip_bracket_notes(sentence)
    sentence = WIKI_MARKUP.sub("", sentence)
    sentence = WIKI_PUNCT_REPEAT.sub(r"\1", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = _strip_leading_punct(sentence)
    sentence = _strip_leading_orphan_letter(sentence)
    if lang_to_group.get(lang) not in LATIN_GROUPS:
        sentence = WIKI_ASCII_WORDS.sub("", sentence)
    sentence = _strip_trailing_orphan_letter(sentence, lang_to_group, lang)
    sentence = _collapse_spaces(sentence)
    return sentence.strip()


def _get_segmenter(lang: str, lang_to_group: dict[str, str]):
    if lang in _PROC_SEGMENTERS:
        return _PROC_SEGMENTERS[lang]
    try:
        import pysbd as _pysbd
        proxy = PYSBD_FALLBACKS.get(lang, lang if lang in PYSBD_SUPPORTED else None)
        seg = _pysbd.Segmenter(language=proxy, clean=True) if proxy else None
    except (ImportError, ValueError):
        seg = None
    _PROC_SEGMENTERS[lang] = seg
    return seg


def extract_sentences_from_wiki(
    lang: str,
    *,
    lang_to_group: dict[str, str],
    articles_per_lang: int,
) -> list[str]:
    segmenter = _get_segmenter(lang, lang_to_group)
    fetch_target = articles_per_lang * 20
    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split="train",
        streaming=True,
    )
    sentences: list[str] = []
    seen = 0
    for article in dataset.take(fetch_target):
        text = article.get("text", "")
        if not isinstance(text, str) or len(text) < _get_min_article_chars(lang, lang_to_group):
            continue
        text = WIKI_MARKUP.sub("", text)
        text = text[: len(text) // 2].strip()
        if not text:
            continue
        sents = segmenter.segment(text) if segmenter else re.split(r"(?<=[.!?])\s+", text)
        for s in sents:
            s = clean_wiki_sentence(s, lang, lang_to_group)
            if _is_valid_sentence(s, lang):
                sentences.append(s)
        seen += 1
        if seen >= articles_per_lang:
            break
    return sentences


def load_or_extract(
    lang: str,
    *,
    lang_to_group: dict[str, str],
    sentences_dir: str,
    articles_per_lang: int,
) -> tuple[str, list[str]]:
    path = parquet_path(sentences_dir, lang)
    if os.path.exists(path):
        cached = pd.read_parquet(path)["sentence"].tolist()
        cleaned = [clean_wiki_sentence(s, lang, lang_to_group) for s in cached]
        cleaned = [s for s in cleaned if _is_valid_sentence(s, lang) and not _has_blocked_artifact(s)]
        if cleaned != cached:
            pd.DataFrame({"sentence": cleaned}).to_parquet(path, index=False)
        return lang, cleaned
    sentences = extract_sentences_from_wiki(
        lang,
        lang_to_group=lang_to_group,
        articles_per_lang=articles_per_lang,
    )
    pd.DataFrame({"sentence": sentences}).to_parquet(path, index=False)
    return lang, sentences


def load_wiki_sentences(
    langs: list[str],
    *,
    lang_to_group: dict[str, str],
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
                sentences_dir=sentences_dir,
                articles_per_lang=articles_per_lang,
            ): lang
            for lang in langs
        }
        for future in tqdm(as_completed(futures), total=len(langs), desc="Languages"):
            lang, sentences = future.result()
            result[lang] = sentences
            tqdm.write(f"  {lang}: {len(sentences)} sentences -> {parquet_path(sentences_dir, lang)}")
    return result


def build_sentence_pools(
    sentence_map: dict[str, list[str]],
    *,
    reserve_fraction: float,
    min_reserved: int,
    max_reserved: int,
) -> tuple[dict[str, deque[str]], dict[str, deque[str]]]:
    reserved: dict[str, deque[str]] = {}
    main: dict[str, deque[str]] = {}
    for lang, sentences in sentence_map.items():
        if not sentences:
            continue
        shuffled = sentences[:]
        random.shuffle(shuffled)
        reserve_target = int(round(len(shuffled) * reserve_fraction))
        reserve_n = min(len(shuffled), max(min_reserved, min(reserve_target, max_reserved)))
        reserved[lang] = deque(shuffled[:reserve_n])
        main[lang] = deque(shuffled[reserve_n:])
    return reserved, main


def draw_sentence(
    lang: str,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
) -> str | None:
    if primary_pool.get(lang):
        return primary_pool[lang].popleft()
    if fallback_pool and fallback_pool.get(lang):
        return fallback_pool[lang].popleft()
    return None


def remaining_sentence_count(
    lang: str,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
) -> int:
    total = len(primary_pool.get(lang, ()))
    if fallback_pool is not None:
        total += len(fallback_pool.get(lang, ()))
    return total


def chunk_list(items: list, n_chunks: int) -> list[list]:
    if n_chunks <= 1:
        return [items]
    chunk_size = (len(items) + n_chunks - 1) // n_chunks
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def partition_sentence_pools(
    pools: dict[str, deque[str]],
    n_workers: int,
) -> list[dict[str, deque[str]]]:
    worker_pools: list[dict[str, deque[str]]] = [dict() for _ in range(n_workers)]
    for lang, dq in pools.items():
        items = list(dq)
        for worker_idx in range(n_workers):
            shard = items[worker_idx::n_workers]
            if shard:
                worker_pools[worker_idx][lang] = deque(shard)
    return worker_pools
