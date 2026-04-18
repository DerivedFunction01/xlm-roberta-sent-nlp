from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from io_utils import write_json_atomic, write_records_parquet
from language import canonical_lang
from paths import PATHS

MAJOR_LATIN_WIKI_LANGS = ("de", "es", "fr", "it", "pt")
WIKI_LEXICON_TOP_N_DEFAULT = 25_000
WIKI_LEXICON_CACHE_VERSION = 2
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
DEFAULT_LEXICON_SOURCES = ("tatoeba", "wiki")


def _wiki_lexicon_cache_dir(cache_dir: str = PATHS["wiki_lexicon"]["cache_dir"]) -> str:
    return cache_dir


def _wiki_lexicon_cache_path(cache_dir: str, lang: str) -> str:
    return os.path.join(cache_dir, f"{canonical_lang(lang)}.parquet")


def _wiki_lexicon_meta_path(cache_dir: str = PATHS["wiki_lexicon"]["cache_dir"]) -> str:
    return PATHS["wiki_lexicon"]["cache_meta"] if cache_dir == PATHS["wiki_lexicon"]["cache_dir"] else os.path.join(
        cache_dir,
        "wiki_latin_lexicon.meta.json",
    )


def _tokenize_words(sentence: str) -> list[str]:
    return [
        token.lower()
        for token in WORD_RE.findall(sentence)
        if token.isalpha()
    ]


def _load_sentence_source(cache_dir: str, lang: str) -> list[str]:
    path = os.path.join(cache_dir, f"{canonical_lang(lang)}.parquet")
    if not os.path.exists(path):
        return []
    frame = pd.read_parquet(path, columns=["sentence"])
    if "sentence" not in frame.columns:
        return []
    return [sentence for sentence in frame["sentence"].astype(str).tolist() if sentence.strip()]


def _lexicon_source_dirs(
    sources: tuple[str, ...],
    *,
    wiki_cache_dir: str,
    tatoeba_cache_dir: str,
) -> dict[str, str]:
    source_dirs = {
        "wiki": wiki_cache_dir,
        "tatoeba": tatoeba_cache_dir,
    }
    unknown_sources = [source for source in sources if source not in source_dirs]
    if unknown_sources:
        raise ValueError(f"Unsupported lexicon source(s): {unknown_sources}")
    return {source: source_dirs[source] for source in sources}


def build_wiki_major_latin_lexicon_cache(
    *,
    wiki_cache_dir: str = PATHS["wiki"]["cache_dir"],
    tatoeba_cache_dir: str = PATHS["tatoeba"]["cache_dir"],
    output_dir: str = PATHS["wiki_lexicon"]["cache_dir"],
    langs: tuple[str, ...] = MAJOR_LATIN_WIKI_LANGS,
    top_n: int = WIKI_LEXICON_TOP_N_DEFAULT,
    sources: tuple[str, ...] = DEFAULT_LEXICON_SOURCES,
) -> dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    source_dirs = _lexicon_source_dirs(
        sources,
        wiki_cache_dir=wiki_cache_dir,
        tatoeba_cache_dir=tatoeba_cache_dir,
    )
    lang_counts: dict[str, int] = {}
    total_rows = 0

    for raw_lang in tqdm(langs, desc="Wiki lexicons"):
        lang = canonical_lang(raw_lang)
        sentences: list[str] = []
        for source_dir in source_dirs.values():
            sentences.extend(_load_sentence_source(source_dir, lang))
        if not sentences:
            continue
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(_tokenize_words(sentence))
        if not counts:
            continue

        top_words = counts.most_common(top_n)
        rows: list[dict[str, Any]] = [
            {
                "lang": lang,
                "rank": rank,
                "word": word,
                "count": int(count),
            }
            for rank, (word, count) in enumerate(top_words, start=1)
        ]
        write_records_parquet(_wiki_lexicon_cache_path(output_dir, lang), rows, columns=["lang", "rank", "word", "count"])
        lang_counts[lang] = len(rows)
        total_rows += len(rows)

    write_json_atomic(
        _wiki_lexicon_meta_path(output_dir),
        {
            "cache_version": WIKI_LEXICON_CACHE_VERSION,
            "langs": sorted(lang_counts),
            "top_n": top_n,
            "sources": list(sources),
            "total_rows": total_rows,
            "source_dirs": source_dirs,
        },
    )
    load_wiki_major_latin_lexicon.cache_clear()
    return lang_counts


@lru_cache(maxsize=16_384)
def load_wiki_major_latin_lexicon(
    lang: str,
    *,
    cache_dir: str = PATHS["wiki_lexicon"]["cache_dir"],
) -> frozenset[str]:
    path = _wiki_lexicon_cache_path(cache_dir, lang)
    if not os.path.exists(path):
        return frozenset()
    frame = pd.read_parquet(path, columns=["word"])
    if "word" not in frame.columns:
        return frozenset()
    return frozenset(
        word.lower().strip()
        for word in frame["word"].astype(str).tolist()
        if isinstance(word, str) and word.strip()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cached top-word lexicon from existing wiki and/or Tatoeba sentence parquets."
    )
    parser.add_argument(
        "--wiki-cache-dir",
        default=PATHS["wiki"]["cache_dir"],
        help="Directory containing cached wiki sentence parquets.",
    )
    parser.add_argument(
        "--tatoeba-cache-dir",
        default=PATHS["tatoeba"]["cache_dir"],
        help="Directory containing cached Tatoeba sentence parquets.",
    )
    parser.add_argument(
        "--output-dir",
        default=PATHS["wiki_lexicon"]["cache_dir"],
        help="Directory where lexicon parquet files will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=WIKI_LEXICON_TOP_N_DEFAULT,
        help="Number of most frequent words to keep per language.",
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        default=list(MAJOR_LATIN_WIKI_LANGS),
        help="Language codes to process.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(DEFAULT_LEXICON_SOURCES),
        choices=("wiki", "tatoeba"),
        help="Sentence sources to merge into the lexicon.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    counts = build_wiki_major_latin_lexicon_cache(
        wiki_cache_dir=args.wiki_cache_dir,
        tatoeba_cache_dir=args.tatoeba_cache_dir,
        output_dir=args.output_dir,
        langs=tuple(args.langs),
        top_n=args.top_n,
        sources=tuple(args.sources),
    )
    total = sum(counts.values())
    print(f"Wrote {len(counts)} lexicon parquet files with {total:,} rows total -> {args.output_dir}")


if __name__ == "__main__":
    main()
