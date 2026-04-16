from __future__ import annotations

import argparse
import os
import re
from typing import Callable

from tqdm.auto import tqdm

from language import LANG_TO_GROUP, LATIN_GROUPS, canonical_lang
from paths import PATHS
from refilter_shared import refilter_cached_sentence_parquets
from text_utils import SENT_SPLIT, _get_segmenter, _split_long_list_like_segment, post_clean_sentences
from wiki_sources import _wiki_use_nltk_secondary


QUOTE_WRAPPER_PAIRS = (
    ('"', '"'),
    ("“", "”"),
    ("‘", "’"),
    ("«", "»"),
    ("‹", "›"),
)
LEADING_QUOTE_NOISE_RE = re.compile(r'^\s*(?:["“”‘’«»‹›]+\s*)+')
TRAILING_QUOTE_NOISE_RE = re.compile(r'(?:\s*["“”‘’«»‹›]+)+\s*$')


def _strip_outer_quote_wrappers(text: str) -> str:
    stripped = text.strip()
    if len(stripped) < 2:
        return stripped

    changed = True
    while changed:
        changed = False
        stripped = stripped.strip()
        if len(stripped) < 2:
            break
        for open_ch, close_ch in QUOTE_WRAPPER_PAIRS:
            if stripped.startswith(open_ch) and stripped.endswith(close_ch):
                inner = stripped[len(open_ch) : len(stripped) - len(close_ch)].strip()
                if inner:
                    stripped = inner
                    changed = True
                    break
    return stripped


def _strip_edge_quote_noise(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    stripped = LEADING_QUOTE_NOISE_RE.sub("", stripped)
    stripped = TRAILING_QUOTE_NOISE_RE.sub("", stripped)
    return stripped.strip()


def _merge_sentence_fragments(fragments: list[str]) -> list[str]:
    if not fragments:
        return []

    merged: list[str] = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if not merged:
            merged.append(fragment)
            continue

        prev = merged[-1]
        prev_words = len(prev.split())
        next_starts_lower = fragment[:1].islower()
        prev_looks_short = prev_words < 4 or prev.endswith((" .", " :", " ;"))
        if prev_looks_short and next_starts_lower:
            merged[-1] = f"{prev} {fragment}".strip()
            continue
        merged.append(fragment)

    return merged


def _segment_wrapped_text(text: str, *, lang: str, segmenter: object | None = None) -> list[str]:
    safe_text = text.strip()
    if not safe_text:
        return []

    if segmenter is None:
        segmenter = _get_segmenter(lang)

    try:
        segments = segmenter.segment(safe_text) if segmenter else SENT_SPLIT.split(safe_text)  # type: ignore[attr-defined]
    except Exception:
        segments = SENT_SPLIT.split(safe_text)

    return [segment.strip() for segment in segments if isinstance(segment, str) and segment.strip()]


def expand_wrapped_sentence_fragments(sentence: str, *, lang: str = "", segmenter: object | None = None) -> list[str]:
    if not isinstance(sentence, str):
        return []

    stripped = sentence.strip()
    if not stripped:
        return []

    blocks = [block.strip() for block in re.split(r"\n\s*\n+", stripped) if block.strip()]
    if not blocks:
        blocks = [stripped]

    expanded: list[str] = []
    for block in blocks:
        inner = _strip_edge_quote_noise(_strip_outer_quote_wrappers(block))
        sentence_chunks = _segment_wrapped_text(inner, lang=lang, segmenter=segmenter)
        if not sentence_chunks:
            sentence_chunks = [inner]
        sentence_chunks = _merge_sentence_fragments(sentence_chunks)
        for chunk in sentence_chunks:
            pieces = _split_long_list_like_segment(chunk)
            if len(pieces) > 1:
                expanded.extend(piece for piece in pieces if piece.strip())
                continue
            expanded.append(chunk)
    return [piece for piece in expanded if piece.strip()]


def _build_transform(source: str) -> Callable[[str, list[str], dict[str, str]], list[str]]:
    source = source.lower().strip()
    if source not in {"wiki", "finetrans"}:
        raise ValueError(f"Unsupported source: {source}")

    def _transform(lang: str, cached: list[str], lang_to_group: dict[str, str]) -> list[str]:
        expanded: list[str] = []
        for sentence in cached:
            expanded.extend(expand_wrapped_sentence_fragments(sentence, lang=lang))

        if source == "wiki":
            return post_clean_sentences(
                expanded,
                lang,
                lang_to_group,
                use_nltk_secondary=_wiki_use_nltk_secondary(lang, lang_to_group),
            )
        return post_clean_sentences(
            expanded,
            lang,
            lang_to_group,
            use_nltk_secondary=lang_to_group.get(canonical_lang(lang)) not in LATIN_GROUPS,
            use_major_latin_leak=True,
        )

    return _transform


def _discover_cached_langs(cache_dir: str) -> list[str]:
    langs: list[str] = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".parquet"):
            continue
        lang = canonical_lang(name[:-8])
        if lang in LANG_TO_GROUP:
            langs.append(lang)
    return langs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split wrapped quote/list sentence caches after fetch.")
    parser.add_argument(
        "--source",
        choices=("wiki", "finetrans"),
        default="wiki",
        help="Which cache layout to process.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the cache directory instead of using the source default.",
    )
    parser.add_argument(
        "--lang",
        help="Only process one language code.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_dir = args.cache_dir or PATHS[args.source]["cache_dir"]
    if args.lang:
        langs = [canonical_lang(args.lang)]
    else:
        langs = _discover_cached_langs(cache_dir)

    transform = _build_transform(args.source)
    updated_counts: dict[str, int] = {}
    for lang in tqdm(langs, desc=f"{args.source} caches"):
        result = refilter_cached_sentence_parquets(
            [lang],
            path_for_lang=lambda current_lang: os.path.join(cache_dir, f"{current_lang}.parquet"),
            lang_to_group=LANG_TO_GROUP,
            should_skip_lang=lambda current_lang, _lang_to_group: current_lang not in LANG_TO_GROUP,
            transform_sentences=transform,
        )
        updated_counts.update(result)

    total_sentences = sum(updated_counts.values())
    print(f"Updated {len(updated_counts)} {args.source} cache(s) ({total_sentences:,} sentences kept).")


if __name__ == "__main__":
    main()
