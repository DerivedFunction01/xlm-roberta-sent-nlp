from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from typing import Any

import pandas as pd

from io_utils import write_records_parquet, write_sentence_parquet
from language import canonical_lang
from text_utils import _sentence_cleanup_reason


def build_refilter_arg_parser(
    *,
    description: str,
    lang_help: str,
    all_langs_help: str,
    leak_out_dir_default: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--lang",
        help=lang_help,
    )
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help=all_langs_help,
    )
    parser.add_argument(
        "--emit-leaks",
        action="store_true",
        help="Write rejected English-looking sentences for the selected language to parquet.",
    )
    parser.add_argument(
        "--leak-out-dir",
        default=leak_out_dir_default,
        help="Directory for leaked-sentence parquet files.",
    )
    return parser


def validate_refilter_args(args: argparse.Namespace) -> None:
    if args.lang and args.all_langs:
        raise SystemExit("--lang and --all-langs cannot be used together.")
    if args.emit_leaks and not args.lang:
        raise SystemExit("--emit-leaks requires --lang.")


def emit_rejected_english_leaks(
    *,
    lang: str,
    leak_out_dir: str,
    collector: Callable[[str], list[dict[str, Any]]],
) -> int:
    leaks = collector(lang)
    os.makedirs(leak_out_dir, exist_ok=True)
    leak_path = os.path.join(leak_out_dir, f"{lang}.parquet")
    write_records_parquet(
        leak_path,
        leaks,
        columns=["lang", "source_index", "sentence", "reason", "use_nltk_secondary"],
    )
    print(f"Wrote {len(leaks):,} rejected sentences -> {leak_path}")
    return len(leaks)


def collect_rejected_english_sentences_from_parquet(
    *,
    lang: str,
    path: str,
    lang_to_group: dict[str, str],
    use_nltk_secondary: bool,
    use_major_latin_leak: bool = False,
) -> list[dict[str, Any]]:
    lang = canonical_lang(lang)
    if lang == "en" or not os.path.exists(path):
        return []
    frame = pd.read_parquet(path)
    if "sentence" not in frame.columns:
        return []
    rejected: list[dict[str, Any]] = []
    for idx, sentence in enumerate(frame["sentence"].astype(str).tolist()):
        reason = _sentence_cleanup_reason(
            sentence,
            lang,
            lang_to_group,
            use_nltk_secondary=use_nltk_secondary,
            use_major_latin_leak=use_major_latin_leak,
        )
        if reason:
            rejected.append(
                {
                    "lang": lang,
                    "source_index": idx,
                    "sentence": sentence,
                    "reason": reason,
                    "use_nltk_secondary": use_nltk_secondary,
                }
            )
    return rejected


def refilter_cached_sentence_parquets(
    langs: list[str],
    *,
    path_for_lang: Callable[[str], str],
    lang_to_group: dict[str, str],
    should_skip_lang: Callable[[str, dict[str, str]], bool],
    transform_sentences: Callable[[str, list[str], dict[str, str]], list[str]],
) -> dict[str, int]:
    updated_counts: dict[str, int] = {}
    for raw_lang in langs:
        lang = canonical_lang(raw_lang)
        if should_skip_lang(lang, lang_to_group):
            continue
        path = path_for_lang(lang)
        if not os.path.exists(path):
            continue
        frame = pd.read_parquet(path)
        if "sentence" not in frame.columns:
            continue
        cached = frame["sentence"].astype(str).tolist()
        cleaned = transform_sentences(lang, cached, lang_to_group)
        if cleaned != cached:
            write_sentence_parquet(path, cleaned)
        updated_counts[lang] = len(cleaned)
    return updated_counts
