from __future__ import annotations

import argparse
import os
from language import LANG_TO_GROUP, canonical_lang
from paths import PATHS
from refilter_shared import (
    build_refilter_arg_parser,
    collect_rejected_english_sentences_from_parquet,
    emit_rejected_english_leaks,
    validate_refilter_args,
)
from finetranslations_sources import refilter_cached_finetranslations_sentences
from text_utils import LATIN_GROUPS


def _parse_args() -> argparse.Namespace:
    parser = build_refilter_arg_parser(
        description="Re-run English-leak post-filtering over cached FineTranslations sentence files.",
        lang_help="Only process one language code (for example no or fr).",
        all_langs_help="Refilter every cached FineTranslations language instead of just the selected one.",
        leak_out_dir_default=os.path.join(PATHS["finetrans"]["cache_dir"], "_english_leaks"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_refilter_args(args)

    if args.lang:
        if args.emit_leaks:
            lang = canonical_lang(args.lang)
            emit_rejected_english_leaks(
                lang=lang,
                leak_out_dir=args.leak_out_dir,
                collector=lambda _lang: collect_rejected_english_sentences_from_parquet(
                    lang=lang,
                    path=os.path.join(PATHS["finetrans"]["cache_dir"], f"{lang}.parquet"),
                    lang_to_group=LANG_TO_GROUP,
                    use_nltk_secondary=LANG_TO_GROUP.get(lang) not in LATIN_GROUPS,
                    use_major_latin_leak=True,
                ),
            )
        updated_counts = refilter_cached_finetranslations_sentences([args.lang], use_major_latin_leak=True)
        total_langs = len(updated_counts)
        total_sentences = sum(updated_counts.values())
        print(f"Refiltered {total_langs} FineTranslations cache ({total_sentences:,} sentences kept).")
        return

    updated_counts = refilter_cached_finetranslations_sentences(use_major_latin_leak=True)
    total_langs = len(updated_counts)
    total_sentences = sum(updated_counts.values())
    print(f"Refiltered {total_langs} FineTranslations caches ({total_sentences:,} sentences kept).")


if __name__ == "__main__":
    main()
