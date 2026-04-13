from __future__ import annotations

import argparse
import os

from io_utils import write_records_parquet
from paths import PATHS
from wiki_sources import collect_rejected_english_sentences, refilter_cached_wiki_sentences


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run wiki post-filtering over cached sentence files."
    )
    parser.add_argument(
        "--lang",
        help="Only process one language code (for example su or fr).",
    )
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help="Refilter every cached wiki language instead of Latin-group languages only.",
    )
    parser.add_argument(
        "--emit-leaks",
        action="store_true",
        help="Write rejected English-looking sentences for the selected language to parquet.",
    )
    parser.add_argument(
        "--leak-out-dir",
        default=os.path.join(PATHS["wiki"]["cache_dir"], "_english_leaks"),
        help="Directory for leaked-sentence parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.lang and args.all_langs:
        raise SystemExit("--lang and --all-langs cannot be used together.")
    if args.emit_leaks and not args.lang:
        raise SystemExit("--emit-leaks requires --lang.")

    if args.lang:
        if args.emit_leaks:
            leaks = collect_rejected_english_sentences(args.lang)
            os.makedirs(args.leak_out_dir, exist_ok=True)
            leak_path = os.path.join(args.leak_out_dir, f"{args.lang}.parquet")
            write_records_parquet(
                leak_path,
                leaks,
                columns=["lang", "source_index", "sentence", "use_nltk_secondary"],
            )
            print(f"Wrote {len(leaks):,} rejected English-looking sentences -> {leak_path}")
        updated_counts = refilter_cached_wiki_sentences([args.lang], latin_only=False)
        total_langs = len(updated_counts)
        total_sentences = sum(updated_counts.values())
        print(f"Refiltered {total_langs} wiki cache ({total_sentences:,} sentences kept).")
        return

    updated_counts = refilter_cached_wiki_sentences(latin_only=not args.all_langs)
    total_langs = len(updated_counts)
    total_sentences = sum(updated_counts.values())
    scope = "Latin-group" if not args.all_langs else "all"
    print(f"Refiltered {total_langs} {scope} wiki caches ({total_sentences:,} sentences kept).")


if __name__ == "__main__":
    main()
