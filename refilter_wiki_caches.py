from __future__ import annotations

import argparse

from wiki_sources import refilter_cached_wiki_sentences


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run wiki post-filtering over cached sentence files."
    )
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help="Refilter every cached wiki language instead of Latin-group languages only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    updated_counts = refilter_cached_wiki_sentences(latin_only=not args.all_langs)
    total_langs = len(updated_counts)
    total_sentences = sum(updated_counts.values())
    scope = "Latin-group" if not args.all_langs else "all"
    print(f"Refiltered {total_langs} {scope} wiki caches ({total_sentences:,} sentences kept).")


if __name__ == "__main__":
    main()
