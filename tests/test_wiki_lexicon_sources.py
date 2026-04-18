from __future__ import annotations

import tempfile
import unittest
import os

from io_utils import write_sentence_parquet
from wiki_lexicon_sources import build_wiki_major_latin_lexicon_cache, load_wiki_major_latin_lexicon


class WikiLexiconCacheTests(unittest.TestCase):
    def test_build_wiki_major_latin_lexicon_cache_writes_top_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_cache_dir = f"{tmpdir}/wiki"
            tatoeba_cache_dir = f"{tmpdir}/tatoeba"
            output_dir = f"{tmpdir}/lexicon"
            os.makedirs(wiki_cache_dir, exist_ok=True)
            write_sentence_parquet(
                f"{wiki_cache_dir}/fr.parquet",
                [
                    "Bonjour le monde.",
                    "Bonjour encore et encore.",
                    "Le monde est beau.",
                ],
            )

            counts = build_wiki_major_latin_lexicon_cache(
                wiki_cache_dir=wiki_cache_dir,
                tatoeba_cache_dir=tatoeba_cache_dir,
                output_dir=output_dir,
                langs=("fr",),
                top_n=10,
            )
            lexicon = load_wiki_major_latin_lexicon("fr", cache_dir=output_dir)

            self.assertEqual(counts["fr"], 7)
            self.assertIn("bonjour", lexicon)
            self.assertIn("monde", lexicon)

    def test_build_wiki_major_latin_lexicon_cache_can_use_tatoeba_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tatoeba_cache_dir = f"{tmpdir}/tatoeba"
            output_dir = f"{tmpdir}/lexicon"
            os.makedirs(tatoeba_cache_dir, exist_ok=True)
            write_sentence_parquet(
                f"{tatoeba_cache_dir}/fr.parquet",
                [
                    "Bonjour le monde.",
                    "Bonjour encore.",
                    "Le monde est calme.",
                ],
            )

            counts = build_wiki_major_latin_lexicon_cache(
                wiki_cache_dir=f"{tmpdir}/missing_wiki",
                tatoeba_cache_dir=tatoeba_cache_dir,
                output_dir=output_dir,
                langs=("fr",),
                top_n=10,
                sources=("tatoeba",),
            )
            lexicon = load_wiki_major_latin_lexicon("fr", cache_dir=output_dir)

            self.assertEqual(counts["fr"], 6)
            self.assertIn("bonjour", lexicon)
            self.assertIn("calme", lexicon)

    def test_build_wiki_major_latin_lexicon_cache_merges_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_cache_dir = f"{tmpdir}/wiki"
            tatoeba_cache_dir = f"{tmpdir}/tatoeba"
            output_dir = f"{tmpdir}/lexicon"
            os.makedirs(wiki_cache_dir, exist_ok=True)
            os.makedirs(tatoeba_cache_dir, exist_ok=True)
            write_sentence_parquet(
                f"{wiki_cache_dir}/fr.parquet",
                [
                    "Bonjour le monde.",
                    "Monde du wiki.",
                ],
            )
            write_sentence_parquet(
                f"{tatoeba_cache_dir}/fr.parquet",
                [
                    "Bonjour le monde.",
                    "Monde de Tatoeba.",
                ],
            )

            counts = build_wiki_major_latin_lexicon_cache(
                wiki_cache_dir=wiki_cache_dir,
                tatoeba_cache_dir=tatoeba_cache_dir,
                output_dir=output_dir,
                langs=("fr",),
                top_n=10,
                sources=("wiki", "tatoeba"),
            )
            lexicon = load_wiki_major_latin_lexicon("fr", cache_dir=output_dir)

            self.assertEqual(counts["fr"], 7)
            self.assertIn("bonjour", lexicon)
            self.assertIn("tatoeba", lexicon)


if __name__ == "__main__":
    unittest.main()
