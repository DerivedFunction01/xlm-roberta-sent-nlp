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
                output_dir=output_dir,
                langs=("fr",),
                top_n=10,
            )
            lexicon = load_wiki_major_latin_lexicon("fr", cache_dir=output_dir)

            self.assertEqual(counts["fr"], 7)
            self.assertIn("bonjour", lexicon)
            self.assertIn("monde", lexicon)


if __name__ == "__main__":
    unittest.main()
