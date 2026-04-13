from __future__ import annotations

import unittest

from wiki_sources import MAX_WIKI_SENTENCES, _wiki_cap_multiplier, max_wiki_sentences_for_lang


class WikiCapTests(unittest.TestCase):
    def test_no_plus_nn_counts_as_full_bucket(self) -> None:
        self.assertEqual(_wiki_cap_multiplier(("no", "nn")), 1.0)
        self.assertEqual(max_wiki_sentences_for_lang("no", source_langs=("no", "nn")), MAX_WIKI_SENTENCES)

    def test_single_language_uses_its_own_multiplier(self) -> None:
        self.assertEqual(_wiki_cap_multiplier(("en",)), 1.5)
        self.assertEqual(
            max_wiki_sentences_for_lang("en", source_langs=("en",)),
            int(round(MAX_WIKI_SENTENCES * 1.5)),
        )


if __name__ == "__main__":
    unittest.main()
