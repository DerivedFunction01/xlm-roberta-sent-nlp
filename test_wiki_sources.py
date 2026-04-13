from __future__ import annotations

import unittest

from language import LANG_TO_GROUP
from wiki_sources import (
    MAX_WIKI_SENTENCES,
    _wiki_cap_multiplier,
    _wiki_use_nltk_secondary,
    max_wiki_sentences_for_lang,
)


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

    def test_wiki_nltk_secondary_is_only_for_minor_latin_groups(self) -> None:
        self.assertTrue(_wiki_use_nltk_secondary("su", LANG_TO_GROUP))
        self.assertFalse(_wiki_use_nltk_secondary("fr", LANG_TO_GROUP))


if __name__ == "__main__":
    unittest.main()
