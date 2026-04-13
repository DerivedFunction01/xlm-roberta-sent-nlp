from __future__ import annotations

import json
import tempfile
import unittest

import text_utils
from io_utils import write_sentence_parquet
from language import LANG_TO_GROUP
from wiki_sources import (
    MAX_WIKI_SENTENCES,
    _wiki_cap_multiplier,
    _wiki_use_nltk_secondary,
    collect_rejected_english_sentences,
    load_or_extract,
    max_wiki_sentences_for_lang,
)


class DummyWordsCorpus:
    def __init__(self, words: list[str]) -> None:
        self._words = words

    def words(self) -> list[str]:
        return self._words


class DummyNltkModule:
    class _Data:
        def find(self, resource: str) -> None:
            return None

    def __init__(self) -> None:
        self.data = self._Data()

    def download(self, name: str, quiet: bool = False, raise_on_error: bool = False) -> None:
        return None


class WikiCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_nltk_module = text_utils.nltk_module
        self._orig_nltk_words = text_utils.nltk_words
        text_utils._nltk_english_secondary_word_set.cache_clear()

    def tearDown(self) -> None:
        text_utils.nltk_module = self._orig_nltk_module
        text_utils.nltk_words = self._orig_nltk_words
        text_utils._nltk_english_secondary_word_set.cache_clear()

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

    def test_collect_rejected_english_sentences_writes_candidates(self) -> None:
        text_utils.nltk_module = DummyNltkModule()
        text_utils.nltk_words = DummyWordsCorpus(["simple", "english", "clear", "speech", "leak"])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_sentence_parquet(
                f"{tmpdir}/su.parquet",
                [
                    "This is the simple English clear speech leak.",
                    "Ngeunaan basa Sunda.",
                ],
            )
            rejected = collect_rejected_english_sentences("su", sentences_dir=tmpdir)

        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["sentence"], "This is the simple English clear speech leak.")

    def test_load_or_extract_creates_meta_for_external_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            write_sentence_parquet(f"{tmpdir}/su.parquet", ["Ngeunaan basa Sunda."])

            lang, path = load_or_extract(
                "su",
                lang_to_group=LANG_TO_GROUP,
                sentences_dir=tmpdir,
            )

            self.assertEqual(lang, "su")
            self.assertEqual(path, f"{tmpdir}/su.parquet")
            with open(f"{tmpdir}/su.meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["lang"], "su")
            self.assertEqual(meta["source_langs"], ["su"])
            self.assertEqual(meta["status"], "complete")


if __name__ == "__main__":
    unittest.main()
