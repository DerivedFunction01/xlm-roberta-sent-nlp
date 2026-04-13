from __future__ import annotations

import string
import unittest

import text_utils
from language import LANGUAGE_GROUPS


def _alpha_token(index: int) -> str:
    letters = string.ascii_lowercase
    chars: list[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        chars.append(letters[remainder])
        if value == 0:
            break
        value -= 1
    return "w" + "".join(reversed(chars))


class DummyWordsCorpus:
    def __init__(self, words: list[str]) -> None:
        self._words = words

    def words(self) -> list[str]:
        return self._words


class EnglishLeakFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_nltk_words = text_utils.nltk_words
        self._lang_to_group = {
            lang: group
            for group, langs in LANGUAGE_GROUPS.items()
            for lang in langs
        }
        text_utils._nltk_english_secondary_word_set.cache_clear()

    def tearDown(self) -> None:
        text_utils.nltk_words = self._orig_nltk_words
        text_utils._nltk_english_secondary_word_set.cache_clear()

    def test_secondary_word_set_keeps_late_corpus_entries(self) -> None:
        corpus_words = [_alpha_token(i) for i in range(50_001)]
        late_word = "zebracarpet"
        text_utils.nltk_words = DummyWordsCorpus(corpus_words + [late_word])
        secondary = text_utils._nltk_english_secondary_word_set()

        self.assertIn(late_word, secondary)

    def test_clean_sentence_drops_english_leakage_for_non_english_lang(self) -> None:
        text_utils.nltk_words = DummyWordsCorpus(
            [
                "simple",
                "english",
                "clear",
                "speech",
                "leak",
            ]
        )
        sentence = "This is the simple English clear speech leak."

        self.assertEqual(
            text_utils.clean_sentence(sentence, "fr", self._lang_to_group),
            "",
        )


if __name__ == "__main__":
    unittest.main()
