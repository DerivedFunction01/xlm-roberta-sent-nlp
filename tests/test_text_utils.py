from __future__ import annotations

import string
import unittest

import text_utils
from language import ALL_LANGS, LANG_TO_GROUP, LANGUAGE_GROUPS


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

    def test_minor_latin_groups_allow_one_local_hit(self) -> None:
        original_local = text_utils._is_local_english_stopword
        original_broad = text_utils._is_broad_english_word
        text_utils._is_local_english_stopword = lambda token: token == "alpha"  # type: ignore[assignment]
        text_utils._is_broad_english_word = lambda token: token in {"alpha", "beta", "gamma", "delta"}  # type: ignore[assignment]
        try:
            sentence = "alpha beta gamma delta"
            self.assertTrue(
                text_utils._looks_like_english_text(sentence, "su", self._lang_to_group)
            )
            self.assertFalse(
                text_utils._looks_like_english_text(sentence, "fr", self._lang_to_group)
            )
        finally:
            text_utils._is_local_english_stopword = original_local
            text_utils._is_broad_english_word = original_broad

    def test_minor_latin_groups_catch_short_english_leak_sentence(self) -> None:
        text_utils.nltk_words = DummyWordsCorpus(
            [
                "combines",
                "neuroscience",
                "economics",
                "psychology",
                "study",
                "choices",
            ]
        )
        sentence = "This combines neuroscience, economics, and psychology to study how we make choices."

        self.assertTrue(
            text_utils.clean_sentence(sentence, "su", self._lang_to_group) == ""
        )

    def test_yiddish_uses_hebrew_pysbd_fallback(self) -> None:
        self.assertEqual(text_utils.PYSBD_FALLBACKS["yi"], "he")

    def test_every_language_alias_has_a_group(self) -> None:
        missing = [lang for lang in ALL_LANGS if lang not in LANG_TO_GROUP]
        self.assertEqual(missing, [], msg=f"Missing language groups for: {missing}")


class TokenCountTests(unittest.TestCase):
    def test_valid_non_digit_non_symbol_token_count_ignores_math_tokens(self) -> None:
        self.assertEqual(
            text_utils._valid_non_digit_non_symbol_token_count("3 + 2 x 5"),
            1,
        )

    def test_valid_non_digit_non_symbol_token_count_counts_real_words(self) -> None:
        self.assertEqual(
            text_utils._valid_non_digit_non_symbol_token_count("यह एक सादा वाक्य है"),
            5,
        )

    def test_valid_non_digit_non_symbol_token_count_ignores_trailing_punctuation(self) -> None:
        self.assertEqual(
            text_utils._valid_non_digit_non_symbol_token_count("This is a valid sentence."),
            5,
        )

    def test_valid_sentence_accepts_cjk_without_whitespace_tokens(self) -> None:
        lang_to_group = {
            lang: group
            for group, langs in LANGUAGE_GROUPS.items()
            for lang in langs
        }
        self.assertTrue(
            text_utils._is_valid_sentence("今天天气很好啊啊啊。", "zh", lang_to_group)
        )

    def test_sentence_split_fallback_handles_cjk_without_spaces(self) -> None:
        pieces = [piece for piece in text_utils.SENT_SPLIT.split("句子一。句子二。") if piece]
        self.assertEqual(pieces, ["句子一。", "句子二。"])

    def test_sentence_split_handles_ascii_and_cjk_punctuation(self) -> None:
        pieces = [piece for piece in text_utils.SENT_SPLIT.split("Hello world. 你好世界。Next.") if piece]
        self.assertEqual(pieces, ["Hello world.", "你好世界。", "Next."])


if __name__ == "__main__":
    unittest.main()
