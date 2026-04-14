from __future__ import annotations

from collections import deque
import unittest
from unittest.mock import patch

import synthetic_build


class DummyTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


class AccentStrippingTests(unittest.TestCase):
    def test_strip_latin_accents_removes_diacritics(self) -> None:
        self.assertEqual(
            synthetic_build._strip_latin_accents("mañana cómo está São Tomé"),
            "manana como esta Sao Tome",
        )

    def test_random_accent_stripping_is_latin_only(self) -> None:
        with patch("synthetic_build.random.random", return_value=0.0):
            self.assertEqual(
                synthetic_build._apply_random_accent_stripping(
                    "mañana cómo está",
                    lang="es",
                    prob=1.0,
                ),
                "manana como esta",
            )
            self.assertEqual(
                synthetic_build._apply_random_accent_stripping(
                    "Привет мир",
                    lang="ru",
                    prob=1.0,
                ),
                "Привет мир",
            )

    def test_pure_doc_build_can_emit_accentless_variant(self) -> None:
        tokenizer = DummyTokenizer()
        primary_pool = {"es": deque(["mañana cómo está"])}
        label2id = {"B-ES": 1, "I-ES": 2}

        with patch("synthetic_build.random.random", return_value=0.0):
            example = synthetic_build.create_pure_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                lang="es",
                label2id=label2id,
                min_sentences=1,
                max_sentences=1,
                strip_punct_prob=0.0,
                accent_strip_prob=1.0,
                format_noise_prob=0.0,
                paragraph_break_prob=0.0,
                uppercase_word_prob=0.0,
                lowercase_word_prob=0.0,
                titlecase_word_prob=0.0,
                merge_word_prob=0.0,
                split_word_prob=0.0,
                typo_char_prob=0.0,
            )

        self.assertEqual(example["original_text"], "manana como esta")
        self.assertEqual(example["tokens"], ["manana", "como", "esta"])

    def test_pure_doc_can_uppercase_full_sentence(self) -> None:
        tokenizer = DummyTokenizer()
        primary_pool = {"en": deque(["Hello World"])}
        label2id = {"B-EN": 1, "I-EN": 2}

        with patch("synthetic_build.random.random", return_value=0.0):
            example = synthetic_build.create_pure_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                lang="en",
                label2id=label2id,
                min_sentences=1,
                max_sentences=1,
                strip_punct_prob=0.0,
                accent_strip_prob=0.0,
                sentence_uppercase_prob=1.0,
                sentence_lowercase_prob=0.0,
                foreign_sentence_prob=0.0,
                format_noise_prob=0.0,
                paragraph_break_prob=0.0,
                uppercase_word_prob=0.0,
                lowercase_word_prob=0.0,
                titlecase_word_prob=0.0,
                merge_word_prob=0.0,
                split_word_prob=0.0,
                typo_char_prob=0.0,
            )

        self.assertEqual(example["original_text"], "HELLO WORLD")
        self.assertEqual(example["tokens"], ["HELLO", "WORLD"])

    def test_pure_doc_can_lowercase_full_sentence(self) -> None:
        tokenizer = DummyTokenizer()
        primary_pool = {"en": deque(["Hello World"])}
        label2id = {"B-EN": 1, "I-EN": 2}

        with patch("synthetic_build.random.random", return_value=0.0):
            example = synthetic_build.create_pure_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                lang="en",
                label2id=label2id,
                min_sentences=1,
                max_sentences=1,
                strip_punct_prob=0.0,
                accent_strip_prob=0.0,
                sentence_uppercase_prob=0.0,
                sentence_lowercase_prob=1.0,
                foreign_sentence_prob=0.0,
                format_noise_prob=0.0,
                paragraph_break_prob=0.0,
                uppercase_word_prob=0.0,
                lowercase_word_prob=0.0,
                titlecase_word_prob=0.0,
                merge_word_prob=0.0,
                split_word_prob=0.0,
                typo_char_prob=0.0,
            )

        self.assertEqual(example["original_text"], "hello world")
        self.assertEqual(example["tokens"], ["hello", "world"])

    def test_homogeneous_doc_stays_single_language(self) -> None:
        tokenizer = DummyTokenizer()
        primary_pool = {"en": deque(["hello", "world", "again"])}
        label2id = {"B-EN": 1, "I-EN": 2}

        with patch("synthetic_build.random.random", return_value=0.0):
            example = synthetic_build.create_pure_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                lang="en",
                label2id=label2id,
                min_sentences=3,
                max_sentences=3,
                strip_punct_prob=0.0,
                accent_strip_prob=0.0,
                foreign_sentence_prob=0.0,
                format_noise_prob=0.0,
                paragraph_break_prob=0.0,
                uppercase_word_prob=0.0,
                lowercase_word_prob=0.0,
                titlecase_word_prob=0.0,
                merge_word_prob=0.0,
                split_word_prob=0.0,
                typo_char_prob=0.0,
            )

        self.assertEqual(example["original_text"], "hello world again")
        self.assertEqual(example["tokens"], ["hello", "world", "again"])

    def test_spliced_doc_can_insert_one_foreign_sentence(self) -> None:
        tokenizer = DummyTokenizer()
        primary_pool = {
            "en": deque(["WORLD!", "Again."]),
            "es": deque(["HOLA!"]),
        }
        label2id = {"B-EN": 1, "I-EN": 2, "B-ES": 3, "I-ES": 4}

        with patch("synthetic_build.random.random", return_value=0.0), patch(
            "synthetic_build.random.choices",
            return_value=["es"],
        ), patch("synthetic_build.random.randrange", return_value=1):
            example = synthetic_build.create_pure_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                lang="en",
                label2id=label2id,
                min_sentences=3,
                max_sentences=3,
                strip_punct_prob=0.0,
                accent_strip_prob=0.0,
                foreign_sentence_prob=1.0,
                splice_strip_next_punct_prob=1.0,
                splice_lowercase_next_prob=1.0,
                format_noise_prob=0.0,
                paragraph_break_prob=0.0,
                uppercase_word_prob=0.0,
                lowercase_word_prob=0.0,
                titlecase_word_prob=0.0,
                merge_word_prob=0.0,
                split_word_prob=0.0,
                typo_char_prob=0.0,
            )

        self.assertEqual(example["original_text"], "WORLD! HOLA! again")
        self.assertEqual(example["tokens"], ["WORLD!", "HOLA!", "again"])

    def test_spliced_doc_foreign_sentence_stays_interior(self) -> None:
        tokenizer = DummyTokenizer()
        primary_pool = {
            "en": deque(["first", "second", "third", "fourth"]),
            "es": deque(["hola"]),
        }
        label2id = {"B-EN": 1, "I-EN": 2, "B-ES": 3, "I-ES": 4}

        with patch("synthetic_build.random.random", return_value=0.0), patch(
            "synthetic_build.random.choices",
            return_value=["es"],
        ), patch("synthetic_build.random.randrange", return_value=1):
            example = synthetic_build.create_pure_synthetic_doc(
                tokenizer=tokenizer,
                primary_pool=primary_pool,
                lang="en",
                label2id=label2id,
                min_sentences=3,
                max_sentences=3,
                strip_punct_prob=0.0,
                accent_strip_prob=0.0,
                foreign_sentence_prob=1.0,
                splice_strip_next_punct_prob=1.0,
                splice_lowercase_next_prob=1.0,
                format_noise_prob=0.0,
                paragraph_break_prob=0.0,
                uppercase_word_prob=0.0,
                lowercase_word_prob=0.0,
                titlecase_word_prob=0.0,
                merge_word_prob=0.0,
                split_word_prob=0.0,
                typo_char_prob=0.0,
            )

        self.assertEqual(example["original_text"], "first hola second")
        self.assertEqual(example["tokens"], ["first", "hola", "second"])


if __name__ == "__main__":
    unittest.main()
