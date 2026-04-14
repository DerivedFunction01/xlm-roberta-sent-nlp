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

    def test_homogeneous_doc_can_splice_one_foreign_sentence(self) -> None:
        tokenizer = DummyTokenizer()
        primary_pool = {
            "en": deque(["hello", "world", "again"]),
            "es": deque(["hola"]),
        }
        label2id = {"B-EN": 1, "I-EN": 2, "B-ES": 3, "I-ES": 4}

        with patch("synthetic_build.random.random", return_value=0.0), patch(
            "synthetic_build.random.choices",
            return_value=["es"],
        ), patch("synthetic_build.random.randrange", return_value=1), patch(
            "synthetic_build.random.randint",
            return_value=3,
        ):
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
                format_noise_prob=0.0,
                paragraph_break_prob=0.0,
                uppercase_word_prob=0.0,
                lowercase_word_prob=0.0,
                titlecase_word_prob=0.0,
                merge_word_prob=0.0,
                split_word_prob=0.0,
                typo_char_prob=0.0,
            )

        self.assertEqual(example["original_text"], "hello hola world")
        self.assertEqual(example["tokens"], ["hello", "hola", "world"])


if __name__ == "__main__":
    unittest.main()
