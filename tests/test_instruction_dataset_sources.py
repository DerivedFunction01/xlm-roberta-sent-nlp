from __future__ import annotations

import unittest

from instruction_dataset_sources import _extract_aya_hindi_texts


class AyaHindiExtractorTests(unittest.TestCase):
    def test_extract_aya_hindi_texts_strips_prefix_before_colon(self) -> None:
        row = {
            "inputs": "प्रश्न: क्या वाक्य 1 और वाक्य 2 एक ही अर्थ व्यक्त करते हैं? हाँ या नहीं?",
            "targets": "उत्तर: हाँ",
        }

        self.assertEqual(
            _extract_aya_hindi_texts(row, {}),
            [
                "क्या वाक्य 1 और वाक्य 2 एक ही अर्थ व्यक्त करते हैं? हाँ या नहीं?",
                "हाँ",
            ],
        )

    def test_extract_aya_hindi_texts_keeps_strings_without_colon(self) -> None:
        row = {
            "inputs": "यह एक सादा वाक्य है",
            "targets": "हाँ",
        }

        self.assertEqual(
            _extract_aya_hindi_texts(row, {}),
            [
                "यह एक सादा वाक्य है",
                "हाँ",
            ],
        )


if __name__ == "__main__":
    unittest.main()
