from __future__ import annotations

import unittest

from instruction_sources import _is_valid_instruction_text
from language import LANGUAGE_GROUPS


class InstructionMathFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._lang_to_group = {
            lang: group
            for group, langs in LANGUAGE_GROUPS.items()
            for lang in langs
        }

    def test_rejects_simple_math_expression(self) -> None:
        text = "3 + 2 x 5"

        self.assertFalse(_is_valid_instruction_text(text, "en", self._lang_to_group))

    def test_rejects_equation_like_expression(self) -> None:
        text = "2 + 2 = 4"

        self.assertFalse(_is_valid_instruction_text(text, "en", self._lang_to_group))


if __name__ == "__main__":
    unittest.main()
