from __future__ import annotations

import unittest

from language import dataset_label_script, is_dataset_label_script_compatible
from script_types import Script


class DatasetLabelScriptTests(unittest.TestCase):
    def test_dataset_label_script_parses_script_suffix(self) -> None:
        self.assertEqual(dataset_label_script("arb_Latn"), Script.LATIN)
        self.assertEqual(dataset_label_script("rus_Cyrl"), Script.CYRILLIC)
        self.assertEqual(dataset_label_script("eng_Latn"), Script.LATIN)

    def test_latin_script_rows_are_filtered_for_non_latin_languages(self) -> None:
        self.assertFalse(is_dataset_label_script_compatible("ar", "arb_Latn"))
        self.assertFalse(is_dataset_label_script_compatible("ja", "jpn_Latn"))
        self.assertTrue(is_dataset_label_script_compatible("ar", "arb_Arab"))
        self.assertTrue(is_dataset_label_script_compatible("en", "eng_Latn"))
