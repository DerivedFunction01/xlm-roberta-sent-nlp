from __future__ import annotations

import tempfile
import unittest

from finetranslations_sources import _config_name_to_lang, _has_finetrans_cache_files
from language import LANG_TO_GROUP


class FinetranslationsCacheTests(unittest.TestCase):
    def test_empty_cache_directory_is_not_treated_as_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(_has_finetrans_cache_files(tmpdir))

    def test_hbo_and_grc_configs_normalize_to_he_and_el(self) -> None:
        self.assertEqual(_config_name_to_lang("hbo", LANG_TO_GROUP), "he")
        self.assertEqual(_config_name_to_lang("grc", LANG_TO_GROUP), "el")

    def test_xh_configs_normalize_to_xh(self) -> None:
        self.assertEqual(_config_name_to_lang("xh", LANG_TO_GROUP), "xh")
        self.assertEqual(_config_name_to_lang("xh_Latn", LANG_TO_GROUP), "xh")


if __name__ == "__main__":
    unittest.main()
