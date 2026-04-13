from __future__ import annotations

import tempfile
import unittest

from finetranslations_sources import _has_finetrans_cache_files


class FinetranslationsCacheTests(unittest.TestCase):
    def test_empty_cache_directory_is_not_treated_as_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(_has_finetrans_cache_files(tmpdir))


if __name__ == "__main__":
    unittest.main()
