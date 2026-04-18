from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from convert_tatoeba_sentences import convert_tatoeba_sentences


class ConvertTatoebaSentencesTests(unittest.TestCase):
    def test_convert_tatoeba_sentences_strips_non_matching_script_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "tatoeba.tsv"
            output_dir = Path(tmpdir) / "out"
            input_path.write_text(
                "1\tfr\tBonjour العربية monde\n",
                encoding="utf-8",
            )

            meta = convert_tatoeba_sentences(
                input_path=input_path,
                output_dir=output_dir,
                flush_rows=1,
                force_rebuild=True,
            )
            table = pq.read_table(output_dir / "fr.parquet")

            self.assertEqual(meta["written_rows"], 1)
            self.assertEqual(table.column("sentence").to_pylist(), ["Bonjour  monde"])

    def test_convert_tatoeba_sentences_drops_rows_with_only_wrong_script_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "tatoeba.tsv"
            output_dir = Path(tmpdir) / "out"
            input_path.write_text(
                "1\tfr\tالعربية\n",
                encoding="utf-8",
            )

            meta = convert_tatoeba_sentences(
                input_path=input_path,
                output_dir=output_dir,
                flush_rows=1,
                force_rebuild=True,
            )

            self.assertEqual(meta["written_rows"], 0)
            self.assertFalse((output_dir / "fr.parquet").exists())


if __name__ == "__main__":
    unittest.main()
