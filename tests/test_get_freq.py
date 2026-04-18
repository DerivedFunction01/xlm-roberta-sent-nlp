from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import get_freq


class _FakeResponse:
    def __init__(self, text: str, *, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.response = self

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise get_freq.requests.HTTPError(response=self)


class _FakeTokenizer:
    def __call__(self, tokens, *, is_split_into_words, truncation, add_special_tokens):
        assert is_split_into_words
        assert truncation
        assert add_special_tokens
        input_ids = [101, *[1000 + idx for idx, _ in enumerate(tokens)], 102]
        attention_mask = [1] * len(input_ids)
        word_ids = [None, *range(len(tokens)), None]

        class _Encoding(dict):
            def __init__(self, input_ids, attention_mask, word_ids):
                super().__init__(input_ids=input_ids, attention_mask=attention_mask)
                self._word_ids = word_ids

            def word_ids(self):
                return self._word_ids

        return _Encoding(input_ids, attention_mask, word_ids)


class _SplitTokenizer:
    def __call__(self, tokens, *, is_split_into_words, truncation, add_special_tokens):
        assert is_split_into_words
        assert truncation
        assert add_special_tokens

        # Simulate XLM-R splitting an Arabic word into three wordpieces.
        input_ids = [101, 2001, 2002, 2003, 102]
        attention_mask = [1] * len(input_ids)
        word_ids = [None, 0, 0, 0, None]

        class _Encoding(dict):
            def __init__(self, input_ids, attention_mask, word_ids):
                super().__init__(input_ids=input_ids, attention_mask=attention_mask)
                self._word_ids = word_ids

            def word_ids(self):
                return self._word_ids

        return _Encoding(input_ids, attention_mask, word_ids)


class GetFreqParsingTests(unittest.TestCase):
    def test_infer_frequency_column_prefers_majority_sample_format(self) -> None:
        sample_rows = [
            ["100", "bonjour"],
            ["90", "salut"],
            ["80", "merci"],
            ["word", "70"],
        ]

        self.assertEqual(get_freq._infer_freq_column(sample_rows), 0)

    def test_fetch_wordlist_handles_mixed_column_order_and_skips_contamination(self) -> None:
        response_text = "\n".join(
            [
                "100 bonjour",
                "90 salut",
                "80 merci",
                "bonjour한 70",
                "60 monde",
            ]
        )

        with patch("get_freq.requests.get", return_value=_FakeResponse(response_text)):
            rows, contaminated_count = get_freq.fetch_wordlist("fr", cutoff=10, min_freq=5)

        self.assertEqual(contaminated_count, 1)
        self.assertEqual([row["word"] for row in rows], ["bonjour", "salut", "merci", "monde"])
        self.assertEqual([row["freq"] for row in rows], [100, 90, 80, 60])

    def test_fetch_wordlist_falls_back_to_full_list_when_50k_missing(self) -> None:
        response_404 = _FakeResponse("", status_code=404)
        response_full = _FakeResponse("100 bonjour\n90 salut\n")

        with patch("get_freq.requests.get", side_effect=[response_404, response_full]):
            rows, contaminated_count = get_freq.fetch_wordlist("fr", cutoff=10, min_freq=5)

        self.assertEqual(contaminated_count, 0)
        self.assertEqual([row["word"] for row in rows], ["bonjour", "salut"])
        self.assertEqual([row["freq"] for row in rows], [100, 90])

    def test_fetch_wordlist_skips_non_words(self) -> None:
        response_text = "\n".join(
            [
                "100 bonjour",
                "90 !!!",
                "80 12345",
                "70 monde",
            ]
        )

        with patch("get_freq.requests.get", return_value=_FakeResponse(response_text)):
            rows, contaminated_count = get_freq.fetch_wordlist("fr", cutoff=10, min_freq=5)

        self.assertEqual(contaminated_count, 0)
        self.assertEqual([row["word"] for row in rows], ["bonjour", "monde"])
        self.assertEqual([row["freq"] for row in rows], [100, 70])

    def test_should_keep_word_drops_minor_latin_gap_overlaps(self) -> None:
        df = pd.DataFrame(
            [
                {"word": "bonjour", "lang": "fr", "freq": 120, "rank": 1, "overlaps": "es", "is_overlap": True},
                {"word": "salut", "lang": "fr", "freq": 30, "rank": 2, "overlaps": "", "is_overlap": False},
                {"word": "hello", "lang": "en", "freq": 90, "rank": 1, "overlaps": "fr", "is_overlap": True},
                {"word": "xin", "lang": "vi", "freq": 80, "rank": 1, "overlaps": "en,fr", "is_overlap": True},
                {"word": "mot", "lang": "vi", "freq": 40, "rank": 2, "overlaps": "", "is_overlap": False},
            ]
        )

        normalized = get_freq._normalize_word_dict(df)
        kept = normalized[normalized.apply(get_freq._should_keep_word, axis=1)].copy()

        self.assertIn("fr", set(kept["lang"].tolist()))
        self.assertIn("en", set(kept["lang"].tolist()))
        self.assertIn("vi", set(kept["lang"].tolist()))
        self.assertNotIn("xin", set(kept.loc[kept["lang"] == "vi", "word"].tolist()))

    def test_finalize_example_marks_split_wordpieces_with_bio_labels(self) -> None:
        example = {
            "tokens": ["دەق"],
            "ner_tags": [get_freq.LABEL2ID["B-CKB"]],
            "original_text": "دەق",
        }

        finalized = get_freq._finalize_example(example, _SplitTokenizer())

        self.assertEqual(
            finalized["labels"],
            [
                -100,
                get_freq.LABEL2ID["B-CKB"],
                get_freq.LABEL2ID["I-CKB"],
                get_freq.LABEL2ID["I-CKB"],
                -100,
            ],
        )
        self.assertEqual(finalized["input_ids"], [101, 2001, 2002, 2003, 102])

    def test_relative_rank_is_per_language_not_absolute_frequency(self) -> None:
        df = pd.DataFrame(
            [
                {"word": "alpha", "lang": "en", "freq": 10_000, "rank": 1, "overlaps": "", "is_overlap": False},
                {"word": "beta", "lang": "en", "freq": 100, "rank": 2, "overlaps": "", "is_overlap": False},
                {"word": "uno", "lang": "es", "freq": 50, "rank": 1, "overlaps": "", "is_overlap": False},
                {"word": "dos", "lang": "es", "freq": 5, "rank": 2, "overlaps": "", "is_overlap": False},
            ]
        )

        normalized = get_freq._normalize_word_dict(df)
        weights = {
            (row["lang"], row["word"]): row["relative_rank"]
            for _, row in normalized.iterrows()
        }

        self.assertEqual(weights[("en", "alpha")], 1.0)
        self.assertEqual(weights[("es", "uno")], 1.0)
        self.assertEqual(weights[("en", "beta")], 0.0)
        self.assertEqual(weights[("es", "dos")], 0.0)

        self.assertEqual(
            get_freq._row_weight(
                pd.Series(
                    {
                        "word": "alpha",
                        "lang": "en",
                        "freq": 10_000,
                        "relative_rank": 0.75,
                        "overlap_langs": set(),
                    }
                )
            ),
            get_freq._row_weight(
                pd.Series(
                    {
                        "word": "alpha",
                        "lang": "en",
                        "freq": 10,
                        "relative_rank": 0.75,
                        "overlap_langs": set(),
                    }
                )
            ),
        )

    def test_repeat_count_uses_relative_rank(self) -> None:
        self.assertEqual(get_freq._repeat_count(0.95, 0), 3)
        self.assertEqual(get_freq._repeat_count(0.60, 0), 2)
        self.assertEqual(get_freq._repeat_count(0.10, 0), 1)
        self.assertEqual(get_freq._repeat_count(0.10, 2), 2)

    def test_write_source_pool_dir_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "freq_word_pools"
            source_df = pd.DataFrame(
                [
                    {
                        "word": "bonjour",
                        "lang": "fr",
                        "freq": 100,
                        "rank": 1,
                        "overlaps": "",
                        "is_overlap": False,
                    },
                    {
                        "word": "hello",
                        "lang": "en",
                        "freq": 90,
                        "rank": 2,
                        "overlaps": "",
                        "is_overlap": False,
                    }
                ]
            )

            language_frames, manifest = get_freq.build_short_text_source_pools(source_df, seed=11)
            get_freq._write_source_pool_dir(path, language_frames, manifest)

            self.assertTrue((path / "fr.parquet").exists())
            self.assertTrue((path / "en.parquet").exists())
            self.assertTrue((path / "manifest.json").exists())
            fr_frame = pd.read_parquet(path / "fr.parquet")
            en_frame = pd.read_parquet(path / "en.parquet")
            self.assertIn("sentence", fr_frame.columns)
            self.assertIn("sentence", en_frame.columns)
            self.assertEqual(fr_frame.iloc[0]["sentence"], "bonjour")
            self.assertEqual(en_frame.iloc[0]["sentence"], "hello")
            self.assertEqual(manifest["cache_version"], get_freq.SOURCE_POOL_CACHE_VERSION)


if __name__ == "__main__":
    unittest.main()
