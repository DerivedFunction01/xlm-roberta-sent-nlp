from __future__ import annotations

import unittest
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

    def test_build_short_text_dataset_drops_minor_latin_major_overlaps(self) -> None:
        df = pd.DataFrame(
            [
                {"word": "bonjour", "lang": "fr", "freq": 120, "rank": 1, "overlaps": "es", "is_overlap": True},
                {"word": "salut", "lang": "fr", "freq": 30, "rank": 2, "overlaps": "", "is_overlap": False},
                {"word": "hello", "lang": "en", "freq": 90, "rank": 1, "overlaps": "fr", "is_overlap": True},
                {"word": "xin", "lang": "vi", "freq": 80, "rank": 1, "overlaps": "en,fr", "is_overlap": True},
                {"word": "mot", "lang": "vi", "freq": 40, "rank": 2, "overlaps": "", "is_overlap": False},
            ]
        )

        train_df, test_df, manifest = get_freq.build_short_text_dataset(
            df,
            seed=7,
            train_fraction=0.8,
        )

        combined = pd.concat([train_df, test_df], ignore_index=True)
        langs = set(combined["lang"].tolist())

        self.assertIn("fr", langs)
        self.assertIn("en", langs)
        self.assertIn("vi", langs)
        self.assertNotIn("vi", set(train_df.loc[train_df["word"] == "xin", "lang"].tolist()))
        self.assertNotIn("vi", set(test_df.loc[test_df["word"] == "xin", "lang"].tolist()))

        fr_rows = combined[combined["lang"] == "fr"]
        self.assertTrue({"unigram", "bigram"}.issubset(set(fr_rows["source_type"].tolist())))
        for _, row in combined.iterrows():
            self.assertEqual(len(row["tokens"]), len(row["ner_tags"]))
            self.assertTrue(all(tag == row["ner_tags"][0] or tag == row["ner_tags"][-1] for tag in row["ner_tags"]))

        self.assertEqual(manifest["seed"], 7)
        self.assertIn("fr", manifest["counts"])

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


if __name__ == "__main__":
    unittest.main()
