from __future__ import annotations

import unittest
from unittest.mock import patch

import get_freq


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


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


if __name__ == "__main__":
    unittest.main()
