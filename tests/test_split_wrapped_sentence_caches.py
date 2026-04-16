from __future__ import annotations

import unittest

from split_wrapped_sentence_caches import expand_wrapped_sentence_fragments


class DummySegmenter:
    def segment(self, text: str) -> list[str]:
        return [piece.strip() for piece in text.split("|") if piece.strip()]


class WrappedSentenceSplitTests(unittest.TestCase):
    def test_expands_quote_wrapped_multi_sentence_text(self) -> None:
        text = '"Je vais tres bien aujourd\'hui. Merci beaucoup pour votre aide."'

        self.assertEqual(
            expand_wrapped_sentence_fragments(text),
            [
                "Je vais tres bien aujourd'hui.",
                "Merci beaucoup pour votre aide.",
            ],
        )

    def test_strips_leading_quote_noise_and_splits_sentence_bundle(self) -> None:
        text = (
            '" "Geese, gulls, and other large fowls were shot with arrows that had long, '
            'five-sided heads of walrus ivory, not very sharp and barbed on one edge."\n\n'
            '"According to . our constitution, prostitution is prohibited. Therefore, Prime '
            'Minister Sheikh Hasina has decided to sanction two crores takas to rehabilitate '
            'them and offer them a new dawn," Osman is quoted as saying.'
        )

        self.assertEqual(
            expand_wrapped_sentence_fragments(text),
            [
                'Geese, gulls, and other large fowls were shot with arrows that had long, five-sided heads of walrus ivory, not very sharp and barbed on one edge.',
                'According to . our constitution, prostitution is prohibited.',
                'Therefore, Prime Minister Sheikh Hasina has decided to sanction two crores takas to rehabilitate them and offer them a new dawn," Osman is quoted as saying.',
            ],
        )

    def test_preserves_embedded_titlecase_quote(self) -> None:
        text = '"Aichi Biodiversity Target 2" addresses the underlying causes of biodiversity loss.'

        self.assertEqual(
            expand_wrapped_sentence_fragments(text),
            ['Aichi Biodiversity Target 2" addresses the underlying causes of biodiversity loss.'],
        )

    def test_expands_long_list_like_text(self) -> None:
        text = (
            "1: This is item one with enough words to trigger the list splitter and keep it stable "
            "2: This is item two with enough words to trigger the list splitter and keep it stable"
        )

        self.assertEqual(
            expand_wrapped_sentence_fragments(text),
            [
                "1: This is item one with enough words to trigger the list splitter and keep it stable",
                "2: This is item two with enough words to trigger the list splitter and keep it stable",
            ],
        )

    def test_uses_provided_segmenter_when_available(self) -> None:
        text = '"One|Two|Three"'

        self.assertEqual(
            expand_wrapped_sentence_fragments(text, lang="fr", segmenter=DummySegmenter()),
            ["One", "Two", "Three"],
        )


if __name__ == "__main__":
    unittest.main()
