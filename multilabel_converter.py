from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from language import ALL_LANGS
from paths import PATHS


def build_label_maps(all_langs: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build NER label maps from the canonical language list."""
    label2id: dict[str, int] = {"O": 0}
    id2label: dict[int, str] = {0: "O"}
    for idx, lang in enumerate(all_langs):
        b_id = 2 * idx + 1
        i_id = 2 * idx + 2
        label2id[f"B-{lang.upper()}"] = b_id
        label2id[f"I-{lang.upper()}"] = i_id
        id2label[b_id] = f"B-{lang.upper()}"
        id2label[i_id] = f"I-{lang.upper()}"
    return label2id, id2label


def extract_example_languages(labels: list[int], id2label: dict[int, str]) -> list[str]:
    """Convert token-level NER labels into a sorted list of present languages."""
    languages: set[str] = set()
    for label_id in labels:
        if label_id <= 0:
            continue
        label = id2label.get(label_id)
        if label is None or not label.startswith(("B-", "I-")):
            continue
        languages.add(label[2:].lower())
    return [lang for lang in ALL_LANGS if lang in languages]


def example_to_multilabel(example: dict[str, Any], id2label: dict[int, str]) -> dict[str, Any]:
    """Map one example to the multilabel classification format."""
    text = example.get("original_text", "")
    if not text:
        tokens = example.get("tokens")
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError:
                tokens = tokens.split()
        text = " ".join(tokens or [])

    label_ids = example.get("labels") or example.get("ner_tags") or []
    if isinstance(label_ids, str):
        label_ids = json.loads(label_ids)

    language_labels = extract_example_languages(label_ids, id2label)
    multi_hot = [1 if lang in language_labels else 0 for lang in ALL_LANGS]

    return {
        "text": text,
        "language_labels": language_labels,
        "labels": multi_hot,
    }


def convert_tokenized_dataset(
    dataset: DatasetDict,
    *,
    id2label: dict[int, str],
    num_samples: int | None = None,
) -> DatasetDict:
    """Convert a token classification train/eval split into multilabel examples."""
    if num_samples is not None:
        dataset = DatasetDict(
            {
                split_name: split.select(range(min(num_samples, len(split))))
                for split_name, split in dataset.items()
            }
        )

    return DatasetDict(
        {
            split_name: split.map(
                lambda ex: example_to_multilabel(ex, id2label),
                batched=False,
                remove_columns=[col for col in split.column_names if col not in {"original_text", "tokens"}],
            )
            for split_name, split in dataset.items()
        }
    )


def load_tokenized_cache(path: Path) -> DatasetDict:
    """Load a tokenized NER dataset cache saved with Hugging Face datasets."""
    return load_from_disk(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a token-level NER dataset into a multilabel classification dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=PATHS["tokenized"]["cache_dir"],
        help="Path to the tokenized NER dataset cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=PATHS["multilabel_dataset"]["cache_dir"],
        help="Directory where the multilabel dataset will be saved.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional max number of examples per split for quick testing.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenized_dataset = load_tokenized_cache(input_dir)
    _, id2label = build_label_maps(ALL_LANGS)

    multilabel_dataset = convert_tokenized_dataset(
        tokenized_dataset,
        id2label=id2label,
        num_samples=args.num_samples,
    )

    multilabel_dataset.save_to_disk(str(output_dir))
    print(f"Saved multilabel dataset to {output_dir}")
    for split_name, split in multilabel_dataset.items():
        print(f"{split_name}: {len(split)} examples")


if __name__ == "__main__":
    main()
