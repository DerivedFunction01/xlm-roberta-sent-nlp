from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from tqdm.auto import tqdm

from paths import PATHS


def compact_multilabel_dataset(input_dir: str | Path, output_dir: str | Path) -> None:
    """Remove unused columns from the multilabel cache and save a compact copy."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_path}")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(input_path))
    compacted_splits = {}
    for split_name in tqdm(dataset.keys(), desc="Compacting splits"):
        split = dataset[split_name]
        removable_columns = [col for col in ("token_type_ids",) if col in split.column_names]
        if removable_columns:
            split = split.remove_columns(removable_columns)
        compacted_splits[split_name] = split

    compacted = DatasetDict(compacted_splits)
    compacted.save_to_disk(str(output_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact multilabel dataset cache without unused columns."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=PATHS["multilabel_dataset"]["cache_dir"],
        help="Path to the existing multilabel dataset cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=Path(PATHS["multilabel_dataset"]["cache_dir"]).with_name("multilabel_dataset_compact"),
        help="Directory where the compact cache will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compact_multilabel_dataset(args.input_dir, args.output_dir)
    print(f"Wrote compact multilabel dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
