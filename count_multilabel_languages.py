# %%
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_from_disk
from tqdm.auto import tqdm

from language import ALL_LANGS
from paths import PATHS


def _chunk_ranges(length: int, num_chunks: int) -> list[tuple[int, int]]:
    """Split a length into contiguous half-open ranges."""
    if length <= 0:
        return []
    num_chunks = max(1, min(num_chunks, length))
    chunk_size = math.ceil(length / num_chunks)
    ranges: list[tuple[int, int]] = []
    for start in range(0, length, chunk_size):
        stop = min(length, start + chunk_size)
        ranges.append((start, stop))
    return ranges


def _count_labels_in_range(
    *,
    dataset_dir: str,
    split_name: str,
    start: int,
    stop: int,
    chunk_size: int,
    position: int,
) -> np.ndarray:
    """Count per-language labels for one contiguous shard of a split."""
    dataset = load_from_disk(dataset_dir)
    split = dataset[split_name]

    counts = np.zeros(len(ALL_LANGS), dtype=np.int64)
    with tqdm(
        total=stop - start,
        desc=f"{split_name} {start}:{stop}",
        position=position,
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for row_start in range(start, stop, chunk_size):
            row_stop = min(stop, row_start + chunk_size)
            batch = split[row_start:row_stop]["labels"]
            counts += np.asarray(batch, dtype=np.int64).sum(axis=0)
            pbar.update(row_stop - row_start)
    return counts


def count_split_parallel(
    *,
    dataset_dir: str,
    split_name: str,
    num_workers: int,
    chunk_size: int,
) -> np.ndarray:
    """Count labels for a split using multiple workers and local tqdm bars."""
    dataset = load_from_disk(dataset_dir)
    split = dataset[split_name]
    ranges = _chunk_ranges(len(split), num_workers)
    if not ranges:
        return np.zeros(len(ALL_LANGS), dtype=np.int64)

    totals = np.zeros(len(ALL_LANGS), dtype=np.int64)
    with ProcessPoolExecutor(max_workers=len(ranges)) as pool:
        futures = {
            pool.submit(
                _count_labels_in_range,
                dataset_dir=dataset_dir,
                split_name=split_name,
                start=start,
                stop=stop,
                chunk_size=chunk_size,
                position=idx,
            ): idx
            for idx, (start, stop) in enumerate(ranges)
        }
        for future in as_completed(futures):
            totals += future.result()
    return totals


def count_dataset_parallel(
    *,
    dataset_dir: str,
    num_workers: int = 8,
    chunk_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Count language presence across all splits in the multilabel dataset."""
    dataset = load_from_disk(dataset_dir)
    return {
        split_name: count_split_parallel(
            dataset_dir=dataset_dir,
            split_name=split_name,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )
        for split_name in dataset.keys()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count how many examples contain each language in the multilabel dataset. "
            "This counts example-level language presence, not token frequency."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=PATHS["multilabel_dataset"]["cache_dir"],
        help="Path to the saved multilabel dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count() // 2,
        help="Number of worker processes to use per split.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="How many examples each worker reads per batch.",
    )
    return parser.parse_args()


def _print_counts(title: str, counts: Iterable[int]) -> None:
    print(title)
    for lang, count in sorted(zip(ALL_LANGS, counts), key=lambda item: (-item[1], item[0])):
        if count:
            print(f"{lang}\t{int(count)}")
    print()


def main() -> None:
    args = _parse_args()
    dataset_dir = str(Path(args.dataset_dir))
    counts_by_split = count_dataset_parallel(
        dataset_dir=dataset_dir,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
    )

    total = np.zeros(len(ALL_LANGS), dtype=np.int64)
    for split_name, counts in counts_by_split.items():
        _print_counts(split_name, counts)
        total += counts

    _print_counts("all_splits", total)


if __name__ == "__main__":
    main()
