from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm.auto import tqdm

from language import ALL_LANGS
from multilabel_converter import build_label_maps
from paths import PATHS

LABEL2ID, _ = build_label_maps(ALL_LANGS)
B_LABEL_IDS = np.array(
    [label_id for label, label_id in LABEL2ID.items() if label.startswith("B-")],
    dtype=np.int64,
)
I_LABEL_IDS = np.array(
    [label_id for label, label_id in LABEL2ID.items() if label.startswith("I-")],
    dtype=np.int64,
)
B_LABEL_ID_SET = set(int(label_id) for label_id in B_LABEL_IDS.tolist())
B_AND_I_LABEL_ID_SET = B_LABEL_ID_SET | set(int(label_id) for label_id in I_LABEL_IDS.tolist())


@dataclass
class SplitCounts:
    """Per-language counts for one split."""

    sentence_counts: np.ndarray
    token_counts: np.ndarray


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
) -> SplitCounts:
    """Count sentence starts and tokens for one contiguous shard of a split."""
    dataset = load_from_disk(dataset_dir)
    split = dataset[split_name]

    sentence_counts = np.zeros(len(ALL_LANGS), dtype=np.int64)
    token_counts = np.zeros(len(ALL_LANGS), dtype=np.int64)

    with tqdm(
        total=stop - start,
        desc=f"{split_name} {start}:{stop}",
        position=position,
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for row_start in range(start, stop, chunk_size):
            row_stop = min(stop, row_start + chunk_size)
            batch_labels = split[row_start:row_stop]["labels"]

            for labels in batch_labels:
                for label_id in labels:
                    if label_id == -100:
                        continue

                    label_id = int(label_id)
                    if label_id in B_AND_I_LABEL_ID_SET:
                        lang_idx = (label_id - 1) // 2
                        token_counts[lang_idx] += 1
                        if label_id in B_LABEL_ID_SET:
                            sentence_counts[lang_idx] += 1

            pbar.update(row_stop - row_start)

    return SplitCounts(sentence_counts=sentence_counts, token_counts=token_counts)


def count_split_parallel(
    *,
    dataset_dir: str,
    split_name: str,
    num_workers: int,
    chunk_size: int,
) -> SplitCounts:
    """Count labels for a split using multiple workers and local tqdm bars."""
    dataset = load_from_disk(dataset_dir)
    split = dataset[split_name]
    ranges = _chunk_ranges(len(split), num_workers)
    if not ranges:
        zeros = np.zeros(len(ALL_LANGS), dtype=np.int64)
        return SplitCounts(sentence_counts=zeros.copy(), token_counts=zeros)

    sentence_totals = np.zeros(len(ALL_LANGS), dtype=np.int64)
    token_totals = np.zeros(len(ALL_LANGS), dtype=np.int64)

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
            result = future.result()
            sentence_totals += result.sentence_counts
            token_totals += result.token_counts

    return SplitCounts(sentence_counts=sentence_totals, token_counts=token_totals)


def count_dataset_parallel(
    *,
    dataset_dir: str,
    num_workers: int = 8,
    chunk_size: int = 100_000,
) -> dict[str, SplitCounts]:
    """Count language sentence starts and tokens across all splits in the tokenized dataset."""
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
            "Count language sentence starts and token coverage in the tokenized NER dataset. "
            "Sentence counts are based on B-* tags; token counts include B-* and I-* tags."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=PATHS["tokenized"]["cache_dir"],
        help="Path to the saved tokenized NER dataset.",
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


def _print_markdown_table(counts_by_split: dict[str, SplitCounts]) -> None:
    """Print a markdown table with per-split counts and percentages, sorted by total sentences."""
    split_names = list(counts_by_split.keys())
    split_sentence_totals = {name: counts_by_split[name].sentence_counts.sum() for name in split_names}
    split_token_totals = {name: counts_by_split[name].token_counts.sum() for name in split_names}
    grand_sentence_total = sum(split_sentence_totals.values())
    grand_token_total = sum(split_token_totals.values())

    header_cols = ["lang"]
    for name in split_names:
        header_cols += [f"{name} sentences", f"{name} tokens"]
    header_cols += ["all sentences", "all tokens"]

    sep_cols = [":---"] + ["---:", "---:"] * (len(split_names) + 1)

    print("| " + " | ".join(header_cols) + " |")
    print("| " + " | ".join(sep_cols) + " |")

    rows = sorted(
        zip(
            ALL_LANGS,
            *[counts_by_split[name].sentence_counts for name in split_names],
            *[counts_by_split[name].token_counts for name in split_names],
        ),
        key=lambda row: -sum(int(row[1 + idx]) for idx in range(len(split_names))),
    )

    for row in rows:
        lang = row[0]
        sentence_counts = row[1 : 1 + len(split_names)]
        token_counts = row[1 + len(split_names) :]
        all_sentence_count = sum(int(count) for count in sentence_counts)
        all_token_count = sum(int(count) for count in token_counts)
        if all_sentence_count == 0 and all_token_count == 0:
            continue

        cells = [lang]
        for name, sentence_count, token_count in zip(split_names, sentence_counts, token_counts):
            sentence_pct = (
                int(sentence_count) / split_sentence_totals[name] * 100 if split_sentence_totals[name] else 0
            )
            token_pct = int(token_count) / split_token_totals[name] * 100 if split_token_totals[name] else 0
            cells += [f"{int(sentence_count)} ({sentence_pct:.2f}%)", f"{int(token_count)} ({token_pct:.2f}%)"]
        all_sentence_pct = all_sentence_count / grand_sentence_total * 100 if grand_sentence_total else 0
        all_token_pct = all_token_count / grand_token_total * 100 if grand_token_total else 0
        cells += [
            f"{all_sentence_count} ({all_sentence_pct:.2f}%)",
            f"{all_token_count} ({all_token_pct:.2f}%)",
        ]

        print("| " + " | ".join(cells) + " |")

    footer = ["**total**"]
    for name in split_names:
        footer += [f"{int(split_sentence_totals[name])} (100.00%)", f"{int(split_token_totals[name])} (100.00%)"]
    footer += [f"{grand_sentence_total} (100.00%)", f"{grand_token_total} (100.00%)"]
    print("| " + " | ".join(footer) + " |")


def main() -> None:
    args = _parse_args()
    dataset_dir = str(Path(args.dataset_dir))
    counts_by_split = count_dataset_parallel(
        dataset_dir=dataset_dir,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
    )

    _print_markdown_table(counts_by_split)


if __name__ == "__main__":
    main()
