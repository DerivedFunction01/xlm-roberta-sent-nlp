from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm.auto import tqdm

from language import ALL_LANGS, LANG_TO_GROUP
from paths import PATHS


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _chunk_ranges(length: int, num_chunks: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    num_chunks = max(1, min(num_chunks, length))
    chunk_size = (length + num_chunks - 1) // num_chunks
    return [(start, min(length, start + chunk_size)) for start in range(0, length, chunk_size)]


def _load_tokenizer(tokenizer_name_or_path: str | None):
    if not tokenizer_name_or_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name_or_path)


def _infer_vocab_size(dataset_dir: str) -> int:
    dataset = load_from_disk(dataset_dir)
    max_token_id = 0
    for split_name, split in dataset.items():
        for start in tqdm(range(0, len(split), 8192), desc=f"scan {split_name}"):
            batch = split[start : start + 8192]
            for ids in batch["input_ids"]:
                if ids:
                    row_max = max(ids)
                    if row_max > max_token_id:
                        max_token_id = row_max
    return max_token_id + 1


def _count_pair_chunk(
    *,
    dataset_dir: str,
    split_name: str,
    start: int,
    stop: int,
    batch_size: int,
    vocab_size: int,
) -> dict[int, int]:
    """Count (label_id, token_id) pairs in one worker slice."""
    dataset = load_from_disk(dataset_dir)
    split = dataset[split_name]

    counts: Counter[int] = Counter()
    for row_start in range(start, stop, batch_size):
        row_stop = min(stop, row_start + batch_size)
        batch = split[row_start:row_stop]
        for input_ids, attention_mask, labels in zip(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        ):
            for token_id, mask, label_id in zip(input_ids, attention_mask, labels):
                if mask and label_id > 0:
                    counts[int(label_id) * int(vocab_size) + int(token_id)] += 1

    return dict(counts)


def _count_split_pairs(
    *,
    dataset_dir: str,
    split_name: str,
    batch_size: int,
    vocab_size: int,
    workers: int,
) -> Counter[int]:
    dataset = load_from_disk(dataset_dir)
    split = dataset[split_name]
    ranges = _chunk_ranges(len(split), workers)
    if not ranges:
        return Counter()

    aggregate: Counter[int] = Counter()
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(
                _count_pair_chunk,
                dataset_dir=dataset_dir,
                split_name=split_name,
                start=start,
                stop=stop,
                batch_size=batch_size,
                vocab_size=vocab_size,
            )
            for start, stop in ranges
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name} workers"):
            aggregate.update(future.result())
    return aggregate


def _label_id_to_lang(label_id: int) -> str:
    if label_id <= 0:
        return "unknown"
    idx = (label_id - 1) // 2
    return ALL_LANGS[idx] if 0 <= idx < len(ALL_LANGS) else "unknown"


def _pair_counts_to_rows(
    pair_counts: Counter[int],
    *,
    split_name: str,
    vocab_size: int,
    token_to_id: dict[str, int] | None,
    id_to_token: dict[int, str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for composite, count in pair_counts.items():
        label_id = composite // vocab_size
        token_id = composite % vocab_size
        lang = _label_id_to_lang(label_id)
        row: dict[str, Any] = {
            "split": split_name,
            "lang": lang,
            "group": LANG_TO_GROUP.get(lang, "Unknown"),
            "label_id": int(label_id),
            "token_id": int(token_id),
            "count": int(count),
        }
        if id_to_token is not None:
            row["token"] = id_to_token.get(int(token_id), "")
        rows.append(row)
    rows.sort(key=lambda row: (row["split"], row["lang"], row["token_id"]))
    return rows


def _lang_rows_to_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    token_lookup: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (row["split"], row["group"], int(row["token_id"]))
        group_counts[key] += int(row["count"])
        if "token" in row and row["token"]:
            token_lookup[(row["split"], int(row["token_id"]))] = row["token"]

    group_rows: list[dict[str, Any]] = []
    for (split_name, group, token_id), count in group_counts.items():
        row: dict[str, Any] = {
            "split": split_name,
            "group": group,
            "token_id": token_id,
            "count": int(count),
        }
        token = token_lookup.get((split_name, token_id))
        if token is not None:
            row["token"] = token
        group_rows.append(row)

    group_rows.sort(key=lambda row: (row["split"], row["group"], row["token_id"]))
    return group_rows


def _lang_rows_to_overall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overall_counts: dict[tuple[str, int], int] = defaultdict(int)
    token_lookup: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (row["split"], int(row["token_id"]))
        overall_counts[key] += int(row["count"])
        if "token" in row and row["token"]:
            token_lookup[key] = row["token"]

    overall_rows: list[dict[str, Any]] = []
    for (split_name, token_id), count in overall_counts.items():
        row: dict[str, Any] = {
            "split": split_name,
            "token_id": token_id,
            "count": int(count),
        }
        token = token_lookup.get((split_name, token_id))
        if token is not None:
            row["token"] = token
        overall_rows.append(row)

    overall_rows.sort(key=lambda row: (row["split"], row["token_id"]))
    return overall_rows


def _write_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count tokenizer usage grouped by language and language bucket. "
            "Outputs long-format parquet files for later analysis."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=PATHS["tokenized"]["cache_dir"],
        help="Path to the tokenized dataset cache.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=PATHS["source_pools"]["cache_meta"],
        help="Path to the source pool manifest for language metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="token_vocab_reports",
        help="Directory where parquet outputs will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of rows to read per inner batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes per split.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer name or local path for token strings.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="How many rows to print in the terminal summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ = _load_json(args.meta)
    dataset = load_from_disk(str(args.dataset_dir))
    tokenizer = _load_tokenizer(args.tokenizer)
    id_to_token: dict[int, str] | None = None
    vocab_size: int

    if tokenizer is not None:
        vocab = tokenizer.get_vocab()
        id_to_token = {token_id: token for token, token_id in vocab.items()}
        vocab_size = max(id_to_token) + 1 if id_to_token else tokenizer.vocab_size or 0
    else:
        vocab_size = _infer_vocab_size(str(args.dataset_dir))

    if vocab_size <= 0:
        raise RuntimeError("Could not determine vocabulary size.")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name in dataset.keys():
        pair_counts = _count_split_pairs(
            dataset_dir=args.dataset_dir,
            split_name=split_name,
            batch_size=args.batch_size,
            vocab_size=vocab_size,
            workers=args.workers,
        )
        split_rows[split_name] = _pair_counts_to_rows(
            pair_counts,
            split_name=split_name,
            vocab_size=vocab_size,
            token_to_id=None,
            id_to_token=id_to_token,
        )

    lang_rows = [row for rows in split_rows.values() for row in rows]
    group_rows = _lang_rows_to_group_rows(lang_rows)
    overall_rows = _lang_rows_to_overall_rows(lang_rows)

    lang_path = output_dir / "token_counts_by_lang.parquet"
    group_path = output_dir / "token_counts_by_group.parquet"
    overall_path = output_dir / "token_counts_overall.parquet"

    _write_parquet(lang_path, lang_rows)
    _write_parquet(group_path, group_rows)
    _write_parquet(overall_path, overall_rows)

    lang_total = len(lang_rows)
    group_total = len(group_rows)
    overall_total = len(overall_rows)

    print(f"Wrote {lang_path}")
    print(f"Wrote {group_path}")
    print(f"Wrote {overall_path}")
    print(f"Rows by lang: {lang_total:,}")
    print(f"Rows by group: {group_total:,}")
    print(f"Rows overall: {overall_total:,}")

    print("\nTop token/lang rows:")
    for row in sorted(lang_rows, key=lambda r: (-int(r["count"]), r["lang"], int(r["token_id"])))[: args.top_k]:
        token_repr = repr(row.get("token", ""))
        print(
            f"  {row['lang']:<5} token_id={row['token_id']:>6} "
            f"count={row['count']:>10,} token={token_repr}"
        )

    print("\nTop bucket rows:")
    for row in sorted(group_rows, key=lambda r: (-int(r["count"]), r["group"], int(r["token_id"])))[: args.top_k]:
        token_repr = repr(row.get("token", ""))
        print(
            f"  {row['group']:<18} token_id={row['token_id']:>6} "
            f"count={row['count']:>10,} token={token_repr}"
        )


if __name__ == "__main__":
    main()
