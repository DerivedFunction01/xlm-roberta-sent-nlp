from __future__ import annotations

import glob
import json
import os
from typing import Any

from datasets import Dataset, load_dataset
from paths import PATHS
import pyarrow as pa




def _write_json_atomic(path: str, payload: dict) -> None:
    """Write JSON through a temporary file so updates are atomic-ish."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _synthetic_worker_temp_path(synthetic_temp_dir: str, worker_idx: int) -> str:
    """Return the temp parquet path for a synthetic-doc worker."""
    return os.path.join(synthetic_temp_dir, f"worker_{worker_idx}.parquet")


def _synthetic_shard_path(synthetic_cache_dir: str, worker_idx: int) -> str:
    """Return the final parquet shard path for a synthetic-doc worker."""
    return os.path.join(synthetic_cache_dir, f"part-{worker_idx:05d}.parquet")


def _synthetic_rows_to_table(rows: list[dict[str, Any]]):
    """Convert a list of row dicts into an Arrow table."""
    if pa is None:
        raise RuntimeError("pyarrow is required for synthetic parquet shards")
    schema = pa.schema(
        [
            ("kind", pa.string()),
            ("original_text", pa.string()),
            ("tokens", pa.string()),
            ("ner_tags", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _append_synthetic_rows(writer, rows: list[dict[str, Any]]) -> None:
    """Append a small batch of synthetic rows to an open Parquet writer."""
    if not rows:
        return
    writer.write_table(_synthetic_rows_to_table(rows))


def _synthetic_example_to_row(kind: str, example: dict[str, Any]) -> dict[str, Any]:
    """Convert one generated example into a parquet row."""
    return {
        "kind": kind,
        "original_text": example.get("original_text", ""),
        "tokens": json.dumps(example["tokens"]),
        "ner_tags": json.dumps(example["ner_tags"]),
    }


def _synthetic_row_to_example(row) -> dict[str, Any]:
    """Convert a parquet row back into an in-memory example dict."""
    if isinstance(row, dict):
        original_text = row.get("original_text", "")
        tokens = row.get("tokens", "[]")
        ner_tags = row.get("ner_tags", "[]")
    else:
        original_text = getattr(row, "original_text", "")
        tokens = getattr(row, "tokens", "[]")
        ner_tags = getattr(row, "ner_tags", "[]")
    example = {
        "original_text": original_text,
        "tokens": json.loads(tokens) if isinstance(tokens, str) else list(tokens),
        "ner_tags": json.loads(ner_tags) if isinstance(ner_tags, str) else list(ner_tags),
    }
    if not example["original_text"]:
        example["original_text"] = " ".join(example["tokens"])
    return example


def _move_synthetic_shard(
    temp_path: str,
    worker_idx: int,
    *,
    synthetic_cache_dir: str,
) -> str:
    """Move a worker temp shard into the final synthetic cache directory."""
    final_path = _synthetic_shard_path(synthetic_cache_dir, worker_idx)
    os.replace(temp_path, final_path)
    meta_temp = temp_path.replace(".parquet", ".meta.json")
    meta_final = final_path.replace(".parquet", ".meta.json")
    if os.path.exists(meta_temp):
        os.replace(meta_temp, meta_final)
    return final_path


def _clear_synthetic_cache_dir(synthetic_cache_dir: str) -> None:
    """Remove old synthetic shard files before rebuilding the cache."""
    if not os.path.exists(synthetic_cache_dir):
        os.makedirs(synthetic_cache_dir, exist_ok=True)
        return
    for path in glob.glob(os.path.join(synthetic_cache_dir, "*.parquet")):
        os.remove(path)
    for path in glob.glob(os.path.join(synthetic_cache_dir, "*.meta.json")):
        os.remove(path)


def _synthetic_cache_shards(synthetic_cache_dir: str) -> list[str]:
    """Return all parquet shards currently stored for synthetic examples."""
    return sorted(glob.glob(os.path.join(synthetic_cache_dir, "*.parquet")))


def _load_synthetic_examples_dataset(synthetic_cache_dir: str):
    """Load the synthetic cache as a Hugging Face dataset backed by parquet shards."""
    shard_paths = _synthetic_cache_shards(synthetic_cache_dir)
    if not shard_paths:
        return None
    try:
        return load_dataset("parquet", data_files=shard_paths, split="train")
    except Exception:
        return None
