from __future__ import annotations

import os
import json

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


def write_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def write_sentence_parquet(path: str, sentences: list[str]) -> None:
    if not sentences:
        pd.DataFrame({"sentence": []}).to_parquet(path, index=False)
        return
    if pa is None or pq is None:
        pd.DataFrame({"sentence": sentences}).to_parquet(path, index=False)
        return
    table = pa.table({"sentence": pa.array(sentences, type=pa.string())})
    pq.write_table(table, path)


def write_records_parquet(path: str, records: list[dict], columns: list[str] | None = None) -> None:
    if columns is None:
        frame = pd.DataFrame.from_records(records)
    else:
        frame = pd.DataFrame.from_records(records, columns=columns)
    tmp_path = f"{path}.tmp"
    frame.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
