#!/usr/bin/env python3
"""Convert Tatoeba's TSV sentence dump into per-language parquet shards.

The source file is expected to be tab-separated with rows like:

    id<TAB>lang_iso3<TAB>sentence_text

The output layout mirrors the per-language caches used by the wiki and
finetranslations loaders: one parquet file per model language, each with a
single `sentence` column.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from language import LANG_ISO2_TO_ISO3
from paths import PATHS


DEFAULT_LANGUAGE_REMAPS = {
    # Chinese varieties collapse into the zh bucket for this model.
    "cmn": "zh",
    "yue": "zh",
    "wuu": "zh",
    "nan": "zh",
    # Norwegian variants can safely be folded into the existing no bucket.
    "nob": "no",
    "nno": "no",
}

PARQUET_SCHEMA = pa.schema([("sentence", pa.string())])
ISO3_TO_ISO2 = {v: k for k, v in LANG_ISO2_TO_ISO3.items()}


def parse_remap(specs: list[str]) -> dict[str, str]:
    remaps: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid remap {spec!r}; expected old=new.")
        old, new = (part.strip() for part in spec.split("=", 1))
        if not old or not new:
            raise ValueError(f"Invalid remap {spec!r}; expected old=new.")
        remaps[old] = new
    return remaps


def normalize_lang(code: str, remaps: dict[str, str]) -> str | None:
    code = (code or "").strip()
    if not code:
        return None
    if code in remaps:
        return remaps[code]
    return ISO3_TO_ISO2.get(code)


def open_writer(path: Path) -> pq.ParquetWriter:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(str(path), PARQUET_SCHEMA)


def write_batch(writer: pq.ParquetWriter, sentences: list[str]) -> None:
    if not sentences:
        return
    table = pa.Table.from_arrays([pa.array(sentences, type=pa.string())], names=["sentence"])
    writer.write_table(table)


def convert_tatoeba_sentences(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    remaps: dict[str, str] = {},
    flush_rows: int = 25_000,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    combined_remaps = {**DEFAULT_LANGUAGE_REMAPS, **remaps}
    input_path = input_path or Path(PATHS["tatoeba"]["source_file"])
    output_dir = output_dir or Path(PATHS["tatoeba"]["cache_dir"])
    cache_meta = Path(PATHS["tatoeba"]["cache_meta"]) if output_dir == Path(PATHS["tatoeba"]["cache_dir"]) else output_dir / "tatoeba.meta.json"

    if not force_rebuild and cache_meta.exists():
        try:
            with cache_meta.open(encoding="utf-8") as f:
                cached_meta = json.load(f)
            parquet_files = sorted(output_dir.glob("*.parquet"))
            if parquet_files:
                return cached_meta
        except json.JSONDecodeError:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, pq.ParquetWriter] = {}
    buffers: defaultdict[str, list[str]] = defaultdict(list)
    counts: Counter[str] = Counter()

    total_rows = 0
    skipped_rows = 0
    flush_rows = max(1, flush_rows)

    try:
        with input_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in tqdm(reader, desc="Tatoeba sentences", unit="rows"):
                total_rows += 1
                if len(row) < 3:
                    skipped_rows += 1
                    continue

                lang_code = row[1].strip()
                sentence = row[2].strip()
                if not sentence:
                    skipped_rows += 1
                    continue

                lang = combined_remaps.get(lang_code) or ISO3_TO_ISO2.get(lang_code)
                if not lang:
                    skipped_rows += 1
                    continue

                buffers[lang].append(sentence)
                counts[lang] += 1

                if total_rows % flush_rows == 0:
                    for flush_lang, flush_sentences in list(buffers.items()):
                        if not flush_sentences:
                            continue
                        writer = writers.get(flush_lang)
                        if writer is None:
                            writer = open_writer(output_dir / f"{flush_lang}.parquet")
                            writers[flush_lang] = writer
                        write_batch(writer, flush_sentences)
                        buffers[flush_lang].clear()
    finally:
        for flush_lang, flush_sentences in list(buffers.items()):
            if not flush_sentences:
                continue
            writer = writers.get(flush_lang)
            if writer is None:
                writer = open_writer(output_dir / f"{flush_lang}.parquet")
                writers[flush_lang] = writer
            write_batch(writer, flush_sentences)
            buffers[flush_lang].clear()

        for writer in writers.values():
            writer.close()

    meta = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_rows": total_rows,
        "written_rows": sum(counts.values()),
        "skipped_rows": skipped_rows,
        "languages_written": dict(sorted(counts.items())),
        "default_remaps": DEFAULT_LANGUAGE_REMAPS,
        "extra_remaps": remaps,
    }
    cache_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(PATHS["tatoeba"]["source_file"]),
        help="Path to the Tatoeba TSV/CSV dump.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(PATHS["tatoeba"]["cache_dir"]),
        help="Directory where per-language parquet files will be written.",
    )
    parser.add_argument(
        "--remap",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Extra language remaps to apply before filtering. Example: yue=zh",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=25_000,
        help="How many input rows to process between parquet flushes.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the parquet cache even if a cached manifest already exists.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    remaps = parse_remap(args.remap)
    meta = convert_tatoeba_sentences(
        args.input_path,
        args.output_dir,
        remaps=remaps,
        flush_rows=args.flush_rows,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
