from __future__ import annotations

import hashlib
import json
import os
import random
from collections import deque
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from io_utils import write_json_atomic
import pyarrow as pa
import pyarrow.parquet as pq
from text_utils import normalize_sentence_for_pool



SOURCE_POOL_CACHE_VERSION = 2


def split_sentence_map_for_synthetic(
    sentence_map: dict[str, list[str]],
    *,
    reserve_fraction: float,
    min_reserved: int,
    max_reserved: int,
) -> tuple[dict[str, deque[str]], dict[str, deque[str]]]:
    """Split one source into reserved and main pools for synthetic sampling."""
    reserved: dict[str, deque[str]] = {}
    main: dict[str, deque[str]] = {}
    for lang, sentences in sentence_map.items():
        if not sentences:
            continue
        shuffled = sentences[:]
        random.shuffle(shuffled)
        reserve_target = int(round(len(shuffled) * reserve_fraction))
        reserve_n = min(len(shuffled), max(min_reserved, min(reserve_target, max_reserved)))
        reserved[lang] = deque(shuffled[:reserve_n])
        main[lang] = deque(shuffled[reserve_n:])
    return reserved, main


def merge_sentence_pools(
    base_reserved: dict[str, deque[str]],
    base_main: dict[str, deque[str]],
    extra_reserved: dict[str, deque[str]],
    extra_main: dict[str, deque[str]],
) -> tuple[dict[str, deque[str]], dict[str, deque[str]]]:
    """Merge one source's pools into the running synthetic pool set."""
    for lang, pool in extra_reserved.items():
        base_reserved.setdefault(lang, deque()).extend(pool)
    for lang, pool in extra_main.items():
        base_main.setdefault(lang, deque()).extend(pool)
    return base_reserved, base_main


def build_source_sentence_pools(
    source_specs: list[dict[str, Any]],
) -> tuple[dict[str, deque[str]], dict[str, deque[str]], list[dict[str, Any]]]:
    """Build merged reserved/main pools from multiple named sources."""
    reserved: dict[str, deque[str]] = {}
    main: dict[str, deque[str]] = {}
    summaries: list[dict[str, Any]] = []

    for spec in source_specs:
        name = str(spec.get("name", "source"))
        sentence_map = spec.get("sentence_map")
        if not sentence_map:
            summaries.append({"name": name, "skipped": True, "reserved": 0, "main": 0})
            continue

        source_reserved, source_main = split_sentence_map_for_synthetic(
            sentence_map,
            reserve_fraction=float(spec["reserve_fraction"]),
            min_reserved=int(spec["min_reserved"]),
            max_reserved=int(spec["max_reserved"]),
        )
        reserved, main = merge_sentence_pools(reserved, main, source_reserved, source_main)
        summaries.append(
            {
                "name": name,
                "skipped": False,
                "reserved": sum(len(pool) for pool in source_reserved.values()),
                "main": sum(len(pool) for pool in source_main.values()),
            }
        )

    return reserved, main, summaries


def draw_sentence(
    lang: str,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
) -> str | None:
    if primary_pool.get(lang):
        return primary_pool[lang].popleft()
    if fallback_pool and fallback_pool.get(lang):
        return fallback_pool[lang].popleft()
    return None


def remaining_sentence_count(
    lang: str,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
) -> int:
    total = len(primary_pool.get(lang, ()))
    if fallback_pool is not None:
        total += len(fallback_pool.get(lang, ()))
    return total


def chunk_list(items: list, n_chunks: int) -> list[list]:
    if n_chunks <= 1:
        return [items]
    chunk_size = (len(items) + n_chunks - 1) // n_chunks
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def partition_sentence_pools(
    pools: dict[str, deque[str]],
    n_workers: int,
) -> list[dict[str, deque[str]]]:
    worker_pools: list[dict[str, deque[str]]] = [dict() for _ in range(n_workers)]
    for lang, dq in pools.items():
        items = list(dq)
        for worker_idx in range(n_workers):
            shard = items[worker_idx::n_workers]
            if shard:
                worker_pools[worker_idx][lang] = deque(shard)
    return worker_pools


def _stable_uint64(*parts: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big", signed=False)


def _parquet_lang_paths(cache_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(cache_dir):
        return []
    paths: list[tuple[str, str]] = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".parquet"):
            continue
        lang = name[:-8]
        paths.append((lang, os.path.join(cache_dir, name)))
    return paths


def load_language_sentences_from_parquet(cache_dir: str, lang: str) -> list[str]:
    path = os.path.join(cache_dir, f"{lang}.parquet")
    if not os.path.exists(path):
        return []
    frame = pd.read_parquet(path)
    if "sentence" not in frame.columns:
        return []
    sentences = frame["sentence"].astype(str).tolist()
    return [sentence for sentence in (normalize_sentence_for_pool(sentence, lang=lang) for sentence in sentences) if sentence]


def load_worker_sentence_pool(path: str) -> dict[str, deque[str]]:
    if not path or not os.path.exists(path):
        return {}
    frame = pd.read_parquet(path)
    if frame.empty or "lang" not in frame.columns or "sentence" not in frame.columns:
        return {}
    pools: dict[str, deque[str]] = {}
    for lang, group in frame.groupby("lang", sort=False):
        pools[str(lang)] = deque(
            sentence
            for sentence in (
                normalize_sentence_for_pool(raw_sentence, lang=str(lang))
                for raw_sentence in group["sentence"].astype(str).tolist()
            )
            if sentence
        )
    return pools


def build_disk_sentence_pool_shards(
    source_specs: list[dict[str, Any]],
    *,
    n_workers: int,
    seed: int,
    pool_cache_dir: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    if pq is None or pa is None:
        raise RuntimeError("pyarrow is required to build disk-backed sentence pool shards")

    os.makedirs(pool_cache_dir, exist_ok=True)
    manifest_path = os.path.join(pool_cache_dir, "sentence_pools.meta.json")

    expected_sources = [
        {
            "name": str(spec.get("name", "source")),
            "cache_dir": str(spec["cache_dir"]),
            "reserve_fraction": float(spec["reserve_fraction"]),
            "min_reserved": int(spec["min_reserved"]),
            "max_reserved": int(spec["max_reserved"]),
        }
        for spec in source_specs
        if spec.get("cache_dir")
    ]

    if not force_rebuild and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            cached = json.load(f)
        if (
            isinstance(cached, dict)
            and cached.get("cache_version") == SOURCE_POOL_CACHE_VERSION
            and cached.get("seed") == seed
            and cached.get("n_workers") == n_workers
            and cached.get("sources") == expected_sources
            and all(os.path.exists(path) for path in cached.get("reserved_shards", []))
            and all(os.path.exists(path) for path in cached.get("main_shards", []))
        ):
            return cached

    reserved_paths = [os.path.join(pool_cache_dir, f"reserved_worker_{i:05d}.parquet") for i in range(n_workers)]
    main_paths = [os.path.join(pool_cache_dir, f"main_worker_{i:05d}.parquet") for i in range(n_workers)]
    temp_reserved_paths = [f"{path}.tmp" for path in reserved_paths]
    temp_main_paths = [f"{path}.tmp" for path in main_paths]

    schema = pa.schema([("lang", pa.string()), ("sentence", pa.string())])
    reserved_writers = [pq.ParquetWriter(path, schema) for path in temp_reserved_paths]
    main_writers = [pq.ParquetWriter(path, schema) for path in temp_main_paths]
    reserved_batches: list[list[dict[str, str]]] = [[] for _ in range(n_workers)]
    main_batches: list[list[dict[str, str]]] = [[] for _ in range(n_workers)]
    batch_size = 2048

    language_stats: dict[str, dict[str, int]] = {}
    source_summaries: list[dict[str, Any]] = []

    def _flush_batch(writer, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()

    try:
        for spec in tqdm(expected_sources, desc="Sentence pool sources"):
            cache_dir = spec["cache_dir"]
            source_name = spec["name"]
            reserve_fraction = float(spec["reserve_fraction"])
            min_reserved = int(spec["min_reserved"])
            max_reserved = int(spec["max_reserved"])
            source_reserved = 0
            source_main = 0

            for lang, path in tqdm(_parquet_lang_paths(cache_dir), desc=f"{source_name} langs", leave=False):
                frame = pd.read_parquet(path, columns=["sentence"])
                if frame.empty or "sentence" not in frame.columns:
                    continue

                total = int(len(frame))
                reserve_target = int(round(total * reserve_fraction))
                reserve_n = min(total, max(min_reserved, min(reserve_target, max_reserved)))
                language_reserved = 0
                language_main = 0

                sentences = [
                    sentence
                    for sentence in (
                        normalize_sentence_for_pool(raw_sentence, lang=lang, seed=seed)
                        for raw_sentence in frame["sentence"].astype(str).tolist()
                    )
                    if sentence
                ]
                for idx, sentence in enumerate(sentences):
                    worker_idx = _stable_uint64(str(seed), source_name, lang, sentence, "worker") % n_workers
                    reserve_key = _stable_uint64(str(seed), source_name, lang, sentence, "reserve")
                    is_reserved = reserve_key % total < reserve_n if total > 0 else False
                    row = {"lang": lang, "sentence": sentence}
                    if is_reserved:
                        reserved_batches[worker_idx].append(row)
                        language_reserved += 1
                        source_reserved += 1
                        if len(reserved_batches[worker_idx]) >= batch_size:
                            _flush_batch(reserved_writers[worker_idx], reserved_batches[worker_idx])
                    else:
                        main_batches[worker_idx].append(row)
                        language_main += 1
                        source_main += 1
                        if len(main_batches[worker_idx]) >= batch_size:
                            _flush_batch(main_writers[worker_idx], main_batches[worker_idx])

                current = language_stats.setdefault(lang, {"total": 0, "reserved": 0, "main": 0})
                current["total"] += total
                current["reserved"] += language_reserved
                current["main"] += language_main

            source_summaries.append(
                {
                    "name": source_name,
                    "cache_dir": cache_dir,
                    "reserve_fraction": reserve_fraction,
                    "min_reserved": min_reserved,
                    "max_reserved": max_reserved,
                    "reserved": source_reserved,
                    "main": source_main,
                }
            )

        for idx in range(n_workers):
            _flush_batch(reserved_writers[idx], reserved_batches[idx])
            _flush_batch(main_writers[idx], main_batches[idx])
    finally:
        for writer in reserved_writers:
            writer.close()
        for writer in main_writers:
            writer.close()

    for temp_path, final_path in zip(temp_reserved_paths, reserved_paths):
        os.replace(temp_path, final_path)
    for temp_path, final_path in zip(temp_main_paths, main_paths):
        os.replace(temp_path, final_path)

    manifest = {
        "cache_version": SOURCE_POOL_CACHE_VERSION,
        "seed": seed,
        "n_workers": n_workers,
        "sources": expected_sources,
        "reserved_shards": reserved_paths,
        "main_shards": main_paths,
        "language_stats": language_stats,
        "source_summaries": source_summaries,
        "total_reserved": sum(item["reserved"] for item in language_stats.values()),
        "total_main": sum(item["main"] for item in language_stats.values()),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
