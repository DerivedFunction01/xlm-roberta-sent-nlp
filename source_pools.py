from __future__ import annotations

import random
from collections import deque
from typing import Any


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
