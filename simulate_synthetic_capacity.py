from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from language import ALL_LANGS, LANG_TO_GROUP
from paths import PATHS
from source_config import DOC_MIX, LANGUAGE_BUCKETS, RUN


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _bucket_weight(lang: str) -> float:
    group = LANG_TO_GROUP.get(lang)
    if group is None:
        return 1.0
    return float(LANGUAGE_BUCKETS.get(group, {}).get("weight", 1.0))


def _language_group(lang: str) -> str:
    return LANG_TO_GROUP.get(lang, "Other")


def _build_language_doc_plan(
    language_stats: dict[str, dict[str, int]],
    *,
    source_key: str,
    target_docs: int,
    docs_per_sentence_estimate: int,
    seed: int,
) -> list[str]:
    """Mirror the synthetic planner used by the training pipeline."""
    capacities: dict[str, int] = {}
    for lang, stats in language_stats.items():
        available = int(stats.get(source_key, 0))
        if available <= 0:
            continue
        capacities[lang] = max(1, available // max(1, docs_per_sentence_estimate))

    if not capacities or target_docs <= 0:
        return []

    rng = random.Random(seed)
    plan: list[str] = []

    for lang in sorted(capacities):
        if len(plan) >= target_docs:
            break
        plan.append(lang)
        capacities[lang] -= 1
        if capacities[lang] <= 0:
            del capacities[lang]

    while len(plan) < target_docs and capacities:
        candidates = list(capacities)
        weights = [capacities[lang] for lang in candidates]
        lang = rng.choices(candidates, weights=weights, k=1)[0]
        plan.append(lang)
        capacities[lang] -= 1
        if capacities[lang] <= 0:
            del capacities[lang]

    rng.shuffle(plan)
    return plan


def _sample_sentence_count(rng: random.Random, phase: str) -> int:
    if phase == "pure":
        low = int(DOC_MIX["pure"]["min_sentences"])
        high = int(DOC_MIX["pure"]["max_sentences"])
    elif phase == "homogeneous":
        low = int(DOC_MIX["homogeneous"]["min_sentences"])
        high = int(DOC_MIX["homogeneous"]["max_sentences"])
    elif phase == "spliced":
        low = int(DOC_MIX["spliced"]["min_sentences"])
        high = int(DOC_MIX["spliced"]["max_sentences"])
    else:
        return 1
    return rng.randint(low, high)


def _simulate_single_run(
    *,
    language_stats: dict[str, dict[str, int]],
    target_docs: int,
    seed: int,
) -> dict[str, Any]:
    """Run a single approximate capacity simulation."""
    rng = random.Random(seed)
    pool_state = {
        "reserved": {lang: int(stats.get("reserved", 0)) for lang, stats in language_stats.items()},
        "main": {lang: int(stats.get("main", 0)) for lang, stats in language_stats.items()},
    }

    pure_target = int(round(target_docs * float(DOC_MIX["pure"]["fraction"])))
    homo_target = int(round(target_docs * float(DOC_MIX["homogeneous"]["fraction"])))
    spliced_target = int(round(target_docs * float(DOC_MIX["spliced"]["fraction"])))

    pure_plan = _build_language_doc_plan(
        language_stats,
        source_key="reserved",
        target_docs=pure_target,
        docs_per_sentence_estimate=3,
        seed=seed + 101,
    )
    homogeneous_plan = _build_language_doc_plan(
        language_stats,
        source_key="main",
        target_docs=homo_target,
        docs_per_sentence_estimate=4,
        seed=seed + 202,
    )
    spliced_plan = _build_language_doc_plan(
        language_stats,
        source_key="main",
        target_docs=spliced_target,
        docs_per_sentence_estimate=4,
        seed=seed + 303,
    )
    mixed_target = max(0, target_docs - len(pure_plan) - len(homogeneous_plan) - len(spliced_plan))

    used_docs = {"pure": 0, "homogeneous": 0, "spliced": 0, "mixed": 0}
    used_sentences = {"reserved": defaultdict(int), "main": defaultdict(int)}
    example_presence = defaultdict(int)
    depleted: list[dict[str, Any]] = []

    def consume(lang: str, pool_name: str, n_sentences: int) -> bool:
        available = pool_state[pool_name].get(lang, 0)
        if available < n_sentences:
            return False
        pool_state[pool_name][lang] = available - n_sentences
        used_sentences[pool_name][lang] += n_sentences
        return True

    for lang in pure_plan:
        n_sentences = _sample_sentence_count(rng, "pure")
        if not consume(lang, "reserved", n_sentences):
            depleted.append(
                {
                    "phase": "pure",
                    "lang": lang,
                    "needed": n_sentences,
                    "available": int(pool_state["reserved"].get(lang, 0)),
                }
            )
            break
        used_docs["pure"] += 1
        example_presence[lang] += 1

    for lang in homogeneous_plan:
        n_sentences = _sample_sentence_count(rng, "homogeneous")
        if not consume(lang, "main", n_sentences):
            depleted.append(
                {
                    "phase": "homogeneous",
                    "lang": lang,
                    "needed": n_sentences,
                    "available": int(pool_state["main"].get(lang, 0)),
                }
            )
            break
        used_docs["homogeneous"] += 1
        example_presence[lang] += 1

    for lang in spliced_plan:
        n_sentences = _sample_sentence_count(rng, "spliced")
        if not consume(lang, "main", n_sentences):
            depleted.append(
                {
                    "phase": "spliced",
                    "lang": lang,
                    "needed": n_sentences,
                    "available": int(pool_state["main"].get(lang, 0)),
                }
            )
            break
        used_docs["spliced"] += 1
        example_presence[lang] += 1

    # Approximate mixed docs as one main-pool sentence per chosen segment.
    for _ in range(mixed_target):
        n_segments = rng.randint(
            int(DOC_MIX["mixed"]["min_segments"]),
            int(DOC_MIX["mixed"]["max_segments"]),
        )
        chosen: list[str] = []
        seen: set[str] = set()
        for _ in range(n_segments):
            candidates = [
                lang
                for lang in ALL_LANGS
                if pool_state["main"].get(lang, 0) > 0 and lang not in seen
            ]
            if not candidates:
                break
            weights = [
                _bucket_weight(lang) * float(pool_state["main"].get(lang, 0))
                for lang in candidates
            ]
            lang = rng.choices(candidates, weights=weights, k=1)[0]
            seen.add(lang)
            chosen.append(lang)
        if not chosen:
            depleted.append({"phase": "mixed", "lang": None, "needed": 1, "available": 0})
            break
        ok = True
        for lang in chosen:
            if not consume(lang, "main", 1):
                ok = False
                depleted.append(
                    {
                        "phase": "mixed",
                        "lang": lang,
                        "needed": 1,
                        "available": int(pool_state["main"].get(lang, 0)),
                    }
                )
                break
        if not ok:
            break
        used_docs["mixed"] += 1
        for lang in chosen:
            example_presence[lang] += 1

    remaining_reserved = {lang: count for lang, count in pool_state["reserved"].items() if count > 0}
    remaining_main = {lang: count for lang, count in pool_state["main"].items() if count > 0}

    return {
        "target_docs": target_docs,
        "planned_docs": {
            "pure": len(pure_plan),
            "homogeneous": len(homogeneous_plan),
            "spliced": len(spliced_plan),
            "mixed": mixed_target,
        },
        "used_docs": used_docs,
        "used_sentences": {
            "reserved": dict(used_sentences["reserved"]),
            "main": dict(used_sentences["main"]),
        },
        "example_presence": dict(example_presence),
        "remaining_reserved": remaining_reserved,
        "remaining_main": remaining_main,
        "depleted": depleted,
    }


def _summarize_run(run: dict[str, Any], language_stats: dict[str, dict[str, int]], *, top_k: int) -> None:
    target_docs = int(run["target_docs"])
    print(f"Target docs: {target_docs:,}")
    print(
        "Planned docs: "
        f"pure={run['planned_docs']['pure']:,} "
        f"homogeneous={run['planned_docs']['homogeneous']:,} "
        f"spliced={run['planned_docs']['spliced']:,} "
        f"mixed={run['planned_docs']['mixed']:,}"
    )
    print(
        "Used docs: "
        f"pure={run['used_docs']['pure']:,} "
        f"homogeneous={run['used_docs']['homogeneous']:,} "
        f"spliced={run['used_docs']['spliced']:,} "
        f"mixed={run['used_docs']['mixed']:,}"
    )

    if run["depleted"]:
        first = run["depleted"][0]
        print(
            "First depletion: "
            f"phase={first['phase']} lang={first['lang']} "
            f"needed={first['needed']} available={first['available']}"
        )
    else:
        print("No depletion during the run.")

    def _top_bottlenecks(pool_name: str, remaining_key: str) -> None:
        print(f"\nTop bottlenecks for {pool_name}:")
        rows = []
        for lang, stats in language_stats.items():
            total = int(stats.get(pool_name, 0))
            remaining = int(run[remaining_key].get(lang, 0))
            used = total - remaining
            if total <= 0:
                continue
            rows.append((lang, used, total, used / total))
        for lang, used, total, frac in sorted(rows, key=lambda item: (-item[3], -item[1], item[0]))[:top_k]:
            print(f"  {lang:<5} used={used:>7,} / {total:<7,} ({frac:6.1%})")

    def _most_headroom(pool_name: str, remaining_key: str) -> None:
        print(f"\nMost headroom for {pool_name}:")
        rows = []
        for lang, stats in language_stats.items():
            total = int(stats.get(pool_name, 0))
            remaining = int(run[remaining_key].get(lang, 0))
            used = total - remaining
            if total <= 0:
                continue
            rows.append((lang, used, total, remaining / total, remaining))
        for lang, used, total, frac, remaining in sorted(rows, key=lambda item: (-item[3], -item[4], item[0]))[:top_k]:
            print(f"  {lang:<5} remaining={remaining:>7,} / {total:<7,} ({frac:6.1%}) used={used:>7,}")

    _top_bottlenecks("reserved", "remaining_reserved")
    _top_bottlenecks("main", "remaining_main")
    _most_headroom("reserved", "remaining_reserved")
    _most_headroom("main", "remaining_main")

    presence = run.get("example_presence", {})
    if isinstance(presence, dict) and presence:
        rows = []
        for lang in ALL_LANGS:
            count = int(presence.get(lang, 0))
            rows.append((lang, count, count / max(1, target_docs)))

        print("\nExample presence share:")
        for lang, count, frac in sorted(rows, key=lambda item: (-item[2], -item[1], item[0]))[:top_k]:
            print(f"  {lang:<5} examples={count:>7,} / {target_docs:<7,} ({frac:6.1%})")

        print("\nLowest presence share:")
        for lang, count, frac in sorted(rows, key=lambda item: (item[2], item[1], item[0]))[:top_k]:
            print(f"  {lang:<5} examples={count:>7,} / {target_docs:<7,} ({frac:6.1%})")


def _estimate_trials(
    *,
    language_stats: dict[str, dict[str, int]],
    target_docs: int,
    seed: int,
    trials: int,
    workers: int,
) -> list[dict[str, Any]]:
    if trials <= 1:
        return [_simulate_single_run(language_stats=language_stats, target_docs=target_docs, seed=seed)]

    seeds = [seed + i * 10_000 for i in range(trials)]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_simulate_single_run, language_stats=language_stats, target_docs=target_docs, seed=trial_seed)
            for trial_seed in seeds
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Trials"):
            results.append(future.result())
    return results


def _format_language_rank(rows: list[tuple[str, int, int, float]], title: str) -> None:
    print(title)
    for lang, used, total, frac in rows:
        print(f"  {lang:<5} used={used:>7,} / {total:<7,} ({frac:6.1%})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate synthetic-data capacity from the cached source-pool metadata. "
            "This is a tuning aid, not an exact reproduction of training."
        )
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=PATHS["source_pools"]["cache_meta"],
        help="Path to sentence_pools.meta.json.",
    )
    parser.add_argument(
        "--target-docs",
        type=int,
        default=int(RUN["target"]),
        help="Synthetic doc target to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the simulation.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of Monte Carlo trials to run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for Monte Carlo trials.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many bottleneck languages to print per pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = _load_json(args.meta)
    language_stats = meta.get("language_stats", {})
    if not isinstance(language_stats, dict) or not language_stats:
        raise RuntimeError(f"No language_stats found in {args.meta}")

    print(f"Loaded {args.meta}")
    print(f"Languages in manifest: {len(language_stats)}")
    print(
        "Doc mix: "
        f"pure={DOC_MIX['pure']['fraction']:.2f}, "
        f"homogeneous={DOC_MIX['homogeneous']['fraction']:.2f}, "
        f"spliced={DOC_MIX['spliced']['fraction']:.2f}, "
        f"mixed={DOC_MIX['mixed']['fraction']:.2f}"
    )
    print()

    results = _estimate_trials(
        language_stats=language_stats,
        target_docs=args.target_docs,
        seed=args.seed,
        trials=args.trials,
        workers=args.workers,
    )

    first = results[0]
    _summarize_run(first, language_stats, top_k=args.top_k)

    if len(results) > 1:
        planned_pure = np.array([r["planned_docs"]["pure"] for r in results], dtype=np.float64)
        planned_homo = np.array([r["planned_docs"]["homogeneous"] for r in results], dtype=np.float64)
        planned_spliced = np.array([r["planned_docs"]["spliced"] for r in results], dtype=np.float64)
        planned_mixed = np.array([r["planned_docs"]["mixed"] for r in results], dtype=np.float64)
        used_pure = np.array([r["used_docs"]["pure"] for r in results], dtype=np.float64)
        used_homo = np.array([r["used_docs"]["homogeneous"] for r in results], dtype=np.float64)
        used_spliced = np.array([r["used_docs"]["spliced"] for r in results], dtype=np.float64)
        used_mixed = np.array([r["used_docs"]["mixed"] for r in results], dtype=np.float64)

        print("\nMonte Carlo summary:")
        print(
            f"  planned pure: {planned_pure.mean():,.1f} +/- {planned_pure.std(ddof=0):,.1f}"
        )
        print(
            f"  planned homogeneous: {planned_homo.mean():,.1f} +/- {planned_homo.std(ddof=0):,.1f}"
        )
        print(
            f"  planned spliced: {planned_spliced.mean():,.1f} +/- {planned_spliced.std(ddof=0):,.1f}"
        )
        print(
            f"  planned mixed: {planned_mixed.mean():,.1f} +/- {planned_mixed.std(ddof=0):,.1f}"
        )
        print(f"  used pure: {used_pure.mean():,.1f} +/- {used_pure.std(ddof=0):,.1f}")
        print(f"  used homogeneous: {used_homo.mean():,.1f} +/- {used_homo.std(ddof=0):,.1f}")
        print(f"  used spliced: {used_spliced.mean():,.1f} +/- {used_spliced.std(ddof=0):,.1f}")
        print(f"  used mixed: {used_mixed.mean():,.1f} +/- {used_mixed.std(ddof=0):,.1f}")


if __name__ == "__main__":
    main()
