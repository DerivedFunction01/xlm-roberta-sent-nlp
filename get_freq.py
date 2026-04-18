from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from io_utils import write_json_atomic
import text_utils
from language import ALL_LANGS, LANG_TO_GROUP, LANGUAGE_GROUPS, canonical_lang
from paths import PATHS
from source_config import MAJOR_LATIN_BUCKETS

BASE = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018"
DEFAULT_INPUT_PARQUET = Path("word_dict.parquet")
DEFAULT_OUTPUT_DIR = Path(PATHS["freq"]["cache_dir"])
DEFAULT_SEED = 42
SOURCE_POOL_CACHE_VERSION = 1

MAJOR_LATIN_LANGS = frozenset(
    lang for bucket in MAJOR_LATIN_BUCKETS for lang in LANGUAGE_GROUPS.get(bucket, ())
)
NORDIC_LANGS = frozenset(LANGUAGE_GROUPS.get("NordicCore", ()))
MINOR_LATIN_GAP_GROUPS = text_utils.ENGLISH_MINOR_LATIN_GROUPS

LANG_CONFIG = {
    "en": {"cutoff": 5650, "min_freq": 5},
    "es": {"cutoff": 4600, "min_freq": 5},
    "fr": {"cutoff": 3200, "min_freq": 5},
    "de": {"cutoff": 1950, "min_freq": 5},
    "it": {"cutoff": 3300, "min_freq": 5},
    "pt": {"cutoff": 2600, "min_freq": 5},
    "da": {"cutoff": 2450, "min_freq": 5},
    "no": {"cutoff": 2000, "min_freq": 5},
    "sv": {"cutoff": 1300, "min_freq": 5},
    "pl": {"cutoff": 3950, "min_freq": 5},
    "tr": {"cutoff": 3250, "min_freq": 5},
    "fi": {"cutoff": 2800, "min_freq": 5},
    "vi": {"cutoff": 1100, "min_freq": 5},
    "id": {"cutoff": 1200, "min_freq": 5},
    "ru": {"cutoff": 2000, "min_freq": 5},
    "uk": {"cutoff": 500, "min_freq": 5},
    "ar": {"cutoff": 1950, "min_freq": 5},
    "hi": {"cutoff": 1950, "min_freq": 5},
}

KEEP_APOSTROPHE_START = False
STRIP_MID_APOSTROPHE = False


def _is_int_token(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _infer_freq_column(sample_rows: list[list[str]]) -> int:
    first_col_hits = 0
    second_col_hits = 0
    for parts in sample_rows:
        if len(parts) < 2:
            continue
        if _is_int_token(parts[0]):
            first_col_hits += 1
        if _is_int_token(parts[1]):
            second_col_hits += 1
    if first_col_hits > second_col_hits:
        return 0
    return 1


def _parse_word_freq(parts: list[str], freq_col: int) -> tuple[str, int] | None:
    if len(parts) < 2:
        return None
    if freq_col not in (0, 1):
        raise ValueError(f"Unsupported frequency column: {freq_col}")
    other_col = 1 - freq_col
    if len(parts) <= other_col:
        return None
    word_col = other_col
    if not _is_int_token(parts[freq_col]):
        if _is_int_token(parts[other_col]):
            freq_col = other_col
            word_col = 1 - freq_col
        else:
            return None
    word = parts[word_col]
    freq = int(parts[freq_col])
    return word, freq


def _fetch_wordlist_text(lang: str) -> str:
    candidates = [f"{BASE}/{lang}/{lang}_50k.txt", f"{BASE}/{lang}/{lang}_full.txt"]
    last_error: Exception | None = None
    for url in candidates:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as err:
            last_error = err
            status_code = getattr(getattr(err, "response", None), "status_code", None)
            if status_code != 404:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not fetch a frequency word list for {lang}")


def build_label_maps(all_langs: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id: dict[str, int] = {"O": 0}
    id2label: dict[int, str] = {0: "O"}
    for idx, lang in enumerate(all_langs):
        b_id = 2 * idx + 1
        i_id = 2 * idx + 2
        label2id[f"B-{lang.upper()}"] = b_id
        label2id[f"I-{lang.upper()}"] = i_id
        id2label[b_id] = f"B-{lang.upper()}"
        id2label[i_id] = f"I-{lang.upper()}"
    return label2id, id2label


LABEL2ID, ID2LABEL = build_label_maps(ALL_LANGS)


def fetch_wordlist(lang: str, cutoff: int, min_freq: int) -> tuple[list[dict], int]:
    lang = canonical_lang(lang)
    lines = _fetch_wordlist_text(lang).splitlines()
    sample_rows = [line.strip().split() for line in lines[: min(10, len(lines))]]
    freq_col = _infer_freq_column(sample_rows)
    rows = []
    contaminated_count = 0
    for i, line in enumerate(lines):
        if i >= cutoff:
            break
        parts = line.strip().split()
        parsed = _parse_word_freq(parts, freq_col)
        if parsed is None:
            continue
        word, freq = parsed
        if freq < min_freq:
            continue
        starts_apos = word.startswith("'")
        has_mid_apos = "'" in word and not starts_apos
        if starts_apos and not KEEP_APOSTROPHE_START:
            continue
        if has_mid_apos and STRIP_MID_APOSTROPHE:
            continue
        word = word.lower()
        if not text_utils._is_valid_word(word):
            continue
        if text_utils._has_script_contamination(word, lang, LANG_TO_GROUP):
            contaminated_count += 1
            continue
        rows.append(
            {
                "word": word,
                "lang": lang,
                "freq": freq,
                "rank": i + 1,
            }
        )
    return rows, contaminated_count


def _parse_overlap_langs(value: Any) -> set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    return {canonical_lang(lang) for lang in value.split(",") if lang.strip()}


def _stable_seed(*parts: str) -> int:
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "big", signed=False)


def _normalize_word_dict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lang"] = df["lang"].map(canonical_lang)
    if "overlaps" not in df.columns:
        overlap = df.groupby("word")["lang"].apply(lambda x: ",".join(sorted(set(x)))).reset_index()
        overlap.columns = ["word", "all_langs"]
        df = df.merge(overlap, on="word")
        df["overlaps"] = df.apply(
            lambda r: ",".join(l for l in r["all_langs"].split(",") if l != r["lang"]), axis=1
        )
        df["is_overlap"] = df["overlaps"] != ""
    df["overlap_langs"] = df["overlaps"].map(_parse_overlap_langs)
    df["overlap_count"] = df["overlap_langs"].map(len)
    df["lang_size"] = df.groupby("lang")["word"].transform("count")
    df["lang_rank_order"] = df.groupby("lang")["rank"].rank(method="first", ascending=True)
    df["relative_rank"] = df.groupby("lang")["lang_rank_order"].transform(
        lambda s: 1.0 if len(s) <= 1 else 1.0 - ((s - 1.0) / (len(s) - 1.0))
    )
    return df


def _should_keep_word(row: pd.Series) -> bool:
    lang = canonical_lang(str(row["lang"]))
    overlap_langs = row["overlap_langs"]
    if LANG_TO_GROUP.get(lang, "") in MINOR_LATIN_GAP_GROUPS and overlap_langs & MAJOR_LATIN_LANGS:
        return False
    return True


def _row_weight(row: pd.Series) -> float:
    relative_rank = float(row["relative_rank"])
    word = str(row["word"])
    lang = canonical_lang(str(row["lang"]))
    overlap_langs = row["overlap_langs"]

    weight = 0.35 + (relative_rank * 1.75)
    if len(word) <= 3:
        weight *= 1.30
    elif len(word) <= 5:
        weight *= 1.10

    if overlap_langs:
        if lang in MAJOR_LATIN_LANGS:
            weight *= 1.10
        elif lang in NORDIC_LANGS:
            weight *= 0.85
        elif lang == "ru" and "uk" in overlap_langs:
            weight *= 1.25
        elif lang == "uk" and "ru" in overlap_langs:
            weight *= 0.75
    return weight


def _repeat_count(relative_rank: float, overlap_count: int) -> int:
    if relative_rank >= 0.85:
        count = 3
    elif relative_rank >= 0.45:
        count = 2
    else:
        count = 1
    if overlap_count > 0:
        count = min(3, max(count, 2))
    return count


def _weighted_sample_without_replacement(
    candidates: list[dict[str, Any]],
    *,
    k: int,
    rng: random.Random,
    forbidden_words: set[str] | None = None,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    pool = [
        row
        for row in candidates
        if forbidden_words is None or str(row["word"]) not in forbidden_words
    ]
    if not pool:
        return []

    chosen: list[dict[str, Any]] = []
    weights = [float(row["sample_weight"]) for row in pool]
    for _ in range(min(k, len(pool))):
        total_weight = sum(weights)
        if total_weight <= 0:
            index = rng.randrange(len(pool))
        else:
            target = rng.random() * total_weight
            running = 0.0
            index = 0
            for idx, weight in enumerate(weights):
                running += weight
                if running >= target:
                    index = idx
                    break
        chosen.append(pool.pop(index))
        weights.pop(index)
    return chosen


def _build_example(
    seed_row: dict[str, Any],
    *,
    lang_pool: list[dict[str, Any]],
    ngram_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    seed_word = str(seed_row["word"])
    lang = canonical_lang(str(seed_row["lang"]))
    context_rows = _weighted_sample_without_replacement(
        lang_pool,
        k=ngram_size - 1,
        rng=rng,
        forbidden_words={seed_word},
    )
    tokens = [seed_word, *[str(row["word"]) for row in context_rows]]
    if len(tokens) > 1:
        rng.shuffle(tokens)

    label_name = f"{lang.upper()}"
    labels = [LABEL2ID[f"B-{label_name}"]]
    labels.extend([LABEL2ID[f"I-{label_name}"] for _ in range(len(tokens) - 1)])

    return {
        "word": seed_word,
        "lang": lang,
        "freq": int(seed_row["freq"]),
        "rank": int(seed_row["rank"]),
        "relative_rank": float(seed_row["relative_rank"]),
        "overlaps": seed_row["overlaps"],
        "overlap_count": int(seed_row["overlap_count"]),
        "is_overlap": bool(seed_row["is_overlap"]),
        "sample_weight": float(seed_row["sample_weight"]),
        "source_type": {1: "unigram", 2: "bigram", 3: "trigram"}[len(tokens)],
        "tokens": tokens,
        "ner_tags": labels,
        "original_text": " ".join(tokens),
    }


def _build_language_examples(
    lang: str,
    lang_df: pd.DataFrame,
    *,
    seed: int,
    tokenizer=None,
) -> list[dict[str, Any]]:
    lang_pool = lang_df.to_dict("records")
    lang_rng = random.Random(_stable_seed(str(seed), lang))
    lang_examples: list[dict[str, Any]] = []

    for row in tqdm(lang_pool, desc=f"{lang} rows", leave=False):
        repeat_count = _repeat_count(float(row["relative_rank"]), int(row["overlap_count"]))
        for repeat_idx in range(repeat_count):
            ngram_size = min(3, repeat_idx + 1)
            example = _build_example(
                row,
                lang_pool=lang_pool,
                ngram_size=ngram_size,
                rng=lang_rng,
            )
            if tokenizer is not None:
                example = _finalize_example(example, tokenizer)
            lang_examples.append(example)

    return lang_examples


def _continuation_label_id(label_id: int) -> int:
    """Map a beginning label to its continuation label for split wordpieces."""
    label_name = ID2LABEL.get(label_id)
    if isinstance(label_name, str) and label_name.startswith("B-"):
        return LABEL2ID[f"I-{label_name.removeprefix('B-')}"]
    return label_id


def _finalize_example(example: dict[str, Any], tokenizer) -> dict[str, Any]:
    """Tokenize the example and expand labels across split wordpieces."""
    tokens = example["tokens"]
    labels = example["ner_tags"]
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        add_special_tokens=True,
    )
    word_ids = encoding.word_ids()
    aligned_labels: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            aligned_labels.append(labels[word_id])
        else:
            aligned_labels.append(_continuation_label_id(labels[word_id]))
        previous_word_id = word_id

    finalized = dict(example)
    finalized["input_ids"] = encoding["input_ids"]
    finalized["attention_mask"] = encoding["attention_mask"]
    finalized["labels"] = aligned_labels
    return finalized


def build_short_text_source_pools(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    normalized = _normalize_word_dict(df)
    kept = normalized[normalized.apply(_should_keep_word, axis=1)].copy()
    kept["sample_weight"] = kept.apply(_row_weight, axis=1)

    language_frames: dict[str, pd.DataFrame] = {}
    lang_counts: dict[str, dict[str, int]] = {}

    grouped = list(kept.sort_values(["lang", "freq", "rank"], ascending=[True, False, True]).groupby("lang", sort=False))
    for lang, lang_df in tqdm(grouped, desc="Building short-text source pools"):
        lang_examples = _build_language_examples(lang, lang_df, seed=seed, tokenizer=None)
        rows = [
            {
                "lang": example["lang"],
                "sentence": example["original_text"],
                "word": example["word"],
                "freq": example["freq"],
                "rank": example["rank"],
                "relative_rank": example["relative_rank"],
                "overlaps": example["overlaps"],
                "overlap_count": example["overlap_count"],
                "is_overlap": example["is_overlap"],
                "sample_weight": example["sample_weight"],
                "source_type": example["source_type"],
            }
            for example in lang_examples
        ]
        language_frames[lang] = pd.DataFrame(rows)
        lang_counts[lang] = {"sentences": len(rows)}

    manifest = {
        "seed": seed,
        "languages": sorted(lang_counts),
        "counts": lang_counts,
        "cache_version": SOURCE_POOL_CACHE_VERSION,
        "source_format": "parquet language shards with sentence column",
        "row_definition": "one generated frequency example per row, repeated according to relative rank and overlap count",
        "source_pool_ready": True,
    }
    return language_frames, manifest


def _load_or_build_word_dict(input_parquet: Path) -> tuple[pd.DataFrame, int]:
    if input_parquet.exists():
        return pd.read_parquet(input_parquet), 0

    dfs: list[pd.DataFrame] = []
    contaminated_total = 0
    for lang, cfg in tqdm(LANG_CONFIG.items(), desc="Fetching word lists"):
        rows, contaminated_count = fetch_wordlist(lang, cfg["cutoff"], cfg["min_freq"])
        contaminated_total += contaminated_count
        dfs.append(pd.DataFrame(rows))

    if not dfs:
        raise RuntimeError("No word lists were loaded")
    df = pd.concat(dfs, ignore_index=True)
    overlap = df.groupby("word")["lang"].apply(lambda x: ",".join(sorted(set(x)))).reset_index()
    overlap.columns = ["word", "all_langs"]
    df = df.merge(overlap, on="word")
    df["overlaps"] = df.apply(
        lambda r: ",".join(l for l in r["all_langs"].split(",") if l != r["lang"]), axis=1
    )
    df["is_overlap"] = df["overlaps"] != ""
    df[["word", "lang", "freq", "rank", "overlaps", "is_overlap"]].to_parquet(input_parquet, index=False)
    return df, contaminated_total


def _write_source_pool_dir(output_dir: Path, language_frames: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for lang, frame in language_frames.items():
        frame.to_parquet(output_dir / f"{lang}.parquet", index=False)
    write_json_atomic(output_dir / "manifest.json", manifest)


def build_freq_source_pool(
    *,
    input_parquet: Path = DEFAULT_INPUT_PARQUET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    force_rebuild: bool = False,
) -> tuple[int, int]:
    manifest_path = output_dir / "manifest.json"
    if not force_rebuild and output_dir.exists() and manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            total_examples = sum(int(stats.get("sentences", 0)) for stats in manifest.get("counts", {}).values())
            return total_examples, 0
        except Exception:
            pass

    df, contaminated_total = _load_or_build_word_dict(input_parquet)
    language_frames, manifest = build_short_text_source_pools(
        df,
        seed=seed,
    )
    _write_source_pool_dir(output_dir, language_frames, manifest)
    total_examples = sum(len(frame) for frame in language_frames.values())
    return total_examples, contaminated_total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a parquet-backed frequency-word source pool for synthetic sampling."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=DEFAULT_INPUT_PARQUET,
        help="Path to the cleaned word dictionary parquet. If missing, it will be built from the raw frequency lists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where language parquet shards will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for short-text generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    total_examples, contaminated_total = build_freq_source_pool(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
        seed=args.seed,
        force_rebuild=True,
    )

    print(f"Done — wrote {total_examples:,} examples across language shards to {args.output_dir}")
    if contaminated_total:
        print(f"Skipped {contaminated_total:,} contaminated raw rows during frequency parsing")


if __name__ == "__main__":
    main()
