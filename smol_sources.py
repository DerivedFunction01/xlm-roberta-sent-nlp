from __future__ import annotations

import json
import os
import random
from typing import Callable

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic, write_sentence_parquet
from language import LANG_TO_GROUP
from paths import PATHS
from source_config import SMOL
from text_utils import _collapse_spaces, _get_segmenter, _is_valid_sentence, _strip_bracket_notes


SMOL_CODE_MAP: dict[str, str] = {
    "pt-PT": "pt",
    "yue": "zh",
    "ar-MA": "ar",
    "arz": "ar",
    "aeb": "ar",
    "ayl": "ar",
    "apd": "ar",
}
MAX_SENTENCES_PER_LANG = 5_000
UNCAPPED_LANGS: set[str] = set()
SMOL_CACHE_VERSION = 2


def _smol_parse_config(config: str) -> tuple[str, str] | None:
    try:
        _, pair = config.split("__", 1)
        sl, tl = pair.split("_", 1)
        return sl, tl
    except ValueError:
        return None


def _smol_add_sentences(bucket: list[str], raw_sentences: object, lang: str, lang_to_group: dict[str, str]) -> None:
    if isinstance(raw_sentences, str):
        raw_sentences = [raw_sentences]
    if not isinstance(raw_sentences, list):
        return
    for sent in raw_sentences:
        if not isinstance(sent, str):
            continue
        sent = _strip_bracket_notes(sent)
        sent = _collapse_spaces(sent)
        if _is_valid_sentence(sent, lang, lang_to_group):
            bucket.append(sent)


def _load_smoldoc(accumulator: dict[str, list[str]], lang_to_group: dict[str, str]) -> tuple[set[str], set[str]]:
    configs = [c for c in get_dataset_config_names("google/smol") if c.startswith("smoldoc__")]
    source_langs_seen: set[str] = set()
    target_langs_seen: set[str] = set()
    for config in tqdm(configs, desc="SmolDoc subsets"):
        parsed = _smol_parse_config(config)
        if parsed is None:
            continue
        sl, tl = parsed
        mapped = SMOL_CODE_MAP.get(tl, tl)
        if mapped not in lang_to_group:
            continue

        ds = load_dataset("google/smol", config, split="train")
        target_bucket = accumulator.setdefault(mapped, [])
        target_langs_seen.add(mapped)
        if sl in lang_to_group and sl not in source_langs_seen:
            source_bucket = accumulator.setdefault(sl, [])
            for row in ds:
                _smol_add_sentences(source_bucket, row.get("srcs") or row.get("src"), sl, lang_to_group)  # type: ignore[arg-type]
            source_langs_seen.add(sl)

        for row in ds:
            _smol_add_sentences(target_bucket, row.get("trgs"), mapped, lang_to_group)  # type: ignore[arg-type]
    return source_langs_seen, target_langs_seen


def _load_smolsent(accumulator: dict[str, list[str]], lang_to_group: dict[str, str]) -> tuple[set[str], set[str]]:
    configs = [c for c in get_dataset_config_names("google/smol") if c.startswith("smolsent__")]
    source_langs_seen: set[str] = set()
    target_langs_seen: set[str] = set()
    for config in tqdm(configs, desc="SmolSent subsets"):
        parsed = _smol_parse_config(config)
        if parsed is None:
            continue
        sl, tl = parsed
        mapped = SMOL_CODE_MAP.get(tl, tl)
        if mapped not in lang_to_group:
            continue

        ds = load_dataset("google/smol", config, split="train", trust_remote_code=True)
        target_bucket = accumulator.setdefault(mapped, [])
        target_langs_seen.add(mapped)
        if sl in lang_to_group and sl not in source_langs_seen:
            source_bucket = accumulator.setdefault(sl, [])
            for row in ds:
                _smol_add_sentences(source_bucket, row.get("srcs") or row.get("src"), sl, lang_to_group)  # type: ignore[arg-type]
            source_langs_seen.add(sl)

        seg = _get_segmenter(mapped)
        for row in ds:
            trg = row.get("trg") or ""  # type: ignore[assignment]
            if not isinstance(trg, str):
                continue
            sents = seg.segment(trg) if seg else [trg]  # type: ignore[union-attr]
            _smol_add_sentences(target_bucket, sents, mapped, lang_to_group)
    return source_langs_seen, target_langs_seen


def _smol_cache_dir(sentences_dir: str) -> str:
    return PATHS["smol"]["cache_dir"] if sentences_dir == PATHS["sentences_dir"] else os.path.join(sentences_dir, "smol_sentences")


def _smol_cache_meta_path(sentences_dir: str) -> str:
    return PATHS["smol"]["cache_meta"] if sentences_dir == PATHS["sentences_dir"] else os.path.join(sentences_dir, "smol_sentences", "smol_sentences.meta.json")


def _smol_cache_path(cache_dir: str, lang: str) -> str:
    return os.path.join(cache_dir, f"{lang}.parquet")


def _smol_legacy_cache_file(sentences_dir: str) -> str:
    return os.path.join(sentences_dir, "smol_sentences.json")


def _load_smol_cache_map(cache_dir: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name in tqdm(sorted(os.listdir(cache_dir)), desc="SMOL cache load"):
        if not name.endswith(".parquet"):
            continue
        path = os.path.join(cache_dir, name)
        frame = pd.read_parquet(path)
        if "sentence" not in frame.columns:
            continue
        lang = name[:-8]
        result[lang] = frame["sentence"].astype(str).tolist()
    return result


def _write_smol_cache_map(cache_dir: str, sentence_map: dict[str, list[str]]) -> dict[str, int]:
    os.makedirs(cache_dir, exist_ok=True)
    counts: dict[str, int] = {}
    for lang in tqdm(sorted(sentence_map), desc="SMOL cache write"):
        sentences = sentence_map[lang]
        path = _smol_cache_path(cache_dir, lang)
        write_sentence_parquet(path, sentences)
        counts[lang] = len(sentences)
    return counts


def _load_smol_meta(cache_meta: str, expected_meta: dict[str, object]) -> dict[str, object] | None:
    if not os.path.exists(cache_meta):
        return None
    with open(cache_meta, encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        return None
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            return None
    return meta


def load_smol_sentences(
    *,
    sentences_dir: str = PATHS["sentences_dir"],
    lang_to_group: dict[str, str] = LANG_TO_GROUP,
    use: bool = SMOL["use"],
    force_rebuild: bool | None = None,
    seed: int = 42,
    max_sentences_per_lang: int = MAX_SENTENCES_PER_LANG,
    uncapped_langs: set[str] | None = None,
) -> dict[str, list[str]] | None:

    if not use:
        return None
    force_rebuild = False if force_rebuild is None else force_rebuild
    cache_dir = _smol_cache_dir(sentences_dir)
    cache_meta = _smol_cache_meta_path(sentences_dir)
    legacy_cache_file = _smol_legacy_cache_file(sentences_dir)
    uncapped_langs = uncapped_langs or UNCAPPED_LANGS
    expected_meta = {
        "cache_version": SMOL_CACHE_VERSION,
        "cache_layout": "per_language_parquet_v1",
        "seed": seed,
        "max_sentences_per_lang": max_sentences_per_lang,
        "uncapped_langs": sorted(uncapped_langs),
    }
    if not force_rebuild:
        cached_meta = _load_smol_meta(cache_meta, expected_meta)
        cache_paths = [name for name in os.listdir(cache_dir)] if os.path.isdir(cache_dir) else []
        parquet_paths = [name for name in cache_paths if name.endswith(".parquet")]
        if cached_meta is not None and parquet_paths:
            print(f"Loading SMOL cache from {cache_dir}")
            cached = _load_smol_cache_map(cache_dir)
            total = sum(len(v) for v in cached.values())
            print(f"  {len(cached)} languages | {total:,} sentences total")
            return cached
        if os.path.exists(legacy_cache_file):
            print(f"Converting legacy SMOL cache from {legacy_cache_file} to parquet")
            with open(legacy_cache_file, encoding="utf-8") as f:
                cached: dict[str, list[str]] = json.load(f)
            _write_smol_cache_map(cache_dir, cached)
            write_json_atomic(
                cache_meta,
                {
                    **expected_meta,
                    "total_sentences": sum(len(v) for v in cached.values()),
                    "languages": len(cached),
                },
            )
            print(f"  {len(cached)} languages | {sum(len(v) for v in cached.values()):,} sentences total")
            return cached

    accumulator: dict[str, list[str]] = {}
    print("Loading SmolDoc ...")
    smoldoc_src_langs, smoldoc_target_langs = _load_smoldoc(accumulator, lang_to_group)
    print("Loading SmolSent ...")
    smolsent_src_langs, smolsent_target_langs = _load_smolsent(accumulator, lang_to_group)
    print(f"SMOL src languages loaded once: {len(smoldoc_src_langs | smolsent_src_langs)}")
    print(f"SMOL target languages loaded: {len(smoldoc_target_langs | smolsent_target_langs)}")

    rng = random.Random(seed)
    result: dict[str, list[str]] = {}
    for lang, sents in sorted(accumulator.items()):
        seen: set[str] = set()
        deduped: list[str] = []
        for sent in sents:
            if sent not in seen:
                seen.add(sent)
                deduped.append(sent)
        rng.shuffle(deduped)
        cap = None if lang in uncapped_langs else max_sentences_per_lang
        result[lang] = deduped if cap is None else deduped[:cap]

    _write_smol_cache_map(cache_dir, result)
    write_json_atomic(
        cache_meta,
        {
            **expected_meta,
            "total_sentences": sum(len(v) for v in result.values()),
            "languages": len(result),
        },
    )

    total = sum(len(v) for v in result.values())
    print(f"\nSMOL sentences cached -> {cache_dir}/")
    print(f"  {len(result)} languages | {total:,} sentences total")
    for lang in sorted(result):
        print(f"  {lang:<6}  {len(result[lang]):>5} sentences")
    return result
