from __future__ import annotations

import json
import os
import random
from typing import Callable

from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

from paths import SENTENCES_DIR, SMOL_CACHE_FILE
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

        ds = load_dataset("google/smol", config, split="train", trust_remote_code=True)
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


def load_smol_sentences(
    *,
    sentences_dir: str = SENTENCES_DIR,
    lang_to_group: dict[str, str],
    use: bool | None = None,
    force_rebuild: bool | None = None,
    seed: int | None = None,
    max_sentences_per_lang: int | None = None,
    uncapped_langs: set[str] | None = None,
) -> dict[str, list[str]] | None:
    from source_config import SMOL

    use = SMOL["use"] if use is None else use
    if not use:
        return None
    force_rebuild = False if force_rebuild is None else force_rebuild
    seed = 42 if seed is None else seed
    max_sentences_per_lang = MAX_SENTENCES_PER_LANG if max_sentences_per_lang is None else max_sentences_per_lang
    cache_file = SMOL_CACHE_FILE if sentences_dir == SENTENCES_DIR else os.path.join(sentences_dir, "smol_sentences.json")
    uncapped_langs = uncapped_langs or UNCAPPED_LANGS
    if not force_rebuild and os.path.exists(cache_file):
        print(f"Loading SMOL cache from {cache_file}")
        with open(cache_file, encoding="utf-8") as f:
            cached: dict[str, list[str]] = json.load(f)
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

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in result.values())
    print(f"\nSMOL sentences cached -> {cache_file}")
    print(f"  {len(result)} languages | {total:,} sentences total")
    for lang in sorted(result):
        print(f"  {lang:<6}  {len(result[lang]):>5} sentences")
    return result
