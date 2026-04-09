from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from instruction_dataset_sources import DEFAULT_INSTRUCTION_SOURCE_SPECS, INSTRUCTION_SOURCE_EXTRACTORS
from io_utils import write_json_atomic, write_sentence_parquet
from language import LATIN_GROUPS, LANG_TO_GROUP
from paths import PATHS
from source_config import INSTRUCT
from text_utils import _collapse_spaces, _strip_bracket_notes


INSTRUCTION_CACHE_VERSION = 2
DEFAULT_MAX_SENTENCES_PER_LANG = INSTRUCT["max_lang"]
DEFAULT_SOURCE_SPECS = DEFAULT_INSTRUCTION_SOURCE_SPECS
_HAS_WORD_OR_IDEOGRAPH = re.compile(r"\w", flags=re.UNICODE)
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}", flags=re.UNICODE)
_MATH_SYMBOL_RE = re.compile(r"[=+\-*/^<>|~]")
_CODE_FENCE_RE = re.compile(r"```|~~~")
_CODE_KEYWORD_RE = re.compile(
    r"\b(import|from|def|class|return|lambda|function|const|let|var|public|private|protected|static|if|else|elif|for|while|try|catch|except|throw|throws|new|switch|case|package|include|using|namespace)\b"
)
_CODE_PATTERN_RE = re.compile(
    r"(^|\n)\s*(?:"
    r"(?:def|class|function|const|let|var)\s+\w+\s*(?:\(|=)|"
    r"(?:import|from)\s+\w+|"
    r"#include\s*<|"
    r"[{}\[\]();]{2,}|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=].*;|"
    r"<[A-Za-z][^>]*>|"
    r".*=>.*"
    r")",
    flags=re.UNICODE,
)


def _instruction_cache_dir(sentences_dir: str) -> str:
    return (
        PATHS["instruction"]["cache_dir"]
        if sentences_dir == PATHS["sentences_dir"]
        else os.path.join(sentences_dir, "instruction_sentences")
    )


def _instruction_cache_meta_path(sentences_dir: str) -> str:
    return (
        PATHS["instruction"]["cache_meta"]
        if sentences_dir == PATHS["sentences_dir"]
        else os.path.join(sentences_dir, "instruction_sentences", "instruction_sentences.meta.json")
    )


def _normalize_text(text: str) -> str:
    text = _strip_bracket_notes(text)
    text = _collapse_spaces(text)
    return text.strip()


def _normalize_source_spec(spec: dict[str, Any]) -> dict[str, Any]:
    config_name = spec.get("config_name")
    return {
        "name": str(spec.get("name") or spec.get("repo_id") or "source"),
        "repo_id": str(spec["repo_id"]),
        "config_name": None if config_name in {None, ""} else str(config_name),
        "split": str(spec.get("split", "train")),
        "lang": str(spec.get("lang", "")),
        "extractor": str(spec.get("extractor", "generic")),
        "trust_remote_code": bool(spec.get("trust_remote_code", False)),
        "max_rows": int(spec.get("max_rows", 0) or 0),
        "allow_code": bool(spec.get("allow_code", False)),
    }


def _source_signature(spec: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    normalized = _normalize_source_spec(spec)
    return tuple(sorted(normalized.items()))


def _load_instruction_meta_raw(cache_meta: str) -> dict[str, Any] | None:
    if not os.path.exists(cache_meta):
        return None
    try:
        with open(cache_meta, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _instruction_meta_state(
    *,
    meta: dict[str, Any] | None,
    expected_meta: dict[str, Any],
) -> str | None:
    if not isinstance(meta, dict):
        return None
    for key, value in expected_meta.items():
        if key == "sources":
            continue
        if meta.get(key) != value:
            return None
    current_sources = meta.get("sources")
    expected_sources = expected_meta.get("sources")
    if not isinstance(current_sources, list) or not isinstance(expected_sources, list):
        return None
    current_sigs = {_source_signature(source) for source in current_sources if isinstance(source, dict)}
    expected_sigs = {_source_signature(source) for source in expected_sources if isinstance(source, dict)}
    if current_sigs == expected_sigs:
        return "exact"
    if current_sigs.issubset(expected_sigs):
        return "subset"
    return None


def _load_instruction_cache_map(cache_dir: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not os.path.isdir(cache_dir):
        return result
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".parquet"):
            continue
        path = os.path.join(cache_dir, name)
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if "sentence" not in frame.columns:
            continue
        lang = name[:-8]
        sentences = [str(sentence) for sentence in frame["sentence"].tolist() if isinstance(sentence, str) and sentence.strip()]
        result[lang] = sentences
    return result


def _write_instruction_cache_map(cache_dir: str, sentence_map: dict[str, list[str]]) -> dict[str, int]:
    os.makedirs(cache_dir, exist_ok=True)
    counts: dict[str, int] = {}
    for lang, sentences in sorted(sentence_map.items()):
        path = os.path.join(cache_dir, f"{lang}.parquet")
        write_sentence_parquet(path, sentences)
        counts[lang] = len(sentences)
    return counts


def _dedupe_sentence_list(sentences: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    deduped: list[str] = []
    removed = 0
    for sentence in sentences:
        if not isinstance(sentence, str):
            removed += 1
            continue
        cleaned = _normalize_text(sentence)
        if not cleaned:
            removed += 1
            continue
        if cleaned in seen:
            removed += 1
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped, removed


def _looks_like_code(text: str) -> bool:
    if _CODE_FENCE_RE.search(text):
        return True
    if _CODE_KEYWORD_RE.search(text) and any(ch in text for ch in (";", "{", "}", "(", ")", "[", "]", "<", ">", "=>", "::")):
        return True
    if _CODE_PATTERN_RE.search(text):
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        code_like_lines = sum(
            1
            for line in lines
            if line.startswith(("    ", "\t"))
            or line.strip().startswith(("#", "//", "/*", "*", "*/"))
            or line.rstrip().endswith((";", "{", "}", ":"))
        )
        if code_like_lines / max(1, len(lines)) >= 0.5:
            return True
    return False


def _is_valid_instruction_text(
    text: str,
    lang: str,
    lang_to_group: dict[str, str],
    *,
    allow_code: bool = False,
) -> bool:
    cleaned = _normalize_text(text)
    if len(cleaned) < 2 or len(cleaned) > 1_500:
        return False
    if not _HAS_WORD_OR_IDEOGRAPH.search(cleaned):
        return False
    if not any(ch.isalpha() for ch in cleaned):
        return False
    token_count = len(_TOKEN_RE.findall(cleaned))
    symbol_count = sum(
        1 for ch in cleaned if not ch.isalnum() and not ch.isspace()
    )
    digit_count = sum(ch.isdigit() for ch in cleaned)
    math_symbol_count = len(_MATH_SYMBOL_RE.findall(cleaned))
    word_count = len(_LATIN_WORD_RE.findall(cleaned))
    is_latin = lang_to_group.get(lang) in LATIN_GROUPS
    if token_count < 4:
        return False
    if digit_count > len(cleaned) * 0.35:
        return False
    if symbol_count > len(cleaned) * 0.45:
        return False
    if math_symbol_count >= 3 and word_count <= 2:
        return False
    if not allow_code and _looks_like_code(cleaned):
        return False
    if is_latin and cleaned.isupper() and len(cleaned) <= 24:
        return False
    return True


def _records_to_sentence_map(records_by_lang: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    sentence_map: dict[str, list[str]] = {}
    for lang, records in records_by_lang.items():
        sentences: list[str] = []
        for record in records:
            text = record.get("text")
            if isinstance(text, str) and text.strip():
                sentences.append(text.strip())
        if sentences:
            sentence_map[lang] = sentences
    return sentence_map


def _normalize_sentence_map(sentence_map: dict[str, list[str]], seed: int, max_sentences_per_lang: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    result: dict[str, list[str]] = {}
    for lang, sentences in sorted(sentence_map.items()):
        deduped, _ = _dedupe_sentence_list(sentences)
        rng.shuffle(deduped)
        result[lang] = deduped[:max_sentences_per_lang]
    return result


def _merge_sentence_maps(base: dict[str, list[str]], new: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {lang: sentences[:] for lang, sentences in base.items()}
    for lang, sentences in new.items():
        merged.setdefault(lang, []).extend(sentences)
    return merged


def _process_source_spec(
    *,
    spec: dict[str, Any],
    lang_to_group: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int]]:
    repo_id = str(spec["repo_id"])
    source_name = str(spec["name"])
    split = str(spec.get("split", "train"))
    config_name = spec.get("config_name")
    lang = str(spec.get("lang", ""))
    extractor_name = str(spec.get("extractor", "generic"))
    trust_remote_code = bool(spec.get("trust_remote_code", False))
    max_rows = int(spec.get("max_rows", 0) or 0)
    allow_code = bool(spec.get("allow_code", False))
    extractor = INSTRUCTION_SOURCE_EXTRACTORS.get(extractor_name)

    sentence_map: dict[str, list[str]] = {}
    extractor_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    if extractor is None:
        tqdm.write(f"  Skipping {repo_id}: unknown extractor '{extractor_name}'")
        return sentence_map, dict(extractor_counts), dict(source_counts)

    try:
        load_kwargs: dict[str, Any] = {
            "split": split,
            "streaming": True,
            "trust_remote_code": trust_remote_code,
        }
        if config_name is not None:
            load_kwargs["name"] = config_name
        ds = load_dataset(repo_id, **load_kwargs)
    except Exception as exc:
        tqdm.write(f"  Skipping {repo_id}: {exc}")
        return sentence_map, dict(extractor_counts), dict(source_counts)

    for row_idx, row in enumerate(tqdm(ds, desc=repo_id, leave=False)):
        if max_rows > 0 and row_idx >= max_rows:
            tqdm.write(f"  Reached row cap for {repo_id}: {max_rows}")
            break
        if not isinstance(row, dict):
            continue
        raw_texts = extractor(row, spec)
        if not raw_texts:
            continue
        for text_idx, text in enumerate(raw_texts):
            if not _is_valid_instruction_text(text, lang, lang_to_group, allow_code=allow_code):
                continue
            sentence_map.setdefault(lang, []).append(_normalize_text(text))
            extractor_counts[extractor_name] += 1
            source_counts[source_name] += 1

    return sentence_map, dict(extractor_counts), dict(source_counts)


def load_instruction_sentences(
    *,
    sentences_dir: str = PATHS["sentences_dir"],
    lang_to_group: dict[str, str] = LANG_TO_GROUP,
    use: bool = INSTRUCT["use"],
    force_rebuild: bool | None = None,
    seed: int = 42,
    max_sentences_per_lang: int = DEFAULT_MAX_SENTENCES_PER_LANG,
    source_specs: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]] | None:
    if not use:
        return None

    force_rebuild = bool(force_rebuild) if force_rebuild is not None else INSTRUCT["rebuild"]
    source_specs = source_specs or INSTRUCT["sources"] or DEFAULT_SOURCE_SPECS
    normalized_sources = [_normalize_source_spec(spec) for spec in source_specs]

    cache_dir = _instruction_cache_dir(sentences_dir)
    cache_meta = _instruction_cache_meta_path(sentences_dir)
    expected_meta = {
        "cache_version": INSTRUCTION_CACHE_VERSION,
        "cache_layout": "per_language_parquet_v1",
        "seed": seed,
        "max_sentences_per_lang": max_sentences_per_lang,
        "sources": normalized_sources,
    }

    existing_meta = _load_instruction_meta_raw(cache_meta)
    cache_state = _instruction_meta_state(meta=existing_meta, expected_meta=expected_meta)
    if not force_rebuild and cache_state == "exact" and existing_meta is not None and existing_meta.get("status") in {"complete", "building"}:
        print(f"Loading instruction cache from {cache_dir}")
        cached = _load_instruction_cache_map(cache_dir)
        if cached:
            total = sum(len(v) for v in cached.values())
            print(f"  {len(cached)} languages | {total:,} sentences total")
            return cached

    incremental_update = (
        not force_rebuild
        and cache_state == "subset"
        and existing_meta is not None
        and existing_meta.get("status") in {"complete", "building"}
        and os.path.isdir(cache_dir)
    )
    if not force_rebuild and (os.path.exists(cache_dir) or os.path.exists(cache_meta)) and not incremental_update:
        raise RuntimeError(
            "Instruction cache metadata does not match the current config. "
            "Refusing to rebuild over the existing cache. "
            "Delete the cache files or set INSTRUCT['rebuild']=True to regenerate them."
        )

    base_result: dict[str, list[str]] = {}
    processed_signatures: set[tuple[tuple[str, Any], ...]] = set()
    if incremental_update and isinstance(existing_meta, dict):
        base_result = _load_instruction_cache_map(cache_dir)
        if not base_result and (existing_meta.get("lang_counts") or existing_meta.get("total_after")):
            incremental_update = False
            base_result = {}
        else:
            processed_signatures = {_source_signature(spec) for spec in existing_meta.get("sources", []) if isinstance(spec, dict)}
            source_specs = [spec for spec in normalized_sources if _source_signature(spec) not in processed_signatures]
            if not source_specs:
                print(f"Loading instruction cache from {cache_dir}")
                total = sum(len(v) for v in base_result.values())
                print(f"  {len(base_result)} languages | {total:,} sentences total")
                return base_result
            print(
                f"Expanding instruction cache with {len(source_specs)} new source(s) "
                f"(reusing {len(processed_signatures)} cached source(s)) ..."
            )

    print(f"Loading instruction sources ({len(source_specs)} dataset(s)) ...")
    print()
    current_sentence_map: dict[str, list[str]] = {lang: sentences[:] for lang, sentences in base_result.items()}
    if not current_sentence_map and incremental_update:
        current_sentence_map = {}
    processed_source_specs: list[dict[str, Any]] = []
    if incremental_update and isinstance(existing_meta, dict) and isinstance(existing_meta.get("sources"), list):
        processed_source_specs = [spec for spec in existing_meta["sources"] if isinstance(spec, dict)]
    source_extractor_counts: dict[str, dict[str, int]] = {}
    source_text_counts: dict[str, int] = {}
    existing_total_before = 0
    if incremental_update and isinstance(existing_meta, dict):
        base_total_after = sum(len(v) for v in base_result.values())
        existing_total_before = int(existing_meta.get("total_before", base_total_after))
    new_total_before = 0
    for spec in tqdm(source_specs, desc="Instruction sources"):
        normalized_spec = _normalize_source_spec(spec)
        source_name = normalized_spec["name"]
        repo_id = normalized_spec["repo_id"]
        lang = normalized_spec["lang"]
        extractor_name = normalized_spec["extractor"]
        config_name = normalized_spec.get("config_name")
        config_suffix = f" / {config_name}" if config_name else ""
        tqdm.write(f"  Reading {source_name} -> {lang}{config_suffix} [{extractor_name}]")
        source_sentence_map, extractor_counts, source_counts = _process_source_spec(spec=normalized_spec, lang_to_group=lang_to_group)
        source_extractor_counts[source_name] = extractor_counts
        source_total_before = int(sum(len(sentences) for sentences in source_sentence_map.values()))
        source_text_counts[source_name] = source_total_before
        new_total_before += source_total_before
        for source_lang, sentences in source_sentence_map.items():
            current_sentence_map.setdefault(source_lang, []).extend(sentences)
        processed_source_specs.append(normalized_spec)
        normalized_sentence_map = _normalize_sentence_map(
            current_sentence_map,
            seed=seed,
            max_sentences_per_lang=max_sentences_per_lang,
        )
        lang_counts = _write_instruction_cache_map(cache_dir, normalized_sentence_map)
        snapshot_total_before = existing_total_before + new_total_before
        snapshot_total_after = sum(lang_counts.values())
        snapshot_total_removed = max(0, snapshot_total_before - snapshot_total_after)
        snapshot_sources = processed_source_specs[:]
        write_json_atomic(
            cache_meta,
            {
                **expected_meta,
                "sources": snapshot_sources,
                "status": "building",
                "deduped": True,
                "total_before": snapshot_total_before,
                "total_after": snapshot_total_after,
                "total_removed": snapshot_total_removed,
                "lang_counts": lang_counts,
                "source_text_counts": source_text_counts,
                "source_extractor_counts": source_extractor_counts,
                "source_row_caps": {spec["name"]: int(spec.get("max_rows", 0) or 0) for spec in normalized_sources},
            },
        )
        tqdm.write(f"  Wrote instruction cache snapshot after {source_name}")

    normalized_sentence_map = _normalize_sentence_map(
        current_sentence_map,
        seed=seed,
        max_sentences_per_lang=max_sentences_per_lang,
    )
    lang_counts = _write_instruction_cache_map(cache_dir, normalized_sentence_map)
    total_sentences = sum(lang_counts.values())
    total_before = existing_total_before + new_total_before
    total_after = total_sentences
    total_removed = max(0, total_before - total_after)

    merged_sources = processed_source_specs
    if incremental_update and isinstance(existing_meta, dict):
        if isinstance(existing_meta.get("source_text_counts"), dict):
            merged_source_text_counts = dict(existing_meta["source_text_counts"])
            merged_source_text_counts.update(source_text_counts)
            source_text_counts = merged_source_text_counts
        if isinstance(existing_meta.get("source_extractor_counts"), dict):
            merged_source_extractor_counts = dict(existing_meta["source_extractor_counts"])
            merged_source_extractor_counts.update(source_extractor_counts)
            source_extractor_counts = merged_source_extractor_counts

    write_json_atomic(
        cache_meta,
        {
            **expected_meta,
            "sources": merged_sources,
            "status": "complete",
            "deduped": True,
            "total_before": total_before,
            "total_after": total_after,
            "total_removed": total_removed,
            "lang_counts": lang_counts,
            "source_text_counts": source_text_counts,
            "source_extractor_counts": source_extractor_counts,
            "source_row_caps": {spec["name"]: int(spec.get("max_rows", 0) or 0) for spec in normalized_sources},
        },
    )

    print(f"\nInstruction sentences cached -> {cache_dir}/")
    print(f"  {len(normalized_sentence_map)} languages | {total_sentences:,} sentences total")
    for lang in sorted(normalized_sentence_map):
        print(f"  {lang:<6}  {len(normalized_sentence_map[lang]):>5} sentences")
    return normalized_sentence_map
