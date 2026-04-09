from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from io_utils import write_json_atomic, write_sentence_parquet
from language import LANG_TO_GROUP
from paths import PATHS
from source_config import INSTRUCT
from text_utils import _collapse_spaces, _strip_bracket_notes


INSTRUCTION_CACHE_VERSION = 1
DEFAULT_MAX_SENTENCES_PER_LANG = INSTRUCT["max_lang"]
DEFAULT_SOURCE_SPECS = [
    {
        "name": "french_instruct",
        "repo_id": "angeluriot/french_instruct",
        "split": "train",
        "lang": "fr",
        "mode": "auto",
        "trust_remote_code": False,
    },
]

_CONVERSATION_KEYS = ("messages", "conversations", "conversation", "dialogue", "turns", "chat", "chatml")
_PAIR_FIELD_SETS = (
    ("instruction", "output"),
    ("instruction", "response"),
    ("instruction", "answer"),
    ("prompt", "completion"),
    ("prompt", "response"),
    ("question", "answer"),
    ("input", "output"),
    ("input", "response"),
)
_TEXT_FIELDS = (
    "text",
    "sentence",
    "content",
    "message",
    "utterance",
    "prompt",
    "instruction",
    "input",
    "question",
    "context",
    "answer",
    "response",
    "output",
    "completion",
    "completion_text",
    "chosen",
    "rejected",
)
_MESSAGE_CONTENT_FIELDS = ("content", "value", "text", "message", "utterance", "answer", "response")
_SKIP_FALLBACK_KEYS = {
    "id",
    "idx",
    "index",
    "lang",
    "language",
    "source",
    "dataset",
    "task",
    "category",
    "categories",
    "label",
    "labels",
    "score",
    "split",
    "source_dataset",
    "source_name",
    "source_id",
    "source_key",
    "provenance_key",
}
_HAS_WORD_OR_IDEOGRAPH = re.compile(r"\w", flags=re.UNICODE)


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
    return {
        "name": str(spec.get("name") or spec.get("repo_id") or "source"),
        "repo_id": str(spec["repo_id"]),
        "split": str(spec.get("split", "train")),
        "lang": str(spec.get("lang", "")),
        "mode": str(spec.get("mode", "auto")),
        "trust_remote_code": bool(spec.get("trust_remote_code", False)),
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


def _is_valid_instruction_text(text: str) -> bool:
    cleaned = _normalize_text(text)
    if len(cleaned) < 2 or len(cleaned) > 1_500:
        return False
    if not _HAS_WORD_OR_IDEOGRAPH.search(cleaned):
        return False
    digit_count = sum(ch.isdigit() for ch in cleaned)
    if digit_count > len(cleaned) * 0.5:
        return False
    return True


def _is_message_like(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in _MESSAGE_CONTENT_FIELDS)


def _extract_message_text(message: Any) -> list[str]:
    if isinstance(message, str):
        cleaned = _normalize_text(message)
        return [cleaned] if cleaned else []
    if isinstance(message, dict):
        for key in _MESSAGE_CONTENT_FIELDS:
            value = message.get(key)
            if isinstance(value, str):
                cleaned = _normalize_text(value)
                if cleaned:
                    return [cleaned]
        nested: list[str] = []
        for value in message.values():
            nested.extend(_extract_message_text(value))
        return nested
    return []


def _extract_conversation_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, list):
        for item in value:
            texts.extend(_extract_message_text(item))
    else:
        texts.extend(_extract_message_text(value))
    return texts


def _iter_fallback_strings(row: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for key, value in row.items():
        if key in _SKIP_FALLBACK_KEYS:
            continue
        if isinstance(value, str):
            cleaned = _normalize_text(value)
            if cleaned:
                yield key, cleaned
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, str):
                    cleaned = _normalize_text(item)
                    if cleaned:
                        yield f"{key}[{idx}]", cleaned
                elif _is_message_like(item):
                    for text in _extract_message_text(item):
                        yield f"{key}[{idx}]", text
        elif isinstance(value, dict):
            if _is_message_like(value):
                for text in _extract_message_text(value):
                    yield key, text
            else:
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        cleaned = _normalize_text(sub_value)
                        if cleaned:
                            yield f"{key}.{sub_key}", cleaned


def _extract_row_texts(row: dict[str, Any], *, mode: str = "auto") -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []

    if mode in {"auto", "conversation"}:
        for field in _CONVERSATION_KEYS:
            value = row.get(field)
            if value is not None:
                texts = _extract_conversation_texts(value)
                if texts:
                    return [(f"{field}[{idx}]", text) for idx, text in enumerate(texts)]

    if mode in {"auto", "paired"}:
        for left, right in _PAIR_FIELD_SETS:
            left_value = row.get(left)
            right_value = row.get(right)
            if isinstance(left_value, str) or isinstance(right_value, str):
                if isinstance(left_value, str):
                    cleaned = _normalize_text(left_value)
                    if cleaned:
                        records.append((left, cleaned))
                if isinstance(right_value, str):
                    cleaned = _normalize_text(right_value)
                    if cleaned:
                        records.append((right, cleaned))
                if records:
                    return records

    if mode in {"auto", "flat"}:
        for field in _TEXT_FIELDS:
            value = row.get(field)
            if isinstance(value, str):
                cleaned = _normalize_text(value)
                if cleaned:
                    records.append((field, cleaned))
        if records:
            return records

    fallback_records = list(_iter_fallback_strings(row))
    if fallback_records:
        return fallback_records
    return []


def _row_to_text_records(
    row: dict[str, Any],
    *,
    repo_id: str,
    lang: str,
    row_idx: int,
    mode: str = "auto",
) -> list[dict[str, str]]:
    extracted = _extract_row_texts(row, mode=mode)
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for item_idx, (field_path, text) in enumerate(extracted):
        cleaned = _normalize_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        records.append(
            {
                "lang": lang,
                "text": cleaned,
                "source_dataset": repo_id,
                "provenance_key": f"{repo_id}:{row_idx}:{field_path}:{item_idx}",
            }
        )
    return records


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
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[str, int]]:
    repo_id = str(spec["repo_id"])
    split = str(spec.get("split", "train"))
    lang = str(spec.get("lang", ""))
    mode = str(spec.get("mode", "auto"))
    trust_remote_code = bool(spec.get("trust_remote_code", False))

    records_by_lang: dict[str, list[dict[str, Any]]] = {}
    field_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    try:
        ds = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=trust_remote_code)
    except Exception as exc:
        tqdm.write(f"  Skipping {repo_id}: {exc}")
        return records_by_lang, dict(field_counts), dict(source_counts)

    for row_idx, row in enumerate(tqdm(ds, desc=repo_id, leave=False)):
        if not isinstance(row, dict):
            continue
        extracted = _row_to_text_records(row, repo_id=repo_id, lang=lang, row_idx=row_idx, mode=mode)
        if not extracted:
            continue
        for record in extracted:
            text = record["text"]
            if not _is_valid_instruction_text(text):
                continue
            records_by_lang.setdefault(lang, []).append(record)
            field_counts[record["provenance_key"].split(":")[-2]] += 1
            source_counts[repo_id] += 1

    return records_by_lang, dict(field_counts), dict(source_counts)


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
    if not force_rebuild and cache_state == "exact" and existing_meta is not None and existing_meta.get("status") == "complete":
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
        and existing_meta.get("status") == "complete"
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
    records_by_lang: dict[str, list[dict[str, Any]]] = {}
    source_field_counts: dict[str, dict[str, int]] = {}
    source_text_counts: dict[str, int] = {}
    for spec in tqdm(source_specs, desc="Instruction sources"):
        normalized_spec = _normalize_source_spec(spec)
        repo_id = normalized_spec["repo_id"]
        lang = normalized_spec["lang"]
        tqdm.write(f"  Reading {normalized_spec['name']} -> {lang}")
        source_records, field_counts, source_counts = _process_source_spec(spec=normalized_spec, lang_to_group=lang_to_group)
        source_field_counts[repo_id] = field_counts
        source_text_counts[repo_id] = int(sum(source_counts.values()))
        for source_lang, records in source_records.items():
            records_by_lang.setdefault(source_lang, []).extend(records)

    sentence_map = _records_to_sentence_map(records_by_lang)
    normalized_sentence_map = _normalize_sentence_map(sentence_map, seed=seed, max_sentences_per_lang=max_sentences_per_lang)
    if incremental_update:
        normalized_sentence_map = _merge_sentence_maps(base_result, normalized_sentence_map)
        normalized_sentence_map = _normalize_sentence_map(
            normalized_sentence_map,
            seed=seed,
            max_sentences_per_lang=max_sentences_per_lang,
        )

    lang_counts = _write_instruction_cache_map(cache_dir, normalized_sentence_map)
    total_sentences = sum(lang_counts.values())
    new_total_before = sum(len(sentences) for sentences in sentence_map.values())
    total_after = total_sentences
    total_before = new_total_before
    total_removed = max(0, total_before - total_after)

    merged_sources = normalized_sources[:]
    if incremental_update and isinstance(existing_meta, dict) and isinstance(existing_meta.get("sources"), list):
        merged_sources = list(existing_meta["sources"]) + [spec for spec in normalized_sources if _source_signature(spec) not in processed_signatures]
        base_total_after = sum(len(v) for v in base_result.values())
        total_before = int(existing_meta.get("total_before", base_total_after)) + new_total_before
        total_after = total_sentences
        total_removed = max(0, total_before - total_after)
        if isinstance(existing_meta.get("source_text_counts"), dict):
            merged_source_text_counts = dict(existing_meta["source_text_counts"])
            merged_source_text_counts.update(source_text_counts)
            source_text_counts = merged_source_text_counts
        if isinstance(existing_meta.get("source_field_counts"), dict):
            merged_source_field_counts = dict(existing_meta["source_field_counts"])
            merged_source_field_counts.update(source_field_counts)
            source_field_counts = merged_source_field_counts

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
            "source_field_counts": source_field_counts,
        },
    )

    print(f"\nInstruction sentences cached -> {cache_dir}/")
    print(f"  {len(normalized_sentence_map)} languages | {total_sentences:,} sentences total")
    for lang in sorted(normalized_sentence_map):
        print(f"  {lang:<6}  {len(normalized_sentence_map[lang]):>5} sentences")
    return normalized_sentence_map
