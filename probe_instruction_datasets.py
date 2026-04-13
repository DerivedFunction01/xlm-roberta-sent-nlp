from __future__ import annotations

import argparse
import json
from itertools import islice
from typing import Any

from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
from tqdm.auto import tqdm


DEFAULT_DATASETS = [
    {"repo_id": "angeluriot/french_instruct", "label": "French"},
    {"repo_id": "DeepMount00/due-chiacchiere", "label": "Italian"},
    {"repo_id": "Cognitive-Lab/Aya_Hindi", "label": "Hindi"},
    {"repo_id": "IndonesiaAI/sft-dataset", "label": "Indonesian"},
    {"repo_id": "Iker/OpenHermes-2.5-English-Spanish", "label": "English/Spanish"},
    {"repo_id": "d0rj/OpenHermes-2.5-ru", "label": "Russian"},
    {"repo_id": "2A2I/Arabic-OpenHermes-2.5", "label": "Arabic"},
    {"repo_id": "nlp-with-deeplearning/ko.openhermes", "label": "Korean"},
    {"repo_id": "stefan-it/nanochat-german-openhermes", "label": "German"},
    {"repo_id": "yhavinga/Openhermes-2.5-dutch-97k", "label": "Dutch"},
    {"repo_id": "cnmoro/Instruct-PTBR-ENUS-11M", "label": "Portuguese/English"},
    {"repo_id": "Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k", "label": "Chinese"},
    {"repo_id": "umarigan/openhermes_tr", "label": "Turkish"},
    {"repo_id": "Aratako/Self-Instruct-Qwen2.5-72B-Instruct-60k", "label": "Japanese"},
]

PREFERRED_SPLITS = ("train", "validation", "test")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe instruction/chat datasets and print schema + sample rows."
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        help="Dataset repository ID to probe. Can be passed multiple times. Defaults to the built-in list.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="How many example rows to print per split.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=2,
        help="Maximum number of dataset configs to probe per repo.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Probe every available split instead of only the first preferred split.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow dataset repositories to execute custom loading code.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Maximum characters to print for any single string value.",
    )
    parser.add_argument(
        "--max-list-items",
        type=int,
        default=6,
        help="Maximum list items to show when summarizing nested values.",
    )
    return parser.parse_args()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def _simple_type(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _summarize_value(value: Any, *, max_chars: int, max_list_items: int, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        if depth >= 1:
            return f"list[{len(value)}]"
        return [
            _summarize_value(item, max_chars=max_chars, max_list_items=max_list_items, depth=depth + 1)
            for item in value[:max_list_items]
        ] + ([f"... +{len(value) - max_list_items} more"] if len(value) > max_list_items else [])
    if isinstance(value, dict):
        if depth >= 1:
            return {k: _simple_type(v) for k, v in list(value.items())[:max_list_items]}
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:max_list_items]:
            result[key] = _summarize_value(item, max_chars=max_chars, max_list_items=max_list_items, depth=depth + 1)
        if len(value) > max_list_items:
            result["..."] = f"+{len(value) - max_list_items} more keys"
        return result
    return _truncate(repr(value), max_chars)


def _collect_string_paths(value: Any, *, prefix: str = "", max_depth: int = 3) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if max_depth < 0:
        return paths
    if isinstance(value, str):
        paths.append((prefix or "<root>", _truncate(value, 120)))
        return paths
    if isinstance(value, list):
        for idx, item in enumerate(value[:5]):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            paths.extend(_collect_string_paths(item, prefix=child_prefix, max_depth=max_depth - 1))
        return paths
    if isinstance(value, dict):
        for key, item in list(value.items())[:12]:
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.extend(_collect_string_paths(item, prefix=child_prefix, max_depth=max_depth - 1))
        return paths
    return paths


def _choose_splits(repo_id: str, config_name: str | None, all_splits: bool) -> list[str]:
    try:
        builder_kwargs: dict[str, Any] = {}
        if config_name is not None:
            builder_kwargs["name"] = config_name
        builder = load_dataset_builder(repo_id, **builder_kwargs)
        split_info = getattr(getattr(builder, "info", None), "splits", None)
        split_names = list(split_info.keys()) if split_info else []
    except Exception as exc:
        print(f"    split discovery failed: {exc}")
        return ["train"]

    if not split_names:
        return ["train"]
    if all_splits:
        return split_names
    for preferred in PREFERRED_SPLITS:
        if preferred in split_names:
            return [preferred]
    return [split_names[0]]


def _iter_example_rows(dataset: Any, num_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in islice(dataset, num_examples):
        if isinstance(row, dict):
            rows.append(row)
        else:
            rows.append({"_row": row})
    return rows


def _probe_split(
    *,
    repo_id: str,
    config_name: str | None,
    split_name: str,
    num_examples: int,
    max_chars: int,
    max_list_items: int,
) -> None:
    config_label = config_name if config_name is not None else "<default>"
    print(f"  Config: {config_label}")
    print(f"  Split: {split_name}")
    try:
        load_kwargs: dict[str, Any] = {
            "split": split_name,
            "streaming": True,
        }
        if config_name is not None:
            load_kwargs["name"] = config_name
        dataset = load_dataset(
            repo_id,
            **load_kwargs,
        )
    except Exception as exc:
        print(f"    load failed: {exc}")
        return

    features = getattr(dataset, "features", None)
    if features is not None:
        print(f"    features: {features}")

    samples = _iter_example_rows(dataset, num_examples)
    if not samples:
        print("    no rows returned")
        return

    first = samples[0]
    print(f"    top-level keys: {sorted(first.keys())}")
    string_paths = _collect_string_paths(first)
    if string_paths:
        print("    string-like paths:")
        for path, preview in string_paths[:12]:
            print(f"      - {path}: {preview}")
    else:
        print("    string-like paths: <none detected>")

    for idx, row in enumerate(samples):
        print(f"    example {idx}:")
        payload = _summarize_value(row, max_chars=max_chars, max_list_items=max_list_items)
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def probe_dataset(
    *,
    repo_id: str,
    num_examples: int,
    max_configs: int,
    all_splits: bool,
    max_chars: int,
    max_list_items: int,
) -> None:
    label = next((item["label"] for item in DEFAULT_DATASETS if item["repo_id"] == repo_id), None)
    title = f"{repo_id}" if label is None else f"{repo_id} ({label})"
    print(f"\n=== {title} ===")
    try:
        configs = get_dataset_config_names(repo_id)
        if not configs:
            configs = [None]
    except Exception as exc:
        print(f"  config discovery failed: {exc}")
        configs = [None]

    print(f"  configs: {configs if configs != [None] else ['<default>']}")
    for config_name in tqdm(configs[:max_configs], desc=f"{repo_id} configs", leave=False):
        splits = _choose_splits(repo_id, config_name, all_splits)
        for split_name in splits:
            _probe_split(
                repo_id=repo_id,
                config_name=config_name,
                split_name=split_name,
                num_examples=num_examples,
                max_chars=max_chars,
                max_list_items=max_list_items,
            )


def main() -> None:
    args = _parse_args()
    repo_ids = args.repo_ids or [item["repo_id"] for item in DEFAULT_DATASETS]
    for repo_id in tqdm(repo_ids, desc="Datasets", leave=False):
        probe_dataset(
            repo_id=repo_id,
            num_examples=args.num_examples,
            max_configs=args.max_configs,
            all_splits=args.all_splits,
            max_chars=args.max_chars,
            max_list_items=args.max_list_items,
        )


if __name__ == "__main__":
    main()
