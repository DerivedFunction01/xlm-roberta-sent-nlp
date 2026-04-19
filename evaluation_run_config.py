from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "DerivedFunction/lang-ner-xlmr"
DEFAULT_TASK_TYPE = "token-classification"
VALID_TASK_TYPES = {"token-classification", "multi-label-classification"}


def _default_config(*, config_id: str, run_name: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": config_id,
        "run": run_name,
        "model_name": DEFAULT_MODEL_NAME,
        "task_type": DEFAULT_TASK_TYPE,
        "multilabel_runner_up_ratio": 0.9,
        "results_dir": f"evaluation_results/{config_id}",
    }
    if run_name == "papluca":
        base.update(
            {
                "sample_size": 2000,
                "batch_size": 32,
            }
        )
    elif run_name == "lid200":
        base.update(
            {
                "langs": [],
                "split": "test",
                "batch_size": 32,
            }
        )
    return base


def _default_manifest() -> dict[str, Any]:
    configs = [
        _default_config(config_id="papluca_default", run_name="papluca"),
        _default_config(config_id="lid200_default", run_name="lid200"),
    ]
    return {
        "runs": [cfg["id"] for cfg in configs],
        "configs": configs,
    }


def _normalize_manifest(config_path: Path, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")

    configs = raw.get("configs")
    runs = raw.get("runs")
    if not isinstance(configs, list) or not isinstance(runs, list):
        raise ValueError(f"Expected top-level 'configs' and 'runs' lists in {config_path}")

    normalized_configs: list[dict[str, Any]] = []
    for entry in configs:
        if not isinstance(entry, dict):
            continue
        config_id = str(entry.get("id", "")).strip()
        run_name = str(entry.get("run", "")).strip().lower()
        if not config_id or not run_name:
            continue
        normalized_configs.append(dict(entry, id=config_id, run=run_name))

    normalized_runs = [str(run_id).strip() for run_id in runs if str(run_id).strip()]
    return {
        "runs": normalized_runs,
        "configs": normalized_configs,
    }


def load_or_create_run_config(*, config_path: Path, run_name: str) -> dict[str, Any]:
    """Load the active config for one benchmark from a shared manifest."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(_default_manifest(), f, ensure_ascii=False, indent=2)
        print(f"Created default config at {config_path}")
        print("Edit the shared manifest, then rerun.")
        raise SystemExit(0)

    with config_path.open(encoding="utf-8") as f:
        manifest = _normalize_manifest(config_path, json.load(f))

    active_configs = {
        config["id"]: config
        for config in manifest["configs"]
        if config["id"] in manifest["runs"]
    }
    matches = [config for config in active_configs.values() if config["run"] == run_name]
    if not matches:
        print(f"No active config for '{run_name}' in {config_path}; skipping.")
        raise SystemExit(0)
    if len(matches) > 1:
        raise ValueError(
            f"Multiple active configs found for '{run_name}' in {config_path}: "
            f"{[config['id'] for config in matches]}"
        )

    config = matches[0]
    config_id = str(config["id"])
    model_name = str(config.get("model_name", "")).strip()
    if not model_name:
        raise ValueError(f"Missing 'model_name' in config '{config_id}' from {config_path}")

    task_type = str(config.get("task_type", "")).strip()
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Invalid or missing 'task_type' in config '{config_id}' from {config_path}: "
            f"expected one of {sorted(VALID_TASK_TYPES)}"
        )

    merged = _default_config(config_id=config_id, run_name=run_name)
    merged.update(config)
    merged["id"] = config_id
    merged["run"] = run_name
    merged["model_name"] = model_name
    merged["task_type"] = task_type
    return merged


def resolve_output_path(*, results_dir: Path, value: Any, default_name: str) -> Path:
    """Resolve an output path relative to a results directory."""
    candidate = Path(str(value).strip()) if value is not None and str(value).strip() else Path(default_name)
    if candidate.is_absolute():
        return candidate
    return results_dir / candidate
