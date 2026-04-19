from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "DerivedFunction/lang-ner-xlmr"
DEFAULT_TASK_TYPE = "token-classification"
VALID_RUN_VALUES = {"papluca", "lid200", "both"}
VALID_TASK_TYPES = {"token-classification", "multi-label-classification"}


def _default_config(*, run_name: str) -> dict[str, Any]:
    if run_name == "papluca":
        return {
            "run": run_name,
            "model_name": DEFAULT_MODEL_NAME,
            "task_type": DEFAULT_TASK_TYPE,
            "sample_size": 2000,
            "batch_size": 32,
            "multilabel_runner_up_ratio": 0.9,
            "results_dir": f"evaluation_results/{run_name}",
        }
    if run_name == "lid200":
        return {
            "run": run_name,
            "model_name": DEFAULT_MODEL_NAME,
            "task_type": DEFAULT_TASK_TYPE,
            "langs": [],
            "split": "test",
            "batch_size": 32,
            "multilabel_runner_up_ratio": 0.9,
            "results_dir": f"evaluation_results/{run_name}",
        }
    return {
        "run": run_name,
        "model_name": DEFAULT_MODEL_NAME,
        "task_type": DEFAULT_TASK_TYPE,
        "results_dir": f"evaluation_results/{run_name}",
    }


def load_or_create_run_config(*, config_path: Path, run_name: str) -> dict[str, Any]:
    """Load a run configuration or create a default one and exit."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = _default_config(run_name=run_name)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"Created default config at {config_path}")
        print("Edit it to choose the model checkpoint, task type, and run target, then rerun.")
        raise SystemExit(0)

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")

    config_run = str(config.get("run", "")).strip().lower()
    if config_run not in VALID_RUN_VALUES:
        raise ValueError(
            f"Invalid or missing 'run' in {config_path}: expected one of {sorted(VALID_RUN_VALUES)}"
        )
    if config_run not in {run_name, "both"}:
        print(f"Config run is '{config_run}', so skipping {run_name}.")
        raise SystemExit(0)

    model_name = str(config.get("model_name", "")).strip()
    if not model_name:
        raise ValueError(f"Missing 'model_name' in {config_path}")

    task_type = str(config.get("task_type", "")).strip()
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Invalid or missing 'task_type' in {config_path}: expected one of {sorted(VALID_TASK_TYPES)}"
        )

    merged = _default_config(run_name=run_name)
    merged.update(config)
    merged["run"] = config_run
    merged["model_name"] = model_name
    merged["task_type"] = task_type
    return merged


def resolve_output_path(*, results_dir: Path, value: Any, default_name: str) -> Path:
    """Resolve an output path relative to a results directory."""
    candidate = Path(str(value).strip()) if value is not None and str(value).strip() else Path(default_name)
    if candidate.is_absolute():
        return candidate
    return results_dir / candidate
