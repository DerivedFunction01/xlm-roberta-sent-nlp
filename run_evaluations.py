#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evaluation_run_config import load_shared_manifest


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / ".evaluation_config.json"
SCRIPT_BY_RUN = {
    "papluca": ROOT / "tests" / "test_papluca.py",
    "lid200": ROOT / "tests" / "test_lid200.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every active evaluation config from the shared manifest.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the shared evaluation manifest JSON.")
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv" / "bin" / "python",
        help="Python executable to use for the benchmark scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = load_shared_manifest(config_path=args.config)

    configs_by_id = {
        str(config["id"]): config
        for config in manifest["configs"]
        if str(config["id"]) in manifest["runs"]
    }

    if not configs_by_id:
        print(f"No active configs found in {args.config}")
        raise SystemExit(0)

    python_exe = args.python
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    for config_id in manifest["runs"]:
        config = configs_by_id.get(str(config_id))
        if config is None:
            raise ValueError(f"Config id '{config_id}' listed in runs but not found in configs")

        run_name = str(config.get("run", "")).strip().lower()
        script = SCRIPT_BY_RUN.get(run_name)
        if script is None:
            raise ValueError(f"Unsupported run type '{run_name}' for config '{config_id}'")

        cmd = [
            str(python_exe),
            str(script),
            "--config",
            str(args.config),
            "--config-id",
            str(config_id),
        ]
        print(f"\n=== Running {config_id} ({run_name}) ===")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
