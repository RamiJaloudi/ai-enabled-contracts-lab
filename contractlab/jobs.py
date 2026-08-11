from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from contractlab.config import TrainingConfig
from contractlab.paths import CONFIGS_DIR, RUNS_DIR, ensure_directories


def launch_training(config: TrainingConfig, wandb_api_key: str | None = None) -> dict[str, Any]:
    ensure_directories()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{config.safe_run_name}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config.output_dir = str(run_dir)
    config_path = CONFIGS_DIR / f"{run_id}.json"
    config.save(config_path)
    log_path = run_dir / "training.log"
    environment = os.environ.copy()
    if wandb_api_key:
        environment["WANDB_API_KEY"] = wandb_api_key
    command = [sys.executable, "-m", "contractlab.cli", "train", "--config", str(config_path)]
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            creationflags=creation_flags,
        )
    launch = {
        "run_id": run_id,
        "pid": process.pid,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "launched_at": time.time(),
    }
    (run_dir / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    return launch


def read_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    result: dict[str, Any] = {"run_dir": str(path), "run_id": path.name, "state": "unknown"}
    for filename, key in (
        ("status.json", "status"),
        ("summary.json", "summary"),
        ("launch.json", "launch"),
    ):
        candidate = path / filename
        if candidate.exists():
            try:
                result[key] = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result[key] = {}
    result["state"] = result.get("status", {}).get("state", "queued")
    log = path / "training.log"
    if log.exists():
        result["log_tail"] = "\n".join(
            log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        )
    return result


def list_runs() -> list[dict[str, Any]]:
    ensure_directories()
    return [read_run(path) for path in sorted(RUNS_DIR.iterdir(), reverse=True) if path.is_dir()]
