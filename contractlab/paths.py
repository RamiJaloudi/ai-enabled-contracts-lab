from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CONFIGS_DIR = ARTIFACTS_DIR / "configs"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"
IMPORTS_DIR = DATA_DIR / "imports"
CURATED_DIR = DATA_DIR / "curated"


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        ARTIFACTS_DIR,
        RUNS_DIR,
        CONFIGS_DIR,
        EXPORTS_DIR,
        IMPORTS_DIR,
        CURATED_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
