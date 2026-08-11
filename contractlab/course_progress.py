from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contractlab.paths import ARTIFACTS_DIR

COURSE_PROGRESS_PATH = ARTIFACTS_DIR / "course_progress.json"


def load_course_progress(path: Path = COURSE_PROGRESS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"completed": [], "notes": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"completed": [], "notes": {}}
    completed = payload.get("completed", [])
    notes = payload.get("notes", {})
    return {
        "completed": [str(item) for item in completed if isinstance(item, str)],
        "notes": {
            str(key): str(value)
            for key, value in notes.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if isinstance(notes, dict)
        else {},
    }


def save_course_progress(
    completed: set[str],
    notes: dict[str, str],
    path: Path = COURSE_PROGRESS_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": sorted(completed),
        "notes": {key: value for key, value in sorted(notes.items()) if value.strip()},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path