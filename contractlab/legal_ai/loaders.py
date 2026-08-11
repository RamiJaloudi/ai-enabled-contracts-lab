from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from contractlab.paths import PROJECT_ROOT


@lru_cache(maxsize=32)
def load_markdown(path: Path) -> str:
    if not path.exists() or path.suffix.casefold() != ".md":
        raise FileNotFoundError(f"Markdown instruction file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt(name: str, prompt_paths: dict[str, Path]) -> str:
    if name not in prompt_paths:
        raise KeyError(f"Unknown prompt: {name}")
    return load_markdown(prompt_paths[name])


def load_skill(name: str) -> str:
    path = PROJECT_ROOT / ".codex" / "skills" / name / "SKILL.md"
    return load_markdown(path)


def load_skills(names: list[str]) -> str:
    return "\n\n".join(load_skill(name) for name in names)
