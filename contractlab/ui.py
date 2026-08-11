from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from contractlab.data import synthetic_dataset
from contractlab.jobs import list_runs
from contractlab.legal_ai import LegalAnalysisOrchestrator
from contractlab.modeling import device_summary
from contractlab.palettes import load_palette_preference, palette_css
from contractlab.paths import IMPORTS_DIR, ensure_directories


def initialize_state() -> None:
    ensure_directories()
    defaults: dict[str, Any] = {
        "dataset_frame": synthetic_dataset(),
        "dataset_name": "AI-Enabled Contracts Lab synthetic clauses",
        "dataset_source": "AI-Enabled Contracts Lab original synthetic examples",
        "dataset_license": "MIT",
        "selected_run_dir": None,
        "completed_lessons": set(),
        "palette": load_palette_preference(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.html(palette_css(st.session_state.palette))


def persist_active_dataset() -> Path:
    ensure_directories()
    path = IMPORTS_DIR / "active_dataset.csv"
    st.session_state.dataset_frame.to_csv(path, index=False)
    return path


@st.cache_resource
def cached_device_summary() -> dict[str, Any]:
    return device_summary()


@st.cache_resource(show_spinner=False)
def cached_legal_ai_orchestrator() -> LegalAnalysisOrchestrator:
    return LegalAnalysisOrchestrator.from_settings()


def page_intro(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def sidebar_context() -> None:
    with st.sidebar:
        st.caption("Current learning context")
        st.write(f"**Dataset:** {st.session_state.dataset_name}")
        frame: pd.DataFrame = st.session_state.dataset_frame
        st.write(f"**Examples:** {len(frame):,}")
        st.write(f"**Labels:** {frame['label'].nunique() if 'label' in frame else 0}")
        device = cached_device_summary()
        if device.get("cuda_available"):
            st.success(f"{device.get('gpu')} · {device.get('vram_gb')} GB VRAM")
        else:
            st.warning("CUDA is unavailable in this Python environment.")
        st.caption("Educational software — not legal advice")


def completed_runs() -> list[dict[str, Any]]:
    return [run for run in list_runs() if run.get("state") == "completed"]


def source_excerpt(path: str | Path, start_marker: str | None = None, max_lines: int = 180) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if start_marker and start_marker in text:
        text = text[text.index(start_marker) :]
    return "\n".join(text.splitlines()[:max_lines])


def json_download(value: dict[str, Any]) -> bytes:
    return json.dumps(value, indent=2, default=str).encode("utf-8")
