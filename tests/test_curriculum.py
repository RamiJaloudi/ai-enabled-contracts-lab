from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from contractlab.course_progress import load_course_progress, save_course_progress
from contractlab.curriculum import LESSONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_curriculum_has_ten_complete_unique_lessons() -> None:
    assert len(LESSONS) == 10
    assert len({lesson.slug for lesson in LESSONS}) == len(LESSONS)
    for lesson in LESSONS:
        assert lesson.title
        assert lesson.outcome
        assert lesson.overview
        assert len(lesson.concepts) >= 4
        assert lesson.mathematics
        assert lesson.experiment
        assert len(lesson.steps) >= 4
        assert len(lesson.checks) >= 2
        assert lesson.deliverable


def test_every_lesson_source_and_marker_resolves() -> None:
    for lesson in LESSONS:
        relative, _, marker = lesson.code_focus.partition(":")
        source = PROJECT_ROOT / relative
        assert source.is_file(), lesson.code_focus
        if marker:
            assert marker in source.read_text(encoding="utf-8"), lesson.code_focus


def test_course_progress_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "course_progress.json"
    save_course_progress(
        {"task", "data"},
        {"task": "Defined the prediction unit.", "data": "Confirmed zero overlap."},
        path,
    )
    assert load_course_progress(path) == {
        "completed": ["data", "task"],
        "notes": {
            "data": "Confirmed zero overlap.",
            "task": "Defined the prediction unit.",
        },
    }


def test_corrupt_course_progress_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "course_progress.json"
    path.write_text("not json", encoding="utf-8")
    assert load_course_progress(path) == {"completed": [], "notes": {}}


def test_guided_course_page_smoke() -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app_pages" / "learn.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Guided classifier course"
    assert len(app.selectbox) == 1
    assert len(app.tabs) == 3
    assert any("Lessons completed" in metric.label for metric in app.metric)