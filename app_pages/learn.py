from __future__ import annotations

import streamlit as st

from contractlab.course_progress import load_course_progress, save_course_progress
from contractlab.curriculum import LESSONS
from contractlab.paths import PROJECT_ROOT
from contractlab.ui import initialize_state, page_intro, source_excerpt

initialize_state()

if not st.session_state.get("course_progress_loaded"):
    saved_progress = load_course_progress()
    st.session_state.completed_lessons = set(saved_progress["completed"])
    st.session_state.course_notes = dict(saved_progress["notes"])
    st.session_state.course_progress_loaded = True

pending_slug = st.session_state.pop("course_pending_slug", None)
valid_slugs = {lesson.slug for lesson in LESSONS}
if pending_slug in valid_slugs:
    st.session_state.course_selected_slug = pending_slug
if st.session_state.get("course_selected_slug") not in valid_slugs:
    st.session_state.course_selected_slug = LESSONS[0].slug

page_intro(
    "Guided classifier course",
    "Ten complete lessons connect task design, data, training, evaluation, and responsible hybrid legal-AI integration.",
)

completed = set(st.session_state.completed_lessons) & valid_slugs
progress = len(completed) / len(LESSONS)
metric_cols = st.columns(3)
metric_cols[0].metric("Lessons completed", f"{len(completed)} / {len(LESSONS)}")
metric_cols[1].metric("Course progress", f"{progress:.0%}")
metric_cols[2].metric("Saved locally", "Yes")
st.progress(progress, text=f"{len(completed)} of {len(LESSONS)} lessons complete")

with st.expander("Course outline", expanded=False):
    for outline_lesson in LESSONS:
        status = "✓" if outline_lesson.slug in completed else "○"
        st.markdown(f"{status} **{outline_lesson.title}** — {outline_lesson.outcome}")

selected_slug = st.selectbox(
    "Current lesson",
    options=[lesson.slug for lesson in LESSONS],
    format_func=lambda slug: next(lesson.title for lesson in LESSONS if lesson.slug == slug),
    key="course_selected_slug",
)
selected_index = next(index for index, item in enumerate(LESSONS) if item.slug == selected_slug)
lesson = LESSONS[selected_index]

st.caption(f"Lesson {selected_index + 1} of {len(LESSONS)}")
st.header(lesson.title)
st.markdown(f"**Learning outcome:** {lesson.outcome}")

learn_tab, code_tab, practice_tab = st.tabs(
    ["Learn the concept", "Inspect the implementation", "Practice and self-check"]
)

with learn_tab:
    st.subheader("Why this matters")
    st.write(lesson.overview)

    st.subheader("Core concepts")
    concept_cols = st.columns(2)
    for index, concept in enumerate(lesson.concepts):
        concept_cols[index % 2].markdown(f"- {concept}")

    st.subheader("Key relationship")
    st.markdown(lesson.mathematics)

    st.subheader("Workflow")
    for step_number, step in enumerate(lesson.steps, start=1):
        st.markdown(f"{step_number}. {step}")

with code_tab:
    file_part = lesson.code_focus.split(":", 1)[0]
    source_path = PROJECT_ROOT / file_part
    marker = lesson.code_focus.split(":", 1)[1] if ":" in lesson.code_focus else None
    st.caption(f"Source: {lesson.code_focus}")
    if source_path.exists():
        st.code(
            source_excerpt(source_path, marker, max_lines=180),
            language="python",
            line_numbers=True,
        )
    else:
        st.error(f"The lesson source file is missing: {file_part}")
    st.info(
        "Read until you can connect the lesson vocabulary to concrete variables, control flow, "
        "saved artifacts, and failure handling in this implementation.",
        icon=":material/code:",
    )

with practice_tab:
    st.subheader("Guided activity")
    st.info(lesson.experiment, icon=":material/science:")

    st.subheader("Completion evidence")
    st.write(lesson.deliverable)

    st.subheader("Check your understanding")
    for question_number, question in enumerate(lesson.checks, start=1):
        st.markdown(f"{question_number}. {question}")

    note_key = f"course_note_{lesson.slug}"
    if note_key not in st.session_state:
        st.session_state[note_key] = st.session_state.course_notes.get(lesson.slug, "")
    notes = st.text_area(
        "Experiment notes",
        key=note_key,
        height=180,
        placeholder="Record your prediction, evidence, observation, and conclusion.",
    )
    if notes != st.session_state.course_notes.get(lesson.slug, ""):
        st.session_state.course_notes[lesson.slug] = notes
        save_course_progress(completed, st.session_state.course_notes)
        st.caption("Notes saved locally.")

navigation_cols = st.columns([1, 2, 1])
with navigation_cols[0]:
    if st.button(
        "Previous lesson",
        disabled=selected_index == 0,
        use_container_width=True,
        icon=":material/arrow_back:",
    ):
        st.session_state.course_pending_slug = LESSONS[selected_index - 1].slug
        st.rerun()

with navigation_cols[1]:
    is_complete = lesson.slug in completed
    completion_label = "Mark lesson incomplete" if is_complete else "Mark lesson complete"
    if st.button(
        completion_label,
        type="secondary" if is_complete else "primary",
        use_container_width=True,
        icon=":material/check_circle:" if not is_complete else ":material/replay:",
    ):
        updated = set(completed)
        if is_complete:
            updated.discard(lesson.slug)
        else:
            updated.add(lesson.slug)
        st.session_state.completed_lessons = updated
        save_course_progress(updated, st.session_state.course_notes)
        st.rerun()

with navigation_cols[2]:
    if st.button(
        "Next lesson",
        disabled=selected_index == len(LESSONS) - 1,
        use_container_width=True,
        icon=":material/arrow_forward:",
    ):
        st.session_state.course_pending_slug = LESSONS[selected_index + 1].slug
        st.rerun()

notes_lines = ["# Guided classifier course notes", ""]
for course_lesson in LESSONS:
    notes_lines.append(f"## {course_lesson.title}")
    lesson_note = st.session_state.course_notes.get(course_lesson.slug, "").strip()
    notes_lines.append(lesson_note or "_No notes recorded._")
    notes_lines.append("")
st.download_button(
    "Download all course notes",
    "\n".join(notes_lines).encode("utf-8"),
    "guided-classifier-course-notes.md",
    "text/markdown",
    icon=":material/download:",
)

if len(completed) == len(LESSONS):
    st.success(
        "Course complete. Continue with a held-out evaluation and trace one request through the hybrid pipeline.",
        icon=":material/school:",
    )
    next_cols = st.columns(2)
    next_cols[0].page_link(
        "app_pages/evaluate.py",
        label="Open Evaluate",
        icon=":material/analytics:",
        use_container_width=True,
    )
    next_cols[1].page_link(
        "app_pages/contract_chat.py",
        label="Open Contract Chat",
        icon=":material/chat:",
        use_container_width=True,
    )

st.caption(
    "Progress and notes are saved to `artifacts/course_progress.json`, which is excluded from Git."
)