from pathlib import Path

import pandas as pd
import streamlit as st

from contractlab.evaluation_chat import answer_metrics_question, prediction_markdown
from contractlab.inference import ClausePredictor
from contractlab.jobs import list_runs
from contractlab.legal_ai.chat_service import answer_evaluation_question, normalize_chat_history
from contractlab.ui import cached_legal_ai_orchestrator, page_intro

page_intro(
    "Evaluate",
    "Use held-out metrics as a map to individual errors—not as a substitute for reading them.",
)

completed = [run for run in list_runs() if run.get("state") == "completed" and run.get("summary")]
if not completed:
    st.info(
        "Complete a training run to unlock checkpoint evaluation. "
        "The concepts below remain available."
    )
    st.subheader("Why macro F1?")
    st.latex(
        r"F1_k = 2\frac{P_kR_k}{P_k+R_k}\qquad "
        r"Macro\ F1=\frac{1}{K}\sum_{k=1}^K F1_k"
    )
    st.write(
        "Macro F1 gives every clause label equal weight, so a common label cannot "
        "hide failure on a rare one."
    )
    st.stop()

selected_id = st.selectbox("Completed run", [run["run_id"] for run in completed])
run = next(item for item in completed if item["run_id"] == selected_id)
summary = run["summary"]
metrics = summary["test_metrics"]

with st.container(horizontal=True):
    st.metric("Test accuracy", f"{metrics['accuracy']:.3f}")
    st.metric("Macro precision", f"{metrics['macro_precision']:.3f}")
    st.metric("Macro recall", f"{metrics['macro_recall']:.3f}")
    st.metric("Macro F1", f"{metrics['macro_f1']:.3f}")

st.caption(
    f"Model: {summary['model_name']} · Test examples: {summary['split_rows']['test']} · "
    f"Data: {summary.get('data_source')} ({summary.get('data_license')})"
)

diagnostics = summary.get("overfitting_diagnostics", {})
release_gate = summary.get("release_gate", {})
if diagnostics.get("status") == "warning" or release_gate.get("passed") is False:
    st.error(
        "Potential overfitting detected; this checkpoint is not release-ready. "
        + " ".join(diagnostics.get("reasons", [])),
        icon=":material/warning:",
    )
elif diagnostics.get("status") == "passed":
    st.success(
        "Generalization gate passed: grouped split integrity and configured train/validation gap checks passed.",
        icon=":material/verified:",
    )
else:
    st.warning(
        "This run predates train/validation gap diagnostics. Retrain before treating it as release-ready.",
        icon=":material/history:",
    )

history_frame = pd.DataFrame(summary.get("history", []))
if {"epoch", "train_macro_f1", "validation_macro_f1"} <= set(history_frame.columns):
    st.subheader("Generalization across epochs")
    st.line_chart(history_frame.set_index("epoch")[["train_macro_f1", "validation_macro_f1"]])
    with st.container(horizontal=True):
        st.metric("Selected epoch", diagnostics.get("selected_epoch", summary.get("best_epoch")))
        st.metric("Train macro F1", f"{diagnostics.get('train_macro_f1', 0):.3f}")
        st.metric("Validation macro F1", f"{diagnostics.get('validation_macro_f1', 0):.3f}")
        st.metric("Generalization gap", f"{diagnostics.get('generalization_gap', 0):.3f}")

st.subheader("Per-label performance")
report = summary["classification_report"]
rows = []
for label in summary["label_names"]:
    values = report.get(label, {})
    rows.append(
        {
            "label": label,
            "precision": values.get("precision", 0),
            "recall": values.get("recall", 0),
            "f1": values.get("f1-score", 0),
            "support": values.get("support", 0),
        }
    )
st.dataframe(pd.DataFrame(rows).sort_values("f1"), hide_index=True)

st.subheader("Confusion matrix")
confusion = pd.DataFrame(summary["confusion_matrix"])
st.scatter_chart(confusion, x="predicted", y="actual", size="count", color="count")
st.caption(
    "Larger marks show more examples for an actual/predicted label pair. "
    "Read the errors below before changing the model."
)

prediction_path = Path(run["run_dir"]) / "test_predictions.csv"
if prediction_path.exists():
    predictions_frame = pd.read_csv(prediction_path)
    errors = predictions_frame[~predictions_frame["correct"].astype(bool)]
    st.subheader("Error analysis")
    if errors.empty:
        st.success(
            "No mistakes in this small held-out split. That is a reason to test harder "
            "data, not proof of production readiness."
        )
    else:
        actual_filter = st.multiselect("Actual labels", sorted(errors["actual"].unique()))
        visible = errors[errors["actual"].isin(actual_filter)] if actual_filter else errors
        st.dataframe(visible[["text", "actual", "predicted", "contract_id"]], hide_index=True)
    st.download_button(
        "Download held-out predictions",
        prediction_path.read_bytes(),
        f"{selected_id}-predictions.csv",
        "text/csv",
    )

with st.expander("Raw evaluation summary"):
    st.json(summary)

st.subheader("Chat with this evaluation")
orchestrator = cached_legal_ai_orchestrator()
st.caption(
    "Ask the configured local generative model about saved held-out evidence, or paste a clause "
    "to classify with the selected checkpoint and discuss its top-k hypotheses."
)
st.badge(
    f"Generative model: {orchestrator.settings.model}",
    icon=":material/smart_toy:",
    color="blue",
)
st.warning(
    "The chat analyzes saved held-out artifacts and checkpoint predictions. It does not "
    "generate legal advice, and its confidence scores are not calibrated.",
    icon=":material/science:",
)


@st.cache_resource(show_spinner=False)
def load_evaluation_predictor(checkpoint: str) -> ClausePredictor:
    return ClausePredictor(checkpoint)


if st.session_state.get("evaluation_chat_run_id") != selected_id:
    st.session_state.evaluation_chat_run_id = selected_id
    st.session_state.evaluation_chat_messages = [
        {
            "role": "assistant",
            "content": (
                f"I am connected to **{selected_id}**. Ask me to analyze why performance is "
                "weak, inspect a label, or paste a clause for a top-three classification."
            ),
        }
    ]

chat_messages = st.session_state.evaluation_chat_messages
with st.container(horizontal=True, horizontal_alignment="right"):
    if st.button("Clear evaluation chat", icon=":material/delete_sweep:"):
        st.session_state.evaluation_chat_messages = chat_messages[:1]
        st.rerun()

for message in chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("source_label"):
            st.caption(f"Response source: {message['source_label']}")
        if message.get("predictions"):
            st.dataframe(
                pd.DataFrame(message["predictions"]),
                hide_index=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="percent"
                    )
                },
            )

suggestions = {
    ":violet[:material/psychology:] Analyze this run": "Analyze why this run is performing poorly and how to improve it.",
    ":blue[:material/analytics:] What was macro F1?": "What was the macro F1?",
    ":orange[:material/troubleshoot:] Which label was weakest?": "Which label was weakest?",
    ":green[:material/article:] Classify a termination clause": (
        "Either party may terminate this Agreement for material breach if the breach remains "
        "uncured for thirty days after written notice."
    ),
}
pending_prompt = st.session_state.pop("evaluation_pending_prompt", None)
if not pending_prompt and not any(message["role"] == "user" for message in chat_messages):
    suggestion = st.pills(
        "Try asking",
        list(suggestions),
        label_visibility="collapsed",
        key="evaluation_chat_suggestion",
    )
    if suggestion:
        st.session_state.evaluation_pending_prompt = suggestions[suggestion]
        st.rerun()

typed_prompt = st.chat_input(
    "Ask about the run or paste a clause",
    key="evaluation_chat_input",
    submit_mode="disable",
)
prompt = pending_prompt or typed_prompt
if prompt:
    history = normalize_chat_history(st.session_state.evaluation_chat_messages)
    st.session_state.evaluation_chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    fallback_response = answer_metrics_question(prompt, summary)
    top_predictions = None
    chat_summary = summary
    with st.chat_message("assistant"):
        with st.status(
            f"Generating with {orchestrator.settings.model}...", expanded=False
        ) as status:
            if fallback_response is None:
                checkpoint = Path(run["run_dir"]) / "best"
                if not checkpoint.exists():
                    fallback_response = (
                        "The selected run has no saved best checkpoint to classify this text."
                    )
                else:
                    try:
                        status.update(label="Classifying with the selected checkpoint...")
                        top_predictions = load_evaluation_predictor(str(checkpoint)).predict_top_k(
                            [prompt], top_k=3
                        )[0]
                        fallback_response = prediction_markdown(top_predictions)
                    except Exception as exc:
                        fallback_response = f"The checkpoint could not classify this text: {exc}"
                chat_summary = {
                    **summary,
                    "ad_hoc_query": prompt,
                    "ad_hoc_predictions": top_predictions,
                }

            status.update(label=f"Generating with {orchestrator.settings.model}...")
            reply = answer_evaluation_question(
                orchestrator,
                summary=chat_summary,
                question=prompt,
                history=history,
                fallback_content=fallback_response,
            )
            if reply.used_local_fallback:
                status.update(
                    label="Generative model unavailable; local fallback used",
                    state="error",
                )
            else:
                status.update(label="Response ready", state="complete")

        st.markdown(reply.content)
        st.caption(f"Response source: {reply.source_label}")
        if top_predictions:
            st.dataframe(
                pd.DataFrame(top_predictions),
                hide_index=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="percent"
                    )
                },
            )
        if reply.used_local_fallback:
            st.warning(
                "Ollama could not complete this response, so the saved-evidence fallback "
                "answered instead.",
                icon=":material/offline_bolt:",
            )

    st.session_state.evaluation_chat_messages.append(
        {
            "role": "assistant",
            "content": reply.content,
            "predictions": top_predictions,
            "source_label": reply.source_label,
            "used_local_fallback": reply.used_local_fallback,
        }
    )
