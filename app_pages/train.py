import os
from pathlib import Path

import streamlit as st

from contractlab.config import TrainingConfig
from contractlab.jobs import launch_training, read_run
from contractlab.modeling import MODEL_PRESETS, device_summary
from contractlab.paths import RUNS_DIR
from contractlab.ui import initialize_state, page_intro, persist_active_dataset

initialize_state()
page_intro(
    "Train a clause classifier",
    "Configure a full Transformer fine-tuning run. Training launches in a separate local process.",
)

device = device_summary()
if not device.get("cuda_available"):
    st.warning(
        "CUDA is unavailable in this environment. Use one epoch and a small batch for a CPU learning run."
    )

with st.form("training_configuration", border=True):
    st.subheader("Experiment configuration")
    run_name = st.text_input("Run name", value="distilbert-contract-clauses")
    selected_preset = st.selectbox("Pretrained model", [preset.name for preset in MODEL_PRESETS])
    model_name = next(preset.model_id for preset in MODEL_PRESETS if preset.name == selected_preset)
    with st.container(horizontal=True):
        epochs = st.number_input("Epochs", 1, 20, 3)
        batch_size = st.selectbox("Device batch size", [2, 4, 8, 16], index=2)
        accumulation = st.selectbox("Gradient accumulation", [1, 2, 4, 8], index=1)
        max_length = st.selectbox("Maximum tokens", [64, 128, 256, 384, 512], index=2)
    with st.container(horizontal=True):
        learning_rate = st.number_input(
            "Learning rate", min_value=0.000001, max_value=0.001, value=0.00002, format="%.6f"
        )
        weight_decay = st.number_input(
            "Weight decay", min_value=0.0, max_value=1.0, value=0.01, format="%.3f"
        )
        seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)
        precision = st.selectbox("Precision", ["auto", "bf16", "fp16", "fp32"])
    freeze_encoder = st.checkbox("Freeze pretrained encoder (comparison exercise)")

    st.subheader("Class imbalance")
    loss_label = st.selectbox(
        "Training loss",
        ["Standard cross-entropy", "Class-weighted cross-entropy", "Class-weighted focal loss"],
        index=1,
        help="Weighted options give rare labels more influence. Compare them with the standard baseline.",
    )
    loss_function = {
        "Standard cross-entropy": "cross_entropy",
        "Class-weighted cross-entropy": "class_weighted",
        "Class-weighted focal loss": "focal",
    }[loss_label]
    focal_gamma = st.number_input(
        "Focal gamma", 0.0, 5.0, 2.0, 0.25, disabled=loss_function != "focal"
    )
    st.caption("Weights use clipped inverse-square-root label frequency from the training split.")

    st.subheader("Generalization safeguards")
    with st.container(horizontal=True):
        early_stopping_patience = st.number_input(
            "Early-stopping patience",
            min_value=0,
            max_value=10,
            value=2,
            help="Stop after this many epochs without a meaningful validation macro-F1 gain. Zero disables early stopping.",
        )
        early_stopping_min_delta = st.number_input(
            "Minimum validation gain",
            min_value=0.0,
            max_value=0.2,
            value=0.001,
            step=0.001,
            format="%.3f",
        )
        overfitting_warning_gap = st.number_input(
            "Train/validation F1 gap limit",
            min_value=0.01,
            max_value=0.9,
            value=0.15,
            step=0.01,
            format="%.2f",
        )
        overfitting_loss_ratio = st.number_input(
            "Validation/train loss ratio limit",
            min_value=1.01,
            max_value=10.0,
            value=1.5,
            step=0.05,
            format="%.2f",
        )
    st.caption(
        "The test set remains sealed until the best validation checkpoint is selected. A failed "
        "generalization gate keeps the checkpoint available for diagnosis but marks it not release-ready."
    )

    st.subheader("Weights & Biases")
    wandb_mode = st.segmented_control(
        "Tracking mode", ["disabled", "offline", "online"], default="disabled"
    )
    wandb_project = st.text_input(
        "W&B project",
        value="ai-enabled-contracts-lab",
        disabled=wandb_mode == "disabled",
    )
    secret_default = ""
    try:
        secret_default = st.secrets.get("wandb_api_key", "")
    except Exception:
        pass
    wandb_key = st.text_input(
        "W&B API key (held only for this process)",
        value=secret_default,
        type="password",
        disabled=wandb_mode != "online",
    )
    st.caption(
        "Only aggregate configuration and metrics are logged. Contract text and predictions are excluded."
    )
    submitted = st.form_submit_button(
        "Launch training", type="primary", icon=":material/play_arrow:"
    )

if submitted:
    dataset_path = persist_active_dataset()
    config = TrainingConfig(
        run_name=run_name,
        dataset_path=str(dataset_path),
        model_name=model_name,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        train_batch_size=int(batch_size),
        eval_batch_size=max(8, int(batch_size)),
        gradient_accumulation_steps=int(accumulation),
        max_length=int(max_length),
        weight_decay=float(weight_decay),
        early_stopping_patience=int(early_stopping_patience),
        early_stopping_min_delta=float(early_stopping_min_delta),
        overfitting_warning_gap=float(overfitting_warning_gap),
        overfitting_loss_ratio=float(overfitting_loss_ratio),
        seed=int(seed),
        precision=precision,
        loss_function=loss_function,
        focal_gamma=float(focal_gamma),
        freeze_encoder=freeze_encoder,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        data_license=st.session_state.dataset_license,
        data_source=st.session_state.dataset_source,
    )
    errors = config.validate()
    if errors:
        for error in errors:
            st.error(error)
    elif wandb_mode == "online" and not (wandb_key or os.environ.get("WANDB_API_KEY")):
        st.error(
            "Online W&B tracking requires an API key in the form, secrets file, or WANDB_API_KEY environment variable."
        )
    else:
        try:
            launch = launch_training(config, wandb_key or None)
            st.session_state.selected_run_dir = launch["run_dir"]
            st.success(f"Launched {launch['run_id']} as process {launch['pid']}.")
        except Exception as exc:
            st.error(f"Could not launch training: {exc}")

run_directories = sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()], reverse=True)
if run_directories:
    st.subheader("Live run monitor")
    default_index = 0
    if st.session_state.selected_run_dir:
        selected_path = Path(st.session_state.selected_run_dir)
        if selected_path in run_directories:
            default_index = run_directories.index(selected_path)
    selected = st.selectbox(
        "Run", run_directories, index=default_index, format_func=lambda path: path.name
    )
    st.session_state.selected_run_dir = str(selected)

    @st.fragment(run_every="2s")
    def monitor_run(run_path: Path) -> None:
        run = read_run(run_path)
        status = run.get("status", {})
        state = run.get("state", "queued")
        message = status.get("message", "Waiting for the trainer to report status…")
        if state == "completed":
            st.success(message)
        elif state == "failed":
            st.error(message)
        else:
            st.info(f"{state.title()}: {message}")
        if status.get("epochs"):
            epoch_fraction = (
                float(status.get("epoch", 1))
                - 1
                + float(status.get("batch", 0)) / max(1, float(status.get("batches", 1)))
            ) / float(status["epochs"])
            st.progress(
                min(max(epoch_fraction, 0.0), 1.0),
                text=f"Epoch {status.get('epoch')}/{status.get('epochs')}",
            )
        if run.get("log_tail"):
            with st.expander("Training log"):
                st.code(run["log_tail"], language="text")
        diagnostics = status.get("overfitting_diagnostics")
        if diagnostics and diagnostics.get("status") == "warning":
            st.error("Potential overfitting detected: " + " ".join(diagnostics.get("reasons", [])))
        elif diagnostics and diagnostics.get("status") == "passed":
            st.success("Current train/validation generalization checks pass.")

    monitor_run(selected)
else:
    st.info("No local training runs yet.")
