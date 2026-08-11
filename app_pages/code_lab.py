from __future__ import annotations

import nbformat
import streamlit as st

from contractlab.codegen import generated_notebook, generated_script
from contractlab.config import TrainingConfig
from contractlab.modeling import MODEL_PRESETS
from contractlab.paths import PROJECT_ROOT
from contractlab.ui import initialize_state, page_intro, persist_active_dataset, source_excerpt

GITHUB_REPOSITORY = "RamiJaloudi/ai-enabled-contracts-lab"
HF_REPOSITORY = "rjaloudi/contra-contracts-bert-cuad-clause-classifier"

initialize_state()
page_intro(
    "Code lab",
    "Inspect the implementation, export reproducible experiments, and prepare reviewed private releases for GitHub and Hugging Face.",
)

st.caption(
    "Code Lab generates commands and artifacts but never stores access tokens. Authenticate with "
    "the official CLIs in your terminal and review every staged file before publishing."
)

source_file = st.selectbox(
    "Implementation file",
    [
        "streamlit_app.py",
        "contractlab/training.py",
        "contractlab/data.py",
        "contractlab/evaluation.py",
        "contractlab/assistant.py",
        "contractlab/legal_ai/orchestrator.py",
        "contractlab/legal_ai/chat_service.py",
        "contractlab/contract_chat_store.py",
        "tools/prepare_huggingface_model.py",
        "contractlab/jobs.py",
    ],
)
st.code(
    source_excerpt(PROJECT_ROOT / source_file, max_lines=1000), language="python", line_numbers=True
)

experiment_tab, github_tab, hugging_face_tab = st.tabs(
    ["Experiment export", "Private GitHub release", "Private Hugging Face release"]
)

with experiment_tab:
    st.subheader("Generate a reproducible experiment")
    with st.form("codegen_config", border=True):
        run_name = st.text_input("Run name", "exported-contract-experiment")
        selected_preset = st.selectbox("Model", [preset.name for preset in MODEL_PRESETS])
        epochs = st.number_input("Epochs", 1, 20, 3)
        learning_rate = st.number_input(
            "Learning rate", min_value=0.000001, max_value=0.001, value=0.00002, format="%.6f"
        )
        batch = st.selectbox("Device batch size", [2, 4, 8, 16], index=2)
        accumulation = st.selectbox("Gradient accumulation", [1, 2, 4, 8], index=1)
        max_length = st.selectbox("Maximum tokens", [64, 128, 256, 384, 512], index=2)
        generate = st.form_submit_button("Generate code", type="primary", icon=":material/code:")

    if generate or "generated_config" not in st.session_state:
        data_path = persist_active_dataset()
        model_id = next(preset.model_id for preset in MODEL_PRESETS if preset.name == selected_preset)
        st.session_state.generated_config = TrainingConfig(
            run_name=run_name,
            dataset_path=str(data_path),
            output_dir=str(PROJECT_ROOT / "artifacts" / "runs" / run_name),
            model_name=model_id,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            train_batch_size=int(batch),
            eval_batch_size=max(8, int(batch)),
            gradient_accumulation_steps=int(accumulation),
            max_length=int(max_length),
            data_source=st.session_state.dataset_source,
            data_license=st.session_state.dataset_license,
        )

    config = st.session_state.generated_config
    script = generated_script(config)
    notebook = generated_notebook(config)
    notebook_bytes = nbformat.writes(notebook).encode("utf-8")

    st.subheader("Generated script")
    edited_script = st.text_area("Editable Python", value=script, height=520)
    with st.container(horizontal=True):
        st.download_button(
            "Download Python script",
            edited_script.encode("utf-8"),
            f"{config.safe_run_name}.py",
            "text/x-python",
            icon=":material/download:",
        )
        st.download_button(
            "Download Jupyter notebook",
            notebook_bytes,
            f"{config.safe_run_name}.ipynb",
            "application/x-ipynb+json",
            icon=":material/download:",
        )
        st.download_button(
            "Download configuration",
            config.save(
                PROJECT_ROOT / "artifacts" / "exports" / f"{config.safe_run_name}.json"
            ).read_bytes(),
            f"{config.safe_run_name}.json",
            "application/json",
            icon=":material/download:",
        )
    st.caption(
        "The generated script calls the same package used by the UI. The handwritten forward, "
        "backward, optimizer, scheduler, checkpoint, and evaluation loop remains visible above."
    )

with github_tab:
    st.subheader("Move an existing project to a private GitHub repository")
    st.markdown(
        "GitHub should contain source, tests, small redistributable samples, documentation, and "
        "configuration—not local contracts, credentials, W&B logs, checkpoints, or generated artifacts."
    )
    st.code(
        f"""# Review before staging
git status --short
git check-ignore -v artifacts/ .env .streamlit/secrets.toml
uv run --extra cpu ruff check .
uv run --extra cpu pytest
uv build

# First private push of an existing local repository
git add <reviewed-files>
git status --short
git diff --cached
git commit -m "Initial private release"
gh repo create {GITHUB_REPOSITORY} --private --source . --remote origin --push

# Verify the destination
git remote -v
gh repo view {GITHUB_REPOSITORY} --web""",
        language="powershell",
    )
    st.warning(
        "Never run `git add -A` until ignored paths and every staged file have been reviewed. "
        "Removing a secret from the latest commit does not remove it from Git history.",
        icon=":material/security:",
    )
    checklist = PROJECT_ROOT / "docs" / "PUBLIC_RELEASE_CHECKLIST.md"
    if checklist.exists():
        st.download_button(
            "Download the GitHub release checklist",
            checklist.read_bytes(),
            checklist.name,
            "text/markdown",
            icon=":material/download:",
        )

with hugging_face_tab:
    st.subheader("Prepare Contra v2 for a private Hugging Face model repository")
    st.markdown(
        "Contra v2 is the 41-label Contracts-BERT classifier. Its held-out macro-F1 is **0.6036** "
        "on the project’s contract-grouped CUAD derivative. It is an educational alpha, not a "
        "whole-contract reasoner or legal authority source."
    )
    st.code(
        f"""# Authenticate outside the app; the token must never enter source control
hf auth login
hf auth whoami

# Rebuild and inspect the upload folder
.\\.venv\\Scripts\\python.exe tools\\prepare_huggingface_model.py `
  --repo-id {HF_REPOSITORY}

# Run a clean local inference smoke test, then upload privately
.\\.venv\\Scripts\\python.exe tools\\prepare_huggingface_model.py `
  --repo-id {HF_REPOSITORY} `
  --push --private""",
        language="powershell",
    )
    bundle = PROJECT_ROOT / "artifacts" / "huggingface" / HF_REPOSITORY.split("/", 1)[1]
    if bundle.exists():
        files = [path for path in bundle.iterdir() if path.is_file()]
        total_mb = sum(path.stat().st_size for path in files) / (1024 * 1024)
        st.success(
            f"Prepared locally: {len(files)} files, {total_mb:.1f} MB. The artifacts directory is Git-ignored.",
            icon=":material/check_circle:",
        )
    st.markdown(
        "The model card records the base model, data lineage, grouped split, per-label limitations, "
        "evaluation results, intended use, and CC BY-SA 4.0 share-alike requirement."
    )
    with st.container(horizontal=True):
        for relative, label in [
            ("docs/HUGGING_FACE_RELEASE.md", "Download the Hugging Face guide"),
            ("docs/models/CONTRA_MODEL_CARD.md", "Download the model card"),
        ]:
            path = PROJECT_ROOT / relative
            if path.exists():
                st.download_button(
                    label,
                    path.read_bytes(),
                    path.name,
                    "text/markdown",
                    icon=":material/download:",
                    key=f"download_{path.name}",
                )

st.info(
    "Recommended promotion path: local validation → private GitHub/model repositories → rendered "
    "documentation review → clean-clone test → public release with limitations and licenses intact.",
    icon=":material/verified_user:",
)