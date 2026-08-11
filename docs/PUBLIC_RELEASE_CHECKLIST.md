# Private-first release checklist

Use this checklist before the first private GitHub or Hugging Face upload and repeat it before changing either repository to public.

## Repository safety

- [ ] Review every tracked file and the full Git history for contracts, personal data, credentials, and private paths.
- [ ] Confirm `.streamlit/secrets.toml`, `.env`, `artifacts/`, `wandb/`, local chat workspaces, checkpoints, and imported data are untracked.
- [ ] Run a secret scanner and inspect every finding manually.
- [ ] Inspect `git status --short --ignored`, the staged file list, and `git diff --cached`.
- [ ] Confirm included data is synthetic or redistributable with attribution.
- [ ] Confirm CUAD-derived files and their CC BY 4.0 attribution remain listed in `NOTICE.md`.
- [ ] Confirm the reviewed CUAD-derived instruction splits are intentionally Git LFS-tracked and attributed; private/local datasets and `data/curated/` remain untracked.
- [ ] Review third-party model and dataset licenses for the intended use.

## Quality

- [ ] Run `uv lock --check`.
- [ ] Run `uv run --extra cpu ruff check .`.
- [ ] Run `uv run --extra cpu pytest`.
- [ ] Run `uv build` and inspect the wheel and source archive contents.
- [ ] Compile the Streamlit pages and run AppTest smoke checks.
- [ ] Test `setup.ps1`, `run.ps1`, and `run-ai-enabled-contracts-lab.bat` on a clean Windows machine.
- [ ] Test the CPU profile and, separately, the CUDA 12.9 profile.
- [ ] Rebuild the Word guide, render every page, and inspect the page images.

## Private GitHub staging

- [ ] `gh auth status` shows `RamiJaloudi`.
- [ ] Repository name is exactly `RamiJaloudi/ai-enabled-contracts-lab`.
- [ ] Create it with `--private`; do not make it public in the initial command.
- [ ] Verify the `origin` URL before pushing.
- [ ] Confirm GitHub Actions uses least-privilege permissions.
- [ ] Enable Dependabot alerts, secret scanning, and private vulnerability reporting.
- [ ] Inspect the private repository from the web UI and perform a clean clone.

## Private Hugging Face staging

- [ ] `hf auth whoami` shows `rjaloudi`.
- [ ] Repository ID is exactly `rjaloudi/contra-contracts-bert-cuad-clause-classifier`.
- [ ] Prepare Contra v2 with `tools/prepare_huggingface_model.py`.
- [ ] Run clean local inference from the prepared directory.
- [ ] Confirm the bundle contains no datasets, contracts, tokens, chat files, W&B files, or private paths.
- [ ] Upload with `--push --private`.
- [ ] Inspect the rendered model card, files, license, metrics, limitations, and all 41 labels.
- [ ] Validate a clean authenticated download before changing visibility.

## Public promotion

- [ ] Obtain an independent review of repository contents and documentation.
- [ ] Verify version consistency in `pyproject.toml`, `CHANGELOG.md`, the guide, and release notes.
- [ ] Add repository topics such as `fine-tuning`, `legal-ai`, `pytorch`, and `streamlit`.
- [ ] Require CI and review on protected branches when collaborators join.
- [ ] Create a version tag only after CI passes.
- [ ] Repeat the legal-advice, privacy, data-lineage, and model-limitations notices in every release.
- [ ] Change visibility only as a separate, deliberate action after the private review is complete.