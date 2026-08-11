# Contributing to AI-Enabled Contracts Lab

Thank you for helping make fine-tuning easier to learn. AI-Enabled Contracts Lab welcomes
documentation, tests, bug fixes, teaching improvements, and carefully scoped
features.

## Development setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and choose one
PyTorch profile.

Windows with a compatible NVIDIA GPU:

```powershell
uv sync --python 3.12 --extra dev --extra cu129
uv run --extra cu129 contractlab doctor
```

CPU-only development and CI:

```powershell
uv sync --python 3.12 --extra dev --extra cpu
uv run --extra cpu contractlab doctor
```

The `cpu` and `cu129` extras conflict by design. Use the same selected extra on
every `uv run` command so uv preserves the intended PyTorch build.

## Before opening a pull request

```powershell
uv run --extra cpu ruff check .
uv run --extra cpu pytest
uv build
```

Use `--extra cu129` instead when testing the CUDA profile. Keep pull requests
focused, add or update tests, and explain user-visible behavior changes. Update
`CHANGELOG.md` under `Unreleased` when appropriate.

## Data and model contributions

Do not commit contracts, credentials, API keys, W&B files, model checkpoints,
or datasets unless redistribution is clearly permitted. Synthetic examples
must not be derived from confidential agreements. Record the source, license,
and transformation process for any proposed public dataset.

## Code style

- Target Python 3.12 and keep lines at or below 100 characters where practical.
- Prefer typed, testable functions in `contractlab/` over logic embedded in UI pages.
- Keep Streamlit pages educational: show what the code does and why it matters.
- Preserve local-first privacy defaults and explicit opt-in for external services.

## Pull requests

By submitting a contribution, you agree that it may be distributed under the
MIT License. Please use the pull request template and respond to review comments
with either a change or a brief technical rationale.
