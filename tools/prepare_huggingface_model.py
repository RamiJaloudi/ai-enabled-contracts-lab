"""Create an upload-ready Hugging Face bundle for a reviewed Contra checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_DEFAULTS = {
    "v1": {
        "run_dir": PROJECT_ROOT / "artifacts" / "runs" / "contra-distilbert-cuad-v1",
        "output": PROJECT_ROOT / "artifacts" / "huggingface" / "contra-distilbert-cuad-clause-classifier",
        "repo_name": "contra-distilbert-cuad-clause-classifier",
    },
    "v2": {
        "run_dir": PROJECT_ROOT / "artifacts" / "runs" / "contra-contracts-bert-weighted-v2",
        "output": PROJECT_ROOT / "artifacts" / "huggingface" / "contra-contracts-bert-cuad-clause-classifier",
        "repo_name": "contra-contracts-bert-cuad-clause-classifier",
    },
}
DEFAULT_NAMESPACE = "rjaloudi"


def base_model_license(model_name: str) -> tuple[str, str]:
    if model_name.startswith("nlpaueb/"):
        return "cc-by-sa-4.0", "Creative Commons Attribution-ShareAlike 4.0"
    return "apache-2.0", "Apache License 2.0"


def validate_repo_id(repo_id: str) -> str:
    normalized = repo_id.strip().rstrip(".")
    if normalized.startswith(("http://", "https://")) or normalized.count("/") != 1:
        raise ValueError("Use a Hugging Face repository ID in namespace/name form, not a URL.")
    namespace, name = normalized.split("/", 1)
    if not namespace or not name or "YOUR_" in normalized:
        raise ValueError("Provide the exact Hugging Face namespace and repository name.")
    return normalized


def _v2_model_card(summary: dict[str, Any], repo_id: str) -> str:
    template_path = PROJECT_ROOT / "docs" / "models" / "CONTRA_MODEL_CARD.md"
    card = template_path.read_text(encoding="utf-8")
    card = card.replace(
        "rjaloudi/contra-contracts-bert-cuad-clause-classifier",
        repo_id,
    )
    metrics = summary["test_metrics"]
    replacements = {
        "0.7639": f"{metrics['accuracy']:.4f}",
        "0.6456": f"{metrics['macro_precision']:.4f}",
        "0.6151": f"{metrics['macro_recall']:.4f}",
        "0.6036": f"{metrics['macro_f1']:.4f}",
        "0.5869": f"{summary['best_validation_macro_f1']:.4f}",
        "1,008": f"{summary['split_rows']['test']:,}",
    }
    for old, new in replacements.items():
        card = card.replace(old, new)
    return card


def _legacy_model_card(summary: dict[str, Any], repo_id: str, version: str) -> str:
    metrics = summary["test_metrics"]
    license_slug, license_name = base_model_license(summary["model_name"])
    return f"""---
license: {license_slug}
library_name: transformers
pipeline_tag: text-classification
language:
- en
base_model: {summary["model_name"]}
datasets:
- theatticusproject/cuad-qa
metrics:
- accuracy
- f1
tags:
- legal
- contracts
- clause-classification
- pytorch
---

# Contra {version}: CUAD clause classifier

Contra {version} is an educational 41-label English contract-clause classifier fine-tuned from
[{summary["model_name"]}](https://huggingface.co/{summary["model_name"]}). The repository ID is
**{repo_id}**.

## Results

| Held-out metric | Value |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Macro precision | {metrics["macro_precision"]:.4f} |
| Macro recall | {metrics["macro_recall"]:.4f} |
| Macro-F1 | {metrics["macro_f1"]:.4f} |
| Test examples | {summary["split_rows"]["test"]:,} |

The source-contract-grouped split prevents a contract from crossing training, validation, and
test partitions. These results are not the official CUAD extractive-QA benchmark.

## Limitations

This educational alpha classifies passages, not whole contracts. Probabilities are uncalibrated,
the taxonomy is closed, rare labels are weak, and human review is required. It is not legal advice
and must not be used as an authority source or automated legal decision-maker.

## Example

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{repo_id}", top_k=3)
print(classifier("This Agreement is governed by the laws of the State of New York."))
```

## Licenses

- Fine-tuned model: {license_name}.
- CUAD-derived training data: CC BY 4.0.
- AI-Enabled Contracts Lab source: MIT.
"""


def model_card(summary: dict[str, Any], repo_id: str, version: str) -> str:
    if version == "v2":
        return _v2_model_card(summary, repo_id)
    return _legacy_model_card(summary, repo_id, version)


def prepare(run_dir: Path, output: Path, repo_id: str, version: str) -> Path:
    checkpoint = run_dir / "best"
    summary_path = run_dir / "summary.json"
    if not checkpoint.is_dir() or not summary_path.exists():
        raise FileNotFoundError(f"Expected best/ and summary.json below {run_dir}.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_v2 = summary["model_name"].startswith("nlpaueb/")
    if (version == "v2") != expected_v2:
        raise ValueError(f"Selected {version} does not match base model {summary['model_name']}.")

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_files = [source for source in checkpoint.iterdir() if source.is_file()]
    for source in checkpoint_files:
        shutil.copy2(source, output / source.name)

    (output / "README.md").write_text(
        model_card(summary, repo_id, version),
        encoding="utf-8",
    )
    shutil.copy2(summary_path, output / "evaluation_summary.json")
    config_path = PROJECT_ROOT / "artifacts" / "configs" / f"{summary['run_name']}.json"
    if config_path.exists():
        shutil.copy2(config_path, output / "training_config.json")

    required = {
        "README.md",
        "config.json",
        "evaluation_summary.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_config.json",
    }
    missing = sorted(required - {path.name for path in output.iterdir() if path.is_file()})
    if missing:
        raise FileNotFoundError(f"Prepared bundle is missing required files: {missing}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSION_DEFAULTS), default="v2")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    defaults = VERSION_DEFAULTS[args.version]
    run_dir = args.run_dir or defaults["run_dir"]
    output = args.output or defaults["output"]
    repo_id = validate_repo_id(
        args.repo_id or f"{DEFAULT_NAMESPACE}/{defaults['repo_name']}"
    )

    target = prepare(run_dir, output, repo_id, args.version)
    print(f"[OK] prepared {target}")
    print(f"[OK] repository ID: {repo_id}")
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=target)
        visibility = "private" if args.private else "public"
        print(f"[OK] uploaded {visibility} repository https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()