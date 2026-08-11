"""Download and validate the reviewed Contra v2 Hugging Face model bundle."""

from __future__ import annotations

import argparse
import errno
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "rjaloudi/contra-contracts-bert-cuad-clause-classifier"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "models" / "contra-v2"
DEFAULT_REVISION = "main"
EXPECTED_LABELS = 41
REQUIRED_FILES = (
    "README.md",
    "config.json",
    "evaluation_summary.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_config.json",
)


def validate_repo_id(repo_id: str) -> str:
    normalized = repo_id.strip().rstrip(".")
    if normalized.startswith(("http://", "https://")) or normalized.count("/") != 1:
        raise ValueError("Use a Hugging Face repository ID in namespace/name form, not a URL.")
    namespace, name = normalized.split("/", 1)
    if not namespace or not name or "YOUR_" in normalized:
        raise ValueError("Provide the exact Hugging Face namespace and repository name.")
    return normalized


def validate_destination(output: Path, minimum_free_gb: float) -> Path:
    destination = output.expanduser().resolve()
    if destination == PROJECT_ROOT or destination == PROJECT_ROOT.parent:
        raise ValueError("Choose a dedicated model directory, not the project or its parent.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination.parent).free
    required_bytes = int(minimum_free_gb * 1024**3)
    if free_bytes < required_bytes:
        free_gb = free_bytes / 1024**3
        raise OSError(
            errno.ENOSPC,
            f"Only {free_gb:.2f} GB is free; at least {minimum_free_gb:.2f} GB is required.",
        )
    return destination


def validate_bundle(output: Path, expected_labels: int = EXPECTED_LABELS) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file()]
    empty = [
        name
        for name in REQUIRED_FILES
        if (output / name).is_file() and not (output / name).stat().st_size
    ]
    if missing or empty:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if empty:
            details.append(f"empty: {', '.join(empty)}")
        raise FileNotFoundError("Downloaded bundle is incomplete (" + "; ".join(details) + ").")

    try:
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Downloaded config.json is not valid UTF-8 JSON.") from exc

    id2label = config.get("id2label")
    label2id = config.get("label2id")
    if not isinstance(id2label, dict) or not isinstance(label2id, dict):
        raise ValueError("config.json must contain id2label and label2id mappings.")
    if len(id2label) != expected_labels or len(label2id) != expected_labels:
        raise ValueError(
            f"Expected {expected_labels} labels, found {len(id2label)} id2label and "
            f"{len(label2id)} label2id entries."
        )
    for label_id, label in id2label.items():
        if str(label2id.get(label)) != str(label_id):
            raise ValueError("config.json id2label and label2id mappings are not exact inverses.")

    return {
        "files": len(REQUIRED_FILES),
        "labels": len(id2label),
        "architecture": config.get("architectures", ["unknown"])[0],
    }


def run_inference_smoke_test(output: Path, expected_labels: int = EXPECTED_LABELS) -> str:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the project dependencies before running the inference smoke test."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        output,
        local_files_only=True,
        use_safetensors=True,
    )
    encoded = tokenizer(
        "This Agreement is governed by the laws of the State of New York.",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    model.eval()
    with torch.inference_mode():
        logits = model(**encoded).logits
    if tuple(logits.shape) != (1, expected_labels):
        raise ValueError(
            f"Expected inference logits shaped (1, {expected_labels}), found {tuple(logits.shape)}."
        )
    top_label_id = int(logits.argmax(dim=-1).item())
    return str(model.config.id2label[top_label_id])


def download_model(
    repo_id: str,
    output: Path,
    revision: str,
    *,
    force_download: bool = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is unavailable. Install the project dependencies first."
        ) from exc

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=output,
            allow_patterns=list(REQUIRED_FILES),
            force_download=force_download,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise RuntimeError(
            f"Cannot access {repo_id}. Confirm the repository exists, then run "
            "'hf auth login' with an account that has access."
        ) from exc
    except RevisionNotFoundError as exc:
        raise RuntimeError(f"Revision '{revision}' does not exist in {repo_id}.") from exc
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise OSError(errno.ENOSPC, "The model download ran out of disk space.") from exc
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and validate the reviewed Contra v2 Hugging Face model."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Branch, tag, or full commit hash to download (default: main).",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Validate files and configuration without loading the model for inference.",
    )
    parser.add_argument(
        "--minimum-free-gb",
        type=float,
        default=1.0,
        help="Fail before downloading when less free space is available (default: 1.0).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.minimum_free_gb < 0:
        raise ValueError("--minimum-free-gb cannot be negative.")
    repo_id = validate_repo_id(args.repo_id)
    output = validate_destination(args.output, args.minimum_free_gb)
    target = download_model(
        repo_id,
        output,
        args.revision,
        force_download=args.force_download,
    )
    summary = validate_bundle(target)
    print(f"[OK] downloaded {repo_id}@{args.revision} to {target}")
    print(f"[OK] validated {summary['files']} release files and {summary['labels']} labels")
    if not args.skip_inference:
        top_label = run_inference_smoke_test(target)
        print(f"[OK] local inference produced 41 logits; top smoke-test label: {top_label}")


if __name__ == "__main__":
    main()
