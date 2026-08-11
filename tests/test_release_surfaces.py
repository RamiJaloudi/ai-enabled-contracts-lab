from __future__ import annotations

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPOSITORY = "RamiJaloudi/ai-enabled-contracts-lab"
HF_REPOSITORY = "rjaloudi/contra-contracts-bert-cuad-clause-classifier"


def read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def test_home_and_code_lab_are_valid_python() -> None:
    for relative in ("app_pages/home.py", "app_pages/code_lab.py"):
        ast.parse(read(relative), filename=relative)


def test_home_explains_hybrid_boundary_and_links_guides() -> None:
    home = read("app_pages/home.py")
    assert "modern hybrid legal-AI architecture" in home
    assert "Classification is a hypothesis, not a legal conclusion" in home
    assert "No RAG" in home
    assert "DOCX/TXT export" in home
    assert "AI_Enabled_Contracts_Lab_User_Guide.docx" in home
    assert GITHUB_REPOSITORY in home
    assert HF_REPOSITORY in home


def test_code_lab_uses_private_first_exact_destinations_without_token_widget() -> None:
    code_lab = read("app_pages/code_lab.py")
    assert GITHUB_REPOSITORY in code_lab
    assert HF_REPOSITORY in code_lab
    assert "--private" in code_lab
    assert "hf auth login" in code_lab
    assert "git diff --cached" in code_lab
    assert "text_input(\"token" not in code_lab.lower()
    assert "password" not in code_lab.lower()


def test_readme_links_complete_release_surfaces() -> None:
    readme = read("README.md")
    assert "## Modern hybrid architecture" in readme
    assert "## Contract Chat" in readme
    assert "## Private-first publishing" in readme
    assert "docs/AI_Enabled_Contracts_Lab_User_Guide.docx" in readme
    assert "docs/HUGGING_FACE_RELEASE.md" in readme
    assert GITHUB_REPOSITORY in readme
    assert HF_REPOSITORY in readme


def test_release_docs_have_no_account_placeholder_and_disclose_limits() -> None:
    files = (
        "docs/HUGGING_FACE_RELEASE.md",
        "docs/models/CONTRA_MODEL_CARD.md",
        "docs/AI_Enabled_Contracts_Lab_User_Guide.md",
    )
    for relative in files:
        content = read(relative)
        assert "YOUR_HF_USERNAME" not in content
        assert HF_REPOSITORY in content

    card = read("docs/models/CONTRA_MODEL_CARD.md")
    assert "0.6036" in card
    assert "Zero-F1 labels | 6" in card
    assert "not a claim of state-of-the-art benchmark performance" in card
    assert "cc-by-sa-4.0" in card


def test_private_first_checklist_covers_both_hosts() -> None:
    checklist = read("docs/PUBLIC_RELEASE_CHECKLIST.md")
    assert "Private GitHub staging" in checklist
    assert "Private Hugging Face staging" in checklist
    assert GITHUB_REPOSITORY in checklist
    assert HF_REPOSITORY in checklist

def test_hugging_face_repo_id_validation() -> None:
    from tools.prepare_huggingface_model import validate_repo_id

    assert validate_repo_id(HF_REPOSITORY + ".") == HF_REPOSITORY
    for invalid in (
        "https://huggingface.co/rjaloudi/model",
        "YOUR_HF_USERNAME/model",
        "missing-namespace",
    ):
        try:
            validate_repo_id(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid repository ID: {invalid}")


def test_hugging_face_downloader_validates_reviewed_bundle(tmp_path: Path) -> None:
    from tools.download_huggingface_model import REQUIRED_FILES, validate_bundle

    labels = {str(index): f"label_{index}" for index in range(41)}
    reverse = {label: int(index) for index, label in labels.items()}
    for filename in REQUIRED_FILES:
        (tmp_path / filename).write_text("placeholder", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["BertForSequenceClassification"],
                "id2label": labels,
                "label2id": reverse,
            }
        ),
        encoding="utf-8",
    )

    summary = validate_bundle(tmp_path)
    assert summary == {
        "files": 7,
        "labels": 41,
        "architecture": "BertForSequenceClassification",
    }


def test_hugging_face_downloader_has_no_token_argument() -> None:
    from tools.download_huggingface_model import build_parser

    help_text = build_parser().format_help()
    assert "--repo-id" in help_text
    assert "--revision" in help_text
    assert "--output" in help_text
    assert "--token" not in help_text


def test_download_workflow_is_linked_from_release_docs() -> None:
    for relative in (
        "README.md",
        "docs/HUGGING_FACE_RELEASE.md",
        "docs/AI_Enabled_Contracts_Lab_User_Guide.md",
    ):
        content = read(relative)
        assert "tools\\download_huggingface_model.py" in content
        assert "hf auth login" in content


def test_generated_v2_card_uses_exact_repo_and_saved_metrics() -> None:
    from tools.prepare_huggingface_model import model_card

    summary = {
        "model_name": "nlpaueb/bert-base-uncased-contracts",
        "best_validation_macro_f1": 0.5869,
        "split_rows": {"test": 1008},
        "test_metrics": {
            "accuracy": 0.7639,
            "macro_precision": 0.6456,
            "macro_recall": 0.6151,
            "macro_f1": 0.6036,
        },
    }
    card = model_card(summary, HF_REPOSITORY, "v2")
    assert HF_REPOSITORY in card
    assert "YOUR_HF_USERNAME" not in card
    assert "0.6036" in card
    assert "cc-by-sa-4.0" in card
