import json

import pandas as pd
import pytest

from contractlab.data import (
    instruction_jsonl_to_classifier,
    load_master_taxonomy,
    load_project_dataset_catalog,
    normalize_dataset,
    prepare_synthetic_396_experiment,
    split_by_contract,
    split_predefined,
    synthetic_dataset,
    validate_dataset,
)


def test_synthetic_dataset_is_balanced_and_valid():
    data = synthetic_dataset()
    report = validate_dataset(data)
    assert report.rows == 80
    assert report.labels == 8
    assert report.contracts == 10
    assert report.duplicates == 0
    assert not report.warnings
    assert data.groupby("label").size().nunique() == 1


def test_group_split_has_no_contract_overlap():
    splits = split_by_contract(synthetic_dataset(), seed=7)
    groups = {name: set(frame["contract_id"]) for name, frame in splits.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert sum(len(frame) for frame in splits.values()) == 80
    assert all(frame["label"].nunique() == 8 for frame in splits.values())


def test_predefined_split_preserves_rows_and_rejects_group_overlap():
    frame = pd.DataFrame(
        [
            {"text": "A", "label": "X", "contract_id": "1", "original_split": "train"},
            {
                "text": "B",
                "label": "X",
                "contract_id": "2",
                "original_split": "validation",
            },
            {"text": "C", "label": "X", "contract_id": "3", "original_split": "test"},
        ]
    )
    splits = split_predefined(frame)
    assert {name: len(part) for name, part in splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }

    frame.loc[2, "contract_id"] = "1"
    with pytest.raises(ValueError, match="Group leakage"):
        split_predefined(frame)


def test_normalize_adds_safe_metadata():
    data = synthetic_dataset()[["text", "label"]].head(3)
    normalized = normalize_dataset(data)
    assert {"contract_id", "source", "license"}.issubset(normalized.columns)


def test_synthetic_396_preparation_groups_template_variants():
    frame = pd.DataFrame(
        [
            {
                "text": "Party A shall follow Payment Terms within 30 days.",
                "label": "PAYMENT_TERMS",
                "clause_name": "Payment Terms",
                "source": "synthetic",
                "split": "train",
            },
            {
                "text": "Party B shall follow Payment Terms within 60 days.",
                "label": "PAYMENT_TERMS",
                "clause_name": "Payment Terms",
                "source": "synthetic",
                "split": "test",
            },
        ]
    )
    prepared = prepare_synthetic_396_experiment(frame, expected_labels=None)
    assert prepared["contract_id"].nunique() == 1
    assert set(prepared["license"]) == {"UNSPECIFIED - LOCAL ONLY"}
    assert prepared["original_split"].tolist() == ["train", "test"]


def test_synthetic_396_preparation_rejects_incomplete_schema():
    with pytest.raises(ValueError, match="missing columns"):
        prepare_synthetic_396_experiment(
            pd.DataFrame(
                [
                    {
                        "text": "Example",
                        "label": "EXAMPLE",
                        "source": "synthetic",
                        "split": "train",
                    }
                ]
            ),
            expected_labels=None,
        )


def _instruction_record(contract: str, label: str, answer: str) -> dict:
    return {
        "task": "clause_qa",
        "messages": [{"role": "assistant", "content": answer}],
        "meta": {"contract_title": contract, "category": label},
    }


def test_instruction_conversion_preserves_metadata_and_splits(tmp_path):
    sources = {}
    for split, contract in (("train", "A"), ("validation", "B"), ("test", "C")):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            json.dumps(_instruction_record(contract, "Termination", f"Clause {contract}")) + "\n",
            encoding="utf-8",
        )
        sources[split] = path

    target = instruction_jsonl_to_classifier(sources, tmp_path / "classifier.csv")
    frame = pd.read_csv(target)
    assert len(frame) == 3
    assert set(frame["contract_id"]) == {"A", "B", "C"}
    assert set(frame["license"]) == {"CC BY 4.0"}
    metadata = json.loads(target.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["contract_overlap_verified"] is True


def test_instruction_conversion_rejects_contract_leakage(tmp_path):
    sources = {}
    for split in ("train", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            json.dumps(_instruction_record("same-contract", "Termination", split)) + "\n",
            encoding="utf-8",
        )
        sources[split] = path
    with pytest.raises(ValueError, match="Contract leakage"):
        instruction_jsonl_to_classifier(sources, tmp_path / "classifier.csv")


def test_master_taxonomy_validates_manifest_and_archive_readiness(tmp_path):
    taxonomy_path = tmp_path / "master_taxonomy.json"
    taxonomy_path.write_text(
        json.dumps({"Domain_A": ["LABEL_A", "LABEL_B"], "Domain_B": ["LABEL_C"]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "domain": "Domain_A",
                "examples": 10,
                "labels": 2,
                "train": 8,
                "validation": 1,
                "test": 1,
                "zip_file": "Domain_A.zip",
            },
            {
                "domain": "Domain_B",
                "examples": 10,
                "labels": 1,
                "train": 8,
                "validation": 1,
                "test": 1,
                "zip_file": "Domain_B.zip",
            },
        ]
    ).to_csv(manifest_path, index=False)
    (tmp_path / "Domain_A.zip").write_bytes(b"placeholder")

    taxonomy, manifest = load_master_taxonomy(taxonomy_path, manifest_path)
    assert len(taxonomy) == 2
    assert manifest["archive_available"].tolist() == [True, False]


def test_master_taxonomy_rejects_split_count_mismatch(tmp_path):
    taxonomy_path = tmp_path / "master_taxonomy.json"
    taxonomy_path.write_text(json.dumps({"Domain_A": ["LABEL_A"]}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "domain": "Domain_A",
                "examples": 10,
                "labels": 1,
                "train": 8,
                "validation": 1,
                "test": 0,
                "zip_file": "Domain_A.zip",
            }
        ]
    ).to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="Split count mismatch"):
        load_master_taxonomy(taxonomy_path, manifest_path)


def test_project_catalog_checks_file_availability(tmp_path):
    (tmp_path / "present.csv").write_text("text,label\nA,B\n", encoding="utf-8")
    catalog = {
        "datasets": [
            {"name": "present", "path": "present.csv"},
            {"name": "missing", "path": "missing.csv"},
        ]
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    frame = load_project_dataset_catalog(path)
    assert frame["available"].tolist() == [True, False]
