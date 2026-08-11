"""Build a reproducible inventory for the repository's publishable datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

DATASET_FILES = [
    ("sample_200.jsonl", "Instruction/chat sample", "CC BY 4.0", "The Atticus Project CUAD v1"),
    ("cuad_classifier_sample.csv", "Classifier sample", "CC BY 4.0", "The Atticus Project CUAD v1"),
    ("clauses.csv", "Full classifier derivative", "CC BY 4.0", "The Atticus Project CUAD v1"),
    (
        "train.jsonl/train.jsonl",
        "Instruction train split",
        "CC BY 4.0",
        "The Atticus Project CUAD v1",
    ),
    (
        "val.jsonl/val.jsonl",
        "Instruction validation split",
        "CC BY 4.0",
        "The Atticus Project CUAD v1",
    ),
    ("test.jsonl/test.jsonl", "Instruction test split", "CC BY 4.0", "The Atticus Project CUAD v1"),
    (
        "contract_taxonomy.json",
        "Combined contract and clause definitions",
        "CC BY 4.0",
        "AI-Enabled Contracts Lab contributors",
    ),
    (
        "taxonomy/contract_types.jsonl",
        "Contract-type definition sources",
        "CC BY 4.0",
        "AI-Enabled Contracts Lab contributors",
    ),
    (
        "taxonomy/clause_definitions.jsonl",
        "Clause-definition sources",
        "CC BY 4.0",
        "AI-Enabled Contracts Lab contributors",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_jsonl(path: Path) -> dict[str, Any]:
    rows = 0
    contracts: set[str] = set()
    tasks: Counter[str] = Counter()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            tasks[str(row.get("task", "unknown"))] += 1
            contract = str((row.get("meta") or {}).get("contract_title", "")).strip()
            if contract:
                contracts.add(contract)
    return {"rows": rows, "contracts": len(contracts), "tasks": dict(sorted(tasks.items()))}


def inspect_csv(path: Path) -> dict[str, Any]:
    rows = 0
    contracts: set[str] = set()
    labels: Counter[str] = Counter()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if row.get("contract_id"):
                contracts.add(row["contract_id"])
            if row.get("label"):
                labels[row["label"]] += 1
    return {"rows": rows, "contracts": len(contracts), "labels": len(labels)}


def contract_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                values.add(json.loads(line)["meta"]["contract_title"])
    return values


def build_manifest() -> dict[str, Any]:
    files = []
    for relative, purpose, license_name, source in DATASET_FILES:
        path = DATASETS / relative
        if relative.startswith(("taxonomy/", "contract_taxonomy")):
            details = (
                {"records": sum(1 for line in path.open(encoding="utf-8") if line.strip())}
                if path.suffix == ".jsonl"
                else {}
            )
        else:
            details = inspect_jsonl(path) if path.suffix == ".jsonl" else inspect_csv(path)
        files.append(
            {
                "path": f"datasets/{relative}",
                "purpose": purpose,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "license": license_name,
                "source": source,
                "redistributable": True,
                **details,
            }
        )

    splits = {
        name: contract_ids(DATASETS / f"{name}.jsonl" / f"{name}.jsonl")
        for name in ("train", "val", "test")
    }
    overlap = (
        (splits["train"] & splits["val"])
        | (splits["train"] & splits["test"])
        | (splits["val"] & splits["test"])
    )
    if overlap:
        raise RuntimeError(f"Contract leakage detected in published splits: {len(overlap)}")

    return {
        "schema_version": 1,
        "generated": date.today().isoformat(),
        "dataset": "AI-Enabled Contracts Lab CUAD derivatives",
        "upstream": "https://github.com/The-Atticus-Project/cuad",
        "dataset_card": "https://huggingface.co/datasets/theatticusproject/cuad-qa",
        "license": "CC BY 4.0",
        "transformation": (
            "Deterministic parsing and templating over CUAD annotations; no language model "
            "generated the included targets. Classifier rows use grounded positive answer spans."
        ),
        "contract_overlap_verified": True,
        "files": files,
    }


def main() -> None:
    target = DATASETS / "MANIFEST.json"
    target.write_text(json.dumps(build_manifest(), indent=2), encoding="utf-8")
    print(f"[OK] wrote {target}")


if __name__ == "__main__":
    main()
