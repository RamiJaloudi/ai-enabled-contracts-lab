from __future__ import annotations

import csv
import json
from collections import Counter
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any
from zipfile import ZipFile

PROJECTION_FIELDS = [
    "text",
    "label",
    "contract_id",
    "source",
    "license",
    "original_split",
    "domain",
    "raw_label",
    "subcategory",
    "jurisdiction",
    "style",
]


def _read_bundle_metadata(bundle: ZipFile) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    taxonomy = json.loads(bundle.read("master_taxonomy.json"))
    manifest_text = TextIOWrapper(bundle.open("manifest.csv"), encoding="utf-8-sig", newline="")
    try:
        manifest = list(csv.DictReader(manifest_text))
    finally:
        manifest_text.close()
    if not isinstance(taxonomy, dict) or not taxonomy:
        raise ValueError("Bundle master_taxonomy.json must be a non-empty object.")
    if {row["domain"] for row in manifest} != set(taxonomy):
        raise ValueError("Bundle manifest domains do not match its master taxonomy.")
    return taxonomy, manifest


def prepare_legal_multidomain_classifier(
    bundle_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Stream all bundle records into a 1,120-class classifier projection.

    Labels are domain-qualified so the nine concepts reused by multiple domains
    remain distinct. The original all-fields records stay in the source bundle;
    this projection contains only fields used for classification and audit.
    """
    source = Path(bundle_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    counts: Counter[str] = Counter()
    labels: set[str] = set()
    domains: set[str] = set()

    try:
        with ZipFile(source) as outer, temporary.open("w", encoding="utf-8", newline="") as output:
            taxonomy, manifest = _read_bundle_metadata(outer)
            writer = csv.DictWriter(output, fieldnames=PROJECTION_FIELDS)
            writer.writeheader()
            for manifest_row in manifest:
                domain = manifest_row["domain"]
                nested_name = f"domain_archives/{manifest_row['zip_file']}"
                if nested_name not in outer.namelist():
                    raise ValueError(f"Bundle is missing archive: {nested_name}")
                allowed_labels = set(taxonomy[domain])
                with ZipFile(BytesIO(outer.read(nested_name))) as archive:
                    for split in ("train", "validation", "test"):
                        member = f"{split}.jsonl"
                        with (
                            archive.open(member) as raw,
                            TextIOWrapper(raw, encoding="utf-8") as records,
                        ):
                            for line_number, line in enumerate(records, start=1):
                                if not line.strip():
                                    continue
                                row: dict[str, Any] = json.loads(line)
                                raw_label = str(row.get("label", "")).strip()
                                row_domain = str(row.get("domain", "")).strip()
                                text = str(row.get("text", "")).strip()
                                record_id = str(row.get("id", "")).strip()
                                if (
                                    row_domain != domain
                                    or raw_label not in allowed_labels
                                    or not text
                                    or not record_id
                                    or str(row.get("split", "")).strip() != split
                                ):
                                    raise ValueError(
                                        f"Invalid record in {nested_name}/{member} "
                                        f"line {line_number}."
                                    )
                                qualified_label = f"{domain}::{raw_label}"
                                writer.writerow(
                                    {
                                        "text": text,
                                        "label": qualified_label,
                                        "contract_id": f"{domain}:{record_id}",
                                        "source": "User-supplied synthetic legal multidomain bundle",
                                        "license": "UNSPECIFIED - LOCAL ONLY",
                                        "original_split": split,
                                        "domain": domain,
                                        "raw_label": raw_label,
                                        "subcategory": row.get("subcategory", ""),
                                        "jurisdiction": row.get("jurisdiction", ""),
                                        "style": row.get("style", ""),
                                    }
                                )
                                counts[split] += 1
                                labels.add(qualified_label)
                                domains.add(domain)
            expected = {
                "train": sum(int(row["train"]) for row in manifest),
                "validation": sum(int(row["validation"]) for row in manifest),
                "test": sum(int(row["test"]) for row in manifest),
            }
            if any(counts[name] != expected[name] for name in expected):
                raise ValueError(f"Bundle split counts do not match manifest: {dict(counts)}")
            expected_assignments = sum(len(values) for values in taxonomy.values())
            if len(labels) != expected_assignments:
                raise ValueError(
                    f"Expected {expected_assignments} domain-qualified labels, found {len(labels)}."
                )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "name": "Contra-Law Experimental classifier projection",
        "source_bundle": str(source),
        "license": "UNSPECIFIED - LOCAL ONLY",
        "rows": sum(counts.values()),
        "domains": len(domains),
        "domain_qualified_labels": len(labels),
        "splits": dict(counts),
        "input_field": "text",
        "target_field": "domain::label",
        "preserved_metadata": [
            "original_split",
            "domain",
            "raw_label",
            "subcategory",
            "jurisdiction",
            "style",
        ],
        "excluded_analysis_fields": [
            "issue",
            "rule_summary",
            "application",
            "counterargument",
            "conclusion",
        ],
        "reason_analysis_fields_excluded": (
            "A BERT sequence-classification head cannot generate structured analysis. "
            "The original bundle preserves those fields for a separate retrieval or "
            "generative experiment."
        ),
        "known_limitations": (
            "Synthetic template and target-name leakage are documented in "
            "datasets/master_taxonomy_audit.json."
        ),
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return target
