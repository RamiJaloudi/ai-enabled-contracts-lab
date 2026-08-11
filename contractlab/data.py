from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"text", "label"}
OPTIONAL_COLUMNS = {"contract_id", "source", "license"}

CUAD_GITHUB_URL = "https://github.com/The-Atticus-Project/cuad"
CUAD_HUGGING_FACE_ID = "theatticusproject/cuad-qa"
CUAD_DATA_ARCHIVE_URL = "https://github.com/The-Atticus-Project/cuad/raw/refs/heads/main/data.zip"
CUAD_LICENSE = "CC BY 4.0"

LABEL_TEMPLATES: dict[str, list[str]] = {
    "Confidentiality": [
        "Each party shall keep all non-public technical and business information confidential for {years} years.",
        "The receiving party may use Confidential Information only to perform its obligations under this Agreement.",
    ],
    "Governing Law": [
        "This Agreement is governed by the laws of the State of {state}, without regard to conflicts principles.",
        "The parties submit to the exclusive jurisdiction of courts located in {state}.",
    ],
    "Termination": [
        "Either party may terminate this Agreement for material breach if the breach is not cured within {days} days.",
        "A party may terminate immediately upon the other party's insolvency or cessation of business.",
    ],
    "Limitation of Liability": [
        "Neither party is liable for indirect, incidental, special, or consequential damages arising from this Agreement.",
        "Except for excluded claims, each party's aggregate liability shall not exceed the fees paid during the prior twelve months.",
    ],
    "Indemnification": [
        "Supplier shall defend and indemnify Customer against third-party claims alleging that the services infringe intellectual property rights.",
        "Each party will indemnify the other for losses caused by its gross negligence or willful misconduct.",
    ],
    "Payment Terms": [
        "Customer shall pay undisputed invoices within {days} days after receipt.",
        "Late undisputed amounts accrue interest at one percent per month or the maximum lawful rate, whichever is lower.",
    ],
    "Intellectual Property": [
        "Each party retains ownership of intellectual property it owned before the Effective Date.",
        "Customer owns its data, and no rights are granted except the limited rights required to provide the services.",
    ],
    "Renewal": [
        "The initial term automatically renews for successive one-year periods unless either party gives {days} days' notice.",
        "Renewal fees may increase only upon written notice delivered at least {days} days before renewal.",
    ],
}

CONTRACT_TYPES = [
    "SaaS agreement",
    "vendor agreement",
    "non-disclosure agreement",
    "employment agreement",
    "license agreement",
    "services agreement",
    "distribution agreement",
    "consulting agreement",
    "purchase agreement",
    "partnership agreement",
]
STATES = ["New York", "Delaware", "California", "Virginia", "Illinois"]


@dataclass(slots=True)
class DatasetReport:
    rows: int
    labels: int
    contracts: int
    duplicates: int
    missing_text: int
    missing_label: int
    missing_contract_id: int
    warnings: list[str]


def synthetic_dataset() -> pd.DataFrame:
    """Return original synthetic clauses suitable for a fast guided run."""
    records: list[dict[str, str]] = []
    for contract_index, contract_type in enumerate(CONTRACT_TYPES, start=1):
        for label_index, (label, variants) in enumerate(LABEL_TEMPLATES.items()):
            template = variants[(contract_index + label_index) % len(variants)]
            text = template.format(
                years=2 + contract_index % 4,
                days=15 + 15 * (contract_index % 4),
                state=STATES[contract_index % len(STATES)],
            )
            text = f"In this {contract_type}, {text[0].lower()}{text[1:]}"
            records.append(
                {
                    "text": text,
                    "label": label,
                    "contract_id": f"synthetic-{contract_index:02d}",
                    "contract_type": contract_type,
                    "source": "AI-Enabled Contracts Lab original synthetic example",
                    "license": "MIT",
                }
            )
    return pd.DataFrame.from_records(records)


def load_table(source: str | Path | bytes, filename: str | None = None) -> pd.DataFrame:
    if isinstance(source, bytes):
        from io import BytesIO

        buffer = BytesIO(source)
        suffix = Path(filename or "upload.csv").suffix.lower()
        if suffix == ".jsonl":
            return pd.read_json(buffer, lines=True)
        if suffix == ".json":
            return pd.read_json(buffer)
        return pd.read_csv(buffer)

    path = Path(source)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def prepare_synthetic_396_experiment(
    frame: pd.DataFrame, expected_labels: int | None = 396
) -> pd.DataFrame:
    """Prepare all 396 synthetic labels with leakage-aware grouping.

    The supplied corpus has no source-contract identifiers. A conservative
    template fingerprint becomes contract_id so near-template variants remain
    together during the normal grouped split. The supplied split is retained
    as original_split for audit, but is not trusted as an evaluation boundary.
    """
    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    required = {"text", "label", "clause_name", "source", "split"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Synthetic 396-label corpus is missing columns: " + ", ".join(sorted(missing))
        )

    normalized_labels = sorted(
        {
            re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
            for value in data["label"].dropna()
        },
        key=len,
        reverse=True,
    )
    label_pattern = re.compile(
        r"(?<![a-z0-9])(?:"
        + "|".join(re.escape(value) for value in normalized_labels if value)
        + r")(?![a-z0-9])"
    )

    def template_group(text: object) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
        normalized = label_pattern.sub(" <clause> ", normalized)
        normalized = re.sub(r"\bparty [a-z]\b", "party <x>", normalized)
        normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"synthetic-template-{digest}"

    data["original_source"] = data["source"]
    data["original_split"] = data["split"]
    data["contract_id"] = data["text"].map(template_group)
    data["source"] = "User-supplied synthetic 396-label corpus"
    data["license"] = "UNSPECIFIED - LOCAL ONLY"
    prepared = normalize_dataset(data)
    if expected_labels is not None and prepared["label"].nunique() != expected_labels:
        raise ValueError(f"Expected {expected_labels} labels, found {prepared['label'].nunique()}.")
    return prepared


def normalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data["label"] = data["label"].fillna("").astype(str).str.strip()
    if "contract_id" not in data:
        data["contract_id"] = [f"row-{index:06d}" for index in range(len(data))]
    data["contract_id"] = data["contract_id"].fillna("").astype(str).str.strip()
    data.loc[data["contract_id"] == "", "contract_id"] = [
        f"ungrouped-{index:06d}" for index in data.index[data["contract_id"] == ""]
    ]
    if "source" not in data:
        data["source"] = "User supplied"
    if "license" not in data:
        data["license"] = "Unknown"
    return data


def validate_dataset(frame: pd.DataFrame) -> DatasetReport:
    data = frame.copy()
    lower_columns = {str(column).strip().lower() for column in data.columns}
    missing_columns = REQUIRED_COLUMNS - lower_columns
    if missing_columns:
        return DatasetReport(
            rows=len(data),
            labels=0,
            contracts=0,
            duplicates=0,
            missing_text=len(data),
            missing_label=len(data),
            missing_contract_id=len(data),
            warnings=[f"Missing required columns: {', '.join(sorted(missing_columns))}"],
        )
    data.columns = [str(column).strip().lower() for column in data.columns]
    missing_text = int(data["text"].fillna("").astype(str).str.strip().eq("").sum())
    missing_label = int(data["label"].fillna("").astype(str).str.strip().eq("").sum())
    missing_contract_id = (
        len(data) if "contract_id" not in data else int(data["contract_id"].isna().sum())
    )
    warnings: list[str] = []
    if "contract_id" not in data:
        warnings.append(
            "No contract_id column: document-level leakage cannot be detected reliably."
        )
    if "license" not in data:
        warnings.append("No license column: record provenance before training or redistribution.")
    label_counts = data["label"].value_counts()
    if len(label_counts) < 2:
        warnings.append("At least two labels are required for classification.")
    if not label_counts.empty and int(label_counts.min()) < 3:
        warnings.append(
            "Some labels have fewer than three examples; all splits may not contain them."
        )
    return DatasetReport(
        rows=len(data),
        labels=int(data["label"].nunique()),
        contracts=int(data["contract_id"].nunique()) if "contract_id" in data else len(data),
        duplicates=int(data.duplicated(subset=["text"]).sum()),
        missing_text=missing_text,
        missing_label=missing_label,
        missing_contract_id=missing_contract_id,
        warnings=warnings,
    )


def split_predefined(
    frame: pd.DataFrame,
    split_column: str = "original_split",
) -> dict[str, pd.DataFrame]:
    """Use declared partitions after validating labels and group separation."""
    data = normalize_dataset(frame)
    column = split_column if split_column in data.columns else "split"
    if column not in data.columns:
        raise ValueError(f"Predefined splitting requires a '{split_column}' or 'split' column.")
    normalized = (
        data[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"val": "validation", "dev": "validation"})
    )
    allowed = {"train", "validation", "test"}
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected predefined split values: {unexpected[:5]}")
    splits = {
        name: data[normalized == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    if any(part.empty for part in splits.values()):
        raise ValueError("Predefined train, validation, and test splits must be non-empty.")

    groups = {name: set(part["contract_id"]) for name, part in splits.items()}
    overlap = (
        (groups["train"] & groups["validation"])
        | (groups["train"] & groups["test"])
        | (groups["validation"] & groups["test"])
    )
    if overlap:
        raise ValueError(f"Group leakage across predefined splits; example: {sorted(overlap)[0]}")
    missing_train_labels = set(data["label"]) - set(splits["train"]["label"])
    if missing_train_labels:
        example = sorted(missing_train_labels)[0]
        raise ValueError(f"Label absent from predefined training split: {example}")
    return splits


def split_by_contract(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split complete source contracts, never individual rows, across partitions."""
    data = normalize_dataset(frame)
    groups = data["contract_id"].drop_duplicates().to_numpy()
    if len(groups) < 3:
        raise ValueError("At least three distinct contract_id values are required.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_end = max(1, int(round(len(groups) * train_ratio)))
    validation_count = max(1, int(round(len(groups) * validation_ratio)))
    validation_end = min(len(groups) - 1, train_end + validation_count)
    train_groups = set(groups[:train_end])
    validation_groups = set(groups[train_end:validation_end])
    test_groups = set(groups[validation_end:])
    if not validation_groups or not test_groups:
        train_groups = set(groups[:-2])
        validation_groups = {groups[-2]}
        test_groups = {groups[-1]}
    return {
        "train": data[data["contract_id"].isin(train_groups)].reset_index(drop=True),
        "validation": data[data["contract_id"].isin(validation_groups)].reset_index(drop=True),
        "test": data[data["contract_id"].isin(test_groups)].reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], directory: str | Path) -> dict[str, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in splits.items():
        path = target / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def load_master_taxonomy(
    taxonomy_path: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Validate a domain-label taxonomy and its referenced archive manifest."""
    taxonomy_file = Path(taxonomy_path)
    manifest_file = Path(manifest_path)
    payload = json.loads(taxonomy_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Master taxonomy must be a non-empty JSON object.")

    taxonomy: dict[str, list[str]] = {}
    for raw_domain, raw_labels in payload.items():
        domain = str(raw_domain).strip()
        if not domain or not isinstance(raw_labels, list) or not raw_labels:
            raise ValueError(f"Invalid taxonomy entry for domain: {raw_domain!r}")
        labels = [str(value).strip() for value in raw_labels]
        if any(not value for value in labels):
            raise ValueError(f"Blank label in taxonomy domain: {domain}")
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate label within taxonomy domain: {domain}")
        taxonomy[domain] = labels

    manifest = pd.read_csv(manifest_file)
    required = {
        "domain",
        "examples",
        "labels",
        "train",
        "validation",
        "test",
        "zip_file",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("Master manifest is missing columns: " + ", ".join(sorted(missing)))
    if manifest["domain"].duplicated().any():
        raise ValueError("Master manifest contains duplicate domain rows.")
    if set(manifest["domain"]) != set(taxonomy):
        raise ValueError("Manifest domains do not match master taxonomy domains.")

    numeric = ["examples", "labels", "train", "validation", "test"]
    for column in numeric:
        manifest[column] = pd.to_numeric(manifest[column], errors="raise").astype(int)
    for row in manifest.itertuples(index=False):
        if row.labels != len(taxonomy[row.domain]):
            raise ValueError(f"Label count mismatch for domain: {row.domain}")
        if row.train + row.validation + row.test != row.examples:
            raise ValueError(f"Split count mismatch for domain: {row.domain}")

    base = manifest_file.parent
    manifest["archive_available"] = manifest["zip_file"].map(
        lambda value: (base / str(value)).is_file()
    )
    manifest["unique_labels"] = manifest["domain"].map(lambda value: len(set(taxonomy[value])))
    return taxonomy, manifest


def load_project_dataset_catalog(path: str | Path) -> pd.DataFrame:
    """Load the lightweight catalog used by Dataset Lab without opening large files."""
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = payload.get("datasets", [])
    if not isinstance(rows, list):
        raise ValueError("Dataset catalog must contain a 'datasets' list.")
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    base = catalog_path.parent
    frame["available"] = frame["path"].map(lambda value: (base / str(value)).exists())
    frame["size_mb"] = frame["path"].map(
        lambda value: (
            round((base / str(value)).stat().st_size / 1_048_576, 2)
            if (base / str(value)).is_file()
            else None
        )
    )
    return frame


def instruction_jsonl_to_classifier(
    sources: Mapping[str, str | Path],
    output: str | Path,
) -> Path:
    """Convert grounded CUAD clause-QA records into classifier rows.

    Only ``clause_qa`` examples are used because their assistant message is a
    positive, verbatim CUAD answer span. Contract titles become grouping IDs,
    preserving the information needed for leakage-resistant splitting.
    """
    records: list[dict[str, str]] = []
    contracts_by_split: dict[str, set[str]] = {}
    input_rows = 0
    for split_name, source in sources.items():
        source_path = Path(source)
        contracts: set[str] = set()
        with source_path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                input_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {source_path} line {line_number}: {exc}"
                    ) from exc
                if row.get("task") != "clause_qa":
                    continue
                meta = row.get("meta") or {}
                contract_id = str(meta.get("contract_title", "")).strip()
                label = str(meta.get("category", "")).strip()
                messages = row.get("messages") or []
                text = next(
                    (
                        str(message.get("content", "")).strip()
                        for message in messages
                        if message.get("role") == "assistant"
                    ),
                    "",
                )
                if not contract_id or not label or not text:
                    raise ValueError(
                        f"Incomplete clause_qa record in {source_path} line {line_number}."
                    )
                contracts.add(contract_id)
                records.append(
                    {
                        "text": text,
                        "label": label,
                        "contract_id": contract_id,
                        "source": "The Atticus Project CUAD v1",
                        "license": CUAD_LICENSE,
                        "original_task": "clause_qa",
                        "original_split": split_name,
                    }
                )
        contracts_by_split[split_name] = contracts

    split_names = list(contracts_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = contracts_by_split[left] & contracts_by_split[right]
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(f"Contract leakage between {left} and {right}; example: {example}")
    if not records:
        raise ValueError("No complete clause_qa records were found.")

    frame = pd.DataFrame.from_records(records).drop_duplicates(
        subset=["text", "label", "contract_id"]
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    metadata = {
        "name": "CUAD classifier rows converted from grounded clause-QA records",
        "source": CUAD_GITHUB_URL,
        "license": CUAD_LICENSE,
        "transformation": (
            "Assistant targets from clause_qa records become classifier text; category becomes "
            "label; contract_title becomes contract_id; exact duplicates are removed."
        ),
        "input_rows": input_rows,
        "output_rows": len(frame),
        "contracts": int(frame["contract_id"].nunique()),
        "labels": int(frame["label"].nunique()),
        "source_splits": {name: len(contracts) for name, contracts in contracts_by_split.items()},
        "contract_overlap_verified": len(contracts_by_split) > 1,
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return target


def prepare_cuad(output: str | Path, max_rows: int | None = None) -> Path:
    """Convert positive CUAD answer spans into an educational multiclass dataset."""
    from urllib.request import Request, urlopen

    request = Request(
        CUAD_DATA_ARCHIVE_URL,
        headers={"User-Agent": "AI-Enabled-Contracts-Lab/0.1"},
    )
    with tempfile.TemporaryFile() as archive_file:
        with urlopen(request, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                archive_file.write(chunk)
        archive_file.seek(0)
        with zipfile.ZipFile(archive_file) as archive:
            with archive.open("CUADv1.json") as dataset_file:
                payload = json.load(dataset_file)

    records: list[dict[str, str]] = []
    for contract in payload.get("data", []):
        title = str(contract.get("title", "unknown-contract")).strip()
        for paragraph in contract.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                raw_category = str(qa.get("id", "Unknown clause")).split("__")[-1]
                category = re.sub(r"_\d+$", "", raw_category).strip()
                answers: Iterable[dict] = qa.get("answers", [])
                for answer in answers:
                    text = str(answer.get("text", "")).strip()
                    if text:
                        records.append(
                            {
                                "text": text,
                                "label": category,
                                "contract_id": title,
                                "source": "The Atticus Project CUAD v1",
                                "license": CUAD_LICENSE,
                            }
                        )
                        if max_rows and len(records) >= max_rows:
                            break
                if max_rows and len(records) >= max_rows:
                    break
            if max_rows and len(records) >= max_rows:
                break
        if max_rows and len(records) >= max_rows:
            break
    if not records:
        raise RuntimeError("CUAD loaded, but no positive answer spans were found.")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).drop_duplicates(subset=["text", "label"]).to_csv(
        target, index=False
    )
    metadata = {
        "name": "CUAD v1 positive answer spans",
        "source": CUAD_GITHUB_URL,
        "download": CUAD_DATA_ARCHIVE_URL,
        "dataset_card": f"https://huggingface.co/datasets/{CUAD_HUGGING_FACE_ID}",
        "license": CUAD_LICENSE,
        "transformation": "Positive answer spans converted to multiclass clause examples.",
        "rows": len(records),
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return target
