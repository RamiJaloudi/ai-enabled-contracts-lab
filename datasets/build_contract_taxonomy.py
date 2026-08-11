"""Build and validate the public contract type and clause definition taxonomy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "taxonomy"
OUTPUT = ROOT / "contract_taxonomy.json"

DOMAIN_DEFINITIONS = {
    "commercial": "Agreements governing goods, services, distribution, licensing, and ongoing business relationships.",
    "corporate": "Agreements concerning ownership, investment, governance, combinations, and business transfers.",
    "employment": "Agreements governing work relationships, compensation, duties, separation, and labor terms.",
    "family": "Agreements concerning family finances, property, support, and parenting, subject to jurisdiction-specific law and court review.",
    "real_estate": "Agreements concerning ownership, transfer, possession, construction, or use of real property.",
    "finance": "Agreements creating or supporting credit, repayment, collateral, guarantees, and financial covenants.",
    "ip_technology": "Agreements governing intellectual property, software, data, technology services, and related rights.",
    "consumer": "Agreements for products or services supplied to individuals, subject to mandatory consumer-protection law.",
    "government": "Agreements involving public procurement, grants, and flow-down requirements.",
    "healthcare": "Agreements concerning health services, protected data, research, providers, and regulated relationships.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc
    return rows


def build() -> dict[str, Any]:
    clauses = read_jsonl(SOURCE_DIR / "clause_definitions.jsonl")
    contract_types = read_jsonl(SOURCE_DIR / "contract_types.jsonl")
    clause_ids = [row["id"] for row in clauses]
    contract_type_ids = [row["id"] for row in contract_types]
    if len(clause_ids) != len(set(clause_ids)):
        raise ValueError("Clause IDs must be unique")
    if len(contract_type_ids) != len(set(contract_type_ids)):
        raise ValueError("Contract type IDs must be unique")
    known_clauses = set(clause_ids)
    for row in contract_types:
        if row["domain"] not in DOMAIN_DEFINITIONS:
            raise ValueError(f"Unknown domain on contract type {row['id']}: {row['domain']}")
        unknown = sorted(set(row["typical_clause_ids"]) - known_clauses)
        if unknown:
            raise ValueError(f"{row['id']} references unknown clause IDs: {unknown}")

    domains = []
    for domain_id, definition in DOMAIN_DEFINITIONS.items():
        domain_types = [row for row in contract_types if row["domain"] == domain_id]
        domains.append(
            {
                "id": domain_id,
                "name": domain_id.replace("_", " ").title(),
                "definition": definition,
                "contract_type_ids": [row["id"] for row in domain_types],
            }
        )
    return {
        "metadata": {
            "title": "AI-Enabled Contracts Lab Contract Type and Clause Taxonomy",
            "version": "0.1.0",
            "created": "2026-08-11",
            "license": "CC BY 4.0",
            "author": "AI-Enabled Contracts Lab contributors",
            "scope": "Non-exhaustive, jurisdiction-neutral educational taxonomy",
            "limitations": "Definitions describe common drafting functions, not legal requirements or advice. Applicability and enforceability vary by jurisdiction, facts, parties, and document purpose.",
        },
        "domains": domains,
        "contract_types": contract_types,
        "clause_definitions": clauses,
        "source_notes": [
            {
                "title": "CUAD v1",
                "url": "https://github.com/The-Atticus-Project/cuad",
                "use": "Commercial clause-category alignment",
            },
            {
                "title": "ContractNLI",
                "url": "https://stanfordnlp.github.io/contract-nli/",
                "use": "NDA document-level hypothesis and evidence task design",
            },
            {
                "title": "LEGAL-BERT Contracts",
                "url": "https://huggingface.co/nlpaueb/bert-base-uncased-contracts",
                "use": "Contract-domain encoder comparison",
            },
        ],
    }


def main() -> None:
    payload = build()
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[OK] wrote {OUTPUT} with {len(payload['domains'])} domains, "
        f"{len(payload['contract_types'])} contract types, and "
        f"{len(payload['clause_definitions'])} clause definitions"
    )


if __name__ == "__main__":
    main()
