from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from contractlab.paths import PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class PlaybookClause:
    key: str
    name: str
    why_it_matters: str
    keywords: tuple[str, ...]
    review_questions: tuple[str, ...]


NDA_CLAUSES: tuple[PlaybookClause, ...] = (
    PlaybookClause(
        "definition",
        "Definition of confidential information",
        "Defines the information protected by the NDA.",
        (
            "confidential information means",
            "confidential information includes",
            "non-public information",
        ),
        ("Is the definition clear?", "Does it cover oral and written disclosures?"),
    ),
    PlaybookClause(
        "exclusions",
        "Exclusions from confidentiality",
        "Carves out information that is public, previously known, independently developed, or lawfully received.",
        (
            "shall not include",
            "publicly available",
            "already known",
            "independently developed",
            "rightfully received",
        ),
        ("Are the standard exclusions present?", "Who must prove that an exclusion applies?"),
    ),
    PlaybookClause(
        "purpose",
        "Permitted purpose and use restriction",
        "Limits use of confidential information to the stated business purpose.",
        (
            "solely for",
            "only for the purpose",
            "permitted purpose",
            "use the confidential information",
        ),
        (
            "Is the permitted purpose specific enough?",
            "Does the restriction cover competitive or commercial use?",
        ),
    ),
    PlaybookClause(
        "nondisclosure",
        "Non-disclosure obligation",
        "Restricts disclosure to unauthorized people or organizations.",
        ("not disclose", "shall not disclose", "keep confidential", "maintain in confidence"),
        ("Who may receive the information?", "Is the obligation mutual or one-way?"),
    ),
    PlaybookClause(
        "representatives",
        "Permitted disclosures to representatives",
        "Allows limited disclosure to personnel and advisers with a need to know.",
        (
            "need to know",
            "representatives",
            "employees and advisors",
            "officers, directors, employees",
        ),
        (
            "Are representatives bound by comparable duties?",
            "Who is responsible for their breaches?",
        ),
    ),
    PlaybookClause(
        "compelled",
        "Required or compelled disclosure",
        "Addresses subpoenas, court orders, and other legally required disclosures.",
        (
            "required by law",
            "court order",
            "subpoena",
            "legally compelled",
            "governmental authority",
        ),
        (
            "Is advance notice required when legally permitted?",
            "Must disclosure be limited to what is required?",
        ),
    ),
    PlaybookClause(
        "care",
        "Standard of care and safeguards",
        "States how the recipient must protect confidential information.",
        ("reasonable care", "same degree of care", "protect against", "security measures"),
        ("Is the care standard objective?", "Are technical or organizational safeguards required?"),
    ),
    PlaybookClause(
        "return",
        "Return or destruction",
        "Controls what happens to confidential information when the relationship or NDA ends.",
        ("return or destroy", "return and destroy", "destroy all", "return all copies"),
        ("Are backups and legal-retention copies addressed?", "Is written certification required?"),
    ),
    PlaybookClause(
        "term",
        "Term and survival",
        "Sets the NDA duration and how long confidentiality duties continue.",
        ("survive", "for a period of", "years after", "term of this agreement", "trade secret"),
        (
            "Is the duration appropriate for ordinary information?",
            "Are trade secrets treated separately?",
        ),
    ),
    PlaybookClause(
        "ownership",
        "No license and ownership",
        "Preserves ownership and limits implied intellectual-property rights.",
        (
            "no license",
            "remains the property",
            "no right, title or interest",
            "intellectual property rights",
        ),
        (
            "Does disclosure grant any express or implied license?",
            "Are feedback and residual knowledge addressed?",
        ),
    ),
    PlaybookClause(
        "remedies",
        "Remedies and injunctive relief",
        "Describes potential relief for unauthorized use or disclosure.",
        ("injunctive relief", "irreparable harm", "equitable relief", "specific performance"),
        ("Are remedies mutual?", "Does the clause preserve other available remedies?"),
    ),
    PlaybookClause(
        "law",
        "Governing law and forum",
        "Identifies the law and forum intended to govern disputes.",
        ("governed by the laws", "governing law", "exclusive jurisdiction", "venue"),
        ("Is the jurisdiction approved?", "Are forum and governing law consistent?"),
    ),
)


def identify_contract_profile(text: str) -> str:
    lowered = text.casefold()
    explicit = any(
        phrase in lowered
        for phrase in (
            "non-disclosure agreement",
            "nondisclosure agreement",
            "confidentiality agreement",
            "mutual nda",
            "unilateral nda",
        )
    )
    detected = sum(any(keyword in lowered for keyword in clause.keywords) for clause in NDA_CLAUSES)
    if explicit:
        return "Non-Disclosure Agreement (NDA)"
    if detected >= 5:
        return "NDA-like confidentiality agreement"
    return "General agreement"


def analyze_nda_playbook(text: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    profile = identify_contract_profile(text)
    rows: list[dict[str, Any]] = []
    for clause in NDA_CLAUSES:
        matches: list[dict[str, Any]] = []
        for finding in findings:
            lowered = str(finding.get("text", "")).casefold()
            score = sum(keyword in lowered for keyword in clause.keywords)
            if score:
                matches.append({"score": score, "finding": finding})
        matches.sort(key=lambda item: int(item["score"]), reverse=True)
        evidence = [item["finding"] for item in matches[:2]]
        rows.append(
            {
                **asdict(clause),
                "status": "Detected" if evidence else "Not detected",
                "evidence_ids": [item.get("passage_id") for item in evidence],
                "evidence": [item.get("text") for item in evidence],
            }
        )
    detected_count = sum(row["status"] == "Detected" for row in rows)
    direction = (
        "Mutual"
        if any(term in text.casefold() for term in ("each party", "both parties", "mutual"))
        else "Unclear or unilateral"
    )
    return {
        "profile": profile,
        "direction": direction,
        "detected_count": detected_count,
        "expected_count": len(rows),
        "coverage_ratio": detected_count / len(rows),
        "clauses": rows,
        "disclaimer": "Not detected means the playbook did not find clear language; it is not a legal conclusion that a clause is absent or required.",
    }


GENERAL_CLAUSE_IDS = (
    "parties",
    "definitions",
    "scope",
    "fees_payment",
    "term",
    "termination",
    "confidentiality",
    "ip_ownership",
    "warranties",
    "indemnity",
    "liability_limit",
    "governing_law",
)

CONTRACT_TYPE_CUES: dict[str, tuple[str, ...]] = {
    "nda": (
        "non-disclosure agreement",
        "nondisclosure agreement",
        "confidentiality agreement",
        "mutual nda",
    ),
    "master_services": (
        "master services agreement",
        "services agreement",
        "software services agreement",
        "professional services agreement",
    ),
    "statement_of_work": ("statement of work", "work order"),
    "saas": ("saas agreement", "software-as-a-service", "subscription agreement"),
    "supply": ("supply agreement", "supplier agreement"),
    "distribution": ("distribution agreement", "distributor agreement"),
    "employment_agreement": ("employment agreement",),
    "offer_letter": ("offer letter",),
    "independent_contractor": ("independent contractor agreement", "consulting agreement"),
    "employment_separation": ("separation agreement", "severance agreement"),
    "commercial_lease": ("commercial lease",),
    "residential_lease": ("residential lease", "rental agreement"),
    "real_estate_purchase": ("real estate purchase agreement", "purchase and sale agreement"),
    "loan": ("loan agreement",),
    "promissory_note": ("promissory note",),
    "security_agreement": ("security agreement",),
    "guaranty": ("guaranty", "guarantee agreement"),
    "software_license": ("software license agreement",),
    "data_processing_addendum": ("data processing addendum", "data protection addendum"),
    "stock_purchase": ("stock purchase agreement", "share purchase agreement"),
    "asset_purchase": ("asset purchase agreement",),
    "merger": ("merger agreement", "agreement and plan of merger"),
    "prenuptial": ("prenuptial agreement", "premarital agreement"),
    "marital_settlement": ("marital settlement agreement",),
    "parenting_plan": ("parenting plan",),
}

CLAUSE_CUES: dict[str, tuple[str, ...]] = {
    "parties": ("between", "party", "parties"),
    "definitions": (" means ", "defined as", "definitions"),
    "scope": ("services", "goods", "scope", "deliverables"),
    "fees_payment": ("invoice", "payment", "fees", "shall pay"),
    "term": ("term", "commence", "effective date"),
    "termination": ("terminate", "termination", "material breach"),
    "renewal": ("renew", "renewal", "successive term"),
    "confidentiality": ("confidential", "non-public", "not disclose"),
    "ip_ownership": ("intellectual property", "ownership", "retains"),
    "warranties": ("warrant", "representation"),
    "indemnity": ("indemnif", "hold harmless", "defend"),
    "liability_limit": ("limitation of liability", "aggregate liability", "consequential damages"),
    "governing_law": ("governed by", "governing law", "jurisdiction"),
    "assignment": ("assign", "assignment"),
    "force_majeure": ("force majeure", "beyond its reasonable control"),
    "privacy_security": ("personal data", "privacy", "security measures"),
    "return_destruction": ("return or destroy", "destroy all", "return all copies"),
}


@lru_cache(maxsize=1)
def load_contract_taxonomy(path: Path | None = None) -> dict[str, Any]:
    target = path or PROJECT_ROOT / "datasets" / "contract_taxonomy.json"
    return json.loads(target.read_text(encoding="utf-8"))


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def identify_contract_type(text: str, taxonomy: dict[str, Any]) -> dict[str, Any] | None:
    header = _normalized(text[:1200])
    stopwords = {"agreement", "contract", "the", "and", "of", "for"}
    ranked: list[tuple[int, dict[str, Any]]] = []
    for contract_type in taxonomy.get("contract_types", []):
        score = 0
        name = _normalized(str(contract_type["name"]))
        if name and name in header:
            score += 40
        for cue in CONTRACT_TYPE_CUES.get(str(contract_type["id"]), ()):
            if _normalized(cue) in header:
                score += 25
        terms = set(_normalized(str(contract_type["id"])).split())
        terms.update(name.split())
        terms -= stopwords
        score += sum(term in header.split() for term in terms)
        ranked.append((score, contract_type))
    score, best = max(ranked, key=lambda item: item[0])
    return best if score >= 3 else None


def analyze_contract_playbook(text: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare any contract with its taxonomy-defined common clause set."""
    taxonomy = load_contract_taxonomy()
    contract_type = identify_contract_type(text, taxonomy)
    clauses_by_id = {item["id"]: item for item in taxonomy.get("clause_definitions", [])}
    clause_ids = (
        contract_type.get("typical_clause_ids", GENERAL_CLAUSE_IDS)
        if contract_type
        else GENERAL_CLAUSE_IDS
    )
    rows: list[dict[str, Any]] = []
    for clause_id in clause_ids:
        clause = clauses_by_id.get(clause_id)
        if not clause:
            continue
        terms = [str(clause["name"]), *clause.get("aliases", []), *CLAUSE_CUES.get(clause_id, ())]
        normalized_terms = [_normalized(term) for term in terms if len(_normalized(term)) >= 4]
        evidence: list[dict[str, Any]] = []
        for finding in findings:
            passage = _normalized(str(finding.get("text", "")))
            label = _normalized(str(finding.get("label", "")))
            clause_name = _normalized(str(clause["name"]))
            label_match = label == clause_name or label == _normalized(clause_id)
            text_match = any(term in passage for term in normalized_terms)
            if label_match or text_match:
                evidence.append(finding)
        rows.append(
            {
                **clause,
                "status": "Detected" if evidence else "Not detected",
                "evidence_ids": [item.get("passage_id") for item in evidence[:3]],
                "evidence": [item.get("text") for item in evidence[:3]],
            }
        )
    detected_count = sum(row["status"] == "Detected" for row in rows)
    profile = contract_type["name"] if contract_type else "General agreement"
    return {
        "profile": profile,
        "contract_type_id": contract_type.get("id") if contract_type else None,
        "domain": contract_type.get("domain", "cross-domain") if contract_type else "cross-domain",
        "definition": contract_type.get(
            "definition",
            "A contract whose specific type was not identified with enough textual evidence.",
        )
        if contract_type
        else "A contract whose specific type was not identified with enough textual evidence.",
        "detected_count": detected_count,
        "expected_count": len(rows),
        "coverage_ratio": detected_count / len(rows) if rows else 0.0,
        "clauses": rows,
        "disclaimer": "Not detected means this educational matcher did not find clear language. It does not mean a clause is legally required, actually absent, deficient, or unenforceable.",
    }
