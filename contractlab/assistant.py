from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from contractlab.playbooks import (
    analyze_contract_playbook,
    analyze_nda_playbook,
    load_contract_taxonomy,
)


@dataclass(slots=True)
class PassageFinding:
    passage_id: str
    text: str
    start: int
    end: int
    label: str
    confidence: float
    explanation: str
    playbook_status: str


KEYWORDS: dict[str, tuple[str, ...]] = {
    "Confidentiality": ("confidential", "non-public", "disclos", "receiving party"),
    "Governing Law": ("governed by", "governing law", "jurisdiction", "venue"),
    "Termination": ("terminate", "termination", "material breach", "cure period"),
    "Limitation of Liability": (
        "limitation of liability",
        "consequential damages",
        "aggregate liability",
        "liable",
    ),
    "Indemnification": ("indemnif", "defend", "third-party claim", "hold harmless"),
    "Payment Terms": ("invoice", "payment", "fees", "late amounts"),
    "Intellectual Property": (
        "intellectual property",
        "ownership",
        "license grant",
        "customer data",
    ),
    "Renewal": ("renew", "renewal", "successive term", "auto-renew"),
}

EXPLANATIONS = {
    "Confidentiality": "Controls how protected information may be used or disclosed.",
    "Governing Law": "Names the law or forum intended to govern disputes.",
    "Termination": "Describes when a party may end the agreement and any cure period.",
    "Limitation of Liability": "Limits categories or amounts of recoverable damages.",
    "Indemnification": "Allocates responsibility for specified third-party claims or losses.",
    "Payment Terms": "States fees, invoicing timing, or payment obligations.",
    "Intellectual Property": "Allocates ownership or permitted use of intellectual property and data.",
    "Renewal": "Describes extension of the term and notice required to prevent renewal.",
    "Other": "No supported clause category was identified with enough evidence.",
}

DEFAULT_PLAYBOOK = {
    "Confidentiality": "Confirm permitted disclosures, exclusions, duration, and return or destruction obligations.",
    "Governing Law": "Confirm the chosen law and forum match the organization's approved jurisdictions.",
    "Termination": "Check breach cure periods, termination for convenience, and post-termination duties.",
    "Limitation of Liability": "Review the cap, excluded damages, carve-outs, and whether limitations are mutual.",
    "Indemnification": "Review covered claims, control of defense, notice, cooperation, and exclusions.",
    "Payment Terms": "Confirm fee schedule, invoice disputes, payment period, taxes, and late charges.",
    "Intellectual Property": "Confirm background IP, deliverable ownership, license scope, and data rights.",
    "Renewal": "Confirm renewal duration, price changes, and notice deadline.",
}


def split_passages(text: str, minimum_length: int = 35) -> list[dict[str, Any]]:
    pattern = re.compile(r"(?:\r?\n){2,}|(?<=[.;])\s+(?=(?:\d+\.?\s+|[A-Z][A-Za-z ]{2,25}:))")
    results: list[dict[str, Any]] = []
    cursor = 0
    for piece in pattern.split(text):
        clean = piece.strip()
        if len(clean) < minimum_length:
            continue
        start = text.find(clean, cursor)
        if start < 0:
            start = cursor
        end = start + len(clean)
        results.append(
            {"passage_id": f"P{len(results) + 1}", "text": clean, "start": start, "end": end}
        )
        cursor = end
    if not results and text.strip():
        clean = text.strip()
        results.append(
            {
                "passage_id": "P1",
                "text": clean,
                "start": text.find(clean),
                "end": text.find(clean) + len(clean),
            }
        )
    return results


def heuristic_prediction(text: str) -> dict[str, Any]:
    taxonomy_version = str(
        load_contract_taxonomy().get("metadata", {}).get("version") or "unknown"
    )
    provenance = {
        "model_version": "heuristic-keyword-v1",
        "taxonomy_version": taxonomy_version,
    }
    lowered = text.lower()
    scores = {
        label: sum(keyword in lowered for keyword in keywords)
        for label, keywords in KEYWORDS.items()
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return {"label": "Other", "confidence": 0.25, **provenance}
    return {
        "label": label,
        "confidence": min(0.55 + score * 0.1, 0.9),
        **provenance,
    }


def extract_facts(text: str) -> dict[str, list[str]]:
    dates = sorted(
        set(
            re.findall(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    money = sorted(set(re.findall(r"\$\s?\d[\d,]*(?:\.\d{2})?", text)))
    obligations = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if re.search(r"\b(?:shall|must|will be required to)\b", sentence, flags=re.IGNORECASE)
    ][:20]
    defined_parties = sorted(
        set(re.findall(r"(?:between|by and between)\s+([^.;]{3,120})", text, flags=re.IGNORECASE))
    )
    return {
        "dates": dates,
        "money": money,
        "obligations": obligations,
        "party_phrases": defined_parties,
    }


def analyze_contract(
    text: str,
    predictor: Callable[[list[str]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    passages = split_passages(text)
    predictions = (
        predictor([passage["text"] for passage in passages])
        if predictor and passages
        else [heuristic_prediction(passage["text"]) for passage in passages]
    )
    findings: list[PassageFinding] = []
    for passage, prediction in zip(passages, predictions, strict=True):
        label = str(prediction["label"])
        confidence = float(prediction["confidence"])
        fallback = heuristic_prediction(passage["text"])
        fallback_is_supported = fallback["label"] != "Other"
        checkpoint_is_weak = confidence < 0.55
        checkpoint_is_out_of_scope = label not in EXPLANATIONS
        if fallback_is_supported and (checkpoint_is_weak or checkpoint_is_out_of_scope):
            label = str(fallback["label"])
            confidence = float(fallback["confidence"])
        findings.append(
            PassageFinding(
                **passage,
                label=label,
                confidence=confidence,
                explanation=EXPLANATIONS.get(label, EXPLANATIONS["Other"]),
                playbook_status="Review" if label in DEFAULT_PLAYBOOK else "Not assessed",
            )
        )
    observed = {finding.label for finding in findings}
    missing = [label for label in DEFAULT_PLAYBOOK if label not in observed]
    finding_rows = [asdict(finding) for finding in findings]
    nda_analysis = analyze_nda_playbook(text, finding_rows)
    contract_analysis = analyze_contract_playbook(text, finding_rows)
    return {
        "findings": finding_rows,
        "facts": extract_facts(text),
        "missing_categories": missing,
        "playbook": DEFAULT_PLAYBOOK,
        "engine": (
            "hybrid checkpoint and transparent playbook"
            if predictor
            else "transparent keyword baseline"
        ),
        "contract_profile": contract_analysis["profile"],
        "contract_analysis": contract_analysis,
        "nda_analysis": nda_analysis,
    }


def answer_with_passages(
    question: str, findings: list[dict[str, Any]], limit: int = 3
) -> list[dict[str, Any]]:
    terms = {term.lower() for term in re.findall(r"[A-Za-z]{3,}", question)}
    ranked: list[tuple[int, dict[str, Any]]] = []
    for finding in findings:
        text_terms = set(re.findall(r"[A-Za-z]{3,}", finding["text"].lower()))
        score = len(terms & text_terms)
        if finding["label"].lower() in question.lower():
            score += 3
        ranked.append((score, finding))
    return [
        finding
        for score, finding in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]
        if score > 0
    ]


def _evidence_excerpt(text: str, limit: int = 480) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def contract_chat_response(prompt: str, result: dict[str, Any]) -> str:
    """Answer contract questions from the analysis result with passage citations."""
    normalized = prompt.casefold()
    findings = result.get("findings", [])
    contract = result.get("contract_analysis", {})
    contract_clauses = contract.get("clauses", [])
    detected = [row for row in contract_clauses if row.get("status") == "Detected"]
    missing = [row for row in contract_clauses if row.get("status") != "Detected"]

    if any(term in normalized for term in ("what type", "contract type", "kind of contract")):
        return (
            f"### Contract profile\n\nThe text most closely matches "
            f"**{contract.get('profile', 'General agreement')}** in the "
            f"**{contract.get('domain', 'cross-domain')}** domain. "
            f"{contract.get('definition', '')}\n\nContract-type matching is an educational "
            "text comparison, not a legal characterization."
        )

    if any(term in normalized for term in ("what is an nda", "what is a nda", "what is nda")):
        return (
            "An **NDA (non-disclosure agreement)** is one type of contract used to restrict "
            "use or disclosure of confidential information for a defined purpose. Common "
            "review concepts include the protected-information definition and exclusions, "
            "permitted use, disclosure to representatives, compelled disclosure, safeguards, "
            "return or destruction, survival, ownership, remedies, and governing law."
        )

    if (
        "nda" in normalized
        or "non-disclosure" in normalized
        or "confidentiality agreement" in normalized
    ):
        nda = result.get("nda_analysis", {})
        nda_clauses = nda.get("clauses", [])
        nda_detected = [row for row in nda_clauses if row.get("status") == "Detected"]
        nda_missing = [row for row in nda_clauses if row.get("status") != "Detected"]
        detected_names = ", ".join(str(row["name"]) for row in nda_detected[:6]) or "none"
        missing_names = ", ".join(str(row["name"]) for row in nda_missing[:6]) or "none"
        return (
            f"### NDA-specific analysis\n\nThe text profile is "
            f"**{nda.get('profile', 'unknown')}** and the NDA matcher detected "
            f"**{nda.get('detected_count', 0)} of {nda.get('expected_count', 0)}** common "
            f"review categories.\n\n**Detected:** {detected_names}.\n\n"
            f"**Not clearly detected:** {missing_names}.\n\n{nda.get('disclaimer', '')}"
        )

    if (
        any(
            term in normalized
            for term in ("clause", "provision", "what was found", "what is included")
        )
        and "missing" not in normalized
    ):
        if not detected:
            return "No taxonomy-defined clause category was detected with enough evidence."
        rows = []
        for row in detected[:12]:
            evidence_ids = ", ".join(row.get("evidence_ids", [])) or "no passage ID"
            rows.append(f"- **{row['name']}** [{evidence_ids}]: {row.get('definition', '')}")
        return "### Clauses detected\n\n" + "\n".join(rows)

    if any(
        term in normalized
        for term in (
            "add",
            "addition",
            "amend",
            "draft",
            "improve",
            "propose",
            "recommend",
            "revise",
            "rewrite",
            "strengthen",
            "suggest",
        )
    ):
        if not missing:
            return (
                "The playbook did not identify an undetected common category. A human reviewer "
                "should still assess whether the detected language needs revision for the "
                "parties' actual business and jurisdictional requirements."
            )
        recommendations = []
        for index, row in enumerate(missing[:3], start=1):
            question = row.get("review_questions", ["Should this topic be addressed?"])[0]
            recommendations.append(
                f"{index}. **{row['name']}** — {row.get('definition', '')}\n"
                f"   - **Decision for the parties:** {question}"
            )
        return (
            "### Candidate additions for human review\n\n"
            + "\n\n".join(recommendations)
            + "\n\nThese are prioritized playbook candidates, not legal requirements. The local "
            "fallback does not generate negotiated clause language; use the generative model or "
            "qualified counsel for drafting tailored to the transaction and jurisdiction."
        )

    if any(term in normalized for term in ("missing", "gap", "absent", "not included")):
        if not missing:
            return (
                "Every common taxonomy category configured for this detected contract type "
                "had some matching evidence, but each passage still requires human review."
            )
        details = "\n".join(
            f"- **{row['name']}** - {row.get('definition', '')}" for row in missing[:12]
        )
        return (
            f"### Common categories not clearly detected\n\n{details}\n\n"
            f"{contract.get('disclaimer', '')}"
        )

    if any(term in normalized for term in ("risk", "review", "concern", "red flag")):
        if not detected:
            return "No supported clause evidence is available yet. Analyze the contract first."
        items = []
        for row in detected[:10]:
            evidence_ids = ", ".join(row.get("evidence_ids", [])) or "no passage ID"
            question = row.get("review_questions", ["Review the cited language."])[0]
            items.append(f"- **{row['name']}** [{evidence_ids}]: {question}")
        return (
            "### Human-review prompts\n\n"
            + "\n".join(items)
            + "\n\nThese are review questions, not conclusions that a provision is risky, "
            "deficient, enforceable, or unenforceable."
        )

    if "summary" in normalized or "summarize" in normalized:
        labeled = [finding for finding in findings if finding.get("label") != "Other"]
        items = [
            f"- **[{finding['passage_id']}] {finding['label']}**: "
            f"{_evidence_excerpt(str(finding['text']), 220)}"
            for finding in labeled[:10]
        ]
        if not items:
            return (
                "The analyzer did not identify enough supported passages to build a cited summary."
            )
        return "### Cited contract summary\n\n" + "\n".join(items)

    if any(term in normalized for term in ("obligation", "must", "shall", "duty")):
        obligations = [
            finding
            for finding in findings
            if re.search(
                r"\b(?:shall|must|will be required to)\b",
                str(finding.get("text", "")),
                re.I,
            )
        ]
        if not obligations:
            return "No obligation language was detected in the segmented passages."
        return "### Obligation language\n\n" + "\n".join(
            f"- **[{finding['passage_id']}]** {_evidence_excerpt(str(finding['text']))}"
            for finding in obligations[:10]
        )

    answers = answer_with_passages(prompt, findings)
    if not answers:
        return (
            "I could not find a sufficiently matching passage. Try naming a contract type, "
            "clause, party, date, obligation, or review topic. I will not fill a gap with "
            "unstated language."
        )
    response = ["### Closest supporting passages"]
    for answer in answers:
        response.append(
            f"**[{answer['passage_id']}] {answer['label']}**\n\n> "
            f"{_evidence_excerpt(str(answer['text']))}"
        )
    response.append(
        "This response retrieves and organizes source language; it does not determine legal effect."
    )
    return "\n\n".join(response)
