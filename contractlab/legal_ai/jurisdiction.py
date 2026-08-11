from __future__ import annotations

import re

from contractlab.legal_ai.schemas import JurisdictionResult


def route_jurisdiction(text: str, supplied: str | None = None) -> JurisdictionResult:
    if supplied and supplied.strip():
        return JurisdictionResult(
            likely_jurisdiction=supplied.strip(),
            confidence=0.85,
            supporting_facts=["Jurisdiction supplied by the user."],
            missing_facts=["Confirm governing-law, forum, and conflict-of-law implications."],
        )

    lowered = text.casefold()
    signals: list[tuple[str, str]] = []
    patterns = {
        "New Jersey": (r"new jersey", r"state of new jersey"),
        "New York": (r"new york", r"state of new york"),
        "Delaware": (r"delaware", r"state of delaware"),
        "California": (r"california", r"state of california"),
        "U.S. federal": (r"united states district court", r"federal law", r"u\.s\.c\."),
        "International or cross-border": (r"international", r"foreign law", r"un convention"),
    }
    for name, cues in patterns.items():
        if any(re.search(cue, lowered) for cue in cues):
            signals.append((name, f"Text contains a {name} jurisdiction signal."))

    if not signals:
        return JurisdictionResult(
            likely_jurisdiction=None,
            confidence=0.0,
            missing_facts=[
                "Governing-law clause",
                "Forum or venue clause",
                "Locations and legal status of the parties",
                "Location of performance and relevant conduct",
            ],
        )

    likely, reason = signals[0]
    return JurisdictionResult(
        likely_jurisdiction=likely,
        alternative_jurisdictions=[name for name, _ in signals[1:]],
        confidence=0.65 if len(signals) == 1 else 0.45,
        supporting_facts=[reason],
        missing_facts=["Confirm that the detected text controls the issue being analyzed."],
    )
