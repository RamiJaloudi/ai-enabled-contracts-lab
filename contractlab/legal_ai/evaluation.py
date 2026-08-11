from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from contractlab.legal_ai.authority import extract_authorities
from contractlab.legal_ai.schemas import LegalAnalysisResult


@dataclass(frozen=True, slots=True)
class LegalQualityMetrics:
    issue_recovery_rate: float
    false_issue_rate: float
    jurisdiction_accuracy: float
    structured_output_validity: float
    missing_fact_coverage: float
    counterargument_coverage: float
    evidence_grounding_rate: float
    unsupported_authority_rate: float

    @property
    def release_gate_passed(self) -> bool:
        return self.structured_output_validity == 1.0 and self.unsupported_authority_rate == 0.0


def _normalized(values: Iterable[str]) -> set[str]:
    return {value.strip().casefold() for value in values if value.strip()}


def evaluate_legal_quality(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> LegalQualityMetrics:
    if len(predictions) != len(references) or not references:
        raise ValueError('Predictions and references must be non-empty and aligned.')
    recovered = expected = false_issues = predicted_count = valid = 0
    jurisdiction_correct = missing_covered = counters_covered = evidence_grounded = 0
    authority_count = unsupported_count = issue_count = 0

    for raw_prediction, reference in zip(predictions, references, strict=True):
        try:
            result = LegalAnalysisResult.model_validate(raw_prediction)
            valid += 1
        except ValidationError:
            continue
        predicted_labels = _normalized(issue.taxonomy_label for issue in result.issues)
        expected_labels = _normalized(reference.get('issues', []))
        recovered += len(predicted_labels & expected_labels)
        expected += len(expected_labels)
        false_issues += len(predicted_labels - expected_labels)
        predicted_count += len(predicted_labels)
        jurisdiction_correct += int(
            (result.jurisdiction or '').casefold()
            == str(reference.get('jurisdiction') or '').casefold()
        )
        for issue in result.issues:
            issue_count += 1
            missing_covered += int(bool(issue.missing_facts))
            counters_covered += int(bool(issue.counterarguments))
            evidence_grounded += int(
                not reference.get('requires_evidence_ids') or bool(issue.evidence_passage_ids)
            )
            issue_text = ' '.join([issue.rule, issue.application, issue.conclusion])
            authorities = extract_authorities(issue_text)
            authority_count += len(authorities)
            allowed = _normalized(reference.get('supplied_authorities', []))
            unsupported_count += sum(item.casefold() not in allowed for item in authorities)

    count = len(references)
    return LegalQualityMetrics(
        issue_recovery_rate=recovered / expected if expected else 1.0,
        false_issue_rate=false_issues / predicted_count if predicted_count else 0.0,
        jurisdiction_accuracy=jurisdiction_correct / count,
        structured_output_validity=valid / count,
        missing_fact_coverage=missing_covered / issue_count if issue_count else 0.0,
        counterargument_coverage=counters_covered / issue_count if issue_count else 0.0,
        evidence_grounding_rate=evidence_grounded / issue_count if issue_count else 0.0,
        unsupported_authority_rate=(
            unsupported_count / authority_count if authority_count else 0.0
        ),
    )
