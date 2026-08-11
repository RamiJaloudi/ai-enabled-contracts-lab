from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ClassifierPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str = "unknown"
    taxonomy_version: str = "unknown"


    prediction_type: Literal['domain', 'doctrine'] = 'doctrine'
    rank: int = Field(default=1, ge=1)
    calibrated: bool = False
    calibration_version: str = 'uncalibrated'
    passage_ids: list[str] = Field(default_factory=list)


class ClassificationRoute(BaseModel):
    model_config = ConfigDict(extra='forbid')

    status: Literal['AUTO_ROUTE', 'REVIEW_REQUIRED', 'ABSTAIN']
    selected_domain: str | None = None
    domain_predictions: list[ClassifierPrediction] = Field(default_factory=list)
    doctrine_predictions: list[ClassifierPrediction] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AuthorityReference(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str = Field(min_length=1)
    status: Literal['USER_SUPPLIED', 'EXTERNALLY_VERIFIED', 'UNVERIFIED']
    verification_required: bool = True


class LegalIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taxonomy_label: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    analysis_confidence: float = Field(ge=0.0, le=1.0)
    rule: str = Field(min_length=1)
    supporting_facts: list[str]
    adverse_facts: list[str]
    missing_facts: list[str]
    application: str = Field(min_length=1)
    counterarguments: list[str]
    jurisdiction_status: Literal[
        "GENERAL_PRINCIPLE",
        "JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED",
        "VERIFIED",
    ]
    conclusion: str = Field(min_length=1)


    evidence_passage_ids: list[str] = Field(default_factory=list)
    confidence_basis: str | None = None
    authorities: list[AuthorityReference] = Field(default_factory=list)


class AnalysisAudit(BaseModel):
    model_config = ConfigDict(extra='forbid')

    analysis_id: str
    created_at: datetime
    input_sha256: str
    provider: str
    generative_model: str
    classifier_model_versions: list[str] = Field(default_factory=list)
    taxonomy_versions: list[str] = Field(default_factory=list)
    prompt_sha256: str
    skill_sha256: str
    schema_sha256: str
    route_status: Literal['AUTO_ROUTE', 'REVIEW_REQUIRED', 'ABSTAIN']
    configured_thresholds: dict[str, float] = Field(default_factory=dict)
    validation_attempts: int = Field(ge=1)
    used_model_fallback: bool = False
    input_truncated: bool = False
    passage_count: int = Field(default=0, ge=0)
    security_findings: list[str] = Field(default_factory=list)


class LegalAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    jurisdiction: str | None = None
    jurisdiction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    classifier_predictions: list[ClassifierPrediction] = Field(default_factory=list)
    issues: list[LegalIssue]


    domain_predictions: list[ClassifierPrediction] = Field(default_factory=list)
    doctrine_predictions: list[ClassifierPrediction] = Field(default_factory=list)
    route_status: Literal['AUTO_ROUTE', 'REVIEW_REQUIRED', 'ABSTAIN'] = 'REVIEW_REQUIRED'
    route_reasons: list[str] = Field(default_factory=list)
    audit: AnalysisAudit | None = None


class JurisdictionResult(BaseModel):
    likely_jurisdiction: str | None
    alternative_jurisdictions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_facts: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    verification_required: bool = True


class ChatResult(BaseModel):
    content: str
    model: str
    provider: str
    used_fallback: bool = False


class DocumentPassage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    passage_id: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    parent_passage_id: str | None = None


class ContractDraftSection(BaseModel):
    model_config = ConfigDict(extra='forbid')

    heading: str = Field(min_length=1)
    paragraphs: list[str] = Field(min_length=1)


class ContractDraft(BaseModel):
    model_config = ConfigDict(extra='forbid')

    title: str = Field(min_length=1)
    contract_type: str = Field(min_length=1)
    jurisdiction: str | None = None
    parties: list[str] = Field(default_factory=list)
    effective_date: str | None = None
    purpose: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    sections: list[ContractDraftSection] = Field(min_length=1)
    drafting_note: str = (
        'Draft for review. Jurisdiction-specific enforceability and current authority require '
        'verification by qualified counsel.'
    )

    def as_text(self) -> str:
        lines = [self.title, 'DRAFT FOR REVIEW', '']
        if self.parties:
            lines.extend(['Parties', *[f'- {party}' for party in self.parties], ''])
        if self.effective_date:
            lines.extend([f'Effective date: {self.effective_date}', ''])
        lines.extend([self.purpose, ''])
        for index, section in enumerate(self.sections, start=1):
            lines.append(f'{index}. {section.heading}')
            lines.extend(section.paragraphs)
            lines.append('')
        if self.unresolved_terms:
            lines.extend(
                [
                    'Terms requiring party or counsel input',
                    *[f'- {term}' for term in self.unresolved_terms],
                    '',
                ]
            )
        if self.assumptions:
            lines.extend(['Drafting assumptions', *[f'- {item}' for item in self.assumptions], ''])
        lines.extend(['Drafting note', self.drafting_note])
        return '\n'.join(lines).strip() + '\n'


class DraftGenerationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    draft: ContractDraft
    model: str
    provider: str
    used_fallback: bool = False
    validation_attempts: int = Field(default=1, ge=1)
    audit: dict[str, Any] = Field(default_factory=dict)
