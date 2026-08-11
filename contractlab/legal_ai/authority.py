from __future__ import annotations

import re

from contractlab.legal_ai.schemas import AuthorityReference, LegalAnalysisResult

AUTHORITY_PATTERNS = (
    re.compile(r'\b[A-Z][A-Za-z.&-]+\s+v\.\s+[A-Z][A-Za-z.&-]+\b'),
    re.compile(r'\b\d+\s+(?:U\.S\.|F\.(?:2d|3d|4th)|N\.J\.|A\.(?:2d|3d))\s+\d+\b'),
    re.compile(r'\b\d+\s+U\.S\.C\.\s+§*\s*\d+[A-Za-z0-9().-]*\b'),
    re.compile(r'\b(?:N\.J\.S\.A\.|C\.F\.R\.)\s*§*\s*[A-Za-z0-9:().-]+\b'),
)


def extract_authorities(text: str) -> list[str]:
    found: list[str] = []
    for pattern in AUTHORITY_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if value not in found:
                found.append(value)
    return found


def contains_authority(text: str) -> bool:
    return bool(extract_authorities(text))


def classify_authority(
    authority: str,
    *,
    source_text: str,
    externally_verified: set[str],
) -> AuthorityReference:
    normalized = authority.casefold()
    verified = {item.casefold() for item in externally_verified}
    if normalized in verified:
        return AuthorityReference(
            text=authority,
            status='EXTERNALLY_VERIFIED',
            verification_required=False,
        )
    if normalized in source_text.casefold():
        return AuthorityReference(
            text=authority,
            status='USER_SUPPLIED',
            verification_required=True,
        )
    return AuthorityReference(
        text=authority,
        status='UNVERIFIED',
        verification_required=True,
    )


def apply_authority_policy(
    result: LegalAnalysisResult,
    *,
    source_text: str,
    externally_verified: set[str] | None = None,
) -> list[str]:
    verified = externally_verified or set()
    findings: list[str] = []
    for issue in result.issues:
        issue_text = ' '.join(
            [issue.rule, issue.application, issue.conclusion, *issue.counterarguments]
        )
        authorities = [
            classify_authority(item, source_text=source_text, externally_verified=verified)
            for item in extract_authorities(issue_text)
        ]
        issue.authorities = authorities
        unverified = [item for item in authorities if item.verification_required]
        if unverified:
            issue.jurisdiction_status = 'JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED'
            findings.append(
                f'Issue {issue.taxonomy_label} contains authority requiring verification.'
            )
        if issue.jurisdiction_status == 'VERIFIED':
            if not authorities or any(item.verification_required for item in authorities):
                issue.jurisdiction_status = 'JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED'
                findings.append(
                    f'Issue {issue.taxonomy_label} claimed VERIFIED without verified authority evidence.'
                )
    return findings
