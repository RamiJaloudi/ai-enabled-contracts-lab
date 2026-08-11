from __future__ import annotations

import re
from typing import Any

from contractlab.legal_ai.jurisdiction import route_jurisdiction
from contractlab.legal_ai.schemas import ContractDraft, ContractDraftSection

CLAUSE_LIBRARY: dict[str, str] = {
    'parties': (
        'This Agreement is entered into by [PARTY A - CONFIRM LEGAL NAME AND ENTITY TYPE] '
        'and [PARTY B - CONFIRM LEGAL NAME AND ENTITY TYPE].'
    ),
    'scope': (
        '[SUPPLIER] shall perform the services described in [SCOPE OR STATEMENT OF WORK], '
        'subject to the responsibilities, assumptions, and change-control process stated there.'
    ),
    'deliverables': (
        '[SUPPLIER] shall provide the deliverables, formats, milestones, and dependencies listed '
        'in [SCHEDULE OR STATEMENT OF WORK].'
    ),
    'acceptance': (
        '[CUSTOMER] shall review each deliverable against [OBJECTIVE ACCEPTANCE CRITERIA] within '
        '[REVIEW PERIOD]. A rejection notice must identify the unmet criteria, after which '
        '[SUPPLIER] shall have [CORRECTION PERIOD] to correct and resubmit the deliverable.'
    ),
    'fees and payment': (
        '[CUSTOMER] shall pay the fees stated in [FEE SCHEDULE]. Invoices, disputed amounts, '
        'taxes, expenses, late charges, and payment timing shall be handled as stated in '
        '[PAYMENT TERMS].'
    ),
    'term': (
        'This Agreement begins on [EFFECTIVE DATE] and continues until [END DATE OR COMPLETION], '
        'unless earlier terminated under this Agreement. Any renewal requires '
        '[RENEWAL MECHANISM AND NOTICE PERIOD].'
    ),
    'termination': (
        'Either party may terminate for an uncured material breach after written notice and '
        '[CURE PERIOD]. Any termination for convenience requires [NOTICE PERIOD] written notice '
        'and is subject to [PAYMENT OR WIND-DOWN TERMS].'
    ),
    'effects of termination': (
        "Upon expiration or termination, each party shall stop using the other party's property, "
        'return or destroy protected information as required, complete [TRANSITION DUTIES], and '
        'pay undisputed amounts accrued through the effective termination date. Provisions that '
        'by their nature should continue shall survive only to the extent stated in this Agreement.'
    ),
    'confidentiality': (
        "Each receiving party shall use the disclosing party's Confidential Information only for "
        'this Agreement, protect it using [REQUIRED SAFEGUARDS], and disclose it only to persons '
        'with a need to know who are bound by appropriate confidentiality duties. Exclusions, '
        'required-disclosure procedures, duration, and return or destruction are stated in '
        '[CONFIDENTIALITY TERMS].'
    ),
    'intellectual property': (
        'Each party retains its pre-existing materials and intellectual property. Ownership and '
        'license rights for deliverables, modifications, feedback, data, and embedded materials '
        'shall follow [SELECTED OWNERSHIP AND LICENSE MODEL]. No ownership transfers except as '
        'expressly stated in this Agreement.'
    ),
    'intellectual property rights': (
        'Each party retains its pre-existing materials and intellectual property. Ownership and '
        'license rights for deliverables, modifications, feedback, data, and embedded materials '
        'shall follow [SELECTED OWNERSHIP AND LICENSE MODEL]. No ownership transfers except as '
        'expressly stated in this Agreement.'
    ),
    'data protection': (
        'Each party shall process personal data only as permitted by [DATA-PROTECTION TERMS], '
        'apply [SECURITY STANDARD], notify the other party of a confirmed incident within '
        '[NOTICE PERIOD], and follow the allocation of controller, processor, and subcontractor '
        'responsibilities in [DATA PROCESSING ADDENDUM].'
    ),
    'security': (
        '[SERVICE PROVIDER] shall maintain the administrative, technical, and physical safeguards '
        'described in [SECURITY SCHEDULE], including access control, incident response, recovery, '
        'and deletion commitments appropriate to the supplied data and services.'
    ),
    'warranties': (
        'Each party warrants that it has authority to enter this Agreement. [SUPPLIER] further '
        'warrants the services will materially conform to [SPECIFICATIONS] during [WARRANTY '
        'PERIOD], subject to [EXCLUSIONS AND REMEDIES].'
    ),
    'indemnification': (
        "Each party's defense and indemnification obligations, covered third-party claims, "
        'exclusions, notice duties, control of defense, cooperation, and settlement authority '
        'shall be those selected in [INDEMNIFICATION SCHEDULE].'
    ),
    'limitation of liability': (
        "The parties' damages exclusions, aggregate liability cap, cap period, and carve-outs "
        'shall be those stated in [LIMITATION OF LIABILITY TERMS]. The parties must confirm how '
        'those limits interact with confidentiality, data security, indemnification, and payment.'
    ),
    'governing law': (
        'This Agreement is governed by the laws of [GOVERNING JURISDICTION], without silently '
        'resolving conflict-of-law questions. The parties select [COURT, VENUE, ARBITRATION, OR '
        'OTHER DISPUTE PROCESS], subject to counsel verification.'
    ),
    'notices': (
        'Formal notices under this Agreement must be in writing and delivered by [PERMITTED '
        'METHODS] to the addresses stated in [NOTICE DETAILS]. A notice is effective at '
        '[DELIVERY OR DEEMED-RECEIPT TIME].'
    ),
    'assignment': (
        'Neither party may assign this Agreement except under [PERMITTED ASSIGNMENT RULES]. Any '
        'change-of-control treatment and required consent shall be stated expressly.'
    ),
    'force majeure': (
        'A party affected by events beyond its reasonable control shall provide prompt notice, '
        'use reasonable mitigation efforts, and resume performance when practicable. The parties '
        'must specify excluded obligations and any termination right after [MAXIMUM DELAY].'
    ),
    'amendment': (
        'An amendment is effective only when set out in a writing that identifies this Agreement '
        'and is signed by authorized representatives of both parties.'
    ),
    'entire agreement': (
        "This Agreement and its identified schedules constitute the parties' complete agreement "
        'on its subject matter and supersede prior discussions on that subject. The parties must '
        'identify the order of precedence for conflicting documents.'
    ),
}


def _normalize(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', value.casefold()).strip()


def _title_and_sections(contract_text: str) -> tuple[str, list[ContractDraftSection]]:
    blocks = [row.strip() for row in re.split(r'(?:\r?\n){2,}', contract_text) if row.strip()]
    title = 'Revised Agreement'
    if blocks and len(blocks[0]) <= 120 and blocks[0].upper() == blocks[0]:
        title = blocks.pop(0).title()
    sections: list[ContractDraftSection] = []
    for index, block in enumerate(blocks, start=1):
        match = re.match(r'^([A-Z][A-Za-z0-9 &/()_-]{1,70})\.\s*(.+)$', block, re.S)
        if match:
            heading = match.group(1).strip()
            body = match.group(2).strip()
        else:
            lines = block.splitlines()
            if len(lines) > 1 and len(lines[0]) <= 70:
                heading = lines[0].rstrip('.:').strip()
                body = '\n'.join(lines[1:]).strip()
            else:
                heading = f'Existing Term {index}'
                body = block
        sections.append(ContractDraftSection(heading=heading, paragraphs=[body]))
    return title, sections


def _source_parties(contract_text: str) -> list[str]:
    match = re.search(
        r'\b(?:between|by and between)\s+(.{2,100}?)\s+and\s+(.{2,100}?)(?:\.|,|\n)',
        contract_text,
        re.I,
    )
    if not match:
        return [
            '[PARTY A - CONFIRM LEGAL NAME AND ENTITY TYPE]',
            '[PARTY B - CONFIRM LEGAL NAME AND ENTITY TYPE]',
        ]
    return [match.group(1).strip(), match.group(2).strip()]


def deterministic_contract_draft(
    *,
    contract_text: str,
    analysis_result: dict[str, Any],
    request: str,
) -> ContractDraft:
    source_title, sections = _title_and_sections(contract_text)
    contract_analysis = analysis_result.get('contract_analysis', {})
    profile = str(contract_analysis.get('profile') or 'Agreement')
    missing = [
        row for row in contract_analysis.get('clauses', []) if row.get('status') != 'Detected'
    ]
    existing_headings = {_normalize(section.heading) for section in sections}
    unresolved: list[str] = []
    for row in missing:
        name = str(row.get('name') or 'Additional Negotiated Terms').strip()
        normalized = _normalize(name)
        if normalized in existing_headings:
            continue
        definition = str(row.get('definition') or '').strip()
        language = CLAUSE_LIBRARY.get(normalized)
        if language is None:
            language = (
                f'[THE PARTIES MUST INSERT NEGOTIATED {name.upper()} TERMS HERE. '
                f'Review focus: {definition or "confirm scope, allocation, and procedures"}.]'
            )
        sections.append(ContractDraftSection(heading=name, paragraphs=[language]))
        questions = [str(item) for item in row.get('review_questions', []) if str(item).strip()]
        unresolved.extend(questions[:1] or [f'Confirm the negotiated terms for {name}.'])

    if 'signatures' not in existing_headings:
        sections.append(
            ContractDraftSection(
                heading='Signatures',
                paragraphs=[
                    '[PARTY A]\nBy: [AUTHORIZED SIGNATORY]\nName: [NAME]\nTitle: [TITLE]\nDate: [DATE]',
                    '[PARTY B]\nBy: [AUTHORIZED SIGNATORY]\nName: [NAME]\nTitle: [TITLE]\nDate: [DATE]',
                ],
            )
        )
    facts = analysis_result.get('facts', {})
    dates = [str(item) for item in facts.get('dates', [])]
    jurisdiction = route_jurisdiction(contract_text).likely_jurisdiction
    return ContractDraft(
        title=source_title if source_title != 'Revised Agreement' else f'Revised {profile}',
        contract_type=profile,
        jurisdiction=jurisdiction,
        parties=_source_parties(contract_text),
        effective_date=dates[0] if dates else '[EFFECTIVE DATE]',
        purpose=(
            'The parties intend this revised Agreement to govern [TRANSACTION PURPOSE]. '
            'It preserves supplied language and adds sections identified for review.'
        ),
        assumptions=[
            'Supplied contract provisions were preserved as source terms rather than restated as verified law.',
            'New business terms remain bracketed placeholders until the parties or counsel select them.',
            f'The drafting request was: {request}',
        ],
        unresolved_terms=list(dict.fromkeys(unresolved))[:20],
        sections=sections,
        drafting_note=(
            'Deterministic complete draft produced because the configured generative model did '
            'not return valid structured output. Review every source term, placeholder, '
            'cross-reference, and added clause. JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED.'
        ),
    )
