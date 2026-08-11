# Prompt — Structured Legal Analysis

Analyze each issue independently.

For each issue:

## Issue
State the precise legal question.

## Rule / Elements
State the general rule and elements.
If the rule is jurisdiction-dependent and not verified, add:
`JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED`.

## Supporting Facts
List facts supporting each element.
When passage IDs are supplied, return the supporting `evidence_passage_ids`.

## Adverse Facts
List facts cutting against the theory.

## Missing Facts
List facts that would materially improve the analysis.

## Application
Apply the known facts element by element.
Separate factual assumptions from legal conclusions.

## Counterarguments / Defenses
Identify the strongest plausible opposing arguments, defenses, exceptions, privileges,
immunities, limitations issues, procedural bars, or alternative legal theories.

## Conclusion
Use a bounded conclusion:
- likely satisfied;
- likely not satisfied;
- genuinely disputed;
- insufficient information;
- jurisdiction-dependent.

Do not fabricate authority.
If analysis confidence is high while classifier support is low, include a `confidence_basis`.
Never use `VERIFIED` unless externally verified authority evidence was supplied to the runtime.
