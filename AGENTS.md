# AGENTS.md — Legal AI Project Instructions

## Project purpose

Build a legal AI application that combines:

1. A fine-tuned BERT encoder for legal-domain and legal-doctrine classification.
2. A local or self-hosted generative LLM for structured legal analysis.
3. Prompt files and reusable skills for issue spotting, jurisdiction analysis, fact extraction,
   counterarguments, and IRAC-style reasoning.
4. The existing synthetic legal datasets as training and evaluation scaffolding.

The application is intended to do more than classify text. Classification is the first routing step.
The final system should analyze legal problems in a structured, bounded, explainable way.

## Core architecture

User facts or legal document
→ preprocessing
→ BERT domain classifier
→ BERT doctrine classifier (top-k)
→ independent issue spotting
→ jurisdiction routing
→ relevant legal skill(s)
→ generative LLM
→ structured legal analysis

Do not add RAG unless the user explicitly requests it later.

## BERT responsibilities

Use BERT for:
- domain classification;
- doctrine / issue classification;
- multi-label candidate generation;
- semantic similarity or reranking where useful;
- token classification / extraction where useful.

Do not use BERT as the primary long-form legal reasoning engine.

Classifier predictions are hypotheses, not legal conclusions.

Always preserve:
- top-k predictions;
- confidence scores;
- model version;
- taxonomy version.

## Generative LLM responsibilities

Use the generative LLM for:
- issue explanation;
- rule/elements organization;
- fact-to-element application;
- counterarguments and defenses;
- missing-fact identification;
- bounded conclusions;
- document summarization and legal drafting assistance.

The generative model must not invent cases, statutes, regulations, docket numbers, quotations,
or citations.

Because this project currently has no RAG layer, the model must clearly distinguish:
- general legal principles;
- model-trained knowledge;
- jurisdiction-specific law that requires verification.

When jurisdiction-specific authority is needed but unavailable, use:
`JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED`.

## Dataset contract

The existing legal datasets contain fields similar to:

- `domain`
- `label`
- `label_id`
- `subcategory`
- `jurisdiction`
- `style`
- `text`
- `issue`
- `rule_summary`
- `application`
- `counterargument`
- `conclusion`
- `source`
- `split`

Use:
- `text -> label_id` for BERT classification.
- `text + metadata -> issue/rule/application/counterargument/conclusion` for generative-model training/evaluation.

Synthetic data is scaffolding, not authoritative law.

## Required legal-analysis behavior

For each materially plausible issue, produce:

1. Issue
2. Rule / elements
3. Supporting facts
4. Adverse facts
5. Missing facts
6. Application
7. Counterarguments / defenses
8. Jurisdiction caveat
9. Conclusion
10. Confidence / uncertainty

Do not state that a claim succeeds merely because the classifier detected the doctrine.

## Jurisdiction behavior

Before giving jurisdiction-dependent analysis:
1. Determine or infer the likely jurisdiction.
2. Distinguish federal, state, New Jersey, administrative, and international law.
3. Flag uncertainty.
4. Never silently apply New Jersey law to a general U.S. problem or vice versa.

Family Law should support both:
- `New Jersey`
- `U.S. general`

Patent, Bankruptcy, Securities, Constitutional Law, and much of Administrative Law are primarily
federal, but do not force federal treatment when the facts do not support it.

## Prompt and skill usage

Prompts live in `/prompts`.

Reusable Codex skills live in `/.codex/skills`.

Before changing legal-analysis behavior:
1. Read this `AGENTS.md`.
2. Read the relevant prompt file.
3. Read the relevant skill.
4. Check the output schema.
5. Update or add tests.

## Engineering standards

Prefer:
- Python 3.12+
- PyTorch
- Hugging Face Transformers
- Pydantic for runtime schemas
- FastAPI for API endpoints
- pytest for tests

Keep model code, prompt orchestration, data loading, and API code separated.

Suggested package layout:

legal_ai/
  api/
  classifiers/
  llm/
  prompts/
  skills/
  schemas/
  services/
  training/
  evaluation/
  utils/

## Configuration

Do not hard-code:
- model paths;
- confidence thresholds;
- taxonomy paths;
- prompt paths;
- device selection;
- max token lengths.

Put these in config files or environment variables.

## Testing requirements

Add tests for:
- dataset loading;
- label mapping;
- classifier top-k output;
- multi-label issue routing;
- jurisdiction routing;
- prompt loading;
- skill loading;
- structured-output validation;
- refusal to invent authority;
- missing-fact behavior;
- low-confidence classifier behavior.

When changing classifier or prompt behavior, run the relevant tests before completing the task.

## Safety and legal-quality requirements

This is legal software. Avoid fabricated authority and false certainty.

The application should prefer:
- "likely";
- "may";
- "depends on";
- "insufficient information";
- "jurisdiction-specific verification required";

when the facts or governing law do not support a definitive answer.

Do not present synthetic examples as actual cases or statutes.

## Current project objective

Build the first production-style pipeline without RAG:

BERT classifier
+ prompt/skill router
+ generative LLM
+ structured legal analysis
+ evaluation harness.

Optimize for modularity so a retrieval layer can be added later without redesigning the classifier or LLM interfaces.
