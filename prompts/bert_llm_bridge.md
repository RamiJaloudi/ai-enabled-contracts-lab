# Prompt — BERT to LLM Bridge

Inputs:
- raw text;
- domain predictions;
- doctrine predictions;
- confidence scores;
- taxonomy version.

Workflow:
1. Keep top-k predictions.
2. Discard nothing solely because it is not top-1.
3. Run independent issue spotting.
4. Merge classifier and issue-spotter candidates.
5. Preserve classifier confidence separately from legal-analysis confidence.
6. Route to one or more domain skills.
7. Run jurisdiction analysis.
8. Send the assembled context to the generative LLM.
9. Validate the LLM response against the legal-analysis schema.

Never convert a classifier score directly into a legal conclusion.
