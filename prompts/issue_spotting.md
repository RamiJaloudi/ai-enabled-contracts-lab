# Prompt — Legal Issue Spotting

Input:
- user text or document excerpt;
- BERT top-k domain predictions, if available;
- BERT top-k doctrine predictions, if available;
- jurisdiction, if known.

Instructions:
1. Read the facts independently.
2. Treat BERT outputs as candidate issues only.
3. Identify all materially plausible legal issues.
4. Map each issue to the closest available taxonomy label.
5. If no label fits, use `UNMAPPED_ISSUE`.
6. For each issue, identify the factual trigger.
7. Include related doctrines and likely defenses.
8. Include missing facts.
9. Return a confidence between 0 and 1.
10. Do not decide the merits merely from issue detection.

Return JSON when called by the application.
