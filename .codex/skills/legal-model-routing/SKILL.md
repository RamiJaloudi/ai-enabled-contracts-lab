---
name: legal-model-routing
description: Coordinate BERT classification with the generative LLM and legal skills.
---

# BERT + Generative LLM Routing Skill

BERT handles classification and issue candidates.
The generative LLM handles analysis.

Runtime:
1. Run domain classifier.
2. Run doctrine classifier.
3. Preserve top-k labels and confidence.
4. Run legal-issue-spotting.
5. Determine jurisdiction.
6. Select relevant domain skill(s).
7. Build LLM context from facts, predictions, prompt, and skill instructions.
8. Generate structured analysis.
9. Validate output schema.
10. Preserve uncertainty.

No RAG is assumed in the current architecture.
