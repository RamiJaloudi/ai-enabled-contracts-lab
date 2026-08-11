Use the repository instructions in AGENTS.md.

Build the first vertical slice of the legal AI system without RAG.

Requirements:
- Load the legal dataset taxonomy and label maps.
- Implement a BERT classifier interface that returns top-k labels with confidence.
- Implement prompt loading from /prompts.
- Implement skill loading from /.codex/skills.
- Implement a model-agnostic generative LLM interface.
- Implement a LegalAnalysisOrchestrator that performs:
  classify -> issue spot -> jurisdiction -> skill route -> LLM analysis -> schema validation.
- Use Pydantic models matching schemas/legal_analysis.schema.json.
- Add FastAPI endpoints POST /classify and POST /analyze.
- Add pytest tests.
- Do not hard-code prompt text in Python.
- Do not implement RAG.
- Never generate or validate fake legal citations.
- Keep classifier confidence separate from legal-analysis confidence.

Before coding, produce a short implementation plan and list the files you will create or modify.
