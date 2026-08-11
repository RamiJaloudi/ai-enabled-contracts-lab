# Codex Build Plan

Ask Codex to implement the project in this order:

1. Dataset loader
   - JSONL loading
   - label maps
   - split validation

2. BERT classifier service
   - domain classifier
   - doctrine classifier
   - top-k predictions
   - confidence scores

3. Prompt loader
   - load markdown prompts from disk
   - no prompt strings hard-coded in Python

4. Skill loader
   - load SKILL.md files by routing name
   - support more than one skill per request

5. Jurisdiction router
   - rule-based first version
   - optional classifier later

6. Generative LLM client
   - local/self-hosted model interface
   - keep provider implementation behind an interface

7. Analysis orchestrator
   - BERT
   - issue spotting
   - jurisdiction
   - skill selection
   - LLM analysis
   - schema validation

8. FastAPI
   Suggested endpoints:
   - POST /classify
   - POST /analyze
   - POST /analyze-document
   - GET /health
   - GET /models

9. Evaluation harness
   Metrics:
   - domain accuracy
   - doctrine top-1 accuracy
   - doctrine top-k recall
   - multi-label F1
   - structured-output validity
   - issue-recovery rate
   - missing-fact quality
   - authority-fabrication rate (target: 0)

10. Tests
   - unit tests
   - integration test for full analysis pipeline
