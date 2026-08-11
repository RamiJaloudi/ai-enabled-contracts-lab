# Legal prompt evaluations

`cases.json` is the fixed, versioned regression set for the legal-analysis and contract-chat
prompts. It tests the user-visible failure modes that ordinary unit tests cannot measure:

- full-draft routing and valid structured generation;
- preservation of supplied deal terms;
- placeholders instead of invented business facts;
- complete contract structure, signatures, and DOCX/TXT artifacts;
- resistance to instructions embedded in source documents;
- separation of recommendation mode from full-contract mode;
- missing facts, counterarguments, jurisdiction caveats, evidence IDs, and authority safety.

Run the current prompts:

```powershell
.venv\Scripts\python.exe -m contractlab.legal_ai.prompt_evaluation `
  --label baseline `
  --output artifacts\prompt-evaluations\baseline.json
```

Run a controlled drafting-prompt variant against the same cases and model settings:

```powershell
.venv\Scripts\python.exe -m contractlab.legal_ai.prompt_evaluation `
  --label contract-drafting-candidate-v3 `
  --override contract_drafting=prompt_evals\contract_drafting_candidate_v3.md `
  --output artifacts\prompt-evaluations\candidate-v3.json
```

The JSON report preserves the dataset hash, every prompt hash, configured model and temperature,
per-assertion results, raw model output, fallback use, and latency. A candidate should be adopted
only when its mean score improves and it introduces no case-level regression. Local model decoding
can vary, so repeat close comparisons before treating a small difference as conclusive.

The active drafting prompt was adopted from candidate v3 after the corrected two-case drafting
comparison improved from 0.798 to 1.000 with no regression. `adoption_manifest.json` records the
model settings, hashes, scores, and release decision.
