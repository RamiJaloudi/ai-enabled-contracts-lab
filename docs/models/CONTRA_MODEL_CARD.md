---
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: text-classification
language:
- en
base_model: nlpaueb/bert-base-uncased-contracts
datasets:
- theatticusproject/cuad-qa
metrics:
- accuracy
- precision
- recall
- f1
tags:
- legal
- contracts
- clause-classification
- bert
- pytorch
model-index:
- name: Contra v2
  results:
  - task:
      type: text-classification
      name: Contract clause classification
    dataset:
      name: Project contract-grouped CUAD derivative
      type: theatticusproject/cuad-qa
      split: test
    metrics:
    - type: accuracy
      value: 0.7639
      name: Accuracy
    - type: f1
      value: 0.6036
      name: Macro-F1
---

# Contra v2: CUAD clause classifier

**Contra v2** is an educational 41-label English contract-clause classifier fine-tuned from [nlpaueb/bert-base-uncased-contracts](https://huggingface.co/nlpaueb/bert-base-uncased-contracts). The repository ID is **rjaloudi/contra-contracts-bert-cuad-clause-classifier**.

It is designed as the encoder-routing layer in [AI-Enabled Contracts Lab](https://github.com/RamiJaloudi/ai-enabled-contracts-lab), where classifier candidates remain separate from generative legal analysis. Its release follows modern practices—safe serialization, explicit lineage, grouped-split metrics, per-label limitations, reproducible configuration, and a clean inference path—but this is not a claim of state-of-the-art benchmark performance.

## Intended use

Use the model to rank candidate CUAD clause categories for already segmented English contract passages, support error-analysis education, and seed a human-reviewed routing workflow. Preserve top-k labels and scores. Treat every prediction as a hypothesis.

Do not use it as a whole-contract reasoner, an authority source, a substitute for counsel, or an automated decision-maker for enforceability, legal rights, risk, fairness, or required language.

## Results

| Held-out metric | Value |
|---|---:|
| Accuracy | 0.7639 |
| Macro precision | 0.6456 |
| Macro recall | 0.6151 |
| Macro-F1 | 0.6036 |
| Test examples | 1,008 |
| Labels | 41 |
| Zero-F1 labels | 6 |

Training used clipped inverse-square-root class weights with cross-entropy. Checkpoint selection used validation macro-F1; the best validation value was 0.5869. The split is grouped by source contract, so no source contract crosses training, validation, and test partitions. Results describe this project's deterministic classifier split, not the official CUAD extractive-QA benchmark.

Macro-F1 weights each label equally and therefore exposes rare-label failures that accuracy can obscure. Consult `evaluation_summary.json` for the full metric and per-label disclosure.

## Training data

Training examples are deterministic classifier derivatives of [CUAD v1](https://github.com/The-Atticus-Project/cuad), curated by The Atticus Project and declared under CC BY 4.0. Positive annotated answer spans become classifier text and CUAD categories become labels. Exact text/label duplicates are removed before training.

CUAD's publisher states that it makes no representations or warranties about the license status of underlying SEC EDGAR contracts. Preserve attribution and review current CUAD terms before reuse.

## How the project uses this model

```text
contract passage
  -> Contra v2 top-k clause hypotheses
  -> confidence and abstention policy
  -> independent issue spotting
  -> jurisdiction and prompt/skill routing
  -> local generative LLM
  -> structured result and human review
```

The project currently has no retrieval-augmented generation layer. The classifier does not supply cases, statutes, regulations, quotations, or verified legal authority.

## Limitations and responsible use

- This is an educational alpha model, not legal advice.
- It classifies individual passages and cannot determine whether a complete contract is adequate.
- Six rare labels have zero held-out F1; additional labels remain weak.
- Softmax confidence is not calibrated and must not be treated as legal certainty.
- The 41-label CUAD taxonomy is closed and incomplete.
- English commercial contracts from CUAD do not represent every jurisdiction, document type, drafting style, industry, party, or time period.
- Training data may contain annotation artifacts and public-filing language unlike private contracts.
- Human review and jurisdiction-specific verification are required for real-world use.

## Example

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="rjaloudi/contra-contracts-bert-cuad-clause-classifier",
    top_k=3,
)

results = classifier(
    "This Agreement is governed by the laws of the State of New York."
)
print(results)
```

For private repositories, authenticate with `hf auth login` before loading.

## Reproducibility files

- `config.json`: architecture and exact label mappings.
- `model.safetensors`: model weights using safe serialization.
- tokenizer files: preprocessing vocabulary and settings.
- `training_config.json`: project training controls and provenance.
- `evaluation_summary.json`: aggregate and per-label evaluation disclosure.

## Licenses

- Fine-tuned model bundle: Creative Commons Attribution-ShareAlike 4.0, following the upstream base-model license.
- Base model: see the upstream Contracts-BERT model card.
- Training data and included dataset derivatives: CC BY 4.0.
- AI-Enabled Contracts Lab source code: MIT.

## Citation

If you use the training data, cite CUAD:

```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={NeurIPS},
  year={2021}
}
```