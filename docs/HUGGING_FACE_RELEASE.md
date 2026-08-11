# Publishing Contra v2 to Hugging Face

The first model release is `rjaloudi/contra-contracts-bert-cuad-clause-classifier`. Publish it **private first**, validate the rendered repository and a clean download, and only then consider making it public.

Contra v2 follows a modern encoder-release pattern: deterministic preparation, data lineage, grouped-split evaluation, per-label disclosure, reproducible configuration, safe serialization, an executable inference example, and an explicit model card. This is strong release engineering, not a claim of state-of-the-art benchmark performance.

## Current local model

| Item | Value |
|---|---|
| Base model | `nlpaueb/bert-base-uncased-contracts` |
| Task | 41-label English contract-clause classification |
| Training loss | Class-weighted cross-entropy |
| Best validation macro-F1 | 0.5869 |
| Held-out accuracy | 0.7639 |
| Held-out macro precision | 0.6456 |
| Held-out macro recall | 0.6151 |
| Held-out macro-F1 | 0.6036 |
| Held-out examples | 1,008 |
| Zero-F1 labels | 6 of 41 |
| Status | Educational alpha |

V2 improves substantially on v1's 0.2986 macro-F1, but rare-label failures remain. It classifies segmented passages; it does not perform whole-contract legal reasoning, determine enforceability, generate verified authority, or provide calibrated probabilities.

## License and lineage

The upstream Contracts-BERT model card declares **CC BY-SA 4.0**, so the Contra v2 model bundle uses `cc-by-sa-4.0`. CUAD-derived training data remains CC BY 4.0. Application source code remains MIT. Share-alike and attribution obligations must stay visible when the model is redistributed.

## 1. Authenticate outside the app

Use the official Hugging Face CLI credential store. Do not paste a token into Streamlit, Code Lab, source code, a notebook, screenshots, or chat.

```powershell
hf auth login
hf auth whoami
```

Use a scoped write token for the intended account. See the official [CLI authentication guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

## 2. Rebuild the private upload folder

```powershell
.\.venv\Scripts\python.exe tools\prepare_huggingface_model.py `
  --version v2 `
  --repo-id rjaloudi/contra-contracts-bert-cuad-clause-classifier
```

The ignored output folder is:

```text
artifacts/huggingface/contra-contracts-bert-cuad-clause-classifier/
```

It contains model weights, tokenizer files, label mapping, training configuration, evaluation summary, and the generated model card. Review every file. Confirm that no source contracts, dataset partitions, W&B files, secrets, user paths, or local chat data are present.

## 3. Run a clean local smoke test

Load the prepared directory with `AutoTokenizer` and `AutoModelForSequenceClassification`, run at least one representative clause, and verify that:

- the model loads without custom code;
- the output has 41 logits;
- `id2label` and `label2id` are exact inverses;
- the top-k labels are readable;
- no private paths appear in configuration or card files.

## 4. Upload privately

```powershell
.\.venv\Scripts\python.exe tools\prepare_huggingface_model.py `
  --version v2 `
  --repo-id rjaloudi/contra-contracts-bert-cuad-clause-classifier `
  --push --private
```

The tool creates or reuses the model repository and uploads only the prepared folder. Hugging Face documents the supported upload methods in its official [upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload) and repository visibility controls in the [repository guide](https://huggingface.co/docs/huggingface_hub/en/guides/repository).

## 5. Inspect the private repository

Open `https://huggingface.co/rjaloudi/contra-contracts-bert-cuad-clause-classifier` while signed in and confirm:

- the license is CC BY-SA 4.0;
- the base model and CUAD lineage are linked;
- metrics are labeled as this project's grouped holdout, not the official CUAD QA benchmark;
- all 41 labels and the six zero-F1 labels are disclosed;
- the example uses the exact repository ID;
- limitations and human-review requirements are prominent;
- the Files tab contains only the seven reviewed release files.

Hugging Face model cards are repository `README.md` files with metadata used for discovery and evaluation reporting; see the official [model card documentation](https://huggingface.co/docs/hub/model-cards).

## 6. Validate from a clean environment

Authenticate in a clean environment with an account that can read the private repository, then run
the project downloader:

```powershell
hf auth login
hf auth whoami
.\.venv\Scripts\python.exe tools\download_huggingface_model.py
```

The default destination is `artifacts/models/contra-v2/`. The tool uses the credential stored by
the official CLI; it deliberately has no token argument. It downloads only the seven reviewed
release files, validates `config.json`, confirms the 41-label mappings are exact inverses, and loads
the saved tokenizer and safetensors weights using local files before checking for 41 logits.

Pin a tag or full commit hash when repeatability matters:

```powershell
.\.venv\Scripts\python.exe tools\download_huggingface_model.py `
  --revision FULL_COMMIT_HASH `
  --output artifacts\models\contra-v2
```

For a direct Hub download without the project validation checks:

```powershell
hf download rjaloudi/contra-contracts-bert-cuad-clause-classifier `
  --local-dir artifacts\models\contra-v2
```

Transformers can also load the repository directly after `hf auth login`:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo_id = "rjaloudi/contra-contracts-bert-cuad-clause-classifier"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
```

See Hugging Face's official [download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)
for `snapshot_download`, revision pinning, file filters, and local directories. A successful clean
download catches missing repository files and accidental reliance on the original training folder.
Do not paste an access token into a command, script, notebook, Streamlit widget, screenshot, or chat.

## 7. Make public only after review

Changing visibility is a separate, deliberate release decision. Before doing so, preserve CUAD attribution, the upstream share-alike license, the legal-use limitations, the exact evaluation split description, and the zero-F1 disclosure. A public model repository is not evidence that the model is legally reliable.

## Private-release checklist

- [ ] `hf auth whoami` shows `rjaloudi`.
- [ ] Repository ID is exactly `rjaloudi/contra-contracts-bert-cuad-clause-classifier`.
- [ ] Prepared bundle contains only the reviewed release files.
- [ ] Clean local inference succeeds and produces 41 logits.
- [ ] Card shows CC BY-SA 4.0 and links the upstream model.
- [ ] CUAD attribution and EDGAR-rights caution remain visible.
- [ ] All 41 labels and six zero-F1 classes are disclosed.
- [ ] No contracts, data partitions, credentials, W&B files, chat data, or private paths are present.
- [ ] Private repository rendering and clean download are verified before any public release.