# AI-Enabled Contracts Lab User and Developer Guide

A guided workbench and visible-code path for learning Transformer fine-tuning in contract analysis

Version 0.1.0 — August 2026

## 1. Purpose and learning outcomes

AI-Enabled Contracts Lab is a local Streamlit workbench for learning how an existing
Transformer model becomes a contract-clause classifier. It combines explanations,
editable settings, real PyTorch code, experiment tracking, evaluation, and a
contract-analysis demonstration. The interface is a guide; the Python package and
command-line interface are the same implementation exposed without the interface.

This guide is for a learner with basic Python experience who wants both a
structured path and direct access to working code. By the end of a complete first
run, you should be able to:

- explain the difference between pretraining, fine-tuning, evaluation, and inference;
- prepare a labeled clause dataset and preserve document-level train/test separation;
- choose a pretrained encoder and understand the main memory trade-offs;
- follow a handwritten PyTorch optimization loop;
- run the classifier training pipeline from the UI, CLI, generated script, or Jupyter;
- interpret macro F1, per-class results, confusion matrices, and individual errors;
- connect a trained classifier to six source-grounded contract-analysis tasks; and
- prepare a clean contribution or release without publishing private contracts or secrets.

> Legal and educational boundary: AI-Enabled Contracts Lab is educational software, not legal
> advice. A model prediction is evidence about model behavior, not a legal
> conclusion. A qualified human must review outputs before they affect legal,
> commercial, employment, compliance, or rights-related decisions.

### 1.1 What version 0.1.0 covers

The first course focuses on multiclass text classification: one clause or passage
is assigned one label such as Confidentiality, Governing Law, Termination,
Limitation of Liability, Indemnification, Payment Terms, Intellectual Property,
or Renewal. The same workbench also introduces:

- tokenization and maximum sequence length;
- PyTorch Dataset and DataLoader objects;
- a pretrained Hugging Face Transformer encoder;
- cross-entropy loss, gradients, AdamW, scheduling, clipping, and accumulation;
- mixed precision and checkpointing;
- optional Weights & Biases tracking;
- local evaluation and error analysis; and
- inference-driven assistant demonstrations.

### 1.2 What the first version intentionally does not do

It does not promise reliable legal interpretation, perform OCR on scanned PDFs,
train a production foundation model, or guarantee coverage of every contract
type. The built-in data is synthetic and is meant to prove the workflow. Real
performance requires representative, licensed, expertly labeled data and a
separate validation plan.

### 1.3 Modern hybrid legal-AI system

The current application extends the classifier course into a no-RAG, production-style hybrid
pipeline. Fine-tuned BERT encoders rank legal domains and doctrine candidates. A router preserves
top-k scores, model and taxonomy versions, confidence thresholds, abstention, independent issue
spotting, and jurisdiction uncertainty. A separate local generative LLM explains issues, applies
facts to elements, identifies missing facts and counterarguments, and drafts documents through
validated schemas.

```text
facts or contract
  -> preprocessing and passage segmentation
  -> BERT domain and doctrine top-k hypotheses
  -> independent issue spotting and jurisdiction routing
  -> legal prompts and reusable skills
  -> local generative LLM
  -> structured analysis or complete draft
  -> evaluation, audit metadata, and human review
```

This division of labor is a modern engineering pattern, not a claim of state-of-the-art benchmark
performance. Contra v2 records 0.6036 macro-F1 on this project's contract-grouped CUAD derivative
and six of 41 labels have zero held-out F1. Classifier confidence is not legal confidence. The
system must not invent authority, and unavailable jurisdiction-specific law is marked
`JURISDICTION_SPECIFIC_VERIFICATION_REQUIRED`. Retrieval-augmented generation is intentionally
out of scope for this release.
## 2. System requirements and installation

### 2.1 Target computer

The primary profile is Windows with an NVIDIA RTX 5070 with 12 GB VRAM. The
repository also provides a CPU profile for contributors, CI, and troubleshooting.

| Component | Required or recommended |
|---|---|
| Operating system | Windows 11 recommended; CI uses Linux |
| Python | Exactly the Python 3.12 series for this project |
| Package manager | uv |
| GPU profile | NVIDIA driver compatible with the PyTorch CUDA 12.9 wheel |
| CPU profile | Any supported x86-64 machine; training is much slower |
| Browser | A current browser for Streamlit at localhost |
| Disk | Allow several GB for packages, downloaded models, checkpoints, and caches |
| Network | Required for first-time package/model downloads and online W&B |

The project uses two mutually exclusive PyTorch extras:

- `cu129` installs the CUDA 12.9 build used by the Windows launchers.
- `cpu` installs the CPU-only build used by GitHub Actions.

Never request both extras in one command.

### 2.2 Install uv

Follow the official uv installation instructions at `https://docs.astral.sh/uv/`.
After installation, open a new PowerShell window and verify:

```powershell
uv --version
```

If PowerShell cannot find `uv`, close and reopen the terminal so the updated
PATH is loaded.

### 2.3 Open the project directory

Every manual command in this guide assumes PowerShell is in the repository root:

```powershell
Set-Location "C:\LLMs and Scripts\LLM_Learning_Material\fine-tune-models"
Get-Location
```

For a future GitHub clone, replace that path with the directory where you cloned
the repository. The prompt should end in `fine-tune-models` or your final
repository name.

### 2.4 Recommended RTX 5070 setup

The setup script selects Python 3.12, installs developer tools, installs the CUDA
profile, and runs the hardware doctor:

```powershell
.\setup.ps1
```

The equivalent visible commands are:

```powershell
uv sync --python 3.12 --extra dev --extra cu129
uv run --extra cu129 contractlab doctor
```

A healthy GPU report should show `cuda_available: true`, the NVIDIA device name,
and approximately 12 GB of device memory. A PyTorch version containing a CUDA
suffix is expected.

### 2.5 CPU-only setup

Use this profile on machines without a compatible NVIDIA GPU and in CI:

```powershell
uv sync --python 3.12 --extra dev --extra cpu
uv run --extra cpu contractlab doctor
```

The report should show `cuda_available: false`. The app still runs, but a
Transformer training experiment can be substantially slower.

### 2.6 Activate the environment only when helpful

`uv run` automatically selects the project environment, so activation is not
required. If you want a conventional interactive shell:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

Exit it later with `deactivate`. Continue to include `--extra cu129` or
`--extra cpu` on `uv run` commands so uv preserves the intended PyTorch build.

## 3. Starting and stopping AI-Enabled Contracts Lab

### 3.1 Simplest launch

Double-click:

```text
run-ai-enabled-contracts-lab.bat
```

The batch file is saved in the repository root beside `streamlit_app.py`. It
checks whether the app is already healthy, opens the existing instance if
possible, or starts Streamlit on:

```text
http://localhost:8501
```

Keep the command window open. Press `Ctrl+C` in that window to stop the server.

### 3.2 Launch from PowerShell

```powershell
.\run.ps1
```

The script prints the URL and starts the same CUDA-profile command used by the
batch launcher.

### 3.3 Run the application command manually

RTX 5070/CUDA profile:

```powershell
uv run --extra cu129 streamlit run streamlit_app.py --server.address=localhost --server.port=8501
```

CPU profile:

```powershell
uv run --extra cpu streamlit run streamlit_app.py --server.address=localhost --server.port=8501
```

Open `http://localhost:8501` if the browser does not open automatically.

### 3.4 Health check and clean shutdown

In a second PowerShell window:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health
```

Status code 200 means Streamlit is healthy. Stop with `Ctrl+C` in the original
server window. If a previous process still owns port 8501, close its terminal or
identify it with:

```powershell
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
```

Do not terminate a process until you verify that it belongs to AI-Enabled Contracts Lab.

## 4. Workbench tour

The navigation sidebar moves between pages. Streamlit reruns the active page
after most widget changes; this is normal. Long training work is launched in a
separate background process so it can continue while the page refreshes.

### 4.1 Home

Home explains the recommended path and the six assistant capabilities. Use it to
confirm the project scope and to return to the learning sequence. No training is
started from this page.

### 4.2 Guided classifier course

The Guided Classifier Course contains ten complete lessons in pipeline order, from task framing
through responsible hybrid integration. Every lesson includes:

- a learning outcome and explanation of why the topic matters;
- core concepts and the relevant mathematical relationship;
- a four-step workflow;
- the exact project source file and function or class when applicable;
- a guided experiment;
- self-check questions; and
- a concrete completion deliverable.

Use the outline, lesson selector, and previous/next controls to navigate. Completion state and
experiment notes are saved locally in `artifacts/course_progress.json`, which is excluded from
Git. **Download all course notes** creates a Markdown learning portfolio. Mark a lesson complete
only after producing its stated evidence; completion is a study record, not a model-quality claim.

Do not rush past the splitting lesson. A randomly split clause dataset may place language from the
same source contract in training and test data, creating an unrealistically optimistic score. The
final lesson traces classifier top-k hypotheses through abstention, independent issue spotting,
jurisdiction routing, structured generation, audit metadata, and human review.

### 4.3 Dataset lab

Dataset Lab supports synthetic examples, audited project datasets, uploads, and
official CUAD conversion. It shows schema validation, label counts, row
inspection, provenance, and a contract-grouped split preview.

The lower taxonomy browser is a knowledge layer: choose one of ten legal
domains, then a contract type, to see its definition, aliases, usual clauses,
clause definitions, and review questions. Download the combined JSON when you
want to inspect or extend the same records in code. Save an active training
dataset only after reviewing text, labels, contract IDs, source, and license.

### 4.4 Model lab

Model Lab compares open Hugging Face presets and links their upstream cards.
Start with DistilBERT to learn the mechanics. Use
`nlpaueb/bert-base-uncased-contracts` for the stronger Contra experiment: its
encoder was pretrained on US contract language, but it is larger and inherits
the upstream CC BY-SA 4.0 license. Model choice does not remove the need for
label definitions, representative examples, and held-out evaluation.

### 4.5 Train

Train creates a `TrainingConfig`, saves it under `artifacts/configs/`, and
launches a background process. Controls include epochs, learning rate, batch
sizes, gradient accumulation, maximum length, precision, freezing, W&B mode,
and imbalance strategy.

Use standard cross-entropy for a transparent baseline. Class-weighted
cross-entropy gives rare labels more influence; focal loss emphasizes hard
examples through `focal_gamma`. These can improve rare-class recall but can
also reduce performance on common labels, so compare on an unchanged held-out
split.

### 4.6 Evaluate

Evaluate displays aggregate and per-label metrics, a confusion matrix, and
individual predictions. The evaluation chat answers questions about the saved
run rather than generating unsupported legal prose. Click **Analyze this run**
to explain the accuracy/macro-F1 gap, zero-F1 labels, weakest classes, common
confusions, and a defensible next experiment. Prediction scores are not
calibrated probabilities.

### 4.7 Contract chat

Contract Chat is the general-purpose, multi-turn contract workspace. It can answer general
contract questions without a document, work from pasted text, or accept optional attachments from
the `+` control built into the chat composer. Supported documents are copied into a conversation-
scoped local workspace, extracted safely, and available to later follow-up questions. Archives are
inspected in memory rather than unpacked onto disk.

The conversation is scrollable and remains available after every answer, so a user can ask
clarifying or follow-up questions in the same way as a modern chat application. **Clear chat**
starts a fresh message history while preserving workspace documents; removing a document is a
separate deliberate action. The Local Workspace appears beneath the conversation for pasted text,
document inspection, and file management.

Response mode may be **Auto**, **Review and analyze**, or **Draft complete contract**. When the user
asks to draft, rewrite, improve, or add missing clauses, the drafting path returns the actual
complete proposed agreement in the chat and matching Word (`.docx`) and plain-text (`.txt`) files.
It preserves supplied terms, uses bracketed placeholders for unknown terms, records assumptions
and negotiation choices, and marks jurisdiction-dependent language for verification. A malformed
or truncated local-model response falls back to a source-preserving deterministic drafter so that
the editable artifacts still contain a complete instrument rather than only recommendations.

Conversations, uploaded originals, extracted text, and generated files are local artifacts and
must never be committed to Git. Clear chat and workspace removal are privacy controls, not a
substitute for disk encryption, access control, retention policy, or professional confidentiality
review.

### 4.8 Contract assistant

Contract Assistant is the more structured document-review demonstration. It accepts pasted text
or TXT, PDF, and DOCX files, classifies passages, identifies a likely contract type, compares the
document with taxonomy concepts, and answers with cited passage IDs. It combines Contra
predictions with transparent definitions and keyword matching when classifier confidence is low
or the requested concept falls outside the CUAD label space.

A concept reported as not detected means the educational matcher found no clear supporting text;
it does not prove that a provision is legally required, absent, deficient, enforceable, or
unenforceable. Verify every cited passage and every governing-law assumption.

### 4.9 Experiments

Experiments lists local runs and saved metrics. When W&B was enabled, it also provides the
corresponding tracking context. Local artifacts remain the reproducibility source of truth.

### 4.10 Code lab

Code Lab has three surfaces. **Experiment export** displays implementation files and generates an
editable script, Jupyter notebook, and JSON configuration. **Private GitHub release** provides the
review, test, commit, and private repository commands. **Private Hugging Face release** prepares
Contra v2, displays its measured limitations, and provides exact private-upload commands for
`rjaloudi/contra-contracts-bert-cuad-clause-classifier`.

Code Lab never requests or stores GitHub or Hugging Face tokens. Authenticate with the official
CLIs in a terminal, inspect every staged or prepared file, and treat private repositories as
review boundaries rather than proof of quality.

### 4.11 Settings

Settings changes the visual palette without changing models or data. Legal Blue is the default.
Parchment, Tropical, Sunset, Forest, and Midnight demonstrate bright and dark viewing preferences.
The selection is saved locally in `artifacts/preferences.json`. Theme CSS affects presentation
only and is never an input feature to training or evaluation.
## 5. First guided classifier experiment

This sequence teaches every stage of the classifier training process while
keeping the first run manageable.

### 5.1 Confirm the environment

From the repository root:

```powershell
.\doctor.ps1
```

If CUDA is unavailable on the target GPU, review the driver, confirm that the
`cu129` extra was selected, and rerun `uv sync` before training.

### 5.2 Study the first lessons

Open Learn and complete at least the lessons on task framing, provenance,
grouped splitting, tokenization, model loading, and the optimization loop.
Write down the input unit, label set, evaluation unit, and intended user. Those
four choices define what the model actually learns.

### 5.3 Prepare the synthetic dataset

Open Dataset lab and choose **Built-in synthetic**. Inspect several rows. Confirm
that:

- `text` contains one clause-like passage;
- `label` is the target class;
- `contract_id` groups passages from the same synthetic agreement;
- `source` identifies the generator; and
- `license` is MIT.

Review the split preview. No `contract_id` should appear in more than one split.
Save the dataset as the active dataset when the validation panel is acceptable.

### 5.4 Choose a model

Open Model lab and select
`distilbert/distilbert-base-uncased`. Read its model card before using it beyond
this project. Keep the first run at maximum length 256. Longer sequences use
more VRAM and compute.

### 5.5 Configure a learning run

Open Train and begin with:

| Setting | First learning run | Why |
|---|---:|---|
| Epochs | 3 | Enough to observe learning without a long run |
| Learning rate | 0.00002 | Common encoder fine-tuning starting point |
| Train batch size | 8 | Appropriate starting point for 12 GB VRAM |
| Eval batch size | 16 | No backward pass, so evaluation can often be larger |
| Gradient accumulation | 2 | Effective train batch of 16 |
| Maximum length | 256 | Balances clause coverage and memory |
| Weight decay | 0.01 | Mild regularization |
| Warmup ratio | 0.1 | Gradual early learning-rate increase |
| Max gradient norm | 1.0 | Clips unusually large gradients |
| Precision | auto | Chooses supported mixed precision on CUDA |
| Freeze encoder | false | Fine-tunes the encoder and classifier |
| W&B mode | disabled | Establish a local baseline first |

Name the run clearly, for example `distilbert-synthetic-baseline`, then start
training. The first model load can download files from Hugging Face.

### 5.6 Watch the training process

Observe training loss, validation loss, macro F1, epoch progress, and device
information. The loop performs these steps:

1. tokenize a batch;
2. move tensors to the selected device;
3. run the model forward;
4. divide loss for gradient accumulation;
5. backpropagate;
6. clip gradients when an optimizer step is due;
7. update parameters with AdamW;
8. advance the learning-rate scheduler;
9. evaluate on validation data; and
10. save the best checkpoint.

Do not compare runs unless their dataset split, label mapping, seed, and metric
definitions are consistent.

### 5.7 Evaluate and record observations

After the status becomes complete, open Evaluate and answer:

- Which labels have the lowest recall?
- Which label pairs are most often confused?
- Are errors caused by wording, truncation, ambiguous labels, or limited data?
- Does macro F1 tell a different story than overall accuracy?
- What single controlled change should the next run test?

Then open Assistant, load the completed checkpoint, and try synthetic or
non-confidential contract text. Verify every cited passage.

## 6. Dataset design and validation

### 6.1 Required schema

| Column | Requirement | Meaning |
|---|---|---|
| `text` | Required | One clause or passage to classify |
| `label` | Required | Target clause category |
| `contract_id` | Strongly recommended | Source-document grouping key |
| `source` | Recommended | Provenance or collection description |
| `license` | Recommended | Redistribution terms or private designation |

CSV should be UTF-8 encoded with a header row. JSONL should contain one JSON
object per line with the same field names.

### 6.2 Grouped splitting

AI-Enabled Contracts Lab splits unique `contract_id` values, then assigns all rows from each
contract to one partition. At least three distinct IDs are required. The default
ratios are approximately 70 percent training, 15 percent validation, and 15
percent test, subject to whole-contract grouping.

Training data updates parameters. Validation data selects the best checkpoint
and guides experiment choices. Test data estimates final performance and should
not be repeatedly used to tune decisions.

### 6.3 Label quality

Write a labeling guide before scaling annotation. Define positive and negative
examples, overlapping concepts, how to handle multi-purpose clauses, and when an
annotator should abstain. If experts disagree systematically, the model cannot
learn a stable target without a policy for that ambiguity.

### 6.4 The five dataset families in this project

Dataset Lab now distinguishes files by learning purpose:

| Dataset | Purpose | Current use |
|---|---|---|
| Built-in synthetic clauses | Fast, balanced classifier lesson | Recommended first run |
| `datasets/cuad_classifier_sample.csv` | Small real-text classifier lesson | 75 rows, 25 labels, 68 contracts |
| Local `train.jsonl`, `val.jsonl`, and `test.jsonl` exports | Four-task instruction/chat research | Convert the grounded clause-QA rows before classifier training |
| Official CUAD | Full reproducible third-party source | Download and convert on demand |
| Local synthetic 396-label corpus | Broad experimental vocabulary pretraining | 39,600 rows, 396 labels, 21 categories; opt-in and local-only |

The large local JSONL files contain 18,008 instruction examples: clause
extraction, clause question answering, contract-type identification, and
missing-clause audit. They are suitable inputs for a future generative-model
phase, not direct inputs to the current multiclass encoder classifier. The
classifier expects one clause span in `text` and one category in `label`.

The audited local split contains 408 training contracts, 51 validation
contracts, and 51 test contracts. AI-Enabled Contracts Lab verifies that no
contract title occurs in more than one source split.

The synthetic 396-label corpus is intentionally a separate classifier task.
Only 13 label names exactly overlap the 41-label CUAD task after normalization,
so combining the rows into Contra's existing classifier head would silently
change the meaning of the benchmark. The corpus can broaden vocabulary and
support staged pretraining experiments, but it cannot by itself prove better
performance on real contracts.

The August 2026 audit found 15,730 duplicate-text rows, 3,680 exact texts
crossing the supplied partitions, and target-label wording in 73.48 percent of
rows. For held-out examples, 93.38 percent had at least 0.80 TF-IDF similarity
to a same-label training example. The application therefore ignores the
supplied row split for evaluation and derives template-group identifiers before
performing its normal grouped split. This reduces obvious leakage; it does not
turn synthetic templates into an independent legal benchmark.


### 6.5 Use the project datasets in Dataset Lab

Open **Dataset lab** and select **Project datasets**. The catalog shows purpose,
row count, source-contract count, license, status, availability, and local file
size without opening every large JSONL file on each Streamlit rerun.

For the short learning path:

1. Click **Use the public CUAD classifier sample**.
2. Read every validation warning; the sample is intentionally small and imbalanced, so some labels have too few examples for reliable evaluation.
3. Inspect the label chart and individual rows.
4. Confirm the leakage preview reports zero shared `contract_id` values.
5. Continue to Model lab and Train only as an educational experiment.

For the fuller local path:

1. Click **Build classifier data from the local instruction splits**.
2. Wait while the application reads the three files, selects grounded `clause_qa` targets, removes exact duplicates, and checks source-contract overlap.
3. Click **Use the converted local CUAD classifier data**.
4. Inspect the resulting 6,702 rows, 510 contracts, and 41 labels.
5. Review the grouped split preview before training.

The converted file and its metadata are stored under `data/curated/`, which
is ignored by Git. The `original_split` column is retained as an audit trail.
The current workbench still creates its own grouped 70/15/15 split from
`contract_id` when training.

For the 396-label experimental path:

1. Choose **Project datasets** and expand **Experimental: synthetic 396-label corpus**.
2. Review the audit metrics and acknowledge that the data is synthetic.
3. Click **Use all 396 experimental labels**.
4. Confirm the active dataset reports 39,600 examples and 396 labels.
5. Use **Active label catalog** to filter the supplied label names and clause
   names by any of the 21 categories.
6. Review the fresh grouped-split preview. With the default seed, all 396 labels
   are represented in training; some labels are absent from the smaller
   validation and test partitions because template groups remain indivisible.
7. Train this as its own run. Do not compare its synthetic test accuracy directly
   with Contra's 41-label CUAD macro F1.

The initial load computes stable template fingerprints and can take several
seconds. Streamlit caches the prepared frame with a bounded cache and invalidates
it when the source file changes. The raw CSV, duplicate directory, and ZIP are
Git-ignored until generation provenance and redistribution rights are supplied.
The audit summary remains public at datasets/synthetic_396_audit.json.


### 6.6 Run the project conversion manually

From the repository root, build or rebuild the public learning sample:

```powershell
.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode sample
```

Build the full local classifier dataset:

```powershell
.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode local
```

The converter uses only the assistant target from records whose task is
`clause_qa`. Those targets are verbatim positive answer spans grounded in CUAD
annotations. It does not use the generated extraction prose, missing-clause
lists, or contract-type responses as classifier clauses. It carries
`meta.contract_title` into `contract_id`, carries `meta.category` into
`label`, writes CUAD attribution and CC BY 4.0 on every row, and fails if one
contract appears in two named input splits.

Inspect the results:

```powershell
Import-Csv datasets\cuad_classifier_sample.csv | Select-Object -First 5
Get-Content datasets\cuad_classifier_sample.metadata.json
Import-Csv data\curated\cuad_local_classifier.csv |
  Group-Object label |
  Sort-Object Count -Descending |
  Select-Object -First 10 Count, Name
Get-Content data\curated\cuad_local_classifier.metadata.json
```

The supported conversion logic is in `contractlab/data.py`; the small script
in `datasets/prepare_project_datasets.py` is a readable command-line wrapper.
The earlier `cuad_pipeline.py`, `split_dataset.py`, and
`distill_tasks.py` files remain research references. The current classifier
does not call a model provider and does not require an Anthropic API key.

### 6.7 Download and prepare official CUAD

CUAD's official GitHub repository is:

```text
https://github.com/The-Atticus-Project/cuad
```

The repository contains the dataset archive, category descriptions, benchmark
code, and supporting material. The application downloads `data.zip` directly
from that official repository and reads `CUADv1.json`:

```powershell
uv run --extra cu129 contractlab prepare-cuad --output data/curated/cuad_clauses.csv
```

For a smaller learning conversion:

```powershell
uv run --extra cu129 contractlab prepare-cuad --output data/curated/cuad_sample.csv --max-rows 500
```

In the UI, choose **Official CUAD**, set the maximum number of spans, and click
**Download and prepare CUAD**. The command writes a CSV and a neighboring
metadata JSON file containing the source repository, archive URL, dataset-card
reference, license, and transformation description. This direct archive path
avoids CUAD's legacy Hugging Face loading script, which is not supported by
Datasets 5.x. Internet access is required only for this download path.

CUAD v1 contains 510 commercial contracts, more than 13,000 expert annotations,
and 41 review categories. The dataset card declares CC BY 4.0, but also warns
that the creators make no representations or warranties concerning the license
status of the underlying EDGAR contracts. Preserve attribution, document your
transformation, and review the current source terms before redistribution.

### 6.8 Use the verified clauses.csv and taxonomy definitions

The published `datasets/clauses.csv` contains 6,702 grounded positive CUAD
answer spans, 510 source contracts, and all 41 CUAD labels. Unlike the earlier
unverified two-column file, every row includes `contract_id`, source, license,
original task, and original split metadata. The earlier file is quarantined
locally and is not included in the public repository.

Use the verified file directly in the UI by choosing **Project datasets**, or
load it manually:

```powershell
.\.venv\Scripts\python.exe -m contractlab.cli train --config artifacts\configs\YOUR_CONFIG.json
```

The definitions dataset is separate because a taxonomy is knowledge, while
`clauses.csv` is supervised training evidence. Open Dataset Lab and scroll to
**Contract taxonomy and definitions**. Select a domain and contract type to see
its plain-language definition, aliases, typical clauses, and review questions.

The same information is available to code:

```powershell
.\.venv\Scripts\python.exe datasets\build_contract_taxonomy.py
Get-Content datasets\contract_taxonomy.json -TotalCount 40
Get-Content datasets\taxonomy\clause_definitions.jsonl -TotalCount 3
```

The combined file contains 10 domains, 47 contract types, and 45 clause
definitions. Edit the JSONL source records, rebuild, and run the tests whenever
you extend it. Definitions help retrieval, explanations, and dataset design;
they do not by themselves teach the classifier to recognize every concept.
Recognition still requires representative labeled examples.

ContractNLI is the recommended optional NDA research extension. It contains 607
NDAs, 17 fixed hypotheses, three decision states, and evidence spans. It is not
bundled because downloading requires explicit acceptance of the official terms.
Review and accept them yourself at https://stanfordnlp.github.io/contract-nli/
before placing a copy in an ignored local data directory.

### 6.9 Data that must not enter Git

Keep real contracts, private annotations, exported customer data, active
datasets, checkpoints, W&B files, and credentials outside version control.
The repository ignores common local locations, but `.gitignore` is not a
security control. Inspect staged files and Git history before publishing.

### 6.10 Add future datasets safely

Treat every new dataset as a registered source, not as anonymous extra rows.
Keep the raw file local first, then record:

- dataset name, author or generator, and creation date;
- source URLs or source-file lineage;
- license and redistribution permission;
- task type, schema, label definitions, and label-to-existing-label mappings;
- row, label, document, duplicate, and null counts;
- split method and a stable document or template grouping key;
- exact and near-duplicate leakage results; and
- intended role: pretraining, fine-tuning, validation, independent test, retrieval,
  definitions, or chat/instruction learning.

Add a lightweight entry to datasets/dataset_catalog.json and a separate audit
JSON. Large or rights-unclear raw files should remain ignored; the public repo
can include their metadata, acquisition instructions, and transformation code.
Never merge label spaces only because names look similar. Create a reviewed
mapping table, preserve original labels, and keep unchanged real-contract
holdouts for comparison.


### 6.11 Use the 560,000-example multidomain bundle

The local legal_multidomain_all_fields_bundle.zip contains 28 nested ZIP files.
Each domain archive has 20,000 examples, 40 labels, three JSONL partitions, a
combined CSV and JSONL, a label map, and a limitations README. Across the bundle,
there are 448,000 training examples, 56,000 validation examples, 56,000 test
examples, 1,120 domain-label assignments, and 1,111 unique label names.

This is a general-law experiment rather than a contract-clause expansion. It
covers domains including torts, evidence, criminal law, constitutional law,
family law, intellectual property, employment, tax, health, and administrative
law. Keep its classifier head separate from Contra. Nine label names occur in
multiple domains, so preparation converts every target to domain::label. For
example, Copyright_Law::FAIR_USE and another domain's FAIR_USE remain distinct.

#### 6.11.1 Audit before training

The structural audit verified all 560,000 IDs, all 28 label maps, all counts,
one consistent 15-field schema, no nulls, and intact nested ZIP files. The
content audit found:

- 227,261 unique texts and 332,739 duplicate-text rows;
- 54,696 exact texts crossing declared train, validation, or test partitions;
- 293,246 rows participating in exact cross-split text reuse;
- 158 normalized template families, 145 of which cross partitions;
- 559,968 rows participating in cross-split template families; and
- the target label phrase inside 100 percent of input texts.

These facts mean a high synthetic score can reflect answer extraction and
template memorization. Record the run as Contra-Law Experimental. Do not compare
its metric table directly with Contra's unchanged, real-text CUAD holdout.

#### 6.11.2 Prepare every example manually

From the repository root:

~~~powershell
.\.venv\Scripts\python.exe tools\prepare_legal_multidomain.py --bundle datasets\legal_multidomain_all_fields_bundle.zip --output data\curated\legal_multidomain_classifier.csv
~~~

The converter streams nested archives without extracting them. It writes
560,000 rows, retains the supplied partitions and useful metadata, and creates
1,120 domain-qualified targets. The projection is approximately 245 MB and is
ignored by Git.

Validate the result:

~~~powershell
$rows = Import-Csv data\curated\legal_multidomain_classifier.csv
$rows.Count
($rows.label | Sort-Object -Unique).Count
$rows | Group-Object original_split | Select-Object Name, Count
~~~

Expected values are 560000 rows, 1120 labels, and partition counts of
448000 / 56000 / 56000.

#### 6.11.3 Run Contra-Law Experimental

The public configuration uses nlpaueb/legal-bert-base-uncased, one epoch,
BF16, maximum length 96, training batch 32, evaluation batch 64, dynamic
batch tokenization, cross-entropy, and offline Weights & Biases logging:

~~~powershell
.\.venv\Scripts\python.exe -m contractlab.cli train --config configs\contra-law-legalbert-1120-exp-v1.json
~~~

Training uses every supplied record by setting deduplicate_examples to false
and split_strategy to predefined. Those choices reproduce the supplied task;
they do not fix its leakage. Dataset Lab displays the live status file under
Master multidomain legal taxonomy.

#### 6.11.4 What all-fields means

The classifier consumes text and predicts domain::label. The bundle also keeps
issue, rule_summary, application, counterargument, and conclusion. A BERT
sequence-classification head cannot generate those structured fields, so the
classifier preparation does not concatenate them into the input. Doing so would
add even more target wording and produce a misleading result.

Keep the original fields for a separate retrieval or small generative-model
experiment. Their repeated language is analysis scaffolding, not authoritative
or current law. Ground any later legal explanation in reviewed primary sources,
jurisdiction, date, and qualified human review.

#### 6.11.5 Publication and licensing

The bundle, manifest, master taxonomy, prepared CSV, W&B run, and checkpoints
remain local-only. The supplied files do not state a redistribution license,
generator, generation prompts, source inputs, or model version. The Git ignore
rules protect these paths, and datasets/master_taxonomy_audit.json publishes
only non-content audit metadata. Do not upload the raw corpus or trained weights
until those rights and lineage questions are resolved.


## 7. Models and training controls

### 7.1 What is being fine-tuned

The pretrained encoder already represents general language patterns. AI-Enabled Contracts Lab
adds a classification head sized to the dataset label set. With
`freeze_encoder=false`, gradients update both encoder and head. With
`freeze_encoder=true`, only the classification head learns; this is faster and
uses less memory but usually adapts less deeply.

### 7.2 Tokenization and truncation

The tokenizer converts text into token IDs and an attention mask. Batches are
padded to a common length. Text beyond `max_length` is truncated, so an
important clause ending may disappear. Inspect text-length distributions before
raising the limit; Transformer attention cost grows quickly with sequence length.

### 7.3 Effective batch size

The effective training batch is:

```text
train_batch_size x gradient_accumulation_steps
```

With batch size 8 and accumulation 2, the optimizer sees 16 examples per update.
Accumulation reduces peak activation memory but does not make computation free.

### 7.4 Precision

`auto` chooses an appropriate mixed-precision mode on CUDA and full precision
on CPU. `fp16` reduces memory and can accelerate supported NVIDIA GPUs.
`bf16` has a wider exponent range but depends on hardware support. Use `fp32`
when diagnosing numerical instability.

### 7.5 Checkpoint selection

After each epoch, AI-Enabled Contracts Lab evaluates on validation data and saves the best
checkpoint. The untouched test split is evaluated after training. A best
checkpoint is not proof of generalization to different organizations, drafting
styles, jurisdictions, time periods, or scan quality.

### 7.6 Contra v2 completed baseline

Contra is the short project name for the learned clause classifier. Contra v2
uses `nlpaueb/bert-base-uncased-contracts`, class-weighted cross-entropy, four
epochs, BF16, maximum length 256, train batch 4, evaluation batch 8, and gradient
accumulation 4 on the same grouped split as v1.

| Held-out metric | Contra v1 | Contra v2 |
|---|---:|---:|
| Accuracy | 0.6062 | 0.7639 |
| Macro F1 | 0.2986 | 0.6036 |
| Zero-F1 labels | 20 of 41 | 6 of 41 |

The experiment nearly doubled macro F1 and reduced the number of classes with
zero F1. Because base model and loss changed together, this single comparison
cannot isolate which change caused how much improvement. Six very rare labels
still received zero F1, so the result is a baseline rather than a finished legal
system. The reviewed Hugging Face bundle is prepared locally under
`artifacts/huggingface/`; uploading requires a separate authenticated review.

### 7.7 CUAD's original trained models and requirements

CUAD's official repository publishes three 2021 fine-tuned checkpoints through
Zenodo: RoBERTa-base at about 100 million parameters, RoBERTa-large at about 300
million, and DeBERTa-v2-xlarge at about 900 million. These are extractive
question-answering models: given a CUAD category question and contract context,
they select a supporting answer span. Contra is a sequence classifier: given a
passage, it predicts one of 41 labels. Keep the tasks, model heads, preprocessing,
and metrics separate.

The CUAD README says its code was tested with Python 3.8, PyTorch 1.7, and
Transformers 4.3/4.4. Those versions document the original reproduction
environment. Do not downgrade this workbench to them. AI-Enabled Contracts Lab
already requires Hugging Face Transformers and uses its modern `AutoTokenizer`
and `AutoModelForSequenceClassification` APIs with the tested Python 3.12 and
PyTorch environment.

For this 12 GB RTX 5070, the RoBERTa-base checkpoint is the appropriate future
span-extraction experiment. RoBERTa-large may fit some inference configurations
but is a poor first full-fine-tuning target; DeBERTa-v2-xlarge is far beyond the
project's practical full-fine-tuning budget. Download no archive until its
source, terms, checksum, serialization format, and intended task are reviewed.

## 8. Weights & Biases

W&B is optional and disabled by default. AI-Enabled Contracts Lab logs aggregate
configuration and metrics, not clause text, uploaded documents, filenames,
predictions, or contract content.

### 8.1 Authenticate safely

The interactive command avoids placing the API key in a committed file:

```powershell
uv run --extra cu129 wandb login
```

You may also set `WANDB_API_KEY` in the current shell, but remember that shell
history and process inspection can expose secrets. Never paste a key into source
code, a notebook cell that will be committed, `.env`, or
`.streamlit/secrets.toml` unless the file is ignored and appropriately
protected.

### 8.2 Modes

| Mode | Behavior |
|---|---|
| `disabled` | No W&B run is created |
| `offline` | Writes local W&B files for later review or synchronization |
| `online` | Sends allowed aggregate run information to the configured project |

Start with disabled, repeat the baseline in offline mode, inspect the generated
W&B directory, and only then choose online mode. Use a descriptive project name
such as `ai-enabled-contracts-lab-learning`.

### 8.3 What to compare

Change one primary variable at a time: model, learning rate, batch strategy,
maximum length, freezing, or data version. Record the hypothesis before the run.
Use W&B charts to compare learning curves, but confirm final findings with local
artifacts and error examples.

## 9. Evaluation and assistant use

### 9.1 Metric meanings

| Metric | Question it answers |
|---|---|
| Accuracy | What fraction of all examples received the expected label? |
| Precision | When the model predicted this label, how often was it correct? |
| Recall | Of the examples that truly have this label, how many did it find? |
| F1 | What harmonic balance did precision and recall achieve? |
| Macro F1 | What is the unweighted mean F1 across labels? |

Macro F1 gives every class equal importance and is therefore useful when label
frequencies differ. It can still conceal dataset shift and annotation problems,
so pair it with per-label results and error review.

### 9.2 Confusion matrix

Rows represent expected labels and columns represent predicted labels. Large
off-diagonal cells identify systematic confusions. Read the original passages
for those cells before changing the model. The better fix may be a clearer label
definition or better data rather than more epochs.

### 9.3 Error analysis loop

Create a small error taxonomy:

- ambiguous or multi-purpose clause;
- missing context outside the passage;
- truncation;
- label-policy disagreement;
- unusual vocabulary or drafting style;
- duplicate or near-duplicate leakage;
- source parsing failure; and
- confident but unsupported prediction.

Count these categories for a sample of errors. Choose the next experiment based
on the dominant cause.

### 9.4 Six assistant capabilities

The assistant combines local passage retrieval, classifier predictions where a
trained checkpoint is available, and deterministic presentation logic. Each
capability must preserve source passages:

1. **Classification** assigns likely clause types and confidence scores.
2. **Extraction** finds passages relevant to a selected clause concept.
3. **Risk flags** surfaces review candidates; it does not decide legal risk.
4. **Obligations** identifies duty-like language and the supporting text.
5. **Summarization** condenses selected passages with citations.
6. **Question answering** returns a limited answer grounded in retrieved passages.

Always compare the response with the uploaded text. If the source is a scanned
PDF with no extractable text, perform OCR in a separate trusted tool before use.

## 10. Running the code manually

This chapter reproduces the workbench workflow without clicking UI controls.
Use `--extra cpu` instead of `--extra cu129` throughout if you selected the
CPU profile.

### 10.1 Inspect available commands

```powershell
uv run --extra cu129 contractlab --help
uv run --extra cu129 contractlab train --help
```

The package entry point is declared in `pyproject.toml` and calls
`contractlab.cli:main`.

### 10.2 Generate a safe synthetic CSV

This command calls the same generator used by Dataset lab:

```powershell
uv run --extra cu129 python -c "from pathlib import Path; from contractlab.data import synthetic_dataset; p=Path('data/imports/manual_synthetic.csv'); p.parent.mkdir(parents=True, exist_ok=True); synthetic_dataset().to_csv(p, index=False); print(p.resolve())"
```

Inspect the result:

```powershell
Get-Content data\imports\manual_synthetic.csv -TotalCount 6
```

The file is ignored by Git because `data/imports/` is a local workspace.

### 10.3 Prepare CUAD project data manually

Generate the public classifier sample and then the full local conversion:

```powershell
.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode sample
.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode local
```

Validate the local conversion through the same functions used by Dataset Lab:

```powershell
.\.venv\Scripts\python.exe -c "from contractlab.data import load_table, normalize_dataset, validate_dataset; p='data/curated/cuad_local_classifier.csv'; d=normalize_dataset(load_table(p)); print(validate_dataset(d)); print(d.groupby('label').size().sort_values(ascending=False).head(10))"
```

To train on this data manually, set `dataset_path` in the configuration to:

```text
data/curated/cuad_local_classifier.csv
```

Keep `data_license` as `CC BY 4.0` and `data_source` as
`The Atticus Project CUAD v1`. Start with one epoch and W&B disabled to confirm
the pipeline before increasing cost. A 41-label real-text experiment is
substantially slower than the 8-label synthetic lesson.

<!-- pagebreak -->

### 10.4 Create a JSON training configuration

Create `artifacts/configs/manual-first-run.json` with this content:

```json
{
  "run_name": "manual-distilbert-baseline",
  "dataset_path": "data/imports/manual_synthetic.csv",
  "output_dir": "",
  "model_name": "distilbert/distilbert-base-uncased",
  "epochs": 3,
  "learning_rate": 0.00002,
  "train_batch_size": 8,
  "eval_batch_size": 16,
  "gradient_accumulation_steps": 2,
  "max_length": 256,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "max_grad_norm": 1.0,
  "seed": 42,
  "precision": "auto",
  "freeze_encoder": false,
  "wandb_mode": "disabled",
  "wandb_project": "ai-enabled-contracts-lab",
  "log_every": 10,
  "data_license": "MIT",
  "data_source": "AI-Enabled Contracts Lab original synthetic examples"
}
```

You can write the file from PowerShell:

```powershell
New-Item -ItemType Directory -Force artifacts\configs | Out-Null
@'
{
  "run_name": "manual-distilbert-baseline",
  "dataset_path": "data/imports/manual_synthetic.csv",
  "output_dir": "",
  "model_name": "distilbert/distilbert-base-uncased",
  "epochs": 3,
  "learning_rate": 0.00002,
  "train_batch_size": 8,
  "eval_batch_size": 16,
  "gradient_accumulation_steps": 2,
  "max_length": 256,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "max_grad_norm": 1.0,
  "seed": 42,
  "precision": "auto",
  "freeze_encoder": false,
  "wandb_mode": "disabled",
  "wandb_project": "ai-enabled-contracts-lab",
  "log_every": 10,
  "data_license": "MIT",
  "data_source": "AI-Enabled Contracts Lab original synthetic examples"
}
'@ | Set-Content -Encoding utf8 artifacts\configs\manual-first-run.json
```

### 10.5 Run training in the foreground

```powershell
uv run --extra cu129 contractlab train --config artifacts/configs/manual-first-run.json
```

The terminal displays progress and final test metrics. The command stays attached
to the training process, unlike the UI background launcher. Keep the computer
awake and do not close the terminal.

### 10.6 Find and inspect the run

```powershell
Get-ChildItem artifacts\runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

Replace `RUN_DIRECTORY` below with the newest directory name:

```powershell
Get-Content artifacts\runs\RUN_DIRECTORY\status.json
Get-Content artifacts\runs\RUN_DIRECTORY\summary.json
Import-Csv artifacts\runs\RUN_DIRECTORY\test_predictions.csv | Select-Object -First 10
```

You can then start Streamlit and use Evaluate or Assistant to open that run.

### 10.7 Export and run a notebook

```powershell
uv run --extra cu129 contractlab export-notebook --config artifacts/configs/manual-first-run.json --output artifacts/exports/manual-first-run.ipynb
```

Launch JupyterLab without permanently adding it to project dependencies:

```powershell
uv run --extra cu129 --with jupyterlab jupyter lab
```

Open the exported notebook, read each Markdown explanation, and execute one cell
at a time. Confirm that the notebook server started from the repository root so
relative paths resolve correctly. Shut it down with `Ctrl+C` in its terminal.

### 10.8 Export a standalone script

Code lab can generate a configured Python script. Save it under
`artifacts/exports/` and run:

```powershell
uv run --extra cu129 python artifacts/exports/YOUR_SCRIPT.py
```

Read the script first. Generated code captures current settings but remains your
responsibility to validate, especially paths, data provenance, and W&B mode.

### 10.9 Run the core function directly

The CLI is intentionally thin. This Python example shows the underlying call:

```python
from contractlab.config import TrainingConfig
from contractlab.training import run_training

config = TrainingConfig.load("artifacts/configs/manual-first-run.json")
errors = config.validate()
if errors:
    raise ValueError(errors)

result = run_training(config)
print(result["test_metrics"])
```

Save it as a local file under `artifacts/exports/` or execute equivalent cells
in Jupyter.

### 10.10 Run tests, lint, and package build

CUDA development machine:

```powershell
uv run --extra cu129 ruff check .
uv run --extra cu129 pytest
uv build
```

CPU contributor machine:

```powershell
uv run --extra cpu ruff check .
uv run --extra cpu pytest
uv build
```

The build command creates a wheel and source distribution under `dist/`.
Those outputs are ignored and should be inspected before any publication.

## 11. Code walkthrough

### 11.1 Application entry point

`streamlit_app.py` configures the Streamlit application and dispatches to the
page modules under `app_pages/`. UI-specific state and presentation should stay
in those page modules or `contractlab/ui.py`.

### 11.2 Configuration

`contractlab/config.py` defines the `TrainingConfig` dataclass. Its
`validate()` method catches invalid epochs, batch sizes, precision modes,
sequence lengths, and W&B modes before training. Its `save()` and `load()`
methods make experiments reproducible as JSON.

### 11.3 Data pipeline

`contractlab/data.py` contains:

- the original synthetic generator;
- CSV/JSON/JSONL loading;
- schema normalization and validation;
- grouped splitting by `contract_id`;
- split persistence; and
- the optional CUAD converter.

The most important invariant is that one contract ID never crosses partitions.

### 11.4 Model and device selection

`contractlab/modeling.py` holds model presets, device reporting, and memory
estimates. The trainer uses Hugging Face `AutoTokenizer` and
`AutoModelForSequenceClassification`, which allows compatible encoder models
to be selected by model ID.

### 11.5 Handwritten PyTorch training loop

`contractlab/training.py` is the central learning file. In conceptual form:

```python
for epoch in range(config.epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        with autocast_context:
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if step % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    validation_metrics = evaluate_model(model, validation_loader)
    save_checkpoint_when_improved(validation_metrics)
```

Read the actual source because it also handles seeding, label mappings, split
files, precision selection, W&B, progress/status files, the final test pass, and
artifact serialization.

### 11.6 Background jobs

`contractlab/jobs.py` saves a configuration and launches
`python -m contractlab.cli train` as a separate process. It writes standard
output and error logs so the UI can refresh without owning the training loop.
The training function itself is shared with manual runs.

### 11.7 Evaluation and inference

`contractlab/evaluation.py` computes aggregate and per-label metrics, confusion
data, and error-analysis structures. `contractlab/inference.py` reloads a saved
checkpoint and label mapping for prediction.

### 11.8 Assistant and document parsing

`contractlab/document_parser.py` extracts text from supported formats.
`contractlab/assistant.py` implements the six learning capabilities with
source-grounded passages. These modules are demonstrations, not a substitute
for a reviewed retrieval, calibration, and legal-quality assurance system.

### 11.9 Code generation

`contractlab/codegen.py` emits configured scripts and notebooks. Code lab makes
those artifacts visible and downloadable. Keep generated outputs local until
you review their data paths and secrets.

## 12. Artifact and directory reference

| Path | Purpose | Publish by default? |
|---|---|---|
| `app_pages/` | Streamlit workbench pages | Yes |
| `contractlab/` | Reusable Python package | Yes |
| `tests/` | Automated unit tests | Yes |
| `docs/` | Guide, screenshots, architecture, and release notes | Yes |
| `datasets/sample_200.jsonl` | Attributed CUAD instruction sample | Yes, CC BY 4.0 |
| `datasets/cuad_classifier_sample.csv` | Classifier tutorial sample | Yes, CC BY 4.0 |
| `datasets/clauses.csv` | Verified full classifier derivative | Yes, CC BY 4.0 |
| `datasets/train.jsonl/`, `val.jsonl/`, `test.jsonl/` | Instruction splits | Yes through Git LFS, CC BY 4.0 |
| `datasets/contract_taxonomy.json` | Contract and clause definitions | Yes, CC BY 4.0 |
| `datasets/dataset_catalog.json` | Audited inventory and optional datasets | Yes |
| `data/imports/` | Active or uploaded local datasets | No |
| `data/curated/` | Downloaded/transformed third-party data | No |
| `artifacts/configs/` | Local experiment configurations | Usually no |
| `artifacts/runs/` | Checkpoints, metrics, splits, predictions | No |
| `artifacts/huggingface/` | Reviewed local release bundles | No; publish separately |
| `artifacts/exports/` | Generated notebooks and scripts | Review individually |
| `wandb/` | Offline W&B runs and logs | No |
| `.streamlit/secrets.toml` | Local secrets | Never |

A completed run can contain:

- `status.json` for live state;
- `history.json` for epoch history;
- `labels.json` for label-to-ID mapping;
- `summary.json` for configuration, provenance, and final metrics;
- `test_predictions.csv` for error analysis;
- `data_splits/` for the exact train/validation/test CSVs; and
- `best/` for the selected model and tokenizer.

Treat split files and predictions as sensitive if the source data is sensitive.

## 13. Troubleshooting

### 13.1 The batch file closes immediately

Open PowerShell in the repository and run `.\run.ps1` so the error remains
visible. Confirm `uv --version`, rerun `.\setup.ps1`, and inspect whether
port 8501 is already in use.

### 13.2 CUDA is unavailable

Run:

```powershell
uv run --extra cu129 contractlab doctor
nvidia-smi
```

Confirm the GPU and driver are visible, then resynchronize:

```powershell
uv sync --python 3.12 --extra dev --extra cu129
```

If `torch` shows a CPU build, make sure the command did not use `--extra cpu`.

### 13.3 CUDA out of memory

Try these changes in order:

1. lower train batch size from 8 to 4 or 2;
2. raise gradient accumulation to preserve an effective batch;
3. lower maximum length from 256 to 192 or 128;
4. close other GPU applications;
5. enable encoder freezing for a teaching run; or
6. choose a smaller model preset.

Restart the failed run with a new run name. Do not assume cached allocations
represent a memory leak.

### 13.4 Model download fails

Confirm network access and the exact Hugging Face model ID. Some models require
accepting terms or authentication. Corporate proxies and TLS inspection may
require organization-specific configuration. Do not disable certificate
verification as a shortcut.

### 13.5 Dataset validation fails

Check exact lowercase meanings for `text` and `label`, remove empty values,
provide at least two labels and three distinct `contract_id` values, and inspect
duplicate text. Save CSV as UTF-8.

### 13.6 Training completes but scores are poor

Use error analysis before extending training. Common causes are too little data,
inconsistent labels, unrepresentative synthetic text, severe class imbalance,
truncation, or mismatch between train and test drafting styles.

### 13.7 W&B authentication or network errors

Switch to `offline` or `disabled` mode. Verify `wandb login` outside the app.
Never paste an API key into an issue or log. Delete and rotate a key that was
exposed.

### 13.8 A PDF contains no text

The PDF is probably scanned or has unusual text encoding. AI-Enabled Contracts Lab does not
perform OCR in version 0.1.0. Use an approved OCR workflow, review the extracted
text, and upload a supported text-bearing file.

## 14. Privacy, licensing, and responsible use

### 14.1 Local-first is not the same as risk-free

AI-Enabled Contracts Lab processes uploads locally by default, but downloaded models,
optional W&B, backups, antivirus tools, synced folders, and operating-system
telemetry may create other data flows. Use the application only within your
organization's rules and with appropriately authorized data.

### 14.2 Secrets

API keys belong in protected environment variables or ignored secret files.
Never commit `.env`, `.streamlit/secrets.toml`, copied console output
containing a token, or notebook outputs that reveal a key.

### 14.3 Third-party licenses

The repository uses MIT for its original code and synthetic examples. A model,
tokenizer, dataset, or downloaded contract collection can have separate terms.
Record the exact version, source URL, license, attribution, and transformation.
Review current source terms before redistribution.

CUAD-derived rows are not relicensed as MIT merely because they are stored in
this repository. Retain the CC BY 4.0 notice and attribution. The CUAD publisher
also disclaims representations and warranties about the license status of the
underlying EDGAR contracts. Treat dataset licensing, contract-party rights, and
your intended use as separate review questions.

### 14.4 Human review and documentation

For any serious evaluation, document intended use, excluded use, population and
contract coverage, label definitions, dataset dates, limitations, error rates,
calibration, reviewers, and the escalation path when the model is uncertain.

## 15. Move the project to a private GitHub repository

The intended repository is
`https://github.com/RamiJaloudi/ai-enabled-contracts-lab`. The safe sequence is local review,
private creation, private web and clean-clone validation, and only then a separate decision about
public visibility. GitHub documents the local-code workflow at
`https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github`.

### 15.1 Understand what belongs in GitHub

Include source code, tests, configuration, small redistributable examples, documentation, model
cards, evaluation summaries, and reproducibility scripts. Keep out private contracts, chat data,
credentials, `.env`, Streamlit secrets, W&B directories, raw checkpoints, prepared Hugging Face
weights, imported datasets, and generated artifacts. Model weights larger than normal Git limits
belong in Hugging Face; use Git LFS only for reviewed repository files that genuinely require it.
GitHub's LFS guide is at
`https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage`.

### 15.2 Review the local repository

```powershell
git status --short --ignored
git check-ignore -v artifacts/ .env .streamlit/secrets.toml
git diff --check
uv lock --check
uv run --extra cpu ruff check .
uv run --extra cpu pytest
uv build
```

Open `docs/PUBLIC_RELEASE_CHECKLIST.md`. Run a trusted secret scanner, inspect all findings,
review package contents, and verify every file to be staged. Do not use `git add -A` until the
entire worktree has been deliberately approved.

### 15.3 Authenticate and verify the destination

```powershell
gh auth status
gh repo view RamiJaloudi/ai-enabled-contracts-lab
git remote -v
```

`gh auth status` must show the intended `RamiJaloudi` account. If the repository already exists,
stop and inspect its default branch, visibility, and remote before changing anything.

### 15.4 Commit reviewed scope and create the private repository

For an initial project release, stage only the reviewed scope, inspect it, and commit:

```powershell
git add <reviewed-files>
git status --short
git diff --cached --stat
git diff --cached
git commit -m "Initial private release"
```

Then create and push the exact private destination:

```powershell
gh repo create RamiJaloudi/ai-enabled-contracts-lab `
  --private `
  --source . `
  --remote origin `
  --push
```

Never put `--public` in the first command. A private repository limits visibility but does not
undo a leaked secret or establish compliance.

### 15.5 Validate privately

In GitHub, confirm repository visibility, README links, Actions permissions, Dependabot,
licensing, security policy, model/data notices, and the absence of ignored material. Clone the
private repository into a clean directory, install dependencies, run tests, build the package,
start Streamlit, open Contract Chat, and download the Word guide.

### 15.6 Prepare for a later public release

Require CI on protected branches, use read-only default workflow permissions, enable secret
scanning and private vulnerability reporting, and require review when collaborators join. Add
`fine-tuning`, `legal-ai`, `pytorch`, `streamlit`, and `contracts` topics. Change visibility only
after an independent review and repeat the educational, privacy, data-lineage, license, and model-
limitation statements in the release notes.

## 16. Load Contra v2 into a private Hugging Face repository

The intended model repository is
`https://huggingface.co/rjaloudi/contra-contracts-bert-cuad-clause-classifier`. Contra v2 is a
41-label Contracts-BERT clause classifier and the first Hugging Face release target. It is a
routing encoder, not the application's generative LLM and not a whole-contract legal reasoner.

### 16.1 Authenticate safely

Use the official CLI credential store:

```powershell
hf auth login
hf auth whoami
```

The account must be `rjaloudi`. Do not paste the token into Streamlit, Code Lab, source code,
notebooks, screenshots, or chat. Official CLI documentation is at
`https://huggingface.co/docs/huggingface_hub/en/guides/cli`.

### 16.2 Prepare the reviewed bundle

```powershell
.\.venv\Scripts\python.exe tools\prepare_huggingface_model.py `
  --version v2 `
  --repo-id rjaloudi/contra-contracts-bert-cuad-clause-classifier
```

The tool copies safe model and tokenizer files into the Git-ignored
`artifacts/huggingface/contra-contracts-bert-cuad-clause-classifier/` directory and generates a
model card, training configuration, and evaluation summary. Inspect every file. The model card
must disclose 0.6036 macro-F1, 1,008 held-out examples, six zero-F1 labels, the contract-grouped
split, CUAD lineage, CC BY-SA 4.0, and intended-use limits.

### 16.3 Run a local inference smoke test

Load the prepared folder using `AutoTokenizer` and `AutoModelForSequenceClassification`. Confirm
that it loads without custom code, produces 41 logits, has inverse `id2label` and `label2id`
mappings, and emits readable top-k labels. Search all text files for credentials, source contracts,
private paths, chat data, W&B references, and raw dataset content.

### 16.4 Upload privately

```powershell
.\.venv\Scripts\python.exe tools\prepare_huggingface_model.py `
  --version v2 `
  --repo-id rjaloudi/contra-contracts-bert-cuad-clause-classifier `
  --push --private
```

Hugging Face documents uploads at
`https://huggingface.co/docs/huggingface_hub/en/guides/upload`, repository visibility at
`https://huggingface.co/docs/huggingface_hub/en/guides/repository`, and model cards at
`https://huggingface.co/docs/hub/model-cards`.

### 16.5 Inspect and validate privately

While signed in, inspect the rendered card and Files tab. Confirm the exact repository ID, license,
base-model link, CUAD attribution, grouped-holdout wording, all labels, limitations, and example.
The expected bundle contains only the reviewed model, tokenizer, configuration, evaluation, and
card files. From a clean authenticated environment, download the private repository and repeat the
41-logit inference test.

### 16.6 Consider public visibility separately

Only after independent review should visibility change. Preserve the share-alike license, data
attribution, rare-label disclosure, human-review requirement, and statement that a public model is
not legal advice or proof of benchmark leadership. The complete release procedure is also in
`docs/HUGGING_FACE_RELEASE.md`.

## 17. Command and configuration reference

### 17.1 Command summary

| Goal | CUDA command |
|---|---|
| Install | `uv sync --python 3.12 --extra dev --extra cu129` |
| Diagnose | `uv run --extra cu129 contractlab doctor` |
| Start UI | `uv run --extra cu129 streamlit run streamlit_app.py` |
| Train any config | `uv run --extra cu129 contractlab train --config CONFIG.json` |
| Repeat Contra v2 | `.\.venv\Scripts\python.exe -m contractlab.cli train --config artifacts\configs\contra-contracts-bert-weighted-v2.json` |
| Prepare CUAD | `uv run --extra cu129 contractlab prepare-cuad --output FILE.csv` |
| Convert CUAD sample | `.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode sample` |
| Convert full CUAD splits | `.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode local` |
| Rebuild definitions | `.\.venv\Scripts\python.exe datasets\build_contract_taxonomy.py` |
| Rebuild data manifest | `.\.venv\Scripts\python.exe datasets\build_manifest.py` |
| Prepare multidomain bundle | `.\.venv\Scripts\python.exe tools\prepare_legal_multidomain.py` |
| Train Contra-Law Experimental | `.\.venv\Scripts\python.exe -m contractlab.cli train --config configs\contra-law-legalbert-1120-exp-v1.json` |
| Prepare Contra HF bundle | `.\.venv\Scripts\python.exe tools\prepare_huggingface_model.py --version v2 --repo-id rjaloudi/contra-contracts-bert-cuad-clause-classifier` |
| Export notebook | `uv run --extra cu129 contractlab export-notebook --config CONFIG.json` |
| JupyterLab | `uv run --extra cu129 --with jupyterlab jupyter lab` |
| Test | `uv run --extra cu129 pytest` |
| Lint | `uv run --extra cu129 ruff check .` |
| Build | `uv build` |

For CPU, replace `--extra cu129` with `--extra cpu`. A CPU install can inspect
the workbench but Contra v2 training will be substantially slower.

### 17.2 TrainingConfig fields

| Field | Purpose |
|---|---|
| `run_name` | Human-readable run identifier, sanitized for a directory name |
| `dataset_path` | CSV, JSON, or JSONL dataset path |
| `output_dir` | Optional explicit run directory; blank uses managed artifacts |
| `model_name` | Hugging Face model identifier or compatible local path |
| `epochs` | Full passes through training data |
| `learning_rate` | Initial AdamW learning rate |
| `train_batch_size` | Examples per training forward pass |
| `eval_batch_size` | Examples per evaluation forward pass |
| `gradient_accumulation_steps` | Forward/backward passes per optimizer update |
| `max_length` | Maximum tokenized sequence length |
| `weight_decay` | AdamW regularization |
| `warmup_ratio` | Fraction of scheduler steps used for warmup |
| `max_grad_norm` | Gradient-clipping threshold |
| `split_strategy` | `grouped` for document/template groups or `predefined` for validated supplied partitions |
| `deduplicate_examples` | Remove repeated text-label rows when true; false reproduces every supplied record |
| `seed` | Random seed for reproducible splits and initialization behavior |
| `precision` | `auto`, `fp32`, `fp16`, or `bf16` |
| `freeze_encoder` | Whether only the classifier head is trainable |
| `loss_function` | `cross_entropy`, `class_weighted`, or `focal` |
| `focal_gamma` | Hard-example emphasis used only by focal loss |
| `wandb_mode` | `disabled`, `offline`, or `online` |
| `wandb_project` | W&B project name when tracking is enabled |
| `log_every` | Training-step interval for progress logging |
| `data_license` | License or private-use designation saved with the run |
| `data_source` | Dataset provenance saved with the run |

### 17.3 Glossary

**Checkpoint:** Saved model and tokenizer state from a selected point in
training.

**Encoder:** Transformer layers that convert tokens into contextual
representations.

**Epoch:** One complete pass through the training partition.

**Fine-tuning:** Updating a pretrained model's parameters for a labeled target
task.

**Gradient accumulation:** Delaying the optimizer step across multiple smaller
batches.

**Leakage:** Information reaching evaluation data in a way that produces
unrealistic performance.

**Macro F1:** Unweighted mean of per-class F1 scores.

**Provenance:** Where data came from, under what terms, and how it was changed.

**Tokenizer:** Component that maps input text to model token IDs and masks.

**Validation split:** Held-out data used for checkpoint selection and experiment
decisions.

### 17.4 Final learning checklist

- [ ] I can start and stop the app and state its local URL.
- [ ] I can explain the dataset fields and prove contract-level split isolation.
- [ ] I can explain every major training control in my configuration.
- [ ] I have read the central PyTorch training loop.
- [ ] I can run the same configuration from the CLI.
- [ ] I can export and execute a notebook or script.
- [ ] I can interpret macro F1, a confusion matrix, and individual errors.
- [ ] I know exactly what W&B sends and have tested offline mode first.
- [ ] I can locate every artifact produced by a run.
- [ ] I will not publish contracts, secrets, checkpoints, or third-party data accidentally.
- [ ] I have completed the private-first release checklist before any GitHub or Hugging Face push.
- [ ] I can explain why Contra's macro-F1 and zero-F1 labels prevent an unsupported state-of-the-art claim.
