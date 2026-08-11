# Datasets

This directory holds the public learning-data bundle and registered local-only
experiments for AI-Enabled Contracts Lab. CUAD derivatives retain CUAD's CC BY
4.0 terms; the project-authored contract taxonomy is separately licensed CC BY
4.0. Rights-unclear raw experiments remain Git-ignored. The repository's Python
source uses the MIT license.

## Inventory

| File | Purpose | Scale |
|---|---|---:|
| `sample_200.jsonl` | Small instruction/chat sample | 200 rows |
| `cuad_classifier_sample.csv` | Fast classifier tutorial | 75 rows / 25 labels |
| `clauses.csv` | Full classifier fine-tuning | 6,702 rows / 510 contracts / 41 labels |
| `train.jsonl/train.jsonl` | Instruction train split | 14,491 rows / 408 contracts |
| `val.jsonl/val.jsonl` | Instruction validation split | 1,784 rows / 51 contracts |
| `test.jsonl/test.jsonl` | Instruction test split | 1,733 rows / 51 contracts |
| `contract_taxonomy.json` | Combined knowledge base | 10 domains / 47 contract types / 45 clause definitions |
| `taxonomy/contract_types.jsonl` | Source contract definitions | 47 definitions |
| `taxonomy/clause_definitions.jsonl` | Source clause concepts | 45 definitions |
| Local synthetic 396-label corpus | Experimental classifier vocabulary | 39,600 rows / 396 labels / local-only |
| manifest.csv | Local multidomain archive index | 28 domains / 560,000 declared examples |
| master_taxonomy.json | Local general-legal label registry | 1,120 assignments / 1,111 unique names |
| legal_multidomain_all_fields_bundle.zip | Local synthetic all-fields source | 560,000 rows / 1,120 domain-qualified labels |

The instruction splits total 18,008 examples. Complete source contracts are
assigned to one partition, and `MANIFEST.json` verifies zero contract overlap.
Large instruction files use Git LFS.
## Local synthetic 396-label experiment

Dataset Lab registers all 396 labels from the user-supplied synthetic corpus as
a separate task. At load time, the app derives stable template-group IDs so
similar templates stay together during its fresh grouped split. The original
train, validation, and test fields are retained only as audit metadata.

The measured audit found substantial exact and near-template leakage, including
15,730 duplicate-text rows and 3,680 exact texts crossing the declared splits.
The data can support pipeline learning, label-vocabulary experiments, and
staged pretraining, but its own test scores are not evidence of performance on
real contracts. See synthetic_396_audit.json for the complete summary.

The source CSV, duplicate directory, and ZIP are intentionally Git-ignored.
Their author or generator, generation inputs, and redistribution license have
not been documented, so the app marks them UNSPECIFIED - LOCAL ONLY. Do not add
them to a public release until those facts and rights are recorded.
## Master multidomain legal experiment

The local bundle contains 28 nested domain archives and 560,000 structurally
complete synthetic records: 448,000 train, 56,000 validation, and 56,000 test.
Each domain has 40 labels and 20,000 examples. Nine label names are reused by
more than one domain, so the classifier target is domain-qualified, such as
Copyright_Law::FAIR_USE. This produces 1,120 unambiguous targets.

The bundle passed ZIP, schema, ID, label-map, row-count, and null validation.
Its content audit also found severe leakage: 332,739 duplicate-text rows,
54,696 exact texts crossing declared splits, 559,968 rows in template families
that cross splits, and the target label phrase in 100 percent of input texts.
This run is therefore named Contra-Law Experimental and remains separate from
Contra's real-text CUAD benchmark.

Prepare all classifier rows without extracting the nested archives:

~~~powershell
.\.venv\Scripts\python.exe tools\prepare_legal_multidomain.py
~~~

Run the published one-epoch Legal-BERT configuration:

~~~powershell
.\.venv\Scripts\python.exe -m contractlab.cli train --config configs\contra-law-legalbert-1120-exp-v1.json
~~~

The classifier consumes text and predicts domain::label. The all-fields source
retains issue, rule summary, application, counterargument, and conclusion for a
later retrieval or generative experiment. A BERT classification head cannot
generate those fields. They are generic synthetic scaffolding rather than
current, jurisdiction-specific legal authority.

The bundle, manifest, taxonomy, projection, and checkpoints are local-only and
Git-ignored. Their license, generator, prompts, source inputs, and model version
must be documented before any public redistribution. See
master_taxonomy_audit.json for machine-readable audit results.





## CUAD source and attribution

- Source: Contract Understanding Atticus Dataset (CUAD) v1
- Curator: The Atticus Project, Inc.
- Repository: https://github.com/The-Atticus-Project/cuad
- Dataset card: https://huggingface.co/datasets/theatticusproject/cuad-qa
- Paper: https://arxiv.org/abs/2103.06268
- License: CC BY 4.0
- Local terms: `LICENSE-CUAD-CC-BY-4.0.txt`

The included targets are deterministic transformations of CUAD annotations; no
language model generated them. The publisher makes no representations or
warranties about the license status of underlying SEC EDGAR contracts. Review
the current terms and the rights relevant to your intended use.

The published `clauses.csv` is the verified replacement for an earlier
unattributed two-column export. It contains text, label, contract ID, source,
license, original task, and split metadata. The unverified original is
quarantined locally and is not published.

## Contract taxonomy and definitions

The taxonomy is project-authored educational reference data. Each clause record
has a plain-language definition, aliases, domains, and review questions. Each
contract-type record has a definition, domain, aliases, and typical clause IDs.
It is broader than CUAD, but is not a jurisdiction-specific legal ontology or
legal advice.

Rebuild and validate the combined file:

```powershell
.\.venv\Scripts\python.exe datasets\build_contract_taxonomy.py
```

Application code can load `datasets/contract_taxonomy.json`; researchers can
edit the two JSONL sources and rebuild it.

## Reproduce and verify

```powershell
.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode sample
.\.venv\Scripts\python.exe datasets\prepare_project_datasets.py --mode local
.\.venv\Scripts\python.exe datasets\build_contract_taxonomy.py
.\.venv\Scripts\python.exe datasets\build_manifest.py
```

## Optional NDA research extension

ContractNLI is a useful next dataset for NDA entailment, contradiction,
not-mentioned decisions, and evidence spans. It is not bundled because its
official download requires explicit acceptance of dataset terms. Visit
https://stanfordnlp.github.io/contract-nli/, accept the terms yourself, and
keep any download in an ignored local directory until redistribution rights
have been independently confirmed.

## Responsible use

These files are for education and research. They are not representative of all
contracts or legal systems and must not drive automated legal decisions.
Preserve attribution and do not add private contracts, credentials,
checkpoints, or W&B artifacts here.
