# Dataset catalog

AI-Enabled Contracts Lab ships an original synthetic learning dataset under
the repository's MIT license and a complete set of CUAD-derived learning files
under CC BY 4.0.

## Built-in synthetic clauses

- Source: original examples created for AI-Enabled Contracts Lab
- License: MIT
- Scale: 80 clauses, 10 synthetic contracts, 8 balanced labels
- Purpose: fast lessons, smoke tests, and pipeline verification
- Limitation: results are not evidence of real-world legal performance

## Local synthetic 396-label experiment

The user-supplied corpus at datasets/contract_clauses_all_39600.csv contains
39,600 synthetic passages across 396 labels and 21 categories. Dataset Lab can
activate every label as a separate experimental classification task. It derives
stable template-group IDs and creates a fresh grouped split because the source
contains no contract IDs.

This is vocabulary expansion data, not a real-contract benchmark. The audit
found 15,730 duplicate-text rows, 3,680 exact texts crossing the supplied
partitions, label-name wording in 73.48 percent of rows, and high template
similarity between training and evaluation examples. Only 13 labels exactly
overlap the 41-label CUAD taxonomy, so the classifier heads are not merged.

The source's generation provenance and redistribution license were not provided.
The raw CSV, duplicate directory, and ZIP remain Git-ignored and are marked
UNSPECIFIED - LOCAL ONLY. The non-content audit record is
datasets/synthetic_396_audit.json. Keep Contra's unchanged CUAD evaluation as
the real-text baseline when testing whether this synthetic pretraining helps.


## Local 560,000-example multidomain experiment

The local all-fields bundle contains 28 general-law domains, 560,000 synthetic
records, 1,120 domain-label assignments, and 1,111 unique label names. The
classifier projection qualifies targets by domain, preserves all supplied
partitions, and uses every record. Legal-BERT is fine-tuned as Contra-Law
Experimental through configs/contra-law-legalbert-1120-exp-v1.json.

This dataset is structurally complete but not an independent legal benchmark.
Every input includes its target label phrase, 332,739 rows repeat text, and
559,968 rows belong to normalized template families that cross the supplied
partitions. The all-fields analysis is generic scaffolding rather than reviewed
law. See ../datasets/master_taxonomy_audit.json for exact results and
../datasets/README.md for preparation commands.

The source files, prepared 245 MB CSV, run data, and model weights remain local
because provenance and redistribution licensing are unspecified. Only the
non-content audit, converter, configuration, and documentation are intended for
the public repository at this stage.


## CUAD-derived public bundle

CUAD v1 is curated by The Atticus Project and contains 510 English commercial
contracts with more than 13,000 expert annotations across 41 review categories.
The official task is span-oriented question answering. This project's
classifier and instruction transformations are educational derivative tasks,
not official CUAD benchmark results.

All six public files, exact sizes, hashes, row counts, contract counts, and task
counts are recorded in [../datasets/MANIFEST.json](../datasets/MANIFEST.json).
The main files are:

- **datasets/clauses.csv** — 6,702-row, metadata-complete classifier derivative.
- **datasets/cuad_classifier_sample.csv** — 75-row classifier tutorial sample.
- **datasets/sample_200.jsonl** — 200-row instruction/chat sample.
- **datasets/train.jsonl**, **val.jsonl**, and **test.jsonl** — 18,008
  deterministic instruction examples grouped into 408/51/51 contracts.

The classifier converter selects grounded clause-QA assistant targets, keeps
contract IDs and original split lineage, rejects cross-split contract overlap,
and removes exact duplicates. Training performs a second contract-grouped split
for the selected experiment and writes its partitions into the ignored run
directory.

## License and caution

- Official repository: https://github.com/The-Atticus-Project/cuad
## Contract and clause definitions

`datasets/contract_taxonomy.json` combines 47 contract-type definitions and
45 clause definitions across ten domains: commercial, corporate, employment,
family, real estate, finance, intellectual-property/technology, consumer,
government, and healthcare. Records include aliases, typical-clause mappings,
and review questions. Dataset Lab provides an interactive browser for the same
information.

The editable sources are `datasets/taxonomy/contract_types.jsonl` and
`datasets/taxonomy/clause_definitions.jsonl`. Run
`datasets/build_contract_taxonomy.py` to validate references and regenerate
the combined JSON. This is educational, non-jurisdiction-specific reference
material rather than legal advice.

## Optional ContractNLI extension

ContractNLI offers 607 NDAs with 17 fixed hypotheses and evidence annotations.
It is a useful future extension for reasoning about NDA obligations and
omissions. It is not included because the official download requires explicit
click-through acceptance. Review and accept the official terms at
https://stanfordnlp.github.io/contract-nli/ before downloading locally.

- Dataset card: https://huggingface.co/datasets/theatticusproject/cuad-qa
- Declared license: CC BY 4.0
- Local legal text:
  [../datasets/LICENSE-CUAD-CC-BY-4.0.txt](../datasets/LICENSE-CUAD-CC-BY-4.0.txt)

The CUAD publisher makes no representations or warranties regarding the
license status of the underlying SEC EDGAR contracts. Preserve attribution,
identify transformations, and review the current source terms before reuse.

## Private and custom data

Users must have authority to process uploaded data. Mark private files with a
designation such as PRIVATE — DO NOT REDISTRIBUTE. Runtime uploads, active
datasets, checkpoints, and quarantined files remain in ignored directories.
Git ignore rules reduce accident risk but do not replace a staged-file and
history review before publishing.
