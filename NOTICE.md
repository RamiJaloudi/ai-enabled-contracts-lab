# Third-party notices

AI-Enabled Contracts Lab source code, documentation, and original synthetic
examples are provided under the repository's MIT License. The dataset
derivatives described below have separate terms.

## Contract Understanding Atticus Dataset (CUAD) v1

The public files under **datasets/** contain selected or reformatted material
derived from Contract Understanding Atticus Dataset (CUAD) v1, curated and
maintained by The Atticus Project, Inc.

- Official repository: https://github.com/The-Atticus-Project/cuad
- Dataset card: https://huggingface.co/datasets/theatticusproject/cuad-qa
- Project: https://www.atticusprojectai.org/cuad
- License: Creative Commons Attribution 4.0 International
- License text: https://creativecommons.org/licenses/by/4.0/
- Local license copy:
  [datasets/LICENSE-CUAD-CC-BY-4.0.txt](datasets/LICENSE-CUAD-CC-BY-4.0.txt)
- Paper: CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review,
  Hendrycks, Burns, Chen, and Ball (2021),
  https://arxiv.org/abs/2103.06268

### Included derivatives

- **datasets/sample_200.jsonl**: 200 deterministic instruction examples.
- **datasets/cuad_classifier_sample.csv**: 75 grounded classifier rows.
- **datasets/clauses.csv**: 6,702 grounded classifier rows with contract IDs,
  source, license, and original split lineage.
- **datasets/train.jsonl**, **val.jsonl**, and **test.jsonl**: 18,008
  deterministic instruction examples grouped by 510 source contracts.

The transformations parse and template CUAD contract text and annotations into
clause extraction, grounded clause Q&A, contract-type identification,
missing-clause audit, and multiclass clause-classification records. No language
model generated the included target answers. The implementation is documented
in the dataset scripts and [datasets/MANIFEST.json](datasets/MANIFEST.json).

The Atticus Project states that it makes no representations or warranties about
the license status of the underlying contracts, which are publicly available
from SEC EDGAR. Review CUAD's current datasheet, privacy policy, disclaimer, and
terms before reuse. This project is not endorsed by The Atticus Project or the
contract parties.

## Base model

Contra is fine-tuned from distilbert/distilbert-base-uncased. Review its current
Hugging Face model card and Apache 2.0 license before redistributing the
fine-tuned model bundle.
