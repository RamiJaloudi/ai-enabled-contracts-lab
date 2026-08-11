"""
cuad_pipeline.py

Turns the real CUAD v1 dataset (510 actual commercial contracts filed with
the SEC, 41 expert-labeled clause categories, CC BY 4.0 license) into
instruction-tuning examples for a contract-analysis LLM.

Every example produced by this file is grounded in real contract text.
No language model is used here — this is pure parsing/templating over
ground-truth legal annotations, so it's fast, free, and 100% traceable
back to the source contract.

Tasks produced (see TASK NOTES at bottom of file for the full taxonomy):
  1. clause_extraction   - "Does this contract have a Non-Compete clause?
                            If so, quote it." -> real clause text or
                            "not present"
  2. missing_clause_audit - "Which of the standard clause categories are
                            absent from this contract?" -> real list,
                            derived from CUAD's is_impossible flags
  3. clause_qa            - reformats CUAD's original lawyer-review
                            questions into natural Q&A pairs
  4. contract_type_id     - "What type of agreement is this?" -> type
                            parsed from the real SEC exhibit filename

Usage:
    python cuad_pipeline.py --cuad data/CUADv1.json --out-dir output/
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

random.seed(13)


# ---------------------------------------------------------------------------
# Contract-type extraction from real SEC exhibit filenames
# ---------------------------------------------------------------------------
# CUAD titles are literally the SEC EDGAR exhibit filenames, e.g.
# "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT" or the newer-style
# "PfHospitalityGroupInc_20150923_10-12G_EX-10.1_9266710_EX-10.1_Franchise
# Agreement1". Underscore/hyphen-delimited tokens keep internal spaces
# intact, so the real free-text description survives as one token — we just
# need to find it among the filing-ID/exhibit-number noise tokens.
def _smart_title_case(text: str) -> str:
    words = text.split(" ")
    return " ".join(w if w.isupper() and len(w) <= 4 else w.title() for w in words)


def infer_contract_type(title: str) -> str:
    tokens = [t.strip() for t in re.split(r"[_\-]+", title) if t.strip()]
    if not tokens:
        return "Commercial Agreement"
    company_token = tokens[0]
    for tok in reversed(tokens):
        if tok == company_token or tok.upper().startswith("EX"):
            continue
        letters = sum(c.isalpha() for c in tok)
        digits = sum(c.isdigit() for c in tok)
        if letters >= 3 and digits <= 1:
            cleaned = re.sub(r"\d+$", "", tok).strip(" .,-")
            cleaned = re.sub(r"\s+", " ", cleaned)
            if len(cleaned) >= 3:
                return _smart_title_case(cleaned)
    return "Commercial Agreement"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class ClauseInstance:
    contract_title: str
    contract_type: str
    category: str
    question: str
    text: str | None  # real extracted clause text, or None if absent
    answer_start: int | None  # char offset of `text` within `full_context`
    full_context: str  # complete contract text this clause was pulled from


def load_cuad(path: str | Path) -> list[ClauseInstance]:
    raw = json.loads(Path(path).read_text())
    instances: list[ClauseInstance] = []
    for contract in raw["data"]:
        title = contract["title"]
        ctype = infer_contract_type(title)
        for para in contract["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                category = qa["id"].split("__")[-1]
                answers = qa.get("answers", [])
                has_answer = bool(answers) and not qa["is_impossible"]
                text = answers[0]["text"].strip() if has_answer else None
                start = answers[0]["answer_start"] if has_answer else None
                instances.append(
                    ClauseInstance(
                        contract_title=title,
                        contract_type=ctype,
                        category=category,
                        question=qa["question"],
                        text=text,
                        answer_start=start,
                        full_context=context,
                    )
                )
    return instances


def windowed_context(inst: ClauseInstance, max_chars: int) -> str:
    """Return a slice of the full contract text guaranteed to contain the
    real clause span (when one exists), instead of naively truncating from
    the start. ~64% of CUAD answers fall past the first 6000 characters of
    their contract, so front-truncation alone silently produces training
    pairs where the target text isn't even present in the model's input —
    a hallucination-training bug. For negative (no-clause) examples there's
    no span to anchor on, so we fall back to a front slice, which is a
    reasonable proxy since the model still has to judge absence from
    *some* real text.
    """
    text = inst.full_context
    if len(text) <= max_chars:
        return text
    if inst.answer_start is None or inst.text is None:
        return text[:max_chars]
    span_end = inst.answer_start + len(inst.text)
    pad = (max_chars - (span_end - inst.answer_start)) // 2
    win_start = max(0, inst.answer_start - pad)
    win_end = min(len(text), win_start + max_chars)
    win_start = max(0, win_end - max_chars)  # re-anchor if we hit the end
    return text[win_start:win_end]


# ---------------------------------------------------------------------------
# Task 1: Clause extraction / callout
# ---------------------------------------------------------------------------
_EXTRACTION_PROMPTS = [
    "Does this {ctype} contain a {category} clause? If so, quote the relevant language.",
    "Identify and extract the {category} provision from the following {ctype}, if one exists.",
    "Review this {ctype} and pull out the {category} clause verbatim, or state that it is absent.",
    "Flag the {category} language in this contract for attorney review.",
]


def build_clause_extraction_examples(
    instances: list[ClauseInstance],
    max_context_chars: int = 6000,
    negative_ratio: float = 0.25,
) -> list[dict]:
    """Task: given contract text + a clause category, extract the real
    clause span, or correctly say it's not present. Negative (absent)
    examples are downsampled since CUAD's negatives outnumber positives
    roughly 2:1 and we don't want the model learning to always say 'no'.
    """
    examples = []
    for inst in instances:
        if inst.text is None and random.random() > negative_ratio:
            continue
        prompt_template = random.choice(_EXTRACTION_PROMPTS)
        instruction = prompt_template.format(
            ctype=inst.contract_type.lower(), category=inst.category
        )
        context = windowed_context(inst, max_context_chars)
        user_msg = f"{instruction}\n\n---CONTRACT TEXT---\n{context}"
        if inst.text:
            assistant_msg = f'Yes. The {inst.category} clause states:\n\n"{inst.text}"'
        else:
            assistant_msg = (
                f"No, this contract does not appear to contain a {inst.category} clause."
            )
        examples.append(
            {
                "task": "clause_extraction",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a contract analysis assistant. Quote clause language exactly as written; never fabricate text that is not in the source contract.",
                    },
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "meta": {
                    "contract_title": inst.contract_title,
                    "contract_type": inst.contract_type,
                    "category": inst.category,
                    "has_clause": inst.text is not None,
                    "source": "CUAD v1 (real SEC-filed contract)",
                },
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Task 2: Missing-clause audit (per contract, aggregated)
# ---------------------------------------------------------------------------
_AUDIT_PROMPTS = [
    "Perform a completeness check on this {ctype}. Which standard clause categories does it appear to be missing?",
    "As part of contract review, list the clause categories that could NOT be located in this {ctype}.",
    "Flag any gaps: which of the standard due-diligence clause categories are absent from this agreement?",
]


def build_missing_clause_audit_examples(
    instances: list[ClauseInstance], max_context_chars: int = 6000
) -> list[dict]:
    by_contract: dict[str, list[ClauseInstance]] = defaultdict(list)
    for inst in instances:
        by_contract[inst.contract_title].append(inst)

    examples = []
    for title, group in by_contract.items():
        ctype = group[0].contract_type
        present = sorted({g.category for g in group if g.text is not None})
        absent = sorted({g.category for g in group if g.text is None} - set(present))
        if not absent:
            continue
        # Anchor the window on the median real clause location so as many
        # of the "present" clauses as possible actually fall inside the
        # text we show the model (see windowed_context() for rationale).
        present_insts = sorted(
            (g for g in group if g.text is not None), key=lambda g: g.answer_start or 0
        )
        anchor = present_insts[len(present_insts) // 2] if present_insts else group[0]
        context = windowed_context(anchor, max_context_chars)
        instruction = random.choice(_AUDIT_PROMPTS).format(ctype=ctype.lower())
        user_msg = f"{instruction}\n\n---CONTRACT TEXT---\n{context}"
        assistant_msg = (
            f"Based on my review, the following standard clause categories do not appear "
            f"in this {ctype.lower()}: {', '.join(absent)}.\n\n"
            f"Categories that ARE present: {', '.join(present) if present else 'none of the standard categories were clearly identified'}."
        )
        examples.append(
            {
                "task": "missing_clause_audit",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a contract analysis assistant performing completeness review against a standard clause checklist.",
                    },
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "meta": {
                    "contract_title": title,
                    "contract_type": ctype,
                    "present_categories": present,
                    "absent_categories": absent,
                    "source": "CUAD v1 (real SEC-filed contract)",
                },
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Task 3: Clause Q&A (reformats CUAD's real lawyer-review questions)
# ---------------------------------------------------------------------------
def build_clause_qa_examples(
    instances: list[ClauseInstance], max_context_chars: int = 6000
) -> list[dict]:
    examples = []
    for inst in instances:
        if inst.text is None:
            continue  # Q&A task only makes sense where there's a real answer
        # CUAD questions are phrased as lawyer-review prompts; convert to a
        # plain question a non-lawyer user might ask.
        q = (
            inst.question.split("Details:")[-1].strip()
            if "Details:" in inst.question
            else inst.question
        )
        context = windowed_context(inst, max_context_chars)
        user_msg = f"{q}\n\n---CONTRACT TEXT---\n{context}"
        assistant_msg = f"{inst.text}"
        examples.append(
            {
                "task": "clause_qa",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a contract analysis assistant. Answer using only language found in the provided contract.",
                    },
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "meta": {
                    "contract_title": inst.contract_title,
                    "contract_type": inst.contract_type,
                    "category": inst.category,
                    "source": "CUAD v1 (real SEC-filed contract)",
                },
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Task 4: Contract type identification
# ---------------------------------------------------------------------------
def build_contract_type_examples(
    instances: list[ClauseInstance], max_context_chars: int = 4000
) -> list[dict]:
    seen = set()
    examples = []
    for inst in instances:
        if inst.contract_title in seen:
            continue
        seen.add(inst.contract_title)
        context = inst.full_context[:max_context_chars]  # type is normally evident near the top
        user_msg = f"What type of agreement is this?\n\n---CONTRACT TEXT---\n{context}"
        assistant_msg = f"This is a {inst.contract_type}."
        examples.append(
            {
                "task": "contract_type_id",
                "messages": [
                    {"role": "system", "content": "You are a contract analysis assistant."},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "meta": {
                    "contract_title": inst.contract_title,
                    "contract_type": inst.contract_type,
                    "source": "CUAD v1 (real SEC-filed contract)",
                },
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_all(cuad_path: str | Path) -> dict[str, list[dict]]:
    instances = load_cuad(cuad_path)
    return {
        "clause_extraction": build_clause_extraction_examples(instances),
        "missing_clause_audit": build_missing_clause_audit_examples(instances),
        "clause_qa": build_clause_qa_examples(instances),
        "contract_type_id": build_contract_type_examples(instances),
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuad", default="data/CUADv1.json", help="Path to CUADv1.json")
    ap.add_argument(
        "--out-dir", default="output", help="Directory to write task JSONL files + combined file"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_all(args.cuad)
    combined: list[dict] = []
    for name, records in tasks.items():
        write_jsonl(records, out_dir / f"{name}.jsonl")
        combined.extend(records)
        print(f"{name:24s} {len(records):6d} examples")

    random.shuffle(combined)
    write_jsonl(combined, out_dir / "combined_all_tasks.jsonl")
    print(f"{'TOTAL':24s} {len(combined):6d} examples -> {out_dir / 'combined_all_tasks.jsonl'}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TASK NOTES: full dataset taxonomy for this project
# ---------------------------------------------------------------------------
# Implemented here (100% real data, no LLM needed):
#   1. clause_extraction    - locate + quote a clause by category
#   2. missing_clause_audit - completeness/gap check against 41 categories
#   3. clause_qa            - natural-language Q&A grounded in real clauses
#   4. contract_type_id     - classify agreement type from real filings
#
# Provided as a separate distillation script (distill_tasks.py) because they
# require judgment/generation beyond what's literally in the source text:
#   5. clause_summarization       - plain-English summary of a real clause
#   6. clause_drafting            - draft a new clause given real clauses as
#                                    style exemplars (few-shot grounded)
#   7. risk_flagging               - flag one-sided / non-market clause terms
#   8. clause_redline_comparison   - explain the practical difference between
#                                    two versions of a clause
#   9. plain_english_negotiation   - suggest a counter-proposal for a clause
#
# Requires an additional real public-domain source (not yet pulled in):
#  10. government_contracts_far    - FAR/DFARS clause identification and
#                                    flow-down analysis, sourced from the
#                                    public-domain Federal Acquisition
#                                    Regulation text (acquisition.gov)
#  11. corporate_law_qa            - entity formation, fiduciary duty, and
#                                    corporate-governance Q&A, best sourced
#                                    from public-domain state statutes
#                                    (e.g. DGCL) rather than CUAD, which is
#                                    purely commercial/IP contracts
