"""
split_dataset.py

Splits the combined dataset into train/val/test by CONTRACT, not by row.
Since many examples in clause_extraction/clause_qa/etc. come from the same
underlying contract, a naive random row-level split would put clauses from
the same document in both train and eval — the model could "recognize" a
contract it partially saw during training and its validation numbers would
look better than its real generalization. Splitting by contract_title
guarantees every document is entirely in one split.

Usage:
    python split_dataset.py --in-dir output/ --out-dir output/split/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

random.seed(42)

TASK_FILES = [
    "clause_extraction.jsonl",
    "clause_qa.jsonl",
    "missing_clause_audit.jsonl",
    "contract_type_id.jsonl",
]


def load_records(in_dir: Path) -> list[dict]:
    records = []
    for fname in TASK_FILES:
        path = in_dir / fname
        if not path.exists():
            print(f"  (skipping missing {fname})")
            continue
        with path.open() as f:
            records.extend(json.loads(line) for line in f)
    return records


def split_contracts(records: list[dict], train_frac: float, val_frac: float) -> dict[str, str]:
    contracts = sorted({r["meta"]["contract_title"] for r in records})
    random.shuffle(contracts)
    n = len(contracts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    assignment = {}
    for i, title in enumerate(contracts):
        if i < n_train:
            assignment[title] = "train"
        elif i < n_train + n_val:
            assignment[title] = "val"
        else:
            assignment[title] = "test"
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="output")
    ap.add_argument("--out-dir", default="output/split")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    # test-frac is whatever remains
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(in_dir)
    print(f"Loaded {len(records)} total examples across {len(TASK_FILES)} task files")

    assignment = split_contracts(records, args.train_frac, args.val_frac)
    n_contracts = len(assignment)
    counts = {"train": 0, "val": 0, "test": 0}
    for v in assignment.values():
        counts[v] += 1
    print(
        f"{n_contracts} unique contracts -> train={counts['train']} val={counts['val']} test={counts['test']}"
    )

    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        split = assignment[r["meta"]["contract_title"]]
        buckets[split].append(r)

    for split, recs in buckets.items():
        random.shuffle(recs)
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        by_task: dict[str, int] = {}
        for r in recs:
            by_task[r["task"]] = by_task.get(r["task"], 0) + 1
        print(f"{split:6s} {len(recs):6d} examples -> {out_path}   {by_task}")

    # Sanity check: verify zero contract overlap between splits
    train_titles = {r["meta"]["contract_title"] for r in buckets["train"]}
    val_titles = {r["meta"]["contract_title"] for r in buckets["val"]}
    test_titles = {r["meta"]["contract_title"] for r in buckets["test"]}
    overlap = (
        (train_titles & val_titles) | (train_titles & test_titles) | (val_titles & test_titles)
    )
    if overlap:
        raise RuntimeError(f"LEAKAGE DETECTED: {len(overlap)} contracts appear in multiple splits")
    print("Verified: zero contract overlap between train/val/test.")


if __name__ == "__main__":
    main()
