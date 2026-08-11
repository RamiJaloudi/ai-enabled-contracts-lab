"""Prepare the CUAD-derived project files for the classifier workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contractlab.data import instruction_jsonl_to_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert grounded CUAD instruction records to classifier CSV."
    )
    parser.add_argument("--mode", choices=("sample", "local"), default="sample")
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    datasets_dir = args.datasets_dir.resolve()
    if args.mode == "sample":
        sources = {"sample": datasets_dir / "sample_200.jsonl"}
        output = args.output or datasets_dir / "cuad_classifier_sample.csv"
    else:
        sources = {
            split: datasets_dir / f"{split}.jsonl" / f"{split}.jsonl"
            for split in ("train", "val", "test")
        }
        output = args.output or Path("data/curated/cuad_local_classifier.csv")

    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing input file(s): " + ", ".join(missing))

    target = instruction_jsonl_to_classifier(sources, output)
    metadata = json.loads(target.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    print(f"[OK] wrote {target}")
    print(
        f"[OK] {metadata['output_rows']} rows, {metadata['contracts']} contracts, "
        f"{metadata['labels']} labels"
    )
    if metadata["contract_overlap_verified"]:
        print("[OK] verified zero source-contract overlap across input splits")


if __name__ == "__main__":
    main()
