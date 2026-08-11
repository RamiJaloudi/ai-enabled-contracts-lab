from __future__ import annotations

import argparse
from pathlib import Path

from contractlab.legal_multidomain import prepare_legal_multidomain_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the local legal multidomain bundle for classifier training."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("datasets/legal_multidomain_all_fields_bundle.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/curated/legal_multidomain_classifier.csv"),
    )
    args = parser.parse_args()
    target = prepare_legal_multidomain_classifier(args.bundle, args.output)
    print(f"[OK] wrote {target}")


if __name__ == "__main__":
    main()
