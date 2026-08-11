"""Map guide headings to rendered PDF page numbers for the static TOC."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_user_guide import heading_inventory, parse_markdown
from pypdf import PdfReader


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description="Map guide headings to PDF pages")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    headings = heading_inventory(parse_markdown(args.source))
    pages = [normalize(page.extract_text() or "") for page in PdfReader(args.pdf).pages]
    body_marker = normalize(
        "AI-Enabled Contracts Lab is a local Streamlit workbench for learning how an existing"
    )
    body_start = next(
        (index for index, text in enumerate(pages) if body_marker in text),
        0,
    )

    mapping: dict[str, int] = {}
    missing: list[str] = []
    for _level, heading, _anchor in headings:
        needle = normalize(heading)
        page_number = next(
            (index + 1 for index in range(body_start, len(pages)) if needle in pages[index]),
            None,
        )
        if page_number is None:
            missing.append(heading)
        else:
            mapping[heading] = page_number

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"[OK] mapped {len(mapping)} headings from page {body_start + 1}")
    if missing:
        print("[WARN] headings not found:")
        for heading in missing:
            print(f"  - {heading}")


if __name__ == "__main__":
    main()
