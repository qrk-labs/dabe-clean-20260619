#!/usr/bin/env python3
"""Replace the Zenodo DOI placeholder inside the Zenodo deposit kit."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PLACEHOLDER = "10.5281/zenodo.TODO"
TARGETS = [
    "zenodo/README.md",
    "zenodo/CHECKLIST.md",
    "zenodo/UPLOAD_FIELDS.md",
    "zenodo/CODE_AND_DATA_AVAILABILITY.md",
    "zenodo/LICENSES.md",
    "zenodo/REPRODUCIBILITY.md",
    "zenodo/artifact_manifest.json",
    "zenodo/metadata/paper_record.json",
    "zenodo/metadata/software_record.json",
    "zenodo/metadata/CITATION.cff",
    "zenodo/metadata/.zenodo.json",
    "zenodo/metadata/codemeta.json",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 zenodo/scripts/replace_doi.py 10.5281/zenodo.<id>")
        return 2

    doi = sys.argv[1].strip()
    if not re.fullmatch(r"10\.5281/zenodo\.[0-9A-Za-z_.-]+", doi):
        print(f"Refusing DOI that does not look like a Zenodo DOI: {doi}")
        return 2

    repo = Path(__file__).resolve().parents[2]
    changed = 0
    for rel in TARGETS:
        path = repo / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new_text = text.replace(PLACEHOLDER, doi)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1

    print(f"Updated {changed} files from {PLACEHOLDER} to {doi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
