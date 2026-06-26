#!/usr/bin/env python3
"""Check that source chapter prose files have matching docs mirror pages.

This repo keeps prose and notebook content in the source tree, and the docs
tree is expected to mirror the prose pages. Notebook files are validated by
separate notebook tests; this script only checks Markdown pages.

Usage:
    python3 check_source_docs_mirror.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = [
    ROOT / "00_Prerequisites",
    ROOT / "01_Hardware_Math_and_Systems",
    ROOT / "02_PyTorch_Algorithms",
    ROOT / "03_Triton_Kernels",
    ROOT / "04_CUDA_and_System_Optimization",
]
DOCS_DIR = ROOT / "docs"
def iter_source_pages() -> set[Path]:
    expected: set[Path] = set()
    for base in SOURCE_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix != ".md":
                continue
            rel = path.relative_to(ROOT)
            expected.add(Path("docs") / rel.with_suffix(".md"))
    return expected


def iter_docs_pages() -> set[Path]:
    actual: set[Path] = set()
    for base in SOURCE_DIRS:
        docs_base = DOCS_DIR / base.name
        if not docs_base.exists():
            continue
        for path in docs_base.rglob("*.md"):
            actual.add(path.relative_to(ROOT))
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that source chapter files have matching docs mirror pages."
    )
    _ = parser.parse_args()

    expected = iter_source_pages()
    actual = iter_docs_pages()

    missing = sorted(expected - actual)
    if not missing:
        print("Source/docs mirror check passed.")
        return 0

    if missing:
        print("Missing docs mirror pages:")
        for path in missing:
            print(f"  - {path.as_posix()}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
