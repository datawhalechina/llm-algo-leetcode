#!/usr/bin/env python3
"""Execute Chapter 0 / Chapter 1 practice notebooks sequentially.

This script is intended for notebooks that are self-contained exercises and
use inline assertions / demo cells, rather than the "question section + answer
section" layout used by later chapters.

Usage:
    python test_chapter0_1_notebooks.py
    python test_chapter0_1_notebooks.py --dir 00_Prerequisites
    python test_chapter0_1_notebooks.py --dir 01_Hardware_Math_and_Systems
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DIRS = [
    ROOT / "00_Prerequisites",
    ROOT / "01_Hardware_Math_and_Systems",
]


def iter_notebooks(base_dirs: list[Path], pattern: str) -> list[Path]:
    notebooks: list[Path] = []
    for base in base_dirs:
        if not base.exists():
            continue
        notebooks.extend(sorted(base.glob(pattern)))
    return notebooks


def run_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict[str, object] = {"__name__": "__main__"}

    print(f"\n{'=' * 72}")
    print(f"Running {path.relative_to(ROOT)}")
    print(f"{'=' * 72}")

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            exec(compile(source, str(path), "exec"), ns, ns)
        except Exception as exc:
            print(f"FAILED at cell {idx}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            return False

    print(f"OK {path.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute Chapter 0 / Chapter 1 practice notebooks sequentially."
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=None,
        help="Directory to scan. Can be passed multiple times. Defaults to both chapter dirs.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ipynb",
        help="Glob pattern to match notebooks. Defaults to all .ipynb files.",
    )
    args = parser.parse_args()

    base_dirs = [Path(d).resolve() for d in args.dir] if args.dir else DEFAULT_DIRS
    notebooks = iter_notebooks(base_dirs, args.pattern)

    if not notebooks:
        print("No notebooks found.")
        return 1

    passed = 0
    failed = 0

    for notebook in notebooks:
        if run_notebook(notebook):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
