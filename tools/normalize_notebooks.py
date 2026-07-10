#!/usr/bin/env python3
"""Normalize notebook JSON so generated validators stop warning about cell ids.

This script adds missing cell ids and applies nbformat's canonical notebook
normalization in place. It is intended for source notebooks under the tutorial
chapters, not for generated docs mirrors.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import nbformat
from nbformat import validator
from nbformat.warnings import MissingIDFieldWarning


DEFAULT_ROOTS = [
    Path("00_Prerequisites"),
    Path("01_Hardware_Math_and_Systems"),
    Path("02_PyTorch_Algorithms"),
    Path("03_Triton_Kernels"),
    Path("04_CUDA_and_System_Optimization"),
]


def iter_notebooks(paths: list[Path]) -> list[Path]:
    notebooks: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".ipynb":
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                notebooks.append(path)
            continue
        if path.is_dir():
            for nb in sorted(path.rglob("*.ipynb")):
                resolved = nb.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                notebooks.append(nb)
    return notebooks


def normalize_notebook(path: Path, *, check_only: bool) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingIDFieldWarning)
        nb = nbformat.read(path, as_version=4)
        _, normalized = validator.normalize(nb, version=4)

    original = path.read_text(encoding="utf-8")
    new_text = nbformat.writes(normalized)
    if new_text == original:
        return False

    if check_only:
        print(f"[check] would normalize {path}")
        return True

    path.write_text(new_text, encoding="utf-8")
    print(f"[normalize] updated {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize notebook JSON and add missing cell ids.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Notebook files or directories to normalize. Defaults to all part source directories.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report notebooks that would change without writing them.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in (args.paths or [str(p) for p in DEFAULT_ROOTS])]
    notebooks = iter_notebooks(input_paths)

    if not notebooks:
        print("No notebooks found.")
        return 0

    changed = 0
    for nb_path in notebooks:
        if normalize_notebook(nb_path, check_only=args.check):
            changed += 1

    if args.check and changed:
        print(f"Found {changed} notebooks that need normalization.")
        return 1

    print(f"Processed {len(notebooks)} notebooks; {'would update' if args.check else 'updated'} {changed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
