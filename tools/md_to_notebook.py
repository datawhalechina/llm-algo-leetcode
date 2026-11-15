#!/usr/bin/env python3
"""Convert Part 0 / Part 1 markdown files into notebook sources.

This helper is meant for the notebook-first migration of Part 0 / Part 1.
It keeps the markdown content and splits fenced code blocks into code cells so
the generated notebook can serve as a first-class source artifact.

Usage:
    python tools/md_to_notebook.py --dir 00_Prerequisites --dry-run
    python tools/md_to_notebook.py --dir 01_Hardware_Math_and_Systems --dry-run
    python tools/md_to_notebook.py --file 00_Prerequisites/02_PyTorch_Tensor_Fundamentals.md --dry-run

Review the preview before omitting --dry-run. Intro and numbered group pages
(for example, 1A.md and 1F.md) remain Markdown and are excluded from conversion.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent.parent


def is_group_page(path: Path) -> bool:
    """Recognize numbered group indexes without limiting their letter to A--E."""
    return bool(re.fullmatch(r"[0-9]+[A-Z]", path.stem))


def iter_source_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".md":
            if path.name == "intro.md" or is_group_page(path):
                continue
            files.append(path)
            continue
        if path.is_dir():
            for md in sorted(path.glob("*.md")):
                if md.name == "intro.md" or is_group_page(md):
                    continue
                files.append(md)
    return files


def markdown_to_cells(text: str, *, split_code: bool) -> list:
    if not split_code:
        content = text.strip("\n")
        return [new_markdown_cell(content)] if content.strip() else []

    cells = []
    md_buf: list[str] = []
    code_buf: list[str] = []
    in_code = False

    def flush_markdown() -> None:
        nonlocal md_buf
        content = "\n".join(md_buf).strip("\n")
        if content.strip():
            cells.append(new_markdown_cell(content))
        md_buf = []

    def flush_code() -> None:
        nonlocal code_buf
        content = "\n".join(code_buf).rstrip("\n")
        if content.strip():
            cells.append(new_code_cell(content))
        code_buf = []

    for line in text.splitlines():
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_markdown()
                in_code = True
            continue

        if in_code:
            code_buf.append(line)
        else:
            md_buf.append(line)

    if in_code:
        flush_code()
    else:
        flush_markdown()

    return cells


def convert_md_to_ipynb(
    md_path: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    split_code: bool = False,
) -> Path:
    ipynb_path = md_path.with_suffix(".ipynb")
    if ipynb_path.exists() and not overwrite:
        return ipynb_path

    text = md_path.read_text(encoding="utf-8")
    cells = markdown_to_cells(text, split_code=split_code)
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
    )

    if dry_run:
        print(f"[dry-run] create notebook {md_path} -> {ipynb_path} ({len(cells)} cells)")
        return ipynb_path

    ipynb_path.write_text(nbformat.writes(nb), encoding="utf-8")
    return ipynb_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Part 0 / Part 1 markdown files to notebook sources.")
    parser.add_argument("--dir", dest="dirs", action="append", help="Directory to scan for Part 0 / Part 1 markdown files.")
    parser.add_argument("--file", dest="files", action="append", help="Specific markdown file to convert.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing notebook files.")
    parser.add_argument(
        "--split-code",
        action="store_true",
        help="Split fenced code blocks into code cells instead of keeping the notebook markdown-only.",
    )
    args = parser.parse_args()

    targets: list[Path] = []
    if args.files:
        targets.extend(Path(p) for p in args.files)
    if args.dirs:
        targets.extend(Path(d) for d in args.dirs)
    if not targets:
        targets.extend([ROOT / "00_Prerequisites", ROOT / "01_Hardware_Math_and_Systems"])

    source_files = iter_source_files(targets)
    if not source_files:
        print("No markdown part files found.")
        return 1

    created = 0
    skipped = 0
    for md_path in source_files:
        ipynb_path = md_path.with_suffix(".ipynb")
        if ipynb_path.exists() and not args.overwrite:
            skipped += 1
            continue
        convert_md_to_ipynb(
            md_path,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            split_code=args.split_code,
        )
        created += 1

    print(f"Processed {len(source_files)} markdown files; created {created}; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
