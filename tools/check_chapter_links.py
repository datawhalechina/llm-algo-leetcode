#!/usr/bin/env python3
"""Check local markdown links for Chapter 0 / Chapter 1 documentation.

Usage:
    python tools/check_chapter_links.py

The script can scan either the docs site content or the source markdown copies
of Chapter 0 / Chapter 1, then verify that local relative links resolve to
existing files.
External URLs and same-page anchors are ignored.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs"


def iter_markdown_files(scope: str) -> Iterable[Path]:
    if scope == "docs":
        roots = [
            DOC_ROOT / "00_Prerequisites",
            DOC_ROOT / "01_Hardware_Math_and_Systems",
        ]
        extras = [DOC_ROOT / "index.md", DOC_ROOT / "guide.md"]
    elif scope == "source":
        roots = [
            ROOT / "00_Prerequisites",
            ROOT / "01_Hardware_Math_and_Systems",
        ]
        extras = [ROOT / "README.md"]
    else:
        raise ValueError(f"unknown scope: {scope}")

    for base in roots:
        if base.exists():
            yield from sorted(base.glob("*.md"))

    for extra in extras:
        if extra.exists():
            yield extra


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def resolve_candidates(source: Path, raw_link: str) -> list[Path]:
    link = raw_link.strip()
    if not link or link.startswith(("#", "http://", "https://", "mailto:")):
        return []

    link = link.split("#", 1)[0].split("?", 1)[0]
    candidates: list[Path] = []

    if link.startswith("/"):
        target = DOC_ROOT / link.lstrip("/")
        candidates.extend([target, target.with_suffix(".md"), target.with_suffix(".html")])
        return candidates

    target = (source.parent / link).resolve()
    candidates.append(target)
    if target.suffix == "":
        candidates.extend(
            [
                target.with_suffix(".md"),
                target.with_suffix(".ipynb"),
                target.with_suffix(".html"),
            ]
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Chapter 0/1 local links.")
    parser.add_argument(
        "--scope",
        choices=["docs", "source"],
        default="docs",
        help="Which tree to scan: docs site content or source markdown copies.",
    )
    args = parser.parse_args()

    missing: list[tuple[Path, str]] = []

    for path in iter_markdown_files(args.scope):
        text = path.read_text(encoding="utf-8")
        for link in LINK_RE.findall(text):
            candidates = resolve_candidates(path, link)
            if candidates and not any(candidate.exists() for candidate in candidates):
                missing.append((path.relative_to(ROOT), link))

    if missing:
        print(f"Missing links for scope={args.scope}:")
        for path, link in missing:
            print(f"- {path}: {link}")
        print(f"\nTotal missing: {len(missing)}")
        return 1

    print(f"All Chapter 0 / 1 local links resolve to existing files for scope={args.scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
