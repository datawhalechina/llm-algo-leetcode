#!/usr/bin/env python3
"""Sync docs/index.md from README.md.

README.md is the source of truth for the project overview. This script mirrors
it into docs/index.md with only link-path adjustments for the docs site.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS_INDEX = ROOT / "docs/index.md"


def sync_docs_index() -> None:
    text = README.read_text(encoding="utf-8")
    text = text.replace("](./docs/", "](./")
    text = text.replace("](./project_test_scripts.md)", "](../project_test_scripts.md)")
    DOCS_INDEX.write_text(text, encoding="utf-8")
    print(f"Synced {DOCS_INDEX.relative_to(ROOT)} from {README.relative_to(ROOT)}")


def main() -> int:
    sync_docs_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
