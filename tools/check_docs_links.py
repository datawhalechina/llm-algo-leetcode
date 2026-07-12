#!/usr/bin/env python3
"""Run the docs verification workflow and scan for broken local markdown links.

The default workflow is:
1. Convert notebook sources into docs/
2. Check source/docs mirror consistency
3. Build the VitePress site
4. Scan rendered markdown files for broken local links

This script is intentionally conservative:
- It ignores fenced code blocks while scanning markdown links.
- It skips external URLs and anchors.
- It treats unresolved relative links as failures.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FENCE_RE = re.compile(r"```.*?```", re.S)
LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd or ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def convert_notebooks(part_dir: str) -> None:
    run_cmd([sys.executable, str(ROOT / "convert_notebook.py"), "--dir", part_dir])


def check_mirror() -> None:
    run_cmd([sys.executable, str(ROOT / "check_source_docs_mirror.py")])


def build_docs() -> None:
    run_cmd(["npm", "run", "docs:build"], cwd=DOCS)


def scan_markdown_links() -> int:
    missing: list[tuple[str, str]] = []
    markdown_files = [
        path
        for path in DOCS.rglob("*.md")
        if "node_modules" not in path.parts
    ]

    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        text = FENCE_RE.sub("", text)
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "tel:", "javascript:", "#")):
                continue
            target = target.split("#", 1)[0].strip()
            if not target or target == "...":
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append((str(path.relative_to(ROOT)), target))

    if missing:
        print(f"missing_count {len(missing)}")
        for source, target in missing:
            print(f"{source} -> {target}")
        return 1

    print("missing_count 0")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run docs build and broken-link checks.")
    parser.add_argument(
        "--part-dir",
        default="02_PyTorch_Algorithms",
        help="Notebook part directory to convert into docs/",
    )
    parser.add_argument("--skip-convert", action="store_true", help="Skip notebook conversion.")
    parser.add_argument("--skip-mirror-check", action="store_true", help="Skip source/docs mirror check.")
    parser.add_argument("--skip-build", action="store_true", help="Skip VitePress docs build.")
    parser.add_argument("--skip-scan", action="store_true", help="Skip markdown link scan.")
    args = parser.parse_args()

    if not args.skip_convert:
        convert_notebooks(args.part_dir)
    if not args.skip_mirror_check:
        check_mirror()
    if not args.skip_build:
        build_docs()
    if not args.skip_scan:
        return scan_markdown_links()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
