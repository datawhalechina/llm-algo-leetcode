#!/usr/bin/env python3
"""Validate Mermaid code blocks with the local Mermaid parser.

The script scans markdown sources for ```mermaid fences and asks Mermaid 11
to parse each block. It reports file paths and line numbers for any block that
fails, which is more precise than waiting for a browser runtime error.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_FILES = [ROOT / "README.md", ROOT / "docs" / "index.md"]
MARKDOWN_FILES.extend(
    sorted(
        path
        for path in (ROOT / "docs").rglob("*.md")
        if "node_modules" not in path.parts
    )
)

MERMAID_FENCE = re.compile(r"^```mermaid\s*$")
FENCE_END = re.compile(r"^```\s*$")


@dataclass
class Block:
    path: Path
    start_line: int
    content: str


def extract_blocks(path: Path) -> list[Block]:
    blocks: list[Block] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False
    block_start = 0
    buffer: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if not in_block and MERMAID_FENCE.match(line):
            in_block = True
            block_start = idx + 1
            buffer = []
            continue

        if in_block and FENCE_END.match(line):
            blocks.append(Block(path=path, start_line=block_start, content="\n".join(buffer)))
            in_block = False
            buffer = []
            continue

        if in_block:
            buffer.append(line)

    return blocks


def parse_with_mermaid(block: Block) -> tuple[bool, str]:
    env = os.environ.copy()
    env["MERMAID_TEXT"] = block.content
    env["MERMAID_FILE"] = str(block.path)
    env["MERMAID_LINE"] = str(block.start_line)

    script = r"""
import DOMPurify from 'dompurify'

DOMPurify.addHook ||= () => {}
DOMPurify.removeHook ||= () => {}
DOMPurify.sanitize ||= (text) => text

const { default: mermaid } = await import('mermaid')

const text = process.env.MERMAID_TEXT ?? ''
mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' })

try {
  await mermaid.parse(text)
  process.stdout.write('ok')
} catch (err) {
  const message = err instanceof Error ? err.message : String(err)
  process.stderr.write(message)
  process.exit(1)
}
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT / "docs",
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, ""
    return False, proc.stderr.strip() or proc.stdout.strip() or "unknown mermaid parse error"


def main() -> int:
    blocks: list[Block] = []
    for path in MARKDOWN_FILES:
        if path.exists():
            blocks.extend(extract_blocks(path))

    failed = 0
    for block in blocks:
        ok, message = parse_with_mermaid(block)
        if not ok:
            failed += 1
            print(f"{block.path}:{block.start_line}: Mermaid parse failed")
            print(message)
            print(block.content)
            print("-" * 80)

    if failed:
        print(f"Found {failed} invalid Mermaid block(s).")
        return 1

    print(f"Checked {len(blocks)} Mermaid block(s): all valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
