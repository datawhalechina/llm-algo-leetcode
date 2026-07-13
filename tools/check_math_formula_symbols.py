#!/usr/bin/env python3
"""Check math formula symbol / delimiter issues in markdown sources.

The script scans markdown files, strips fenced code blocks, extracts TeX/LaTeX
math expressions, and asks MathJax to parse them with strict error handling.
It is intentionally conservative: it reports delimiter imbalance, parse errors,
and suspicious control characters that often come from broken escape sequences.

Usage:
    python tools/check_math_formula_symbols.py
    python tools/check_math_formula_symbols.py --dir docs/02_PyTorch_Algorithms
    python tools/check_math_formula_symbols.py --file README.md --file docs/index.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN_ROOTS = [ROOT / "README.md", ROOT / "docs" / "index.md", ROOT / "docs"]

CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
BLOCK_DOLLAR_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
BLOCK_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.S)
INLINE_PAREN_RE = re.compile(r"\\\((.+?)\\\)")
INLINE_DOLLAR_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MATHJAX_ROOT = ROOT / "docs"
MATHJAX_NODE_SCRIPT = r"""
const fs = require('fs')
const { mathjax } = require('./node_modules/mathjax-full/js/mathjax.js')
const { TeX } = require('./node_modules/mathjax-full/js/input/tex.js')
const { SVG } = require('./node_modules/mathjax-full/js/output/svg.js')
const { liteAdaptor } = require('./node_modules/mathjax-full/js/adaptors/liteAdaptor.js')
const { RegisterHTMLHandler } = require('./node_modules/mathjax-full/js/handlers/html.js')
const { AllPackages } = require('./node_modules/mathjax-full/js/input/tex/AllPackages.js')

const payload = JSON.parse(fs.readFileSync(0, 'utf8') || '[]')
const adaptor = liteAdaptor()
RegisterHTMLHandler(adaptor)
const tex = new TeX({
  packages: AllPackages,
  formatError: (_jax, err) => { throw err },
})
const svg = new SVG({ fontCache: 'none' })
const doc = mathjax.document('', { InputJax: tex, OutputJax: svg })

const results = []
for (const item of payload) {
  try {
    if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(item.content)) {
      throw new Error('control character found in formula')
    }
    doc.convert(item.content, { display: item.display })
    results.push({ ok: true })
  } catch (err) {
    results.push({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    })
  }
}
process.stdout.write(JSON.stringify(results))
"""


@dataclass
class Formula:
    path: Path
    line: int
    content: str
    display: bool
    kind: str


def iter_markdown_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".md":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.md") if "node_modules" not in p.parts))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def strip_code_fences(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    return CODE_FENCE_RE.sub(repl, text)


def replace_span(text: str, start: int, end: int) -> str:
    chars = list(text)
    for idx in range(start, end):
        if chars[idx] != "\n":
            chars[idx] = " "
    return "".join(chars)


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def extract_formulas(path: Path) -> list[Formula]:
    raw = path.read_text(encoding="utf-8")
    text = strip_code_fences(raw)
    formulas: list[Formula] = []

    # Display math with $$...$$
    for match in BLOCK_DOLLAR_RE.finditer(text):
        content = match.group(1).strip()
        if content:
            formulas.append(Formula(path=path, line=line_number(text, match.start()), content=content, display=True, kind="$$"))
    masked = text
    for match in BLOCK_DOLLAR_RE.finditer(text):
        masked = replace_span(masked, match.start(), match.end())

    # Display math with \[...\]
    for match in BLOCK_BRACKET_RE.finditer(masked):
        content = match.group(1).strip()
        if content:
            formulas.append(Formula(path=path, line=line_number(masked, match.start()), content=content, display=True, kind="\\[\\]"))
    for match in BLOCK_BRACKET_RE.finditer(masked):
        masked = replace_span(masked, match.start(), match.end())

    # Inline math with \(...\)
    for match in INLINE_PAREN_RE.finditer(masked):
        content = match.group(1).strip()
        if content:
            formulas.append(Formula(path=path, line=line_number(masked, match.start()), content=content, display=False, kind="\\(\\)"))
    for match in INLINE_PAREN_RE.finditer(masked):
        masked = replace_span(masked, match.start(), match.end())

    # Inline math with $...$
    for idx, line in enumerate(masked.splitlines(), start=1):
        pos = 0
        while True:
            start = line.find("$", pos)
            if start == -1:
                break
            if start + 1 < len(line) and line[start + 1] == "$":
                pos = start + 2
                continue
            if start > 0 and line[start - 1] == "\\":
                pos = start + 1
                continue
            end = line.find("$", start + 1)
            if end == -1:
                break
            if end > 0 and line[end - 1] == "\\":
                pos = end + 1
                continue
            content = line[start + 1 : end].strip()
            if content:
                formulas.append(Formula(path=path, line=idx, content=content, display=False, kind="$"))
            pos = end + 1

    return formulas


def validate_formulas(formulas: list[Formula]) -> list[tuple[Formula, str]]:
    if not formulas:
        return []

    payload = [
        {"content": item.content, "display": item.display, "kind": item.kind, "line": item.line}
        for item in formulas
    ]
    proc = subprocess.run(
        ["node", "--input-type=commonjs", "-e", MATHJAX_NODE_SCRIPT],
        cwd=MATHJAX_ROOT,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or "MathJax validation failed unexpectedly")

    results = json.loads(proc.stdout or "[]")
    failures: list[tuple[Formula, str]] = []
    for item, result in zip(formulas, results):
        if not result.get("ok"):
            failures.append((item, result.get("error", "unknown math parse error")))
    return failures


def scan_file(path: Path) -> list[tuple[Formula, str]]:
    raw = path.read_text(encoding="utf-8")
    control_hits = [
        (line_number(raw, match.start()), repr(match.group(0)))
        for match in CONTROL_CHAR_RE.finditer(raw)
    ]
    formulas = extract_formulas(path)
    failures = validate_formulas(formulas)

    if control_hits:
        # Attach control-char issues as pseudo failures tied to the first formula-like line.
        first_line = control_hits[0][0]
        failures.append(
            (
                Formula(path=path, line=first_line, content="control character found in markdown", display=False, kind="control"),
                f"control character(s) found: {', '.join(ch for _, ch in control_hits[:5])}",
            )
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check math formula symbols and delimiters in markdown sources.")
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Specific markdown file to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=None,
        help="Directory to scan recursively. Can be passed multiple times.",
    )
    args = parser.parse_args()

    targets: list[Path] = []
    if args.file:
        targets.extend(Path(p).resolve() for p in args.file)
    if args.dir:
        targets.extend(Path(p).resolve() for p in args.dir)
    if not targets:
        targets = DEFAULT_MARKDOWN_ROOTS

    markdown_files = iter_markdown_files(targets)
    if not markdown_files:
        print("No markdown files found.")
        return 1

    failures: list[tuple[Formula, str]] = []
    checked = 0
    formula_count = 0
    for path in markdown_files:
        checked += 1
        file_formulas = extract_formulas(path)
        formula_count += len(file_formulas)
        failures.extend(scan_file(path))

    if failures:
        for item, message in failures:
            print(f"{item.path.relative_to(ROOT)}:{item.line}: math formula parse failed")
            print(f"  kind: {item.kind}")
            print(f"  content: {item.content}")
            print(f"  error: {message}")
            print("-" * 80)
        print(f"Checked {checked} markdown file(s), {formula_count} formula(s), {len(failures)} issue(s) found.")
        return 1

    print(f"Checked {checked} markdown file(s), {formula_count} formula(s): all valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
