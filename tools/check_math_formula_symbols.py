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
    python tools/check_math_formula_symbols.py --github-compatible --static-only --file lesson.ipynb

GitHub mode checks known static risks only. It makes no network requests and
does not establish that GitHub's final math renderer accepts every expression.
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
    cell: int | None = None


def iter_markdown_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix in {".md", ".ipynb"}:
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*")
                                if p.suffix in {".md", ".ipynb"}
                                and not {"node_modules", ".ipynb_checkpoints"}.intersection(p.parts)))
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


def extract_formulas(path: Path, raw: str | None = None) -> list[Formula]:
    if raw is None:
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


def markdown_sources(path: Path) -> list[tuple[int | None, str]]:
    """Read prose only; notebook cell numbers are one-based."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        notebook = json.loads(raw)
        return [(index + 1, "".join(cell.get("source", [])))
                for index, cell in enumerate(notebook.get("cells", []))
                if cell.get("cell_type") == "markdown"]
    return [(None, raw)]


def mask_code(text: str) -> str:
    """Mask fenced/inline code while retaining offsets and line numbers."""
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line.rstrip("\n"))
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
            lines.append(re.sub(r"[^\n]", " ", line))
        elif marker:
            fence = (marker[1][0], len(marker[1]))
            lines.append(re.sub(r"[^\n]", " ", line))
        else:
            lines.append(line)
    masked = "".join(lines)
    return re.sub(r"(`+)(?!`)(.*?)(?<!`)\1(?!`)",
                  lambda match: re.sub(r"[^\n]", " ", match[0]), masked, flags=re.S)


def github_compatibility_issues(path: Path, raw: str, cell: int | None = None) -> list[tuple[Formula, str]]:
    """Flag observed compatibility risks, not proof of final GitHub rendering."""
    text = mask_code(raw)
    pattern = re.compile(r"(?<!\\)\$\$(.*?)\$\$|(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)", re.S)
    issues = []
    for match in pattern.finditer(text):
        display = match[1] is not None
        content = match[1] if display else match[2]
        item = Formula(path, line_number(text, match.start()), content.strip(), display, "github-risk", cell)
        if not display and match.start() and not text[match.start() - 1].isspace():
            issues.append((item, "GitHub边界风险：行内公式起始符前补空格（尤其中文标点后）"))
        if re.search(r"\\operatorname\b", content):
            issues.append((item, "GitHub宏兼容风险：operatorname已有禁止宏报错；按含义选择兼容写法"))
        if re.search(r"(?<!\\)\\[,!;:%#{}]", content):
            issues.append((item, "GitHub转义风险：排版间距、百分号、井号或花括号可能先被Markdown处理"))
        if re.search(r"\^\*", content):
            issues.append((item, r"GitHub强调标记风险：裸星号上标改用^{\ast}"))
        if re.search(r"\\(?:bar|hat)\{[^}]+\}_", content):
            issues.append((item, "GitHub下标边界风险：重音命令花括号后紧接下划线，需实际渲染复核"))
    for match in re.finditer(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", text, re.S):
        item = Formula(path, line_number(text, match.start()), match[0], match[2] is not None, "github-risk", cell)
        issues.append((item, r"双端分隔符风险：当前站点使用$...$或$$...$$，不要直接使用\(...\)或\[...\]"))
    return issues


def scan_source(path: Path, raw: str, *, github_compatible: bool = False,
                static_only: bool = False, cell: int | None = None) -> list[tuple[Formula, str]]:
    """Check one Markdown document or notebook Markdown cell."""
    control_hits = [
        (line_number(raw, match.start()), repr(match.group(0)))
        for match in CONTROL_CHAR_RE.finditer(raw)
    ]
    formulas = extract_formulas(path, mask_code(raw))
    for formula in formulas:
        formula.cell = cell
    failures = [] if static_only else validate_formulas(formulas)
    if github_compatible:
        failures.extend(github_compatibility_issues(path, raw, cell))

    if control_hits:
        # Attach control-char issues as pseudo failures tied to the first formula-like line.
        first_line = control_hits[0][0]
        failures.append(
            (
                Formula(path=path, line=first_line, content="control character found in markdown", display=False, kind="control", cell=cell),
                f"control character(s) found: {', '.join(ch for _, ch in control_hits[:5])}",
            )
        )
    return failures


def scan_file(path: Path, *, github_compatible: bool = False,
              static_only: bool = False) -> list[tuple[Formula, str]]:
    """Check Markdown or prose cells without executing notebooks or using network."""
    return [issue for cell, raw in markdown_sources(path)
            for issue in scan_source(path, raw, github_compatible=github_compatible,
                                     static_only=static_only, cell=cell)]


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
    parser.add_argument("--github-compatible", action="store_true",
                        help="Also report known GitHub compatibility risks; no network calls.")
    parser.add_argument("--static-only", action="store_true",
                        help="With --github-compatible, skip local MathJax; only Python is required.")
    args = parser.parse_args()
    if args.static_only and not args.github_compatible:
        parser.error("--static-only requires --github-compatible")

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
        formula_count += sum(len(extract_formulas(path, mask_code(raw)))
                             for _, raw in markdown_sources(path))
        failures.extend(scan_file(path, github_compatible=args.github_compatible,
                                  static_only=args.static_only))

    if failures:
        for item, message in failures:
            location = os.path.relpath(item.path, ROOT)
            cell = f":cell {item.cell}" if item.cell is not None else ""
            print(f"{location}{cell}:{item.line}: math formula issue")
            print(f"  kind: {item.kind}")
            print(f"  content: {item.content}")
            print(f"  error: {message}")
            print("-" * 80)
        print(f"Checked {checked} markdown file(s), {formula_count} formula(s), {len(failures)} issue(s) found.")
        return 1

    detail = ("no known static compatibility risks; final GitHub display not verified"
              if args.github_compatible else "all valid")
    print(f"Checked {checked} file(s), {formula_count} formula(s): {detail}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
