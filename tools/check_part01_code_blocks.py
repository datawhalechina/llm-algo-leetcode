#!/usr/bin/env python3
"""Check code blocks in Part 0 / Part 1 markdown pages.

The script validates fenced code blocks in the Part 0 and Part 1 docs by:
1. checking Python syntax for runnable fences; and
2. optionally executing CPU-friendly pages sequentially to catch runtime errors.

GPU-heavy pages are syntax-checked by default and only executed when a CUDA
device is available.

Usage:
    python tools/check_part01_code_blocks.py
    python tools/check_part01_code_blocks.py --dir docs/00_Prerequisites
    python tools/check_part01_code_blocks.py --syntax-only
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    ROOT / "docs" / "00_Prerequisites",
    ROOT / "docs" / "01_Hardware_Math_and_Systems",
]
RUNNABLE_LANGS = {"", "python", "py"}
SKIPPED_LANGS = {"bash", "sh", "shell", "text", "mermaid", "json", "yaml", "yml", "html", "diff"}


@dataclass
class CodeBlock:
    path: Path
    start_line: int
    lang: str
    source: str


def iter_markdown_files(base_dirs: list[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for base in base_dirs:
        if base.is_file() and base.suffix == ".md":
            files.append(base)
        elif base.exists():
            files.extend(sorted(p for p in base.rglob(pattern) if p.suffix == ".md" and "node_modules" not in p.parts))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def parse_env_kind(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"\*\*环境：\*\*\s*([^|]+)", text)
    if not m:
        return "unknown"
    return m.group(1).strip().lower()


def extract_code_blocks(path: Path) -> list[CodeBlock]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[CodeBlock] = []
    in_block = False
    lang = ""
    start_line = 0
    buffer: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if not in_block:
            if line.startswith("```"):
                fence = line[3:].strip().split(maxsplit=1)
                lang = fence[0].lower() if fence and fence[0] else ""
                in_block = True
                start_line = idx + 1
                buffer = []
        else:
            if line.startswith("```"):
                blocks.append(CodeBlock(path=path, start_line=start_line, lang=lang, source="\n".join(buffer)))
                in_block = False
                lang = ""
                buffer = []
            else:
                buffer.append(line)

    return blocks


def is_runnable(block: CodeBlock) -> bool:
    return block.lang in RUNNABLE_LANGS


def syntax_check(block: CodeBlock) -> tuple[bool, str]:
    try:
        ast.parse(block.source, filename=f"{block.path}:{block.start_line}")
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno}, col {exc.offset})"


def has_gpu() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def runtime_check(path: Path, blocks: list[CodeBlock], timeout: int) -> tuple[bool, str]:
    runnable_blocks = [b for b in blocks if is_runnable(b) and b.source.strip()]
    if not runnable_blocks:
        return True, ""

    script = [
        "import traceback",
        "ns = {'__name__': '__main__'}",
        f"blocks = {json.dumps([{'line': b.start_line, 'source': b.source} for b in runnable_blocks], ensure_ascii=False)}",
        "for block in blocks:",
        "    try:",
        "        exec(compile(block['source'], f\"{path}:{block['line']}\", 'exec'), ns, ns)",
        "    except Exception as exc:",
        "        print(f\"FAILED at {path}:{block['line']}: {exc.__class__.__name__}: {exc}\")",
        "        traceback.print_exc()",
        "        raise",
    ]
    code = "\n".join(script).replace("{path}", str(path.relative_to(ROOT)))

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode == 0:
        return True, ""
    return False, proc.stderr.strip() or proc.stdout.strip() or "runtime execution failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Part 0 / Part 1 code blocks.")
    parser.add_argument("--dir", action="append", default=None, help="Directory to scan. Can be passed multiple times.")
    parser.add_argument("--pattern", default="*.md", help="Glob pattern to match markdown files.")
    parser.add_argument("--syntax-only", action="store_true", help="Only check Python syntax.")
    parser.add_argument("--runtime-only", action="store_true", help="Only execute runnable blocks.")
    parser.add_argument("--timeout", type=int, default=120, help="Runtime execution timeout per file (seconds).")
    args = parser.parse_args()

    base_dirs = [Path(d).resolve() for d in args.dir] if args.dir else DEFAULT_DIRS
    markdown_files = iter_markdown_files(base_dirs, args.pattern)
    if not markdown_files:
        print("No markdown files found.")
        return 1

    run_runtime = not args.syntax_only
    run_syntax = not args.runtime_only
    gpu_available = has_gpu()

    passed = 0
    failed = 0
    skipped = 0

    for path in markdown_files:
        env_kind = parse_env_kind(path)
        blocks = extract_code_blocks(path)
        syntax_failures = []
        runtime_failure = None

        if run_syntax:
            for block in blocks:
                if is_runnable(block) and block.source.strip():
                    ok, message = syntax_check(block)
                    if not ok:
                        syntax_failures.append((block, message))

        if run_runtime:
            should_skip_runtime = "gpu" in env_kind and "cpu" not in env_kind and not gpu_available
            if should_skip_runtime:
                skipped += 1
                print(f"SKIP {path.relative_to(ROOT)} (environment={env_kind}, no GPU available)")
            else:
                ok, message = runtime_check(path, blocks, args.timeout)
                if not ok:
                    runtime_failure = message

        if syntax_failures or runtime_failure:
            failed += 1
            print(f"\n{'=' * 72}")
            print(f"FAILED {path.relative_to(ROOT)}")
            print(f"{'=' * 72}")
            for block, message in syntax_failures:
                print(f"{block.path.relative_to(ROOT)}:{block.start_line}: syntax error")
                print(f"  lang: {block.lang or 'plain'}")
                print(f"  error: {message}")
                print(block.source)
                print("-" * 80)
            if runtime_failure:
                print(f"{path.relative_to(ROOT)}: runtime error")
                print(runtime_failure)
            continue

        passed += 1

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped runtime: {skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
