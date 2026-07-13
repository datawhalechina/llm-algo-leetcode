#!/usr/bin/env python3
"""Unified verification entrypoint for the tutorial repo.

This script orchestrates the existing part-specific converters and checks
without changing their individual responsibilities.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(cmd: list[str], *, cwd: Path | None = None) -> None:
    display_cwd = cwd.relative_to(ROOT) if cwd else Path(".")
    print(f"[verify] ({display_cwd}) $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_python(script: str, *args: str) -> None:
    run_step([sys.executable, script, *args], cwd=ROOT)


def run_docs_build() -> None:
    run_step(["npm", "run", "docs:build"], cwd=ROOT / "docs")


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_candidates(source: Path, raw_link: str) -> list[Path]:
    link = raw_link.strip()
    if not link or link.startswith(("#", "http://", "https://", "mailto:")):
        return []

    link = link.split("#", 1)[0].split("?", 1)[0]
    candidates: list[Path] = []

    if link.startswith("/"):
        target = ROOT / "docs" / link.lstrip("/")
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


def check_chapter3_intro_links() -> None:
    """Verify source/docs entry-page links for Part 3."""
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    paths = [
        ROOT / "03_Triton_Kernels/intro.md",
        ROOT / "docs/03_Triton_Kernels/intro.md",
    ]

    missing: list[tuple[str, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for link in link_re.findall(text):
            candidates = resolve_candidates(path, link)
            if candidates and not any(candidate.exists() for candidate in candidates):
                missing.append((str(path.relative_to(ROOT)), link))

    if missing:
        print("[verify] Part 3 intro link check failed:")
        for path, link in missing:
            print(f"  - {path}: {link}")
        raise SystemExit(1)

    print("[verify] Part 3 intro links are valid in source and docs.")


def check_chapter4_intro_links() -> None:
    """Verify source/docs entry-page links for Part 4."""
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    paths = [
        ROOT / "04_CUDA_and_System_Optimization/intro.md",
        ROOT / "docs/04_CUDA_and_System_Optimization/intro.md",
    ]

    missing: list[tuple[str, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for link in link_re.findall(text):
            candidates = resolve_candidates(path, link)
            if candidates and not any(candidate.exists() for candidate in candidates):
                missing.append((str(path.relative_to(ROOT)), link))

    if missing:
        print("[verify] Part 4 intro link check failed:")
        for path, link in missing:
            print(f"  - {path}: {link}")
        raise SystemExit(1)

    print("[verify] Part 4 intro links are valid in source and docs.")


def verify_chapter0_1(*, build_docs: bool) -> None:
    # Legacy compatibility path for the Part 0 / Part 1 notebook-first closeout.
    run_python("tools/convert_chapter0_1.py")
    run_python("tools/check_chapter_links.py", "--scope", "source")
    run_python("tools/check_chapter_links.py", "--scope", "docs")
    run_python("tools/test_chapter0_1_notebooks.py")
    if build_docs:
        run_docs_build()


def verify_chapter2(*, build_docs: bool) -> None:
    run_python("tools/convert_notebook.py", "--dir", "02_PyTorch_Algorithms")
    run_python("tools/check_source_docs_mirror.py")
    if has_cuda():
        run_python("tools/test_notebook_answers.py", "--all", "--dir", "02_PyTorch_Algorithms", "--mode", "both")
    else:
        print("[verify] GPU not available, skipping Part 2 notebook answer tests.")
    if build_docs:
        run_docs_build()


def verify_chapter3(*, build_docs: bool) -> None:
    run_python("tools/convert_notebook.py", "--dir", "03_Triton_Kernels")
    run_python("tools/check_source_docs_mirror.py")
    if has_cuda():
        run_python("tools/test_notebook_answers.py", "--all", "--dir", "03_Triton_Kernels", "--mode", "both")
    else:
        print("[verify] GPU not available, skipping Part 3 notebook answer tests.")
    check_chapter3_intro_links()
    if build_docs:
        run_docs_build()


def verify_chapter4(*, build_docs: bool) -> None:
    run_python("tools/convert_notebook.py", "--dir", "04_CUDA_and_System_Optimization")
    run_python("tools/check_source_docs_mirror.py")
    if has_cuda():
        run_python("tools/test_notebook_answers.py", "--all", "--dir", "04_CUDA_and_System_Optimization", "--mode", "both")
    else:
        print("[verify] GPU not available, skipping Part 4 notebook answer tests.")
    check_chapter4_intro_links()
    if build_docs:
        run_docs_build()


def verify_all(*, build_docs: bool) -> None:
    verify_chapter0_1(build_docs=False)
    verify_chapter2(build_docs=False)
    verify_chapter3(build_docs=build_docs)
    verify_chapter4(build_docs=build_docs)
    if build_docs:
        # Part 3 already built docs; keep the all-target behavior explicit.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified verification entrypoint.")

    sub = parser.add_subparsers(dest="target")
    sub.required = False
    sub.add_parser("part0_1", help="Verify Part 0 / Part 1.")
    sub.add_parser("part2", help="Verify Part 2.")
    sub.add_parser("part3", help="Verify Part 3.")
    sub.add_parser("part4", help="Verify Part 4.")
    sub.add_parser("chapter0_1", help="Verify Part 0 / Part 1 (legacy compatibility).")
    sub.add_parser("chapter2", help="Verify Part 2 (legacy compatibility).")
    sub.add_parser("chapter3", help="Verify Part 3 (legacy compatibility).")
    sub.add_parser("chapter4", help="Verify Part 4 (legacy compatibility).")
    sub.add_parser("all", help="Verify all Parts.")

    raw_args = sys.argv[1:]
    build_docs = "--no-build" not in raw_args
    filtered_args = [arg for arg in raw_args if arg != "--no-build"]
    args = parser.parse_args(filtered_args)
    target = args.target or "all"

    if target in {"part0_1", "chapter0_1"}:
        verify_chapter0_1(build_docs=build_docs)
    elif target in {"part2", "chapter2"}:
        verify_chapter2(build_docs=build_docs)
    elif target in {"part3", "chapter3"}:
        verify_chapter3(build_docs=build_docs)
    elif target in {"part4", "chapter4"}:
        verify_chapter4(build_docs=build_docs)
    elif target == "all":
        verify_all(build_docs=build_docs)
    else:
        raise SystemExit(f"unknown target: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
