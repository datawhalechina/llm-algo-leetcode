#!/usr/bin/env python3
"""Unified verification entrypoint for the tutorial repo.

This script orchestrates the existing part-specific converters and checks
without changing their individual responsibilities.
"""

from __future__ import annotations

import argparse
import json
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
    """Check the Part 3 entry, group navigation and chapter numbering."""
    check_kernel_part_navigation("03_Triton_Kernels", "3")


def check_chapter4_intro_links() -> None:
    """Check the Part 4 entry, group navigation and chapter numbering."""
    check_kernel_part_navigation("04_CUDA_and_System_Optimization", "4")


def check_kernel_part_navigation(part_dir: str, group_prefix: str) -> None:
    """Check source/mirror navigation, lesson H1 numbers and sidebar coverage."""
    link_re = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
    problems: list[str] = []
    source = ROOT / part_dir
    chapters = [
        path for path in source.iterdir()
        if re.match(r"^\d{2}_", path.name) and path.suffix in {".md", ".ipynb"}
    ]
    numbers = sorted(int(path.name[:2]) for path in chapters)
    if numbers != list(range(1, len(chapters) + 1)):
        problems.append(f"{part_dir}: lesson numbers must be unique and consecutive")
    sidebar = (ROOT / "docs/.vitepress/config.mts").read_text(encoding="utf-8")
    sidebar_labels = dict(
        (stem, label) for label, stem in re.findall(
            r"\{ text: '([^']+)', link: '/" + re.escape(part_dir) + r"/([^']+)' \}", sidebar
        )
    )
    for path in chapters:
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            text = next(
                "".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
        else:
            text = path.read_text(encoding="utf-8")
        if not text.startswith(f"# {path.name[:2]}. ") or " | " not in text.splitlines()[0]:
            problems.append(f"{path.relative_to(ROOT)}: H1 number or bilingual title mismatch")
        if f"/{part_dir}/{path.stem}'" not in sidebar:
            problems.append(f"{path.relative_to(ROOT)}: missing sidebar entry")
        elif not sidebar_labels.get(path.stem, "").startswith(f"{path.name[:2]}. "):
            problems.append(f"{path.relative_to(ROOT)}: sidebar number mismatch")
    for base in (source, ROOT / "docs" / part_dir):
        pages = [base / "intro.md", *(base / f"{group_prefix}_{n}.md" for n in range(1, 6))]
        group_targets: set[str] = set()
        for page in pages:
            if not page.exists():
                problems.append(f"{page.relative_to(ROOT)}: missing navigation page")
                continue
            text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
            for label, link in link_re.findall(text):
                if not label.strip():
                    problems.append(f"{page.relative_to(ROOT)}: empty link label")
                candidates = resolve_candidates(page, link)
                if base == source:
                    candidates.extend(p.with_suffix(".ipynb") for p in list(candidates) if p.suffix == ".md")
                if candidates and not any(p.exists() for p in candidates):
                    problems.append(f"{page.relative_to(ROOT)}: missing target {link}")
                target = Path(link.split("#", 1)[0])
                if page.name != "intro.md" and target.parent == Path("."):
                    group_targets.add(target.stem)
                    label_number = re.match(r"^(\d{2})(?=[.\s]|$)", label)
                    target_number = re.match(r"^(\d{2})_", target.stem)
                    if label_number and target_number and label_number[1] != target_number[1]:
                        problems.append(f"{page.relative_to(ROOT)}: link number mismatch {label} -> {link}")
        missing = {p.stem for p in chapters} - group_targets
        if missing:
            problems.append(f"{base.relative_to(ROOT)}: lessons absent from group pages: {sorted(missing)}")
    if problems:
        print("[verify] Navigation check failed:\n" + "\n".join(f"  - {p}" for p in problems))
        raise SystemExit(1)
    print(f"[verify] Part {group_prefix}: numbering, H1, sidebar and all five source/docs groups passed.")


def verify_chapter0_1(*, build_docs: bool) -> None:
    # Canonical Part 0 / Part 1 verification entrypoint:
    # - convert source notebooks to docs mirrors
    # - check source/docs chapter links
    # - run sequential code-cell execution validation
    run_python("tools/check_chapter_links.py", "--scope", "source")
    run_python("tools/convert_chapter0_1.py")
    # Foundation pages link directly to kernel/runtime lessons. Generate their
    # mirrors even in a clean checkout; conversion does not execute GPU code.
    run_python("tools/convert_notebook.py", "--dir", "03_Triton_Kernels", "--dir", "04_CUDA_and_System_Optimization")
    run_python("tools/check_chapter_links.py", "--scope", "docs")
    run_python("tools/test_chapter0_1_notebooks.py")
    if build_docs:
        run_docs_build()


def audit_chapter0_1() -> None:
    # Supplemental audit entrypoint for closeout work:
    # - codecell_run: per-notebook code execution audit
    # - structure_only: section/link/cell-id audit
    run_python("tools/audit_chapter0_1_notebooks.py", "--profile", "all")


def verify_chapter2(*, build_docs: bool) -> None:
    run_python("tools/check_svg_layout.py")
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
        run_python("tools/test_notebook_answers.py", "--all", "--dir", "03_Triton_Kernels", "--mode", "both", "--strict-question-failure", "--timeout", "180")
    else:
        print("[verify] GPU not available, skipping Part 3 notebook answer tests.")
    check_chapter3_intro_links()
    if build_docs:
        run_docs_build()


def verify_chapter4(*, build_docs: bool) -> None:
    run_python("tools/convert_notebook.py", "--dir", "04_CUDA_and_System_Optimization")
    run_python("tools/check_source_docs_mirror.py")
    if has_cuda():
        run_python("tools/test_notebook_answers.py", "--all", "--dir", "04_CUDA_and_System_Optimization", "--mode", "both", "--strict-question-failure", "--timeout", "180")
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
    sub.add_parser("part0_1", help="Canonical verify entry for Part 0 / Part 1.")
    sub.add_parser("part0_1_audit", help="Supplemental audit for Part 0 / Part 1: codecell_run + structure_only.")
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
    elif target == "part0_1_audit":
        audit_chapter0_1()
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
