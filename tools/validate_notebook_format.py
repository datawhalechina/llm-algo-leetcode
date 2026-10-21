#!/usr/bin/env python3
"""Audit source notebooks without modifying them.

The checker separates Jupyter schema problems from tutorial-specific naming
and metadata conventions. By default it scans the source notebooks in
Part 00–04 and prints a Markdown report. It exits successfully while the
repository is being migrated; pass ``--fail-on-issues`` to use it as a gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat
from nbformat import ValidationError
from nbformat.warnings import MissingIDFieldWarning

from check_mechanism_step4_tables import (
    audit as audit_mechanism_step4,
    registered_project_paths,
)


DEFAULT_ROOTS = [
    Path("00_Prerequisites"),
    Path("01_Hardware_Math_and_Systems"),
    Path("02_PyTorch_Algorithms"),
    Path("03_Triton_Kernels"),
    Path("04_CUDA_and_System_Optimization"),
]

FILENAME_NUMBER_RE = re.compile(r"^(\d{2})_")
TITLE_NUMBER_RE = re.compile(r"^#\s+(\d{1,2})\.")
COLAB_NOTEBOOK_RE = re.compile(r"github/[^)\s]+/(\d{2}_[^)\s]+\.ipynb)")
REGISTERED_PROJECT_PATHS = registered_project_paths()


def _is_part02_mechanism(path: Path) -> bool:
    """Return whether a source notebook uses the Part 02 mechanism template."""
    return (
        path.parent.name == "02_PyTorch_Algorithms"
        and path.resolve() not in REGISTERED_PROJECT_PATHS
        and "Reserved" not in path.name
    )


@dataclass(frozen=True)
class Issue:
    path: str
    severity: str
    code: str
    message: str
    cells: tuple[int, ...] = ()


def iter_notebooks(paths: list[Path]) -> list[Path]:
    """Return unique source notebooks in stable path order."""
    notebooks: dict[Path, Path] = {}
    for path in paths:
        if path.is_file() and path.suffix == ".ipynb":
            notebooks[path.resolve()] = path
        elif path.is_dir():
            for notebook in path.rglob("*.ipynb"):
                if any(part.startswith(".") for part in notebook.parts):
                    continue
                notebooks[notebook.resolve()] = notebook
    return sorted(notebooks.values(), key=lambda item: item.as_posix())


def _cell_numbers(indices: list[int]) -> tuple[int, ...]:
    """Convert zero-based indices to reader-facing cell numbers."""
    return tuple(index + 1 for index in indices)


def _first_markdown_source(cells: list[dict[str, Any]]) -> str:
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "".join(source) if isinstance(source, list) else str(source)
    return ""


def audit_notebook(path: Path) -> list[Issue]:
    """Return format and tutorial-contract issues for one notebook."""
    relative_path = path.as_posix()
    issues: list[Issue] = []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [Issue(relative_path, "P0", "invalid_json", str(exc))]

    if not isinstance(raw, dict):
        return [Issue(relative_path, "P0", "invalid_notebook_root", "Notebook 根对象不是 JSON object")]

    cells = raw.get("cells")
    if not isinstance(cells, list):
        return [Issue(relative_path, "P0", "missing_cells", "缺少合法的 cells 列表")]

    major = raw.get("nbformat")
    minor = raw.get("nbformat_minor")
    if major != 4:
        issues.append(Issue(relative_path, "P0", "unsupported_nbformat", f"nbformat={major!r}，应为 4"))
    if not isinstance(minor, int) or minor < 5:
        issues.append(
            Issue(
                relative_path,
                "P1",
                "old_nbformat_minor",
                f"nbformat_minor={minor!r}；统一格式要求至少为 5",
            )
        )

    missing_ids: list[int] = []
    empty_ids: list[int] = []
    markdown_code_fields: list[int] = []
    raw_code_fields: list[int] = []
    code_missing_execution_count: list[int] = []
    code_missing_outputs: list[int] = []
    invalid_tags: list[int] = []
    ids: defaultdict[str, list[int]] = defaultdict(list)

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            issues.append(
                Issue(relative_path, "P0", "invalid_cell", "Cell 不是 JSON object", _cell_numbers([index]))
            )
            continue

        cell_id = cell.get("id")
        if "id" not in cell:
            missing_ids.append(index)
        elif not isinstance(cell_id, str) or not cell_id.strip():
            empty_ids.append(index)
        else:
            ids[cell_id].append(index)

        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            if "execution_count" in cell or "outputs" in cell:
                markdown_code_fields.append(index)
        elif cell_type == "raw":
            if "execution_count" in cell or "outputs" in cell:
                raw_code_fields.append(index)
        elif cell_type == "code":
            if "execution_count" not in cell:
                code_missing_execution_count.append(index)
            if "outputs" not in cell:
                code_missing_outputs.append(index)
        else:
            issues.append(
                Issue(
                    relative_path,
                    "P0",
                    "invalid_cell_type",
                    f"未知 cell_type={cell_type!r}",
                    _cell_numbers([index]),
                )
            )

        metadata = cell.get("metadata", {})
        if isinstance(metadata, dict) and "tags" in metadata:
            tags = metadata["tags"]
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                invalid_tags.append(index)

    if missing_ids:
        issues.append(
            Issue(relative_path, "P1", "missing_cell_id", "Cell 缺少 id", _cell_numbers(missing_ids))
        )
    if empty_ids:
        issues.append(Issue(relative_path, "P1", "empty_cell_id", "Cell id 为空", _cell_numbers(empty_ids)))

    duplicate_id_cells = sorted(
        index for indices in ids.values() if len(indices) > 1 for index in indices
    )
    if duplicate_id_cells:
        duplicate_values = sorted(cell_id for cell_id, indices in ids.items() if len(indices) > 1)
        issues.append(
            Issue(
                relative_path,
                "P0",
                "duplicate_cell_id",
                f"重复 id：{', '.join(duplicate_values)}",
                _cell_numbers(duplicate_id_cells),
            )
        )

    for indices, code, message in (
        (markdown_code_fields, "markdown_has_code_fields", "Markdown cell 带有 execution_count 或 outputs"),
        (raw_code_fields, "raw_has_code_fields", "Raw cell 带有 execution_count 或 outputs"),
        (
            code_missing_execution_count,
            "code_missing_execution_count",
            "Code cell 缺少 execution_count",
        ),
        (code_missing_outputs, "code_missing_outputs", "Code cell 缺少 outputs"),
        (invalid_tags, "invalid_tags", "metadata.tags 必须是字符串列表"),
    ):
        if indices:
            issues.append(Issue(relative_path, "P1", code, message, _cell_numbers(indices)))

    filename_match = FILENAME_NUMBER_RE.match(path.name)
    first_markdown = _first_markdown_source(cells)
    title_match = TITLE_NUMBER_RE.match(first_markdown.lstrip())
    if filename_match and title_match:
        filename_number = filename_match.group(1)
        title_number = title_match.group(1).zfill(2)
        if filename_number != title_number:
            issues.append(
                Issue(
                    relative_path,
                    "P1",
                    "filename_title_number_mismatch",
                    f"文件编号 {filename_number} 与 H1 编号 {title_number} 不一致",
                    (1,),
                )
            )

    colab_match = COLAB_NOTEBOOK_RE.search(first_markdown)
    if colab_match and colab_match.group(1) != path.name:
        issues.append(
            Issue(
                relative_path,
                "P1",
                "colab_filename_mismatch",
                f"Colab 指向 {colab_match.group(1)}，当前文件为 {path.name}",
                (1,),
            )
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingIDFieldWarning)
            nbformat.validate(nbformat.from_dict(raw))
    except ValidationError as exc:
        issues.append(
            Issue(
                relative_path,
                "P0",
                "schema_validation_error",
                str(exc).splitlines()[0],
            )
        )
    except Exception as exc:  # pragma: no cover - protects full-repository audits
        issues.append(Issue(relative_path, "P0", "schema_validation_crash", repr(exc)))

    if _is_part02_mechanism(path):
        step4_issues = audit_mechanism_step4(path)
        if step4_issues:
            issues.append(
                Issue(
                    relative_path,
                    "P1",
                    "mechanism_step4_contract",
                    "；".join(step4_issues),
                )
            )

        def tagged_indices(tag: str) -> list[int]:
            return [
                index
                for index, cell in enumerate(cells)
                if cell.get("cell_type") == "code"
                and isinstance(cell.get("metadata"), dict)
                and tag in cell["metadata"].get("tags", [])
            ]

        test_heading_indices = [
            index
            for index, cell in enumerate(cells)
            if cell.get("cell_type") == "markdown"
            and any(
                line.strip() == "## 测试"
                for line in "".join(cell.get("source", [])).splitlines()
            )
        ]
        imports_indices = tagged_indices("tutorial-imports")
        question_indices = tagged_indices("tutorial-question")
        test_indices = tagged_indices("tutorial-test")
        if all(
            len(indices) == 1
            for indices in (
                imports_indices,
                question_indices,
                test_heading_indices,
                test_indices,
            )
        ):
            ordered_indices = [
                imports_indices[0],
                question_indices[0],
                test_heading_indices[0],
                test_indices[0],
            ]
            if ordered_indices != sorted(ordered_indices):
                issues.append(
                    Issue(
                        relative_path,
                        "P1",
                        "mechanism_test_section_order",
                        "机制练习应按 imports → question → ## 测试 → test 排列",
                        _cell_numbers(ordered_indices),
                    )
                )

    return issues


def _part_name(path: str) -> str:
    return path.split("/", 1)[0]


def render_markdown(notebooks: list[Path], issues: list[Issue]) -> str:
    affected = {issue.path for issue in issues}
    issue_counts = Counter(issue.code for issue in issues)
    severity_counts = Counter(issue.severity for issue in issues)
    notebooks_by_part = Counter(_part_name(path.as_posix()) for path in notebooks)
    affected_by_part = Counter(_part_name(path) for path in affected)

    lines = [
        "# Notebook 格式审计问题清单",
        "",
        "> 由 `python tools/validate_notebook_format.py` 生成；本轮仅扫描，不自动修改 Notebook。",
        "",
        "## 总览",
        "",
        f"- 扫描 Notebook：{len(notebooks)}",
        f"- 无格式问题：{len(notebooks) - len(affected)}",
        f"- 存在问题：{len(affected)}",
        f"- 问题记录：{len(issues)}（P0：{severity_counts['P0']}；P1：{severity_counts['P1']}）",
        "",
        "| Part | 扫描数 | 问题文件数 |",
        "| --- | ---: | ---: |",
    ]
    for part in sorted(notebooks_by_part):
        lines.append(f"| `{part}` | {notebooks_by_part[part]} | {affected_by_part[part]} |")

    lines.extend(
        [
            "",
            "## 问题类型",
            "",
            "| 规则 | 记录数 |",
            "| --- | ---: |",
        ]
    )
    for code, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{code}` | {count} |")

    lines.extend(
        [
            "",
            "## 修复优先级",
            "",
            "- P0：JSON、cell ID 或 Jupyter schema 无效；修复后才能作为强制门禁。",
            "- P1：格式版本、cell 字段、tags、编号或 Colab 路径不一致；按 Part 分批规范化。",
            "- 修复时先处理 P0，再处理 P1；每批完成后重新运行本脚本和对应 Notebook 测试。",
            "",
            "## 逐文件问题",
            "",
        ]
    )

    issues_by_path: defaultdict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        issues_by_path[issue.path].append(issue)

    if not issues_by_path:
        lines.append("当前扫描范围内没有发现格式问题。")
    else:
        for path in sorted(issues_by_path):
            lines.append(f"### `{path}`")
            lines.append("")
            for issue in sorted(issues_by_path[path], key=lambda item: (item.severity, item.code)):
                cells = f"；cell {', '.join(map(str, issue.cells))}" if issue.cells else ""
                lines.append(f"- **{issue.severity} · `{issue.code}`**：{issue.message}{cells}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_text(notebooks: list[Path], issues: list[Issue]) -> str:
    affected = {issue.path for issue in issues}
    lines = [f"Scanned {len(notebooks)} notebooks; {len(affected)} files have issues."]
    for issue in issues:
        cells = f" cells={','.join(map(str, issue.cells))}" if issue.cells else ""
        lines.append(f"[{issue.severity}] {issue.path}: {issue.code}: {issue.message}{cells}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit notebook schema and tutorial metadata without edits.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Notebook files or directories. Defaults to source notebooks in Part 00–04.",
    )
    parser.add_argument("--format", choices=("markdown", "text", "json"), default="markdown")
    parser.add_argument("--fail-on-issues", action="store_true", help="Return exit code 1 when issues exist.")
    args = parser.parse_args()

    input_paths = [Path(value) for value in args.paths] if args.paths else DEFAULT_ROOTS
    notebooks = iter_notebooks(input_paths)
    issues = [issue for notebook in notebooks for issue in audit_notebook(notebook)]

    if args.format == "json":
        output = json.dumps(
            {
                "notebook_count": len(notebooks),
                "affected_count": len({issue.path for issue in issues}),
                "issues": [asdict(issue) for issue in issues],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    elif args.format == "text":
        output = render_text(notebooks, issues)
    else:
        output = render_markdown(notebooks, issues)

    sys.stdout.write(output)
    return 1 if args.fail_on_issues and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
