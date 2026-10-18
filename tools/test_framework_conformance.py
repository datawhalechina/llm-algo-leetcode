#!/usr/bin/env python3
"""Execute optional framework and interface checks from mechanism notebooks."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

from check_mechanism_step4_tables import discover_part02_mechanisms


REQUIRED_CODE_TAGS = (
    "tutorial-imports",
    "tutorial-answer",
    "tutorial-framework-check",
)
FRAMEWORK_TAG = "tutorial-framework-check"
SKIP_TAG = "skip_answer_test"
RESULT_MARKER = "__FRAMEWORK_CONFORMANCE_RESULT__="
DEFAULT_CHECK_KIND = "framework_conformance"
FRAMEWORK_HEADING = "## 框架一致性核对（可选）"
VALID_CHECK_KINDS = {
    "framework_conformance",
    "data_interface",
    "downstream_contract",
}


def tagged_cells(notebook: nbformat.NotebookNode, tag: str) -> list[nbformat.NotebookNode]:
    return [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and tag in cell.metadata.get("tags", [])
    ]


def structure_issues(notebook: nbformat.NotebookNode) -> list[str]:
    issues: list[str] = []
    for tag in REQUIRED_CODE_TAGS:
        count = len(tagged_cells(notebook, tag))
        if count != 1:
            issues.append(f"{tag} 应恰好标注 1 个代码单元，当前为 {count} 个")

    framework_cells = tagged_cells(notebook, FRAMEWORK_TAG)
    if framework_cells and SKIP_TAG not in framework_cells[0].metadata.get("tags", []):
        issues.append(f"{FRAMEWORK_TAG} 单元必须同时标注 {SKIP_TAG}")
    if framework_cells:
        profile = framework_cells[0].metadata.get("framework_profile")
        if not isinstance(profile, str) or not profile.strip():
            issues.append(f"{FRAMEWORK_TAG} 单元必须登记非空 framework_profile")
        check_kind = framework_cells[0].metadata.get("check_kind", DEFAULT_CHECK_KIND)
        if check_kind not in VALID_CHECK_KINDS:
            issues.append(
                f"{FRAMEWORK_TAG} 单元的 check_kind 必须属于 {sorted(VALID_CHECK_KINDS)}"
            )
    else:
        check_kind = DEFAULT_CHECK_KIND

    if framework_cells:
        framework_index = notebook.cells.index(framework_cells[0])
        if framework_index == 0 or notebook.cells[framework_index - 1].cell_type != "markdown":
            issues.append("框架核对代码前必须紧邻对应 Markdown 层级")
        else:
            framework_markdown = str(notebook.cells[framework_index - 1].source)
            if '<a id="framework-check"></a>' not in framework_markdown:
                issues.append('框架核对层级缺少稳定锚点 <a id="framework-check"></a>')
            if FRAMEWORK_HEADING not in framework_markdown:
                issues.append(f"框架核对层级应使用固定二级标题：{FRAMEWORK_HEADING}")
            nonempty_lines = [
                line.strip() for line in framework_markdown.splitlines() if line.strip()
            ]
            if not nonempty_lines or not nonempty_lines[-1].startswith("|"):
                issues.append("框架核对层级必须以核对表结束，表格后不能追加文字")

        try:
            framework_tree = ast.parse(str(framework_cells[0].source))
        except SyntaxError as exc:
            issues.append(f"框架核对代码无法解析：{exc}")
        else:
            check_functions = [
                node
                for node in ast.walk(framework_tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("check_")
            ]
            if not check_functions:
                issues.append("框架核对代码必须定义 check_ 开头的主核对函数")
            elif not any(ast.get_docstring(node) for node in check_functions):
                issues.append("框架核对主函数必须使用一句 docstring 说明比较契约")
    return issues


def notebook_profile(notebook: nbformat.NotebookNode) -> str | None:
    cells = tagged_cells(notebook, FRAMEWORK_TAG)
    if len(cells) != 1:
        return None
    profile = cells[0].metadata.get("framework_profile")
    return profile.strip() if isinstance(profile, str) and profile.strip() else None


def discover_framework_notebooks() -> list[Path]:
    discovered: list[Path] = []
    for path in discover_part02_mechanisms():
        notebook = nbformat.read(path, as_version=4)
        if tagged_cells(notebook, FRAMEWORK_TAG):
            discovered.append(path)
    return discovered


def select_notebooks(paths: list[Path], profile: str | None) -> list[Path]:
    if profile is None:
        return paths
    selected: list[Path] = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        if notebook_profile(notebook) == profile:
            selected.append(path)
    return selected


def build_script(notebook: nbformat.NotebookNode) -> str:
    sources = [tagged_cells(notebook, tag)[0].source for tag in REQUIRED_CODE_TAGS]
    report_code = f"""
if not isinstance(framework_check, dict):
    raise TypeError("framework_check 必须是字典")
print({RESULT_MARKER!r} + json.dumps(framework_check, ensure_ascii=False, sort_keys=True))
"""
    return "\n\n".join([
        sources[0],
        sources[1],
        sources[2],
        "import json",
        report_code,
    ])


def parse_report(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if not lines:
        raise ValueError("执行结果中缺少 framework_check 报告")
    payload = json.loads(lines[-1][len(RESULT_MARKER):])
    if not isinstance(payload, dict):
        raise TypeError("framework_check 报告不是字典")
    return payload


def report_issues(report: dict, require_passed: bool, expected_kind: str) -> list[str]:
    status = report.get("status")
    if status == "skipped":
        issues = []
        if not isinstance(report.get("reason"), str) or not report["reason"].strip():
            issues.append("skipped 报告必须包含非空 reason")
        if require_passed:
            issues.append("当前门禁要求真实执行，不能使用 skipped 结果")
        return issues
    if status != "passed":
        return [f"status 应为 passed 或 skipped，当前为 {status!r}"]

    issues: list[str] = []
    report_kind = report.get("check_kind", DEFAULT_CHECK_KIND)
    if report_kind != expected_kind:
        issues.append(
            f"check_kind 应为 {expected_kind!r}，当前为 {report_kind!r}"
        )
    versions = report.get("versions")
    if not isinstance(versions, dict) or not versions:
        issues.append("passed 报告必须包含非空 versions")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks:
        issues.append("passed 报告必须包含非空 checks")
    else:
        failed = [name for name, value in checks.items() if value is not True]
        if failed:
            issues.append(f"以下具名检查未通过：{failed}")
    return issues


def run_notebook_check(path: Path, require_passed: bool, timeout: int) -> bool:
    notebook = nbformat.read(path, as_version=4)
    issues = structure_issues(notebook)
    if issues:
        print(f"FAIL {path}")
        for issue in issues:
            print(f"  - {issue}")
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as handle:
        handle.write(build_script(notebook))
        script_path = Path(handle.name)

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=path.resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"FAIL {path}\n  - 执行超过 {timeout} 秒")
        return False
    finally:
        script_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        print(f"FAIL {path}\n  - 框架核对单元执行失败")
        if completed.stdout.strip():
            print(completed.stdout.rstrip())
        if completed.stderr.strip():
            print(completed.stderr.rstrip())
        return False

    try:
        report = parse_report(completed.stdout)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"FAIL {path}\n  - {exc}")
        return False

    framework_cell = tagged_cells(notebook, FRAMEWORK_TAG)[0]
    expected_kind = framework_cell.metadata.get("check_kind", DEFAULT_CHECK_KIND)
    issues = report_issues(report, require_passed, expected_kind)
    if issues:
        print(f"FAIL {path}")
        for issue in issues:
            print(f"  - {issue}")
        return False

    status = report["status"].upper()
    suffix = f": {report.get('reason')}" if report["status"] == "skipped" else ""
    print(f"{status} {path}{suffix}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证机制 Notebook 的可选框架与接口核对单元。"
    )
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--all-part02",
        action="store_true",
        help="自动发现 Part 02 中带有框架或接口核对单元的机制 Notebook。",
    )
    parser.add_argument(
        "--profile",
        help="只执行 framework_profile 与该值一致的 Notebook。",
    )
    parser.add_argument(
        "--require-passed",
        action="store_true",
        help="将 skipped 视为失败；应在 Notebook 指定的框架环境中使用。",
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if args.all_part02 and args.notebooks:
        parser.error("--all-part02 不能与显式 Notebook 路径同时使用")
    paths = discover_framework_notebooks() if args.all_part02 else args.notebooks
    if not paths:
        parser.error("请传入 Notebook，或使用 --all-part02")
    notebooks = select_notebooks(paths, args.profile)
    if not notebooks:
        profile_suffix = f" framework_profile={args.profile!r}" if args.profile else ""
        parser.error(f"没有找到可执行的框架或接口核对 Notebook；筛选条件：{profile_suffix or '无'}")

    profile_label = args.profile or "all"
    print(f"Selected {len(notebooks)} notebook(s); framework_profile={profile_label}")

    results = [
        run_notebook_check(path, args.require_passed, args.timeout)
        for path in notebooks
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
