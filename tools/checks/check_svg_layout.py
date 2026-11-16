#!/usr/bin/env python3
"""Audit referenced SVG canvas metadata and same-page scale consistency.

The checker scans canonical source notebooks and topic Markdown. Existing
issues may be recorded in a baseline while newly introduced issues fail CI.
This lets the repository migrate old diagrams without allowing new drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = (
    "00_Prerequisites",
    "01_Hardware_Math_and_Systems",
    "02_PyTorch_Algorithms",
    "03_Triton_Kernels",
    "04_CUDA_and_System_Optimization",
    "05_Integrated_Projects_and_Decisions",
    "topic_discussion",
)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+?\.svg)(?:\s+[\"'][^\"']*[\"'])?\)", re.I)
NUMBER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:px)?$")


@dataclass(frozen=True)
class SvgInfo:
    path: Path
    width: float | None
    height: float | None
    viewbox_width: float | None
    viewbox_height: float | None
    max_font_size: float | None
    median_node_height: float | None

    @property
    def aspect(self) -> float | None:
        if not self.viewbox_width or not self.viewbox_height:
            return None
        return self.viewbox_width / self.viewbox_height

    @property
    def normalized_font_size(self) -> float | None:
        if not self.max_font_size or not self.viewbox_width:
            return None
        return self.max_font_size / self.viewbox_width

    @property
    def normalized_node_height(self) -> float | None:
        if not self.median_node_height or not self.viewbox_height:
            return None
        return self.median_node_height / self.viewbox_height


@dataclass(frozen=True)
class Issue:
    key: str
    message: str


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def numeric_dimension(raw: str | None) -> float | None:
    if raw is None or not NUMBER_RE.match(raw.strip()):
        return None
    return float(raw.removesuffix("px"))


def parse_svg(path: Path) -> tuple[SvgInfo | None, list[Issue]]:
    issues: list[Issue] = []
    rel = relative(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return None, [Issue(f"invalid_xml:{rel}", f"{rel}: SVG XML 无法解析：{exc}")]

    width = numeric_dimension(root.attrib.get("width"))
    height = numeric_dimension(root.attrib.get("height"))
    raw_viewbox = root.attrib.get("viewBox")
    viewbox_width = viewbox_height = None
    if raw_viewbox:
        try:
            values = [float(value) for value in re.split(r"[ ,]+", raw_viewbox.strip())]
            if len(values) == 4 and values[2] > 0 and values[3] > 0:
                viewbox_width, viewbox_height = values[2], values[3]
        except ValueError:
            pass

    if viewbox_width is None or viewbox_height is None:
        issues.append(Issue(f"missing_viewbox:{rel}", f"{rel}: 缺少有效 viewBox"))
    if width is None or height is None:
        issues.append(
            Issue(
                f"missing_explicit_size:{rel}",
                f"{rel}: 根元素必须同时声明数值 width 和 height",
            )
        )
    elif viewbox_width and viewbox_height:
        intrinsic_aspect = width / height
        viewbox_aspect = viewbox_width / viewbox_height
        if max(intrinsic_aspect, viewbox_aspect) / min(intrinsic_aspect, viewbox_aspect) > 1.01:
            issues.append(
                Issue(
                    f"distorted_canvas:{rel}",
                    f"{rel}: width/height 与 viewBox 宽高比不一致，浏览器可能拉伸图片",
                )
            )

    text = path.read_text(encoding="utf-8")
    font_sizes = [
        float(value)
        for pattern in (
            r"font-size\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)",
            r"font\s*:[^;{}]*?(\d+(?:\.\d+)?)px",
        )
        for value in re.findall(pattern, text, re.I)
    ]
    node_heights: list[float] = []
    if viewbox_width and viewbox_height:
        ignored_roles = {"background", "panel", "container", "glyph", "glyph-group", "decoration"}

        def iter_rects(node: ET.Element, inherited_role: str = ""):
            role = node.attrib.get("data-layout-role", inherited_role)
            if node.tag.rsplit("}", 1)[-1] == "rect":
                yield node, role
            for child in node:
                yield from iter_rects(child, role)

        for node, layout_role in iter_rects(root):
            rect_width = numeric_dimension(node.attrib.get("width"))
            rect_height = numeric_dimension(node.attrib.get("height"))
            if not rect_width or not rect_height:
                continue
            class_names = set(node.attrib.get("class", "").split())
            is_visual_glyph = bool(class_names & {"panel", "glyph", "q", "kv"})
            is_background = rect_width >= viewbox_width * 0.9 and rect_height >= viewbox_height * 0.9
            if not is_background and not is_visual_glyph and layout_role not in ignored_roles:
                node_heights.append(rect_height)

    return SvgInfo(
        path,
        width,
        height,
        viewbox_width,
        viewbox_height,
        max(font_sizes) if font_sizes else None,
        median(node_heights) if node_heights else None,
    ), issues


def source_pages() -> list[Path]:
    pages: list[Path] = []
    for directory in SOURCE_DIRS:
        root = ROOT / directory
        if not root.exists():
            continue
        pages.extend(root.rglob("*.md"))
        pages.extend(root.rglob("*.ipynb"))
    return sorted(path for path in pages if ".ipynb_checkpoints" not in path.parts)


def page_text(path: Path) -> str:
    if path.suffix == ".md":
        return path.read_text(encoding="utf-8")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )


def resolve_svg(page: Path, raw: str) -> Path | None:
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    candidates = [(page.parent / raw).resolve()]
    if raw.startswith("/"):
        candidates.append((ROOT / "docs" / raw.lstrip("/")).resolve())
    else:
        candidates.append((ROOT / raw).resolve())
    return next((path for path in candidates if path.exists()), None)


def spread(values: list[float]) -> float:
    return max(values) / min(values) if len(values) >= 2 and min(values) > 0 else 1.0


def audit(
    *,
    aspect_threshold: float,
    font_scale_threshold: float,
    node_scale_threshold: float,
) -> list[Issue]:
    issues: list[Issue] = []
    cache: dict[Path, SvgInfo | None] = {}
    for page in source_pages():
        try:
            references = IMAGE_RE.findall(page_text(page))
        except (OSError, json.JSONDecodeError):
            continue
        images: list[SvgInfo] = []
        for raw in references:
            target = resolve_svg(page, raw)
            if target is None:
                continue
            if target not in cache:
                cache[target], svg_issues = parse_svg(target)
                issues.extend(svg_issues)
            if cache[target] is not None:
                images.append(cache[target])  # type: ignore[arg-type]

        aspects = [image.aspect for image in images if image.aspect]
        if len(aspects) < 2:
            continue
        font_scales = [value for image in images if (value := image.normalized_font_size)]
        node_scales = [value for image in images if (value := image.normalized_node_height)]
        aspect_spread = spread(aspects)
        font_spread = spread(font_scales)
        node_spread = spread(node_scales)
        if (
            aspect_spread > aspect_threshold
            or font_spread > font_scale_threshold
            or node_spread > node_scale_threshold
        ):
            rel = relative(page)
            detail = ", ".join(
                f"{image.path.name}={image.viewbox_width:g}×{image.viewbox_height:g}"
                for image in images
                if image.viewbox_width and image.viewbox_height
            )
            issues.append(
                Issue(
                    f"page_scale_spread:{rel}",
                    f"{rel}: 同页 SVG 视觉尺度差异过大（宽高比 {aspect_spread:.2f}×，"
                    f"归一化字号 {font_spread:.2f}×，归一化节点高度 {node_spread:.2f}×）；{detail}",
                )
            )
    return sorted(set(issues), key=lambda issue: issue.key)


def load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("known_issues", []))


def write_baseline(path: Path, issues: list[Issue]) -> None:
    payload = {
        "schema_version": "svg-layout-baseline/v1",
        "description": "存量 SVG 布局问题；修复后应从基线移除，不得加入新的例外。",
        "known_issues": [issue.key for issue in issues],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "tools/svg_layout_baseline.json",
        help="Known-issue baseline used to reject only new regressions.",
    )
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail on every current issue.")
    parser.add_argument("--aspect-threshold", type=float, default=1.25)
    parser.add_argument("--font-scale-threshold", type=float, default=1.25)
    parser.add_argument("--node-scale-threshold", type=float, default=1.25)
    args = parser.parse_args()

    issues = audit(
        aspect_threshold=args.aspect_threshold,
        font_scale_threshold=args.font_scale_threshold,
        node_scale_threshold=args.node_scale_threshold,
    )
    baseline_path = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    if args.update_baseline:
        write_baseline(baseline_path, issues)
        print(f"Wrote {len(issues)} issue keys to {relative(baseline_path)}")
        return 0

    baseline = set() if args.strict else load_baseline(baseline_path)
    current = {issue.key for issue in issues}
    new_issues = [issue for issue in issues if issue.key not in baseline]
    new_page_warnings = [issue for issue in new_issues if issue.key.startswith("page_scale_spread:")]
    new_errors = [issue for issue in new_issues if not issue.key.startswith("page_scale_spread:")]
    resolved = sorted(baseline - current)

    print(
        f"SVG layout audit: current={len(issues)}, baseline={len(baseline)}, "
        f"new_errors={len(new_errors)}, new_page_warnings={len(new_page_warnings)}, "
        f"resolved={len(resolved)}"
    )
    for issue in new_errors:
        print(f"ERROR {issue.message}")
    for issue in new_page_warnings:
        print(f"WARNING {issue.message}")
    for key in resolved:
        print(f"RESOLVED {key}（请从 baseline 删除）")
    if not new_errors and not new_page_warnings:
        print("OK 未发现新增 SVG 尺度回归")
    return 1 if new_errors or (args.strict and new_page_warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
