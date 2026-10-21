#!/usr/bin/env python3
"""Validate structural arrow rules that are easy to miss in SVG review."""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


MOVETO_RE = re.compile(r"(?<![A-Za-z])[Mm](?=[\s+\-.0-9])")
CLASS_MARKER_RE = re.compile(r"\.([A-Za-z_][\w-]*)\s*\{[^}]*marker-end\s*:", re.DOTALL)
URL_MARKER_RE = re.compile(r"url\(#([^)]+)\)")
PATH_TOKEN_RE = re.compile(r"[A-Za-z]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def path_endpoints(data: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return endpoints for the absolute/relative M/L/H/V paths used by diagrams."""
    tokens = PATH_TOKEN_RE.findall(data)
    if not tokens or tokens[0] not in {"M", "m"}:
        return None
    if any(token.isalpha() and token not in "MmLlHhVv" for token in tokens):
        return None
    if PATH_TOKEN_RE.sub("", data).strip(" ,\t\r\n"):
        return None
    index = 0
    command = ""
    x = y = 0.0
    start: tuple[float, float] | None = None
    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if index >= len(tokens):
                return None
        if command in "MmLl":
            if index + 1 >= len(tokens) or any(
                value.isalpha() for value in tokens[index:index + 2]
            ):
                return None
            next_x, next_y = float(tokens[index]), float(tokens[index + 1])
            if command.islower():
                next_x += x
                next_y += y
            x, y = next_x, next_y
            index += 2
            if start is None:
                start = (x, y)
            if command in "Mm":
                command = "l" if command == "m" else "L"
        elif command in "Hh":
            if tokens[index].isalpha():
                return None
            next_x = float(tokens[index])
            x = x + next_x if command == "h" else next_x
            index += 1
        elif command in "Vv":
            if tokens[index].isalpha():
                return None
            next_y = float(tokens[index])
            y = y + next_y if command == "v" else next_y
            index += 1
        else:
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
    return (start, (x, y)) if start is not None else None


def point_on_boundary(node: ET.Element, point: tuple[float, float], tolerance: float = 1.5) -> bool:
    x, y = point
    tag = local_name(node.tag)
    if tag == "rect":
        left = float(node.attrib.get("x", 0))
        top = float(node.attrib.get("y", 0))
        right = left + float(node.attrib.get("width", 0))
        bottom = top + float(node.attrib.get("height", 0))
        within_x = left - tolerance <= x <= right + tolerance
        within_y = top - tolerance <= y <= bottom + tolerance
        return within_x and within_y and min(
            abs(x - left), abs(x - right), abs(y - top), abs(y - bottom)
        ) <= tolerance
    if tag == "circle":
        cx = float(node.attrib.get("cx", 0))
        cy = float(node.attrib.get("cy", 0))
        radius = float(node.attrib.get("r", 0))
        return abs(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 - radius) <= tolerance
    return False


def audit(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return [f"XML 解析失败：{exc}"]

    root = tree.getroot()
    styles = "\n".join(
        node.text or "" for node in root.iter() if local_name(node.tag) == "style"
    )
    marker_classes = set(CLASS_MARKER_RE.findall(styles))
    marker_ids = {
        node.attrib.get("id") for node in root.iter()
        if local_name(node.tag) == "marker" and node.attrib.get("id")
    }
    nodes_by_id = {
        node.attrib["id"]: node for node in root.iter()
        if node.attrib.get("id") and local_name(node.tag) in {"rect", "circle"}
    }

    for node in root.iter():
        if local_name(node.tag) != "path":
            continue
        classes = set(node.attrib.get("class", "").split())
        marker_value = node.attrib.get("marker-end", "")
        has_arrow = bool(classes & marker_classes) or bool(marker_value)
        if not has_arrow:
            continue
        data = node.attrib.get("d", "")
        if len(MOVETO_RE.findall(data)) > 1:
            issues.append(
                "一条带箭头的 path 含多个 M/m 子路径；请拆成‘一条语义边一个 path’"
            )
        for marker_id in URL_MARKER_RE.findall(marker_value):
            if marker_id not in marker_ids:
                issues.append(f"marker-end 引用了不存在的 marker：{marker_id}")

        source_id = node.attrib.get("data-source")
        target_id = node.attrib.get("data-target")
        if source_id or target_id:
            if not source_id or not target_id:
                issues.append("带连接信息的箭头必须同时声明 data-source 与 data-target")
                continue
            source = nodes_by_id.get(source_id)
            target = nodes_by_id.get(target_id)
            if source is None:
                issues.append(f"data-source 引用了不存在的节点：{source_id}")
            if target is None:
                issues.append(f"data-target 引用了不存在的节点：{target_id}")
            endpoints = path_endpoints(data)
            if endpoints is None:
                issues.append(
                    f"无法解析 {source_id} → {target_id} 的路径；带端点校验的边仅支持 M/L/H/V"
                )
            elif source is not None and target is not None:
                start, end = endpoints
                if not point_on_boundary(source, start):
                    issues.append(
                        f"{source_id} → {target_id} 的起点 {start} 未连接 source 边界"
                    )
                if not point_on_boundary(target, end):
                    issues.append(
                        f"{source_id} → {target_id} 的终点 {end} 未连接 target 边界"
                    )

    for marker_class in marker_classes:
        match = re.search(rf"\.{re.escape(marker_class)}\s*\{{([^}}]*)\}}", styles, re.DOTALL)
        if match:
            for marker_id in URL_MARKER_RE.findall(match.group(1)):
                if marker_id not in marker_ids:
                    issues.append(f"样式 .{marker_class} 引用了不存在的 marker：{marker_id}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.files:
        issues = audit(path)
        if issues:
            failed = True
            for issue in issues:
                print(f"{path}: {issue}")
        else:
            print(f"OK {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
