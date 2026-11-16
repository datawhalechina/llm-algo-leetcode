#!/usr/bin/env python3
"""Focused tests for SVG layout metadata parsing."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_svg_layout import parse_svg


class SvgLayoutTests(unittest.TestCase):
    def test_complete_canvas_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok.svg"
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="600" '
                'viewBox="0 0 1240 600"/>',
                encoding="utf-8",
            )
            _, issues = parse_svg(path)
            self.assertEqual([], issues)

    def test_missing_size_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.svg"
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 600"/>',
                encoding="utf-8",
            )
            _, issues = parse_svg(path)
            self.assertTrue(any(issue.key.startswith("missing_explicit_size:") for issue in issues))

    def test_distorted_canvas_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distorted.svg"
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" '
                'viewBox="0 0 1200 400"/>',
                encoding="utf-8",
            )
            _, issues = parse_svg(path)
            self.assertTrue(any(issue.key.startswith("distorted_canvas:") for issue in issues))

    def test_layout_roles_exclude_panels_and_nested_glyphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roles.svg"
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="400" '
                'viewBox="0 0 1200 400">'
                '<rect width="1200" height="400"/>'
                '<rect data-layout-role="panel" width="400" height="240"/>'
                '<g data-layout-role="glyph-group"><rect width="30" height="40"/></g>'
                '<rect class="q" width="30" height="40"/>'
                '<rect width="180" height="72"/>'
                '</svg>',
                encoding="utf-8",
            )
            info, issues = parse_svg(path)
            self.assertEqual([], issues)
            self.assertIsNotNone(info)
            self.assertEqual(72, info.median_node_height)


if __name__ == "__main__":
    unittest.main()
