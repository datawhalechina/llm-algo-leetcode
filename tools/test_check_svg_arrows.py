"""Regression tests for annotated SVG arrow endpoints."""

from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from check_svg_arrows import audit, path_endpoints, point_on_boundary


class SvgArrowTests(unittest.TestCase):
    def test_absolute_paths(self):
        self.assertEqual(path_endpoints("M 10 20 H 40 V 60 L 80 90"), ((10, 20), (80, 90)))

    def test_relative_and_implicit_lines(self):
        self.assertEqual(path_endpoints("m 10 20 5 5 h 10 v -5"), ((10, 20), (25, 20)))

    def test_commas_and_exponents(self):
        self.assertEqual(path_endpoints("M1e1,20 L30,4e1"), ((10, 20), (30, 40)))

    def test_invalid_paths_are_rejected_without_crashing(self):
        for data in ("", "M", "M0", "M0 0 L", "M0 0 L1 H2", "M0 0 H V2", "L1 2",
                     "M0 0 C1 2 3 4 5 6", "M0 0 Z", "M0 0 L1e999 2", "M0 0 @ L1 2"):
            with self.subTest(data=data):
                self.assertIsNone(path_endpoints(data))

    def test_rect_boundary_excludes_interior(self):
        node = ET.fromstring('<rect x="10" y="20" width="30" height="40"/>')
        self.assertTrue(point_on_boundary(node, (40, 40)))
        self.assertFalse(point_on_boundary(node, (25, 40)))

    def test_circle_boundary(self):
        node = ET.fromstring('<circle cx="10" cy="20" r="5"/>')
        self.assertTrue(point_on_boundary(node, (15, 20)))
        self.assertFalse(point_on_boundary(node, (10, 20)))

    def arrow_issues(self, data, attributes='data-source="a" data-target="b"'):
        root = ET.fromstring(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<defs><marker id="arrow"/></defs>'
            '<rect id="a" x="0" y="0" width="20" height="20"/>'
            '<rect id="b" x="50" y="0" width="20" height="20"/>'
            f'<path d="{data}" marker-end="url(#arrow)" {attributes}/></svg>'
        )
        with patch("check_svg_arrows.ET.parse", return_value=ET.ElementTree(root)):
            return audit(Path("fixture.svg"))

    def test_connected_arrow(self):
        self.assertEqual(self.arrow_issues("M20 10 H50"), [])

    def test_disconnected_target_is_reported(self):
        self.assertTrue(any("target 边界" in item for item in self.arrow_issues("M20 10 H40")))

    def test_incomplete_annotation_is_reported(self):
        self.assertTrue(self.arrow_issues("M20 10 H50", 'data-source="a"'))

    def test_missing_node_is_reported(self):
        issues = self.arrow_issues("M20 10 H50", 'data-source="a" data-target="missing"')
        self.assertTrue(any("不存在的节点" in item for item in issues))

    def test_unannotated_curves_remain_supported_by_structural_audit(self):
        self.assertEqual(self.arrow_issues("M20 10 C30 0 40 0 50 10", ""), [])


if __name__ == "__main__":
    unittest.main()
