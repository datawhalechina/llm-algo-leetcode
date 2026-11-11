"""Check performance-analysis navigation and generated mirrors without GPU runs."""

from pathlib import Path
import re
import sys
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from convert_notebook import normalize_generated_markdown, normalize_markdown_links
from check_svg_arrows import audit as audit_svg_arrows


TOPIC = ROOT / "topic_discussion/performance_analysis"


class PerformanceAnalysisDocsTests(unittest.TestCase):
    def test_diagram_arrows_have_valid_endpoints_and_fixed_marker_units(self):
        assets = ROOT / "docs/public/topic_discussion/performance_analysis"
        diagrams = sorted(assets.glob("*.svg"))
        self.assertTrue(diagrams)
        for diagram in diagrams:
            with self.subTest(diagram=diagram.name):
                self.assertEqual(audit_svg_arrows(diagram), [])
                root = ET.parse(diagram).getroot()
                for marker in root.findall(".//{http://www.w3.org/2000/svg}marker"):
                    self.assertEqual(marker.get("markerUnits"), "userSpaceOnUse")

    def test_all_topic_pages_have_matching_generated_mirrors(self):
        self.assertTrue((TOPIC / "formula_reference.md").is_file())
        for source in sorted(TOPIC.glob("*.md")):
            with self.subTest(page=source.name):
                target = ROOT / "docs" / source.relative_to(ROOT)
                expected = normalize_generated_markdown(
                    normalize_markdown_links(source.read_text(encoding="utf-8"))
                )
                self.assertEqual(target.read_text(encoding="utf-8"), expected)

    def test_setup_and_asset_headings_follow_the_fixed_structure(self):
        expected = {
            "quickstart.md": [
                "1. 选择环境", "2. 首次创建", "3. 中断续装与重新进入",
                "4. 手动创建（可选）", "5. 验证与常见问题",
            ],
            "experiment_assets.md": [
                "环境与实验入口", "机制实验", "项目实验", "结果版本与复现",
            ],
        }
        for name, headings in expected.items():
            with self.subTest(page=name):
                text = (TOPIC / name).read_text(encoding="utf-8")
                self.assertEqual(re.findall(r"^## (.+)$", text, re.M), headings)
                self.assertEqual(text.count("<details>"), 1)
                self.assertEqual(text.count("</details>"), 1)
                self.assertNotRegex(text, r"(?m)^#### ")

    def test_creation_command_ends_its_section(self):
        text = (TOPIC / "quickstart.md").read_text(encoding="utf-8")
        self.assertIn('<a id="perf-cuda-mechanism-quickstart"></a>', text)
        section = text.split("## 2. 首次创建", 1)[1].split("## 3.", 1)[0]
        before, code = section.split("```bash", 1)
        self.assertTrue(before.strip())
        command, after = code.split("```", 1)
        self.assertIn("--profile PERF-CUDA-MECHANISM", command)
        self.assertFalse(after.strip())

    def test_local_source_and_mirror_file_links_exist(self):
        for base in (TOPIC, ROOT / "docs/topic_discussion/performance_analysis"):
            for page in sorted(base.glob("*.md")):
                text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
                for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                    if re.match(r"[A-Za-z][\w+.-]*:|#", target):
                        continue
                    relative = target.split("#", 1)[0]
                    if relative:
                        with self.subTest(page=page.name, target=target):
                            self.assertTrue((page.parent / relative).is_file())


if __name__ == "__main__":
    unittest.main()
