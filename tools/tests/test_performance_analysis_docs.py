"""Check performance documentation, canonical ownership and legacy redirects."""

from pathlib import Path
import re
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.convert_notebook import normalize_generated_markdown, normalize_markdown_links
from tools.check_svg_arrows import audit as audit_svg_arrows

TOPIC = ROOT / "topic_discussion/performance_optimization"
LEGACY_TOPIC = ROOT / "topic_discussion/performance_analysis"


class PerformanceAnalysisDocsTests(unittest.TestCase):
    def test_all_legacy_analysis_pages_redirect_and_preserve_fragments(self):
        """旧路径仅兼容跳转，保留深链接，不再持有第二套正文。"""
        pages = list(LEGACY_TOPIC.glob("*.md"))
        self.assertEqual(len(pages), 14)
        for page in pages:
            text = page.read_text()
            self.assertIn("window.location.search + window.location.hash", text)
            self.assertIn("withBase('/topic_discussion/performance_optimization/", text)
            self.assertNotIn("\n## ", text)
            self.assertNotIn("\n|", text)
            target = re.search(r"\]\(([^)]+)\)", text).group(1)
            self.assertTrue((page.parent / target).is_file(), page.name)

    def test_performance_support_pages_have_one_canonical_owner(self):
        """旧方法入口兼容保留，但不再维护第二套手册、问题链和术语。"""
        target = ROOT / "topic_discussion/performance_optimization"
        for name in ("walkthrough.md", "casebook.md", "glossary.md"):
            old = (LEGACY_TOPIC / name).read_text()
            self.assertIn(f"../performance_optimization/{name}", old)
            self.assertNotIn("\n|", old)
            self.assertNotIn("![", old)
            self.assertTrue((target / name).is_file())
        glossary = (target / "glossary.md").read_text()
        terms = [line.split("|")[1].strip() for line in glossary.splitlines()
                 if line.startswith("| ") and not line.startswith("| 中文")]
        self.assertEqual(len(terms), len(set(terms)))
        self.assertIn("配对比值", glossary)
        self.assertIn("边际收益", glossary)

    def test_analysis_method_entry_belongs_to_the_performance_route(self):
        config = (ROOT / "docs/.vitepress/config.mts").read_text()
        route = config.split("text: '主学习路线：性能优化'", 1)[1].split("text: '主学习路线：算子优化'", 1)[0]
        self.assertIn("/topic_discussion/performance_optimization/methods/intro", route)
        independent = config.split("text: '基础支撑专题'", 1)[1]
        self.assertNotIn("/topic_discussion/performance_analysis/intro", independent)
        self.assertIn("所属路线：性能优化", (TOPIC / "methods/intro.md").read_text())

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
        for source in sorted([*TOPIC.rglob("*.md"), *LEGACY_TOPIC.glob("*.md")]):
            with self.subTest(page=source.name):
                target = ROOT / "docs" / source.relative_to(ROOT)
                expected = normalize_generated_markdown(
                    normalize_markdown_links(source.read_text(encoding="utf-8"))
                )
                self.assertEqual(target.read_text(encoding="utf-8"), expected)

    def test_setup_and_asset_headings_follow_the_fixed_structure(self):
        expected = {
            "gpu_quickstart.md": [
                "1. 选择环境", "2. 首次创建", "3. 中断续装与重新进入",
                "4. 手动创建（可选）", "5. 验证与常见问题",
            ],
            "measurement_assets.md": [
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
        text = (TOPIC / "gpu_quickstart.md").read_text(encoding="utf-8")
        self.assertIn('<a id="perf-cuda-mechanism-quickstart"></a>', text)
        section = text.split("## 2. 首次创建", 1)[1].split("## 3.", 1)[0]
        before, code = section.split("```bash", 1)
        self.assertTrue(before.strip())
        command, after = code.split("```", 1)
        self.assertIn("--profile PERF-CUDA-MECHANISM", command)
        self.assertFalse(after.strip())

    def test_local_source_and_mirror_file_links_exist(self):
        for base in (TOPIC, ROOT / "docs/topic_discussion/performance_optimization", LEGACY_TOPIC,
                     ROOT / "docs/topic_discussion/performance_analysis"):
            for page in sorted(base.rglob("*.md")):
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
