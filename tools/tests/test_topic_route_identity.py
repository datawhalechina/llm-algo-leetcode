"""Check shared topic identities separately from individual topic documentation."""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TopicRouteIdentityTests(unittest.TestCase):
    def test_topic_intro_identity_categories_match_the_overview(self):
        """主线、基础、横向与领域身份统一；兼容页不计作专题。"""
        counts = {}
        compatible = []
        for page in (ROOT / "topic_discussion").glob("*/intro.md"):
            text = page.read_text()
            match = re.search(r"^> 专题类型：([^　\n]+)　学习目标：(.+)$", text, re.M)
            if match:
                category = match.group(1)
                counts[category] = counts.get(category, 0) + 1
                self.assertTrue(match.group(2).endswith("。"), page)
            else:
                compatible.append(page.parent.name)
        self.assertEqual(counts, {"主学习路线": 5, "基础支撑专题": 2,
                                  "横向工程专题": 4, "领域专题": 2})
        self.assertEqual(set(compatible), {"performance_analysis", "memory_performance_tuning"})
        architecture = ROOT / "topic_discussion/llm_architecture_evolution/intro.md"
        combined = ROOT / "topic_discussion/model_architecture_and_training/intro.md"
        self.assertIn("专题类型：基础支撑专题", architecture.read_text())
        self.assertIn("专题类型：主学习路线", combined.read_text())


if __name__ == "__main__":
    unittest.main()
