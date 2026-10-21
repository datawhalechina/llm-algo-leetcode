"""Regression checks for post-training evidence links and navigation entries."""

from pathlib import Path
import unittest

from convert_notebook import normalize_markdown_links
from sync_docs_navigation import NAV_FILES


class PostTrainingDocsSyncTests(unittest.TestCase):
    def test_result_links_follow_mirror_depth(self):
        for prefix in ("../", "../../", "../../../"):
            with self.subTest(prefix=prefix):
                source = f"[evidence]({prefix}benchmarks/results/run.json)"
                expected = f"[evidence](../{prefix}benchmarks/results/run.json)"
                self.assertEqual(normalize_markdown_links(source), expected)

    def test_result_link_keeps_fragment(self):
        source = "[evidence](../benchmarks/results/run.json#candidate)"
        self.assertEqual(
            normalize_markdown_links(source),
            "[evidence](../../benchmarks/results/run.json#candidate)",
        )

    def test_existing_lesson_and_image_links(self):
        source = "[lesson](./13_SFT_Training_Loop.ipynb) ![image](../docs/public/sft.svg)"
        self.assertEqual(
            normalize_markdown_links(source),
            "[lesson](./13_SFT_Training_Loop.md) ![image](../public/sft.svg)",
        )

    def test_non_result_links_keep_their_depth(self):
        source = "[guide](../guide.md) [remote](https://example.com/run.json)"
        self.assertEqual(normalize_markdown_links(source), source)

    def test_navigation_uses_current_post_training_pages(self):
        entries = NAV_FILES["topic_discussion"]
        root = Path(__file__).resolve().parents[1] / "topic_discussion"
        for name in (
            "08_trajectory_lifecycle_and_environment_interaction.md",
            "09_rl_execution_system_and_efficiency.md",
            "10_evaluation_and_adoption_decision.md",
        ):
            relative = f"post_training_optimization/{name}"
            with self.subTest(page=name):
                self.assertEqual(entries.count(relative), 1)
                self.assertTrue((root / relative).is_file())
        self.assertNotIn(
            "post_training_optimization/08_rl_engineering_and_efficiency.md", entries
        )
        self.assertNotIn(
            "post_training_optimization/09_preference_data_and_project_decision.md", entries
        )


if __name__ == "__main__":
    unittest.main()
