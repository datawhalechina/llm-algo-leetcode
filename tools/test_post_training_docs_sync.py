"""Regression checks for post-training evidence links and navigation entries."""

from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
from unittest.mock import patch

from check_svg_arrows import audit as audit_svg_arrows
from convert_notebook import normalize_generated_markdown, normalize_markdown_links
from sync_docs_navigation import NAV_FILES
import sync_docs_navigation


def audit_topic_svg(path, *, require_connections=True):
    """Check real arrow paths, including class and inline marker declarations.

    Comparison panels need no markers. CSS support covers the simple class
    selectors used by this topic, including multiple classes on one element.
    Normalize only a temporary tree so the shared endpoint audit also sees
    inline styles; never rewrite the source SVG during a check.
    """
    root = ET.parse(path).getroot()
    local = lambda node: node.tag.rsplit("}", 1)[-1]
    styles = "\n".join(node.text or "" for node in root.iter() if local(node) == "style")
    rules = []
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", styles):
        for selector in selectors.split(","):
            selector = selector.strip()
            if re.fullmatch(r"(?:\.[\w-]+)+", selector):
                rules.append((set(re.findall(r"\.([\w-]+)", selector)), body))
    markers = {node.get("id"): node for node in root.iter() if local(node) == "marker"}
    issues = []
    for node in root.iter():
        if local(node) != "path":
            continue
        value = node.get("marker-end", "")
        classes = set(node.get("class", "").split())
        matching = [(required, body) for required, body in rules if required <= classes]
        for _, body in sorted(matching, key=lambda rule: len(rule[0])):
            declarations = re.findall(r"(?:^|;)\s*marker-end\s*:\s*([^;]+)", body)
            if declarations:
                value = declarations[-1].strip()
        inline = re.findall(r"(?:^|;)\s*marker-end\s*:\s*([^;]+)", node.get("style", ""))
        if inline:
            value = inline[-1].strip()
        if not value or value == "none":
            continue
        node.set("marker-end", value)
        if require_connections and (not node.get("data-source") or not node.get("data-target")):
            issues.append(f"arrow {node.get('d', '')} requires data-source and data-target")
        references = re.findall(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)", value)
        if not references:
            issues.append("arrow must reference a local marker")
        for marker_id in references:
            marker = markers.get(marker_id)
            if marker is None:
                issues.append(f"missing marker: {marker_id}")
            elif marker.get("markerUnits") != "userSpaceOnUse":
                issues.append(f"marker {marker_id} requires userSpaceOnUse")
    with TemporaryDirectory() as directory:
        normalized = Path(directory) / "diagram.svg"
        ET.ElementTree(root).write(normalized, encoding="unicode")
        issues.extend(audit_svg_arrows(normalized))
    return issues


class PostTrainingDocsSyncTests(unittest.TestCase):
    def test_lifecycle_feedback_branches_and_verifier_input(self):
        path = Path(__file__).resolve().parents[1] / "docs/public/topic_discussion/post_training_optimization/training_lifecycle_context.svg"
        root = ET.parse(path).getroot()
        nodes = {node.get("id"): node for node in root.iter() if node.get("id")}
        choice = nodes["feedback-choice"]
        self.assertTrue(choice.tag.endswith("rect"))
        self.assertEqual([choice.get(key) for key in ("x", "y", "width", "height")], ["1080", "180", "140", "82"])
        edges = [node for node in root.iter() if node.tag.endswith("path") and node.get("class") in {"line", "data"}]
        self.assertEqual(len(edges), 19)
        self.assertEqual({node.get("data-target") for node in edges if node.get("data-source") == "feedback-choice"}, {"offline-feedback", "online-sampling"})
        self.assertEqual([(node.get("data-source"), node.get("d")) for node in edges if node.get("data-target") == "verifier"], [("online-sampling", "M985 342H1080")])
        self.assertEqual(audit_topic_svg(path), [])

    def test_svg_arrow_rules_are_conditional_and_attribute_order_independent(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.svg"
            def check(body):
                path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg">{body}</svg>', encoding="utf-8")
                return audit_topic_svg(path)
            self.assertEqual(check('<path d="M0 0H10" stroke="black"/>'), [])
            nodes = '<rect id="a" x="0" y="0" width="10" height="10"/><rect id="b" x="30" y="0" width="10" height="10"/>'
            marker = '<defs><marker id="arrow" markerUnits="userSpaceOnUse"/></defs>'
            for declaration, styles in (
                ('marker-end="url(#arrow)"', ''),
                ('style="marker-end: url(#arrow)"', ''),
                ('class="extra flow"', '<style>.flow{marker-end:url(#arrow)}</style>'),
            ):
                with self.subTest(declaration=declaration):
                    edge = f'<path data-target="b" {declaration} d="M10 5H30" data-source="a"/>'
                    self.assertEqual(check(marker + nodes + styles + edge), [])
                    self.assertTrue(check(marker + nodes + styles + edge.replace(' data-source="a"', '')))
                    self.assertTrue(check(marker.replace('userSpaceOnUse', 'strokeWidth') + nodes + styles + edge))
                    self.assertTrue(check(marker + nodes + styles + edge.replace('M10 5', 'M15 5')))

    def test_scoped_topic_sync_preserves_other_directories_and_titles(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "topic_discussion" / "example"
            source.mkdir(parents=True)
            text = "# 保留专题标题\n\n[lesson](../../02_PyTorch_Algorithms/13.ipynb)\n\n[worker](../../tools/worker.py)\n"
            (source / "01_example.md").write_text(text, encoding="utf-8")
            other = root / "docs/topic_discussion/other/intro.md"
            other.parent.mkdir(parents=True)
            other.write_text("untouched", encoding="utf-8")
            with patch.object(sync_docs_navigation, "ROOT", root), patch.object(sync_docs_navigation, "DOCS", root / "docs"):
                sync_docs_navigation.sync_topic("example")
            mirror = root / "docs/topic_discussion/example/01_example.md"
            self.assertEqual(mirror.read_text(encoding="utf-8"), normalize_generated_markdown(normalize_markdown_links(text)))
            self.assertEqual(other.read_text(encoding="utf-8"), "untouched")

    def test_scoped_topic_sync_rejects_path_traversal(self):
        for topic in ("", ".", "..", "../other", "/tmp/example", "nested/topic", "nested\\topic"):
            with self.subTest(topic=topic), self.assertRaises(ValueError):
                sync_docs_navigation.sync_topic(topic)

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

    def test_repository_tool_links_follow_mirror_depth(self):
        source = "[worker](../../tools/verl_mechanism_worker.py)"
        self.assertEqual(
            normalize_markdown_links(source),
            "[worker](../../../tools/verl_mechanism_worker.py)",
        )

    def test_post_training_asset_and_quickstart_headings(self):
        root = Path(__file__).resolve().parents[1]
        topic = root / "topic_discussion/post_training_optimization"
        expected = {
            "experiment_assets.md": [
                "环境与实验入口", "机制实验", "项目实验", "结果版本与复现",
            ],
            "quickstart.md": [
                "1. 选择环境", "2. 首次创建", "3. 中断续装与重新进入",
                "4. 手动创建（可选）", "5. 验证与常见问题",
            ],
        }
        for name, headings in expected.items():
            with self.subTest(page=name):
                text = (topic / name).read_text(encoding="utf-8")
                self.assertEqual(re.findall(r"^## (.+)$", text, re.M), headings)
                self.assertNotRegex(text, r"(?m)^#### ")
                self.assertEqual(text.count("<details>"), 1)
                self.assertEqual(text.count("</details>"), 1)

    def test_post_training_reference_mirrors(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("glossary.md", "formula_reference.md", "experiment_assets.md", "quickstart.md", "walkthrough.md", "casebook.md"):
            with self.subTest(page=name):
                relative = Path("topic_discussion/post_training_optimization") / name
                expected = normalize_generated_markdown(
                    normalize_markdown_links((root / relative).read_text(encoding="utf-8"))
                )
                self.assertEqual((root / "docs" / relative).read_text(encoding="utf-8"), expected)

    def test_profile_descriptions_precede_creation_commands(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "topic_discussion/post_training_optimization/quickstart.md").read_text(encoding="utf-8")
        creation = text.split("## 2. 首次创建", 1)[1].split("## 3.", 1)[0]
        profiles = re.findall(r"^### (PT-[A-Z-]+)\n(.*?)(?=^### |\Z)", creation, re.M | re.S)
        self.assertEqual(len(profiles), 6)
        for profile, section in profiles:
            with self.subTest(profile=profile):
                before, command = section.split("```bash", 1)
                self.assertTrue(before.strip())
                _, after = command.split("```", 1)
                after = re.sub(r'<a id="[^"]+"></a>', "", after)
                self.assertFalse(after.strip())

    def test_post_training_body_and_intro_mirrors(self):
        root = Path(__file__).resolve().parents[1]
        topic = root / "topic_discussion/post_training_optimization"
        pages = [topic / "intro.md", *sorted(topic.glob("[0-9][0-9]_*.md"))]
        self.assertEqual(len(pages), 11)
        for page in pages:
            with self.subTest(page=page.name):
                expected = normalize_generated_markdown(normalize_markdown_links(page.read_text(encoding="utf-8")))
                mirror = root / "docs" / page.relative_to(root)
                self.assertEqual(mirror.read_text(encoding="utf-8"), expected)

    def test_body_diagrams_have_connected_arrows_and_explicit_fonts(self):
        root = Path(__file__).resolve().parents[1]
        topic = root / "topic_discussion/post_training_optimization"
        for page in sorted(topic.glob("*.md")):
            images = re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", page.read_text(encoding="utf-8"))
            if re.match(r"[0-9][0-9]_", page.name):
                self.assertTrue(images, page.name)
            for relative in images:
                image = (page.parent / relative).resolve()
                with self.subTest(page=page.name, image=image.name):
                    self.assertTrue(image.is_file())
                    # Overview arrows now receive the same strict identity
                    # checks as body arrows; unresolved geometry must fail.
                    self.assertEqual(audit_topic_svg(image), [])
                    text = image.read_text(encoding="utf-8")
                    self.assertNotRegex(text, r"\bfont\s*:")

    def test_intro_overview_count_and_stage_anchors(self):
        topic = Path(__file__).resolve().parents[1] / "topic_discussion/post_training_optimization"
        intro = (topic / "intro.md").read_text(encoding="utf-8")
        images = re.findall(r"!\[[^\]]*\]\(([^)]+\.svg)\)", intro)
        self.assertEqual([Path(image).name for image in images], [
            "post_training_roadmap.svg", "post_training_knowledge_map.svg",
        ])
        for name, anchor in (("walkthrough.md", "stage-selection"), ("04_end_to_end_experiment.md", "sft-upgrade-conditions")):
            self.assertIn(f"./{name}#{anchor}", intro)
            self.assertEqual((topic / name).read_text(encoding="utf-8").count(f'<a id="{anchor}"></a>'), 1)

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
