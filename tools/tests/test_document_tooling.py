"""Check document discovery and CLI imports using temporary files only."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import check_mechanism_step4_tables as step4
from tools import check_source_docs_mirror as mirror
from tools import convert_notebook as converter


@contextmanager
def working_directory(path):
    """Use a temporary cwd and restore it even when an assertion fails."""
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class DocumentToolingTests(unittest.TestCase):
    def test_cli_help_works_without_site_packages(self):
        """Both startup modes must import without running document writes."""
        for name in ("check_mechanism_step4_tables", "sync_docs_navigation"):
            for arguments in ([str(ROOT / "tools" / f"{name}.py"), "--help"],
                              ["-m", f"tools.{name}", "--help"]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, "-S", *arguments], cwd=ROOT,
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_script_help_works_outside_repository(self):
        """Absolute script paths must not require the repository as cwd."""
        with tempfile.TemporaryDirectory() as directory:
            for name in ("check_mechanism_step4_tables", "sync_docs_navigation"):
                result = subprocess.run(
                    [sys.executable, "-S", str(ROOT / "tools" / f"{name}.py"), "--help"],
                    cwd=directory, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_nested_sources_and_example_configs_are_discovered(self):
        """Nested project pages are included; caches and hidden paths are not."""
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            part = Path("05_Integrated_Projects_and_Decisions")
            expected = {
                part / "01_Project/index.md",
                part / "01_Project/lesson.ipynb",
                part / "01_Project/project_config.example.yaml",
            }
            excluded = {
                part / ".ipynb_checkpoints/lesson.ipynb",
                part / "benchmarks/result.md",
                part / "model_cache/model.md",
                part / "__pycache__/cached.md",
                part / "01_Project/other.yaml",
            }
            for path in expected | excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            args = argparse.Namespace(files=None, dirs=[str(part)])
            self.assertEqual(set(map(Path, converter.collect_targets(args))), expected)

    def test_generated_markdown_normalization_is_idempotent(self):
        text = "# Example  \n\n[page](./example.ipynb)\n\n"
        normalized = converter.normalize_generated_markdown(
            converter.normalize_markdown_links(text)
        )
        self.assertEqual(normalized, "# Example\n\n[page](./example.md)\n")
        self.assertEqual(converter.normalize_generated_markdown(normalized), normalized)

    def test_explicit_empty_directory_does_not_fall_back_to_all_parts(self):
        """A local conversion must not silently select unrelated chapters."""
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            unrelated = Path("02_PyTorch_Algorithms/lesson.md")
            unrelated.parent.mkdir()
            unrelated.touch()
            selected = Path("05_Integrated_Projects_and_Decisions/empty")
            selected.mkdir(parents=True)
            args = argparse.Namespace(files=None, dirs=[str(selected)])
            self.assertEqual(converter.collect_targets(args), [])

    def test_mirror_discovery_ignores_non_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = root / "05_Integrated_Projects_and_Decisions"
            for name in (".hidden/page.md", "benchmarks/result.md",
                         "model_cache/cached.md", "__pycache__/cached.md"):
                path = part / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (part / "directory.md").mkdir()
            with patch.object(mirror, "ROOT", root), patch.object(mirror, "SOURCE_DIRS", [part]):
                self.assertEqual(mirror.iter_source_pages(), set())

    def test_mirror_checks_include_example_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = root / "05_Integrated_Projects_and_Decisions"
            source = part / "01_Project/project_config.example.yaml"
            source.parent.mkdir(parents=True)
            source.touch()
            target = root / "docs" / source.relative_to(root)
            with patch.object(mirror, "ROOT", root), patch.object(mirror, "DOCS_DIR", root / "docs"), patch.object(mirror, "SOURCE_DIRS", [part]):
                self.assertEqual(mirror.iter_source_pages(), {target.relative_to(root)})
                self.assertEqual(mirror.iter_docs_pages(), set())
                target.parent.mkdir(parents=True)
                target.touch()
                self.assertEqual(mirror.iter_docs_pages(), {target.relative_to(root)})

    def test_step4_reads_the_shared_registry(self):
        with patch.object(step4, "load_project_registry", return_value={
            "EXAMPLE": {"notebook": "02_PyTorch_Algorithms/example.ipynb"}
        }):
            self.assertEqual(step4.registered_project_paths(), {
                (step4.REPO_ROOT / "02_PyTorch_Algorithms/example.ipynb").resolve()
            })


if __name__ == "__main__":
    unittest.main()
