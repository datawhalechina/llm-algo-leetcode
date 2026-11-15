"""Check migration selection and preview safety using temporary fixtures only."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "md_to_notebook.py"
SPEC = importlib.util.spec_from_file_location("md_to_notebook_under_test", SCRIPT)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class MarkdownMigrationTests(unittest.TestCase):
    """Keep group indexes and existing notebooks safe during source selection."""

    def setUp(self):
        """Create a small source directory outside the tutorial tree."""
        self.temporary = tempfile.TemporaryDirectory(prefix="markdown-migration-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("intro.md", "1A.md", "1F.md", "1G.md", "01_Lesson.md"):
            (self.root / name).write_text("# Example\n", encoding="utf-8")
        self.lesson = self.root / "01_Lesson.md"

    def run_preview(self, *arguments):
        """Run the public CLI in preview mode from a non-repository directory."""
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, arguments), "--dry-run"],
            cwd=self.root, capture_output=True, text=True, timeout=20,
        )

    def snapshot(self):
        """Capture all fixture names and bytes to detect any preview writes."""
        return {path.name: path.read_bytes() for path in self.root.iterdir()}

    def test_group_names_include_f_and_future_groups(self):
        """Group selection must not stop at the old A--E range."""
        for name in ("0A.md", "1F.md", "1G.md", "3Z.md", "12A.md"):
            with self.subTest(name=name):
                self.assertTrue(CONVERTER.is_group_page(Path(name)))
        for name in ("01_Lesson.md", "1F_notes.md", "intro.md", "19.md"):
            with self.subTest(name=name):
                self.assertFalse(CONVERTER.is_group_page(Path(name)))

    def test_directory_selection_excludes_intro_and_groups(self):
        """Directory conversion should select only the lesson fixture."""
        self.assertEqual(CONVERTER.iter_source_files([self.root]), [self.lesson])

    def test_explicit_files_use_the_same_exclusions(self):
        """An explicit --file target must not bypass group-page filtering."""
        targets = [self.root / name for name in ("intro.md", "1F.md", "01_Lesson.md")]
        self.assertEqual(CONVERTER.iter_source_files(targets), [self.lesson])

    def test_directory_preview_does_not_write(self):
        """A directory preview reports the lesson without creating notebooks."""
        before = self.snapshot()
        result = self.run_preview("--dir", self.root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[dry-run]", result.stdout)
        self.assertIn("01_Lesson.ipynb", result.stdout)
        self.assertNotIn("1F.ipynb", result.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_single_file_preview_does_not_write(self):
        """A single-file preview also preserves all fixture contents."""
        before = self.snapshot()
        result = self.run_preview("--file", self.lesson)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("01_Lesson.ipynb", result.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_explicit_group_preview_has_no_conversion_targets(self):
        """Selecting only 1F produces the existing no-source exit status."""
        before = self.snapshot()
        result = self.run_preview("--file", self.root / "1F.md")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("No markdown part files found", result.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_existing_notebook_is_skipped_without_overwrite(self):
        """Both the helper and CLI preserve a pre-existing notebook by default."""
        notebook = self.lesson.with_suffix(".ipynb")
        notebook.write_bytes(b"existing notebook fixture\n")
        before = self.snapshot()
        self.assertEqual(CONVERTER.convert_md_to_ipynb(self.lesson), notebook)
        result = self.run_preview("--file", self.lesson)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skipped 1", result.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_overwrite_preview_still_preserves_existing_notebook(self):
        """Even --overwrite remains read-only when combined with --dry-run."""
        self.lesson.with_suffix(".ipynb").write_bytes(b"existing notebook fixture\n")
        before = self.snapshot()
        result = self.run_preview("--file", self.lesson, "--overwrite")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[dry-run]", result.stdout)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
