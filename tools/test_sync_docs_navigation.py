"""Check opt-in topic mirror pruning without touching repository documents."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import sync_docs_navigation as sync


class ScopedTopicPruningTests(unittest.TestCase):
    """Keep cleanup bounded to the chosen topic's obsolete Markdown files."""

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.docs = self.root / "docs"
        self.source = self.root / "topic_discussion" / "example"
        self.target = self.docs / "topic_discussion" / "example"
        self.source.mkdir(parents=True)
        self.target.mkdir(parents=True)
        (self.source / "intro.md").write_text("# Example\n", encoding="utf-8")
        self.addCleanup(patch.stopall)
        patch.object(sync, "ROOT", self.root).start()
        patch.object(sync, "DOCS", self.docs).start()

    def run_sync(self, *, prune=False):
        """Run the real sync function with output captured for small tests."""
        with redirect_stdout(io.StringIO()):
            sync.sync_topic("example", prune=prune)

    def test_default_keeps_stale_mirror(self):
        """Existing callers retain the previous non-pruning behavior."""
        stale = self.target / "old.md"
        stale.write_text("preserve by default", encoding="utf-8")
        self.run_sync()
        self.assertTrue(stale.exists())
        self.assertEqual((self.target / "intro.md").read_text(), "# Example\n")

    def test_prune_removes_only_selected_stale_markdown(self):
        """Remove deleted pages and empty folders, preserving assets and siblings."""
        obsolete = self.target / "obsolete"
        obsolete.mkdir()
        (obsolete / "old.md").write_text("old", encoding="utf-8")
        keep = self.target / "records.json"
        keep.write_text("{}", encoding="utf-8")
        sibling = self.docs / "topic_discussion" / "other" / "old.md"
        sibling.parent.mkdir()
        sibling.write_text("untouched", encoding="utf-8")
        nested = self.source / "nested" / "current.md"
        nested.parent.mkdir()
        nested.write_text("# Current\n", encoding="utf-8")
        self.run_sync(prune=True)
        self.assertFalse(obsolete.exists())
        self.assertEqual(keep.read_text(), "{}")
        self.assertEqual(sibling.read_text(), "untouched")
        self.assertEqual((self.target / "nested/current.md").read_text(), "# Current\n")

    def test_prune_is_repeatable(self):
        """A second sync produces identical current content."""
        self.run_sync(prune=True)
        first = (self.target / "intro.md").read_bytes()
        self.run_sync(prune=True)
        self.assertEqual((self.target / "intro.md").read_bytes(), first)

    def test_empty_source_stops_before_cleanup(self):
        """An accidentally empty source cannot erase the mirror."""
        (self.source / "intro.md").unlink()
        stale = self.target / "old.md"
        stale.write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_sync(prune=True)
        self.assertEqual(stale.read_text(), "keep")

    def test_symlinked_topic_mirror_is_rejected(self):
        """Do not write or prune through a redirected topic directory."""
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "old.md").write_text("keep", encoding="utf-8")
        self.target.rmdir()
        self.target.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_sync(prune=True)
        self.assertEqual((outside / "old.md").read_text(), "keep")
        self.assertFalse((outside / "intro.md").exists())

    def test_symlinked_file_is_rejected_before_writes(self):
        """Validate every target before replacing any existing mirror."""
        outside = self.root / "outside.md"
        outside.write_text("keep", encoding="utf-8")
        (self.target / "old.md").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.run_sync(prune=True)
        self.assertEqual(outside.read_text(), "keep")
        self.assertFalse((self.target / "intro.md").exists())

    def test_topic_traversal_is_rejected(self):
        """The opt-in flag does not broaden accepted topic names."""
        for topic in ("", ".", "..", "../other", "/tmp/example", "nested/topic"):
            with self.subTest(topic=topic), self.assertRaises(ValueError):
                sync.sync_topic(topic, prune=True)

    def test_cli_requires_a_topic_for_pruning(self):
        """Reject global pruning rather than guessing a directory."""
        with patch("sys.argv", ["sync_docs_navigation.py", "--prune"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                sync.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
