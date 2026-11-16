"""Keep the historical text-based question fixer strictly preview-only."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "standardize_part02_project_failures.py"
SPEC = importlib.util.spec_from_file_location("legacy_question_migration", SCRIPT)
LEGACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEGACY)


class LegacyQuestionMigrationTests(unittest.TestCase):
    """Use disposable fixture content, never real tutorial notebooks."""

    def setUp(self):
        """Provide content that the old heuristic would have rewritten."""
        temporary = tempfile.TemporaryDirectory(prefix="legacy-question-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / LEGACY.TARGETS[0]
        self.path.write_text(json.dumps({"cells": [
            {"cell_type": "code", "source": ["import math\n", "def example():\n", "    return 1\n"]},
            {"cell_type": "markdown", "source": ["## 参考代码与解析"]},
            {"cell_type": "code", "source": ["def answer():\n", "    return 2\n"]},
        ]}), encoding="utf-8")
        self.original = self.path.read_bytes()

    def test_old_helper_defaults_to_preview(self):
        """Calling the previous helper name must no longer overwrite content."""
        self.assertTrue(LEGACY._rewrite_notebook(self.path))
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_explicit_write_is_rejected_before_reading(self):
        """A write request fails before any file access or modification."""
        with self.assertRaises(ValueError):
            LEGACY._rewrite_notebook(self.root / "absent.ipynb", dry_run=False)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_main_only_reports_planned_changes(self):
        """The default CLI path remains read-only even for matching content."""
        with patch.object(LEGACY, "NOTEBOOK_DIR", self.root), \
                patch.object(sys, "argv", [str(SCRIPT), "--file", self.path.name]), \
                patch("builtins.print") as output:
            LEGACY.main()
        output.assert_any_call("would change 1 notebooks; wrote 0")
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_cli_refuses_write_option(self):
        """A real subprocess must fail a write request before selecting files."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--write"], cwd=self.root,
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("已冻结写入", result.stderr)
        self.assertEqual(self.path.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
