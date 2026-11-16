"""Check classified implementations and legacy imports without Node or GPU."""

from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ChecksMigrationTests(unittest.TestCase):
    def test_legacy_and_classified_imports_share_one_module(self):
        for name in ("check_source_docs_mirror", "check_mermaid_blocks"):
            with self.subTest(name=name):
                old = importlib.import_module(f"tools.{name}")
                new = importlib.import_module(f"tools.checks.{name}")
                self.assertIs(old, new)
                self.assertEqual(new.ROOT, ROOT)

    def test_bare_legacy_import_preserves_module_identity(self):
        probe = (
            "import importlib, sys; "
            f"sys.path.insert(0, {str(ROOT / 'tools')!r}); "
            "old = importlib.import_module(sys.argv[1]); "
            "new = importlib.import_module('tools.checks.' + sys.argv[1]); "
            "assert old is new"
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in ("check_source_docs_mirror", "check_mermaid_blocks"):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", probe, name], cwd=directory,
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_and_new_mirror_commands_work_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for folder in ("tools", "tools/checks"):
                result = subprocess.run(
                    [sys.executable, "-S", str(ROOT / folder / "check_source_docs_mirror.py")],
                    cwd=directory, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertIn("Source/docs mirror check passed.", result.stdout)

    def test_missing_mirror_failure_reaches_legacy_main(self):
        old = importlib.import_module("tools.check_source_docs_mirror")
        new = importlib.import_module("tools.checks.check_source_docs_mirror")
        with patch.object(new, "iter_source_pages", return_value={Path("docs/missing.md")}), \
                patch.object(new, "iter_docs_pages", return_value=set()), \
                patch.object(sys, "argv", ["check_source_docs_mirror.py"]), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(old.main(), 1)

    def test_mermaid_extracts_blocks_and_uses_repository_node_directory(self):
        module = importlib.import_module("tools.checks.check_mermaid_blocks")
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "fixture.md"
            page.write_text("# Example\n```mermaid\nflowchart LR\nA --> B\n```\n")
            blocks = module.extract_blocks(page)
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].start_line, 3)
            completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
            with patch.object(module.subprocess, "run", return_value=completed) as run:
                self.assertEqual(module.parse_with_mermaid(blocks[0]), (True, ""))
            self.assertEqual(run.call_args.kwargs["cwd"], ROOT / "docs")
            self.assertEqual(run.call_args.kwargs["env"]["MERMAID_TEXT"], "flowchart LR\nA --> B")

    def test_mermaid_parse_failure_remains_nonzero(self):
        module = importlib.import_module("tools.checks.check_mermaid_blocks")
        block = module.Block(ROOT / "README.md", 3, "invalid diagram")
        completed = SimpleNamespace(returncode=1, stdout="", stderr="parse failure")
        with patch.object(module.subprocess, "run", return_value=completed), \
                patch.object(module, "MARKDOWN_FILES", [ROOT / "README.md"]), \
                patch.object(module, "extract_blocks", return_value=[block]), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 1)

    def test_legacy_cli_propagates_main_exit_status(self):
        probe = (
            "import importlib, runpy, sys; "
            "module = importlib.import_module('tools.checks.' + sys.argv[1]); "
            "module.main = lambda: int(sys.argv[2]); "
            f"runpy.run_path(str(__import__('pathlib').Path({str(ROOT)!r}) / 'tools' / "
            "(sys.argv[1] + '.py')), run_name='__main__')"
        )
        for name in ("check_source_docs_mirror", "check_mermaid_blocks"):
            for code in (0, 1):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", probe, name, str(code)],
                    cwd=ROOT, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, code, result.stderr)


if __name__ == "__main__":
    unittest.main()
