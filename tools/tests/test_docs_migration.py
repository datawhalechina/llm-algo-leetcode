"""Verify document-tool migration with temporary inputs and output trees."""

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMES = ("convert_notebook", "convert_chapter0_1",
         "sync_docs_navigation", "sync_docs_index")


class DocsMigrationTests(unittest.TestCase):
    def run_python(self, arguments, directory):
        return subprocess.run(
            [sys.executable, "-S", *map(str, arguments)], cwd=directory,
            capture_output=True, text=True, timeout=15,
        )

    def test_legacy_imports_share_classified_modules(self):
        for name in NAMES:
            old = importlib.import_module(f"tools.{name}")
            new = importlib.import_module(f"tools.docs.{name}")
            self.assertIs(old, new)
            if hasattr(new, "ROOT"):
                self.assertEqual(new.ROOT, ROOT)

    def test_bare_imports_remain_available_outside_repository(self):
        probe = (
            "import importlib, sys; "
            f"sys.path.insert(0, {str(ROOT / 'tools')!r}); "
            "old = importlib.import_module(sys.argv[1]); "
            "new = importlib.import_module('tools.docs.' + sys.argv[1]); "
            "assert old is new"
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in NAMES:
                result = self.run_python(["-c", probe, name], directory)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_converter_and_navigation_help_work_in_all_entry_modes(self):
        for name in ("convert_notebook", "sync_docs_navigation"):
            entries = [
                [ROOT / "tools" / f"{name}.py"],
                [ROOT / "tools/docs" / f"{name}.py"],
                ["-m", f"tools.{name}"],
                ["-m", f"tools.docs.{name}"],
            ]
            for entry in entries:
                result = self.run_python([*entry, "--help"], ROOT)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def make_fixture(self, root, part):
        source = root / part
        source.mkdir(parents=True)
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# 01. 示例\n", "**难度：** Easy\n"]},
                {"cell_type": "code", "source": ["raise AssertionError('must not execute')"]},
            ]
        }
        (source / "01_Example.ipynb").write_text(json.dumps(notebook))
        (source / "intro.md").write_text("# Intro\n\n[lesson](./01_Example.ipynb)\n")
        return source

    def output_tree(self, root):
        return {str(p.relative_to(root)): p.read_bytes()
                for p in (root / "docs").rglob("*") if p.is_file()}

    def test_converter_script_outputs_match_in_temporary_roots(self):
        outputs = []
        for folder in ("tools", "tools/docs"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self.make_fixture(root, "05_Integrated_Projects_and_Decisions/01_Project")
                (source / "project_config.example.yaml").write_text("project: fixture\n")
                before = {p: p.read_bytes() for p in source.iterdir()}
                sibling = root / "docs/03_Triton_Kernels/keep.md"
                sibling.parent.mkdir(parents=True)
                sibling.write_text("preserve")
                result = self.run_python([
                    ROOT / folder / "convert_notebook.py", "--dir",
                    "05_Integrated_Projects_and_Decisions",
                ], root)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(sibling.read_text(), "preserve")
                self.assertEqual({p: p.read_bytes() for p in source.iterdir()}, before)
                tree = self.output_tree(root)
                self.assertIn("docs/05_Integrated_Projects_and_Decisions/01_Project/01_Example.md", tree)
                self.assertEqual(tree["docs/05_Integrated_Projects_and_Decisions/01_Project/project_config.example.yaml"], b"project: fixture\n")
                outputs.append(tree)
        self.assertEqual(outputs[0], outputs[1])

    def test_foundation_converter_outputs_match_in_temporary_roots(self):
        outputs = []
        for folder in ("tools", "tools/docs"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_fixture(root, "00_Prerequisites")
                self.make_fixture(root, "01_Hardware_Math_and_Systems")
                result = self.run_python([ROOT / folder / "convert_chapter0_1.py"], root)
                self.assertEqual(result.returncode, 0, result.stderr)
                tree = self.output_tree(root)
                self.assertIn("docs/00_Prerequisites/01_Example.md", tree)
                self.assertIn("docs/01_Hardware_Math_and_Systems/intro.md", tree)
                outputs.append(tree)
        self.assertEqual(outputs[0], outputs[1])

    def test_module_dry_run_does_not_create_docs(self):
        for name in ("tools.convert_notebook", "tools.docs.convert_notebook"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_fixture(root, "02_PyTorch_Algorithms")
                probe = (
                    f"import runpy, sys; sys.path.insert(0, {str(ROOT)!r}); "
                    f"sys.argv = [{name!r}, '--dir', '02_PyTorch_Algorithms', '--dry-run']; "
                    f"runpy.run_module({name!r}, run_name='__main__')"
                )
                result = self.run_python(["-c", probe], root)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("[dry-run]", result.stdout)
                self.assertFalse((root / "docs").exists())

    def test_homepage_sync_uses_patched_paths_through_legacy_import(self):
        old = importlib.import_module("tools.sync_docs_index")
        new = importlib.import_module("tools.docs.sync_docs_index")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readme = root / "README.md"
            output = root / "docs/index.md"
            output.parent.mkdir()
            readme.write_text("# Example\n\n[guide](./docs/guide.md)\n")
            with patch.object(new, "ROOT", root), patch.object(new, "README", readme), \
                    patch.object(new, "DOCS_INDEX", output), redirect_stdout(io.StringIO()):
                self.assertEqual(old.main(), 0)
            self.assertEqual(output.read_text(), "# Example\n\n[guide](./guide.md)\n")

    def test_legacy_entrypoints_propagate_main_exit_codes(self):
        probe = (
            "import importlib, runpy, sys; "
            "module = importlib.import_module('tools.docs.' + sys.argv[1]); "
            "module.main = lambda: int(sys.argv[2]); "
            f"runpy.run_path(str(__import__('pathlib').Path({str(ROOT)!r}) / 'tools' / "
            "(sys.argv[1] + '.py')), run_name='__main__')"
        )
        for name in NAMES:
            for code in (0, 1):
                result = self.run_python(["-c", probe, name, str(code)], ROOT)
                self.assertEqual(result.returncode, code, result.stderr)


if __name__ == "__main__":
    unittest.main()
