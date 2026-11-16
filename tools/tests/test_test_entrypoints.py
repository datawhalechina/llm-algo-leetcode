"""Check real test counts, CLI compatibility, and discovery after migration."""

import ast
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SUITE_SPECS = (
    ("test_foundation_performance_evidence.py", "FoundationPerformanceTests",
     "test_shape_keys_do_not_claim_actual_compilation"),
    ("test_chapter_link_checks.py", "FoundationLinkTests",
     "test_notebook_prose_is_checked_without_code_or_outputs"),
    ("test_check_svg_arrows.py", "SvgArrowTests", "test_absolute_paths"),
    ("test_svg_layout.py", "SvgLayoutTests", "test_complete_canvas_passes"),
    ("test_math_formula_compatibility.py", "MathCompatibilityTests",
     "test_spaced_formula_and_sequence"),
)


class TestEntrypointCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suites = []
        for filename, class_name, selected in SUITE_SPECS:
            canonical = ROOT / "tools/tests" / filename
            module = ast.parse(canonical.read_text(encoding="utf-8"))
            case = next(node for node in module.body
                        if isinstance(node, ast.ClassDef) and node.name == class_name)
            methods = [node.name for node in case.body
                       if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
            if not methods or selected not in methods:
                raise AssertionError(f"Missing test cases or selected method: {filename}")
            cls.suites.append({
                "canonical": canonical,
                "legacy": ROOT / "tools" / filename,
                "selected": f"{class_name}.{selected}",
                "count": len(methods),
            })

    def run_python(self, arguments, cwd=ROOT):
        return subprocess.run(
            [sys.executable, "-S", *map(str, arguments)], cwd=cwd,
            capture_output=True, text=True, timeout=20,
        )

    def assert_passed_count(self, result, expected):
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        match = re.search(r"Ran (\d+) tests? in", output)
        self.assertIsNotNone(match, output)
        self.assertEqual(int(match.group(1)), expected, output)

    def test_new_and_old_script_commands_run_the_same_suite(self):
        for suite in self.suites:
            for entry in (suite["canonical"], suite["legacy"]):
                with self.subTest(entry=entry):
                    self.assert_passed_count(self.run_python([entry]), suite["count"])

    def test_source_root_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="tutorial-entrypoint-") as temporary:
            for suite in self.suites:
                for entry in (suite["canonical"], suite["legacy"]):
                    with self.subTest(entry=entry):
                        result = self.run_python([entry], cwd=temporary)
                        self.assert_passed_count(result, suite["count"])

    def test_discovery_from_both_directories_collects_each_case_once(self):
        for suite in self.suites:
            for start in ("tools", "tools/tests"):
                with self.subTest(start=start, suite=suite["canonical"].name):
                    result = self.run_python([
                        "-m", "unittest", "discover", "-s", start,
                        "-p", suite["canonical"].name,
                    ])
                    self.assert_passed_count(result, suite["count"])

    def test_explicit_modules_execute_real_tests(self):
        # Migrated check modules use load_tests, not exported TestCase aliases.
        for suite in self.suites[1:]:
            stem = suite["canonical"].stem
            for module in (f"tools.{stem}", f"tools.tests.{stem}"):
                with self.subTest(module=module):
                    result = self.run_python(["-m", "unittest", module])
                    self.assert_passed_count(result, suite["count"])

    def test_math_ci_targets_the_canonical_suite(self):
        workflow = (ROOT / ".github/workflows/test_notebooks.yml").read_text(encoding="utf-8")
        self.assertIn("run: python -m unittest tools.tests.test_math_formula_compatibility", workflow)
        self.assertIn("run: python -m unittest tools.tests.test_test_entrypoints", workflow)
        self.assertNotIn("run: python -m unittest tools.test_math_formula_compatibility\n", workflow)

    def test_legacy_launcher_preserves_test_selection_arguments(self):
        for suite in self.suites:
            for entry in (suite["canonical"], suite["legacy"]):
                with self.subTest(entry=entry):
                    self.assert_passed_count(self.run_python([entry, suite["selected"]]), 1)

    def test_real_assertion_failure_propagates_through_both_entries(self):
        # Change only a child process: no source file or saved result is edited.
        probe = """
import runpy
import sys
import unittest

entry, selected = sys.argv[1:]
sys.argv = [entry, selected]

def fail_assertion(self, *args, **kwargs):
    raise self.failureException('deliberate_entrypoint_assertion_probe')

unittest.TestCase.assertEqual = fail_assertion
runpy.run_path(entry, run_name='__main__')
"""
        for suite in self.suites:
            for entry in (suite["canonical"], suite["legacy"]):
                with self.subTest(entry=entry):
                    result = self.run_python(["-c", probe, entry, suite["selected"]])
                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn("deliberate_entrypoint_assertion_probe", output)
                    self.assertIn("FAILED (failures=1)", output)


if __name__ == "__main__":
    unittest.main()
