"""Check CLI compatibility and test discovery for the first tools migration."""

import ast
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "tools/tests/test_foundation_performance_evidence.py"
LEGACY = ROOT / "tools/test_foundation_performance_evidence.py"
TEST_CLASS = "FoundationPerformanceTests"
SINGLE_TEST = f"{TEST_CLASS}.test_shape_keys_do_not_claim_actual_compilation"


class TestEntrypointCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = ast.parse(CANONICAL.read_text(encoding="utf-8"))
        case = next(node for node in module.body
                    if isinstance(node, ast.ClassDef) and node.name == TEST_CLASS)
        cls.expected_count = sum(
            isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            for node in case.body
        )
        if cls.expected_count == 0:
            raise AssertionError("The canonical suite must contain test cases")

    def run_python(self, arguments, cwd=ROOT):
        return subprocess.run(
            [sys.executable, *map(str, arguments)], cwd=cwd,
            capture_output=True, text=True, timeout=20,
        )

    def assert_passed_count(self, result, expected):
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        match = re.search(r"Ran (\d+) tests? in", output)
        self.assertIsNotNone(match, output)
        self.assertEqual(int(match.group(1)), expected, output)

    def test_new_and_old_script_commands_run_the_same_suite(self):
        for entry in (CANONICAL, LEGACY):
            with self.subTest(entry=entry):
                self.assert_passed_count(self.run_python([entry]), self.expected_count)

    def test_source_root_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="tutorial-entrypoint-") as temporary:
            for entry in (CANONICAL, LEGACY):
                with self.subTest(entry=entry):
                    result = self.run_python([entry], cwd=temporary)
                    self.assert_passed_count(result, self.expected_count)

    def test_discovery_from_both_directories_collects_each_case_once(self):
        for start in ("tools", "tools/tests"):
            with self.subTest(start=start):
                result = self.run_python([
                    "-m", "unittest", "discover", "-s", start,
                    "-p", CANONICAL.name,
                ])
                self.assert_passed_count(result, self.expected_count)

    def test_legacy_launcher_preserves_test_selection_arguments(self):
        for entry in (CANONICAL, LEGACY):
            with self.subTest(entry=entry):
                self.assert_passed_count(self.run_python([entry, SINGLE_TEST]), 1)

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
        for entry in (CANONICAL, LEGACY):
            with self.subTest(entry=entry):
                result = self.run_python(["-c", probe, entry, SINGLE_TEST])
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn("deliberate_entrypoint_assertion_probe", output)
                self.assertIn("FAILED (failures=1)", output)


if __name__ == "__main__":
    unittest.main()
