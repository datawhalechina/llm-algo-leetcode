"""Check collection and legacy entrypoints for the split project contract tests."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tests import test_part02_project_structure as part02


class ProjectContractEntrypoints(unittest.TestCase):
    """Run source checks in child processes without invoking experiments."""

    def run_python(self, arguments, cwd=ROOT):
        return subprocess.run([sys.executable, "-S", *map(str, arguments)], cwd=cwd,
                              capture_output=True, text=True, timeout=30)

    def assert_count(self, result, expected):
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        match = re.search(r"Ran (\d+) tests? in", output)
        self.assertIsNotNone(match, output)
        self.assertEqual(int(match.group(1)), expected, output)

    def test_legacy_script_and_module_run_all_ten_checks(self):
        self.assert_count(self.run_python([ROOT / "tools/test_project_contracts.py"]), 10)
        self.assert_count(self.run_python(["-m", "unittest", "tools.test_project_contracts"]), 10)

    def test_focused_scripts_work_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            for filename, count in (("test_project_contracts.py", 2),
                                    ("test_part02_project_structure.py", 6),
                                    ("test_quantization_project_contracts.py", 2)):
                with self.subTest(filename=filename):
                    self.assert_count(self.run_python([ROOT / "tools/tests" / filename], cwd=temporary), count)
            self.assert_count(self.run_python([ROOT / "tools/test_project_contracts.py"], cwd=temporary), 10)

    def test_discovery_does_not_collect_the_legacy_wrapper_twice(self):
        for start in ("tools", "tools/tests"):
            self.assert_count(self.run_python(["-m", "unittest", "discover", "-s", start,
                                               "-p", "test_project_contracts.py"]), 2)

    def test_every_tagged_project_is_actually_checked(self):
        expected = {
            "66_Inference_Performance_Comparison.ipynb", "60_Decoder_Block_Stability_Benchmark.ipynb",
            "78_Attention_Access_Pattern_Benchmark.ipynb", "71_MLA_KV_Cache_Architecture_Benchmark.ipynb",
            "77_Long_Sequence_Memory_Architecture_Benchmark.ipynb", "61_Model_Architecture_Exploration.ipynb",
            "82_Quantization_Artifact_Evaluation_Project.ipynb", "67_Quantized_Inference_and_Deployment.ipynb",
            "83_Quantized_Serving_Profile_Benchmark.ipynb", "88_Distilled_Student_Delivery_Project.ipynb",
        }
        with patch.object(part02, "_assert_tagged_project_step5_contract",
                          wraps=part02._assert_tagged_project_step5_contract) as check:
            part02.Part02ProjectStructureTests().test_tagged_project_layouts()
        self.assertEqual(check.call_count, 10)
        self.assertEqual({call.args[1] for call in check.call_args_list}, expected)

    def test_performance_test_imports_do_not_shadow_canonical_contract_tests(self):
        probe = """
import sys, unittest
sys.path.insert(0, sys.argv[1])
import tools.tests.test_performance_analysis_docs
import tools.tests.test_performance_result_readers
suite = unittest.defaultTestLoader.discover(sys.argv[1] + '/tools/tests', pattern='test_project_contracts.py')
result = unittest.TextTestRunner().run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
        self.assert_count(self.run_python(["-c", probe, ROOT]), 2)

    def test_failed_assertion_propagates_through_both_script_entries(self):
        probe = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
import tools.project_contracts as contracts
contracts.validate_run_result = lambda _: ['deliberate_failure']
entry = sys.argv[2]
sys.argv = [entry]
runpy.run_path(entry, run_name='__main__')
"""
        for entry in (ROOT / "tools/test_project_contracts.py", ROOT / "tools/tests/test_project_contracts.py"):
            result = self.run_python(["-c", probe, ROOT, entry])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAILED (failures=1)", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
