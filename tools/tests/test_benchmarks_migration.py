"""Check experiment entry migration with temporary JSON and mocked HTTP only."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMES = ("aggregate_distributed_evidence", "benchmark_inference_backend")


class BenchmarksMigrationTests(unittest.TestCase):
    def run_python(self, arguments, directory=ROOT):
        return subprocess.run(
            [sys.executable, "-S", *map(str, arguments)], cwd=directory,
            capture_output=True, text=True, timeout=15,
        )

    def entries(self, name):
        return [
            [str(ROOT / "tools" / f"{name}.py")],
            [str(ROOT / "tools/benchmarks" / f"{name}.py")],
            ["-m", f"tools.{name}"],
            ["-m", f"tools.benchmarks.{name}"],
        ]

    def test_imports_share_one_implementation(self):
        for name in NAMES:
            self.assertIs(importlib.import_module(f"tools.{name}"),
                          importlib.import_module(f"tools.benchmarks.{name}"))

    def test_bare_imports_work_outside_repository(self):
        probe = (
            "import importlib, sys; "
            f"sys.path.insert(0, {str(ROOT / 'tools')!r}); "
            "old = importlib.import_module(sys.argv[1]); "
            "new = importlib.import_module('tools.benchmarks.' + sys.argv[1]); "
            "assert old is new"
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in NAMES:
                result = self.run_python(["-c", probe, name], directory)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_works_for_all_entry_modes(self):
        for name in NAMES:
            for entry in self.entries(name):
                result = self.run_python([*entry, "--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output", result.stdout)

    def test_direct_script_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in NAMES:
                for entry in self.entries(name)[:2]:
                    result = self.run_python([*entry, "--help"], directory)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_aggregate_preserves_results_and_relative_output_paths(self):
        documents = []
        for entry in self.entries("aggregate_distributed_evidence"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = {
                    "workload": {"fixture": "same"}, "hardware": {}, "metrics": {},
                    "evidence_level": "real_multi_gpu_smoke", "failure": None,
                    "decision": "measured",
                }
                input_file = root / "input.json"
                input_file.write_text(json.dumps(source))
                before = input_file.read_bytes()
                # Module execution also uses the temporary cwd via an explicit root import.
                if entry[0] == "-m":
                    probe = (
                        f"import runpy, sys; sys.path.insert(0, {str(ROOT)!r}); "
                        "module = sys.argv[1]; sys.argv = sys.argv[1:]; "
                        "runpy.run_module(module, run_name='__main__')"
                    )
                    command = ["-c", probe, entry[1]]
                else:
                    command = entry
                result = self.run_python([
                    *command, "--project", "79", "--input", "input.json",
                    "--output", "nested/report.json",
                ], root)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((root / "nested/report.json").read_text())
                self.assertEqual(report, json.loads(result.stdout))
                self.assertEqual(report["decision"]["decision"], "tune")
                self.assertEqual(report["role"], "training_state_parallel")
                self.assertEqual(report["sources"][0], {"path": "input.json", **source})
                self.assertEqual(input_file.read_bytes(), before)
                documents.append(report)
        self.assertTrue(all(item == documents[0] for item in documents))

    def test_unreadable_sources_are_reported_without_fabricating_success(self):
        module = importlib.import_module("tools.benchmarks.aggregate_distributed_evidence")
        self.assertEqual(module._decision([])[0], "not_started")
        for record in ({"failure": "stopped", "evidence_level": "real_benchmark"},
                       {"failure": None, "evidence_level": "real_multi_gpu_smoke"}):
            self.assertEqual(module._decision([record])[0], "tune")
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_python([
                ROOT / "tools/aggregate_distributed_evidence.py", "--project", "81",
                "--input", "absent.json", "--output", "result.json",
            ], directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["decision"]["decision"], "tune")
            self.assertEqual(report["sources"], [])
            self.assertIn("absent.json", report["failure"]["reason"])

    def test_client_entrypoints_write_raw_and_normalized_mocked_reports(self):
        probe = f"""
import json, runpy, sys
from unittest.mock import patch
sys.path.insert(0, {str(ROOT)!r})
mode, target = sys.argv[1:3]
sys.argv = [target, *sys.argv[3:]]
class Response:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __iter__(self):
        payload = {{"choices": [{{"delta": {{"content": "Answer: 42"}}, "finish_reason": "stop"}}],
                   "usage": {{"prompt_tokens": 6, "completion_tokens": 4}}}}
        return iter([("data: " + json.dumps(payload) + "\\n").encode(), b"data: [DONE]\\n"])
with patch("urllib.request.urlopen", return_value=Response()) as request:
    try:
        if mode == "module": runpy.run_module(target, run_name="__main__")
        else: runpy.run_path(target, run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0): raise
    request.assert_called_once()
    body = json.loads(request.call_args.args[0].data)
    assert body["model"] == "fixture-model"
    assert body["messages"][0]["content"] == "Add 35 and 7"
"""
        for entry in self.entries("benchmark_inference_backend"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workload = root / "workload.jsonl"
                workload.write_text(json.dumps({"request_id": "r1", "prompt": "Add 35 and 7", "expected_answer": "42"}) + "\n")
                mode, target = ("module", entry[1]) if entry[0] == "-m" else ("script", entry[0])
                result = self.run_python([
                    "-c", probe, mode, target, "--model", "fixture-model", "--workload", workload,
                    "--num-prompts", "1", "--warmup", "0", "--include-requests",
                    "--output", "nested/client.json",
                ], root)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((root / "nested/client.json").read_text())
                self.assertEqual(report, json.loads(result.stdout))
                self.assertEqual(report["metrics"]["successful_requests"], 1)
                self.assertEqual(report["results"][0]["output_text"], "Answer: 42")
                self.assertEqual(report["normalized_result"]["metrics"]["successful_requests"], 1)

    def test_invalid_repeat_arguments_fail_before_requests(self):
        for entry in self.entries("benchmark_inference_backend"):
            result = self.run_python([*entry, "--model", "fixture", "--repeats", "0"])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--repeats", result.stderr)

    def test_legacy_entrypoints_propagate_failure_exit_codes(self):
        probe = (
            "import importlib, runpy, sys; "
            "module = importlib.import_module('tools.benchmarks.' + sys.argv[1]); "
            "module.main = lambda: int(sys.argv[2]); "
            f"runpy.run_path(str(__import__('pathlib').Path({str(ROOT)!r}) / 'tools' / "
            "(sys.argv[1] + '.py')), run_name='__main__')"
        )
        for name in NAMES:
            for code in (0, 2):
                result = self.run_python(["-c", probe, name, str(code)])
                self.assertEqual(result.returncode, code, result.stderr)


if __name__ == "__main__":
    unittest.main()
