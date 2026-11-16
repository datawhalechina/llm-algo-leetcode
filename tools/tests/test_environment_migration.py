"""Check environment entrypoints without installing or registering a real environment."""

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

NAMES = ("install_environment_profile", "environment_preflight", "register_notebook_runtime")


class EnvironmentMigrationTests(unittest.TestCase):
    def run_python(self, arguments, directory=ROOT):
        return subprocess.run(
            [sys.executable, "-S", *map(str, arguments)], cwd=directory,
            capture_output=True, text=True, timeout=15,
        )

    def entries(self, name):
        return [
            [ROOT / "tools" / f"{name}.py"],
            [ROOT / "tools/environment" / f"{name}.py"],
            ["-m", f"tools.{name}"],
            ["-m", f"tools.environment.{name}"],
        ]

    def test_imports_share_one_implementation(self):
        for name in NAMES:
            self.assertIs(importlib.import_module(f"tools.{name}"),
                          importlib.import_module(f"tools.environment.{name}"))
        registration = importlib.import_module("tools.environment.register_notebook_runtime")
        self.assertEqual(registration.repository_root(), ROOT)

    def test_bare_imports_work_outside_repository(self):
        probe = (
            "import importlib, sys; "
            f"sys.path.insert(0, {str(ROOT / 'tools')!r}); "
            "old = importlib.import_module(sys.argv[1]); "
            "new = importlib.import_module('tools.environment.' + sys.argv[1]); "
            "assert old is new"
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in NAMES:
                result = self.run_python(["-c", probe, name], directory)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_supports_script_and_module_modes(self):
        for name in NAMES:
            for entry in self.entries(name):
                result = self.run_python([*entry, "--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_direct_script_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in NAMES:
                for entry in self.entries(name)[:2]:
                    result = self.run_python([*entry, "--help"], directory)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_profile_listing_matches_without_hardware_probe(self):
        results = []
        for entry in self.entries("environment_preflight"):
            result = self.run_python([*entry, "--list-profiles"])
            self.assertEqual(result.returncode, 0, result.stderr)
            results.append(json.loads(result.stdout))
        self.assertTrue(results[0])
        self.assertTrue(all(value == results[0] for value in results))

    def test_registration_only_writes_the_supplied_temporary_directory(self):
        for entry in self.entries("register_notebook_runtime"):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "site-packages"
                registration = target / "llm_algo_leetcode_repo.pth"
                result = self.run_python([*entry, "--check", "--site-packages", target])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(target.exists())
                result = self.run_python([*entry, "--site-packages", target])
                self.assertEqual(result.returncode, 0, result.stderr)
                before = registration.read_bytes()
                self.assertEqual(before.decode(), f"{ROOT}\n")
                report = json.loads(result.stdout)
                self.assertIsNone(report["subdirectory_import"])
                result = self.run_python([*entry, "--check", "--site-packages", target])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(registration.read_bytes(), before)

    def test_installer_dry_run_does_not_execute_subprocesses(self):
        module = importlib.import_module("tools.environment.install_environment_profile")
        profile = {"title": "fixture", "requirements": [], "constraint_files": []}
        with patch.object(sys, "argv", ["installer", "--profile", "FIXTURE", "--env-name", "fixture", "--dry-run", "--no-kernel"]), \
                patch.object(module, "bootstrap_project_root", return_value=ROOT), \
                patch.object(module, "load_environment_profile", return_value=profile), \
                patch.object(module.shutil, "which", return_value="/fake/conda"), \
                patch.object(module.subprocess, "run", side_effect=AssertionError("must not execute")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(module.main(), 0)
            self.assertIn("conda create", output.getvalue())
            self.assertIn("未创建或预检环境", output.getvalue())

    def test_preflight_passes_options_and_preserves_blocked_exit_code(self):
        module = importlib.import_module("tools.environment.environment_preflight")
        for status, expected in (("ok", 0), ("blocked", 2)):
            report = {"status": status, "ready": status == "ok", "reasons": [], "next_actions": []}
            with patch.object(sys, "argv", ["preflight", "--gpu", "--packages", "example"]), \
                    patch.dict(sys.modules, {"torch": None}), \
                    patch.object(module, "bootstrap_project_root", return_value=ROOT), \
                    patch.object(module, "read_exact_constraints", return_value={}), \
                    patch.object(module, "environment_preflight", return_value=report) as check, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), expected)
                self.assertIsNone(check.call_args.args[0])
                self.assertTrue(check.call_args.kwargs["require_gpu"])
                self.assertEqual(check.call_args.kwargs["required_packages"], ("example",))

    def test_legacy_entrypoints_preserve_failure_exit_codes(self):
        probe = (
            "import importlib, runpy, sys; "
            "module = importlib.import_module('tools.environment.' + sys.argv[1]); "
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
