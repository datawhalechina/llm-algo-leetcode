"""Check sidecar conversion safety without touching repository evidence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.normalize_inference_results import normalize_file as normalize_inference
from tools.training_memory_result_schema import normalize_file as normalize_memory


class ResultMigrationSafetyTests(unittest.TestCase):
    """Apply the same input and overwrite protections to both result formats."""

    def setUp(self):
        """Provide minimal historical reports in a disposable directory."""
        temporary = tempfile.TemporaryDirectory(prefix="result-migration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "73_raw.json"
        self.source.write_text(json.dumps({"normalized_result": {"value": 1}}), encoding="utf-8")
        self.original = self.source.read_bytes()

    def convert(self, kind, target=None, *, dry_run=False):
        """Exercise public conversion functions rather than a stand-in writer."""
        if kind == "inference":
            return normalize_inference(self.source, output_path=target, dry_run=dry_run)
        return normalize_memory(self.source, target, dry_run=dry_run)

    def test_default_sidecar_preserves_raw_input(self):
        """Both default paths stay alongside, not on top of, raw measurements."""
        for kind, suffix in (("inference", "_normalized"), ("memory", "_v1")):
            with self.subTest(kind=kind):
                target = self.convert(kind)
                self.assertEqual(target, self.root / f"73_raw{suffix}.json")
                self.assertTrue(target.is_file())
                self.assertEqual(self.source.read_bytes(), self.original)

    def test_legacy_inference_metrics_keep_their_values(self):
        """The safety wrapper must not change the existing conversion mapping."""
        self.source.write_text(json.dumps({
            "model": "fixture-model", "metrics": {
                "e2e_ms": {"p99": 12.5}, "output_token_throughput_per_s": 42,
            },
        }), encoding="utf-8")
        original = self.source.read_bytes()
        report = json.loads(self.convert("inference").read_text(encoding="utf-8"))
        self.assertEqual(report["config"]["model"], "fixture-model")
        self.assertEqual(report["metrics"]["p99_ms"], 12.5)
        self.assertEqual(report["metrics"]["throughput_tokens_per_s"], 42)
        self.assertEqual(self.source.read_bytes(), original)

    def test_legacy_memory_comparison_keeps_its_values(self):
        """Candidate savings and ratios still derive from original measurements."""
        self.source.write_text(json.dumps({
            "baseline": {"peak_memory_mb": 1000, "samples_per_s": 10},
            "tuned": {"peak_memory_mb": 700, "samples_per_s": 8},
        }), encoding="utf-8")
        original = self.source.read_bytes()
        report = json.loads(self.convert("memory").read_text(encoding="utf-8"))
        tuned = next(row for row in report["candidates"] if row["name"] == "tuned")
        self.assertEqual(tuned["metrics"]["memory_saving_mb"], 300)
        self.assertEqual(tuned["metrics"]["throughput_ratio"], 0.8)
        self.assertEqual(self.source.read_bytes(), original)

    def test_identical_repeat_leaves_existing_file_untouched(self):
        """An identical rerun reuses the result without changing its timestamp."""
        for kind in ("inference", "memory"):
            with self.subTest(kind=kind):
                target = self.convert(kind)
                before = (target.read_bytes(), target.stat().st_mtime_ns)
                self.assertEqual(self.convert(kind), target)
                self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)

    def test_conflicting_output_is_preserved(self):
        """Different existing content is never replaced, including in preview."""
        target = self.root / "conflict.json"
        target.write_bytes(b'{"existing": true}\n')
        for kind in ("inference", "memory"):
            for dry_run in (False, True):
                with self.subTest(kind=kind, dry_run=dry_run):
                    with self.assertRaises(FileExistsError):
                        self.convert(kind, target, dry_run=dry_run)
                    self.assertEqual(target.read_bytes(), b'{"existing": true}\n')
                    self.assertEqual(self.source.read_bytes(), self.original)

    def test_input_cannot_be_selected_as_output(self):
        """Explicit output paths cannot bypass raw-evidence protection."""
        for kind in ("inference", "memory"):
            for dry_run in (False, True):
                with self.subTest(kind=kind, dry_run=dry_run):
                    with self.assertRaises(ValueError):
                        self.convert(kind, self.source, dry_run=dry_run)
                    self.assertEqual(self.source.read_bytes(), self.original)

    def test_hard_link_to_input_is_rejected(self):
        """Different path spelling does not make the same input file safe."""
        alias = self.root / "hardlink.json"
        alias.hardlink_to(self.source)
        for kind in ("inference", "memory"):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    self.convert(kind, alias)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_symbolic_output_is_rejected(self):
        """Neither raw aliases nor redirected output links can be written."""
        for name, destination in (("raw_alias", self.source), ("dangling", self.root / "absent.json")):
            alias = self.root / f"{name}.json"
            alias.symlink_to(destination)
            for kind in ("inference", "memory"):
                with self.subTest(kind=kind, name=name):
                    with self.assertRaises(ValueError):
                        self.convert(kind, alias)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse((self.root / "absent.json").exists())

    def test_preview_creates_neither_file_nor_directory(self):
        """Planning a nested output leaves the filesystem unchanged."""
        for kind in ("inference", "memory"):
            with self.subTest(kind=kind):
                target = self.root / kind / "nested" / "report.json"
                self.assertEqual(self.convert(kind, target, dry_run=True), target)
                self.assertFalse(target.parent.exists())

    def test_cli_previews_work_as_scripts_and_modules(self):
        """Both public CLI forms support read-only planning outside the repo."""
        for name, arguments in (
            ("normalize_inference_results", ["--output", str(self.root / "new" / "report.json")]),
            ("normalize_training_memory_results", ["--output-dir", str(self.root / "new")]),
        ):
            for module in (False, True):
                with self.subTest(name=name, module=module):
                    entry = ["-m", f"tools.{name}"] if module else [str(ROOT / "tools" / f"{name}.py")]
                    result = subprocess.run(
                        [sys.executable, *entry, str(self.source), *arguments, "--dry-run"],
                        cwd=ROOT if module else self.root,
                        capture_output=True, text=True, timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("[dry-run]", result.stdout)
                    self.assertFalse((self.root / "new").exists())

    def test_cli_failure_preserves_conflicting_output(self):
        """A conflicting conversion produces a failing process, not success."""
        for name, arguments, output in (
            ("normalize_inference_results", ["--output", str(self.root / "conflict.json")], self.root / "conflict.json"),
            ("normalize_training_memory_results", ["--output-dir", str(self.root)], self.root / "73_raw_v1.json"),
        ):
            with self.subTest(name=name):
                output.write_bytes(b"existing evidence\n")
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / f"{name}.py"), str(self.source), *arguments],
                    cwd=self.root, capture_output=True, text=True, timeout=20,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_bytes(), b"existing evidence\n")
                self.assertEqual(self.source.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
