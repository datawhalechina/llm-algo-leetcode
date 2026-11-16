from __future__ import annotations

import json
import statistics
import unittest
import tempfile
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

from tools.performance_delivery_benchmark import (
    CANDIDATE_ORDER,
    MODEL_SNAPSHOT,
    REPEATS,
    VARIANTS,
    validate_project_protocol,
    run_benchmark, build_review_config, model_path_for, WORKLOAD_ID,
    _record_for_variant, SEED, _run_one_repeat,
)
from tools.performance_delivery_runtime import sha256_file, review_delivery


class PerformanceDeliveryBenchmarkTests(unittest.TestCase):
    def _protocol(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        return yaml.safe_load((root / "05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/project_config.example.yaml").read_text(encoding="utf-8"))

    def test_candidate_matrix_has_baseline_two_singles_and_composition(self) -> None:
        self.assertEqual(CANDIDATE_ORDER, ["fused_adamw", "activation_checkpoint"])
        self.assertEqual(set(VARIANTS), {"baseline", *CANDIDATE_ORDER, "fused_checkpoint"})
        self.assertEqual(VARIANTS["fused_checkpoint"]["components"], CANDIDATE_ORDER)
        self.assertEqual(REPEATS, 3)
        self.assertEqual(len(MODEL_SNAPSHOT), 40)

    def test_example_protocol_matches_runner_workload_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol_path = root / "05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/project_config.example.yaml"
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is unavailable in this static test environment")
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(validate_project_protocol(protocol), [])

    def test_notebook_code_cells_are_syntactically_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebook_path = root / "05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/Performance_Optimization_Delivery_Project.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{notebook_path.name}:cell-{index}", "exec")

    def test_invalid_protocol_thresholds_do_not_pass(self):
        for metric, value in (("loss", float("nan")), ("peak_memory_mb", float("inf")), ("loss", True)):
            protocol = self._protocol()
            protocol["gates"]["maximum_guardrail_regression_ratio"][metric] = value
            self.assertTrue(validate_project_protocol(protocol))
        self.assertTrue(validate_project_protocol(None))

    def test_bad_labels_and_existing_results_stop_before_preflight(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.performance_delivery_benchmark.preflight") as check:
            root = Path(temporary)
            for label in (None, "../outside", "replace_with_unique_label"):
                with self.assertRaises(ValueError):
                    run_benchmark(None, None, root, label)
            (root / "benchmarks/results/part05_performance_delivery/existing").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_benchmark(None, None, root, "existing")
            check.assert_not_called()

    def _capture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        model = model_path_for(root)
        model.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
            (model / name).write_bytes(b"cpu-fixture")
        identity = {"workload_id": WORKLOAD_ID, "model_revision": MODEL_SNAPSHOT,
                    "model_file_sha256": {name: sha256_file(model / name) for name in ("config.json", "model.safetensors", "tokenizer_config.json")}}
        def sample(torch, model_class, model_path, variant, seed):
            return {"step_time_ms": {"baseline": 100, "fused_adamw": 90, "activation_checkpoint": 105, "fused_checkpoint": 80}[variant],
                    "peak_memory_mb": 1000, "peak_reserved_mb": 1100, "loss": 1.0,
                    "first_loss": 1.0, "tokens_per_s": 100, "seed": seed}
        def profile(torch, model_class, model_path, variant, seed, trace):
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({"traceEvents": [{"name": "fixture"}]}), encoding="utf-8")
            return {"variant": variant, "trace_path": str(trace), "top_device_operators": [{"name": "fixture", "self_device_time_us": 1}]}
        with patch("tools.performance_delivery_benchmark.preflight", return_value={"workload_identity": identity, "model_assets_complete": True}), patch(
            "tools.performance_delivery_benchmark._run_one_repeat", side_effect=sample
        ), patch("tools.performance_delivery_benchmark._profile_variant", side_effect=profile):
            manifest = run_benchmark(None, None, root, "cpu_capture_test")
        return root, manifest

    def test_mock_capture_round_trip_and_composition_reprofile(self):
        root, manifest = self._capture()
        config = build_review_config(self._protocol(), root, "cpu_capture_test")
        report = review_delivery(config, root)
        self.assertEqual(report["decision"], "accept")
        self.assertEqual(report["decision_detail"]["selected_route"], "composition")
        self.assertTrue(any(source["role"] == "project_capture_manifest" for source in report["measurement_sources"]))
        self.assertEqual(len(manifest["repeat_orders"]), REPEATS)

    def test_changed_measurement_or_trace_is_rejected_before_review(self):
        for relative in ("baseline.json", "traces/baseline.json"):
            root, manifest = self._capture()
            path = root / "benchmarks/results/part05_performance_delivery/cpu_capture_test" / relative
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                build_review_config(self._protocol(), root, "cpu_capture_test")

    def test_changed_base_weights_prevent_review(self):
        root, manifest = self._capture()
        (model_path_for(root) / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "基础模型"):
            build_review_config(self._protocol(), root, "cpu_capture_test")

    def test_manifest_identity_and_required_hashes_are_checked(self):
        for key, value in (("run_label", "other"), ("file_sha256", {})):
            root, manifest = self._capture()
            manifest[key] = value
            path = root / "benchmarks/results/part05_performance_delivery/cpu_capture_test/capture_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_review_config(self._protocol(), root, "cpu_capture_test")

    def test_partial_samples_keep_original_pair_ids(self):
        sample = {"step_time_ms": 100, "peak_memory_mb": 1000, "peak_reserved_mb": 1100,
                  "loss": 1, "tokens_per_s": 100, "seed": SEED + 1}
        record = _record_for_variant("fused_adamw", [sample], {"id": "fixture"}, Path("unused"), run_label="partial")
        self.assertEqual(record["samples"][0]["pair_id"], "pair-02")
        self.assertFalse(record["execution_path"]["verified"])

    def test_training_objects_are_released_before_cache_cleanup(self):
        active = weakref.WeakSet()
        class Item:
            pass
        def components(*args):
            model, optimizer = Item(), Item()
            model.config = Mock(vocab_size=10)
            active.update((model, optimizer))
            return model, optimizer
        def cleanup(torch):
            self.assertEqual(len(active), 0)
        torch = Mock()
        torch.cuda.max_memory_allocated.return_value = 1024
        torch.cuda.max_memory_reserved.return_value = 1024
        with patch("tools.performance_delivery_benchmark._load_train_components", side_effect=components), patch(
            "tools.performance_delivery_benchmark._seeded_tokens", return_value=Item()
        ), patch("tools.performance_delivery_benchmark.WARMUP_STEPS", 0), patch(
            "tools.performance_delivery_benchmark.MEASURED_STEPS", 0
        ), patch("tools.performance_delivery_benchmark._release_cuda", side_effect=cleanup) as release:
            # Empty synthetic step arrays deliberately raise, exercising finally.
            with self.assertRaises(statistics.StatisticsError):
                _run_one_repeat(torch, None, Path("unused"), "baseline", SEED)
            release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
