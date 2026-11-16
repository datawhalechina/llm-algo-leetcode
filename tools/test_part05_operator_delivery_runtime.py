"""CPU-only decision tests for the Part 05 operator delivery project."""

from __future__ import annotations

import unittest
import hashlib
import json
import statistics
import tempfile
import weakref
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from tools.part05_operator_delivery_runtime import (
    SEMANTIC_ID, VARIANTS, evaluate_measurements, validate_protocol,
    close_operator_delivery, run_operator_delivery, _measure_pair,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load(
    (ROOT / "05_Integrated_Projects_and_Decisions/05_Operator_Optimization_Delivery/project_config.example.yaml").read_text(encoding="utf-8")
)


def measurement(sequence: int, speedup: float = 1.06) -> dict:
    return {
        "sequence_length": sequence,
        "max_absolute_error": 0.001,
        "max_relative_error": 0.01,
        "max_scaled_error": 0.1,
        "numerical_close": True,
        "speedup_ratio": speedup,
        "torch_sdpa_auto": {
            "median_latency_ms": speedup,
            "repeat_latency_ms": [speedup] * PROTOCOL["workload"]["repeats"],
            "repeat_execution_order": [list(VARIANTS if index % 2 == 0 else reversed(VARIANTS)) for index in range(PROTOCOL["workload"]["repeats"])],
            "peak_allocated_mb": 1000,
            "profile": {"trace_path": "traces/sdpa.json", "trace_sha256": "a" * 64, "sdpa_dispatch_events": ["aten::_scaled_dot_product_flash_attention"]},
        },
        "triton_flash_attention": {
            "median_latency_ms": 1.0,
            "repeat_latency_ms": [1.0] * PROTOCOL["workload"]["repeats"],
            "repeat_execution_order": [list(VARIANTS if index % 2 == 0 else reversed(VARIANTS)) for index in range(PROTOCOL["workload"]["repeats"])],
            "peak_allocated_mb": 900,
            "profile": {"trace_path": "traces/triton.json", "trace_sha256": "b" * 64, "triton_kernel_observed": True},
        },
    }


def record(rows=None):
    """Build a complete synthetic record for CPU-only gate checks."""
    return {
        "schema_version": "part05-operator-delivery/v1", "semantic_id": SEMANTIC_ID,
        "status": "complete",
        "run_label": "unit", "protocol": deepcopy(PROTOCOL),
        "measurements": rows if rows is not None else [measurement(n) for n in PROTOCOL["workload"]["sequence_lengths"]],
    }


class OperatorDeliveryProtocolTests(unittest.TestCase):
    def test_registered_protocol_is_valid(self) -> None:
        self.assertEqual(validate_protocol(PROTOCOL), [])

    def test_rejects_a_different_shape_matrix(self) -> None:
        changed = {**PROTOCOL, "workload": {**PROTOCOL["workload"], "sequence_lengths": [128, 512]}}
        self.assertIn("sequence_lengths must be [128, 512, 1024, 2048]", validate_protocol(changed))

    def test_complete_evidence_can_pass(self) -> None:
        decision = evaluate_measurements(record(), PROTOCOL)
        self.assertEqual(decision["decision"], "accept")
        self.assertTrue(all(decision["gates"].values()))

    def test_valid_but_slow_candidate_is_tune(self) -> None:
        rows = [measurement(n, speedup=1.01 if n >= 1024 else 0.95) for n in PROTOCOL["workload"]["sequence_lengths"]]
        decision = evaluate_measurements(record(rows), PROTOCOL)
        self.assertEqual(decision["decision"], "tune")
        self.assertFalse(decision["gates"]["long_sequence_speedup"])

    def test_missing_measurement_is_pending(self) -> None:
        decision = evaluate_measurements(record([measurement(128)]), PROTOCOL)
        self.assertEqual(decision["decision"], "pending")

    def test_missing_candidate_trace_is_rejected(self) -> None:
        rows = [measurement(n) for n in PROTOCOL["workload"]["sequence_lengths"]]
        rows[0]["triton_flash_attention"]["profile"]["triton_kernel_observed"] = False
        decision = evaluate_measurements(record(rows), PROTOCOL)
        self.assertEqual(decision["decision"], "reject")

    def test_invalid_protocol_is_rejected(self):
        for section, key, value in (
            ("workload", "repeats", 0), ("workload", "repeats", 3),
            ("workload", "batch_size", True), ("workload", "hidden_size", 513),
            ("workload", "num_key_value_heads", 4), ("workload", "warmup", -1),
            ("gates", "maximum_peak_memory_gib", float("nan")),
            ("gates", "minimum_speedup_long_sequence", float("inf")),
            ("gates", "short_sequence_lengths", [512]),
        ):
            with self.subTest(section=section, key=key):
                protocol = deepcopy(PROTOCOL)
                protocol[section][key] = value
                self.assertTrue(validate_protocol(protocol))
        self.assertTrue(validate_protocol(None))
        self.assertTrue(validate_protocol({}))

    def test_duplicate_shape_cannot_count_as_complete(self):
        raw = record()
        raw["measurements"].append(measurement(128))
        self.assertEqual(evaluate_measurements(raw, PROTOCOL)["decision"], "pending")

    def test_bad_metrics_are_pending(self):
        for field in ("speedup_ratio", "max_absolute_error", "max_relative_error", "max_scaled_error"):
            for value in (float("nan"), float("inf"), -1, True, "invalid"):
                with self.subTest(field=field, value=value):
                    raw = record()
                    raw["measurements"][0][field] = value
                    self.assertEqual(evaluate_measurements(raw, PROTOCOL)["decision"], "pending")

    def test_inconsistent_speedup_is_pending(self):
        raw = record()
        raw["measurements"][0]["speedup_ratio"] = 100
        self.assertEqual(evaluate_measurements(raw, PROTOCOL)["decision"], "pending")

    def test_gate_uses_median_of_paired_speedups(self):
        raw = record()
        row = raw["measurements"][0]
        left = [1, 2, 100, 1, 2, 100, 1, 2, 100, 1]
        right = [1, 1, 10, 10, 1, 1, 10, 10, 1, 1]
        for variant, samples in zip(VARIANTS, (left, right)):
            row[variant]["repeat_latency_ms"] = samples
            row[variant]["median_latency_ms"] = statistics.median(samples)
        row["speedup_ratio"] = statistics.median(a / b for a, b in zip(left, right))
        self.assertTrue(evaluate_measurements(raw, PROTOCOL)["gates"]["measurement_complete"])
        row["speedup_ratio"] = statistics.median(left) / statistics.median(right)
        self.assertEqual(evaluate_measurements(raw, PROTOCOL)["decision"], "pending")

    def test_missing_repeat_or_unbalanced_order_is_pending(self):
        for field, value in (("repeat_latency_ms", [1]), ("repeat_execution_order", []), ("median_latency_ms", 99)):
            raw = record()
            raw["measurements"][0][VARIANTS[0]][field] = value
            self.assertEqual(evaluate_measurements(raw, PROTOCOL)["decision"], "pending")

    def test_scaled_error_over_tolerance_is_rejected(self):
        raw = record()
        raw["measurements"][0]["max_scaled_error"] = 1.01
        self.assertEqual(evaluate_measurements(raw, PROTOCOL)["decision"], "reject")

    def test_generic_sdpa_wrapper_does_not_prove_backend(self):
        raw = record()
        raw["measurements"][0][VARIANTS[0]]["profile"]["sdpa_dispatch_events"] = ["aten::scaled_dot_product_attention"]
        self.assertEqual(evaluate_measurements(raw, PROTOCOL)["decision"], "reject")

    def test_changed_protocol_cannot_reuse_measurements(self):
        protocol = deepcopy(PROTOCOL)
        protocol["gates"]["minimum_speedup_long_sequence"] = 0.01
        with self.assertRaises(ValueError):
            evaluate_measurements(record(), protocol)

    def test_invalid_label_stops_before_cuda(self):
        torch = Mock()
        for label in (None, "../outside", "replace_with_unique_label"):
            with self.assertRaises(ValueError):
                run_operator_delivery(torch, PROTOCOL, ROOT, label)
        torch.cuda.is_available.assert_not_called()

    def _persisted_case(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        raw = record()
        run_dir = root / PROTOCOL["outputs"]["directory"] / raw["run_label"]
        traces = run_dir / "traces"
        traces.mkdir(parents=True)
        for row in raw["measurements"]:
            for variant in VARIANTS:
                path = traces / f"{variant}_{row['sequence_length']}.json"
                name = "aten::_scaled_dot_product_flash_attention" if variant == VARIANTS[0] else "flash_attn_fwd_kernel"
                path.write_text(json.dumps({"traceEvents": [{"name": name}]}), encoding="utf-8")
                profile = row[variant]["profile"]
                profile["trace_path"] = str(path.relative_to(root))
                profile["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw["result_path"] = str((run_dir / "raw_result.json").relative_to(root))
        (run_dir / "raw_result.json").write_text(json.dumps(raw), encoding="utf-8")
        return root, raw, run_dir

    def test_close_checks_real_trace_files_and_is_idempotent(self):
        root, raw, run_dir = self._persisted_case()
        first = close_operator_delivery(root, raw["run_label"], PROTOCOL)
        self.assertEqual(first["decision"], "accept")
        self.assertEqual(first, close_operator_delivery(root, raw["run_label"], PROTOCOL))

    def test_close_rejects_missing_or_changed_trace(self):
        for missing in (True, False):
            root, raw, run_dir = self._persisted_case()
            path = root / raw["measurements"][0][VARIANTS[1]]["profile"]["trace_path"]
            if missing:
                path.unlink()
            else:
                path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                close_operator_delivery(root, raw["run_label"], PROTOCOL)

    def test_close_rejects_trace_without_target_event_even_with_valid_hash(self):
        root, raw, run_dir = self._persisted_case()
        profile = raw["measurements"][0][VARIANTS[1]]["profile"]
        path = root / profile["trace_path"]
        path.write_text(json.dumps({"traceEvents": [{"name": "unrelated"}]}), encoding="utf-8")
        profile["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (run_dir / "raw_result.json").write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "目标执行事件"):
            close_operator_delivery(root, raw["run_label"], PROTOCOL)

    def test_close_rejects_trace_outside_current_run(self):
        root, raw, run_dir = self._persisted_case()
        profile = raw["measurements"][0][VARIANTS[1]]["profile"]
        original = root / profile["trace_path"]
        outside = root / "outside.json"
        outside.write_bytes(original.read_bytes())
        profile["trace_path"] = str(outside)
        (run_dir / "raw_result.json").write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "超出本轮目录"):
            close_operator_delivery(root, raw["run_label"], PROTOCOL)

    def test_changed_raw_record_cannot_overwrite_decision(self):
        root, raw, run_dir = self._persisted_case()
        close_operator_delivery(root, raw["run_label"], PROTOCOL)
        raw["additional_information"] = "changed"
        (run_dir / "raw_result.json").write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(FileExistsError):
            close_operator_delivery(root, raw["run_label"], PROTOCOL)

    def test_paired_measurement_discards_outputs_during_timing(self):
        calls = []
        class Output:
            """Track output lifetimes without retaining the measured objects."""
        active = weakref.WeakSet()
        def decode(*args, **kwargs):
            calls.append((kwargs["variant"], len(active)))
            result = Output()
            active.add(result)
            return result
        torch = Mock()
        torch.cuda.max_memory_allocated.return_value = 1024
        with patch("tools.part05_operator_delivery_runtime._decoder_prefill", side_effect=decode), patch(
            "tools.part05_operator_delivery_runtime._profile_once", return_value={}
        ), patch("tools.part05_operator_delivery_runtime.time.perf_counter", side_effect=[0, .002, .003, .004, .005, .006, .007, .009]):
            baseline, candidate = _measure_pair(torch, Mock(device="cuda:0"), [], heads=8, warmup=0, repeats=2,
                                                 trace_dir=Path("unused"), sequence=128)
        self.assertEqual([variant for variant, _ in calls[:4]], [VARIANTS[0], VARIANTS[1], VARIANTS[1], VARIANTS[0]])
        self.assertTrue(all(count == 0 for _, count in calls[:4]))
        self.assertEqual(len(baseline["repeat_latency_ms"]), 2)
        self.assertEqual(len(candidate["repeat_latency_ms"]), 2)


if __name__ == "__main__":
    unittest.main()
