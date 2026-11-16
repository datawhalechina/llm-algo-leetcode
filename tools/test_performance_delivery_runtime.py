from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.performance_delivery_runtime import (
    CONFIG_SCHEMA,
    MEASUREMENT_SCHEMA,
    SEMANTIC_ID,
    evaluate_performance_candidate,
    review_delivery,
    validate_composition_order,
    validate_delivery_config,
    decide_delivery,
    save_json_record_once,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sample_measurement(workload: dict, values: list[tuple[float, float, float]], *, components=None) -> dict:
    record = {
        "schema_version": MEASUREMENT_SCHEMA,
        "workload_identity": workload,
        "execution_path": {"verified": True, "evidence_path": "trace.json"},
        "correctness": {"passed": True, "evidence_path": "correctness.json"},
        "samples": [
            {
                "pair_id": f"pair-{index + 1}",
                "metrics": {"step_time_ms": elapsed, "peak_memory_mb": memory, "loss": loss},
            }
            for index, (elapsed, memory, loss) in enumerate(values)
        ],
    }
    if components is not None:
        record["components"] = components
    return record


def sample_config(root: Path, workload: dict, *, label: str = "independent-project-test") -> dict:
    return {
        "schema_version": CONFIG_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "environment_profile": "PERF-DELIVERY",
        "run_label": label,
        "workload_identity": workload,
        "baseline_measurement": {"result_path": "baseline.json"},
        "candidate_order": ["fused_optimizer"],
        "candidates": {"fused_optimizer": {"result_path": "candidate.json", "conflicts_with": []}},
        "composition": {"result_path": "composition.json"},
        "reprofile": {"result_path": "reprofile.json"},
        "rollback": {"ready": True, "evidence_path": "rollback.json"},
        "gates": {
            "primary_metric": "step_time_ms", "primary_direction": "lower",
            "minimum_primary_improvement_ratio": 0.05,
            "maximum_guardrail_regression_ratio": {"peak_memory_mb": 0.01, "loss": 0.01},
            "minimum_samples": 3, "require_reprofile": True,
            "require_rollback_ready": True,
        },
        "outputs": {"directory": "results"},
        "context_sources": [],
        "capture_status": "complete",
    }


class PerformanceDeliveryRuntimeTests(unittest.TestCase):
    def test_config_and_review_do_not_require_part02_handoff(self) -> None:
        workload = {"family": "training", "workload_id": "part05-v1", "model": "tiny"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = sample_config(root, workload)
            config["candidate_order"] = []
            config["candidates"] = {}
            config["baseline_measurement"]["result_path"] = ""
            config["rollback"] = {"ready": False, "evidence_path": ""}
            config["context_sources"] = [{"label": "Part 02 · 74", "path": "benchmarks/results/74_not_a_gate.json"}]
            self.assertEqual(validate_delivery_config(config, require_evidence_paths=False), [])

            result = review_delivery(config, root)
            self.assertEqual(result["decision"], "tune")
            self.assertIn("project_baseline_measurement", result["decision_detail"]["missing_evidence"])
            self.assertIn("no_project_candidates_configured", result["decision_detail"]["missing_evidence"])
            self.assertNotIn("no_qualified_upstream_candidates", result["decision_detail"]["missing_evidence"])
            self.assertEqual(result["context_sources"][0]["status"], "not_provided")

    def test_paired_measurements_pass_and_measure_guardrails(self) -> None:
        workload = {
            "family": "training", "workload_id": "train-v1", "model": "tiny",
            "input": {"batch_size": 1}, "hardware": "gpu0", "runtime": {"torch": "2.x"},
        }
        baseline = sample_measurement(workload, [(100, 1000, 1.0), (102, 1000, 1.0), (98, 1000, 1.0)])
        candidate = sample_measurement(workload, [(90, 995, 1.005), (92, 995, 1.005), (88, 995, 1.005)])
        gates = {
            "primary_metric": "step_time_ms", "primary_direction": "lower",
            "minimum_primary_improvement_ratio": 0.05,
            "maximum_guardrail_regression_ratio": {"peak_memory_mb": 0.01, "loss": 0.01},
            "minimum_samples": 3,
        }
        decision = evaluate_performance_candidate(baseline, candidate, gates, workload)
        self.assertEqual(decision["decision"], "accept")
        self.assertGreater(decision["primary_improvement_ratio"], 0.05)

    def test_workload_or_pair_mismatch_is_rejected(self) -> None:
        workload = {"family": "training", "workload_id": "train-v1"}
        baseline = sample_measurement(workload, [(100, 1000, 1.0)] * 3)
        candidate = sample_measurement({**workload, "workload_id": "other"}, [(80, 900, 1.0)] * 3)
        candidate["samples"][0]["pair_id"] = "other-pair"
        gates = {
            "primary_metric": "step_time_ms", "primary_direction": "lower",
            "minimum_primary_improvement_ratio": 0.01,
            "maximum_guardrail_regression_ratio": {}, "minimum_samples": 3,
        }
        decision = evaluate_performance_candidate(baseline, candidate, gates, workload)
        self.assertEqual(decision["decision"], "reject")
        self.assertIn("workload_or_pair_mismatch", decision["hard_failures"])

    def test_composition_conflicts_are_checked_in_both_directions(self) -> None:
        candidates = {"a": {"conflicts_with": ["b"]}, "b": {"conflicts_with": []}}
        self.assertFalse(validate_composition_order(["a", "b"], candidates)["valid"])
        self.assertFalse(validate_composition_order(["b", "a"], candidates)["valid"])

    def test_independent_project_accepts_own_measured_composition(self) -> None:
        workload = {
            "family": "training", "workload_id": "train-v1", "model": "tiny",
            "input": {"batch_size": 1}, "hardware": "gpu0", "runtime": {"torch": "2.x"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "baseline.json", sample_measurement(workload, [(100, 1000, 1.0), (102, 1000, 1.0), (98, 1000, 1.0)]))
            write_json(root / "candidate.json", sample_measurement(workload, [(90, 995, 1.0), (92, 995, 1.0), (88, 995, 1.0)]))
            write_json(root / "composition.json", sample_measurement(workload, [(80, 990, 1.0), (82, 990, 1.0), (78, 990, 1.0)], components=["fused_optimizer"]))
            write_json(root / "trace.json", {"captured": True})
            write_json(root / "rollback.json", {"rollback_ready": True, "restore_action": "reload baseline and disable candidate flags"})
            write_json(root / "reprofile.json", {"workload_identity": workload, "bottleneck_before": "compute", "bottleneck_after": "input", "evidence_path": "trace.json"})
            config = sample_config(root, workload)
            self.assertEqual(validate_delivery_config(config), [])
            result = review_delivery(config, root)
            self.assertEqual(result["decision"], "accept")
            self.assertEqual(result["composition_decision"]["decision"], "accept")
            self.assertEqual(len(result["measurement_sources"]), 6)
            config["capture_status"] = "incomplete"
            incomplete_result = review_delivery(config, root)
            self.assertEqual(incomplete_result["decision"], "tune")
            self.assertIn("project_capture_incomplete", incomplete_result["decision_detail"]["missing_evidence"])

    def _pair(self):
        workload = {"workload_id": "test_training"}
        baseline = sample_measurement(workload, [(100, 1000, 1.0)] * 3)
        candidate = sample_measurement(workload, [(80, 995, 1.0)] * 3)
        gates = sample_config(ROOT, workload)["gates"]
        return baseline, candidate, gates, workload

    def test_invalid_gates_are_rejected_before_comparison(self):
        for key, value in (("minimum_samples", True), ("minimum_samples", 0),
                           ("minimum_primary_improvement_ratio", float("nan")),
                           ("maximum_guardrail_regression_ratio", {"loss": float("inf")}),
                           ("primary_direction", [])):
            with self.subTest(key=key, value=value):
                base, candidate, gates, workload = self._pair()
                gates[key] = value
                with self.assertRaises(ValueError):
                    evaluate_performance_candidate(base, candidate, gates, workload)

    def test_config_rejects_bad_sections_and_labels(self):
        self.assertTrue(validate_delivery_config(None))
        for key, value in (("gates", None), ("outputs", []), ("candidate_order", [["nested"]]),
                           ("run_label", "../outside")):
            config = sample_config(ROOT, {"id": "test"})
            config[key] = value
            self.assertTrue(validate_delivery_config(config))

    def test_flags_require_literal_true_on_baseline_and_candidate(self):
        for side in ("baseline", "candidate"):
            for field, key in (("correctness", "passed"), ("execution_path", "verified")):
                base, candidate, gates, workload = self._pair()
                (base if side == "baseline" else candidate)[field][key] = "false"
                self.assertEqual(evaluate_performance_candidate(base, candidate, gates, workload)["decision"], "reject")

    def test_invalid_metrics_are_rejected(self):
        for value in (float("nan"), float("inf"), -1, True):
            base, candidate, gates, workload = self._pair()
            candidate["samples"][0]["metrics"]["step_time_ms"] = value
            self.assertEqual(evaluate_performance_candidate(base, candidate, gates, workload)["decision"], "reject")

    def test_zero_baseline_pair_cannot_be_silently_dropped(self):
        base, candidate, gates, workload = self._pair()
        base["samples"][0]["metrics"]["step_time_ms"] = 0
        candidate["samples"][0]["metrics"]["step_time_ms"] = 0
        result = evaluate_performance_candidate(base, candidate, gates, workload)
        self.assertEqual(result["decision"], "tune")
        self.assertIsNone(result["primary_improvement_ratio"])

    def test_malformed_or_duplicated_samples_are_rejected(self):
        for value in (None, [None] * 3, [{"pair_id": []}] * 3):
            base, candidate, gates, workload = self._pair()
            candidate["samples"] = value
            self.assertEqual(evaluate_performance_candidate(base, candidate, gates, workload)["decision"], "reject")

    def test_pair_seed_mismatch_is_rejected(self):
        base, candidate, gates, workload = self._pair()
        base["samples"][0]["seed"] = 42
        candidate["samples"][0]["seed"] = 43
        self.assertEqual(evaluate_performance_candidate(base, candidate, gates, workload)["decision"], "reject")

    def test_extra_identity_difference_is_rejected(self):
        base, candidate, gates, workload = self._pair()
        candidate["workload_identity"] = {**workload, "dtype": "changed"}
        self.assertEqual(evaluate_performance_candidate(base, candidate, gates, workload)["decision"], "reject")

    def test_conflicting_composition_cannot_be_selected(self):
        result = decide_delivery(
            rollback={"ready": True}, candidate_order=["single"],
            candidate_decisions={"single": {"decision": "accept", "primary_improvement_ratio": 0.1}},
            composition_decision={"decision": "accept", "primary_improvement_ratio": 0.5},
            composition_check={"valid": False}, reprofile=None,
            gates={"require_reprofile": False},
        )
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["selected_route"], "single")
        self.assertNotIn("composition", result["accepted_routes"])

    def test_unconfigured_route_cannot_be_selected(self):
        result = decide_delivery(
            rollback={"ready": True}, candidate_order=["single"],
            candidate_decisions={"other": {"decision": "accept", "primary_improvement_ratio": 0.5}},
            composition_decision=None, composition_check={"valid": True}, reprofile=None,
            gates={"require_reprofile": False, "require_composition": False},
        )
        self.assertEqual(result["decision"], "tune")
        self.assertIsNone(result["selected_route"])

    def test_selected_route_requires_its_own_reprofile(self):
        result = decide_delivery(
            rollback={"ready": True}, candidate_order=["single"],
            candidate_decisions={"single": {"decision": "accept", "primary_improvement_ratio": 0.1}},
            composition_decision=None, composition_check={"valid": True},
            reprofile={"bottleneck_before": ["compute"], "bottleneck_after": {"other": ["copy"]}, "evidence_path": "trace.json"},
            gates={"require_reprofile": True, "require_composition": False},
        )
        self.assertEqual(result["decision"], "tune")
        self.assertIn("reprofile:selected_route_missing", result["missing_evidence"])

    def test_higher_direction_allows_over_one_hundred_percent_target(self):
        base, candidate, gates, workload = self._pair()
        gates.update(primary_direction="higher", minimum_primary_improvement_ratio=2)
        for sample in candidate["samples"]:
            sample["metrics"]["step_time_ms"] = 400
        self.assertEqual(evaluate_performance_candidate(base, candidate, gates, workload)["decision"], "accept")

    def test_save_refuses_nonfinite_values_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            with self.assertRaises(ValueError):
                save_json_record_once(path, {"value": float("nan")})
            self.assertFalse(path.exists())
            save_json_record_once(path, {"value": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_json_record_once(path, {"value": 2})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
