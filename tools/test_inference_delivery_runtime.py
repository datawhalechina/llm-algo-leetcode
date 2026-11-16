from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import unittest

from tools.inference_delivery_runtime import (
    CONFIG_SCHEMA,
    RESULT_SCHEMA,
    SEMANTIC_ID,
    build_delivery_record,
    evaluate_service_candidate,
    compare_output_fidelity,
    missing_candidate_results,
    select_service_candidate,
    validate_delivery_config,
    validate_deployment_handoff,
)


def _config() -> dict:
    return {
        "schema_version": CONFIG_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "environment_profile": "INF-VLLM-011-CU128",
        "run_label": "smoke-v1",
        "active_candidate": "baseline",
        "inputs": {"deployment_handoff": None},
        "workload": {
            "protocol_version": "inference-service-frozen-workload-v2",
            "path": "workload.jsonl",
            "num_prompts": 5,
            "max_tokens": 32,
            "concurrency": 1,
            "warmup": 1,
            "repeats": 3,
        },
        "candidates": {
            "baseline": {
                "role": "baseline",
                "model": "model",
                "model_revision": "a" * 40,
                "backend": "vllm",
                "dtype": "bf16",
                "cache_dir": "model_cache",
                "served_model_name": "model",
                "cache_policy": "disabled",
                "max_model_len": 1024,
                "max_num_seqs": 4,
                "scheduling_policy": "fcfs",
                "gpu_memory_utilization": 0.5,
                "enforce_eager": True,
                "result_path": "baseline.json",
            },
            "candidate": {
                "role": "candidate",
                "model": "model",
                "model_revision": "a" * 40,
                "backend": "vllm",
                "dtype": "bf16",
                "cache_dir": "model_cache",
                "served_model_name": "model",
                "cache_policy": "enabled",
                "max_model_len": 1024,
                "max_num_seqs": 8,
                "scheduling_policy": "fcfs",
                "gpu_memory_utilization": 0.5,
                "enforce_eager": True,
                "result_path": "candidate.json",
            },
        },
        "gates": {
            "minimum_success_rate": 1.0,
            "minimum_answer_accuracy": 1.0,
            "minimum_complete_correct_rate": 1.0,
            "minimum_response_completion_rate": 1.0,
            "maximum_answer_accuracy_regression": 0.0,
            "maximum_ttft_p95_ms": 100.0,
            "maximum_tpot_p95_ms": 20.0,
            "maximum_ttft_regression_ratio": 0.10,
            "maximum_tpot_regression_ratio": 0.05,
            "minimum_output_throughput_tps": 10.0,
            "maximum_peak_memory_gib": 11.0,
        },
    }


def _result(*, tpot: float, ttft: float, throughput: float, peak_mb: float) -> dict:
    return {
        "schema_version": "inference-benchmark/v1",
        "config": {
            "model": "model",
            "prompt_tokens": 64,
            "generated_tokens": 32,
            "batch": 1,
            "concurrency": 1,
        },
        "metrics": {
            "successful_requests": 5,
            "failed_requests": 0,
            "ttft_ms": {"p95": ttft},
            "tpot_ms": {"p95": tpot},
            "throughput_tokens_per_s": throughput,
            "peak_memory_mb": peak_mb,
        },
        "failure": {"failure_reason": None, "retest_path": None},
    }


def _task_quality(*, accuracy: float = 1.0, complete_correct: float = 1.0) -> dict:
    return {
        "expected_answer_accuracy": accuracy,
        "valid_answer_format_rate": 1.0,
        "unique_answer_tasks": 5,
        "answer_repeat_consistency": 1.0,
        "response_completion_rate": 1.0,
        "complete_correct_rate": complete_correct,
        "truncated_response_rate": 0.0,
    }


def main() -> None:
    config = _config()
    assert validate_delivery_config(config) == []
    placeholder_config = _config()
    placeholder_config["run_label"] = "replace_with_unique_label"
    assert validate_delivery_config(placeholder_config)
    assert validate_delivery_config(placeholder_config, allow_placeholder=True) == []
    handoff = {
        "schema_version": "inference-deployment-handoff/v1",
        "semantic_id": "INF-SINGLE-GPU-DEPLOY",
        "run_label": "handoff-v1",
        "workload": {},
        "candidate_records": [
            {"group": "G2", "handoff_status": "ready_for_integrated_evaluation"}
        ],
        "decision": "accept",
        "rollback": {"ready": True},
    }
    checked = validate_deployment_handoff(handoff)
    assert checked["rollback_ready"] is True
    assert checked["ready_candidates"][0]["group"] == "G2"
    fidelity = compare_output_fidelity(
        {"results": [{"ok": True, "request_id": "r1", "prompt": "p", "output_text": "42"}]},
        {"results": [{"ok": True, "request_id": "r1", "prompt": "p", "output_text": "42"}]},
    )
    assert fidelity == {"status": "measured", "matched_requests": 1, "compared_requests": 1, "score": 1.0}
    assert compare_output_fidelity(
        {"results": [{"ok": True, "request_id": "r1", "prompt": "p", "output_text": "42"}]},
        {"results": [{"ok": True, "request_id": "r1", "prompt": "p", "output_text": "41"}]},
    )["score"] == 0.0

    baseline = _result(tpot=18.0, ttft=80.0, throughput=12.0, peak_mb=9 * 1024)
    candidate = _result(tpot=12.0, ttft=70.0, throughput=18.0, peak_mb=10 * 1024)
    accepted = evaluate_service_candidate(
        candidate,
        baseline,
        config["gates"],
        output_fidelity_score=1.0,
        baseline_output_fidelity_score=1.0,
        task_quality=_task_quality(),
        baseline_task_quality=_task_quality(),
        artifact_verified=True,
        rollback_ready=True,
    )
    assert accepted["decision"] == "accept"
    missing_memory = _result(tpot=12.0, ttft=70.0, throughput=18.0, peak_mb=10 * 1024)
    missing_memory["metrics"]["peak_memory_mb"] = None
    assert evaluate_service_candidate(
        missing_memory,
        baseline,
        config["gates"],
        output_fidelity_score=1.0,
        baseline_output_fidelity_score=1.0,
        task_quality=_task_quality(),
        baseline_task_quality=_task_quality(),
        artifact_verified=True,
        rollback_ready=True,
    )["decision"] == "tune"
    assert evaluate_service_candidate(
        candidate,
        baseline,
        config["gates"],
        output_fidelity_score=1.0,
        baseline_output_fidelity_score=1.0,
        task_quality=_task_quality(),
        baseline_task_quality=_task_quality(),
        artifact_verified=False,
        rollback_ready=True,
    )["decision"] == "reject"
    text_drift_only = evaluate_service_candidate(
        candidate,
        baseline,
        config["gates"],
        output_fidelity_score=0.55,
        baseline_output_fidelity_score=1.0,
        task_quality=_task_quality(),
        baseline_task_quality=_task_quality(),
        artifact_verified=True,
        rollback_ready=True,
    )
    assert text_drift_only["decision"] == "accept"
    incorrect_answer = evaluate_service_candidate(
        candidate,
        baseline,
        config["gates"],
        output_fidelity_score=0.55,
        baseline_output_fidelity_score=1.0,
        task_quality=_task_quality(accuracy=0.8, complete_correct=0.8),
        baseline_task_quality=_task_quality(),
        artifact_verified=True,
        rollback_ready=True,
    )
    assert incorrect_answer["decision"] == "tune"
    selection = select_service_candidate(
        {"baseline": accepted, "candidate": accepted},
        {"baseline": baseline, "candidate": candidate},
    )
    assert selection["selected_candidate"] == "candidate"
    delivery = build_delivery_record(
        run_label="smoke-v1", workload=config["workload"],
        results={"baseline": baseline}, decisions={"baseline": accepted},
        selection={"decision": "accept", "selected_candidate": "baseline"},
    )
    assert delivery["source_handoff"] is None
    assert missing_candidate_results(config["candidates"], {"baseline": baseline}) == ["candidate"]
    assert missing_candidate_results(config["candidates"], {"baseline": baseline, "candidate": candidate}) == []
    config["inputs"]["deployment_handoff"] = ""
    assert any("deployment_handoff" in issue for issue in validate_delivery_config(config))
    print("inference delivery runtime: ok")


class InferenceDeliveryRuntimeTests(unittest.TestCase):
    """Exercise saved-record decisions without a service, GPU or file writes."""

    def evaluate(self, *, candidate=None, quality=None, gates=None):
        return evaluate_service_candidate(
            _result(tpot=12, ttft=70, throughput=18, peak_mb=10 * 1024) if candidate is None else candidate,
            _result(tpot=18, ttft=80, throughput=12, peak_mb=9 * 1024),
            _config()["gates"] if gates is None else gates,
            output_fidelity_score=1.0, baseline_output_fidelity_score=1.0,
            task_quality=_task_quality() if quality is None else quality,
            baseline_task_quality=_task_quality(), artifact_verified=True, rollback_ready=True,
        )

    def test_existing_scenarios(self):
        """Keep all original protocol, decision and output-comparison assertions."""
        with redirect_stdout(io.StringIO()):
            main()

    def test_invalid_roles_return_errors_instead_of_stop_iteration(self):
        for role in ("candidate", "unknown", None):
            config = _config()
            config["candidates"]["baseline"]["role"] = role
            self.assertTrue(validate_delivery_config(config))

    def test_non_mapping_sections_return_errors(self):
        self.assertTrue(validate_delivery_config(None))
        for field in ("workload", "inputs", "candidates", "gates"):
            config = _config()
            config[field] = None
            with self.subTest(field=field):
                self.assertTrue(validate_delivery_config(config))

    def test_missing_or_non_string_labels_are_rejected(self):
        for value in (None, True, 123, "", "  "):
            config = _config()
            config["run_label"] = value
            self.assertTrue(any("run_label" in e for e in validate_delivery_config(config)))

    def test_frozen_workload_answers_and_request_ids(self):
        """Validate the v2 input fixture, not the missing v1 historical input."""
        path = Path(__file__).resolve().parents[1] / "benchmarks/workloads/part05_inference_service_shared_prefix.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({row["request_id"] for row in rows}), len(rows))
        for row in rows:
            cost = re.search(r"costs (\d+) dollars", row["prompt"])
            fee = re.search(r"fee of (\d+) dollars", row["prompt"])
            self.assertIsNotNone(cost)
            self.assertIsNotNone(fee)
            self.assertEqual(row["expected_answer"], int(cost.group(1)) + int(fee.group(1)))
            self.assertEqual(row["chat_template_kwargs"], {"enable_thinking": False})

    def test_result_schema_matches_registered_project(self):
        from tools.project_contracts import load_project_registry

        self.assertEqual(load_project_registry()[SEMANTIC_ID]["target_contract"], RESULT_SCHEMA)

    def test_workload_counts_require_integers_and_warmup_can_be_zero(self):
        for field in ("num_prompts", "max_tokens", "concurrency", "repeats", "warmup"):
            for value in (None, True, "3", 1.5, -1):
                config = _config()
                config["workload"][field] = value
                with self.subTest(field=field, value=value):
                    self.assertTrue(any(f"workload.{field}" in e for e in validate_delivery_config(config)))
        config = _config()
        config["workload"]["warmup"] = 0
        self.assertEqual(validate_delivery_config(config), [])

    def test_candidate_capacity_and_device_options_require_valid_types(self):
        cases = [
            ("max_num_seqs", "8"), ("max_num_seqs", True),
            ("max_model_len", -1), ("max_model_len", 1.5),
            ("enforce_eager", "true"), ("gpu_memory_utilization", "0.5"),
            ("gpu_memory_utilization", float("nan")), ("gpu_memory_utilization", float("inf")),
            ("gpu_memory_utilization", 0), ("gpu_memory_utilization", 1.1),
        ]
        for field, value in cases:
            config = _config()
            for item in config["candidates"].values():
                item[field] = value
            with self.subTest(field=field, value=value):
                self.assertTrue(any(field in e for e in validate_delivery_config(config)))

    def test_invalid_gates_are_rejected_in_configuration(self):
        for field in _config()["gates"]:
            for value in (True, "1", -1, float("nan"), float("inf")):
                config = _config()
                config["gates"][field] = value
                with self.subTest(field=field, value=value):
                    self.assertTrue(any(field in e for e in validate_delivery_config(config)))
        config = _config()
        config["gates"]["minimum_answer_accuracy"] = 1.1
        self.assertTrue(validate_delivery_config(config))

    def test_nonfinite_metrics_do_not_accept_a_candidate(self):
        for field in ("ttft_ms", "tpot_ms", "throughput_tokens_per_s", "peak_memory_mb"):
            for value in (float("nan"), float("inf"), -1, True):
                candidate = _result(tpot=12, ttft=70, throughput=18, peak_mb=10 * 1024)
                candidate["metrics"][field] = {"p95": value} if field in {"ttft_ms", "tpot_ms"} else value
                with self.subTest(field=field, value=value):
                    result = self.evaluate(candidate=candidate)
                    self.assertEqual(result["decision"], "tune")
                    self.assertTrue(result["missing_evidence"])

    def test_nonfinite_or_out_of_range_quality_is_missing_evidence(self):
        for field in ("expected_answer_accuracy", "complete_correct_rate", "response_completion_rate"):
            for value in (float("nan"), float("inf"), -0.1, 1.1, "1", True):
                quality = _task_quality()
                quality[field] = value
                with self.subTest(field=field, value=value):
                    self.assertEqual(self.evaluate(quality=quality)["decision"], "tune")

    def test_direct_evaluation_rejects_invalid_thresholds(self):
        for value in (float("nan"), float("inf"), "100", -1):
            gates = _config()["gates"]
            gates["maximum_ttft_p95_ms"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(gates=gates)

    def test_config_validation_does_not_modify_input(self):
        config = _config()
        before = deepcopy(config)
        self.assertEqual(validate_delivery_config(config), [])
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
