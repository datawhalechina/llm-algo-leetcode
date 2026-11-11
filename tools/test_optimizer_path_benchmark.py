"""CPU tests for the optimizer-path project contracts and decision gates."""

from __future__ import annotations

import unittest

from tools.optimizer_path_benchmark import (
    OPTIMIZER_PATHS,
    decide_optimizer_path,
    validate_optimizer_candidates,
)


def candidate_rows(*, learning_rate: float = 1e-4, scheduler: str | None = None):
    return [
        {
            "name": name,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "scheduler": scheduler,
            "path": path,
        }
        for name, path in OPTIMIZER_PATHS.items()
    ]


class OptimizerPathContractTests(unittest.TestCase):
    def test_fixed_adamw_paths_pass(self):
        result = validate_optimizer_candidates(
            candidate_rows(), learning_rate=1e-4, weight_decay=0.01
        )
        self.assertTrue(result["valid"], result["errors"])

    def test_learning_rate_or_scheduler_drift_fails(self):
        rows = candidate_rows()
        rows[1]["learning_rate"] = 2e-4
        rows[2]["scheduler"] = "cosine"
        result = validate_optimizer_candidates(
            rows, learning_rate=1e-4, weight_decay=0.01
        )
        self.assertFalse(result["valid"])
        self.assertIn("foreach:learning_rate_mismatch", result["errors"])
        self.assertIn("fused:scheduler_must_be_fixed", result["errors"])

    def test_end_to_end_gain_is_required_for_accept(self):
        records = [
            {
                "name": "single_tensor", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 10.0, "optimizer_step_ms": 2.0,
                "peak_allocated_mb": 900.0, "max_parameter_error": 0.0,
            },
            {
                "name": "foreach", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 9.0, "optimizer_step_ms": 1.2,
                "peak_allocated_mb": 920.0, "max_parameter_error": 1e-6,
            },
            {
                "name": "fused", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 10.1, "optimizer_step_ms": 0.8,
                "peak_allocated_mb": 905.0, "max_parameter_error": 2e-6,
            },
        ]
        result = decide_optimizer_path(records)
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["selected"], "foreach")
        fused = next(item for item in result["candidates"] if item["name"] == "fused")
        self.assertFalse(fused["eligible"])
        self.assertGreater(fused["optimizer_step_speedup"], 1.0)

    def test_incomparable_workloads_are_pending(self):
        records = [
            {
                "name": "single_tensor", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 10.0, "optimizer_step_ms": 2.0,
                "peak_allocated_mb": 900.0, "max_parameter_error": 0.0,
            },
            {
                "name": "foreach", "status": "ok", "workload_id": "mlp-v2",
                "seed": 42, "training_step_ms": 8.0, "optimizer_step_ms": 1.0,
                "peak_allocated_mb": 900.0, "max_parameter_error": 0.0,
            },
            {
                "name": "fused", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 9.8, "optimizer_step_ms": 0.8,
                "peak_allocated_mb": 900.0, "max_parameter_error": 0.0,
            },
        ]
        result = decide_optimizer_path(records)
        self.assertEqual(result["decision"], "pending")
        self.assertEqual(result["reason"], "incomparable_workloads")

    def test_incomplete_candidate_set_is_pending(self):
        records = [
            {
                "name": name, "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 10.0, "optimizer_step_ms": 2.0,
                "peak_allocated_mb": 900.0, "max_parameter_error": 0.0,
            }
            for name in ("single_tensor", "fused")
        ]
        result = decide_optimizer_path(records)
        self.assertEqual(result["decision"], "pending")
        self.assertEqual(result["reason"], "invalid_measurements")
        self.assertIn("candidate_set_mismatch", result["errors"])

    def test_correctness_or_memory_guardrail_can_block_adoption(self):
        records = [
            {
                "name": "single_tensor", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 10.0, "optimizer_step_ms": 2.0,
                "peak_allocated_mb": 900.0, "max_parameter_error": 0.0,
            },
            {
                "name": "foreach", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 8.0, "optimizer_step_ms": 1.0,
                "peak_allocated_mb": 1200.0, "max_parameter_error": 2e-3,
            },
            {
                "name": "fused", "status": "ok", "workload_id": "mlp-v1",
                "seed": 42, "training_step_ms": 9.8, "optimizer_step_ms": 0.8,
                "peak_allocated_mb": 900.0, "max_parameter_error": 0.0,
            },
        ]
        result = decide_optimizer_path(records)
        self.assertEqual(result["decision"], "tune")
        self.assertFalse(result["candidates"][0]["eligible"])

    def test_non_finite_measurement_cannot_be_adopted(self):
        records = [
            {
                "name": name,
                "status": "ok",
                "workload_id": "mlp-v1",
                "seed": 42,
                "training_step_ms": 10.0,
                "optimizer_step_ms": 2.0,
                "peak_allocated_mb": 900.0,
                "max_parameter_error": 0.0,
            }
            for name in OPTIMIZER_PATHS
        ]
        records[1]["training_step_ms"] = float("nan")
        result = decide_optimizer_path(records)
        self.assertEqual(result["decision"], "pending")
        self.assertIn("non_finite_evidence", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
