import unittest

from tools.memory_strategy_runtime import (
    decide_memory_boundary,
    decide_memory_strategy_project,
    evaluate_memory_budget_sensitivity,
    summarize_memory_boundary,
    summarize_memory_strategy_candidates,
)


class MemoryStrategyRuntimeTest(unittest.TestCase):
    def test_budget_sensitivity_reports_stable_and_changed_selection(self):
        candidates = [
            {"name": "baseline", "status": "ok", "peak_memory_mb": 1000.0, "samples_per_s": 2.0, "eval_loss": 1.0},
            {"name": "checkpoint", "status": "ok", "peak_memory_mb": 800.0, "samples_per_s": 1.6, "eval_loss": 1.02},
            {"name": "offload", "status": "ok", "peak_memory_mb": 700.0, "samples_per_s": 0.7, "eval_loss": 1.02},
        ]
        base_budget = {
            "memory_cap_mb": 1100.0,
            "min_samples_per_s": 1.0,
            "min_memory_saving_mb": 100.0,
            "min_throughput_ratio": 0.7,
        }
        quality_floor = {"max_val_loss": 1.1}
        stable = evaluate_memory_budget_sensitivity(candidates, [
            {"name": "reference", "budget": base_budget, "quality_floor": quality_floor},
            {"name": "stricter_memory", "budget": {**base_budget, "memory_cap_mb": 1050.0}, "quality_floor": quality_floor},
        ])
        self.assertEqual(stable["status"], "stable")
        changed = evaluate_memory_budget_sensitivity(candidates, [
            {"name": "reference", "budget": base_budget, "quality_floor": quality_floor},
            {"name": "stricter_throughput", "budget": {**base_budget, "min_samples_per_s": 1.8}, "quality_floor": quality_floor},
        ])
        self.assertEqual(changed["status"], "threshold_sensitive")
        empty = evaluate_memory_budget_sensitivity(candidates, [
            {"name": "too_strict", "budget": {**base_budget, "memory_cap_mb": 650.0}, "quality_floor": quality_floor},
        ])
        self.assertEqual(empty["status"], "no_feasible_candidate")

    def test_valid_baseline_is_available_to_decision(self):
        candidates = [
            {
                "name": "baseline",
                "status": "ok",
                "peak_memory_mb": 1000.0,
                "samples_per_s": 2.0,
                "eval_loss": 1.0,
            },
            {
                "name": "checkpoint",
                "status": "ok",
                "peak_memory_mb": 990.0,
                "samples_per_s": 1.8,
                "eval_loss": 1.0,
            },
        ]
        budget = {
            "memory_cap_mb": 1200.0,
            "min_samples_per_s": 1.0,
            "min_memory_saving_mb": 100.0,
            "min_throughput_ratio": 0.7,
        }
        summary = summarize_memory_strategy_candidates(
            candidates, budget, {"max_val_loss": 1.1}
        )

        self.assertTrue(summary["baseline_available"])
        self.assertEqual(
            decide_memory_strategy_project(summary)["reason"],
            "memory_saving_below_meaningful_threshold",
        )

    def test_boundary_accepts_capacity_extension_with_throughput_guardrail(self):
        def candidate(name, memory, throughput, status="ok"):
            result = {"name": name, "status": status}
            if status == "ok":
                result.update(
                    peak_memory_mb=memory,
                    samples_per_s=throughput,
                    eval_loss=1.0,
                )
            return result

        records = [
            {
                "seq_len": 1024,
                "candidates": [
                    candidate("baseline", 1100.0, 2.0),
                    candidate("checkpoint", 900.0, 1.6),
                ],
            },
            {
                "seq_len": 1280,
                "candidates": [
                    candidate("baseline", None, None, "oom"),
                    candidate("checkpoint", 1050.0, 1.2),
                ],
            },
        ]
        budget = {
            "min_memory_saving_mb": 512.0,
            "min_throughput_ratio": 0.7,
        }
        summary = summarize_memory_boundary(
            records, budget, {"max_val_loss": 1.1}
        )

        self.assertEqual(summary["baseline_max_runnable_seq_len"], 1024)
        self.assertEqual(summary["baseline_oom_seq_len"], 1280)
        self.assertEqual(summary["eligible_strategies"], ["checkpoint"])
        decision = decide_memory_boundary(summary)
        self.assertEqual(
            decision["decision"], "accept_for_activation_memory_scope"
        )
        self.assertEqual(decision["scope"], "activation_memory_isolation")
        self.assertFalse(decision["promotable_to_full_training"])

    def test_boundary_requests_longer_ladder_when_baseline_never_ooms(self):
        records = [
            {
                "seq_len": 1024,
                "candidates": [
                    {
                        "name": "baseline",
                        "status": "ok",
                        "peak_memory_mb": 1000.0,
                        "samples_per_s": 2.0,
                        "eval_loss": 1.0,
                    }
                ],
            }
        ]
        summary = summarize_memory_boundary(
            records,
            {"min_memory_saving_mb": 512.0, "min_throughput_ratio": 0.7},
            {"max_val_loss": 1.1},
        )

        self.assertEqual(decide_memory_boundary(summary)["decision"], "measure")
        self.assertEqual(
            decide_memory_boundary(summary)["reason"],
            "baseline_boundary_not_reached",
        )


if __name__ == "__main__":
    unittest.main()
