"""Shared runtime logic for the memory budget decision project."""

from typing import Dict, List


def validate_memory_budget(
    budget: Dict[str, float], quality_floor: Dict[str, float]
) -> Dict[str, object]:
    required_budget_keys = ["memory_cap_mb", "min_samples_per_s"]
    required_quality_keys = ["max_val_loss"]
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    return {"is_valid": not missing_keys, "missing_keys": missing_keys}


def summarize_memory_strategies(
    candidates: List[Dict[str, float]],
    budget: Dict[str, float],
    quality_floor: Dict[str, float],
) -> Dict[str, object]:
    feasible: List[Dict[str, float]] = []
    quality_failed = 0
    invalid_count = 0
    oom_count = 0
    evaluations = []

    for candidate in candidates:
        if candidate.get("status", "ok") == "oom":
            oom_count += 1
            evaluations.append({"name": candidate.get("name"), "status": "oom", "reasons": ["oom"]})
            continue
        memory = candidate.get("peak_memory_mb")
        throughput = candidate.get("samples_per_s")
        eval_loss = candidate.get("eval_loss", candidate.get("val_loss"))
        if not all(isinstance(value, (int, float)) for value in (memory, throughput, eval_loss)):
            invalid_count += 1
            evaluations.append({"name": candidate.get("name"), "status": "invalid", "reasons": ["missing_or_non_numeric_metric"]})
            continue
        memory_ok = memory <= budget["memory_cap_mb"]
        speed_ok = throughput >= budget["min_samples_per_s"]
        quality_ok = eval_loss <= quality_floor["max_val_loss"]
        if not quality_ok:
            quality_failed += 1
        reasons = []
        if not memory_ok:
            reasons.append("memory_over_budget")
        if not speed_ok:
            reasons.append("throughput_below_floor")
        if not quality_ok:
            reasons.append("quality_over_floor")
        evaluations.append({
            "name": candidate.get("name"),
            "status": "feasible" if not reasons else "rejected",
            "reasons": reasons,
        })
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    feasible.sort(
        key=lambda item: (
            item["peak_memory_mb"],
            -item["samples_per_s"],
            item.get("eval_loss", item.get("val_loss")),
        )
    )
    baseline = next(
        (
            item
            for item in candidates
            if item.get("name") == "baseline"
            and item.get("status", "ok") == "ok"
            and all(
                isinstance(item.get(key), (int, float))
                for key in ("peak_memory_mb", "samples_per_s")
            )
        ),
        None,
    )
    best = feasible[0] if feasible else None
    return {
        "candidate_count": len(candidates),
        "measured_count": len(candidates) - oom_count - invalid_count,
        "oom_count": oom_count,
        "invalid_count": invalid_count,
        "feasible_count": len(feasible),
        "best_candidate": best["name"] if best else None,
        "quality_failed_count": quality_failed,
        "feasible_names": [item["name"] for item in feasible],
        "baseline_peak_memory_mb": baseline["peak_memory_mb"] if baseline else None,
        "throughput_ratio": (
            best["samples_per_s"] / baseline["samples_per_s"]
            if baseline and best
            else None
        ),
        "best_peak_memory_mb": best["peak_memory_mb"] if best else None,
        "memory_saving_mb": (
            baseline["peak_memory_mb"] - best["peak_memory_mb"]
            if baseline and best
            else 0.0
        ),
        "min_memory_saving_mb": budget.get("min_memory_saving_mb", 512.0),
        "min_throughput_ratio": budget.get("min_throughput_ratio", 0.70),
        "evaluations": evaluations,
    }


def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    feasible_count = summary["feasible_count"]
    best_candidate = summary["best_candidate"]
    quality_failed_count = summary["quality_failed_count"]

    if feasible_count == 0:
        return {
            "decision": "reject",
            "reason": "no_strategy_meets_budget_and_quality",
            "next_action": "tighten_batch_or_rework_memory_plan",
        }
    meaningful_memory_gain = summary.get("memory_saving_mb", 0.0) >= summary.get(
        "min_memory_saving_mb", 512.0
    )
    acceptable_throughput = (
        summary.get("throughput_ratio") is None
        or summary.get("throughput_ratio")
        >= summary.get("min_throughput_ratio", 0.70)
    )
    if (
        best_candidate in {"checkpoint", "offload", "hybrid"}
        and meaningful_memory_gain
        and acceptable_throughput
    ):
        return {
            "decision": "accept",
            "reason": "compression_strategy_is_best_feasible_option",
            "next_action": "promote_to_training_run",
        }
    if quality_failed_count > 0:
        return {
            "decision": "tune",
            "reason": "compression_needs_quality_recovery",
            "next_action": "adjust_checkpoint_scope_or_batch_plan",
        }
    if not meaningful_memory_gain:
        return {
            "decision": "tune",
            "reason": "memory_saving_below_meaningful_threshold",
            "next_action": "test_larger_pressure_or_optimizer_compression",
        }
    if not acceptable_throughput:
        return {
            "decision": "tune",
            "reason": "throughput_loss_exceeds_budget",
            "next_action": "reduce_compression_scope",
        }
    return {
        "decision": "tune",
        "reason": "baseline_still_best_under_current_budget",
        "next_action": "revisit_memory_strategy_mix",
    }
