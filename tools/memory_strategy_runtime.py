"""Shared runtime logic for the activation/checkpoint/offload benchmark."""

import math
from typing import Dict, List


def validate_strategy_budget(
    budget: Dict[str, float], quality_floor: Dict[str, float]
) -> Dict[str, object]:
    required_budget_keys = [
        "memory_cap_mb",
        "min_samples_per_s",
        "min_memory_saving_mb",
        "min_throughput_ratio",
    ]
    required_quality_keys = ["max_val_loss"]
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    numeric_values = {key: budget.get(key) for key in required_budget_keys}
    numeric_values.update({key: quality_floor.get(key) for key in required_quality_keys})
    invalid_keys = [
        key
        for key, value in numeric_values.items()
        if key not in missing_keys
        and (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
        )
    ]
    memory_cap = budget.get("memory_cap_mb")
    min_throughput = budget.get("min_samples_per_s")
    min_saving = budget.get("min_memory_saving_mb")
    throughput_ratio = budget.get("min_throughput_ratio")
    max_loss = quality_floor.get("max_val_loss")
    if isinstance(memory_cap, (int, float)) and memory_cap <= 0:
        invalid_keys.append("memory_cap_mb")
    if isinstance(min_throughput, (int, float)) and min_throughput < 0:
        invalid_keys.append("min_samples_per_s")
    if isinstance(min_saving, (int, float)) and min_saving < 0:
        invalid_keys.append("min_memory_saving_mb")
    if isinstance(throughput_ratio, (int, float)) and not 0 <= throughput_ratio <= 1:
        invalid_keys.append("min_throughput_ratio")
    if isinstance(max_loss, (int, float)) and max_loss < 0:
        invalid_keys.append("max_val_loss")
    invalid_keys = list(dict.fromkeys(invalid_keys))
    return {
        "is_valid": not missing_keys and not invalid_keys,
        "missing_keys": missing_keys,
        "invalid_keys": invalid_keys,
    }


def summarize_memory_strategy_candidates(
    candidates: List[Dict[str, object]],
    budget: Dict[str, float],
    quality_floor: Dict[str, float],
) -> Dict[str, object]:
    feasible: List[Dict[str, float]] = []
    quality_failed = 0
    invalid_count = 0
    oom_count = 0

    for candidate in candidates:
        if candidate.get("status", "ok") == "oom":
            oom_count += 1
            continue
        memory = candidate.get("peak_memory_mb")
        throughput = candidate.get("samples_per_s")
        eval_loss = candidate.get("eval_loss", candidate.get("val_loss"))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (memory, throughput, eval_loss)):
            invalid_count += 1
            continue
        memory_ok = memory <= budget["memory_cap_mb"]
        speed_ok = throughput >= budget["min_samples_per_s"]
        quality_ok = eval_loss <= quality_floor["max_val_loss"]
        if not quality_ok:
            quality_failed += 1
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
                and math.isfinite(item.get(key))
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
        "baseline_available": baseline is not None,
        "best_peak_memory_mb": best["peak_memory_mb"] if best else None,
        "memory_saving_mb": (
            baseline["peak_memory_mb"] - best["peak_memory_mb"]
            if baseline and best
            else 0.0
        ),
        "throughput_ratio": (
            best["samples_per_s"] / baseline["samples_per_s"]
            if baseline and best
            else None
        ),
        "min_memory_saving_mb": budget["min_memory_saving_mb"],
        "min_throughput_ratio": budget["min_throughput_ratio"],
    }


def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
    feasible_count = summary["feasible_count"]
    best_candidate = summary["best_candidate"]
    quality_failed_count = summary["quality_failed_count"]

    if not summary.get("baseline_available", False):
        return {
            "decision": "reject",
            "reason": "baseline_missing_or_invalid",
            "next_action": "rerun_baseline_before_comparing_candidates",
        }
    if feasible_count == 0:
        return {
            "decision": "reject",
            "reason": "no_strategy_meets_budget_and_quality",
            "next_action": "rework_checkpoint_or_offload_scope",
        }
    meaningful_memory_gain = summary.get("memory_saving_mb", 0.0) >= summary[
        "min_memory_saving_mb"
    ]
    acceptable_throughput = (
        summary.get("throughput_ratio") is not None
        and summary["throughput_ratio"] >= summary["min_throughput_ratio"]
    )
    if (
        best_candidate != "baseline"
        and meaningful_memory_gain
        and acceptable_throughput
    ):
        return {
            "decision": "accept",
            "reason": "strategy_is_best_feasible_option",
            "next_action": "promote_to_training_run",
        }
    if quality_failed_count > 0:
        return {
            "decision": "tune",
            "reason": "strategy_needs_quality_recovery",
            "next_action": "adjust_checkpoint_granularity_or_offload_scope",
        }
    if not meaningful_memory_gain:
        return {
            "decision": "tune",
            "reason": "memory_saving_below_meaningful_threshold",
            "next_action": "test_pressure_or_offload_scope",
        }
    if not acceptable_throughput:
        return {
            "decision": "tune",
            "reason": "throughput_loss_exceeds_budget",
            "next_action": "reduce_checkpoint_or_offload_scope",
        }
    return {
        "decision": "tune",
        "reason": "baseline_still_best_under_current_budget",
        "next_action": "revisit_strategy_mix",
    }


def evaluate_memory_budget_sensitivity(
    candidates: List[Dict[str, object]],
    scenarios: List[Dict[str, object]],
) -> Dict[str, object]:
    """Re-evaluate one measured candidate set under named budget scenarios."""
    results = []
    for scenario in scenarios:
        name = scenario.get("name")
        budget = scenario.get("budget")
        quality_floor = scenario.get("quality_floor")
        if not isinstance(name, str) or not name or not isinstance(budget, dict) or not isinstance(quality_floor, dict):
            raise ValueError("each sensitivity scenario needs a name, budget, and quality_floor")
        validation = validate_strategy_budget(budget, quality_floor)
        if not validation["is_valid"]:
            raise ValueError(f"invalid sensitivity scenario {name}: {validation}")
        summary = summarize_memory_strategy_candidates(candidates, budget, quality_floor)
        decision = decide_memory_strategy_project(summary)
        results.append({
            "name": name,
            "budget": budget,
            "quality_floor": quality_floor,
            "summary": summary,
            "decision": decision,
        })

    if not results:
        raise ValueError("at least one sensitivity scenario is required")
    reference = results[0]
    stable = all(
        row["decision"].get("decision") == reference["decision"].get("decision")
        and row["summary"].get("best_candidate") == reference["summary"].get("best_candidate")
        and row["summary"].get("feasible_names") == reference["summary"].get("feasible_names")
        for row in results[1:]
    )
    no_feasible = any(row["summary"].get("feasible_count", 0) == 0 for row in results)
    return {
        "scenario_count": len(results),
        "status": "no_feasible_candidate" if no_feasible else "stable" if stable else "threshold_sensitive",
        "reference_scenario": reference["name"],
        "results": results,
    }


def summarize_memory_boundary(
    length_records: List[Dict[str, object]],
    budget: Dict[str, float],
    quality_floor: Dict[str, float],
) -> Dict[str, object]:
    """Summarize maximum runnable length and common-length trade-offs."""

    strategy_names = list(
        dict.fromkeys(
            candidate.get("name")
            for record in length_records
            for candidate in record.get("candidates", [])
            if isinstance(candidate.get("name"), str)
        )
    )
    successful: Dict[str, Dict[int, Dict[str, object]]] = {
        name: {} for name in strategy_names
    }
    quality_failed: List[Dict[str, object]] = []
    for record in length_records:
        seq_len = record.get("seq_len")
        if not isinstance(seq_len, int) or seq_len <= 0:
            continue
        for candidate in record.get("candidates", []):
            name = candidate.get("name")
            eval_loss = candidate.get("eval_loss")
            metrics_ok = all(
                isinstance(candidate.get(key), (int, float))
                and math.isfinite(candidate[key])
                for key in ("peak_memory_mb", "samples_per_s", "eval_loss")
            )
            if candidate.get("status") == "ok" and metrics_ok:
                if eval_loss <= quality_floor["max_val_loss"]:
                    successful[name][seq_len] = candidate
                else:
                    quality_failed.append({"name": name, "seq_len": seq_len})

    max_runnable = {
        name: max(rows) if rows else None for name, rows in successful.items()
    }
    baseline_rows = successful.get("baseline", {})
    baseline_max = max_runnable.get("baseline")
    baseline_oom_lengths = [
        record["seq_len"]
        for record in length_records
        for candidate in record.get("candidates", [])
        if candidate.get("name") == "baseline" and candidate.get("status") == "oom"
    ]
    comparisons = []
    eligible = []
    for name in strategy_names:
        if name == "baseline":
            continue
        candidate_rows = successful[name]
        common_lengths = sorted(set(baseline_rows) & set(candidate_rows))
        comparison_length = common_lengths[-1] if common_lengths else None
        if comparison_length is None:
            throughput_ratio = None
            memory_saving_mb = None
        else:
            baseline = baseline_rows[comparison_length]
            candidate = candidate_rows[comparison_length]
            throughput_ratio = candidate["samples_per_s"] / baseline["samples_per_s"]
            memory_saving_mb = baseline["peak_memory_mb"] - candidate["peak_memory_mb"]
        capacity_extended = (
            baseline_max is not None
            and max_runnable[name] is not None
            and max_runnable[name] > baseline_max
        )
        meaningful_memory_gain = (
            memory_saving_mb is not None
            and memory_saving_mb >= budget["min_memory_saving_mb"]
        )
        acceptable_throughput = (
            throughput_ratio is not None
            and throughput_ratio >= budget["min_throughput_ratio"]
        )
        accepted = acceptable_throughput and (
            capacity_extended or meaningful_memory_gain
        )
        comparison = {
            "name": name,
            "max_runnable_seq_len": max_runnable[name],
            "comparison_seq_len": comparison_length,
            "capacity_extended": capacity_extended,
            "memory_saving_mb": memory_saving_mb,
            "throughput_ratio": throughput_ratio,
            "meaningful_memory_gain": meaningful_memory_gain,
            "acceptable_throughput": acceptable_throughput,
            "accepted": accepted,
        }
        comparisons.append(comparison)
        if accepted:
            eligible.append(name)

    return {
        "tested_seq_lens": [record.get("seq_len") for record in length_records],
        "baseline_available": baseline_max is not None,
        "baseline_max_runnable_seq_len": baseline_max,
        "baseline_oom_seq_len": min(baseline_oom_lengths) if baseline_oom_lengths else None,
        "boundary_observed": bool(baseline_oom_lengths),
        "max_runnable_seq_len": max_runnable,
        "quality_failed": quality_failed,
        "comparisons": comparisons,
        "eligible_strategies": eligible,
        "min_memory_saving_mb": budget["min_memory_saving_mb"],
        "min_throughput_ratio": budget["min_throughput_ratio"],
    }


def decide_memory_boundary(summary: Dict[str, object]) -> Dict[str, object]:
    """Decide the activation-memory scope without promoting it to full training."""

    if not summary.get("baseline_available"):
        return {
            "decision": "reject",
            "reason": "baseline_missing_or_invalid",
            "next_action": "rerun_boundary_from_the_first_length",
        }
    if summary.get("eligible_strategies"):
        return {
            "decision": "accept_for_activation_memory_scope",
            "reason": "capacity_or_memory_gain_meets_throughput_guardrail",
            "next_action": "validate_candidate_in_the_matched_full_training_workload",
            "scope": "activation_memory_isolation",
            "promotable_to_full_training": False,
        }
    if summary.get("quality_failed"):
        return {
            "decision": "tune",
            "reason": "boundary_strategy_failed_quality_floor",
            "next_action": "inspect_numerics_before_capacity_claim",
        }
    if not summary.get("boundary_observed"):
        return {
            "decision": "measure",
            "reason": "baseline_boundary_not_reached",
            "next_action": "extend_sequence_length_ladder",
        }
    return {
        "decision": "tune",
        "reason": "no_strategy_meets_boundary_and_throughput_guardrails",
        "next_action": "adjust_checkpoint_or_offload_scope",
    }
