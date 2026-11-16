"""Contracts and decision logic for the Part 05 inference delivery project."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "inference-service-delivery-config/v2"
HANDOFF_SCHEMA = "inference-deployment-handoff/v1"
RESULT_SCHEMA = "inference-service-delivery/v2"
SEMANTIC_ID = "INF-INTEGRATED-DELIVERY"


def _finite_number(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _rate(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and 0 <= number <= 1 else None


def _gate_errors(gates: Any) -> list[str]:
    """Reject missing or invalid thresholds before comparisons can accept a run."""
    if not isinstance(gates, Mapping):
        return ["gates 必须是 mapping"]
    rates = {
        "minimum_success_rate", "minimum_answer_accuracy", "minimum_complete_correct_rate",
        "minimum_response_completion_rate", "maximum_answer_accuracy_regression",
    }
    positive = {"maximum_ttft_p95_ms", "maximum_tpot_p95_ms", "maximum_peak_memory_gib"}
    required = rates | positive | {
        "maximum_ttft_regression_ratio", "maximum_tpot_regression_ratio",
        "minimum_output_throughput_tps",
    }
    errors = []
    for key in sorted(required):
        if key not in gates:
            errors.append(f"gates 缺少 {key}")
            continue
        number = _finite_number(gates[key])
        if number is None or number < 0 or (key in rates and number > 1) or (key in positive and number == 0):
            errors.append(f"gates.{key} 必须是有效范围内的有限数值")
    return errors


def _metric(record: Mapping[str, Any], key: str, statistic: str | None = None) -> float | None:
    metrics = record.get("metrics", {})
    value = metrics.get(key) if isinstance(metrics, Mapping) else None
    if isinstance(value, Mapping):
        value = value.get(statistic or "p50")
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def _has_failure(record: Mapping[str, Any]) -> bool:
    failure = record.get("failure")
    if failure in (None, {}, "", "not_recorded"):
        return False
    if isinstance(failure, Mapping):
        return any(value not in (None, "", {}, []) for value in failure.values())
    return True


def load_json_record(path: str | Path) -> dict[str, Any]:
    """Load one JSON object without guessing around a missing path."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"项目记录不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"项目记录必须是 JSON object：{source}")
    return dict(payload)


def normalized_backend_result(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared inference record stored by the backend benchmark."""

    normalized = record.get("normalized_result")
    if isinstance(normalized, Mapping):
        return dict(normalized)
    if record.get("schema_version") == "inference-benchmark/v1" and "metrics" in record:
        return dict(record)
    raise ValueError("后端结果缺少 normalized_result 或 inference-benchmark/v1 记录")


def validate_delivery_config(
    config: Mapping[str, Any], *, allow_placeholder: bool = False,
) -> list[str]:
    """Validate identities, workload and gates before contacting a backend."""

    errors: list[str] = []
    if not isinstance(config, Mapping):
        return ["config 必须是 mapping"]
    if config.get("schema_version") != CONFIG_SCHEMA:
        errors.append(f"schema_version 必须是 {CONFIG_SCHEMA}")
    if config.get("semantic_id") != SEMANTIC_ID:
        errors.append(f"semantic_id 必须是 {SEMANTIC_ID}")
    if config.get("environment_profile") != "INF-VLLM-011-CU128":
        errors.append("本项目默认统一使用 INF-VLLM-011-CU128 环境")
    workload = config.get("workload", {})
    if not isinstance(workload, Mapping):
        errors.append("workload 必须是 mapping")
        workload = {}
    if workload.get("protocol_version") != "inference-service-frozen-workload-v2":
        errors.append("workload.protocol_version 必须是 inference-service-frozen-workload-v2")
    raw_label = config.get("run_label")
    run_label = raw_label.strip() if isinstance(raw_label, str) else ""
    if not run_label or (run_label == "replace_with_unique_label" and not allow_placeholder):
        errors.append("run_label 必须替换为唯一值")
    inputs = config.get("inputs", {})
    if not isinstance(inputs, Mapping):
        errors.append("inputs 必须是 mapping")
    handoff = inputs.get("deployment_handoff") if isinstance(inputs, Mapping) else None
    if handoff is not None and not str(handoff).strip():
        errors.append("inputs.deployment_handoff 应为非空路径或 null")
    for key in ("path", "num_prompts", "max_tokens", "concurrency", "warmup", "repeats"):
        if key not in workload:
            errors.append(f"workload 缺少 {key}")
    if "path" in workload and (not isinstance(workload["path"], str) or not workload["path"].strip()):
        errors.append("workload.path 必须是非空路径")
    for key in ("num_prompts", "max_tokens", "concurrency", "repeats", "warmup"):
        if key in workload and (type(workload[key]) is not int or workload[key] < (0 if key == "warmup" else 1)):
            errors.append(f"workload.{key} 必须是{'非负' if key == 'warmup' else '正'}整数")
    candidates = config.get("candidates")
    if not isinstance(candidates, Mapping) or not candidates:
        errors.append("candidates 至少包含一个服务候选")
    else:
        active = config.get("active_candidate")
        if not isinstance(active, str) or active not in candidates:
            errors.append("active_candidate 必须指向 candidates 中的候选")
        roles = [item.get("role") for item in candidates.values() if isinstance(item, Mapping)]
        if roles.count("baseline") != 1:
            errors.append("candidates 必须且只能包含一个 baseline")
        if len(candidates) != 2 or roles.count("candidate") != 1:
            errors.append("本实验固定比较一个 baseline 与一个 candidate")
        backend_names = {str(item.get("backend", "")).lower() for item in candidates.values() if isinstance(item, Mapping)}
        if len(backend_names) > 1:
            errors.append("一次运行只允许一个 backend；切换后端时应复用同一工作负载逐个测量")
        for name, item in candidates.items():
            if not isinstance(item, Mapping):
                errors.append(f"candidate {name} 必须是 mapping")
                continue
            for key in (
                "role", "model", "model_revision", "cache_dir", "served_model_name", "backend", "dtype",
                "cache_policy", "max_model_len", "max_num_seqs", "scheduling_policy",
                "gpu_memory_utilization", "enforce_eager", "result_path",
            ):
                if not str(item.get(key, "")).strip():
                    errors.append(f"candidate {name} 缺少 {key}")
            if item.get("cache_policy") not in {"enabled", "disabled"}:
                errors.append(f"candidate {name}.cache_policy 必须是 enabled 或 disabled")
            if not re.fullmatch(r"[0-9a-f]{40}", str(item.get("model_revision", ""))):
                errors.append(f"candidate {name}.model_revision 必须是固定的 40 位提交号")
            if item.get("scheduling_policy") not in {"fcfs", "priority"}:
                errors.append(f"candidate {name}.scheduling_policy 不受支持")
            for key in ("max_num_seqs", "max_model_len"):
                if type(item.get(key)) is not int or item[key] <= 0:
                    errors.append(f"candidate {name}.{key} 必须是正整数")
            if type(item.get("enforce_eager")) is not bool:
                errors.append(f"candidate {name}.enforce_eager 必须是布尔值")
            utilization = _finite_number(item.get("gpu_memory_utilization"))
            if utilization is None or not 0 < utilization <= 1:
                errors.append(f"candidate {name}.gpu_memory_utilization 必须在 (0, 1] 内")
        if len(candidates) == 2 and roles.count("baseline") == roles.count("candidate") == 1:
            baseline_item = next(item for item in candidates.values() if item.get("role") == "baseline")
            candidate_item = next(item for item in candidates.values() if item.get("role") == "candidate")
            fixed_fields = (
                "model", "model_revision", "cache_dir", "served_model_name", "backend", "dtype",
                "max_model_len", "scheduling_policy", "gpu_memory_utilization", "enforce_eager",
            )
            changed = [key for key in fixed_fields if baseline_item.get(key) != candidate_item.get(key)]
            if changed:
                errors.append(f"基线与候选必须固定相同服务条件；不一致字段：{changed}")
            if baseline_item.get("cache_policy") != "disabled" or candidate_item.get("cache_policy") != "enabled":
                errors.append("基线须关闭缓存，候选须开启缓存")
            if not (
                type(baseline_item.get("max_num_seqs")) is int
                and type(candidate_item.get("max_num_seqs")) is int
                and candidate_item["max_num_seqs"] > baseline_item["max_num_seqs"]
            ):
                errors.append("候选 max_num_seqs 必须高于基线")
    errors.extend(_gate_errors(config.get("gates", {})))
    return errors


def validate_deployment_handoff(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an optional Part 02 handoff used only for provenance."""

    required = {
        "schema_version",
        "semantic_id",
        "run_label",
        "workload",
        "candidate_records",
        "decision",
        "rollback",
    }
    missing = sorted(required - set(record))
    if missing:
        raise KeyError(f"单卡部署交接清单缺少字段：{missing}")
    if record.get("schema_version") != HANDOFF_SCHEMA:
        raise ValueError(f"交接清单版本必须是 {HANDOFF_SCHEMA}")
    if record.get("semantic_id") != "INF-SINGLE-GPU-DEPLOY":
        raise ValueError("交接清单 semantic_id 必须是 INF-SINGLE-GPU-DEPLOY")
    ready = [
        dict(item)
        for item in record.get("candidate_records", [])
        if isinstance(item, Mapping) and item.get("handoff_status") == "ready_for_integrated_evaluation"
    ]
    return {
        "ready_candidates": ready,
        "handoff_decision": record.get("decision"),
        "rollback_ready": bool(record.get("rollback", {}).get("ready")),
    }


def validate_comparable_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Check that a measured candidate uses the frozen Part 05 workload."""

    fixed_config = ("model", "prompt_tokens", "generated_tokens", "batch", "concurrency")
    baseline_config = baseline.get("config", {})
    candidate_config = candidate.get("config", {})
    mismatches = [
        key for key in fixed_config if baseline_config.get(key) != candidate_config.get(key)
    ]
    return {"comparable": not mismatches, "mismatches": mismatches}


def evaluate_service_candidate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    output_fidelity_score: float | None,
    baseline_output_fidelity_score: float | None,
    task_quality: Mapping[str, Any] | None = None,
    baseline_task_quality: Mapping[str, Any] | None = None,
    artifact_verified: bool,
    rollback_ready: bool,
) -> dict[str, Any]:
    """Apply availability, task-quality, SLO, and capacity gates."""

    gate_errors = _gate_errors(gates)
    if gate_errors:
        raise ValueError("服务门槛配置无效：" + "; ".join(gate_errors))
    comparison = validate_comparable_runs(baseline, candidate)
    hard_failures: list[str] = []
    missing_evidence: list[str] = []
    failed_requests = _metric(candidate, "failed_requests")
    successful_requests = _metric(candidate, "successful_requests")
    total_requests = None
    if failed_requests is not None and successful_requests is not None:
        total_requests = failed_requests + successful_requests
    success_rate = successful_requests / total_requests if total_requests else None

    if not comparison["comparable"]:
        hard_failures.append("workload_mismatch")
    if _has_failure(candidate):
        hard_failures.append("backend_failure")
    if not artifact_verified:
        hard_failures.append("artifact_unverified")
    if not rollback_ready:
        hard_failures.append("rollback_unavailable")

    ttft_p95 = _metric(candidate, "ttft_ms", "p95")
    tpot_p95 = _metric(candidate, "tpot_ms", "p95")
    baseline_ttft_p95 = _metric(baseline, "ttft_ms", "p95")
    baseline_tpot_p95 = _metric(baseline, "tpot_ms", "p95")
    throughput = _metric(candidate, "throughput_tokens_per_s")
    peak_memory_mb = _metric(candidate, "peak_memory_mb")
    peak_memory_gib = peak_memory_mb / 1024 if peak_memory_mb is not None else None
    task_quality = task_quality or {}
    baseline_task_quality = baseline_task_quality or {}
    answer_accuracy = _rate(task_quality.get("expected_answer_accuracy"))
    complete_correct_rate = _rate(task_quality.get("complete_correct_rate"))
    response_completion_rate = _rate(task_quality.get("response_completion_rate"))
    baseline_answer_accuracy = _rate(baseline_task_quality.get("expected_answer_accuracy"))
    answer_accuracy_regression = (
        baseline_answer_accuracy - answer_accuracy
        if isinstance(baseline_answer_accuracy, (int, float))
        and isinstance(answer_accuracy, (int, float)) else None
    )
    evidence = {
        "success_rate": success_rate,
        # Exact text match remains visible, but task correctness is gated separately.
        "output_fidelity": output_fidelity_score,
        "output_fidelity_delta": (
            output_fidelity_score - baseline_output_fidelity_score
            if isinstance(output_fidelity_score, (int, float))
            and isinstance(baseline_output_fidelity_score, (int, float)) else None
        ),
        "valid_answer_format_rate": _rate(task_quality.get("valid_answer_format_rate")),
        "unique_answer_tasks": (
            task_quality["unique_answer_tasks"]
            if type(task_quality.get("unique_answer_tasks")) is int
            and task_quality["unique_answer_tasks"] > 0 else None
        ),
        "answer_repeat_consistency": _rate(task_quality.get("answer_repeat_consistency")),
        "expected_answer_accuracy": answer_accuracy,
        "complete_correct_rate": complete_correct_rate,
        "response_completion_rate": response_completion_rate,
        "truncated_response_rate": _rate(task_quality.get("truncated_response_rate")),
        "answer_accuracy_regression": answer_accuracy_regression,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "ttft_regression_ratio": (
            ttft_p95 / baseline_ttft_p95
            if ttft_p95 is not None and baseline_ttft_p95 not in (None, 0) else None
        ),
        "tpot_regression_ratio": (
            tpot_p95 / baseline_tpot_p95
            if tpot_p95 is not None and baseline_tpot_p95 not in (None, 0) else None
        ),
        "output_throughput_tps": throughput,
        "peak_memory_gib": peak_memory_gib,
    }
    for key, value in evidence.items():
        if value is None and key not in {"output_fidelity", "output_fidelity_delta"}:
            missing_evidence.append(key)

    failed_gates: list[str] = []
    if success_rate is not None and success_rate < gates["minimum_success_rate"]:
        failed_gates.append("success_rate")
    if answer_accuracy is not None and answer_accuracy < gates["minimum_answer_accuracy"]:
        failed_gates.append("answer_accuracy")
    if complete_correct_rate is not None and complete_correct_rate < gates["minimum_complete_correct_rate"]:
        failed_gates.append("complete_correct_rate")
    if response_completion_rate is not None and response_completion_rate < gates["minimum_response_completion_rate"]:
        failed_gates.append("response_completion_rate")
    if answer_accuracy_regression is not None and answer_accuracy_regression > gates["maximum_answer_accuracy_regression"]:
        failed_gates.append("answer_accuracy_regression")
    if ttft_p95 is not None and ttft_p95 > gates["maximum_ttft_p95_ms"]:
        failed_gates.append("ttft_p95")
    if tpot_p95 is not None and tpot_p95 > gates["maximum_tpot_p95_ms"]:
        failed_gates.append("tpot_p95")
    if evidence["ttft_regression_ratio"] is not None and evidence["ttft_regression_ratio"] > 1 + gates["maximum_ttft_regression_ratio"]:
        failed_gates.append("ttft_regression")
    if evidence["tpot_regression_ratio"] is not None and evidence["tpot_regression_ratio"] > 1 + gates["maximum_tpot_regression_ratio"]:
        failed_gates.append("tpot_regression")
    if throughput is not None and throughput < gates["minimum_output_throughput_tps"]:
        failed_gates.append("throughput")
    if peak_memory_gib is not None and peak_memory_gib > gates["maximum_peak_memory_gib"]:
        failed_gates.append("peak_memory")

    if hard_failures:
        decision = "reject"
    elif missing_evidence or failed_gates:
        decision = "tune"
    else:
        decision = "accept"
    return {
        "decision": decision,
        "hard_failures": hard_failures,
        "missing_evidence": missing_evidence,
        "failed_gates": failed_gates,
        "comparison": comparison,
        "evidence": evidence,
    }


def select_service_candidate(
    decisions: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one accepted candidate by TPOT, then TTFT, then throughput."""

    accepted = [name for name, item in decisions.items() if item.get("decision") == "accept"]
    if not accepted:
        fallback = "tune" if any(item.get("decision") == "tune" for item in decisions.values()) else "reject"
        return {"decision": fallback, "selected_candidate": None, "accepted_candidates": []}

    def rank(name: str) -> tuple[float, float, float]:
        record = results[name]
        tpot = _metric(record, "tpot_ms", "p95")
        ttft = _metric(record, "ttft_ms", "p95")
        throughput = _metric(record, "throughput_tokens_per_s")
        return (
            tpot if tpot is not None else float("inf"),
            ttft if ttft is not None else float("inf"),
            -(throughput if throughput is not None else 0.0),
        )

    selected = min(accepted, key=rank)
    return {"decision": "accept", "selected_candidate": selected, "accepted_candidates": sorted(accepted)}


def missing_candidate_results(
    configured_candidates: Mapping[str, Any],
    measured_results: Mapping[str, Any],
) -> list[str]:
    """Return configured candidates that still need a measured result."""

    return sorted(set(configured_candidates) - set(measured_results))


def compare_output_fidelity(
    baseline_record: Mapping[str, Any], candidate_record: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare exact request text as a descriptive signal, not task correctness."""

    def request_outputs(record: Mapping[str, Any]) -> dict[tuple[str, ...], list[str]]:
        runs = record.get("runs")
        run_records = runs if isinstance(runs, list) and runs else [record]
        collected: dict[tuple[str, ...], list[str]] = {}
        for run_index, run in enumerate(run_records):
            if not isinstance(run, Mapping):
                continue
            rows = run.get("results")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping) or row.get("ok") is not True:
                    continue
                key = (str(row.get("request_id", "")), str(row.get("prompt", "")))
                collected.setdefault((str(run_index), *key), []).append(str(row.get("output_text", "")))
        return collected

    baseline = request_outputs(baseline_record)
    candidate = request_outputs(candidate_record)
    keys = sorted(set(baseline) & set(candidate))
    compared = sum(min(len(baseline[key]), len(candidate[key])) for key in keys)
    matched = sum(
        sum(left == right for left, right in zip(sorted(baseline[key]), sorted(candidate[key])))
        for key in keys
    )
    return {
        "status": "measured" if compared else "missing_request_outputs",
        "matched_requests": matched,
        "compared_requests": compared,
        "score": matched / compared if compared else None,
    }


def build_delivery_record(
    *,
    run_label: str,
    handoff_path: str | Path | None = None,
    workload: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable final record without hiding candidate-level evidence."""

    source = Path(handoff_path) if handoff_path else None
    digest = hashlib.sha256(source.read_bytes()).hexdigest() if source and source.is_file() else None
    return {
        "schema_version": RESULT_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "source_handoff": (
            {"path": str(source), "sha256": digest} if source else None
        ),
        "workload": dict(workload),
        "candidate_results": {name: dict(value) for name, value in results.items()},
        "candidate_decisions": {name: dict(value) for name, value in decisions.items()},
        "decision": selection.get("decision"),
        "selected_candidate": selection.get("selected_candidate"),
        "rollback_candidate": "baseline",
        "evidence_level": "integrated_system_evidence",
    }


def save_delivery_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Save a delivery decision without silently replacing an earlier run."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"结果已存在，请更换 run_label：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
