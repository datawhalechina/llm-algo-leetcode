"""Evidence contracts and delivery decisions for Part 05 performance projects."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "performance-optimization-delivery-config/v2"
MEASUREMENT_SCHEMA = "performance-delivery-measurement/v1"
RESULT_SCHEMA = "performance-optimization-delivery/v2"
SEMANTIC_ID = "PERF-INTEGRATED-DELIVERY"


def load_json_record(path: str | Path) -> dict[str, Any]:
    """Load one explicit JSON object; never search for a substitute file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"性能记录不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"性能记录必须是 JSON object：{source}")
    return dict(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_record_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    root = Path(project_root).resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"证据路径须位于项目根目录内：{value}")
    return resolved


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _gate_errors(gates: Mapping[str, Any]) -> list[str]:
    if not isinstance(gates, Mapping):
        return ["gates 必须是映射"]
    errors = []
    if not isinstance(gates.get("primary_metric"), str) or not gates["primary_metric"].strip():
        errors.append("primary_metric 必须是非空指标名")
    if gates.get("primary_direction", "lower") not in ("lower", "higher"):
        errors.append("primary_direction 必须是 lower 或 higher")
    value = gates.get("minimum_primary_improvement_ratio")
    if not _finite(value) or value < 0 or (gates.get("primary_direction", "lower") == "lower" and value >= 1):
        errors.append("minimum_primary_improvement_ratio 须为有限非负数；lower 方向须小于 1")
    if type(gates.get("minimum_samples")) is not int or gates["minimum_samples"] <= 0:
        errors.append("minimum_samples 必须是正整数")
    thresholds = gates.get("maximum_guardrail_regression_ratio")
    if not isinstance(thresholds, Mapping):
        errors.append("maximum_guardrail_regression_ratio 必须是映射")
    elif any(not isinstance(key, str) or not key.strip() or not _finite(value) or value < 0 for key, value in thresholds.items()):
        errors.append("守护指标名须非空，回归阈值须为有限非负数")
    for key in ("require_composition", "require_reprofile", "require_rollback_ready"):
        if key in gates and type(gates[key]) is not bool:
            errors.append(f"{key} 必须是布尔值")
    return errors


def _samples(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = record.get("samples")
    return values if isinstance(values, list) and all(isinstance(item, Mapping) for item in values) else []


def validate_delivery_config(
    config: Mapping[str, Any], *, require_run_label: bool = True,
    require_evidence_paths: bool = False,
) -> list[str]:
    """Check project identity and configuration shape before reading evidence."""

    errors: list[str] = []
    if not isinstance(config, Mapping):
        return ["配置必须是映射"]
    for section in ("baseline_measurement", "gates", "outputs", "rollback", "composition", "reprofile", "capture_manifest"):
        if section in config and not isinstance(config[section], Mapping):
            errors.append(f"{section} 必须是映射")
    if errors:
        return errors
    capture = config.get("capture_manifest")
    if capture and (not isinstance(capture.get("path"), str) or not capture["path"]
                    or not isinstance(capture.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", capture["sha256"])):
        errors.append("capture_manifest 须包含路径及有效 SHA256")
    if config.get("schema_version") != CONFIG_SCHEMA:
        errors.append(f"schema_version 必须是 {CONFIG_SCHEMA}")
    if config.get("semantic_id") != SEMANTIC_ID:
        errors.append(f"semantic_id 必须是 {SEMANTIC_ID}")
    if config.get("environment_profile") not in ("PERF-CUDA-MECHANISM", "PERF-DELIVERY"):
        errors.append("environment_profile 必须是 PERF-CUDA-MECHANISM 或 PERF-DELIVERY")
    label = config.get("run_label")
    if require_run_label and (
        not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", label) or label == "replace_with_unique_label"
    ):
        errors.append("run_label 必须替换为唯一值")
    workload = config.get("workload_identity")
    if not isinstance(workload, Mapping) or not workload:
        errors.append("workload_identity 必须是非空 JSON object")
    baseline = config.get("baseline_measurement", {})
    if not isinstance(baseline, Mapping):
        errors.append("baseline_measurement 必须是映射")
    elif require_evidence_paths and not str(baseline.get("result_path", "")).strip():
        errors.append("baseline_measurement.result_path 不能为空")

    order = config.get("candidate_order")
    candidates = config.get("candidates")
    if not isinstance(order, list):
        errors.append("candidate_order 必须是列表")
        order = []
    if any(not isinstance(name, str) or not name.strip() for name in order):
        errors.append("candidate_order 中每项须为非空候选名")
        order = []
    if len(order) != len(set(order)):
        errors.append("candidate_order 不能包含重复候选")
    if not isinstance(candidates, Mapping):
        errors.append("candidates 必须是映射")
        candidates = {}
    unknown = [name for name in order if name not in candidates]
    if unknown:
        errors.append(f"candidate_order 中存在未配置候选：{unknown}")
    for name in order:
        item = candidates.get(name, {})
        if not isinstance(item, Mapping) or (require_evidence_paths and not str(item.get("result_path", "")).strip()):
            errors.append(f"候选 {name} 缺少 result_path")
        elif not isinstance(item.get("conflicts_with", []), list) or any(not isinstance(value, str) for value in item.get("conflicts_with", [])):
            errors.append(f"候选 {name} 的 conflicts_with 须为候选名列表")

    gates = config.get("gates", {})
    errors.extend(_gate_errors(gates))
    for key in (
        "primary_metric",
        "minimum_primary_improvement_ratio",
        "maximum_guardrail_regression_ratio",
        "minimum_samples",
    ):
        if key not in gates:
            errors.append(f"gates 缺少 {key}")
    if gates.get("primary_direction", "lower") not in ("lower", "higher"):
        errors.append("gates.primary_direction 必须是 lower 或 higher")
    if not isinstance(gates.get("maximum_guardrail_regression_ratio", {}), Mapping):
        errors.append("gates.maximum_guardrail_regression_ratio 必须是映射")
    if not str(config.get("outputs", {}).get("directory", "")).strip():
        errors.append("outputs.directory 不能为空")
    else:
        directory = config["outputs"]["directory"]
        if not isinstance(directory, str) or Path(directory).is_absolute() or ".." in Path(directory).parts:
            errors.append("outputs.directory 必须是仓库内相对路径")
    rollback = config.get("rollback", {})
    if not isinstance(rollback, Mapping) or not isinstance(rollback.get("ready"), bool):
        errors.append("rollback.ready 必须是布尔值")
    elif rollback.get("ready") and not str(rollback.get("evidence_path", "")).strip():
        errors.append("rollback.ready 为 true 时必须提供 rollback.evidence_path")
    context_sources = config.get("context_sources", [])
    if not isinstance(context_sources, list):
        errors.append("context_sources 必须是列表")
    elif any(not isinstance(item, Mapping) or not item.get("label") or not item.get("path") for item in context_sources):
        errors.append("context_sources 每项都必须包含 label 和 path")
    return errors


def validate_composition_order(
    order: Sequence[str], candidates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Reject unknown, duplicated or conflicting candidate combinations."""

    issues: list[str] = []
    if len(order) != len(set(order)):
        issues.append("duplicate_candidate")
    unknown = [name for name in order if name not in candidates]
    if unknown:
        issues.append(f"unknown_candidates:{','.join(unknown)}")
    enabled: set[str] = set()
    for name in order:
        if name not in candidates:
            continue
        conflicts = set(candidates[name].get("conflicts_with", []))
        reverse_conflicts = {
            prior for prior in enabled
            if name in set(candidates[prior].get("conflicts_with", []))
        }
        conflict_names = (conflicts & enabled) | reverse_conflicts
        if conflict_names:
            issues.append(f"conflict:{name}:{','.join(sorted(conflict_names))}")
        enabled.add(name)
    return {"valid": not issues, "issues": issues, "ordered_candidates": list(order)}


def _validate_measurement_record(record: Mapping[str, Any], *, minimum_samples: int) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != MEASUREMENT_SCHEMA:
        errors.append(f"schema_version 必须是 {MEASUREMENT_SCHEMA}")
    if not isinstance(record.get("workload_identity"), Mapping):
        errors.append("workload_identity 必须是 JSON object")
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) < minimum_samples:
        errors.append(f"samples 至少需要 {minimum_samples} 个配对样本")
        return errors
    pair_ids = [item.get("pair_id") for item in samples if isinstance(item, Mapping)]
    if len(pair_ids) != len(samples) or any(value is None for value in pair_ids):
        errors.append("每个 sample 都必须包含 pair_id")
    if any(not isinstance(value, str) or not value for value in pair_ids):
        errors.append("pair_id 必须是非空字符串")
    elif len(pair_ids) != len(set(pair_ids)):
        errors.append("samples 中 pair_id 不能重复")
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            errors.append(f"samples[{index}] 必须是映射")
            continue
        metrics = sample.get("metrics", {})
        if not isinstance(metrics, Mapping) or not metrics:
            errors.append(f"samples[{index}].metrics 不能为空")
            continue
        for metric, value in metrics.items():
            if not _finite(value) or value < 0:
                errors.append(f"samples[{index}].metrics.{metric} 必须是有限非负数值")
    return errors


def validate_comparable_result(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], expected_workload: Mapping[str, Any]
) -> dict[str, Any]:
    """Check workload identity, measurement shape, correctness and execution path."""

    baseline_identity = baseline.get("workload_identity", {})
    candidate_identity = candidate.get("workload_identity", {})
    baseline_identity = baseline_identity if isinstance(baseline_identity, Mapping) else {}
    candidate_identity = candidate_identity if isinstance(candidate_identity, Mapping) else {}
    identity_mismatches = [
        key for key, value in expected_workload.items()
        if baseline_identity.get(key) != value or candidate_identity.get(key) != value
    ]
    baseline_pairs = [(item.get("pair_id"), item.get("seed")) for item in _samples(baseline)]
    candidate_pairs = [(item.get("pair_id"), item.get("seed")) for item in _samples(candidate)]
    pair_mismatch = baseline_pairs != candidate_pairs
    execution_verified = all(isinstance(record.get("execution_path"), Mapping) and record["execution_path"].get("verified") is True for record in (baseline, candidate))
    correctness_passed = all(isinstance(record.get("correctness"), Mapping) and record["correctness"].get("passed") is True for record in (baseline, candidate))
    return {
        "comparable": not identity_mismatches and not pair_mismatch and baseline_identity == candidate_identity,
        "identity_mismatches": identity_mismatches,
        "pair_ids_match": not pair_mismatch,
        "execution_path_verified": execution_verified,
        "correctness_passed": correctness_passed,
    }


def evaluate_performance_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gates: Mapping[str, Any],
    expected_workload: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate paired measurements; all positive improvement ratios mean better."""

    errors = _gate_errors(gates)
    if errors:
        raise ValueError("性能门槛配置错误：" + "; ".join(errors))
    minimum_samples = gates["minimum_samples"]
    baseline_errors = _validate_measurement_record(baseline, minimum_samples=minimum_samples)
    candidate_errors = _validate_measurement_record(candidate, minimum_samples=minimum_samples)
    comparison = validate_comparable_result(baseline, candidate, expected_workload)
    hard_failures: list[str] = []
    if not comparison["comparable"]:
        hard_failures.append("workload_or_pair_mismatch")
    if not comparison["execution_path_verified"]:
        hard_failures.append("execution_path_unverified")
    if not comparison["correctness_passed"]:
        hard_failures.append("correctness_failed")
    errors = baseline_errors + candidate_errors
    if errors:
        hard_failures.append("invalid_measurement_record")
        return {
            "decision": "reject", "hard_failures": hard_failures, "record_errors": errors,
            "missing_evidence": [], "failed_gates": [], "comparison": comparison,
            "primary_metric": gates["primary_metric"], "primary_improvement_ratio": None, "guardrails": {},
        }

    primary = gates["primary_metric"]
    base_values = [item.get("metrics", {}).get(primary) for item in baseline.get("samples", [])]
    candidate_values = [item.get("metrics", {}).get(primary) for item in candidate.get("samples", [])]
    missing_metrics = (
        not base_values
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in base_values + candidate_values)
    )
    failed_gates: list[str] = []
    missing_evidence: list[str] = []
    primary_improvement = None
    if not missing_metrics and any(value <= 0 for value in base_values):
        missing_metrics = True
        missing_evidence.append(f"positive_baseline_metric:{primary}")
    if missing_metrics:
        missing_evidence.append(f"paired_metric:{primary}")
    elif len(base_values) != len(candidate_values):
        hard_failures.append("paired_sample_count_mismatch")
    elif gates.get("primary_direction", "lower") == "lower":
        pair_ratios = [1 - candidate_value / base_value for base_value, candidate_value in zip(base_values, candidate_values) if base_value != 0]
        primary_improvement = statistics.median(pair_ratios) if pair_ratios else None
    else:
        pair_ratios = [candidate_value / base_value - 1 for base_value, candidate_value in zip(base_values, candidate_values) if base_value != 0]
        primary_improvement = statistics.median(pair_ratios) if pair_ratios else None
    if primary_improvement is None and not missing_metrics:
        missing_evidence.append(f"nonzero_baseline_metric:{primary}")
    if primary_improvement is not None and primary_improvement < gates["minimum_primary_improvement_ratio"]:
        failed_gates.append("primary_improvement")

    guardrail_results: dict[str, Any] = {}
    thresholds = gates.get("maximum_guardrail_regression_ratio", {})
    for metric, maximum_regression in thresholds.items():
        base_metric = [item.get("metrics", {}).get(metric) for item in baseline.get("samples", [])]
        candidate_metric = [item.get("metrics", {}).get(metric) for item in candidate.get("samples", [])]
        if not base_metric or not candidate_metric or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in base_metric + candidate_metric
        ):
            missing_evidence.append(f"guardrail_metric:{metric}")
            guardrail_results[metric] = {"status": "missing"}
            continue
        base_median = statistics.median(base_metric)
        candidate_median = statistics.median(candidate_metric)
        regression = max(0.0, (candidate_median - base_median) / max(abs(base_median), 1e-12))
        guardrail_results[metric] = {
            "baseline_median": base_median,
            "candidate_median": candidate_median,
            "regression_ratio": regression,
            "maximum_regression_ratio": maximum_regression,
        }
        if regression > maximum_regression:
            failed_gates.append(f"guardrail:{metric}")

    if hard_failures:
        decision = "reject"
    elif missing_evidence or failed_gates:
        decision = "tune"
    else:
        decision = "accept"
    return {
        "decision": decision,
        "hard_failures": hard_failures,
        "record_errors": errors,
        "missing_evidence": missing_evidence,
        "failed_gates": failed_gates,
        "comparison": comparison,
        "primary_metric": primary,
        "primary_improvement_ratio": primary_improvement,
        "guardrails": guardrail_results,
    }


def decide_delivery(
    *,
    rollback: Mapping[str, Any],
    candidate_order: Sequence[str],
    candidate_decisions: Mapping[str, Mapping[str, Any]],
    composition_decision: Mapping[str, Any] | None,
    composition_check: Mapping[str, Any],
    reprofile: Mapping[str, Any] | None,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose the best measured route, allowing a single change to ship alone."""

    hard_failures: list[str] = []
    missing: list[str] = []
    if not rollback.get("ready") and gates.get("require_rollback_ready", True):
        missing.append("project_rollback_unavailable")
    if not candidate_order:
        missing.append("no_project_candidates_configured")
    if composition_decision is None and gates.get("require_composition", True):
        missing.append("composition_measurement_missing")

    accepted_routes = [
        {"route": name, "decision": item}
        for name in candidate_order
        if (item := candidate_decisions.get(name, {})).get("decision") == "accept"
    ]
    if composition_decision and composition_decision.get("decision") == "accept" and composition_check.get("valid") is True:
        accepted_routes.append({"route": "composition", "decision": composition_decision})
    selected_route = max(
        accepted_routes,
        key=lambda item: item["decision"].get("primary_improvement_ratio") or 0.0,
        default=None,
    )
    if selected_route is None and candidate_decisions and all(
        item.get("decision") == "reject" for item in candidate_decisions.values()
    ) and (composition_decision is None or composition_decision.get("decision") == "reject"):
        hard_failures.append("all_project_routes_rejected")
    elif selected_route is None:
        missing.append("no_project_route_passed_performance_and_quality_gates")

    if gates.get("require_reprofile", True):
        required_reprofile = ("bottleneck_before", "bottleneck_after", "evidence_path")
        absent = [key for key in required_reprofile if not (reprofile or {}).get(key)]
        if absent:
            missing.extend(f"reprofile:{key}" for key in absent)
        after = (reprofile or {}).get("bottleneck_after")
        if selected_route and isinstance(after, Mapping) and not after.get(selected_route["route"]):
            missing.append("reprofile:selected_route_missing")

    if hard_failures:
        decision = "reject"
        reason = "hard_delivery_gate_failed"
    elif missing:
        decision = "tune"
        reason = "required_delivery_evidence_missing_or_not_qualified"
    else:
        decision = "accept"
        reason = "selected_route_and_delivery_gates_passed"
    return {
        "decision": decision,
        "reason": reason,
        "selected_route": selected_route["route"] if selected_route else None,
        "accepted_routes": [item["route"] for item in accepted_routes],
        "hard_failures": sorted(set(hard_failures)),
        "missing_evidence": sorted(set(missing)),
    }


def review_delivery(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Review the project's own workload, measurements and rollback evidence.

    This function never launches a model or GPU measurement. Candidate files
    must be produced by their execution environments and use the measurement
    schema defined by this project.
    """

    errors = validate_delivery_config(config)
    if errors:
        raise ValueError("项目配置未通过检查：" + "; ".join(errors))
    root = Path(project_root).resolve()
    order = config["candidate_order"]
    candidates = config["candidates"]
    composition_check = validate_composition_order(order, candidates)
    candidate_decisions: dict[str, Any] = {}
    measurement_sources: list[dict[str, str]] = []
    context_sources: list[dict[str, str]] = []
    composition_decision = None
    reprofile = None
    missing_inputs: list[str] = []
    if config.get("capture_status") != "complete":
        missing_inputs.append("project_capture_incomplete")
    capture = config.get("capture_manifest")
    if capture:
        capture_path = _resolve_record_path(root, capture["path"])
        if not capture_path.is_file() or sha256_file(capture_path) != capture.get("sha256"):
            missing_inputs.append("project_capture_manifest_hash_mismatch")
        else:
            measurement_sources.append({"role": "project_capture_manifest", "path": str(capture_path.relative_to(root)), "sha256": capture["sha256"]})
    for source in config.get("context_sources", []):
        source_path = _resolve_record_path(root, source["path"])
        context_sources.append({
            "label": source["label"],
            "path": source["path"],
            "status": "available" if source_path.is_file() else "not_provided",
            **({"sha256": sha256_file(source_path)} if source_path.is_file() else {}),
        })
    baseline_measurement = None
    baseline_ref = config["baseline_measurement"].get("result_path", "")
    baseline_path = _resolve_record_path(root, baseline_ref) if baseline_ref else None
    if baseline_path is not None and baseline_path.is_file():
        baseline_measurement = load_json_record(baseline_path)
        measurement_sources.append({
            "role": "project_baseline",
            "path": str(baseline_path.relative_to(root)),
            "sha256": sha256_file(baseline_path),
        })
    else:
        missing_inputs.append("project_baseline_measurement")

    if order:
        if baseline_measurement is None:
            missing_inputs.append("paired_candidate_measurements_not_evaluated")
        else:
            for name in order:
                candidate_ref = candidates[name].get("result_path")
                candidate_path = _resolve_record_path(root, candidate_ref) if candidate_ref else None
                if candidate_path is None or not candidate_path.is_file():
                    missing_inputs.append(f"candidate_measurement:{name}")
                    continue
                candidate = load_json_record(candidate_path)
                measurement_sources.append({
                    "role": f"candidate:{name}",
                    "path": str(candidate_path.relative_to(root)),
                    "sha256": sha256_file(candidate_path),
                })
                candidate_decisions[name] = evaluate_performance_candidate(
                    baseline_measurement, candidate, config["gates"], config["workload_identity"]
                )

        composition_ref = config.get("composition", {}).get("result_path")
        if composition_ref and baseline_measurement is not None:
            composition_path = _resolve_record_path(root, composition_ref)
            if composition_path.is_file():
                composition_record = load_json_record(composition_path)
                measurement_sources.append({
                    "role": "composition",
                    "path": str(composition_path.relative_to(root)),
                    "sha256": sha256_file(composition_path),
                })
                if not composition_check["valid"]:
                    composition_decision = {"decision": "reject", "hard_failures": ["composition_conflicts"]}
                elif composition_record.get("components") != order:
                    composition_decision = {
                        "decision": "reject",
                        "hard_failures": ["composition_components_do_not_match_configured_order"],
                    }
                else:
                    composition_decision = evaluate_performance_candidate(
                        baseline_measurement, composition_record, config["gates"], config["workload_identity"]
                    )
            elif config["gates"].get("require_composition", True):
                missing_inputs.append("composition_measurement")
        elif config["gates"].get("require_composition", True):
            missing_inputs.append("composition_measurement")
    else:
        missing_inputs.append("no_project_candidates_configured")

    reprofile_ref = config.get("reprofile", {}).get("result_path")
    if reprofile_ref:
        reprofile_path = _resolve_record_path(root, reprofile_ref)
        if reprofile_path.is_file():
            reprofile = load_json_record(reprofile_path)
            if reprofile.get("workload_identity") != config["workload_identity"]:
                missing_inputs.append("post_composition_reprofile_workload_mismatch")
            evidence_ref = reprofile.get("evidence_path")
            evidence_path = _resolve_record_path(root, evidence_ref) if evidence_ref else None
            if evidence_path is None or not evidence_path.is_file():
                missing_inputs.append("post_composition_reprofile_evidence_file")
            else:
                measurement_sources.append({
                    "role": "post_composition_reprofile_evidence",
                    "path": str(evidence_path.relative_to(root)),
                    "sha256": sha256_file(evidence_path),
                })
            measurement_sources.append({
                "role": "post_composition_reprofile",
                "path": str(reprofile_path.relative_to(root)),
                "sha256": sha256_file(reprofile_path),
            })
        else:
            missing_inputs.append("post_composition_reprofile")

    rollback = config.get("rollback", {})
    rollback_evidence = rollback.get("evidence_path")
    if rollback.get("ready"):
        rollback_path = _resolve_record_path(root, rollback_evidence)
        if rollback_path.is_file():
            rollback_record = load_json_record(rollback_path)
            if rollback_record.get("rollback_ready") is not True or not str(rollback_record.get("restore_action", "")).strip():
                missing_inputs.append("project_rollback_evidence_invalid")
            measurement_sources.append({
                "role": "project_rollback",
                "path": str(rollback_path.relative_to(root)),
                "sha256": sha256_file(rollback_path),
            })
        else:
            missing_inputs.append("project_rollback_evidence")

    decision = decide_delivery(
        rollback=rollback,
        candidate_order=order,
        candidate_decisions=candidate_decisions,
        composition_decision=composition_decision,
        composition_check=composition_check,
        reprofile=reprofile,
        gates=config["gates"],
    )
    decision["missing_evidence"] = sorted(set(decision["missing_evidence"] + missing_inputs))
    if decision["decision"] == "accept" and decision["missing_evidence"]:
        decision["decision"] = "tune"
        decision["reason"] = "required_delivery_evidence_missing_or_not_qualified"

    return {
        "schema_version": RESULT_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": config["run_label"],
        "environment_profile": config["environment_profile"],
        "workload_identity": dict(config["workload_identity"]),
        "rollback_ready": bool(rollback.get("ready")),
        "context_sources": context_sources,
        "measurement_sources": measurement_sources,
        "candidate_decisions": candidate_decisions,
        "composition_check": composition_check,
        "composition_decision": composition_decision,
        "reprofile": reprofile,
        "decision": decision["decision"],
        "decision_detail": decision,
    }


def save_json_record_once(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Write a result only to a new path; never overwrite an earlier run."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"结果已存在，不覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    return target
