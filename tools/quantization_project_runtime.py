"""Shared decisions and evidence lifecycle for quantization project notebooks."""

from __future__ import annotations

from math import isclose, isfinite
from pathlib import PurePosixPath
from typing import Any


def artifact_gate_plan(root, run_label: str, upstream_run_label: str, stage: str = "prepare") -> dict:
    """解析固定运行合同与路径，不读取模型、使用GPU或创建结果。"""
    from pathlib import Path
    import re

    if stage not in {"prepare", "measure", "close"}:
        raise ValueError("stage must be prepare, measure or close")
    for label in (run_label, upstream_run_label):
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", label):
            raise ValueError("run labels must be safe, nonempty filename components")
    if stage != "prepare" and run_label == "replace_with_unique_label":
        raise ValueError("measure / close require a unique RUN_LABEL")
    root = Path(root).resolve()
    directory = root / "benchmarks/results/82_quantization_artifact_evaluation"
    return {
        "root": root, "stage": stage, "run_label": run_label, "upstream_run_label": upstream_run_label,
        "source_path": root / f"benchmarks/results/40_llmcompressor_gptq_awq_{upstream_run_label}.json",
        "loader_path": directory / f"loader_{run_label}.json",
        "raw_path": directory / f"82_artifact_gate_{run_label}.json",
        "report_path": directory / f"82_artifact_gate_{run_label}_companion.json",
        "reference_path": directory / f"82_artifact_gate_{run_label}_reference.json",
        "receipt_path": directory / f"82_artifact_gate_{run_label}_registration.json",
        "contract": {"semantic_id": "QUANT-ARTIFACT-GATE", "schema_version": "quantization-evidence/v1",
                     "protocol_version": "artifact-gate-lifecycle/v2", "environment_profile": "INF-VLLM-011-CU128",
                     "backend": "vllm", "max_model_len": 512, "max_tokens": 16,
                     "gpu_memory_utilization": 0.75, "max_memory_mb": 0.0,
                     "memory_measurement": "device_free_memory_delta_not_peak"},
    }


def _gate_read_json(path):
    import json
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return value


def _gate_digest(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_file_reference(plan, path):
    return {"path": str(path.resolve().relative_to(plan["root"])), "sha256": _gate_digest(path)}


def prepare_artifact_gate(plan: dict) -> dict:
    """只读校验唯一上游、获选候选和磁盘内容；环境GPU预检留在measure。"""
    from pathlib import Path
    from tools.quantization_runtime import validate_candidate_bundle

    issues, bundle, snapshot = [], None, None
    try:
        source = _gate_read_json(plan["source_path"])
        if (source.get("chapter") != "40" or source.get("run_label") != plan["upstream_run_label"]
                or source.get("run_mode") != "real_gpu" or source.get("failure")
                or source.get("decision") != "accept_weight_ptq_candidate"):
            raise ValueError("upstream identity, execution or accepted decision does not match")
        selected = source.get("mechanism_metrics", {}).get("selected_candidate")
        if selected not in {"gptq", "awq", "rtn"} or not isinstance(source.get("candidates", {}).get(selected), dict):
            raise ValueError("upstream selected candidate is invalid")
        bundle_path = plan["root"] / "benchmarks/artifacts/40_gptq_awq" / plan["upstream_run_label"] / selected / "candidate_bundle.json"
        bundle = _gate_read_json(bundle_path)
        validation = validate_candidate_bundle(bundle, project_root=plan["root"])
        if not validation["ready"]:
            raise ValueError("; ".join(validation["issues"]))
        if Path(validation["artifact_path"]).resolve() != bundle_path.parent.resolve():
            raise ValueError("bundle artifact_path differs from selected candidate directory")
        producer = bundle["evidence"]
        declared_source = Path(producer["source_result_path"])
        if not declared_source.is_absolute():
            declared_source = plan["root"] / declared_source
        if declared_source.resolve() != plan["source_path"].resolve() or producer["producer_semantic_id"] != "QUANT-WEIGHT-PTQ":
            raise ValueError("bundle source lineage does not match the selected upstream")
        workload, dataset = source["workload"], source["dataset"]
        if bundle["model"] != {"id": workload["model_id"], "revision": workload["resolved_model_commit"]} or bundle["quantization"]["method"] != selected:
            raise ValueError("bundle model or method does not match upstream")
        for section, prefix in (("calibration", "calibration"), ("evaluation", "evaluation")):
            recorded = bundle[section]
            expected = {"dataset": dataset["dataset_id"], "revision": dataset["revision"], "config": dataset.get("config"),
                        "split": dataset[prefix + "_split"], "sample_ids_hash": dataset[prefix + "_ids_hash"],
                        "sample_count": dataset[prefix + "_samples"]}
            if any(recorded.get(key) != value for key, value in expected.items()) or not _positive_int(recorded.get("sample_count")):
                raise ValueError("bundle dataset identity/count differs from upstream")
        evaluation = bundle["evaluation"]
        expected_quality = {"direction": "lower_is_better", "metric_name": "token_weighted_nll",
                            "baseline_value": source["baseline"]["token_weighted_nll"],
                            "candidate_value": source["candidates"][selected]["token_weighted_nll"],
                            "max_degradation_ratio": workload["max_nll_ratio_vs_bf16"]}
        if any(evaluation.get(key) != value for key, value in expected_quality.items()):
            raise ValueError("bundle quality values do not match the upstream selected candidate")
        snapshot = {"source": _gate_file_reference(plan, plan["source_path"]),
                    "bundle": _gate_file_reference(plan, bundle_path),
                    "artifact_path": str(bundle_path.parent.resolve()), "artifact_id": bundle["artifact_id"],
                    "model_revision": bundle["model"]["revision"], "selected_candidate": selected}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        issues.append(str(error))
    return {"ready": not issues, "issues": issues, "bundle": bundle, "snapshot": snapshot,
            "environment_profile": plan["contract"]["environment_profile"], "gpu_preflight_executed": False}


def _require_gate_inputs(plan):
    prepared = prepare_artifact_gate(plan)
    if not prepared["ready"]:
        raise ValueError("artifact inputs invalid: " + "; ".join(prepared["issues"]))
    return prepared


def artifact_loader_invocation(plan: dict, prepared: dict) -> dict:
    """记录加载对象及固定调用条件，读取时逐项重算，不使用当前Kernel缓存。"""
    return {"run_label": plan["run_label"], "upstream_run_label": plan["upstream_run_label"],
            "inputs": prepared["snapshot"], "contract": plan["contract"]}


def _validate_gate_loader(plan, prepared, loader):
    if loader.get("schema_version") != "quantization-artifact-loader/v2":
        raise ValueError("loader lacks the v2 input identity; use a new label for a new measurement")
    if loader.get("invocation") != artifact_loader_invocation(plan, prepared):
        raise ValueError("loader input identity or fixed invocation changed")
    if loader.get("backend") != "vllm" or loader.get("load_status") not in {"loaded", "failed", "roundtrip_failed"}:
        raise ValueError("loader backend/status is invalid")
    if loader.get("status") not in {"ok", "failed"}:
        raise ValueError("loader execution status is invalid")
    if loader.get("status") == "ok" and (not loader.get("backend_version") or loader.get("hardware", {}).get("cuda_available") is not True):
        raise ValueError("successful loader lacks runtime/hardware identity")


def _artifact_gate_payload(plan, prepared, loader):
    """从当前磁盘输入与已保存加载记录重算唯一准入结论。"""
    bundle = prepared["bundle"]
    decision = decide_artifact_adoption(validate_candidate_contract(bundle), evaluate_candidate_evidence(bundle), loader,
                                        max_memory_mb=plan["contract"]["max_memory_mb"])
    return {"schema_version": plan["contract"]["schema_version"], "semantic_id": plan["contract"]["semantic_id"],
            "run_label": plan["run_label"], "config": plan["contract"], "manifest": bundle, "decision": decision,
            "lineage": {**prepared["snapshot"], "loader": _gate_file_reference(plan, plan["loader_path"])},
            "loader": loader,
            "quality": {**bundle["evaluation"], "eval_split": bundle["evaluation"]["split"],
                        "peak_memory_mb": loader.get("peak_memory_mb"), "roundtrip_quality": loader.get("roundtrip_quality")},
            "compatibility": {"backend": "vllm", "version": loader.get("backend_version"),
                              "loader_status": loader.get("load_status"), "kernel": loader.get("kernel"),
                              "fallback": loader.get("fallback"), "failure": loader.get("failure")},
            "hardware": loader.get("hardware", {}),
            "gate": {"ready": decision["decision"] == "accept", "issues": [] if decision["decision"] == "accept" else [decision["reason"]],
                     "evidence_level": "real_project_benchmark"}}


def _gate_save_or_verify(path, payload):
    """内容一致时只读复用；原子发布完整JSON，不覆盖任何同名文件。"""
    import json
    import os
    import tempfile

    if path.exists():
        if _gate_read_json(path) != payload:
            raise FileExistsError(f"evidence conflict; choose another RUN_LABEL: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".record-", dir=path.parent) as temporary:
        from pathlib import Path
        staging = Path(temporary) / "record.json"
        with staging.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, path)
        except FileExistsError:
            if _gate_read_json(path) != payload:
                raise FileExistsError(f"evidence conflict; choose another RUN_LABEL: {path}")
            return False
    return True


def save_artifact_gate_measurement(plan: dict, prepared: dict, loader: dict) -> dict:
    """measure只发布加载记录与原始准入证据，不生成报告或交接。"""
    if plan["stage"] != "measure":
        raise ValueError("measurement writes require the measure stage")
    current = _require_gate_inputs(plan)
    if prepared.get("snapshot") != current["snapshot"]:
        raise ValueError("artifact inputs changed during loader execution")
    _validate_gate_loader(plan, current, loader)
    loader_written = _gate_save_or_verify(plan["loader_path"], loader)
    payload = _artifact_gate_payload(plan, current, loader)
    raw_written = _gate_save_or_verify(plan["raw_path"], payload)
    return {"status": "measured" if loader_written or raw_written else "already_measured", "decision": payload["decision"]}


def _gate_report(plan, payload):
    from tools.quantization_result_schema import artifact_gate_record
    from tools.project_asset_relations import validate_asset_payload
    report = {**artifact_gate_record(payload), "run_label": plan["run_label"], "lineage": payload["lineage"]}
    issues = validate_asset_payload("quantization.artifact_gate", report)
    if issues:
        raise ValueError("invalid report schema: " + "; ".join(issues))
    return report


def read_artifact_gate(plan: dict) -> dict:
    """5.5只读验证来源、调用身份、原始记录、报告与已登记文件哈希。"""
    prepared = _require_gate_inputs(plan)
    if not plan["raw_path"].exists():
        if any(plan[key].exists() for key in ("report_path", "reference_path", "receipt_path")):
            raise ValueError("final assets exist without measurement evidence")
        if plan["loader_path"].exists():
            _validate_gate_loader(plan, prepared, _gate_read_json(plan["loader_path"]))
            return {"status": "incomplete_measurement", "decision": {"decision": "pending"}, "next_action": "resume_measure_using_saved_loader"}
        return {"status": "pending", "decision": {"decision": "pending"}, "next_action": "run_measure"}
    loader = _gate_read_json(plan["loader_path"])
    _validate_gate_loader(plan, prepared, loader)
    payload = _gate_read_json(plan["raw_path"])
    if payload != _artifact_gate_payload(plan, prepared, loader):
        raise ValueError("measurement identity, hashes, metrics or decision do not match current evidence")
    report = _gate_report(plan, payload)
    if plan["report_path"].exists() and _gate_read_json(plan["report_path"]) != report:
        raise ValueError("report conflicts with revalidated measurement")
    if plan["reference_path"].exists():
        from tools.project_asset_relations import make_asset_reference
        if report["decision"]["decision"] != "accept" or not plan["report_path"].exists():
            raise ValueError("handoff requires an accepted, saved report")
        if _gate_read_json(plan["reference_path"]) != make_asset_reference("quantization.artifact_gate", plan["report_path"]):
            raise ValueError("handoff hash or schema conflicts with report")
    if plan["receipt_path"].exists():
        receipt = _gate_read_json(plan["receipt_path"])
        if receipt != _gate_receipt(plan, payload, report):
            raise ValueError("registration receipt conflicts with evidence")
    return {"status": "validated", "decision": report["decision"], "payload": payload, "report": report,
            "summary": {"run_label": plan["run_label"], "artifact_id": report["artifact"]["artifact_id"],
                        "model_revision": report["artifact"]["model_revision"], "backend": report["backend"],
                        "quality": report["quality"], "memory_measurement": plan["contract"]["memory_measurement"],
                        "device_free_memory_delta_mb": report["metrics"]["peak_memory_mb"], "failure": report["failure"]}}


def _gate_receipt(plan, payload, report):
    files = {"source": plan["root"] / payload["lineage"]["source"]["path"],
             "bundle": plan["root"] / payload["lineage"]["bundle"]["path"], "loader": plan["loader_path"],
             "measurement": plan["raw_path"], "report": plan["report_path"]}
    if report["decision"]["decision"] == "accept":
        files["handoff"] = plan["reference_path"]
    return {"schema_version": "quantization-project-registration/v1", "run_label": plan["run_label"],
            "semantic_id": plan["contract"]["semantic_id"], "report_schema": report["schema_version"],
            "decision": report["decision"], "files": {role: _gate_file_reference(plan, path) for role, path in files.items()}}


def close_artifact_gate(plan: dict) -> dict:
    """close重新读取证据，幂等补齐报告、接受引用与登记回执；不调用后端。"""
    if plan["stage"] != "close":
        raise ValueError("final assets require the close stage")
    checked = read_artifact_gate(plan)
    if checked["status"] != "validated":
        return {"status": checked["status"], "decision": checked["decision"], "next_action": checked["next_action"], "registration_receipt": None}
    report, payload = checked["report"], checked["payload"]
    wrote = _gate_save_or_verify(plan["report_path"], report)
    if report["decision"]["decision"] == "accept":
        from tools.project_asset_relations import make_asset_reference
        wrote = _gate_save_or_verify(plan["reference_path"], make_asset_reference("quantization.artifact_gate", plan["report_path"])) or wrote
    receipt = _gate_receipt(plan, payload, report)
    wrote = _gate_save_or_verify(plan["receipt_path"], receipt) or wrote
    read_artifact_gate(plan)
    return {"status": "closed" if wrote else "already_complete", "decision": report["decision"],
            "report_path": str(plan["report_path"]), "handoff_path": str(plan["reference_path"]) if report["decision"]["decision"] == "accept" else None,
            "registration_receipt": {"path": str(plan["receipt_path"]), **receipt}}


FIXED_PAIR_FIELDS = (
    "model_revision", "tokenizer_revision", "workload_id", "backend", "hardware_id", "cache_policy",
)
SUPPORTED_PROFILES = {"latency", "balanced", "throughput"}
SUPPORTED_DIRECTIONS = {"max", "min"}


def _positive_int(value: Any) -> bool:
    """配置整数不能由布尔、浮点或字符串截断得到。"""
    return type(value) is int and value > 0


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _relative_file(value: Any) -> bool:
    return _text(value) and not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts and "\\" not in value and value != "."


def candidate_contract_facts(candidate: dict[str, Any]) -> dict[str, Any]:
    """只核对声明、文件角色覆盖和哈希格式；真实文件内容由磁盘校验另行核对。"""
    candidate = _mapping(candidate)
    model, quantization, files = (_mapping(candidate.get(key)) for key in ("model", "quantization", "files"))
    identity_ready = candidate.get("schema_version") == "quantization-artifact-candidate/v1" and all(
        _text(value) for value in (candidate.get("artifact_id"), model.get("id"), model.get("revision"))
    ) and model.get("revision") not in {"main", "master", "latest"}
    quantization_ready = all(_text(quantization.get(key)) for key in ("method", "format")) and all(
        _positive_int(quantization.get(key)) for key in ("bits", "group_size")
    )
    inventory = files.get("inventory")
    valid_inventory = isinstance(inventory, list) and bool(inventory) and all(
        isinstance(item, dict) and _relative_file(item.get("path"))
        and isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
        and all(char in "0123456789abcdefABCDEF" for char in item["sha256"])
        for item in inventory
    )
    paths = [item["path"] for item in inventory] if valid_inventory else []
    weights, tokenizer, config = files.get("weights"), files.get("tokenizer"), files.get("config")
    roles_ready = all(isinstance(group, list) and bool(group) and all(_relative_file(path) for path in group) for group in (weights, tokenizer)) and _relative_file(config)
    files_ready = bool(valid_inventory and len(set(paths)) == len(paths) and roles_ready and all(
        path in paths for path in [*weights, config, *tokenizer]
    ))
    issues = [name for name, ready in (("identity", identity_ready), ("quantization", quantization_ready), ("files", files_ready)) if not ready]
    return {"identity_ready": bool(identity_ready), "quantization_ready": bool(quantization_ready), "files_ready": files_ready, "issues": issues}


def validate_candidate_contract(candidate: dict[str, Any]) -> dict[str, Any]:
    """组合身份、量化配置和完整文件角色三个静态门槛。"""
    facts = candidate_contract_facts(candidate)
    return {"ready": facts["identity_ready"] and facts["quantization_ready"] and facts["files_ready"], **facts}


def candidate_evidence_facts(candidate: dict[str, Any]) -> dict[str, Any]:
    """复核登记的数据划分与质量数值，不把不同摘要当作集合无交集证明。"""
    candidate = _mapping(candidate)
    calibration, evaluation = (_mapping(candidate.get(key)) for key in ("calibration", "evaluation"))
    snapshot_ready = all(_text(calibration.get(key)) and calibration.get(key) == evaluation.get(key) for key in ("dataset", "revision")) and calibration.get("config") == evaluation.get("config")
    split_isolated = snapshot_ready and all(_text(section.get("split")) for section in (calibration, evaluation)) and calibration["split"] != evaluation["split"]
    hashes_distinct = all(_text(section.get("sample_ids_hash")) for section in (calibration, evaluation)) and calibration["sample_ids_hash"] != evaluation["sample_ids_hash"]
    baseline, value = evaluation.get("baseline_value"), evaluation.get("candidate_value")
    values_ready = _finite_number(baseline, positive=True) and _finite_number(value)
    observed_ratio = value / baseline if values_ready else None
    direction = evaluation.get("direction")
    limit = evaluation.get("max_degradation_ratio") if direction == "lower_is_better" else evaluation.get("min_retention_ratio")
    ratio_declared = evaluation.get("ratio_vs_baseline")
    ratio_consistent = "ratio_vs_baseline" not in evaluation or (
        _finite_number(ratio_declared) and observed_ratio is not None and isclose(ratio_declared, observed_ratio, rel_tol=1e-6, abs_tol=1e-9)
    )
    # sample_ids_isolated 保留旧接口名，仅表示摘要不同；无交集证明另由明确字段表达。
    return {"split_isolated": bool(split_isolated), "sample_hashes_distinct": bool(hashes_distinct),
            "sample_ids_isolated": bool(hashes_distinct), "sample_disjointness_proven": False,
            "direction": direction, "observed_ratio": observed_ratio, "quality_limit": limit,
            "quality_inputs_ready": evaluation.get("passed") is True and values_ready and observed_ratio is not None and isfinite(observed_ratio) and _finite_number(limit, positive=True) and ratio_consistent,
            "isolation_basis": "registered_dataset_splits"}


def evaluate_candidate_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    """按指标方向重新计算质量判断，并核对已登记的数据划分。"""
    facts = candidate_evidence_facts(candidate)
    ratio, limit, direction = facts["observed_ratio"], facts["quality_limit"], facts["direction"]
    quality_ready = bool(facts["quality_inputs_ready"] and (
        (direction == "lower_is_better" and ratio <= limit)
        or (direction == "higher_is_better" and ratio >= limit)
    ))
    issues = [name for name, ready in (("split_overlap", facts["split_isolated"]), ("sample_identity_missing_or_equal", facts["sample_hashes_distinct"]), ("quality", quality_ready)) if not ready]
    return {"ready": facts["split_isolated"] and facts["sample_hashes_distinct"] and quality_ready,
            **facts, "quality_ready": quality_ready, "issues": issues}


def artifact_loader_facts(loader_result: dict[str, Any], max_memory_mb: float) -> dict[str, bool]:
    """检查真实加载记录的原子事实；预算缺测与测量超标分别处理。"""
    loader_result = _mapping(loader_result)
    roundtrip = _mapping(loader_result.get("roundtrip_quality"))
    count, completed = roundtrip.get("prompt_count"), roundtrip.get("non_empty_generations")
    peak = loader_result.get("peak_memory_mb")
    return {"load_complete": loader_result.get("load_status") == "loaded" and not loader_result.get("failure") and loader_result.get("status") not in {"failed", "error"},
            "roundtrip_complete": _positive_int(count) and type(completed) is int and completed == count,
            "budget_valid": _finite_number(max_memory_mb),
            "memory_missing": peak is None,
            "memory_valid": _finite_number(peak),
            "memory_within_budget": _finite_number(peak) and _finite_number(max_memory_mb) and peak <= max_memory_mb}


def decide_artifact_adoption(contract_check: dict[str, Any], evidence_check: dict[str, Any], loader_result: dict[str, Any] | None, *, max_memory_mb: float = 0.0) -> dict[str, str]:
    """组合三层证据，预算缺测保持 pending，真实超预算为 tune。"""
    if contract_check.get("ready") is not True or evidence_check.get("ready") is not True:
        issues = [*contract_check.get("issues", []), *evidence_check.get("issues", [])]
        return {"decision": "reject", "reason": "; ".join(issues) or "静态或质量门禁未通过"}
    if loader_result is None:
        return {"decision": "pending", "reason": "CPU 门禁已通过，尚未执行真实加载"}
    facts = artifact_loader_facts(loader_result, max_memory_mb)
    loader_ready = facts["load_complete"] and facts["roundtrip_complete"]
    memory_ready = facts["budget_valid"] and (max_memory_mb == 0 or facts["memory_within_budget"])
    if not loader_ready:
        return {"decision": "reject", "reason": "真实加载与往返生成未完成"}
    if not facts["budget_valid"] or (not facts["memory_missing"] and not facts["memory_valid"]):
        return {"decision": "reject", "reason": "显存预算或测量数值无效"}
    if max_memory_mb > 0 and facts["memory_missing"]:
        return {"decision": "pending", "reason": "已启用预算但尚无有效显存证据"}
    if not memory_ready:
        return {"decision": "tune", "reason": "加载成功，但显存记录超过预算"}
    return {"decision": "accept", "reason": "候选可进入浮点—量化部署基准实验"}


def _positive_ratio(candidate_value: float, baseline_value: float) -> float:
    if not _finite_number(baseline_value, positive=True) or not _finite_number(candidate_value) or candidate_value < 0:
        raise ValueError("指标必须满足 baseline > 0 且 candidate >= 0")
    return candidate_value / baseline_value


def validate_deployment_pair(
    gate: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """检查产物血缘与浮点、量化配对条件。"""
    gate_ready = gate.get("decision") == "accept"
    artifact_matches = _text(candidate.get("artifact_id")) and candidate.get("artifact_id") == gate.get("artifact_id")
    model_revision_matches = _text(candidate.get("model_revision")) and candidate.get("model_revision") == gate.get("model_revision")
    fixed_mismatches = [
        field for field in FIXED_PAIR_FIELDS
        if not _text(baseline.get(field)) or baseline.get(field) != candidate.get(field)
    ]
    issues = []
    if not gate_ready:
        issues.append("artifact_gate_not_accepted")
    if not artifact_matches:
        issues.append("artifact_id_mismatch")
    if not model_revision_matches:
        issues.append("model_revision_mismatch")
    issues.extend(f"fixed_field_mismatch:{field}" for field in fixed_mismatches)
    return {
        "ready": gate_ready and artifact_matches and model_revision_matches and not fixed_mismatches,
        "issues": issues,
        "fixed_mismatches": fixed_mismatches,
    }


def summarize_deployment_pair(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """计算候选相对基线的延迟、吞吐、显存和质量比值。"""
    tpot_ratio = _positive_ratio(float(candidate["tpot_ms"]), float(baseline["tpot_ms"]))
    throughput_ratio = _positive_ratio(
        float(candidate["throughput_tps"]), float(baseline["throughput_tps"])
    )
    if baseline.get("peak_memory_mb") is None or candidate.get("peak_memory_mb") is None:
        memory_reduction_ratio = None
    else:
        memory_reduction_ratio = 1.0 - _positive_ratio(
            float(candidate["peak_memory_mb"]), float(baseline["peak_memory_mb"])
        )
    quality_ratio = _positive_ratio(float(candidate["quality"]), float(baseline["quality"]))
    return {
        "tpot_ratio": round(tpot_ratio, 6),
        "throughput_ratio": round(throughput_ratio, 6),
        "memory_reduction_ratio": (
            None if memory_reduction_ratio is None else round(memory_reduction_ratio, 6)
        ),
        "quality_ratio": round(quality_ratio, 6),
    }


def decide_quantized_deployment(
    contract: dict[str, Any],
    comparison: dict[str, Any],
    evidence: dict[str, Any],
    *,
    max_tpot_ratio: float = 1.20,
    min_throughput_ratio: float = 0.90,
    min_memory_reduction_ratio: float = 0.20,
    max_quality_ratio: float = 1.10,
) -> dict[str, str]:
    """根据配对证据、质量和资源门槛形成部署判断。"""
    if not all(_finite_number(v) and v >= 0 for v in
               (max_tpot_ratio, min_throughput_ratio, min_memory_reduction_ratio, max_quality_ratio)):
        raise ValueError("deployment thresholds must be finite nonnegative numbers")
    if any(not _finite_number(comparison.get(k)) or comparison[k] < 0
           for k in ("tpot_ratio", "throughput_ratio", "quality_ratio")):
        return {"decision": "pending", "reason": "部署比较缺少有效数值"}
    memory_gain = comparison.get("memory_reduction_ratio")
    evidence_ready = (
        evidence.get("real_backend") is True
        and evidence.get("quantized_path") is True
        and evidence.get("fallback") is False
    )
    quality_ready = comparison.get("quality_ratio", float("inf")) <= max_quality_ratio
    resource_ready = any(
        (
            comparison.get("tpot_ratio", float("inf")) <= max_tpot_ratio,
            comparison.get("throughput_ratio", 0.0) >= min_throughput_ratio,
            memory_gain is not None and memory_gain >= min_memory_reduction_ratio,
        )
    )
    if not contract.get("ready"):
        return {"decision": "reject", "reason": "配对条件不一致"}
    if not evidence_ready:
        return {"decision": "pending", "reason": "真实量化执行证据尚未完整"}
    if not quality_ready:
        return {"decision": "reject", "reason": "候选评测质量退化超过预算"}
    if not resource_ready:
        return {"decision": "tune", "reason": "资源或性能门槛尚未达到"}
    return {"decision": "accept", "reason": "质量、执行路径和资源门槛均成立"}


def validate_deployment_lineage(deployment: dict[str, Any]) -> dict[str, Any]:
    """检查量化部署结果能否作为服务配置实验的上游资产。"""
    artifact = deployment.get("artifact", {})
    lineage = deployment.get("lineage", {})
    accepted = (
        deployment.get("stage") == "quantized_deployment"
        and deployment.get("decision", {}).get("decision") == "accept"
    )
    artifact_ready = bool(artifact.get("artifact_id") and artifact.get("path"))
    gate_lineage_ready = bool(
        lineage.get("artifact_gate_reference") and lineage.get("artifact_gate_sha256")
    )
    issues = []
    if not accepted:
        issues.append("deployment_not_accepted")
    if not artifact_ready:
        issues.append("artifact_identity_or_path_missing")
    if not gate_lineage_ready:
        issues.append("artifact_gate_lineage_missing")
    return {
        "ready": accepted and artifact_ready and gate_lineage_ready,
        "issues": issues,
    }


def _positive_workload(workload: dict[str, Any]) -> bool:
    return all(
        _positive_int(workload.get(key))
        for key in ("num_prompts", "concurrency", "max_tokens", "repeats")
    )


def _valid_slo_rules(slo: dict[str, Any]) -> bool:
    return all(
        isinstance(rule, dict) and rule.get("direction") in SUPPORTED_DIRECTIONS
        and _finite_number(rule.get("limit")) and rule["limit"] >= 0
        for rule in slo.values()
    )


def validate_serving_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """检查服务配置的工作负载、服务目标和主指标。"""
    workload = _mapping(profile.get("workload"))
    slo = _mapping(profile.get("slo"))
    workload_ready = bool(workload.get("id")) and _positive_workload(workload)
    slo_ready = bool(slo) and _valid_slo_rules(slo)
    primary_ready = profile.get("primary_metric") in slo
    issues = []
    if profile.get("name") not in SUPPORTED_PROFILES:
        issues.append("unsupported_profile")
    if not workload_ready:
        issues.append("invalid_workload")
    if not slo_ready:
        issues.append("invalid_slo")
    if not primary_ready:
        issues.append("primary_metric_missing_from_slo")
    return {"ready": not issues, "issues": issues}


def _meets_slo(metrics: dict[str, Any], slo: dict[str, Any]) -> bool:
    for metric, rule in slo.items():
        value = metrics.get(metric)
        if not _finite_number(value) or value < 0:
            return False
        if rule["direction"] == "max" and value > rule["limit"]:
            return False
        if rule["direction"] == "min" and value < rule["limit"]:
            return False
    return True


def _slo_headroom(metrics: dict[str, Any], slo: dict[str, Any]) -> float:
    headroom = 0.0
    for metric, rule in slo.items():
        value, limit = metrics[metric], rule["limit"]
        scale = max(abs(float(limit)), 1.0)
        headroom += (
            (limit - value) if rule["direction"] == "max" else (value - limit)
        ) / scale
    return headroom


def choose_primary_record(
    records: list[dict[str, Any]],
    primary_metric: str,
    direction: str,
    slo: dict[str, Any],
) -> dict[str, Any] | None:
    """从达到服务目标的记录中选择主指标最优项。"""
    if not records:
        return None
    best = records[0]
    for candidate in records[1:]:
        candidate_value = candidate["metrics"][primary_metric]
        best_value = best["metrics"][primary_metric]
        primary_is_better = (
            candidate_value < best_value if direction == "max" else candidate_value > best_value
        )
        primary_is_tied = candidate_value == best_value
        if primary_is_better or (
            primary_is_tied
            and _slo_headroom(candidate["metrics"], slo) > _slo_headroom(best["metrics"], slo)
        ):
            best = candidate
    return best


def select_serving_candidate(
    records: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    expected_artifact_id: str | None = None,
) -> dict[str, Any]:
    """过滤不可比记录并选择满足服务目标的候选。"""
    if not validate_serving_profile(profile)["ready"]:
        raise ValueError("服务配置不完整")
    expected_artifact = expected_artifact_id or next(
        (record.get("artifact_id") for record in records if record.get("artifact_id")), None
    )
    comparable = [
        record
        for record in records
        if record.get("status") == "ok"
        and record.get("artifact_id") == expected_artifact
        and record.get("profile_name") == profile["name"]
        and record.get("workload_id") == profile["workload"]["id"]
        and record.get("evidence_level") == "real_project_benchmark"
    ]
    feasible = [
        record for record in comparable if _meets_slo(record.get("metrics", {}), profile["slo"])
    ]
    primary_metric = profile["primary_metric"]
    direction = profile["slo"][primary_metric]["direction"]
    selected = choose_primary_record(feasible, primary_metric, direction, profile["slo"])
    return {
        "decision": "accept" if selected else "tune",
        "selected": selected,
        "comparable_count": len(comparable),
        "feasible_count": len(feasible),
        "excluded": [record.get("name", "unnamed") for record in records if record not in feasible],
    }
