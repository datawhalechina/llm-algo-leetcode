"""Fixed contracts and resumable evidence for quantized deployment and serving.

Import, prepare, read and close are CPU-only. Backend imports occur only in measure.
No historical records are migrated or amended by these helpers.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from statistics import median, mean
import subprocess
import sys

from tools import quantization_project_runtime as gate
from tools.project_asset_relations import make_asset_reference, validate_asset_payload
from tools.quantization_result_schema import quantized_deployment_record, serving_profile_record
from tools.quantization_runtime import validate_candidate_bundle


def project_plan(root, project: str, label: str, upstream: str, stage="prepare") -> dict:
    """Load one immutable contract; each project has exactly one direct upstream."""
    guard = gate.artifact_gate_plan(root, label, upstream, stage)
    if project not in {"67", "83"}:
        raise ValueError("project must be 67 or 83")
    root = guard["root"]
    source_dir = "82_quantization_artifact_evaluation" if project == "67" else "67_quantized_deployment"
    source_name = "82_artifact_gate" if project == "67" else "67_paired_deployment"
    result_dir = root / "benchmarks/results" / ("67_quantized_deployment" if project == "67" else "83_quantized_serving_profile")
    prefix = "67_paired_deployment" if project == "67" else "83_profile_summary"
    profiles = {
        "latency": {"workload": {"id": "fixed-chat-latency", "num_prompts": 8, "concurrency": 1, "max_tokens": 64, "repeats": 3},
                    "primary_metric": "ttft_p95_ms", "slo": {"ttft_p95_ms": {"direction": "max", "limit": 500.}, "failure_rate": {"direction": "max", "limit": 0.}}},
        "balanced": {"workload": {"id": "fixed-chat-balanced", "num_prompts": 12, "concurrency": 4, "max_tokens": 128, "repeats": 3},
                     "primary_metric": "tpot_p95_ms", "slo": {"tpot_p95_ms": {"direction": "max", "limit": 80.}, "throughput_tps": {"direction": "min", "limit": 20.}, "failure_rate": {"direction": "max", "limit": 0.}}},
        "throughput": {"workload": {"id": "fixed-chat-throughput", "num_prompts": 24, "concurrency": 8, "max_tokens": 256, "repeats": 3},
                       "primary_metric": "throughput_tps", "slo": {"throughput_tps": {"direction": "min", "limit": 40.}, "failure_rate": {"direction": "max", "limit": 0.}}},
    }
    contract = {"protocol": "quantized-deployment-lifecycle/v3", "project": project,
                "semantic_id": "QUANT-INFERENCE-DEPLOY" if project == "67" else "QUANT-SERVING-PROFILE",
                "environment_profile": "INF-VLLM-011-CU128", "backend": "vllm", "dtype": "bfloat16",
                "max_model_len": 2048, "gpu_memory_utilization": .75, "enforce_eager": True,
                "cache_policy": "disabled", "warmup": 1, "temperature": 0.,
                "profiles": profiles if project == "83" else {},
                "pair": {"num_prompts": 3, "max_tokens": 64, "concurrency": 1, "repeats": 1, "pairs": 3},
                "limits": {"max_tpot_ratio": 1.2, "min_throughput_ratio": .9, "min_memory_reduction_ratio": .2}}
    return {"root": root, "project": project, "run_label": label, "upstream_run_label": upstream, "stage": stage,
            "contract": contract, "result_dir": result_dir,
            "source_path": root / "benchmarks/results" / source_dir / f"{source_name}_{upstream}_reference.json",
            "workload_path": root / "benchmarks/workloads/fixed.jsonl",
            **{key: result_dir / f"{prefix}_{label}{suffix}.json" for key, suffix in
               (("raw_path", ""), ("report_path", "_companion"), ("reference_path", "_reference"), ("receipt_path", "_registration"))}}


def _reference_record(plan, reference, asset):
    """Resolve the direct upstream only, under this repository root."""
    if reference.get("schema_version") != "project-asset-reference/v1" or reference.get("asset_id") != asset:
        raise ValueError("upstream reference schema or asset does not match")
    path = Path(reference.get("result_path", ""))
    path = path if path.is_absolute() else plan["root"] / path
    path = path.resolve()
    path.relative_to(plan["root"])
    if gate._gate_digest(path) != reference.get("sha256"):
        raise ValueError("upstream SHA256 mismatch")
    record = gate._gate_read_json(path)
    errors = validate_asset_payload(asset, record, consumer=plan["contract"]["semantic_id"])
    if errors or record.get("decision", {}).get("decision") != "accept":
        raise ValueError("upstream is not accepted or violates schema: " + "; ".join(errors))
    if record.get("run_label") != plan["upstream_run_label"]:
        raise ValueError("upstream run_label mismatch; legacy records need separate review")
    return record, path


def _cached_float_model(model, revision):
    """Locate an immutable, already-prepared HF snapshot; never fetch a branch tip."""
    import re
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("baseline requires the immutable 40-character model commit")
    cache = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    snapshot = cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    if not (snapshot / "config.json").is_file() or not (snapshot / "tokenizer_config.json").is_file():
        raise ValueError("prepare the exact floating model/tokenizer snapshot in the HF cache before measuring")
    index = snapshot / "model.safetensors.index.json"
    weights = set(gate._gate_read_json(index).get("weight_map", {}).values()) if index.is_file() else {"model.safetensors"}
    if not weights or any(not (snapshot / name).is_file() for name in weights):
        raise ValueError("floating baseline snapshot lacks weight files")
    return snapshot


def _tokenizer_signature(path):
    """Compare tokenizer data and generation token rules, ignoring save-path metadata."""
    files = {name: gate._gate_digest(path / name) for name in ("tokenizer.json", "tokenizer.model") if (path / name).is_file()}
    if not files:
        raise ValueError("tokenizer data missing from the pinned snapshot")
    config_path = path / "tokenizer_config.json"
    config = gate._gate_read_json(config_path) if config_path.is_file() else {}
    return {"files": files, "rules": {k: config.get(k) for k in ("chat_template", "bos_token", "eos_token", "pad_token", "add_bos_token", "add_eos_token")}}


def prepare_project(plan: dict) -> dict:
    """Read direct lineage, artifact bytes and workload; do not touch CUDA or write files."""
    try:
        reference = gate._gate_read_json(plan["source_path"])
        asset = "quantization.artifact_gate" if plan["project"] == "67" else "quantization.deployment_candidate"
        upstream, upstream_path = _reference_record(plan, reference, asset)
        expected_stage = "artifact_gate" if plan["project"] == "67" else "quantized_deployment"
        if upstream.get("stage") != expected_stage:
            raise ValueError("upstream stage does not match the project input")
        artifact = upstream["artifact"]
        manifest = artifact["manifest"]
        validation = validate_candidate_bundle(manifest, project_root=plan["root"])
        if not validation["ready"]:
            raise ValueError("; ".join(validation["issues"]))
        if (artifact["artifact_id"] != manifest["artifact_id"] or artifact["model_revision"] != manifest["model"]["revision"]
                or artifact["model"] != manifest["model"]["id"]):
            raise ValueError("artifact identity/revision differs from manifest")
        artifact_path = Path(validation["artifact_path"]).resolve()
        declared_path = Path(artifact["path"])
        declared_path = declared_path if declared_path.is_absolute() else plan["root"] / declared_path
        if declared_path.resolve() != artifact_path:
            raise ValueError("upstream artifact path differs from embedded manifest")
        if plan["project"] == "67":
            quality = upstream["quality"]
            if any(quality.get(k) != manifest["evaluation"].get(k) for k in
                   ("metric_name", "direction", "baseline_value", "candidate_value", "max_degradation_ratio")):
                raise ValueError("upstream quality differs from embedded candidate evaluation")
        if plan["project"] == "83" and not gate.validate_deployment_lineage(upstream)["ready"]:
            raise ValueError("deployment lacks embedded gate lineage")
        config = gate._gate_read_json(artifact_path / "config.json")
        method = config.get("quantization_config", {}).get("quant_method")
        if method not in {"compressed-tensors", "gptq", "awq"}:
            raise ValueError("unsupported artifact runtime format; GPTQ/AWQ algorithms do not imply the same loader format")
        workload_text = plan["workload_path"].read_text(encoding="utf-8")
        lines = [json.loads(line) for line in workload_text.splitlines() if line.strip()]
        if not lines or any(not isinstance(row, dict) or not (row.get("messages") or row.get("prompt")) for row in lines):
            raise ValueError("fixed workload must contain nonempty prompts or chat messages")
        snapshot = {"reference": gate._gate_file_reference(plan, plan["source_path"]),
                    "upstream": gate._gate_file_reference(plan, upstream_path),
                    "workload": gate._gate_file_reference(plan, plan["workload_path"]),
                    "artifact_id": artifact["artifact_id"], "artifact_path": str(artifact_path),
                    "manifest": manifest, "runtime_format": method}
        if plan["project"] == "67":
            float_path = _cached_float_model(artifact["model"], artifact["model_revision"])
            snapshot["float_model_path"] = str(float_path)
            snapshot["tokenizer"] = _tokenizer_signature(float_path)
            if snapshot["tokenizer"] != _tokenizer_signature(artifact_path):
                raise ValueError("floating and quantized tokenizer data or token rules differ")
        return {"ready": True, "issues": [], "snapshot": snapshot, "upstream": upstream, "artifact": artifact}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return {"ready": False, "issues": [str(error)], "snapshot": None}


def _prepared(plan):
    result = prepare_project(plan)
    if not result["ready"]:
        raise ValueError("input preflight: " + "; ".join(result["issues"]))
    return result


def invocation(plan, prepared):
    """Bind run identity, source hashes and all fixed measurement conditions."""
    return {"run_label": plan["run_label"], "upstream_run_label": plan["upstream_run_label"],
            "inputs": prepared["snapshot"], "contract": plan["contract"]}


def preflight_project(plan):
    """Only a fresh measure stage may run the environment/GPU probe."""
    prepared = prepare_project(plan)
    if plan["stage"] != "measure" or not prepared["ready"] or plan["raw_path"].exists():
        return prepared
    try:
        completed = subprocess.run([sys.executable, str(plan["root"] / "tools/environment_preflight.py"),
                                    "--profile", plan["contract"]["environment_profile"]],
                                   capture_output=True, text=True, cwd=plan["root"], timeout=60)
        profile = json.loads(completed.stdout)
        if not isinstance(profile, dict) or completed.returncode or profile.get("ready") is not True:
            raise ValueError("environment profile not ready: " + str(profile))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {**prepared, "ready": False, "issues": [str(error)]}
    return {**prepared, "environment": profile}


def member_path(plan, key):
    return plan["result_dir"] / f"{plan['project']}_{plan['run_label']}_{key}.json"


def _execute_member(plan, prepared, key, workload, role):
    """Run one server, pin model and tokenizer paths, save complete logs, always stop it."""
    from tools.backend_runtime import build_vllm_quantization_args, probe_vllm_quantization_support
    from tools.inference_project_runtime import start_optional_vllm, stop_optional_vllm, run_backend_benchmark, runtime_snapshot
    import torch

    fixed, artifact = plan["contract"], prepared["artifact"]
    quantized = role != "baseline"
    model_path = prepared["snapshot"]["artifact_path"] if quantized else prepared["snapshot"]["float_model_path"]
    runtime_format = prepared["snapshot"]["runtime_format"]
    # compressed-tensors is declared in config.json and auto-detected by the loader.
    args = build_vllm_quantization_args(probe_vllm_quantization_support(), runtime_format) if quantized and runtime_format != "compressed-tensors" else None
    log_path = plan["result_dir"] / f"{key}_{plan['run_label']}.log"
    raw_path = plan["result_dir"] / f"{key}_{plan['run_label']}_backend.json"
    if log_path.exists() or raw_path.exists():
        raise FileExistsError("interrupted backend output exists without a complete member; preserve it and use a new run label")
    server = handle = None
    try:
        server, handle, port, dtype, resolved = start_optional_vllm(
            model_id=model_path, model_source="local", dtype=fixed["dtype"],
            served_model_name=artifact["model"], max_model_len=fixed["max_model_len"],
            gpu_memory_utilization=fixed["gpu_memory_utilization"], enforce_eager=True,
            enable_prefix_caching=False, quantization_args=args, generation_config="vllm", log_path=log_path)
        raw = run_backend_benchmark(project=plan["project"], base_url=f"http://127.0.0.1:{port}",
            model=artifact["model"], label=key, output=raw_path, workload=str(plan["workload_path"]),
            num_prompts=workload["num_prompts"], max_tokens=workload["max_tokens"], concurrency=workload["concurrency"],
            warmup=fixed["warmup"], repeats=workload["repeats"], temperature=0., backend="vllm", dtype=dtype,
            cache_policy="disabled", role=role, include_requests=True)
        properties = torch.cuda.get_device_properties(0)
        hardware = {**runtime_snapshot(torch), "device_uuid": str(getattr(properties, "uuid", ""))}
        if not hardware["device_uuid"]:
            raise ValueError("runtime lacks the physical GPU UUID for paired identity")
        result = {"status": "ok", "raw": raw, "backend_version": importlib.metadata.version("vllm"),
                "hardware": hardware, "dtype": dtype, "resolved_model": resolved,
                "execution": {"real_backend": True, "quantized_path": None, "fallback": None,
                              "launch_args": args, "scope": "loader_arguments_not_kernel_execution"}, "failure": None}
    finally:
        if server is not None:
            stop_optional_vllm(server, handle)
    # Shutdown may append log lines: hash only after the process and log handle close.
    result.update(backend_raw=gate._gate_file_reference(plan, raw_path), backend_log=gate._gate_file_reference(plan, log_path))
    return result


def _member(plan, prepared, key, workload, role, runner):
    path = member_path(plan, key)
    expected = {"invocation": invocation(plan, prepared), "key": key, "workload": workload, "role": role}
    if path.exists():
        record = gate._gate_read_json(path)
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("saved member identity/contract mismatch")
        _check_member_files(plan, record)
        return record
    try:
        observed = runner(plan, prepared, key, workload, role)
    except Exception as error:
        observed = {"status": "failed", "failure": {"type": type(error).__name__, "reason": str(error)}}
    record = {**expected, "observed": observed}
    if _prepared(plan)["snapshot"] != prepared["snapshot"]:
        raise ValueError("inputs changed during measurement")
    gate._gate_save_or_verify(path, record)
    return record


def _check_member_files(plan, record):
    observed = record.get("observed", {})
    if observed.get("status") == "ok":
        inputs = record["invocation"]["inputs"]
        expected_path = inputs["float_model_path"] if record["role"] == "baseline" else inputs["artifact_path"]
        if observed.get("resolved_model") != expected_path or observed.get("dtype") != plan["contract"]["dtype"]:
            raise ValueError("actual model path or dtype differs from fixed invocation")
    for key in ("backend_raw", "backend_log"):
        ref = record.get("observed", {}).get(key)
        if ref:
            path = plan["root"] / ref["path"]
            if gate._gate_digest(path) != ref["sha256"]:
                raise ValueError("backend file hash changed")


def measure_project(plan, runner=None):
    """Resume complete members without rerunning; raw project evidence precedes close."""
    if plan["stage"] != "measure":
        raise ValueError("measure requires measure stage")
    prepared = _prepared(plan)
    if plan["raw_path"].exists():
        return read_project(plan)
    runner = runner or _execute_member
    keys = []
    if plan["project"] == "67":
        work = {k: v for k, v in plan["contract"]["pair"].items() if k != "pairs"}
        for index in range(plan["contract"]["pair"]["pairs"]):
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            for role in order:
                key = f"pair{index + 1}_{role}"
                _member(plan, prepared, key, work, role, runner)
                keys.append(key)
    else:
        for name, spec in plan["contract"]["profiles"].items():
            _member(plan, prepared, name, spec["workload"], "candidate", runner)
            keys.append(name)
    payload = {"schema_version": "quantized-project-measurement/v3", "invocation": invocation(plan, prepared),
               "members": {key: gate._gate_file_reference(plan, member_path(plan, key)) for key in keys}}
    gate._gate_save_or_verify(plan["raw_path"], payload)
    return read_project(plan)


def _metrics(member):
    observed = member["observed"]
    if observed.get("status") != "ok" or observed.get("failure"):
        return None
    raw = observed.get("raw", {})
    runs = raw.get("runs") or [raw]
    if len(runs) != member["workload"]["repeats"]:
        return None
    total = failed = 0
    for run in runs:
        work, measured = run.get("workload", {}), run.get("metrics", {})
        count, errors = work.get("requests"), measured.get("failed_requests")
        if (type(count) is not int or count != member["workload"]["num_prompts"]
                or type(errors) is not int or not 0 <= errors <= count
                or measured.get("successful_requests") != count - errors
                or any(work.get(k) != member["workload"][k] for k in ("concurrency", "max_tokens"))):
            return None
        total += count
        failed += errors
    def averaged(key, stat=None):
        values = [run.get("metrics", {}).get(key) for run in runs]
        if stat:
            values = [value.get(stat) if isinstance(value, dict) else None for value in values]
        return mean(values) if all(gate._finite_number(v) and v >= 0 for v in values) else None
    return {"tpot_ms": averaged("tpot_ms", "p50"), "ttft_p95_ms": averaged("ttft_ms", "p95"),
            "tpot_p95_ms": averaged("tpot_ms", "p95"), "throughput_tps": averaged("output_token_throughput_per_s"),
            "failure_rate": failed / total, "peak_memory_mb": None}


def _deployment_decision(plan, prepared, members):
    """Recompute paired ratios, identity checks and execution sufficiency from saved members."""
    quality = prepared["upstream"]["quality"]
    comparisons, missing, failed = [], [], []
    for index in range(1, plan["contract"]["pair"]["pairs"] + 1):
        b, c = members[f"pair{index}_baseline"], members[f"pair{index}_candidate"]
        bm, cm = _metrics(b), _metrics(c)
        if bm is None or cm is None:
            missing.append(index)
            continue
        if bm["failure_rate"] or cm["failure_rate"]:
            failed.append(index)
            continue
        bo, co = b["observed"], c["observed"]
        if any(not bo.get(k) or bo.get(k) != co.get(k) for k in ("backend_version", "hardware", "dtype")):
            return {"decision": "reject", "reason": "paired runtime identity changed"}, {}
        if not all(gate._finite_number(m.get(k), positive=True) for m in (bm, cm) for k in ("tpot_ms", "throughput_tps")):
            missing.append(index)
            continue
        comparisons.append(gate.summarize_deployment_pair({**bm, "quality": quality["baseline_value"]},
                                                        {**cm, "quality": quality["candidate_value"]}))
    if missing or failed:
        return {"decision": "pending", "reason": "paired observations missing or requests failed", "missing": missing, "failed": failed}, {}
    comparison = {k: median(row[k] for row in comparisons) if all(row[k] is not None for row in comparisons) else None for k in comparisons[0]}
    evidence = {key: all(m["observed"].get("execution", {}).get(key) is value for m in members.values()
                         if m["role"] == "candidate") for key, value in (("real_backend", True), ("quantized_path", True), ("fallback", False))}
    evidence["fallback"] = not evidence["fallback"]
    decision = gate.decide_quantized_deployment({"ready": True}, comparison, evidence,
               **plan["contract"]["limits"], max_quality_ratio=quality["max_degradation_ratio"])
    return decision, {"comparison": comparison, "pairs": comparisons, "execution_evidence": evidence}


def _derive_report(plan, prepared, members):
    artifact = prepared["artifact"]
    lineage = {"artifact_gate_reference": str(plan["source_path"].relative_to(plan["root"])),
               "artifact_gate_sha256": prepared["snapshot"]["upstream"]["sha256"]} if plan["project"] == "67" else {
               **prepared["upstream"]["lineage"], "deployment_reference": prepared["snapshot"]["reference"]}
    if plan["project"] == "67":
        decision, details = _deployment_decision(plan, prepared, members)
        sample = next(iter(members.values()))["observed"]
        payload = {"artifact": artifact, "backend": "vllm", "backend_version": sample.get("backend_version"),
                   "hardware": sample.get("hardware", {}), "workload": plan["contract"]["pair"],
                   "quality": prepared["upstream"]["quality"], "lineage": lineage,
                   "status": "ok" if decision["decision"] == "accept" else decision["decision"],
                   "evidence_level": "real_project_benchmark", "decision": decision, **details}
        report = quantized_deployment_record(payload)
    else:
        rows, decisions = {}, {}
        for name, spec in plan["contract"]["profiles"].items():
            metrics = _metrics(members[name])
            observed = members[name]["observed"]
            record = {"name": name, "artifact_id": artifact["artifact_id"], "profile_name": name,
                      "workload_id": spec["workload"]["id"], "evidence_level": "real_project_benchmark",
                      "status": "ok" if metrics is not None else "failed", "metrics": metrics or {}}
            result = gate.select_serving_candidate([record], {"name": name, **spec}, expected_artifact_id=artifact["artifact_id"])
            if metrics is None or any(not gate._finite_number(metrics.get(k)) for k in spec["slo"]):
                result = {**result, "decision": "pending"}
            decisions[name] = result["decision"]
            rows[name] = serving_profile_record({"artifact_id": artifact["artifact_id"], "profile": name,
                "workload_id": spec["workload"]["id"], **spec, **(metrics or {}), "backend": "vllm",
                "backend_version": observed.get("backend_version"), "hardware": observed.get("hardware", {}),
                "status": record["status"], "decision": {"decision": result["decision"]}, "lineage": lineage,
                "evidence_level": "real_project_benchmark", "failure": observed.get("failure")})
        identities = [(m["observed"].get("backend_version"), m["observed"].get("hardware"), m["observed"].get("dtype")) for m in members.values()]
        incomplete = any(v == "pending" for v in decisions.values())
        overall = "pending" if incomplete else "accept" if "accept" in decisions.values() else "tune"
        upstream_identity = (prepared["upstream"]["backend"].get("version"), prepared["upstream"].get("hardware"), plan["contract"]["dtype"])
        if not incomplete and (any(not all(x) for x in identities) or any(x != upstream_identity for x in identities)):
            overall = "reject"
        report = {"schema_version": "quantization-evidence/v1", "semantic_id": plan["contract"]["semantic_id"],
                  "stage": "serving_profile_summary", "artifact": artifact, "profiles": rows,
                  "profile_decisions": decisions, "decision": {"decision": overall,
                  "reason": "all three profiles must be measured; accept applies only to listed feasible profiles"}, "lineage": lineage}
    return {**report, "run_label": plan["run_label"], "invocation": invocation(plan, prepared),
            "measurement": gate._gate_file_reference(plan, plan["raw_path"])}


def read_project(plan):
    """Validate disk evidence and existing final assets without launching or writing."""
    prepared = _prepared(plan)
    if not plan["raw_path"].exists():
        if any(plan[k].exists() for k in ("report_path", "reference_path", "receipt_path")):
            raise ValueError("final assets exist without original measurement")
        return {"status": "pending", "decision": {"decision": "pending"}, "next_action": "measure_or_resume_complete_members"}
    raw = gate._gate_read_json(plan["raw_path"])
    if raw.get("schema_version") != "quantized-project-measurement/v3" or raw.get("invocation") != invocation(plan, prepared):
        raise ValueError("measurement schema/identity/source changed")
    expected = [f"pair{i}_{role}" for i in range(1, plan["contract"]["pair"]["pairs"] + 1) for role in ("baseline", "candidate")] if plan["project"] == "67" else list(plan["contract"]["profiles"])
    if set(raw.get("members", {})) != set(expected):
        raise ValueError("measurement member matrix is incomplete")
    members = {}
    for key, reference in raw["members"].items():
        path = member_path(plan, key)
        if reference != gate._gate_file_reference(plan, path):
            raise ValueError("saved member hash/path changed")
        member = gate._gate_read_json(path)
        role = key.split("_")[-1] if plan["project"] == "67" else "candidate"
        workload = {k: v for k, v in plan["contract"]["pair"].items() if k != "pairs"} if plan["project"] == "67" else plan["contract"]["profiles"][key]["workload"]
        if (member.get("invocation"), member.get("key"), member.get("role"), member.get("workload")) != (invocation(plan, prepared), key, role, workload):
            raise ValueError("member runtime contract changed")
        _check_member_files(plan, member)
        members[key] = member
    report = _derive_report(plan, prepared, members)
    if plan["report_path"].exists() and gate._gate_read_json(plan["report_path"]) != report:
        raise ValueError("report differs from recomputed evidence")
    if plan["reference_path"].exists():
        if plan["project"] != "67" or report["decision"]["decision"] != "accept" or not plan["report_path"].exists():
            raise ValueError("nonaccepted or incomplete evidence has a handoff")
        if gate._gate_read_json(plan["reference_path"]) != make_asset_reference("quantization.deployment_candidate", plan["report_path"]):
            raise ValueError("deployment handoff changed")
    for path, expected_row, reference_path in _profile_exports(plan, report):
        if path.exists() and gate._gate_read_json(path) != expected_row:
            raise ValueError("profile companion differs from recomputed evidence")
        if reference_path.exists() and (not path.exists() or gate._gate_read_json(reference_path) != make_asset_reference("quantization.serving_profile", path)):
            raise ValueError("profile reference changed")
    if plan["project"] == "83":
        allowed = {p for path, _, ref in _profile_exports(plan, report) for p in (path, ref)}
        for name in plan["contract"]["profiles"]:
            for suffix in ("companion", "reference"):
                path = plan["result_dir"] / f"83_{name}_{plan['run_label']}_{suffix}.json"
                if path.exists() and path not in allowed:
                    raise ValueError("unaccepted profile has an export")
    if plan["receipt_path"].exists() and gate._gate_read_json(plan["receipt_path"]) != _receipt(plan, report):
        raise ValueError("registration receipt changed")
    return {"status": "validated", "decision": report["decision"], "report": report}


def _profile_exports(plan, report):
    """Only feasible profiles from a complete matrix may be offered to project 72."""
    if plan["project"] != "83" or report["decision"]["decision"] != "accept":
        return []
    return [(plan["result_dir"] / f"83_{name}_{plan['run_label']}_companion.json",
             {**row, "run_label": plan["run_label"], "measurement": report["measurement"]},
             plan["result_dir"] / f"83_{name}_{plan['run_label']}_reference.json")
            for name, row in report["profiles"].items() if row["decision"]["decision"] == "accept"]


def _receipt(plan, report):
    paths = [plan["source_path"], plan["raw_path"], plan["report_path"]]
    if plan["project"] == "67" and report["decision"]["decision"] == "accept":
        paths.append(plan["reference_path"])
    for path, _, reference in _profile_exports(plan, report):
        paths.extend([path, reference])
    return {"schema_version": "quantized-project-registration/v1", "run_label": plan["run_label"],
            "decision": report["decision"], "files": [gate._gate_file_reference(plan, p) for p in paths]}


def close_project(plan):
    """Publish final assets idempotently, with accept-only deployment handoff."""
    if plan["stage"] != "close":
        raise ValueError("close requires close stage")
    checked = read_project(plan)
    if checked["status"] != "validated":
        return checked
    report = checked["report"]
    gate._gate_save_or_verify(plan["report_path"], report)
    if plan["project"] == "67" and report["decision"]["decision"] == "accept":
        gate._gate_save_or_verify(plan["reference_path"], make_asset_reference("quantization.deployment_candidate", plan["report_path"]))
    for path, row, reference in _profile_exports(plan, report):
        gate._gate_save_or_verify(path, row)
        gate._gate_save_or_verify(reference, make_asset_reference("quantization.serving_profile", path))
    gate._gate_save_or_verify(plan["receipt_path"], _receipt(plan, report))
    read_project(plan)
    return {"status": "closed", "decision": report["decision"], "report": str(plan["report_path"]), "receipt": str(plan["receipt_path"])}
