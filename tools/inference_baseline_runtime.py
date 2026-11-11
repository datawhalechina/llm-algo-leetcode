"""Fixed-workload baseline collection and read-only evidence checks for project 66."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path


PROTOCOL = "client-stream-baseline/v3"


def check_baseline_identity(expected, observed):
    """Require complete, matching model, tokenizer, workload and measurement identities."""
    keys = ("model_id", "model_revision", "tokenizer_revision", "workload_sha256",
            "profile_id", "dtype", "timing_protocol")
    missing = [key for key in keys if not expected.get(key) or not observed.get(key)]
    mismatches = [key for key in keys if expected.get(key) and observed.get(key)
                  and expected[key] != observed[key]]
    identity_ok = not missing and not mismatches
    return {"ok": identity_ok, "missing": missing, "mismatches": mismatches}


def check_baseline_request(row, max_tokens):
    """Separate missing evidence from invalid order, empty output and stop failures."""
    if row.get("ok") is not True:
        return {"ok": False, "missing": [], "failures": ["request_failed"]}
    fields = ("submit_offset_ms", "first_content_offset_ms", "last_content_offset_ms",
              "completion_offset_ms", "output_tokens", "finish_reason",
              "estimated_output_tokens", "stream_done_received")
    missing = [key for key in fields if row.get(key) is None]
    failures = [] if str(row.get("output_text", "")).strip() else ["empty_output"]
    if missing:
        return {"ok": False, "missing": missing, "failures": failures}
    times = [row[key] for key in fields[:4]]
    finite_times = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in times)
    if not finite_times:
        return {"ok": False, "missing": [], "failures": failures + ["nonfinite_time"]}
    submitted, first, last, completed = times
    tokens = row["output_tokens"]
    exact_count = row["estimated_output_tokens"] is False
    integer_count = isinstance(tokens, int) and not isinstance(tokens, bool)
    reason = row["finish_reason"]
    time_order_ok = 0 <= submitted <= first <= last <= completed
    token_count_ok = integer_count and 1 <= tokens <= max_tokens
    stop_ok = reason == "stop" or (reason == "length" and tokens == max_tokens)
    failures.extend(name for name, passed in (
        ("time_order", time_order_ok), ("token_count", token_count_ok),
        ("stop_reason", stop_ok), ("incomplete_stream", row["stream_done_received"] is True),
    ) if not passed)
    if not exact_count:
        missing.append("exact_output_tokens")
    return {"ok": not failures and not missing, "missing": missing, "failures": failures}


def check_baseline_totals(rows, metrics, expected_requests):
    """Reconcile per-run samples, token counts and throughput over the full window."""
    successful = [row for row in rows if row.get("ok") is True]
    failed_count = len(rows) - len(successful)
    output_counts = [row.get("output_tokens") for row in successful]
    count_values_valid = all(isinstance(value, int) and not isinstance(value, bool)
                             and value >= 0 for value in output_counts)
    total_tokens = sum(output_counts) if count_values_valid else None
    required = ("successful_requests", "failed_requests", "output_tokens",
                "output_token_throughput_per_s", "duration_s")
    missing = [key for key in required if metrics.get(key) is None]
    duration = metrics.get("duration_s")
    throughput = metrics.get("output_token_throughput_per_s")
    finite_window = isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0
    finite_rate = isinstance(throughput, (int, float)) and math.isfinite(throughput) and throughput >= 0
    expected_rate = total_tokens / duration if finite_window and total_tokens is not None else None
    completions = [row.get("completion_offset_ms") for row in rows]
    finite_completions = [value for value in completions
                          if isinstance(value, (int, float)) and math.isfinite(value)]
    window_covers_records = (finite_window and
                             duration * 1000 + 1.0 >= max(finite_completions, default=0.0))
    relative_tolerance, absolute_tolerance = 5e-4, 0.001
    counts_ok = (len(rows) == expected_requests
                 and metrics.get("successful_requests") == len(successful)
                 and metrics.get("failed_requests") == failed_count)
    tokens_ok = total_tokens is not None and metrics.get("output_tokens") == total_tokens
    throughput_ok = (finite_rate and window_covers_records and expected_rate is not None
                     and math.isclose(throughput, expected_rate,
                                      rel_tol=relative_tolerance, abs_tol=absolute_tolerance))
    failures = []
    if len(rows) != expected_requests or (
        all(metrics.get(key) is not None for key in ("successful_requests", "failed_requests"))
        and not counts_ok
    ):
        failures.append("sample_counts")
    if metrics.get("output_tokens") is not None and not tokens_ok:
        failures.append("token_total")
    if duration is not None and throughput is not None and not throughput_ok:
        failures.append("throughput_window")
    return {"ok": not failures and not missing, "missing": missing, "failures": failures,
            "observed_requests": len(rows), "observed_output_tokens": total_tokens}


def decide_baseline_readiness(evidence):
    """Give failures precedence over missing evidence; ready means referenceable only."""
    failures = list(evidence.get("failures", []))
    missing = list(evidence.get("missing", []))
    must_reject = bool(failures)
    can_reference = not failures and not missing
    status = "reject" if must_reject else ("ready" if can_reference else "tune")
    return {"status": status, "failures": failures, "missing": missing,
            "scope": "fixed_workload_baseline_not_task_quality",
            "next_action": "按身份和协议匹配引用" if can_reference else "修复失败或补采缺失证据"}


def sha256_file(path):
    """Return a source fingerprint without changing its contents."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repository_path(root, relative):
    """Resolve a repository-owned path and reject absolute or escaping references."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("合同路径必须是非空仓库相对路径")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("合同路径不能为绝对路径或包含 ..")
    resolved = (Path(root) / candidate).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("合同路径解析后越出仓库")
    return resolved


def _validate_experiment_contract(snapshot):
    """Validate supported design parameters without selecting model-specific defaults."""
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "schema_version", "contract_id", "contract_version", "semantic_id",
        "baseline_schema", "parameters"
    }:
        raise ValueError("实验合同字段不完整或包含未知字段")
    if snapshot["schema_version"] != "inference-baseline-contract/v1":
        raise ValueError("不支持的实验合同 schema")
    if snapshot["semantic_id"] != "INF-FLOAT-BASELINE":
        raise ValueError("实验合同项目身份不匹配")
    if snapshot["baseline_schema"] != "inference-baseline/v3":
        raise ValueError("不支持的基线报告合同")
    if not isinstance(snapshot["contract_id"], str) or not snapshot["contract_id"].strip():
        raise ValueError("缺少 contract_id")
    if not isinstance(snapshot["contract_version"], str) or not re.fullmatch(
        r"v[1-9][0-9]*(?:\.[0-9]+)*", snapshot["contract_version"]
    ):
        raise ValueError("contract_version 应使用 v3 等版本标识")
    params = snapshot["parameters"]
    required = {
        "profile_id", "model_id", "model_source", "dtype", "max_model_len",
        "gpu_memory_utilization", "enforce_eager", "enable_prefix_caching",
        "scheduling_policy", "workload_path", "workload_sha256", "num_prompts",
        "max_tokens", "concurrency", "warmup", "repeats", "temperature", "timing_protocol"
    }
    if not isinstance(params, dict) or set(params) != required:
        raise ValueError("实验参数缺失或包含未知参数")
    for key in ("max_model_len", "num_prompts", "max_tokens", "concurrency", "repeats"):
        if type(params[key]) is not int or params[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if type(params["warmup"]) is not int or params["warmup"] < 0:
        raise ValueError("warmup 必须为非负整数")
    for key in ("gpu_memory_utilization", "temperature"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} 必须为有限数值")
    if not 0 < params["gpu_memory_utilization"] <= 1 or params["temperature"] < 0:
        raise ValueError("显存比例或 temperature 超出范围")
    for key in ("enforce_eager", "enable_prefix_caching"):
        if type(params[key]) is not bool:
            raise ValueError(f"{key} 必须为布尔值")
    if params["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("dtype 必须显式指定浮点精度")
    if params["model_source"] not in ("huggingface", "modelscope", "auto"):
        raise ValueError("不支持的模型来源")
    if params["scheduling_policy"] not in ("fcfs", "priority"):
        raise ValueError("不支持的调度策略")
    if params["timing_protocol"] != PROTOCOL:
        raise ValueError("测量协议变化需要升级采集与核验代码，不能只改合同")
    for key in ("profile_id", "model_id", "workload_path"):
        if not isinstance(params[key], str) or not params[key].strip():
            raise ValueError(f"{key} 必须为非空字符串")
    if not isinstance(params["workload_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", params["workload_sha256"]
    ):
        raise ValueError("workload_sha256 必须为 SHA256")


def _contract_digest(snapshot):
    """Hash canonical JSON values; formatting changes do not change experiment meaning."""
    content = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _contract_from_binding(root, binding):
    """Validate an archived design snapshot without opening the current default file."""
    if not isinstance(binding, dict) or set(binding) != {"source_path", "sha256", "snapshot"}:
        raise ValueError("实验合同绑定不完整")
    _repository_path(root, binding["source_path"])
    snapshot = binding["snapshot"]
    _validate_experiment_contract(snapshot)
    _repository_path(root, snapshot["parameters"]["workload_path"])
    if binding["sha256"] != _contract_digest(snapshot):
        raise ValueError("实验合同快照 SHA256 不匹配")
    return {"schema_version": snapshot["baseline_schema"],
            "semantic_id": snapshot["semantic_id"], **copy.deepcopy(snapshot["parameters"])}


def load_baseline_plan(root, run_label, run_stage):
    """Load a default design for new runs, or the saved design for read/close."""
    if run_stage not in {"prepare", "measure", "read", "close"}:
        raise ValueError("RUN_STAGE 只能为 prepare / measure / read / close")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_label):
        raise ValueError("RUN_LABEL 只允许字母、数字、下划线和连字符，最多 96 字符")
    if run_stage != "prepare" and run_label == "replace_with_unique_label":
        raise ValueError("真实执行、读取和登记前请填写唯一 RUN_LABEL")
    root = Path(root).resolve()
    plan = {"root": str(root), "run_label": run_label, "run_stage": run_stage,
            "result_path": f"benchmarks/results/66_g0_vllm_{run_label}.json",
            "raw_path": f"benchmarks/results/66_g0_vllm_raw_{run_label}.json",
            "log_path": f"benchmarks/results/66_backend_{run_label}.log",
            "registration_path": f"benchmarks/results/66_baseline_registration_{run_label}.json",
            "reference_path": f"benchmarks/results/66_baseline_reference_{run_label}.json"}
    if run_stage in {"read", "close"}:
        result = root / plan["result_path"]
        report = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {}
        if not isinstance(report, dict):
            raise ValueError("测量报告必须是 JSON 对象")
        if "experiment_contract" in report:
            binding = copy.deepcopy(report["experiment_contract"])
            contract = _contract_from_binding(root, binding)
            return {**plan, "contract": contract, "experiment_contract": binding,
                    "contract_source": "saved_measurement"}
        return {**plan, "contract": {}, "experiment_contract": None,
                "contract_source": "legacy_unbound" if result.is_file() else "missing_result"}

    from tools.project_contracts import get_project

    project = get_project("INF-FLOAT-BASELINE", root / "benchmarks/project_registry.json")
    source = project.get("default_experiment_contract")
    contract_path = _repository_path(root, source)
    snapshot = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_experiment_contract(snapshot)
    binding = {"source_path": source, "sha256": _contract_digest(snapshot), "snapshot": snapshot}
    contract = _contract_from_binding(root, binding)
    workload = _repository_path(root, contract["workload_path"])
    if not workload.is_file():
        raise FileNotFoundError(workload)
    if sha256_file(workload) != contract["workload_sha256"]:
        raise ValueError("工作负载内容与版本化合同 SHA256 不匹配")
    return {**plan, "contract": contract, "experiment_contract": binding,
            "contract_source": "registry_default"}


def audit_baseline_report(report, contract, run_label):
    """Audit saved samples read-only, keeping historical protocols explicitly separate."""
    failures, missing, run_checks = [], [], []
    if report.get("failure"):
        failures.append("runtime_failure")
    if report.get("baseline_protocol_version") != PROTOCOL or "experiment_contract" not in report:
        missing.append("current_protocol_or_bound_contract; historical_report_is_unchanged")
        return {"failures": failures, "missing": missing, "runs": [],
                "decision": decide_baseline_readiness({"failures": failures, "missing": missing}),
                "historical_protocol": True,
                "historical_summary": {"model": report.get("model"),
                    "metrics": report.get("metrics"), "quality_check": report.get("quality")}}
    if report.get("run_label") != run_label:
        failures.append("run_label_mismatch")
    if report.get("baseline_contract") != contract:
        failures.append("contract_mismatch")
    identity = report.get("measurement_identity") or {}
    revision = (report.get("model_identity") or {}).get("resolved_snapshot")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        missing.append("immutable_model_revision")
    expected_identity = {key: contract[key] for key in (
        "model_id", "workload_sha256", "profile_id", "dtype", "timing_protocol")}
    expected_identity.update(model_revision=revision, tokenizer_revision=revision)
    identity_check = check_baseline_identity(expected_identity, identity)
    failures.extend("identity." + key for key in identity_check["mismatches"])
    missing.extend("identity." + key for key in identity_check["missing"])
    resource = report.get("resource_measurement") or {}
    peak = (report.get("metrics") or {}).get("peak_memory_mb")
    if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        missing.append("device_peak_memory")
    if resource.get("scope") != "visible_gpu_0_total_device_usage":
        missing.append("device_peak_scope")
    if not isinstance(resource.get("sample_count"), int) or resource["sample_count"] <= 0:
        missing.append("device_peak_samples")
    if not (report.get("environment_preflight") or {}).get("ready"):
        missing.append("profile_preflight")
    runs = report.get("runs") or [report]
    if len(runs) != contract["repeats"]:
        failures.append("repeat_count")
    for index, run in enumerate(runs, 1):
        rows = run.get("results")
        if not isinstance(rows, list) or not rows:
            missing.append(f"run{index}.request_records")
            continue
        totals = check_baseline_totals(rows, run.get("metrics") or {}, contract["num_prompts"])
        actual_workload = run.get("workload") or {}
        for key, expected in (("requests", contract["num_prompts"]),
                              ("max_tokens", contract["max_tokens"]),
                              ("concurrency", contract["concurrency"]),
                              ("warmup", contract["warmup"]), ("repeats", 1)):
            if actual_workload.get(key) is None:
                missing.append(f"run{index}.workload.{key}")
            elif actual_workload[key] != expected:
                failures.append(f"run{index}.workload.{key}")
        failures.extend(f"run{index}." + key for key in totals["failures"])
        missing.extend(f"run{index}." + key for key in totals["missing"])
        requests = [check_baseline_request(row, contract["max_tokens"]) for row in rows]
        for number, check in enumerate(requests, 1):
            failures.extend(f"run{index}.request{number}." + key for key in check["failures"])
            missing.extend(f"run{index}.request{number}." + key for key in check["missing"])
        valid = [row for row, check in zip(rows, requests) if check["ok"]]
        run_checks.append({
            "repeat": index, "requests": len(rows), "valid_requests": len(valid),
            "length_limited_requests": sum(row.get("finish_reason") == "length" for row in rows),
            "single_token_requests": sum(row.get("output_tokens") == 1 for row in rows),
            "totals": totals,
            "observations": [{
                "request_id": row.get("request_id"),
                "ttft_content_ms": row["first_content_offset_ms"] - row["submit_offset_ms"],
                "content_span_ms": row["last_content_offset_ms"] - row["first_content_offset_ms"],
                "stream_tail_ms": row["completion_offset_ms"] - row["last_content_offset_ms"],
                "request_e2e_ms": row["completion_offset_ms"] - row["submit_offset_ms"],
                "content_tpot_proxy_ms": (
                    (row["last_content_offset_ms"] - row["first_content_offset_ms"])
                    / (row["output_tokens"] - 1) if row["output_tokens"] > 1 else None),
            } for row in valid],
        })
        summaries = {}
        for key in ("ttft_content_ms", "content_tpot_proxy_ms", "request_e2e_ms", "stream_tail_ms"):
            values = sorted(item[key] for item in run_checks[-1]["observations"]
                            if item[key] is not None)
            summaries[key] = {
                "samples": len(values),
                "p50": values[max(0, math.ceil(0.5 * len(values)) - 1)] if values else None,
                "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None,
            }
        run_checks[-1]["client_metrics"] = summaries
    evidence = {"failures": failures, "missing": missing}
    return {**evidence, "runs": run_checks, "decision": decide_baseline_readiness(evidence),
            "historical_protocol": False, "identity": identity,
            "resource": {"device_peak_mb": peak, **resource}}


def read_baseline_evidence(plan):
    """Read and fingerprint existing results; never repair or rewrite historical files."""
    root = Path(plan["root"])
    path = root / plan["result_path"]
    if not path.is_file():
        evidence = {"failures": [], "missing": ["result_file"]}
        return {**evidence, "decision": decide_baseline_readiness(evidence), "runs": []}
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("baseline_protocol_version") == PROTOCOL and "experiment_contract" in report:
        binding = report["experiment_contract"]
        bound_contract = _contract_from_binding(root, binding)
        if bound_contract != plan["contract"] or binding != plan.get("experiment_contract"):
            raise RuntimeError("合同绑定与已加载计划不同，请重新加载该标签")
    evidence = audit_baseline_report(report, plan["contract"], plan["run_label"])
    evidence.update(result_path=plan["result_path"], result_sha256=sha256_file(path))
    if not evidence.get("historical_protocol"):
        raw = root / plan["raw_path"]
        if not raw.is_file():
            evidence["missing"].append("raw_file")
        elif sha256_file(raw) != report.get("raw_sha256"):
            evidence["failures"].append("raw_sha256_mismatch")
        else:
            evidence["raw_sha256"] = sha256_file(raw)
            raw_report = json.loads(raw.read_text(encoding="utf-8"))
            if any(report.get(key) != raw_report.get(key) for key in (
                "results", "runs", "metrics", "workload", "resource_measurement"
            )):
                evidence["failures"].append("raw_content_mismatch")
        if not (root / plan["log_path"]).is_file():
            evidence["missing"].append("backend_log")
        else:
            evidence["log_sha256"] = sha256_file(root / plan["log_path"])
        evidence["decision"] = decide_baseline_readiness(evidence)
    return evidence


def collect_baseline(plan, preflight):
    """Run one fixed service measurement, record failures and always stop the server."""
    from tools import inference_project_runtime as runtime
    from tools.project_runtime import save_json_record_once

    if plan["run_stage"] != "measure" or not preflight.get("ready"):
        raise RuntimeError("只有 measure 且 Profile 预检通过时允许启动服务")
    root, contract = Path(plan["root"]), plan["contract"]
    if _contract_from_binding(root, plan.get("experiment_contract")) != contract:
        raise RuntimeError("运行参数与绑定的实验合同不一致")
    if sha256_file(root / contract["workload_path"]) != contract["workload_sha256"]:
        raise RuntimeError("预检后工作负载发生变化，请重新准备合同")
    paths = [root / plan[key] for key in ("raw_path", "result_path", "log_path",
                                        "registration_path", "reference_path")]
    if any(path.exists() for path in paths):
        raise FileExistsError("本标签已有资产；测量必须使用新 RUN_LABEL")
    common = {"schema_version": "inference-benchmark/v1", "semantic_id": "INF-FLOAT-BASELINE",
              "run_label": plan["run_label"], "baseline_protocol_version": PROTOCOL,
              "experiment_contract": copy.deepcopy(plan["experiment_contract"]),
              "baseline_contract": contract, "environment_preflight": preflight,
              "run_mode": "real_backend",
              "collector_sha256": sha256_file(root / "tools/benchmark_inference_backend.py")
              if (root / "tools/benchmark_inference_backend.py").is_file() else None}
    server = log_handle = None
    try:
        server, log_handle, port, dtype, model_path = runtime.start_optional_vllm(
            model_id=contract["model_id"], model_source=contract["model_source"],
            cache_dir=root / "model_cache", dtype=contract["dtype"],
            max_model_len=contract["max_model_len"],
            gpu_memory_utilization=contract["gpu_memory_utilization"],
            enforce_eager=contract["enforce_eager"],
            enable_prefix_caching=contract["enable_prefix_caching"],
            scheduling_policy=contract["scheduling_policy"],
            served_model_name=contract["model_id"], log_path=root / plan["log_path"],
        )
        report = runtime.run_backend_benchmark(
            project="66", base_url=f"http://127.0.0.1:{port}", model=contract["model_id"],
            label=plan["run_label"], output=root / plan["raw_path"],
            workload=contract["workload_path"], num_prompts=contract["num_prompts"],
            max_tokens=contract["max_tokens"], concurrency=contract["concurrency"],
            warmup=contract["warmup"], repeats=contract["repeats"],
            temperature=contract["temperature"], backend="vllm", dtype=dtype,
            batch=1, cache_policy="enabled" if contract["enable_prefix_caching"] else "disabled",
            role="baseline",
            measure_device_peak_memory=True, include_requests=True,
        )
        report = copy.deepcopy(report)
        revision = Path(model_path).resolve().name
        identity = {key: contract[key] for key in (
            "model_id", "workload_sha256", "profile_id", "dtype", "timing_protocol")}
        identity.update(model_revision=revision, tokenizer_revision=revision)
        report.update(common, model_identity={"requested_id": contract["model_id"],
                      "resolved_snapshot": revision, "selected_dtype": dtype},
                      measurement_identity=identity, raw_sha256=sha256_file(root / plan["raw_path"]),
                      evidence_level="real_backend_baseline_measurement",
                      quality={"task_quality": "not_evaluated",
                               "check": "transport_output_and_stop_integrity_only"})
        save_json_record_once(root / plan["result_path"], report)
    except Exception as exc:
        if not (root / plan["result_path"]).exists():
            save_json_record_once(root / plan["result_path"], {
                **common, "evidence_level": "failed_real_backend_attempt",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            })
        raise
    finally:
        if server is not None:
            runtime.stop_optional_vllm(server, log_handle)


def close_baseline(plan, evidence):
    """Write an immutable receipt and qualified reference; recover partial close safely."""
    from tools.project_runtime import save_json_record_once

    if plan["run_stage"] != "close":
        raise RuntimeError("只有 close 可以登记报告与引用")
    if evidence.get("historical_protocol") or "result_sha256" not in evidence:
        raise RuntimeError("历史报告只读保留；新协议登记需要新标签测量结果")
    fresh = read_baseline_evidence(plan)
    if fresh != evidence:
        raise RuntimeError("读取后的证据发生变化，请重新运行读取与核对")
    root = Path(plan["root"])
    receipt = {"schema_version": "inference-baseline-registration/v1",
               "semantic_id": "INF-FLOAT-BASELINE", "run_label": plan["run_label"],
               "experiment_contract": copy.deepcopy(plan["experiment_contract"]),
               "contract": plan["contract"], "decision": evidence["decision"],
               "source": {"path": plan["result_path"], "sha256": evidence["result_sha256"]},
               "raw_source": {"path": plan["raw_path"], "sha256": evidence.get("raw_sha256")}}
    receipt["log_source"] = {"path": plan["log_path"], "sha256": evidence.get("log_sha256")}
    outputs = [(root / plan["registration_path"], receipt)]
    if evidence["decision"]["status"] == "ready":
        outputs.append((root / plan["reference_path"], {
            "schema_version": "inference-baseline-reference/v1",
            "semantic_id": "INF-FLOAT-BASELINE", "run_label": plan["run_label"],
            "experiment_contract": copy.deepcopy(plan["experiment_contract"]),
            "status": "ready", "scope": "matching_identity_and_measurement_protocol_only",
            "contract": plan["contract"], "source": receipt["source"],
            "raw_source": receipt["raw_source"], "task_quality": "not_evaluated",
        }))
    elif (root / plan["reference_path"]).exists():
        raise FileExistsError("非 ready 证据已有引用文件，停止登记")
    for path, payload in outputs:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"同标签登记内容冲突：{path}")
    already_complete = all(path.exists() for path, _ in outputs)
    for path, payload in outputs:
        if not path.exists():
            save_json_record_once(path, payload)
    return {"status": "already_complete" if already_complete else "registered",
            "run_label": plan["run_label"], "decision": evidence["decision"]["status"],
            "schema_version": receipt["schema_version"],
            "files": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
                      for path, _ in outputs]}
