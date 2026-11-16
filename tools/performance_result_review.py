"""Read-only CPU checks for the four performance mechanism result formats.

Recomputes recorded statistics, not GPU correctness or task-level quality.
Historical protocols remain historical; no missing measurement is synthesized.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from random import Random


FORMATS = {
    "PERF-TRACE": ("performance-trace-mechanism", "perf-trace-critical-path", ("v2", "v3", "v4")),
    "PERF-MEMORY-TIMELINE": ("performance-memory-timeline-mechanism", "perf-memory-peak-overlap", ("v2",)),
    "PERF-ROOFLINE": ("performance-roofline-mechanism", "perf-empirical-roofline", ("v2",)),
    "PERF-CANDIDATE": ("performance-candidate-mechanism", "perf-decoder-mlp-candidate-matrix", ("v1", "v2")),
}


def _require(condition, field):
    if not condition:
        raise ValueError(f"性能结果核对失败：{field}")


def _number(value, field, *, positive=False, integer=False):
    _require(type(value) is int if integer else type(value) in (int, float), field)
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    _require(valid, field)
    return value


def _match(saved, derived, field):
    if type(derived) in (bool, str):
        valid = type(saved) is type(derived) and saved == derived
    else:
        try:
            valid = (type(saved) in (int, float) and math.isfinite(saved)
                     and math.isfinite(derived)
                     and math.isclose(saved, derived, rel_tol=1e-9, abs_tol=1e-9))
        except OverflowError:
            valid = False
    _require(valid, field)


def _fields(saved, derived, prefix):
    _require(isinstance(saved, dict), prefix)
    for key, value in derived.items():
        _match(saved.get(key), value, f"{prefix}.{key}")


def _samples(values, count, field):
    _require(isinstance(values, list) and len(values) == count, f"{field}: sample_count")
    for index, value in enumerate(values):
        _number(value, f"{field}[{index}]", positive=True)
    return values


def _finite_json(value, field="record"):
    """Reject non-finite JSON extensions even in auxiliary measurement fields."""
    if type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        _require(finite, field)
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{field}[{index}]")


def _trace(record, workload, notes):
    metrics = record["metrics"]
    events = metrics["selected_events"]
    _require(isinstance(events, dict) and bool(events), "metrics.selected_events")
    for name, event in events.items():
        for key in ("cpu_time_total_us", "cuda_time_total_us"):
            _number(event[key], f"{name}.{key}")
        keys = ("cpu_annotation_count", "gpu_annotation_count") if workload["version"] == "v4" else ("count",)
        for key in keys:
            _number(event[key], f"{name}.{key}", integer=True)
    if workload["version"] == "v2":
        notes.append("旧版无独立墙钟重复样本，仅核对已有事件统计。")
        return {}
    count = _number(workload["benchmark_repeats"], "benchmark_repeats", positive=True, integer=True)
    times = {}
    for name in ("sync_each_step", "sync_at_end"):
        times[name] = statistics.median(_samples(metrics["timing_samples_ms"][name], count, name))
        _match(metrics["wall_time_ms"][name], times[name], f"wall_time_ms.{name}")
    speedup = times["sync_each_step"] / times["sync_at_end"]
    _match(metrics["sync_at_end_speedup"], speedup, "sync_at_end_speedup")
    annotations_complete = False
    if workload["version"] == "v4":
        expected = {}
        for path in times:
            expected[path] = 1
            for suffix in ("h2d_copy", "cuda_matmul", "host_sync"):
                expected[f"{path}.{suffix}"] = (1 if path == "sync_at_end" and suffix == "host_sync"
                                                else workload["profile_steps"])
        missing = sorted(set(expected) - set(events))
        _require(metrics["missing_required_events"] == missing, "missing_required_events")
        mismatches = {name: {"expected": value, "actual": events.get(name, {}).get("cpu_annotation_count", 0)}
                      for name, value in expected.items() if "." in name
                      and events.get(name, {}).get("cpu_annotation_count", 0) != value}
        _require(metrics["cpu_annotation_count_mismatches"] == mismatches, "cpu_annotation_count_mismatches")
        annotations_complete = not missing and not mismatches
    else:
        notes.append("旧版事件count未分离CPU/GPU，不能作为当前注解计数检查。")
    notes.append("事件统计与墙钟核对不替代输出数值核对或目标GPU内核追踪。")
    notes.append("decision保留原记录措辞；本次只重算统计与注解完整性，不据此采用同步策略。")
    return {"wall_time_ms": times, "sync_at_end_speedup": speedup,
            "cpu_annotations_complete": annotations_complete}


def _memory(record, workload, notes):
    count = _number(workload["repeat_steps"], "repeat_steps", positive=True, integer=True)
    groups = {}
    for name in ("baseline", "candidate"):
        rows = record["samples"][name]
        _require(isinstance(rows, list) and len(rows) == count, f"samples.{name}: sample_count")
        for row in rows:
            for key in ("latency_ms", "peak_allocated_bytes", "peak_reserved_bytes"):
                _number(row[key], f"samples.{name}.{key}", positive=True, integer=key != "latency_ms")
            _number(row["checksum"], f"samples.{name}.checksum")
            _require(row["peak_reserved_bytes"] >= row["peak_allocated_bytes"], f"{name}: reserved < allocated")
        groups[name] = {
            "latency_median_ms": statistics.median(row["latency_ms"] for row in rows),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows),
        }
        _fields(record[name], groups[name], name)
    base, candidate = groups["baseline"], groups["candidate"]
    comparison = {
        "allocated_reduction_ratio": 1 - candidate["peak_allocated_bytes"] / base["peak_allocated_bytes"],
        "reserved_reduction_ratio": 1 - candidate["peak_reserved_bytes"] / base["peak_reserved_bytes"],
        "latency_ratio": candidate["latency_median_ms"] / base["latency_median_ms"],
    }
    _fields(record["comparison"], comparison, "comparison")
    base_checksum = record["samples"]["baseline"][-1]["checksum"]
    close = abs(base_checksum - record["samples"]["candidate"][-1]["checksum"]) <= 1e-3 * abs(base_checksum)
    _match(record["quality"]["checksum_close"], close, "quality.checksum_close")
    decision = "accept_for_peak_reduction" if close and candidate["peak_allocated_bytes"] < base["peak_allocated_bytes"] else "tune"
    _match(record["decision"], decision, "decision")
    notes.append("按原协议核对末次校验和与构造张量峰值；不是完整输出或训练质量验证。")
    return {"groups": groups, "comparison": comparison, "decision": decision}


def _roofline(record, workload, notes):
    count = _number(workload["repeat_steps"], "repeat_steps", positive=True, integer=True)
    metrics = record["metrics"]
    seconds = {name: statistics.median(_samples(metrics[name + "_samples_ms"], count, name)) / 1000
               for name in ("copy", "peak_matmul", "candidate_matmul")}
    for key in ("copy_elements", "peak_matmul_size", "candidate_matmul_size"):
        _number(workload[key], key, positive=True, integer=True)
    _require(workload["matmul_precision"] in ("highest", "high", "medium"), "matmul_precision")
    bandwidth = 2 * workload["copy_elements"] * 4 / seconds["copy"]
    compute = 2 * workload["peak_matmul_size"] ** 3 / seconds["peak_matmul"]
    size = workload["candidate_matmul_size"]
    flops, byte_count = float(2 * size ** 3), float(3 * size ** 2 * 4)
    intensity = flops / byte_count
    measured = flops / seconds["candidate_matmul"]
    bound = min(compute, intensity * bandwidth)
    ceilings = {"bandwidth_bytes_per_s": bandwidth, "compute_flops_per_s": compute}
    roofline = {"arithmetic_intensity": intensity, "machine_balance": compute / bandwidth,
                "regime": "bandwidth_bound" if intensity * bandwidth < compute else "compute_bound",
                "bound_flops": bound, "measured_flops": measured, "roofline_efficiency": measured / bound}
    _fields(record["empirical_ceilings"], ceilings, "empirical_ceilings")
    _fields(metrics["candidate_workload"], {"flops": flops, "minimum_bytes": byte_count,
                                         "arithmetic_intensity": intensity}, "candidate_workload")
    _fields(metrics["roofline"], roofline, "roofline")
    _match(record["decision"], "ready_for_bottleneck_hypothesis", "decision")
    notes.append("经验参考比值允许超过1；重算吞吐不代表矩阵输出正确或已确认实际内核。")
    return {"empirical_ceilings": ceilings, "roofline": roofline}


def _candidate(record, workload, notes):
    count = _number(workload["repeat_steps"], "repeat_steps", positive=True, integer=True)
    tokens = workload["token_counts"]
    _require(isinstance(tokens, list) and bool(tokens), "token_counts")
    for token in tokens:
        _number(token, "token_count", positive=True, integer=True)
    _require(len(set(tokens)) == len(tokens), "duplicate token_counts")
    for key in ("minimum_speedup", "minimum_cosine"):
        _number(workload[key], key, positive=True)
    _require(workload["minimum_speedup"] >= 1 and workload["minimum_cosine"] <= 1, "candidate thresholds")
    current = workload["version"] == "v2"
    if current:
        protocol = record["matrix"]["protocol"]
        _fields(protocol, {"measurement_order": "rotating_order_within_each_round", "paired_rounds": count}, "protocol")
        _number(workload["maximum_p95_ratio"], "maximum_p95_ratio", positive=True)
    else:
        notes.append("v1没有轮换协议与P95门槛，保留旧版逐负载判断。")
    variants = record["matrix"]["variants"]
    names = ("baseline_fp32_serial", "bf16_serial", "fp32_overlap")
    _require(set(variants) == set(names), "matrix.variants")
    loads = {}
    for name in names:
        rows = variants[name]["loads"]
        _require(isinstance(rows, list) and len(rows) == len(tokens), f"{name}: load_count")
        by_token = {}
        for row in rows:
            token = _number(row["token_count"], "token_count", positive=True, integer=True)
            _require(token in tokens and token not in by_token, f"{name}: duplicate_or_unknown_load")
            values = _samples(row["latency"]["samples_ms"], count, f"{name}/{token}")
            latency = {"median_ms": statistics.median(values),
                       "p95_ms": sorted(values)[int(0.95 * (len(values) - 1))]}
            _fields(row["latency"], latency, f"{name}/{token}.latency")
            for key in ("model_bytes", "buffer_bytes", "matrix_peak_allocated_bytes") if current else ("model_bytes", "peak_allocated_bytes"):
                _number(row[key], f"{name}.{key}", positive=True, integer=True)
            cosine = row["cosine_vs_fp32"]
            _require(type(cosine) in (int, float) and math.isfinite(cosine), "cosine_vs_fp32")
            if current:
                _require(-1 <= cosine <= 1, "cosine_vs_fp32 range")
            elif not -1 <= cosine <= 1:
                notes.append(f"旧版{name}/{token}余弦值{cosine}超出[-1,1]；原值保留，旧质量标记仅复算，不升级为数值正确性通过。")
            by_token[token] = row
        loads[name] = by_token
    saved = record["comparisons"]
    _require(isinstance(saved, list) and len(saved) == 2 * len(tokens), "comparisons: count")
    comparisons = []
    seen = set()
    for item in saved:
        name, token = item["candidate"], item["token_count"]
        _require(type(token) is int and name in names[1:] and token in tokens and (name, token) not in seen, "comparison identity")
        seen.add((name, token))
        base, row = loads[names[0]][token], loads[name][token]
        speedup = base["latency"]["median_ms"] / row["latency"]["median_ms"]
        quality = row["cosine_vs_fp32"] >= workload["minimum_cosine"]
        direction = row["model_bytes"] < base["model_bytes"] if name == "bf16_serial" else row["latency"]["median_ms"] < base["latency"]["median_ms"]
        derived = {"speedup": speedup, "model_bytes_reduction": 1 - row["model_bytes"] / base["model_bytes"],
                   "cosine_vs_fp32": row["cosine_vs_fp32"], "quality_passed": quality, "direction_passed": direction}
        tail = True
        if current:
            ratio = row["latency"]["p95_ms"] / base["latency"]["p95_ms"]
            tail = ratio <= workload["maximum_p95_ratio"]
            derived.update(p95_ratio=ratio, tail_passed=tail,
                           active_bytes_reduction=1 - (row["model_bytes"] + row["buffer_bytes"]) / (base["model_bytes"] + base["buffer_bytes"]),
                           matrix_peak_allocated_bytes=row["matrix_peak_allocated_bytes"])
        else:
            derived["peak_bytes_delta"] = row["peak_allocated_bytes"] - base["peak_allocated_bytes"]
        derived["decision"] = ("reject" if not quality or not direction else
                               "accept_for_load" if speedup >= workload["minimum_speedup"] and tail else "tune")
        _fields(item, derived, f"comparison/{name}/{token}")
        comparisons.append({"candidate": name, "token_count": token, **derived})
    _match(record["decision"], "review_per_load", "decision")
    notes.append("余弦门槛来自独立前向，不能替代重叠路径逐元素或任务质量验证。")
    return {"comparisons": comparisons}


def review_performance_result(record, *, semantic_id, run_label, workload_id, current_version):
    """Validate a supported record and recompute available statistics without mutation."""
    _require(isinstance(record, dict), "JSON object")
    _finite_json(record)
    _require(semantic_id in FORMATS, "unsupported semantic_id")
    family, expected_workload, versions = FORMATS[semantic_id]
    _require(record.get("semantic_id") == semantic_id, "semantic_id")
    _require(record.get("run_label") == run_label and isinstance(run_label, str) and bool(run_label), "run_label")
    workload = record.get("workload")
    _require(isinstance(workload, dict), "workload")
    version = workload.get("version")
    _require(version in versions and current_version == versions[-1], "workload.version")
    schema_version = "v1" if semantic_id == "PERF-CANDIDATE" else version
    _require(record.get("schema_version") == f"{family}/{schema_version}", "schema_version")
    _require(workload.get("id") == workload_id == expected_workload, "workload.id")
    _require(record.get("profile_id") == "PERF-CUDA-MECHANISM", "profile_id")
    _require("failure" in record and record["failure"] is None, "failure")
    for key in ("framework", "hardware"):
        _require(isinstance(record.get(key), dict) and bool(record[key]), key)
    dimensions = {
        "PERF-TRACE": ("matrix_size", "profile_steps"),
        "PERF-MEMORY-TIMELINE": ("rows", "hidden_size"),
        "PERF-ROOFLINE": ("copy_elements", "peak_matmul_size", "candidate_matmul_size"),
        "PERF-CANDIDATE": ("hidden_size", "intermediate_size", "pipeline_batches"),
    }
    for key in dimensions[semantic_id]:
        _number(workload.get(key), f"workload.{key}", positive=True, integer=True)
    _number(workload.get("warmup_steps"), "workload.warmup_steps", integer=True)
    if semantic_id == "PERF-CANDIDATE":
        _number(workload.get("seed"), "workload.seed", integer=True)
    notes = []
    if version != current_version:
        notes.append(f"历史协议{version}；按该版本检查，不补齐为{current_version}。")
    validator = {"PERF-TRACE": _trace, "PERF-MEMORY-TIMELINE": _memory,
                 "PERF-ROOFLINE": _roofline, "PERF-CANDIDATE": _candidate}[semantic_id]
    try:
        derived = validator(record, workload, notes)
    except (KeyError, TypeError, IndexError, OverflowError, ZeroDivisionError) as error:
        raise ValueError(f"性能结果字段缺失、类型或数值范围错误：{error}") from error
    return {"status": "record_statistics_checked" if version == current_version else "historical_record_checked",
            "workload_version": version, "current_protocol": version == current_version,
            "recomputed": derived, "notes": notes}


def read_performance_result(path, **expected):
    """Read one byte snapshot; return its unchanged data, review and source SHA256."""
    path = Path(path)
    source = path.read_bytes()
    record = json.loads(source)
    review = review_performance_result(record, **expected)
    review["source_sha256"] = hashlib.sha256(source).hexdigest()
    if expected["semantic_id"] == "PERF-TRACE":
        trace = path.with_suffix(".trace.json")
        artifacts = record.get("artifacts", {})
        _number(artifacts.get("trace_bytes"), "artifacts.trace_bytes", positive=True, integer=True)
        _require(isinstance(artifacts.get("trace_path"), str)
                 and Path(artifacts["trace_path"]).name == trace.name, "artifacts.trace_path identity")
        if trace.is_file():
            _require(trace.stat().st_size == artifacts["trace_bytes"], "trace_bytes")
        else:
            review["notes"].append("本地缺少同名轨迹；JSON统计可读，暂不能展开轨迹检查。")
        review["local_trace_exists"] = trace.is_file()
        review["notes"].append("轨迹按结果旁的同名文件定位；原记录未保存轨迹哈希，文件大小一致不证明内容一致。")
    return record, review


def read_input_io_result(path, *, semantic_id, run_label):
    """Check input summaries or recompute saved I/O pipeline samples, read-only."""
    source = Path(path).read_bytes()
    record = json.loads(source)
    _require(isinstance(record, dict), "JSON object")
    _finite_json(record)
    _require(semantic_id in ("PERF-INPUT", "PERF-IO") and record.get("semantic_id") == semantic_id, "semantic_id")
    _require(record.get("run_label") == run_label, "run_label")
    _require(record.get("profile_id") == "PERF-CUDA-MECHANISM", "profile_id")
    _require("failure" in record and record["failure"] is None, "failure")
    workload, metrics, correctness = record["workload"], record["metrics"], record["correctness"]
    notes, derived = [], {}
    schema = record["schema_version"]
    if semantic_id == "PERF-INPUT":
        _require(schema == "performance-mechanism-gpu/v1", "schema_version")
        for key in ("batch_size", "seq_len", "feature_dim", "repeat_steps", "candidate_workers", "prefetch_factor"):
            _number(workload[key], key, positive=True, integer=True)
        _number(workload["warmup_steps"], "warmup_steps", integer=True)
        specs = {"pageable_single": (0, False, False), "pinned_single": (0, True, True),
                 "prefetched_pinned": (workload["candidate_workers"], True, True)}
        _require(set(metrics) == set(specs), "input variants")
        for name, (workers, pinned, non_blocking) in specs.items():
            row = metrics[name]
            _number(row["workers"], "workers", integer=True)
            _fields(row, {"workers": workers, "pin_memory": pinned, "non_blocking": non_blocking}, name)
            for key in ("batch_wait_median_ms", "h2d_median_ms", "step_median_ms", "step_p95_ms",
                        "samples_per_second", "peak_allocated_mb"):
                _number(row[key], f"{name}.{key}", positive=key not in ("batch_wait_median_ms",))
            _require(row["step_p95_ms"] >= row["step_median_ms"], f"{name}: p95 < median")
            _require(type(row["checksum"]) in (int, float), f"{name}.checksum")
        base, candidate = metrics["pageable_single"], metrics["prefetched_pinned"]
        diff = abs(candidate["checksum"] - base["checksum"]) / max(abs(base["checksum"]), 1e-12)
        passed = diff <= 1e-5
        _fields(correctness, {"checksum_relative_diff": diff, "passed": passed}, "correctness")
        qualifies = passed and candidate["samples_per_second"] > base["samples_per_second"] and candidate["step_p95_ms"] < base["step_p95_ms"]
        accepted = ("accept_prefetch_candidate", "accept_prefetch_for_throughput_and_tail_latency")
        _require(record["decision"] in accepted if qualifies else record["decision"] == "tune_input_pipeline", "decision")
        derived = {"checksum_relative_diff": diff, "prefetch_throughput_and_tail_conditions": qualifies}
        notes.append("只有汇总，不能重算中位数、P95或吞吐；累计校验只比较基线与预取配置，不代表逐元素或完整训练质量验证。")
        status = "summary_fields_checked"
    else:
        versions = {"performance-mechanism-gpu/v1": "v1", "performance-io-mechanism/v2": "v2",
                    "performance-io-mechanism/v3": "v3"}
        _require(schema in versions, "schema_version")
        version = versions[schema]
        if version != "v1":
            _require(workload["version"] == version, "workload.version")
        for key in ("tensor_mib", "repeats"):
            _number(workload[key], key, positive=True, integer=True)
        _number(workload["warmup"], "warmup", integer=True)
        _require(record["storage"]["cache_scope"] == "local_warm_cache", "storage.cache_scope")
        phases = metrics if version == "v1" else metrics["phase_service_times"]
        names = ("artifact_save", "artifact_load_warm_cache", "pageable_h2d", "pinned_non_blocking_h2d", "output_d2h_and_save")
        _require(set(phases) == set(names), "phase_service_times")
        for name in names:
            for key in ("median_ms", "p95_ms"):
                _number(phases[name][key], f"{name}.{key}", positive=True)
            _require(phases[name]["p95_ms"] >= phases[name]["median_ms"], f"{name}: p95 < median")
        notes.append("分段只有汇总，不能重算其分布；独立阶段不相加为完整任务时间。读取JSON不重新加载临时张量文件。")
        if version == "v1":
            _require(type(correctness["passed"]) is bool, "correctness.passed")
            notes.append("历史v1没有流水线样本；保留原决策，不能复核流水线收益或文件往返内容。")
            status = "summary_fields_checked"
        else:
            count = workload["repeats"]
            pipeline = metrics["pipeline_critical_path"]
            summaries = {}
            for name, sample_key in (("serial_copy_then_compute", "serial_samples_ms"), ("double_buffer_overlap", "overlap_samples_ms")):
                values = _samples(pipeline[sample_key], count, sample_key)
                summaries[name] = {"median_ms": statistics.median(values), "p95_ms": sorted(values)[int(0.95 * (count - 1))]}
                _fields(pipeline[name], summaries[name], name)
            base, overlap = summaries["serial_copy_then_compute"], summaries["double_buffer_overlap"]
            derived = {"median_speedup": base["median_ms"] / overlap["median_ms"],
                       "median_exposed_reduction_ms": base["median_ms"] - overlap["median_ms"],
                       "p95_reduction_ms": base["p95_ms"] - overlap["p95_ms"]}
            _fields(pipeline, derived, "pipeline")
            for key in ("artifact_roundtrip_passed", "pipeline_passed"):
                _require(type(correctness[key]) is bool, key)
            diff = _number(correctness["pipeline_checksum_relative_diff"], "pipeline_checksum_relative_diff")
            _match(correctness["pipeline_passed"], diff <= 1e-5, "pipeline_passed")
            if version == "v3":
                for key in ("artifact_load_passed", "output_reload_passed"):
                    _require(type(correctness[key]) is bool, key)
                _require(not correctness["artifact_roundtrip_passed"] or
                         (correctness["artifact_load_passed"] and correctness["output_reload_passed"]), "file check contradiction")
            else:
                notes.append("历史v2往返字段仅支持主机—设备—主机张量检查，不能证明加载文件或输出重载内容一致。")
            threshold = _number(workload["minimum_speedup"], "minimum_speedup", positive=True)
            _require(threshold >= 1, "minimum_speedup < 1")
            if not correctness["artifact_roundtrip_passed"] or not correctness["pipeline_passed"]:
                decision = "reject_correctness_failure"
            elif derived["median_speedup"] >= threshold and overlap["p95_ms"] < base["p95_ms"]:
                decision = "accept_overlap_for_local_pipeline"
            elif overlap["median_ms"] < base["median_ms"]:
                decision = "tune_overlap_small_gain"
            else:
                decision = "keep_serial_path"
            _match(record["decision"], decision, "decision")
            derived.update(summaries=summaries, decision=decision)
            notes.append("重算只覆盖复制＋计算流水线；末次累计校验不等于逐轮逐元素检查，时间差不直接等于暴露I/O减少。")
            status = "record_statistics_checked" if version == "v3" else "historical_record_checked"
    return record, {"status": status, "source_sha256": hashlib.sha256(source).hexdigest(),
                    "recomputed": derived, "notes": notes}


def read_composition_result(path, *, run_label):
    """Recompute historical or guarded composition decisions without upgrading evidence."""
    source = Path(path).read_bytes()
    record = json.loads(source)
    _require(isinstance(record, dict), "JSON object")
    _finite_json(record)
    _require(record.get("semantic_id") == "PERF-COMPOSITION", "semantic_id")
    _require(record.get("run_label") == run_label, "run_label")
    schema = record.get("schema_version")
    _require(schema in ("performance-composition-mechanism/v1", "performance-composition-mechanism/v2"), "schema_version")
    guarded = schema.endswith("/v2")
    workload = record["workload"]
    version = workload["version"]
    _require(version in ("v1", "v2"), "workload.version")
    _require(workload["id"] == "perf-decoder-mlp-composition-matrix", "workload.id")
    _require(record.get("profile_id") == "PERF-CUDA-MECHANISM", "profile_id")
    _require("failure" in record and record["failure"] is None, "failure")
    if guarded:
        _require(version == "v2" and record.get("decision_version") == "guarded_fallback/v2", "decision_version")
    else:
        _require(record.get("decision_version") is None, "unknown historical decision_version")
    notes = ["独立前向相似度不替代双缓冲逐元素验证；服务时间最长不等于已确认瓶颈。"]
    if not guarded:
        notes.append("历史schema v1按最快单项复算；合格回退规则另列复核，不补写为原实验已执行。")
    count = _number(workload["repeat_steps"], "repeat_steps", positive=True, integer=True)
    tokens = workload["token_counts"]
    _require(isinstance(tokens, list) and bool(tokens), "token_counts")
    for token in tokens:
        _number(token, "token_count", positive=True, integer=True)
    _require(len(set(tokens)) == len(tokens), "duplicate token_counts")
    gain = _number(workload["minimum_marginal_gain"], "minimum_marginal_gain")
    _require(gain < 1, "minimum_marginal_gain >= 1")
    cosine_threshold = workload["minimum_cosine"]
    _require(type(cosine_threshold) in (int, float) and -1 <= cosine_threshold <= 1, "minimum_cosine")
    tail_limit = _number(workload["maximum_p95_ratio"], "maximum_p95_ratio", positive=True) if version == "v2" else None
    if version == "v2":
        _fields(record["matrix"]["protocol"], {"measurement_order": "rotating_order_within_each_round",
                                                "paired_rounds": count}, "protocol")
    else:
        notes.append("工作负载v1没有P95门槛与轮换协议，不能据此确认当前尾延迟条件。")
    names = ("baseline_fp32_serial", "bf16_serial", "fp32_overlap", "bf16_overlap")
    variants = record["matrix"]["variants"]
    _require(set(variants) == set(names), "matrix.variants")
    loads = {}
    for name in names:
        rows = variants[name]["loads"]
        _require(isinstance(rows, list) and len(rows) == len(tokens), f"{name}: load_count")
        loads[name] = {}
        for row in rows:
            token = _number(row["token_count"], "token_count", positive=True, integer=True)
            _require(token in tokens and token not in loads[name], "duplicate_or_unknown_load")
            values = _samples(row["latency"]["samples_ms"], count, f"{name}/{token}")
            _fields(row["latency"], {"median_ms": statistics.median(values),
                                    "p95_ms": sorted(values)[int(0.95 * (len(values) - 1))]}, f"{name}/{token}.latency")
            cosine = row["cosine_vs_fp32"]
            _require(type(cosine) in (int, float), "cosine_vs_fp32")
            if version == "v2":
                _require(-1 <= cosine <= 1, "cosine_vs_fp32 range")
            elif not -1 <= cosine <= 1:
                notes.append(f"历史{name}/{token}余弦值{cosine}越界，保留原值，不视为数值正确性通过。")
            loads[name][token] = row
    comparisons = record["comparisons"]
    _require(isinstance(comparisons, list) and len(comparisons) == len(tokens), "comparisons count")
    results, seen = [], set()
    for item in comparisons:
        token = item["token_count"]
        _require(type(token) is int and token in tokens and token not in seen, "comparison identity")
        seen.add(token)
        base = loads[names[0]][token]
        combo = loads[names[3]][token]
        singles = {name: loads[name][token] for name in names[1:3]}
        base_ms, combo_ms = base["latency"]["median_ms"], combo["latency"]["median_ms"]
        checks = {name: {"quality_passed": row["cosine_vs_fp32"] >= cosine_threshold,
                         "tail_passed": row["latency"]["p95_ms"] / base["latency"]["p95_ms"] <= tail_limit if tail_limit else None,
                         "faster_than_baseline": row["latency"]["median_ms"] < base_ms}
                  for name, row in singles.items()}
        eligible = {name: row for name, row in singles.items() if all(checks[name].values())}
        guarded_best = min(eligible, key=lambda name: (eligible[name]["latency"]["median_ms"], name)) if eligible else None
        best = guarded_best if guarded else min(singles, key=lambda name: (singles[name]["latency"]["median_ms"], name))
        reference_name = best or names[0]
        reference = singles[best] if best else base
        baseline_gain = (base_ms - combo_ms) / base_ms
        marginal_gain = (reference["latency"]["median_ms"] - combo_ms) / reference["latency"]["median_ms"]
        p95_ratio = combo["latency"]["p95_ms"] / reference["latency"]["p95_ms"]
        tail = p95_ratio <= tail_limit if tail_limit else True
        quality = combo["cosine_vs_fp32"] >= cosine_threshold
        keep = quality and tail and marginal_gain >= gain
        if guarded:
            keep = keep and marginal_gain > 0 and baseline_gain > 0
        decision = "keep_combination" if keep else "keep_single" if best else "rollback_baseline"
        derived = {"best_single": best, "gain_vs_baseline": baseline_gain,
                   "gain_vs_best_single": marginal_gain if best else None,
                   "combination_cosine_vs_fp32": combo["cosine_vs_fp32"],
                   "decision": decision, "rollback_to": reference_name}
        services = combo["service_times"]
        for key in ("compute_median_ms", "h2d_median_ms"):
            _number(services[key], key, positive=True)
        _fields(item["combination_service_times"], services, "combination_service_times")
        derived["dominant_service_time"] = "compute" if services["compute_median_ms"] >= services["h2d_median_ms"] else "h2d"
        if version == "v2":
            derived.update(combination_p95_ratio_vs_best_single=p95_ratio if best else None, tail_passed=tail)
        if guarded:
            for name in singles:
                _fields(item["single_checks"][name], checks[name], f"single_checks.{name}")
            derived.update(reference_name=reference_name, gain_vs_reference=marginal_gain,
                           combination_p95_ratio_vs_reference=p95_ratio)
        for key, value in derived.items():
            if value is None:
                _require(key in item and item[key] is None, key)
            else:
                _match(item.get(key), value, key)
        results.append({"token_count": token, **derived, "guarded_review_reference": (guarded_best or names[0]) if tail_limit else None,
                        "single_checks_recomputed": checks})
    _match(record["decision"], "review_per_load", "decision")
    return record, {"status": "record_statistics_checked" if guarded else "historical_record_checked",
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                    "recomputed_comparisons": results, "notes": notes}


def read_regression_result(path, *, run_label):
    """Recompute paired bootstrap statistics and recorded decisions on the CPU.

    Failed measurements remain readable. Missing historical threshold settings
    are reported, not silently replaced with today's notebook defaults.
    """
    source = Path(path).read_bytes()
    record = json.loads(source)
    _require(isinstance(record, dict), "JSON object")
    _finite_json(record)
    _require(record.get("semantic_id") == "PERF-REGRESSION", "semantic_id")
    _require(record.get("run_label") == run_label, "run_label")
    workload = record["workload"]
    version = workload["version"]
    _require(version in ("v3", "v4"), "workload.version")
    _require(record.get("schema_version") == f"performance-regression-mechanism/{version}", "schema_version")
    _require(workload.get("id") == "perf-paired-regression", "workload.id")
    notes = ["回归通过只表示未确认超过允许变慢幅度，不表示获得加速。"]
    for key in ("paired_trials", "minimum_pairs", "bootstrap_samples"):
        _number(workload[key], key, positive=True, integer=True)
    _number(workload["seed"], "seed", integer=True)
    declared_threshold = workload.get("regression_threshold")
    if declared_threshold is None:
        notes.append("历史记录未登记regression_threshold；仅按原decision中的边界复算，不能证明该门槛已预先设定。")
    else:
        _number(declared_threshold, "regression_threshold")
    loads = record["metrics"]["load_matrix"] if version == "v4" else [{
        "token_count": workload["batch_size"], "paired_runs": record["metrics"]["paired_runs"],
        "summary": record["metrics"]["summary"], "quality": record["quality"],
        "failure": record["failure"], "decision": record["decision"],
    }]
    tokens = workload["token_counts"] if version == "v4" else [workload["batch_size"]]
    _require(isinstance(tokens, list) and bool(tokens), "token_counts")
    for token in tokens:
        _number(token, "token_count", positive=True, integer=True)
    _require(len(set(tokens)) == len(tokens) and len(loads) == len(tokens), "load_count")
    results, seen, failures = [], set(), []
    for load in loads:
        token = load["token_count"]
        _require(type(token) is int and token in tokens and token not in seen, "load identity")
        seen.add(token)
        pairs = load["paired_runs"]
        _require(isinstance(pairs, list), "paired_runs")
        planned = workload["paired_trials"]
        if version == "v4":
            for key, value in (("planned_pairs", planned), ("completed_pairs", len(pairs))):
                if key in load:
                    _match(load[key], value, key)
                else:
                    notes.append(f"历史负载{token}缺少{key}；使用工作负载计划量和实际配对列表核对，不补写字段。")
        ids, ratios = [], []
        for pair in pairs:
            trial = _number(pair["trial_id"], "trial_id", integer=True)
            _require(trial < planned and trial not in ids, "duplicate_or_out_of_plan_trial")
            ids.append(trial)
            base = _number(pair["baseline_ms"], "baseline_ms", positive=True)
            candidate = _number(pair["candidate_ms"], "candidate_ms", positive=True)
            ratios.append(_number(candidate / base, "paired_ratio", positive=True))
        _require(ids == list(range(len(pairs))), "paired_run_order")
        summary = None
        if ratios:
            rng = Random(workload["seed"] + token if version == "v4" else workload["seed"])
            medians = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                             for _ in range(workload["bootstrap_samples"]))
            summary = {"pair_count": len(ratios), "median_ratio": statistics.median(ratios),
                       "slower_fraction": sum(value > 1 for value in ratios) / len(ratios),
                       "ci_low": medians[int(round((len(medians) - 1) * 0.05))],
                       "ci_high": medians[int(round((len(medians) - 1) * 0.95))]}
            _fields(load["summary"], summary, f"load/{token}.summary")
        else:
            _require(load["summary"] is None, "summary_without_pairs")
        quality = load["quality"]["outputs_allclose"]
        _require(type(quality) is bool, "quality.outputs_allclose")
        if load["failure"] is not None:
            _require(isinstance(load["failure"], dict), "load.failure")
            decision = {"decision": "reject", "reason": "candidate_execution_failed"}
            failures.append({"token_count": token, **load["failure"]})
        elif len(pairs) != planned:
            decision = {"decision": "tune", "reason": "incomplete_pairs"}
        else:
            boundary = (1 + declared_threshold if declared_threshold is not None
                        else load["decision"]["regression_boundary"])
            _number(boundary, "regression_boundary", positive=True)
            _require(boundary >= 1, "regression_boundary < 1")
            if not quality:
                state, reason = "reject", "quality_guard_failed"
            elif len(pairs) < workload["minimum_pairs"]:
                state, reason = "tune", "insufficient_pairs"
            elif summary["ci_low"] > boundary:
                state, reason = "reject", "latency_regression"
            elif summary["ci_high"] <= boundary:
                state, reason = "accept", "no_confirmed_regression"
            else:
                state, reason = "tune", "interval_crosses_threshold"
            decision = {"decision": state, "reason": reason, "regression_boundary": boundary}
        _fields(load["decision"], decision, f"load/{token}.decision")
        results.append({"token_count": token, "summary": summary, "decision": decision})
    if version == "v4":
        _require(record["failure"] == (failures or None), "overall failure list")
        decisions = [item["decision"]["decision"] for item in results]
        state = "reject" if "reject" in decisions else "tune" if "tune" in decisions else "accept"
        reason = {"reject": "one_or_more_loads_rejected", "tune": "one_or_more_loads_inconclusive",
                  "accept": "no_confirmed_regression_across_loads"}[state]
        _fields(record["decision"], {"decision": state, "reason": reason}, "overall decision")
    return record, {"status": "record_statistics_checked" if version == "v4" and declared_threshold is not None
                    else "historical_record_checked", "source_sha256": hashlib.sha256(source).hexdigest(),
                    "recomputed_loads": results, "notes": notes}
