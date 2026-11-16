"""Measurement helpers for the Part 05 FlashAttention delivery project."""

from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_ID = "OP-INTEGRATED-DELIVERY"
SEQUENCE_LENGTHS = (128, 512, 1024, 2048)
VARIANTS = ("torch_sdpa_auto", "triton_flash_attention")
RESULT_ROOT = "benchmarks/results/part05_operator_delivery"
SDPA_BACKENDS = ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
                 "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_attention_math")


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Validate the small fixed protocol before any CUDA work begins."""
    errors: list[str] = []
    if not isinstance(protocol, Mapping):
        return ["protocol must be a mapping"]
    if any(not isinstance(protocol.get(key), Mapping) for key in ("workload", "candidates", "gates", "outputs")):
        return ["workload, candidates, gates and outputs must be mappings"]
    if protocol.get("schema_version") != "operator-optimization-delivery-config/v1":
        errors.append("unexpected schema_version")
    if protocol.get("semantic_id") != SEMANTIC_ID:
        errors.append("unexpected semantic_id")
    if protocol.get("environment_profile") != "PERF-CUDA-MECHANISM":
        errors.append("environment_profile must be PERF-CUDA-MECHANISM")
    workload = protocol.get("workload", {})
    if workload.get("sequence_lengths") != list(SEQUENCE_LENGTHS):
        errors.append("sequence_lengths must be [128, 512, 1024, 2048]")
    if workload.get("dtype") != "bfloat16" or workload.get("causal") is not True:
        errors.append("workload must use causal bfloat16 attention")
    if workload.get("batch_size") != 1 or workload.get("layers") != 2:
        errors.append("workload must use batch_size=1 and layers=2")
    candidates = protocol.get("candidates", {})
    if candidates.get("baseline") != VARIANTS[0] or candidates.get("candidate") != VARIANTS[1]:
        errors.append("candidate mapping does not match the registered implementations")
    gates = protocol.get("gates", {})
    if gates.get("require_kernel_trace") is not True:
        errors.append("require_kernel_trace must be true")
    for key in ("batch_size", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "layers", "repeats"):
        if type(workload.get(key)) is not int or workload[key] <= 0:
            errors.append(f"workload.{key} must be a positive integer")
    for key in ("seed", "warmup"):
        if type(workload.get(key)) is not int or workload[key] < 0:
            errors.append(f"workload.{key} must be a nonnegative integer")
    if not errors:
        if (workload["hidden_size"] != workload["num_attention_heads"] * workload["head_dim"]
                or workload["num_key_value_heads"] != workload["num_attention_heads"]
                or workload["head_dim"] % 2 or workload["head_dim"] > 128):
            errors.append("attention dimensions must match dense MHA and even head_dim <= 128")
        if workload["repeats"] < 2 or workload["repeats"] % 2:
            errors.append("repeats must be even and >= 2 for balanced paired order")
    if workload.get("model") != "synthetic_two_layer_decoder":
        errors.append("model must be synthetic_two_layer_decoder")
    for key in ("max_relative_error_rtol", "max_absolute_error_atol", "minimum_speedup_long_sequence", "maximum_short_sequence_regression", "maximum_peak_memory_gib"):
        value = _number(gates.get(key))
        if value is None or value < 0 or (key == "maximum_peak_memory_gib" and value == 0):
            errors.append(f"gates.{key} must be a finite nonnegative number (memory > 0)")
    if gates.get("long_sequence_lengths") != [1024, 2048] or gates.get("short_sequence_lengths") != [128, 512]:
        errors.append("long/short sequence gates must cover the registered shape matrix")
    if gates.get("max_absolute_error_atol") == 0:
        errors.append("max_absolute_error_atol must be positive for scaled-error checking")
    if protocol["outputs"].get("directory") != RESULT_ROOT:
        errors.append(f"outputs.directory must be {RESULT_ROOT}")
    return errors


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _validate_label(label: str) -> None:
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", label) or label == "replace_with_unique_label":
        raise ValueError("请填写 3–64 位唯一 RUN_LABEL（字母、数字、点、下划线或连字符）。")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_cuda(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("本项目需要 CUDA GPU。")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16。")
    properties = torch.cuda.get_device_properties(0)
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < 2 * 1024**3:
        raise RuntimeError("GPU 可用显存不足 2 GiB；请先关闭其他 GPU 任务。")
    return properties


def preflight(torch: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    issues = validate_protocol(protocol)
    if issues:
        raise ValueError("invalid operator delivery protocol: " + "; ".join(issues))
    properties = _require_cuda(torch)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "status": "ok",
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "free_memory_gib": round(free_bytes / 1024**3, 3),
        "total_memory_gib": round(total_bytes / 1024**3, 3),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _apply_rope(torch: Any, value: Any) -> Any:
    batch, heads, sequence, dimension = value.shape
    positions = torch.arange(sequence, device=value.device, dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=value.device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=value.device)) / dimension)
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos().to(value.dtype)[None, None, :, :]
    sine = angles.sin().to(value.dtype)[None, None, :, :]
    even, odd = value[..., 0::2], value[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2)


def _build_weights(torch: Any, hidden: int, heads: int, layers: int, seed: int, device: Any) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = []
    for _ in range(layers):
        def matrix(rows: int, columns: int) -> Any:
            return (torch.randn((rows, columns), generator=generator, dtype=torch.float32) * (columns ** -0.5)).to(device=device, dtype=torch.bfloat16)
        weights.append({
            "qkv": matrix(hidden, 3 * hidden),
            "out": matrix(hidden, hidden),
            "up": matrix(hidden, 4 * hidden),
            "down": matrix(4 * hidden, hidden),
            "norm1": torch.ones(hidden, device=device, dtype=torch.bfloat16),
            "norm2": torch.ones(hidden, device=device, dtype=torch.bfloat16),
        })
    return weights


def _decoder_prefill(torch: Any, source: Any, weights: list[dict[str, Any]], *, heads: int, variant: str) -> Any:
    from tools.triton_attention_runtime import triton_flash_attention

    hidden = source.shape[-1]
    head_dim = hidden // heads
    value = source
    for layer in weights:
        normalized = torch.nn.functional.layer_norm(value, (hidden,), layer["norm1"])
        qkv = normalized @ layer["qkv"]
        q, k, v = qkv.reshape(1, value.shape[1], 3, heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = _apply_rope(torch, q.contiguous()), _apply_rope(torch, k.contiguous())
        if variant == "torch_sdpa_auto":
            attended = torch.nn.functional.scaled_dot_product_attention(q, k, v.contiguous(), is_causal=True)
        elif variant == "triton_flash_attention":
            attended = triton_flash_attention(q, k, v.contiguous(), is_causal=True)
        else:
            raise ValueError(f"unknown variant: {variant}")
        value = value + attended.transpose(1, 2).reshape(1, value.shape[1], hidden) @ layer["out"]
        normalized = torch.nn.functional.layer_norm(value, (hidden,), layer["norm2"])
        value = value + torch.nn.functional.gelu(normalized @ layer["up"]) @ layer["down"]
    return value


def _profile_once(torch: Any, source: Any, weights: list[dict[str, Any]], *, heads: int, variant: str, trace_path: Path) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        _decoder_prefill(torch, source, weights, heads=heads, variant=variant)
        torch.cuda.synchronize(source.device)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    names = [event.key for event in profiler.key_averages()]
    sdpa_dispatch = sorted(name for name in names if "scaled_dot_product" in name or "flash" in name.lower())
    triton_observed = any("flash_attn_fwd_kernel" in name for name in names)
    if variant == "torch_sdpa_auto" and not any(marker in name for marker in SDPA_BACKENDS for name in sdpa_dispatch):
        raise RuntimeError("profiler 未能识别 PyTorch SDPA 的实际 dispatch 路径")
    if variant == "triton_flash_attention" and not triton_observed:
        raise RuntimeError("profiler 未观察到 Triton FlashAttention kernel")
    return {"sdpa_dispatch_events": sdpa_dispatch, "triton_kernel_observed": triton_observed,
            "trace_path": str(trace_path), "trace_sha256": _sha256(trace_path)}


def _measure_pair(torch: Any, source: Any, weights: list[dict[str, Any]], *, heads: int, warmup: int, repeats: int, trace_dir: Path, sequence: int) -> tuple[dict[str, Any], dict[str, Any]]:
    durations: dict[str, list[float]] = {name: [] for name in VARIANTS}
    peaks: dict[str, list[float]] = {name: [] for name in VARIANTS}
    for variant in VARIANTS:
        for _ in range(warmup):
            _decoder_prefill(torch, source, weights, heads=heads, variant=variant)
    torch.cuda.synchronize(source.device)
    for repeat_index in range(repeats):
        order = VARIANTS if repeat_index % 2 == 0 else tuple(reversed(VARIANTS))
        for variant in order:
            torch.cuda.synchronize(source.device)
            torch.cuda.reset_peak_memory_stats(source.device)
            start = time.perf_counter()
            output = _decoder_prefill(torch, source, weights, heads=heads, variant=variant)
            torch.cuda.synchronize(source.device)
            durations[variant].append((time.perf_counter() - start) * 1000)
            peaks[variant].append(torch.cuda.max_memory_allocated(source.device) / 1024**2)
            del output
    results = []
    for variant in VARIANTS:
        profile = _profile_once(
            torch, source, weights, heads=heads, variant=variant,
            trace_path=trace_dir / f"{variant}_s{sequence}.json",
        )
        results.append({
            "median_latency_ms": statistics.median(durations[variant]),
            "repeat_latency_ms": durations[variant],
            "repeat_execution_order": [list(VARIANTS if index % 2 == 0 else reversed(VARIANTS)) for index in range(repeats)],
            "peak_allocated_mb": round(max(peaks[variant]), 3),
            "profile": profile,
            "output": _decoder_prefill(torch, source, weights, heads=heads, variant=variant),
        })
    return results[0], results[1]


def _valid_row_metrics(row: Mapping[str, Any], repeats: int) -> bool:
    """Verify paired samples and recompute summaries rather than trusting ratios."""
    samples = []
    expected_order = [list(VARIANTS if index % 2 == 0 else reversed(VARIANTS)) for index in range(repeats)]
    for name in VARIANTS:
        result = row.get(name)
        if not isinstance(result, Mapping):
            return False
        values = result.get("repeat_latency_ms")
        if not isinstance(values, list) or len(values) != repeats or any(_number(value) is None or value <= 0 for value in values):
            return False
        median = _number(result.get("median_latency_ms"))
        peak = _number(result.get("peak_allocated_mb"))
        if (median is None or not math.isclose(median, statistics.median(values), rel_tol=1e-6)
                or peak is None or peak <= 0 or result.get("repeat_execution_order") != expected_order):
            return False
        samples.append(values)
    speedup = _number(row.get("speedup_ratio"))
    expected_speedup = statistics.median(left / right for left, right in zip(*samples))
    if speedup is None or speedup <= 0 or not math.isclose(speedup, expected_speedup, rel_tol=1e-6):
        return False
    return all(_number(row.get(key)) is not None and row[key] >= 0 for key in (
        "max_absolute_error", "max_relative_error", "max_scaled_error",
    ))


def _trace_claims_valid(row: Mapping[str, Any]) -> bool:
    profiles = [row[name].get("profile") for name in VARIANTS]
    if any(not isinstance(profile, Mapping) for profile in profiles):
        return False
    for profile in profiles:
        if not isinstance(profile.get("trace_path"), str) or not profile["trace_path"]:
            return False
        digest = profile.get("trace_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False
    events = profiles[0].get("sdpa_dispatch_events")
    return (isinstance(events, list) and bool(events) and all(isinstance(event, str) and event for event in events)
            and any(marker in name for marker in SDPA_BACKENDS for name in events)
            and profiles[1].get("triton_kernel_observed") is True)


def evaluate_measurements(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered numerical, speed, memory, and trace gates."""
    issues = validate_protocol(protocol)
    if issues:
        raise ValueError("invalid operator delivery protocol: " + "; ".join(issues))
    if (record.get("schema_version") != "part05-operator-delivery/v1" or record.get("semantic_id") != SEMANTIC_ID
            or record.get("protocol") != dict(protocol) or record.get("status") != "complete"):
        raise ValueError("测量记录的项目身份或协议与当前配置不一致。")
    _validate_label(record.get("run_label"))
    rows = record.get("measurements", [])
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) or type(row.get("sequence_length")) is not int for row in rows):
        rows = []
    expected_lengths = protocol["workload"]["sequence_lengths"]
    by_length = {row.get("sequence_length"): row for row in rows}
    complete = (len(rows) == len(by_length) == len(expected_lengths) and set(by_length) == set(expected_lengths)
                and all(_valid_row_metrics(row, protocol["workload"]["repeats"]) for row in rows))
    gates = protocol["gates"]
    numerical_pass = complete and all(
        row.get("numerical_close") is True and row["max_scaled_error"] <= 1
        for row in rows
    )
    traces_pass = complete and all(_trace_claims_valid(row) for row in rows)
    long_lengths = gates["long_sequence_lengths"]
    long_pass = complete and all(by_length[length]["speedup_ratio"] >= 1 + gates["minimum_speedup_long_sequence"] for length in long_lengths)
    short_lengths = gates["short_sequence_lengths"]
    short_pass = complete and all(
        1 / by_length[length]["speedup_ratio"] - 1 <= gates["maximum_short_sequence_regression"]
        for length in short_lengths
    )
    memory_limit_mb = gates["maximum_peak_memory_gib"] * 1024
    memory_pass = complete and all(
        max(row["torch_sdpa_auto"]["peak_allocated_mb"], row["triton_flash_attention"]["peak_allocated_mb"]) <= memory_limit_mb
        for row in rows
    )
    if not complete or not traces_pass or not numerical_pass:
        decision = "reject" if complete else "pending"
    elif long_pass and short_pass and memory_pass:
        decision = "accept"
    else:
        decision = "tune"
    return {
        "schema_version": "part05-operator-delivery-decision/v1",
        "semantic_id": SEMANTIC_ID,
        "run_label": record.get("run_label"),
        "decision": decision,
        "gates": {
            "measurement_complete": complete,
            "numerical_correctness": numerical_pass,
            "both_execution_paths_traced": traces_pass,
            "long_sequence_speedup": long_pass,
            "short_sequence_regression": short_pass,
            "memory_budget": memory_pass,
        },
        "result_path": record.get("result_path"),
        "next_action": {
            "accept": "保留固定配置和两条 profiler 轨迹作为交付证据。",
            "tune": "保持数值与路径约束，调整候选实现或适用序列范围后另用新标签复测。",
            "reject": "修复数值错误、执行路径或证据记录后再测。",
            "pending": "补齐未完成的固定形状测量。",
        }[decision],
    }


def run_operator_delivery(torch: Any, protocol: Mapping[str, Any], project_root: str | Path, run_label: str) -> dict[str, Any]:
    """Run paired two-layer prefill measurements and save immutable evidence."""
    issues = validate_protocol(protocol)
    if issues:
        raise ValueError("invalid operator delivery protocol: " + "; ".join(issues))
    _validate_label(run_label)
    properties = _require_cuda(torch)
    root = Path(project_root).resolve()
    output_dir = root / protocol["outputs"]["directory"] / run_label
    output_dir.mkdir(parents=True, exist_ok=False)
    workload = protocol["workload"]
    hidden, heads, layers = workload["hidden_size"], workload["num_attention_heads"], workload["layers"]
    device = torch.device("cuda:0")
    rows = []
    for sequence in workload["sequence_lengths"]:
        generator = torch.Generator(device="cpu").manual_seed(workload["seed"] + sequence)
        source = torch.randn((1, sequence, hidden), generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
        weights = _build_weights(torch, hidden, heads, layers, workload["seed"], device)
        torch_result, triton_result = _measure_pair(
            torch, source, weights, heads=heads,
            warmup=workload["warmup"], repeats=workload["repeats"],
            trace_dir=output_dir / "traces", sequence=sequence,
        )
        torch_output, triton_output = torch_result.pop("output"), triton_result.pop("output")
        for result in (torch_result, triton_result):
            result["profile"]["trace_path"] = str(Path(result["profile"]["trace_path"]).relative_to(root))
        if not bool(torch.isfinite(torch_output).all().item()) or not bool(torch.isfinite(triton_output).all().item()):
            raise RuntimeError("两层预填充输出包含 NaN/Inf，保留轨迹并停止本轮测量。")
        difference = (torch_output.float() - triton_output.float()).abs()
        denominator = torch_output.float().abs().clamp_min(1e-6)
        max_relative_error = float((difference / denominator).max().item())
        max_absolute_error = float(difference.max().item())
        scaled_error = difference / (protocol["gates"]["max_absolute_error_atol"] + protocol["gates"]["max_relative_error_rtol"] * torch_output.float().abs())
        numerical_close = bool(torch.allclose(
            torch_output.float(), triton_output.float(),
            rtol=protocol["gates"]["max_relative_error_rtol"],
            atol=protocol["gates"]["max_absolute_error_atol"],
        ))
        rows.append({
            "sequence_length": sequence,
            "torch_sdpa_auto": torch_result,
            "triton_flash_attention": triton_result,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
            "max_scaled_error": float(scaled_error.max().item()),
            "numerical_close": numerical_close,
            "speedup_ratio": statistics.median(left / right for left, right in zip(torch_result["repeat_latency_ms"], triton_result["repeat_latency_ms"])),
        })
        del source, weights, torch_output, triton_output, difference, denominator, scaled_error
        torch.cuda.empty_cache()
    record = {
        "schema_version": "part05-operator-delivery/v1",
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "status": "complete",
        "device": properties.name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "protocol": dict(protocol),
        "measurements": rows,
    }
    result_path = output_dir / "raw_result.json"
    record["result_path"] = str(result_path.relative_to(root))
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return record


def close_operator_delivery(project_root: str | Path, run_label: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Read measured evidence and write one immutable decision record."""
    issues = validate_protocol(protocol)
    if issues:
        raise ValueError("invalid operator delivery protocol: " + "; ".join(issues))
    _validate_label(run_label)
    root = Path(project_root).resolve()
    run_dir = root / protocol["outputs"]["directory"] / run_label
    result_path = run_dir / "raw_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"先完成 RUN_STAGE='measure'：{result_path}")
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if record.get("run_label") != run_label or record.get("semantic_id") != SEMANTIC_ID:
        raise ValueError("测量记录的项目身份或 RUN_LABEL 不匹配")
    decision = evaluate_measurements(record, protocol)
    decision["result_path"] = str(result_path.relative_to(root))
    if decision["gates"]["measurement_complete"]:
        for row in record["measurements"]:
            for name in VARIANTS:
                profile = row[name].get("profile")
                if not isinstance(profile, Mapping):
                    raise ValueError("性能轨迹记录缺失。")
                trace_name = profile.get("trace_path")
                if not isinstance(trace_name, str) or not trace_name:
                    raise ValueError("性能轨迹文件路径缺失。")
                trace = Path(trace_name)
                if not trace.is_absolute():
                    trace = root / trace
                if not trace.resolve().is_relative_to((run_dir / "traces").resolve()) or not trace.is_file() or _sha256(trace) != profile.get("trace_sha256"):
                    raise ValueError(f"性能轨迹缺失、超出本轮目录或 SHA256 不匹配：{trace}")
                payload = json.loads(trace.read_text(encoding="utf-8"))
                events = payload.get("traceEvents", [])
                names = [event.get("name", "") for event in events if isinstance(event, Mapping) and isinstance(event.get("name"), str)]
                observed = (any(marker in event for marker in SDPA_BACKENDS for event in names) if name == VARIANTS[0]
                            else any("flash_attn_fwd_kernel" in event for event in names))
                if not observed:
                    raise ValueError(f"轨迹没有记录本路径的目标执行事件：{trace}")
    decision["raw_result_sha256"] = _sha256(result_path)
    decision_path = run_dir / "decision.json"
    if decision_path.exists():
        existing = json.loads(decision_path.read_text(encoding="utf-8"))
        if existing != decision:
            raise FileExistsError(f"同一 RUN_LABEL 已有不同决策记录：{decision_path}")
    else:
        with decision_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    decision["decision_path"] = str(decision_path.relative_to(root))
    return decision
