"""Single-device AdamW execution-path benchmark used by Part 02 · 75."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any


OPTIMIZER_PATHS: dict[str, dict[str, Any]] = {
    "single_tensor": {"foreach": False, "fused": False},
    "foreach": {"foreach": True, "fused": False},
    "fused": {"foreach": None, "fused": True},
}

WORKLOADS: dict[str, dict[str, int]] = {
    "many_small": {"tensor_count": 256, "elements_per_tensor": 65_536},
    "few_large": {"tensor_count": 4, "elements_per_tensor": 4_194_304},
}


def validate_optimizer_candidates(
    candidates: list[dict[str, Any]], *, learning_rate: float, weight_decay: float
) -> dict[str, Any]:
    """Check that variants change only the AdamW execution path."""
    names = [item.get("name") for item in candidates]
    expected = set(OPTIMIZER_PATHS)
    errors: list[str] = []
    if len(names) != len(set(names)):
        errors.append("duplicate_candidate")
    if len(names) != len(expected) or set(names) != expected:
        errors.append("candidate_set_mismatch")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        errors.append("invalid_learning_rate")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        errors.append("invalid_weight_decay")
    for item in candidates:
        name = item.get("name")
        if name not in OPTIMIZER_PATHS:
            continue
        if item.get("optimizer") != "AdamW":
            errors.append(f"{name}:optimizer_changed")
        if item.get("scheduler") is not None:
            errors.append(f"{name}:scheduler_must_be_fixed")
        if item.get("learning_rate") != learning_rate:
            errors.append(f"{name}:learning_rate_mismatch")
        if item.get("weight_decay") != weight_decay:
            errors.append(f"{name}:weight_decay_mismatch")
        if item.get("path") != OPTIMIZER_PATHS[name]:
            errors.append(f"{name}:execution_path_mismatch")
    return {"valid": not errors, "errors": errors}


def decide_optimizer_path(
    records: list[dict[str, Any]],
    *,
    minimum_training_speedup: float = 1.05,
    maximum_peak_memory_increase_mb: float = 256.0,
    maximum_parameter_error: float = 1e-4,
) -> dict[str, Any]:
    """Select only numerically comparable paths with an end-to-end step gain."""
    required = {
        "name",
        "status",
        "workload_id",
        "seed",
        "training_step_ms",
        "optimizer_step_ms",
        "peak_allocated_mb",
        "max_parameter_error",
    }
    errors: list[str] = []
    if not records:
        return {"decision": "pending", "reason": "no_measurements", "candidates": []}
    if any(not required.issubset(record) for record in records):
        errors.append("missing_required_measurement")
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)):
        errors.append("duplicate_candidate")
    if set(names) != set(OPTIMIZER_PATHS):
        errors.append("candidate_set_mismatch")
    baseline = next((record for record in records if record.get("name") == "single_tensor"), None)
    if baseline is None:
        errors.append("baseline_missing")
    numeric_fields = (
        "training_step_ms",
        "optimizer_step_ms",
        "peak_allocated_mb",
        "max_parameter_error",
    )
    for record in records:
        try:
            values = [float(record[field]) for field in numeric_fields]
        except (TypeError, ValueError, KeyError):
            errors.append(f"{record.get('name', 'candidate')}:invalid_numeric_evidence")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{record.get('name', 'candidate')}:non_finite_evidence")
        elif any(value < 0 for value in values) or values[0] == 0 or values[1] == 0:
            errors.append(f"{record.get('name', 'candidate')}:invalid_measurement_range")
    if errors:
        return {"decision": "pending", "reason": "invalid_measurements", "errors": errors, "candidates": []}

    identity_fields = ("workload_id", "seed")
    identity = tuple(baseline[field] for field in identity_fields)
    if any(tuple(record[field] for field in identity_fields) != identity for record in records):
        return {"decision": "pending", "reason": "incomparable_workloads", "candidates": []}
    if baseline["status"] != "ok" or baseline["training_step_ms"] <= 0:
        return {"decision": "pending", "reason": "baseline_unusable", "candidates": []}

    evaluated = []
    for record in records:
        if record["name"] == "single_tensor":
            continue
        if record["status"] != "ok" or record["training_step_ms"] <= 0:
            evaluated.append({"name": record["name"], "eligible": False, "reason": "run_failed"})
            continue
        speedup = baseline["training_step_ms"] / record["training_step_ms"]
        memory_increase = record["peak_allocated_mb"] - baseline["peak_allocated_mb"]
        numeric_error = record["max_parameter_error"]
        eligible = (
            speedup >= minimum_training_speedup
            and memory_increase <= maximum_peak_memory_increase_mb
            and numeric_error <= maximum_parameter_error
        )
        evaluated.append(
            {
                "name": record["name"],
                "eligible": eligible,
                "training_speedup": round(speedup, 4),
                "optimizer_step_speedup": round(
                    baseline["optimizer_step_ms"] / record["optimizer_step_ms"], 4
                ) if record["optimizer_step_ms"] > 0 else None,
                "peak_memory_increase_mb": round(memory_increase, 3),
                "max_parameter_error": numeric_error,
                "reason": "meets_gates" if eligible else "does_not_meet_all_gates",
            }
        )
    selected = [item for item in evaluated if item["eligible"]]
    selected.sort(key=lambda item: (-item["training_speedup"], item["peak_memory_increase_mb"]))
    decision = "accept" if selected else "tune"
    return {
        "decision": decision,
        "reason": "end_to_end_gain_and_guardrails_passed" if selected else "no_path_meets_adoption_gates",
        "selected": selected[0]["name"] if selected else None,
        "candidates": evaluated,
        "thresholds": {
            "minimum_training_speedup": minimum_training_speedup,
            "maximum_peak_memory_increase_mb": maximum_peak_memory_increase_mb,
            "maximum_parameter_error": maximum_parameter_error,
        },
    }


def _make_parameter_tensors(torch: Any, device: Any, workload_name: str, seed: int) -> list[Any]:
    spec = WORKLOADS[workload_name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.nn.Parameter(
            torch.randn(spec["elements_per_tensor"], generator=generator, dtype=torch.float32).to(device)
        )
        for _ in range(spec["tensor_count"])
    ]


def _make_model(torch: Any, device: Any, seed: int) -> Any:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
    )
    return model.to(device)


def _make_adamw(torch: Any, params: Any, name: str, learning_rate: float, weight_decay: float) -> Any:
    path = OPTIMIZER_PATHS[name]
    return torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=path["foreach"],
        fused=path["fused"],
    )


def _optimizer_only_run(
    torch: Any, name: str, workload_name: str, *, learning_rate: float,
    weight_decay: float, seed: int, warmup: int, iterations: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    params = _make_parameter_tensors(torch, device, workload_name, seed)
    grad_generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    for param in params:
        param.grad = torch.randn(param.shape, generator=grad_generator, dtype=param.dtype).to(device)
    optimizer = _make_adamw(torch, params, name, learning_rate, weight_decay)
    for _ in range(warmup):
        optimizer.step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    starts, ends = [], []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.step()
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize(device)
    elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    del optimizer, params
    torch.cuda.empty_cache()
    return {
        "optimizer_step_ms": statistics.median(elapsed),
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
    }


def _training_step_run(
    torch: Any, name: str, *, learning_rate: float, weight_decay: float,
    seed: int, warmup: int, iterations: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    model = _make_model(torch, device, seed)
    optimizer = _make_adamw(torch, model.parameters(), name, learning_rate, weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed + 200)
    inputs = torch.randn((16, 512), generator=generator, dtype=torch.float32).to(device)
    targets = torch.randn((16, 512), generator=generator, dtype=torch.float32).to(device)

    def train_step() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(warmup):
        train_step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    starts, ends = [], []
    final_loss = None
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        final_loss = train_step()
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize(device)
    elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    result = {
        "training_step_ms": statistics.median(elapsed),
        "final_loss": float(final_loss.item()),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "parameters": [param.detach().cpu().clone() for param in model.parameters()],
    }
    del optimizer, model, inputs, targets
    torch.cuda.empty_cache()
    return result


def run_optimizer_path_benchmark(
    *, learning_rate: float, weight_decay: float, seed: int,
    warmup: int = 5, iterations: int = 30, repeats: int = 3,
) -> dict[str, Any]:
    """Measure fixed AdamW paths on two equal-size layouts and a toy full step."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("optimizer path benchmark requires one CUDA GPU")
    if iterations < 5 or warmup < 1 or repeats < 3:
        raise ValueError("measurement requires warmup >= 1, iterations >= 5 and repeats >= 3")

    device_name = torch.cuda.get_device_name(0)
    result: dict[str, Any] = {
        "schema_version": "optimizer-path-benchmark/v1",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "scheduler": None,
        "optimizer": "AdamW",
        "paths": {},
    }

    for workload_name, spec in WORKLOADS.items():
        result["paths"][workload_name] = {}
        for name in OPTIMIZER_PATHS:
            runs = []
            for repeat_index in range(repeats):
                run = _optimizer_only_run(
                    torch, name, workload_name,
                    learning_rate=learning_rate, weight_decay=weight_decay,
                    seed=seed, warmup=warmup, iterations=iterations,
                )
                run["repeat"] = repeat_index + 1
                runs.append(run)
            result["paths"][workload_name][name] = {
                "status": "ok",
                "tensor_count": spec["tensor_count"],
                "total_elements": spec["tensor_count"] * spec["elements_per_tensor"],
                "optimizer_step_ms_median": statistics.median(row["optimizer_step_ms"] for row in runs),
                "optimizer_step_ms_range": max(row["optimizer_step_ms"] for row in runs) - min(row["optimizer_step_ms"] for row in runs),
                "peak_allocated_mb_median": statistics.median(row["peak_allocated_mb"] for row in runs),
                "peak_reserved_mb_median": statistics.median(row["peak_reserved_mb"] for row in runs),
                "runs": runs,
            }

    training_runs: dict[str, list[dict[str, Any]]] = {name: [] for name in OPTIMIZER_PATHS}
    candidate_names = list(OPTIMIZER_PATHS)
    for repeat_index in range(repeats):
        rotation = repeat_index % len(candidate_names)
        repeat_order = candidate_names[rotation:] + candidate_names[:rotation]
        repeat_rows: dict[str, dict[str, Any]] = {}
        for name in repeat_order:
            run = _training_step_run(
                torch, name, learning_rate=learning_rate, weight_decay=weight_decay,
                seed=seed, warmup=warmup, iterations=iterations,
            )
            run["repeat"] = repeat_index + 1
            repeat_rows[name] = run
        reference_parameters = repeat_rows["single_tensor"].pop("parameters")
        for name, row in repeat_rows.items():
            if name == "single_tensor":
                row["max_parameter_error_vs_single_tensor"] = 0.0
            else:
                parameters = row.pop("parameters")
                row["max_parameter_error_vs_single_tensor"] = max(
                    float((candidate - expected).abs().max().item())
                    for expected, candidate in zip(reference_parameters, parameters, strict=True)
                )
            training_runs[name].append(row)
    training_summary = {}
    for name, runs in training_runs.items():
        training_summary[name] = {
            "status": "ok",
            "training_step_ms_median": statistics.median(row["training_step_ms"] for row in runs),
            "training_step_ms_range": max(row["training_step_ms"] for row in runs) - min(row["training_step_ms"] for row in runs),
            "final_loss_median": statistics.median(row["final_loss"] for row in runs),
            "peak_allocated_mb_median": statistics.median(row["peak_allocated_mb"] for row in runs),
            "peak_reserved_mb_median": statistics.median(row["peak_reserved_mb"] for row in runs),
            "max_parameter_error_vs_single_tensor": max(
                row["max_parameter_error_vs_single_tensor"] for row in runs
            ),
            "runs": [{key: value for key, value in row.items() if key != "parameters"} for row in runs],
        }
    result["training_step"] = training_summary
    result["measurement_order"] = (
        "optimizer-only: fixed candidate order; training-step: candidate order rotates across repeats"
    )
    result["evidence_scope"] = "single_gpu_torch_adamw_execution_path"
    return result
