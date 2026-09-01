"""Shared runtime helpers for fine-tuning projects 60--65.

The helpers only validate configuration and persist reports. They do not
start training, change hyperparameters, or invent measurements.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

RUN_MODES = {"cpu", "dry_run", "real_gpu"}


def locate_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "tools").is_dir() and (candidate / "02_PyTorch_Algorithms").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("未找到项目根目录，请从仓库启动 Notebook 或先 clone 仓库。")


def ensure_output_path(path: str | Path, root: str | Path | None = None) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = locate_repo_root(root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def validate_project_config(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("project", "model", "dtype"):
        if not str(config.get(key, "")).strip():
            errors.append(f"missing config: {key}")
    for key in ("batch_size", "seq_len", "steps", "seed"):
        if key in config and int(config[key]) <= 0:
            errors.append(f"invalid config: {key}")
    if "val_ratio" in config and not 0.0 < float(config["val_ratio"]) < 1.0:
        errors.append("val_ratio must be between 0 and 1")
    run_mode = config.get("run_mode", "cpu")
    if run_mode not in RUN_MODES:
        errors.append(f"run_mode must be one of {sorted(RUN_MODES)}")
    return errors


def runtime_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    if torch_module is None:
        return snapshot
    snapshot["torch"] = getattr(torch_module, "__version__", None)
    snapshot["torch_cuda"] = getattr(getattr(torch_module, "version", None), "cuda", None)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        snapshot["device"] = cuda.get_device_name(0)
        snapshot["device_capability"] = list(cuda.get_device_capability(0))
        # `is_bf16_supported()` may report emulation support on older GPUs.
        # Keep both signals so a report does not confuse allocation support
        # with native BF16 acceleration.
        try:
            snapshot["bf16_supported_including_emulation"] = bool(cuda.is_bf16_supported())
            snapshot["bf16_native_supported"] = bool(
                cuda.is_bf16_supported(including_emulation=False)
            )
        except TypeError:
            snapshot["bf16_supported_including_emulation"] = bool(cuda.is_bf16_supported())
            snapshot["bf16_native_supported"] = None
    else:
        snapshot["device"] = "cpu"
        snapshot["device_capability"] = None
        snapshot["bf16_supported_including_emulation"] = False
        snapshot["bf16_native_supported"] = False
    return snapshot


def preflight_runtime(
    torch_module: Any | None,
    run_mode: str = "cpu",
    *,
    min_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Check execution prerequisites without loading a model or training.

    ``cpu`` never requires CUDA. ``dry_run`` reports GPU readiness when CUDA
    exists but does not fail merely because the current machine has no GPU.
    ``real_gpu`` requires CUDA and can enforce a minimum device capacity.
    """

    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {sorted(RUN_MODES)}")

    snapshot = runtime_snapshot(torch_module)
    cuda_available = bool(snapshot.get("cuda_available", False))
    total_memory_gb = None
    if cuda_available and torch_module is not None:
        total_memory_gb = round(
            torch_module.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )

    ready = True
    reasons: list[str] = []
    if run_mode == "real_gpu" and not cuda_available:
        ready = False
        reasons.append("CUDA is unavailable")
    if min_memory_gb is not None and total_memory_gb is not None and total_memory_gb < min_memory_gb:
        ready = False
        reasons.append(f"GPU memory {total_memory_gb} GB < required {min_memory_gb} GB")

    return {
        "run_mode": run_mode,
        "ready": ready,
        "reasons": reasons,
        "runtime": snapshot,
        "gpu_memory_gb": total_memory_gb,
        "next_action": "run_experiment" if ready and run_mode == "real_gpu" else (
            "enable_gpu_or_use_cpu" if not ready else "review_preflight_then_choose_mode"
        ),
    }


def save_project_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> Path:
    """Normalize, validate and save one 60--65 project report."""

    from fine_tuning_result_schema import normalize_and_validate

    normalized = normalize_and_validate(report)
    output = ensure_output_path(path, root)
    output.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(normalized, ensure_ascii=False, indent=2))
    return output
