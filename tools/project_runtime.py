"""Low-risk runtime helpers shared by real training project notebooks.

These helpers validate configuration and record environment metadata. They do
not choose hyperparameters, retry OOMs, or change an experiment's workload.
"""

from __future__ import annotations

import os
import platform
import sys
import importlib.util
from pathlib import Path
from typing import Any, Mapping


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Find an existing checkout from the current directory or an override."""

    override = os.environ.get("LLM_ALGO_PROJECT_ROOT")
    current = Path(override or start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "benchmarks").is_dir() and (candidate / "02_PyTorch_Algorithms").is_dir():
            return candidate
    return current


def bootstrap_project_root(start: str | Path | None = None) -> Path:
    """Locate the checkout and make ``tools`` importable in a notebook.

    A notebook opened directly from ``docs`` or Colab does not necessarily
    inherit the repository directory as its current working directory.  This
    helper changes only ``sys.path``; it does not change the working directory
    or install packages.
    """

    roots = []
    override = os.environ.get("LLM_ALGO_PROJECT_ROOT")
    if override:
        roots.append(Path(override).expanduser())
    if start is not None:
        roots.append(Path(start).expanduser())
    roots.extend((Path.cwd(), Path("/content/llm-algo-leetcode")))
    for item in roots:
        current = item.resolve()
        for candidate in (current, *current.parents):
            if (candidate / "tools").is_dir() and (candidate / "02_PyTorch_Algorithms").is_dir():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                return candidate
    raise RuntimeError("未找到项目根目录，请先 clone 仓库，或设置 LLM_ALGO_PROJECT_ROOT。")


def ensure_output_path(root: str | Path, relative_path: str | Path) -> Path:
    """Create the parent directory and return an absolute result path."""

    output = Path(relative_path)
    if not output.is_absolute():
        output = Path(root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Fail early on invalid shared training benchmark parameters."""

    positive = ("batch_size", "seq_len", "warmup", "iters")
    for key in positive:
        value = config.get(key)
        if not isinstance(value, int) or value < 0 or (key in {"batch_size", "seq_len", "iters"} and value == 0):
            raise ValueError(f"{key} 必须是有效的非负整数，当前为 {value!r}")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed 必须是整数")
    if config.get("learning_rate") is not None and float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate 必须大于 0")


def runtime_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    """Return reproducibility metadata without requiring CUDA."""

    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    if torch_module is None:
        return snapshot
    snapshot["torch"] = getattr(torch_module, "__version__", None)
    snapshot["torch_cuda"] = getattr(torch_module.version, "cuda", None)
    available = bool(torch_module.cuda.is_available())
    snapshot["cuda_available"] = available
    if available:
        snapshot["device"] = torch_module.cuda.get_device_name(0)
        snapshot["device_capability"] = list(torch_module.cuda.get_device_capability(0))
    return snapshot


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def environment_preflight(
    torch_module: Any | None = None,
    *,
    required_packages: tuple[str, ...] = (),
    optional_packages: tuple[str, ...] = (),
    require_gpu: bool = False,
    min_gpu_memory_gb: float | None = None,
    require_native_bf16: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe, actionable preflight report.

    This is deliberately a check, not an installer.  Missing packages are
    reported with an install hint; CUDA/PyTorch replacements remain an
    explicit user decision because reinstalling them can invalidate a cloud
    runtime or turn a CUDA wheel into a CPU-only environment.
    """

    runtime = runtime_snapshot(torch_module)
    cuda_available = bool(runtime.get("cuda_available", False))
    packages = {
        "required": {name: _module_available(name) for name in required_packages},
        "optional": {name: _module_available(name) for name in optional_packages},
    }
    reasons: list[str] = []
    actions: list[str] = []
    missing_required = [name for name, present in packages["required"].items() if not present]
    if missing_required:
        reasons.append("缺少必需 Python 包：" + ", ".join(missing_required))
        actions.append("使用当前内核的 sys.executable 安装缺少的普通 Python 包，然后重启内核")

    gpu_memory_gb = None
    native_bf16 = None
    if cuda_available and torch_module is not None:
        props = torch_module.cuda.get_device_properties(0)
        gpu_memory_gb = round(props.total_memory / (1024**3), 2)
        try:
            native_bf16 = bool(torch_module.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            native_bf16 = None

    if require_gpu and not cuda_available:
        reasons.append("当前 PyTorch 不可使用 CUDA；可能安装了 CPU 版 torch，或 Colab 尚未启用 GPU")
        actions.append("先在运行时设置中选择 GPU，并确认 torch.version.cuda 非 None；不要直接覆盖当前 torch")
    if min_gpu_memory_gb is not None and gpu_memory_gb is not None and gpu_memory_gb < min_gpu_memory_gb:
        reasons.append(f"GPU 总显存 {gpu_memory_gb} GB 小于要求 {min_gpu_memory_gb} GB")
        actions.append("降低 workload，或换用显存更大的 GPU")
    if require_native_bf16 and native_bf16 is not True:
        reasons.append("当前 GPU 没有确认原生 BF16 加速；不能把 including_emulation=True 当作硬件加速")
        actions.append("改用 FP16，或明确记录为 BF16 容量探测而非 BF16 加速实验")

    output = None
    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            probe = output.parent / ".preflight_write_probe"
            probe.touch(exist_ok=False)
            probe.unlink()
        except FileExistsError:
            probe.unlink(missing_ok=True)
        except OSError as exc:
            reasons.append(f"结果目录不可写：{output.parent}（{exc}）")
            actions.append("检查项目根目录和 benchmarks/results 的权限")

    status = "blocked" if reasons and (require_gpu or missing_required or require_native_bf16) else ("warning" if reasons else "ok")
    return {
        "status": status,
        "ready": status != "blocked",
        "runtime": runtime,
        "gpu_memory_gb": gpu_memory_gb,
        "native_bf16_supported": native_bf16,
        "packages": packages,
        "required_packages_missing": missing_required,
        "reasons": reasons,
        "next_actions": actions,
        "result_path": str(output) if output else None,
    }


def require_input_file(path: str | Path, label: str) -> Path:
    """Validate a required upstream report before a downstream project runs."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} 不存在：{input_path}。请先完成上游项目并保存 JSON。")
    if input_path.suffix.lower() != ".json":
        raise ValueError(f"{label} 必须是 JSON 文件：{input_path}")
    return input_path


def standard_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared experiment fields without removing legacy fields."""

    return {
        "model": config.get("model", config.get("model_id")),
        "backend": config.get("backend", "torch"),
        "dtype": config.get("dtype", config.get("amp_dtype")),
        "optimizer": config.get("optimizer"),
        "batch_size": config.get("batch_size"),
        "seq_len": config.get("seq_len"),
        "warmup": config.get("warmup"),
        "iters": config.get("iters"),
        "seed": config.get("seed"),
        "device": config.get("device"),
        "torch": config.get("torch"),
        "cuda": config.get("cuda", config.get("torch_cuda")),
        "workload": config.get("workload"),
    }


def standard_training_metrics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Map legacy candidate names to the common training result columns."""

    return {
        "step_time_ms": candidate.get("step_time_ms"),
        "throughput_samples_per_s": candidate.get(
            "throughput_samples_per_s", candidate.get("samples_per_s")
        ),
        "peak_memory_mb": candidate.get(
            "peak_memory_mb", candidate.get("peak_mem_mb")
        ),
        "peak_reserved_mb": candidate.get("peak_reserved_mb"),
        "loss": candidate.get("loss"),
        "eval_loss": candidate.get("eval_loss", candidate.get("val_loss")),
        "status": candidate.get("status", "ok"),
    }
