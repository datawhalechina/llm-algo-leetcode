"""Shared profiler collection for the real GPU optimization project."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


def collect_training_trace(
    train_step: Callable[[], Any],
    *,
    torch_module: Any,
    output_dir: str | Path,
    warmup: int,
    iters: int,
    batch_size: int,
) -> dict[str, Any]:
    """Profile fixed optimizer steps and return comparable trace metadata."""

    if warmup < 0 or iters <= 0 or batch_size <= 0:
        raise ValueError("warmup must be >= 0, iters and batch_size must be > 0")
    if not torch_module.cuda.is_available():
        raise RuntimeError("collect_training_trace requires CUDA")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        train_step()
    torch_module.cuda.synchronize()
    torch_module.cuda.reset_peak_memory_stats()

    activities = [
        torch_module.profiler.ProfilerActivity.CPU,
        torch_module.profiler.ProfilerActivity.CUDA,
    ]
    # iters=1 时不能先 warmup 再 active，否则循环结束前不会产出 trace。
    schedule = torch_module.profiler.schedule(
        wait=0, warmup=1 if iters > 1 else 0,
        active=max(1, iters - 1) if iters > 1 else 1,
        repeat=1,
    )
    handler = torch_module.profiler.tensorboard_trace_handler(str(output_path))
    losses: list[float] = []
    with torch_module.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        start = time.perf_counter()
        for _ in range(iters):
            value = train_step()
            losses.append(float(value.detach().item() if hasattr(value, "detach") else value))
            profiler.step()
        torch_module.cuda.synchronize()
        elapsed = time.perf_counter() - start
        top_operators = profiler.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        )

    return {
        "step_time_ms": round(elapsed * 1000 / iters, 3),
        "samples_per_s": round(batch_size * iters / elapsed, 3),
        "loss": round(losses[-1], 6),
        "peak_memory_mb": round(torch_module.cuda.max_memory_allocated() / (1024**2), 2),
        "peak_reserved_mb": round(torch_module.cuda.max_memory_reserved() / (1024**2), 2),
        "top_operators": top_operators,
        "profile": {
            "tool": "torch.profiler",
            "activities": [activity.name for activity in activities],
            "trace_dir": str(output_path),
            "warmup": warmup,
            "iters": iters,
            "status": "collected",
        },
    }
