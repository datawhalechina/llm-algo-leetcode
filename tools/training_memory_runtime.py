"""Shared measurement helpers for the real training-memory notebooks.

Model construction and strategy setup stay in the notebooks.  This module
only keeps timing and CUDA memory accounting consistent across projects 73,
76 and 74.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def synchronize_if_cuda(torch_module: Any) -> None:
    """Synchronize CUDA before and after a timed region when available."""

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def measure_training_run(
    train_step: Callable[[], float],
    *,
    torch_module: Any,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict[str, float | str]:
    """Measure complete optimizer steps and return common result fields."""

    if batch_size <= 0 or warmup < 0 or iters <= 0:
        raise ValueError("batch_size must be > 0, warmup >= 0 and iters > 0")

    for _ in range(warmup):
        train_step()
    synchronize_if_cuda(torch_module)
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    losses = [float(train_step()) for _ in range(iters)]
    synchronize_if_cuda(torch_module)
    elapsed = time.perf_counter() - start

    result: dict[str, float | str] = {
        "step_time_ms": round(elapsed * 1000 / iters, 3),
        "samples_per_s": round(batch_size * iters / elapsed, 3),
        "loss": round(losses[-1], 6),
    }
    if torch_module.cuda.is_available():
        result.update(
            {
                "peak_mem_mb": round(torch_module.cuda.max_memory_allocated() / (1024**2), 2),
                "peak_reserved_mb": round(torch_module.cuda.max_memory_reserved() / (1024**2), 2),
            }
        )
    return result
