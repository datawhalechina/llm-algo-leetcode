"""Short-lived local Gloo worlds for Part 02 framework-conformance checks."""

from __future__ import annotations

import os
import tempfile
import traceback
from collections.abc import Callable
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp


def _distributed_entry(
    rank: int,
    worker: Callable[[int, int], dict[str, Any]],
    world_size: int,
    init_method: str,
    queue: Any,
) -> None:
    """Initialize one Gloo rank, execute ``worker``, and report a serializable result."""
    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        queue.put({"rank": rank, "result": worker(rank, world_size), "error": None})
    except Exception:
        queue.put({"rank": rank, "result": None, "error": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_local_gloo_world(
    worker: Callable[[int, int], dict[str, Any]], *, world_size: int = 2
) -> list[dict[str, Any]]:
    """Run ``worker`` in a temporary local Gloo world and return results ordered by rank."""
    if world_size < 2:
        raise ValueError("world_size must be at least 2 for a multiprocess conformance check")
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("the current PyTorch build does not provide the Gloo backend")
    context = mp.get_context("fork")
    queue = context.SimpleQueue()
    rendezvous_fd, rendezvous_path = tempfile.mkstemp(prefix="part02_gloo_")
    os.close(rendezvous_fd)
    try:
        mp.start_processes(
            _distributed_entry,
            args=(worker, world_size, f"file://{rendezvous_path}", queue),
            nprocs=world_size,
            join=True,
            start_method="fork",
        )
        messages = [queue.get() for _ in range(world_size)]
    finally:
        if os.path.exists(rendezvous_path):
            os.unlink(rendezvous_path)
    failures = [message for message in messages if message["error"]]
    if failures:
        raise RuntimeError(f"local Gloo worker failed: {failures}")
    return [message["result"] for message in sorted(messages, key=lambda item: item["rank"])]
