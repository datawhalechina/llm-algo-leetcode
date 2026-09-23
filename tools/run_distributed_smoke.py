"""Run small, real torch.distributed probes for Part 02 communication projects.

This is deliberately a smoke runner, not a model benchmark.  It verifies that
the requested collective can be initialized and executed on the current
multi-GPU environment, then writes a machine-readable result for the notebook
to reference.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=("29", "46", "47", "79", "80", "81"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="nccl")
    return parser.parse_args()


def _sync_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def main() -> int:
    args = _args()
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    result = {
        "project": args.project,
        "role": "gpu_smoke",
        "backend": args.backend,
        "rank": rank,
        "world_size": world_size,
        "hardware": torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu",
        "load_status": "not_started",
        "failure": None,
        "evidence_level": "real_multi_gpu_smoke",
    }

    try:
        if world_size < 2:
            raise RuntimeError("真实多卡 smoke 至少需要 WORLD_SIZE=2。")
        if not torch.cuda.is_available():
            raise RuntimeError("真实多卡 NCCL smoke 需要 CUDA GPU。")
        if args.backend != "nccl":
            raise ValueError("当前 runner 只把 NCCL 作为真实 GPU smoke backend。")

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.backend, init_method="env://")
        device = torch.device("cuda", local_rank)
        payload = torch.ones(1024, device=device)
        dist.barrier()
        _sync_cuda(torch)
        start = time.perf_counter()

        if args.project in {"47", "80"}:
            received = torch.empty_like(payload)
            dist.all_to_all_single(received, payload)
            operation = "all_to_all_single"
        elif args.project == "29":
            gathered = [torch.empty_like(payload) for _ in range(world_size)]
            dist.all_gather(gathered, payload)
            dist.all_reduce(payload)
            operation = "all_gather_then_all_reduce"
        else:
            dist.all_reduce(payload)
            operation = "all_reduce"

        _sync_cuda(torch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        dist.barrier()
        result.update({
            "operation": operation,
            "payload_bytes": payload.numel() * payload.element_size(),
            "elapsed_ms": round(elapsed_ms, 4),
            "load_status": "initialized_and_collective_completed",
        })
    except Exception as exc:  # keep failure evidence instead of hiding environment problems
        result["load_status"] = "failed"
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    output = Path(args.output)
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["load_status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
