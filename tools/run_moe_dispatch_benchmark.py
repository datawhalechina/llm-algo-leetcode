"""Benchmark local versus cross-rank MoE token dispatch on CUDA/NCCL.

This is a synthetic dispatch microbenchmark: it measures the cost of two
All-to-All transfers around a small local expert compute.  It does not claim
to evaluate a real router, capacity overflow, or model quality.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-mode", choices=("local", "cross_rank_balanced"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens-per-rank", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "bf16"), default="bf16")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if min(args.tokens_per_rank, args.hidden_size, args.experts, args.top_k, args.repeats) <= 0:
        raise ValueError("tokens-per-rank、hidden-size、experts、top-k 和 repeats 必须为正整数")
    if args.top_k != 1:
        raise ValueError("当前合成 dispatch runner 只测 top-k=1；真实多专家路由交给 backend benchmark")

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    result: dict[str, object] = {
        "project": "80",
        "strategy": f"moe_dispatch_{args.route_mode}",
        "workload": {
            "route_mode": args.route_mode,
            "tokens_per_rank": args.tokens_per_rank,
            "hidden_size": args.hidden_size,
            "world_size": world_size,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "experts": {"count": args.experts, "top_k": args.top_k},
        "routing": {"kind": "synthetic_balanced", "capacity_factor": None},
        "hardware": torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu",
        "metrics": {},
        "failure": None,
        "evidence_level": "real_multi_gpu_synthetic_moe_dispatch",
        "decision": "not_started",
    }

    try:
        if world_size < 2:
            raise RuntimeError("MoE dispatch benchmark 至少需要两个 torchrun rank")
        if args.tokens_per_rank % world_size:
            raise ValueError("tokens-per-rank 必须能被 world size 整除，便于等量 All-to-All")
        if not torch.cuda.is_available():
            raise RuntimeError("MoE dispatch benchmark 需要 CUDA GPU")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        hidden = torch.randn(args.tokens_per_rank, args.hidden_size, device=device, dtype=dtype)
        weight = torch.randn(args.hidden_size, args.hidden_size, device=device, dtype=dtype)
        chunk_tokens = args.tokens_per_rank // world_size

        def one_step() -> tuple[float, float]:
            if args.route_mode == "local":
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = hidden @ weight
                torch.cuda.synchronize()
                return (time.perf_counter() - start) * 1000.0, 0.0
            send = hidden.reshape(world_size, chunk_tokens, args.hidden_size).contiguous()
            received = torch.empty_like(send)
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_to_all_single(received, send)
            torch.cuda.synchronize()
            after_dispatch = time.perf_counter()
            local_output = received.reshape(args.tokens_per_rank, args.hidden_size) @ weight
            torch.cuda.synchronize()
            before_combine = time.perf_counter()
            returned = torch.empty_like(send)
            dist.all_to_all_single(returned, local_output.reshape_as(send))
            torch.cuda.synchronize()
            end = time.perf_counter()
            dispatch_ms = (after_dispatch - start) * 1000.0
            combine_ms = (end - before_combine) * 1000.0
            return (end - start) * 1000.0, dispatch_ms + combine_ms

        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        total_ms, comm_ms = 0.0, 0.0
        for _ in range(args.repeats):
            elapsed, comm = one_step()
            total_ms += elapsed
            comm_ms += comm
        step_ms = total_ms / args.repeats
        communication_ms = comm_ms / args.repeats
        local = torch.tensor([step_ms, communication_ms], dtype=torch.float64, device=device)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        max_step_ms = max(item[0].item() for item in gathered)
        payload_bytes = args.tokens_per_rank * args.hidden_size * hidden.element_size()
        result["metrics"] = {
            "step_time_ms_per_rank": [round(item[0].item(), 4) for item in gathered],
            "communication_ms_per_rank": [round(item[1].item(), 4) for item in gathered],
            "max_step_time_ms": round(max_step_ms, 4),
            "max_communication_ms": round(max(item[1].item() for item in gathered), 4),
            "dispatch_and_combine_bytes_per_rank": 0 if args.route_mode == "local" else 2 * payload_bytes,
            "cross_rank_token_ratio": 0.0 if args.route_mode == "local" else 1.0,
            "load_imbalance_ratio": 1.0,
            "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3),
            "tokens_per_s_proxy": round(args.tokens_per_rank / max_step_ms * 1000.0, 3),
        }
        result["decision"] = "dispatch_path_completed"
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["decision"] = "dispatch_path_unavailable"
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
