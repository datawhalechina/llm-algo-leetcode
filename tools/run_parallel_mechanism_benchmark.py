"""Run real multi-GPU mechanism benchmarks for Part 02 parallelism lessons.

Each mode executes a complete mechanism around the collective: a sharded MLP,
a repeated collective matrix, MoE token dispatch + expert compute + combine,
context-parallel K/V exchange, communication-compute overlap, or replica load
distribution. Launch it with ``torchrun``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _format_version(value) -> str:
    """Normalize library versions returned as tuples, integers, or strings."""
    if isinstance(value, (tuple, list)):
        return ".".join(map(str, value))
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "tp_mlp",
            "collective_matrix",
            "ep_dispatch",
            "cp_attention",
            "comm_overlap",
            "replica_routing",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--payload-elements", default="262144,1048576,4194304")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--in-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=11008)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--tokens-per-rank", type=int, default=256)
    parser.add_argument("--routing", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--compute-dim", type=int, default=2048)
    return parser.parse_args()


def _timed(torch, dist, fn, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    dist.barrier()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    local = torch.tensor([sum(samples) / len(samples)], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    per_rank = [round(item.item(), 5) for item in gathered]
    return max(per_rank), per_rank


def _run_tp(torch, dist, args, device, dtype) -> dict:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if args.hidden_dim % world_size:
        raise ValueError("hidden-dim 必须能被 world size 整除")
    torch.manual_seed(17)
    x = torch.randn(args.batch_size, args.in_dim, device=device, dtype=dtype)
    w1 = torch.randn(args.in_dim, args.hidden_dim, device=device, dtype=dtype) / args.in_dim**0.5
    w2 = torch.randn(args.hidden_dim, args.out_dim, device=device, dtype=dtype) / args.hidden_dim**0.5
    hidden_per_rank = args.hidden_dim // world_size
    start = rank * hidden_per_rank
    end = start + hidden_per_rank
    w1_local = w1[:, start:end].contiguous()
    w2_local = w2[start:end, :].contiguous()
    dense_reference = torch.nn.functional.gelu(x @ w1) @ w2
    del w1, w2

    def tp_forward():
        partial = torch.nn.functional.gelu(x @ w1_local) @ w2_local
        dist.all_reduce(partial)
        return partial

    output = tp_forward()
    max_error = (output.float() - dense_reference.float()).abs().max()
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    del dense_reference, output
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms, per_rank = _timed(torch, dist, tp_forward, args.warmup, args.repeats)
    local_peak = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / 1024**2], device=device, dtype=torch.float64
    )
    gathered_peak = [torch.empty_like(local_peak) for _ in range(world_size)]
    dist.all_gather(gathered_peak, local_peak)
    return {
        "strategy": "column_row_sharded_mlp",
        "metrics": {
            "max_abs_error": float(max_error.item()),
            "max_rank_step_ms": elapsed_ms,
            "rank_step_ms": per_rank,
            "peak_memory_mb_per_rank": [round(item.item(), 3) for item in gathered_peak],
            "all_reduce_payload_bytes_per_rank": args.batch_size * args.out_dim * torch.tensor([], dtype=dtype).element_size(),
            "tokens_per_s_proxy": round(args.batch_size / elapsed_ms * 1000.0, 3),
        },
    }


def _run_collectives(torch, dist, args, device, dtype) -> dict:
    world_size = dist.get_world_size()
    rows = []
    for elements in [int(item) for item in args.payload_elements.split(",") if item]:
        if elements <= 0 or elements % world_size:
            raise ValueError("每个 payload-elements 必须为正且能被 world size 整除")
        local = torch.ones(elements, device=device, dtype=dtype)
        element_size = local.element_size()
        gather_output = torch.empty(elements * world_size, device=device, dtype=dtype)
        scatter_input = torch.ones(elements, device=device, dtype=dtype)
        scatter_output = torch.empty(elements // world_size, device=device, dtype=dtype)
        all_to_all_output = torch.empty_like(local)
        operations = {
            "all_reduce": lambda: dist.all_reduce(local),
            "all_gather_into_tensor": lambda: dist.all_gather_into_tensor(gather_output, local),
            "reduce_scatter_tensor": lambda: dist.reduce_scatter_tensor(scatter_output, scatter_input),
            "all_to_all_single": lambda: dist.all_to_all_single(all_to_all_output, local),
        }
        for name, operation in operations.items():
            elapsed_ms, per_rank = _timed(torch, dist, operation, args.warmup, args.repeats)
            payload_bytes = elements * element_size
            rows.append({
                "operation": name,
                "payload_bytes_per_rank": payload_bytes,
                "max_rank_elapsed_ms": elapsed_ms,
                "rank_elapsed_ms": per_rank,
                "effective_payload_gbps": round(payload_bytes / (elapsed_ms / 1000.0) / 1e9, 5),
            })
    return {"strategy": "collective_size_matrix", "metrics": {"rows": rows}}


def _destination_counts(tokens: int, world_size: int, routing: str) -> list[int]:
    if tokens % world_size:
        raise ValueError("tokens-per-rank 必须能被 world size 整除")
    if routing == "balanced":
        return [tokens // world_size] * world_size
    hot = tokens * 3 // 4
    remaining = tokens - hot
    quotient, remainder = divmod(remaining, world_size - 1)
    counts = [hot] + [quotient + (index < remainder) for index in range(world_size - 1)]
    return counts


def _run_ep(torch, dist, args, device, dtype) -> dict:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    send_counts = _destination_counts(args.tokens_per_rank, world_size, args.routing)
    send_count_tensor = torch.tensor(send_counts, device=device, dtype=torch.int64)
    recv_count_tensor = torch.empty_like(send_count_tensor)
    dist.all_to_all_single(recv_count_tensor, send_count_tensor)
    recv_counts = [int(item) for item in recv_count_tensor.tolist()]
    torch.manual_seed(100 + rank)
    dispatched = torch.randn(args.tokens_per_rank, args.hidden_dim, device=device, dtype=dtype)
    expert_weight = torch.randn(args.hidden_dim, args.hidden_dim, device=device, dtype=dtype) / args.hidden_dim**0.5

    def dispatch_expert_combine():
        received = torch.empty(sum(recv_counts), args.hidden_dim, device=device, dtype=dtype)
        dist.all_to_all_single(received, dispatched, recv_counts, send_counts)
        expert_output = torch.nn.functional.gelu(received @ expert_weight)
        combined = torch.empty_like(dispatched)
        dist.all_to_all_single(combined, expert_output, send_counts, recv_counts)
        return combined

    combined = dispatch_expert_combine()
    if combined.shape != dispatched.shape or not torch.isfinite(combined).all():
        raise RuntimeError("dispatch / combine 输出形状或数值异常")
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms, per_rank = _timed(torch, dist, dispatch_expert_combine, args.warmup, args.repeats)
    local_received = torch.tensor([sum(recv_counts)], device=device, dtype=torch.float64)
    received_all = [torch.empty_like(local_received) for _ in range(world_size)]
    dist.all_gather(received_all, local_received)
    loads = [int(item.item()) for item in received_all]
    mean_load = sum(loads) / len(loads)
    return {
        "strategy": f"ep_{args.routing}_routing",
        "metrics": {
            "max_rank_step_ms": elapsed_ms,
            "rank_step_ms": per_rank,
            "received_tokens_per_rank": loads,
            "load_skew_ratio": round((max(loads) - min(loads)) / mean_load, 5) if mean_load else 0.0,
            "dispatch_bytes_per_rank": args.tokens_per_rank * args.hidden_dim * dispatched.element_size(),
            "peak_memory_mb_per_rank": round(torch.cuda.max_memory_allocated(device) / 1024**2, 3),
        },
    }


def _run_cp_attention(torch, dist, args, device, dtype) -> dict:
    """Measure local-query attention with a real all-gather of sharded K/V."""
    world_size = dist.get_world_size()
    if args.seq_len % world_size:
        raise ValueError("seq-len 必须能被 world size 整除")
    local_len = args.seq_len // world_size
    torch.manual_seed(301 + dist.get_rank())
    q_local = torch.randn(1, args.num_heads, local_len, args.head_dim, device=device, dtype=dtype)
    k_local = torch.randn_like(q_local)
    v_local = torch.randn_like(q_local)
    gathered_k = [torch.empty_like(k_local) for _ in range(world_size)]
    gathered_v = [torch.empty_like(v_local) for _ in range(world_size)]

    def exchange_kv():
        dist.all_gather(gathered_k, k_local)
        dist.all_gather(gathered_v, v_local)
        return torch.cat(gathered_k, dim=2), torch.cat(gathered_v, dim=2)

    k_full, v_full = exchange_kv()

    def compute_only():
        return torch.nn.functional.scaled_dot_product_attention(q_local, k_full, v_full)

    def exchange_then_compute():
        k_global, v_global = exchange_kv()
        return torch.nn.functional.scaled_dot_product_attention(q_local, k_global, v_global)

    reference = compute_only()
    candidate = exchange_then_compute()
    max_error = float((reference.float() - candidate.float()).abs().max().item())
    compute_ms, compute_per_rank = _timed(torch, dist, compute_only, args.warmup, args.repeats)
    total_ms, total_per_rank = _timed(torch, dist, exchange_then_compute, args.warmup, args.repeats)
    element_size = q_local.element_size()
    local_kv_bytes = 2 * args.num_heads * local_len * args.head_dim * element_size
    return {
        "strategy": "context_parallel_kv_all_gather",
        "metrics": {
            "max_abs_error": max_error,
            "compute_only_max_rank_ms": compute_ms,
            "exchange_compute_max_rank_ms": total_ms,
            "exposed_exchange_ms_proxy": round(max(total_ms - compute_ms, 0.0), 5),
            "compute_only_rank_ms": compute_per_rank,
            "exchange_compute_rank_ms": total_per_rank,
            "local_kv_bytes_per_rank": local_kv_bytes,
            "replicated_kv_bytes_per_rank": local_kv_bytes * world_size,
            "kv_exchange_bytes_per_rank": local_kv_bytes,
        },
    }


def _run_comm_overlap(torch, dist, args, device, dtype) -> dict:
    """Compare serial and asynchronous collective-compute schedules."""
    elements = int(args.payload_elements.split(",")[0])
    if elements <= 0:
        raise ValueError("payload-elements 必须为正整数")
    base = torch.randn(elements, device=device, dtype=dtype)
    payload = torch.empty_like(base)
    torch.manual_seed(401 + dist.get_rank())
    x = torch.randn(args.compute_dim, args.compute_dim, device=device, dtype=dtype)
    weight = torch.randn_like(x)

    def serial_schedule():
        payload.copy_(base)
        dist.all_reduce(payload)
        return x @ weight

    def overlap_schedule():
        payload.copy_(base)
        work = dist.all_reduce(payload, async_op=True)
        output = x @ weight
        work.wait()
        return output

    serial_ms, serial_per_rank = _timed(torch, dist, serial_schedule, args.warmup, args.repeats)
    overlap_ms, overlap_per_rank = _timed(torch, dist, overlap_schedule, args.warmup, args.repeats)
    return {
        "strategy": "async_collective_compute_overlap",
        "metrics": {
            "serial_max_rank_ms": serial_ms,
            "overlap_max_rank_ms": overlap_ms,
            "saved_ms": round(serial_ms - overlap_ms, 5),
            "speedup": round(serial_ms / overlap_ms, 5) if overlap_ms else None,
            "serial_rank_ms": serial_per_rank,
            "overlap_rank_ms": overlap_per_rank,
            "payload_bytes_per_rank": base.numel() * base.element_size(),
            "compute_dim": args.compute_dim,
        },
    }


def _assign_replica_requests(costs: list[int], world_size: int, strategy: str) -> list[list[int]]:
    assignments = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for index, cost in enumerate(costs):
        if strategy == "round_robin":
            target = index % world_size
        else:
            target = min(range(world_size), key=lambda rank: loads[rank])
        assignments[target].append(cost)
        loads[target] += cost
    return assignments


def _run_replica_routing(torch, dist, args, device, dtype) -> dict:
    """Run the same skewed request set under two replica assignment policies."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    request_costs = [1, 8, 2, 7, 1, 6, 2, 5]
    matrix_dim = min(args.compute_dim, 1024)
    torch.manual_seed(501 + rank)
    x = torch.randn(matrix_dim, matrix_dim, device=device, dtype=dtype)
    weight = torch.randn_like(x)
    rows = []
    for strategy in ("round_robin", "least_loaded"):
        local_costs = _assign_replica_requests(request_costs, world_size, strategy)[rank]

        def run_local_queue():
            output = None
            for cost in local_costs:
                for _ in range(cost):
                    output = x @ weight
            return output

        elapsed_ms, per_rank = _timed(torch, dist, run_local_queue, args.warmup, args.repeats)
        rows.append({
            "strategy": strategy,
            "max_rank_makespan_ms": elapsed_ms,
            "rank_elapsed_ms": per_rank,
            "request_cost_units_per_rank": [
                sum(items) for items in _assign_replica_requests(request_costs, world_size, strategy)
            ],
            "throughput_requests_per_s_proxy": round(len(request_costs) / elapsed_ms * 1000.0, 5),
        })
    return {"strategy": "replica_routing_comparison", "metrics": {"rows": rows}}


def main() -> int:
    args = _parse_args()
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    result = {
        "project": {
            "tp_mlp": "29",
            "collective_matrix": "46",
            "ep_dispatch": "47",
            "cp_attention": "PAR-CONTEXT",
            "comm_overlap": "48",
            "replica_routing": "PAR-REPLICA",
        }[args.mode],
        "workload": vars(args) | {"world_size": world_size},
        "hardware": {},
        "metrics": {},
        "failure": None,
        "evidence_level": "environment_preflight",
        "decision": "not_started",
    }
    try:
        if world_size < 2 or not torch.cuda.is_available():
            raise RuntimeError("真实并行 benchmark 至少需要两张 CUDA GPU")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
        result["hardware"] = {
            "gpu": torch.cuda.get_device_name(local_rank),
            "world_size": world_size,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": _format_version(torch.cuda.nccl.version()),
        }
        if args.mode == "tp_mlp":
            measured = _run_tp(torch, dist, args, device, dtype)
        elif args.mode == "collective_matrix":
            measured = _run_collectives(torch, dist, args, device, dtype)
        elif args.mode == "ep_dispatch":
            measured = _run_ep(torch, dist, args, device, dtype)
        elif args.mode == "cp_attention":
            measured = _run_cp_attention(torch, dist, args, device, dtype)
        elif args.mode == "comm_overlap":
            measured = _run_comm_overlap(torch, dist, args, device, dtype)
        else:
            measured = _run_replica_routing(torch, dist, args, device, dtype)
        result.update(measured)
        result["evidence_level"] = "real_multi_gpu_mechanism_benchmark"
        result["decision"] = "benchmark_completed"
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["decision"] = "benchmark_failed"
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
