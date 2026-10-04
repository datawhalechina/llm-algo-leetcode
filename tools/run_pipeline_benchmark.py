"""Run a real DeepSpeed PipelineModule benchmark on a synthetic workload.

The runner records the framework's default training schedule.  It does not
claim to benchmark a separately implemented fill-drain schedule; the CPU
lesson in Part 02 · 28 remains the place where those event dependencies are
compared directly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-stages", type=int, required=True)
    parser.add_argument("--micro-batches", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "bf16"), default="bf16")
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    for field in ("pipeline_stages", "micro_batches", "width", "layers", "micro_batch", "seq_len", "repeats"):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field.replace('_', '-')} 必须为正整数")


def main() -> int:
    args = _args()
    _validate(args)

    # Imports stay inside main so CPU-first validation does not need DeepSpeed.
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    result: dict[str, object] = {
        "project": "28",
        "strategy": "deepspeed_pipeline_default_schedule",
        "workload": {
            "model": "synthetic_transformer_mlp",
            "pipeline_stages": args.pipeline_stages,
            "micro_batches": args.micro_batches,
            "width": args.width,
            "layers": args.layers,
            "micro_batch": args.micro_batch,
            "seq_len": args.seq_len,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "world_size": world_size,
        },
        "hardware": torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu",
        "metrics": {},
        "failure": None,
        "evidence_level": "real_multi_gpu_pipeline",
        "decision": "not_started",
    }

    try:
        if world_size != args.pipeline_stages:
            raise RuntimeError("torchrun WORLD_SIZE 必须与 pipeline-stages 相同")
        if not torch.cuda.is_available():
            raise RuntimeError("Pipeline benchmark 需要 CUDA GPU")
        try:
            import deepspeed
            from deepspeed.pipe import LayerSpec, PipelineModule
        except ImportError as exc:
            raise RuntimeError("请安装 DeepSpeed：pip install deepspeed") from exc

        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="nccl")
        device = torch.device("cuda", local_rank)

        class ToyBlock(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.up = nn.Linear(width, 4 * width, bias=False)
                self.down = nn.Linear(4 * width, width, bias=False)

            def forward(self, x):
                return x + self.down(torch.nn.functional.gelu(self.up(x)))

        layers = [LayerSpec(ToyBlock, args.width) for _ in range(args.layers)]
        model = PipelineModule(
            layers=layers,
            loss_fn=nn.MSELoss(),
            num_stages=args.pipeline_stages,
            partition_method="parameters",
        )
        config = {
            "train_micro_batch_size_per_gpu": args.micro_batch,
            "gradient_accumulation_steps": args.micro_batches,
            "bf16": {"enabled": args.dtype == "bf16"},
            "fp16": {"enabled": False},
        }
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        def data_iter():
            while True:
                x = torch.randn(args.micro_batch, args.seq_len, args.width, device=device)
                y = torch.zeros_like(x)
                yield x, y

        batches = data_iter()
        for _ in range(args.warmup):
            engine.train_batch(data_iter=batches)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.repeats):
            engine.train_batch(data_iter=batches)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / args.repeats
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        elapsed = torch.tensor([elapsed_ms], device=device, dtype=torch.float64)
        memory = torch.tensor([memory_mb], device=device, dtype=torch.float64)
        elapsed_all = [torch.empty_like(elapsed) for _ in range(world_size)]
        memory_all = [torch.empty_like(memory) for _ in range(world_size)]
        dist.all_gather(elapsed_all, elapsed)
        dist.all_gather(memory_all, memory)
        max_step_ms = max(item.item() for item in elapsed_all)
        result["metrics"] = {
            "step_time_ms_per_rank": [round(item.item(), 4) for item in elapsed_all],
            "peak_memory_mb_per_rank": [round(item.item(), 3) for item in memory_all],
            "max_step_time_ms": round(max_step_ms, 4),
            "max_peak_memory_mb": round(max(item.item() for item in memory_all), 3),
            "tokens_per_s_proxy": round(
                args.micro_batch * args.micro_batches * args.seq_len / max_step_ms * 1000,
                3,
            ),
        }
        result["decision"] = "pipeline_completed"
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["decision"] = "pipeline_unavailable"
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
