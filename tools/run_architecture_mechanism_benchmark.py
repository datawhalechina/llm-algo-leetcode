"""Run controlled single-GPU mechanism benchmarks for architecture lessons.

The runner measures representation and state-update costs.  It deliberately
does not claim model-quality or production-backend evidence; those belong to
the architecture project notebooks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("列表必须包含正整数")
    return values


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("kv_representation", "linear_recurrence", "memory_growth"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--sequence-lengths", type=_parse_int_list, default=[512, 2048, 8192])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--attention-every", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def _measure(torch, fn, warmup: int, repeats: int, device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = fn()
        torch.cuda.synchronize(device)
        if not torch.isfinite(output).all():
            raise RuntimeError("GPU 机制输出包含非有限值")
        samples.append((time.perf_counter() - started) * 1000.0)
    return sum(samples) / len(samples), torch.cuda.max_memory_allocated(device) / 1024**2


def _sdpa(torch, query, key, value, enable_gqa: bool = False):
    if enable_gqa:
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, enable_gqa=True
            )
        except TypeError as exc:
            raise RuntimeError("当前 PyTorch 不支持 SDPA enable_gqa；请升级后复测") from exc
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)


def _run_kv_representation(torch, args, device, dtype) -> dict:
    if args.query_heads % args.kv_heads:
        raise ValueError("query-heads 必须能被 kv-heads 整除")
    if args.latent_dim <= 0 or args.rope_dim <= 0:
        raise ValueError("latent-dim 与 rope-dim 必须为正整数")
    rows = []
    element_size = torch.tensor([], dtype=dtype).element_size()
    for sequence_length in args.sequence_lengths:
        torch.manual_seed(701 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )

        mha_key = torch.randn(
            args.batch_size, args.query_heads, sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        mha_value = torch.randn_like(mha_key)
        elapsed_ms, peak_mb = _measure(
            torch, lambda: _sdpa(torch, query, mha_key, mha_value),
            args.warmup, args.repeats, device,
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "mha_full_kv",
            "cache_bytes": 2 * mha_key.numel() * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "reconstruction": False,
        })
        del mha_key, mha_value

        gqa_key = torch.randn(
            args.batch_size, args.kv_heads, sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        gqa_value = torch.randn_like(gqa_key)
        elapsed_ms, peak_mb = _measure(
            torch, lambda: _sdpa(torch, query, gqa_key, gqa_value, enable_gqa=True),
            args.warmup, args.repeats, device,
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "gqa_shared_kv",
            "cache_bytes": 2 * gqa_key.numel() * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "reconstruction": False,
        })
        del gqa_key, gqa_value

        latent = torch.randn(
            args.batch_size, sequence_length, args.latent_dim, device=device, dtype=dtype
        )
        rope_state = torch.randn(
            args.batch_size, sequence_length, args.rope_dim, device=device, dtype=dtype
        )
        key_projection = torch.randn(
            args.latent_dim, args.kv_heads * args.head_dim, device=device, dtype=dtype
        ) / args.latent_dim**0.5
        value_projection = torch.randn_like(key_projection)

        def latent_attention():
            key = (latent @ key_projection).view(
                args.batch_size, sequence_length, args.kv_heads, args.head_dim
            ).transpose(1, 2)
            value = (latent @ value_projection).view(
                args.batch_size, sequence_length, args.kv_heads, args.head_dim
            ).transpose(1, 2)
            return _sdpa(torch, query, key, value, enable_gqa=True)

        elapsed_ms, peak_mb = _measure(
            torch, latent_attention, args.warmup, args.repeats, device
        )
        rows.append({
            "sequence_length": sequence_length,
            "representation": "latent_kv_proxy",
            "cache_bytes": (latent.numel() + rope_state.numel()) * element_size,
            "mean_step_ms": round(elapsed_ms, 5),
            "peak_memory_mb": round(peak_mb, 3),
            "reconstruction": True,
        })
        del latent, rope_state, key_projection, value_projection, query
        torch.cuda.empty_cache()
    return {"comparison_type": "controlled_representation_proxy", "rows": rows}


def _run_memory_growth(torch, args, device, dtype) -> dict:
    if args.attention_every <= 0 or args.num_layers <= 0:
        raise ValueError("num-layers 与 attention-every 必须为正整数")
    recurrent_layers = args.num_layers
    hybrid_attention_layers = max(1, args.num_layers // args.attention_every)
    hybrid_recurrent_layers = args.num_layers - hybrid_attention_layers
    rows = []
    element_size = torch.tensor([], dtype=dtype).element_size()
    for sequence_length in args.sequence_lengths:
        torch.manual_seed(801 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        key_history = torch.randn(
            args.batch_size, args.query_heads, sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        value_history = torch.randn_like(key_history)
        recurrent_state = torch.randn(
            args.batch_size, args.query_heads, args.head_dim, args.head_dim,
            device=device, dtype=dtype,
        )
        key_now = torch.randn(
            args.batch_size, args.query_heads, args.head_dim, device=device, dtype=dtype
        )
        value_now = torch.randn_like(key_now)

        def attention_step():
            output = None
            for _ in range(args.num_layers):
                output = _sdpa(torch, query, key_history, value_history)
            return output

        def recurrent_step():
            state = recurrent_state
            output = None
            update = key_now.unsqueeze(-1) * value_now.unsqueeze(-2)
            for _ in range(recurrent_layers):
                state = 0.99 * state + update
                output = torch.matmul(query, state)
            return output

        def hybrid_step():
            output = None
            for _ in range(hybrid_attention_layers):
                output = _sdpa(torch, query, key_history, value_history)
            state = recurrent_state
            update = key_now.unsqueeze(-1) * value_now.unsqueeze(-2)
            for _ in range(hybrid_recurrent_layers):
                state = 0.99 * state + update
                output = torch.matmul(query, state)
            return output

        strategies = (
            ("explicit_attention", attention_step,
             args.num_layers * 2 * key_history.numel() * element_size),
            ("fixed_recurrent_state", recurrent_step,
             recurrent_layers * recurrent_state.numel() * element_size),
            ("hybrid_attention_recurrent", hybrid_step,
             hybrid_attention_layers * 2 * key_history.numel() * element_size
             + hybrid_recurrent_layers * recurrent_state.numel() * element_size),
        )
        for name, fn, state_bytes in strategies:
            elapsed_ms, peak_mb = _measure(torch, fn, args.warmup, args.repeats, device)
            rows.append({
                "sequence_length": sequence_length,
                "representation": name,
                "state_bytes": state_bytes,
                "mean_step_ms": round(elapsed_ms, 5),
                "peak_memory_mb": round(peak_mb, 3),
                "attention_layers": args.num_layers if name == "explicit_attention" else (
                    hybrid_attention_layers if name.startswith("hybrid") else 0
                ),
            })
        del query, key_history, value_history, recurrent_state, key_now, value_now
        torch.cuda.empty_cache()
    return {"comparison_type": "controlled_state_interface_proxy", "rows": rows}


def _run_linear_recurrence(torch, args, device, dtype) -> dict:
    """Compare one-layer explicit history with a fixed recurrent state."""
    rows = []
    element_size = torch.tensor([], dtype=dtype).element_size()
    for sequence_length in args.sequence_lengths:
        torch.manual_seed(751 + sequence_length)
        query = torch.randn(
            args.batch_size, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
        )
        key_history = torch.randn(
            args.batch_size, args.query_heads, sequence_length, args.head_dim,
            device=device, dtype=dtype,
        )
        value_history = torch.randn_like(key_history)
        recurrent_state = torch.randn(
            args.batch_size, args.query_heads, args.head_dim, args.head_dim,
            device=device, dtype=dtype,
        )
        key_now = torch.randn(
            args.batch_size, args.query_heads, args.head_dim, device=device, dtype=dtype
        )
        value_now = torch.randn_like(key_now)

        def explicit_history_step():
            return _sdpa(torch, query, key_history, value_history)

        def recurrent_state_step():
            update = key_now.unsqueeze(-1) * value_now.unsqueeze(-2)
            updated_state = 0.99 * recurrent_state + update
            return torch.matmul(query, updated_state)

        candidates = (
            ("explicit_attention_history", explicit_history_step,
             2 * key_history.numel() * element_size),
            ("fixed_recurrent_state", recurrent_state_step,
             recurrent_state.numel() * element_size),
        )
        for name, fn, state_bytes in candidates:
            elapsed_ms, peak_mb = _measure(torch, fn, args.warmup, args.repeats, device)
            rows.append({
                "sequence_length": sequence_length,
                "representation": name,
                "state_bytes": state_bytes,
                "mean_step_ms": round(elapsed_ms, 5),
                "peak_memory_mb": round(peak_mb, 3),
            })
        del query, key_history, value_history, recurrent_state, key_now, value_now
        torch.cuda.empty_cache()
    return {"comparison_type": "controlled_single_layer_state_proxy", "rows": rows}


def main() -> int:
    args = _args()
    import torch

    result = {
        "mechanism": args.mode,
        "workload": vars(args),
        "hardware": {},
        "metrics": {},
        "failure": None,
        "evidence_level": "environment_preflight",
        "decision": "not_started",
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("架构机制 GPU 实验需要 CUDA GPU")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
        result["hardware"] = {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": args.dtype,
        }
        if args.mode == "kv_representation":
            result["metrics"] = _run_kv_representation(torch, args, device, dtype)
        elif args.mode == "linear_recurrence":
            result["metrics"] = _run_linear_recurrence(torch, args, device, dtype)
        else:
            result["metrics"] = _run_memory_growth(torch, args, device, dtype)
        result["evidence_level"] = "single_gpu_mechanism_benchmark"
        result["decision"] = "mechanism_measurement_completed"
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["decision"] = "mechanism_measurement_failed"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
