"""Load one quantized artifact with vLLM and write a project-82 loader record."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--invocation-json", required=True, help="固定输入身份与调用合同，由项目入口生成")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    from tools import quantization_project_runtime as runtime
    from tools.project_runtime import resolve_project_root

    invocation = json.loads(args.invocation_json)
    plan = runtime.artifact_gate_plan(resolve_project_root(), invocation["run_label"], invocation["upstream_run_label"], "measure")
    prepared = runtime.prepare_artifact_gate(plan)
    if not prepared["ready"] or invocation != runtime.artifact_loader_invocation(plan, prepared):
        raise ValueError("loader inputs do not match the requested snapshot")
    if output.resolve() != plan["loader_path"].resolve() or Path(args.artifact).resolve() != Path(prepared["snapshot"]["artifact_path"]):
        raise ValueError("loader input/output paths differ from the fixed contract")
    contract = plan["contract"]
    if (args.max_model_len, args.max_tokens, args.gpu_memory_utilization) != (contract["max_model_len"], contract["max_tokens"], contract["gpu_memory_utilization"]):
        raise ValueError("loader command differs from the fixed invocation")
    if output.exists():
        raise FileExistsError(f"loader evidence already exists: {output}")
    result = {
        "schema_version": "quantization-artifact-loader/v2",
        "backend": "vllm",
        "invocation": invocation,
        "memory_measurement": "device_free_memory_delta_not_peak",
        "status": "failed",
        "framework": {
            "python": platform.python_version(),
            "torch": None,
            "cuda": None,
        },
        "hardware": {
            "cuda_available": False,
            "device": None,
            "total_memory_bytes": None,
        },
        "backend_version": None,
        "load_status": "failed",
        "roundtrip_quality": {"non_empty_generations": 0, "prompt_count": 2},
        "peak_memory_mb": None,
        "kernel": "not_verified_by_loader_gate",
        "fallback": "not_verified_by_loader_gate",
        "failure": None,
    }
    try:
        import torch
        from vllm import LLM, SamplingParams

        result["framework"].update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        })
        result["hardware"].update({
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_bytes": (
                int(torch.cuda.get_device_properties(0).total_memory)
                if torch.cuda.is_available()
                else None
            ),
        })
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable")
        result["backend_version"] = importlib.metadata.version("vllm")
        free_before, _ = torch.cuda.mem_get_info()
        engine = LLM(
            model=str(Path(args.artifact).resolve()),
            dtype="auto",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=False,
        )
        prompts = ["量化部署验证：请简要解释校准。", "Quantization deployment check:"]
        outputs = engine.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
            use_tqdm=False,
        )
        generated = [item.outputs[0].text if item.outputs else "" for item in outputs]
        free_after, _ = torch.cuda.mem_get_info()
        non_empty = sum(bool(text.strip()) for text in generated)
        result.update({
            "status": "ok" if non_empty == len(prompts) else "failed",
            "load_status": "loaded" if non_empty == len(prompts) else "roundtrip_failed",
            "roundtrip_quality": {
                "non_empty_generations": non_empty,
                "prompt_count": len(prompts),
            },
            "peak_memory_mb": round(max(0, free_before - free_after) / (1024 ** 2), 3),
            "failure": None if non_empty == len(prompts) else {"reason": "empty generation"},
        })
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
    current = runtime.prepare_artifact_gate(plan)
    if not current["ready"] or current["snapshot"] != prepared["snapshot"]:
        result.update(status="failed", load_status="failed", failure={"stage": "input_revalidation", "reason": "artifact inputs changed during execution"})
    runtime._gate_save_or_verify(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
