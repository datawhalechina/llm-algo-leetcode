"""Command-line preflight for running one project notebook independently.

Examples:
  python tools/environment_preflight.py --gpu --packages transformers
  python tools/environment_preflight.py --gpu --native-bf16 --output benchmarks/results/73.json
"""

from __future__ import annotations

import argparse
import json
import sys

from project_runtime import bootstrap_project_root, environment_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="检查项目 Notebook 的 Python、PyTorch、GPU 和结果目录")
    parser.add_argument("--gpu", action="store_true", help="要求 CUDA 可用")
    parser.add_argument("--native-bf16", action="store_true", help="要求确认原生 BF16 加速")
    parser.add_argument("--min-gpu-memory-gb", type=float, default=None)
    parser.add_argument("--packages", nargs="*", default=(), help="必需的 Python 模块名")
    parser.add_argument("--optional-packages", nargs="*", default=(), help="可选的 Python 模块名")
    parser.add_argument("--output", default=None, help="结果 JSON 路径，同时检查父目录可写")
    args = parser.parse_args()

    root = bootstrap_project_root()
    try:
        import torch
    except ImportError:
        torch = None
    report = environment_preflight(
        torch,
        required_packages=tuple(args.packages),
        optional_packages=tuple(args.optional_packages),
        require_gpu=args.gpu,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
        require_native_bf16=args.native_bf16,
        output_path=args.output,
    )
    report["project_root"] = str(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "blocked" else 2


if __name__ == "__main__":
    sys.exit(main())
