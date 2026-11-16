"""Command-line preflight for running one project notebook independently.

Examples:
  python tools/environment_preflight.py --gpu --packages transformers
  python tools/environment_preflight.py --gpu --native-bf16 --output benchmarks/results/73.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.project_runtime import (
    bootstrap_project_root,
    environment_preflight,
    load_environment_profile,
    load_environment_profiles,
    read_exact_constraints,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="检查项目 Notebook 的 Python、PyTorch、GPU 和结果目录")
    parser.add_argument("--profile", default=None, help="环境 Profile ID，例如 Q-CUDA-MECHANISM")
    parser.add_argument("--list-profiles", action="store_true", help="列出可用环境 Profile 后退出")
    parser.add_argument("--gpu", action="store_true", help="要求 CUDA 可用")
    parser.add_argument("--native-bf16", action="store_true", help="要求确认原生 BF16 加速")
    parser.add_argument("--min-gpu-memory-gb", type=float, default=None)
    parser.add_argument("--packages", nargs="*", default=(), help="必需的 Python 模块名")
    parser.add_argument("--optional-packages", nargs="*", default=(), help="可选的 Python 模块名")
    parser.add_argument("--output", default=None, help="结果 JSON 路径，同时检查父目录可写")
    args = parser.parse_args()

    root = bootstrap_project_root()
    if args.list_profiles:
        print(json.dumps(load_environment_profiles(root), ensure_ascii=False, indent=2))
        return 0

    profile = load_environment_profile(root, args.profile) if args.profile else {}
    required_modules = profile.get("required_modules", {})
    profile_gpu = profile.get("gpu", {})
    profile_isolation = profile.get("isolation", {})
    expected_versions = read_exact_constraints(
        root, tuple(profile.get("constraint_files", ()))
    )

    try:
        import torch
    except ImportError:
        torch = None
    report = environment_preflight(
        torch,
        required_packages=tuple(dict.fromkeys((*required_modules.keys(), *args.packages))),
        optional_packages=tuple(args.optional_packages),
        require_gpu=bool(args.gpu or profile_gpu.get("required", False)),
        min_gpu_memory_gb=(
            args.min_gpu_memory_gb
            if args.min_gpu_memory_gb is not None
            else profile_gpu.get("minimum_memory_gb")
        ),
        require_native_bf16=args.native_bf16,
        output_path=args.output,
        expected_distribution_versions=expected_versions,
        expected_exact_distribution_versions=profile.get(
            "exact_distribution_versions", {}
        ),
        forbidden_distributions=tuple(profile.get("forbidden_distributions", ())),
        expected_torch_cuda=profile_gpu.get("torch_cuda"),
        expected_torch_version=profile_gpu.get("torch_version"),
        minimum_gpu_devices=int(profile_gpu.get("minimum_devices", 1)),
        python_minimum=profile.get("python", {}).get("minimum"),
        python_maximum_exclusive=profile.get("python", {}).get("maximum_exclusive"),
        environment_profile=args.profile,
        require_clean_sys_path=bool(profile_isolation.get("require_clean_sys_path", False)),
        require_user_site_disabled=bool(
            profile_isolation.get("require_user_site_disabled", False)
        ),
    )
    if profile_isolation.get("require_project_runtime", False):
        notebook_directory = root / "02_PyTorch_Algorithms"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from tools.notebook_runtime import package_version; "
                    "from tools.project_runtime import resolve_project_root; "
                    "from tools.quantization_runtime import state_bytes"
                ),
            ],
            cwd=notebook_directory,
            capture_output=True,
            text=True,
        )
        report["project_runtime_import"] = {
            "working_directory": str(notebook_directory),
            "importable": completed.returncode == 0,
            "stderr": completed.stderr.strip(),
        }
        if completed.returncode != 0:
            report["status"] = "blocked"
            report["ready"] = False
            report["reasons"].append("从 Part 02 Notebook 工作目录无法导入仓库公共运行时 tools")
            report["next_actions"].append(
                "在仓库根目录执行当前环境的 python tools/register_notebook_runtime.py，然后重启 Kernel"
            )
    if profile:
        report["profile"] = {
            "title": profile.get("title"),
            "purpose": profile.get("purpose"),
            "requirements": profile.get("requirements", []),
            "documentation": profile.get("documentation", {}),
            "evidence_scope": profile.get("evidence_scope"),
        }
    report["project_root"] = str(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "blocked" else 2


if __name__ == "__main__":
    sys.exit(main())
