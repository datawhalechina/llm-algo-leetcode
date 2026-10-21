"""Create an isolated Conda environment from a tutorial Profile registry entry.

This command is intentionally conservative: an existing environment is not
modified unless ``--reuse`` is explicit.  It never removes an environment.

Example:
  python tools/install_environment_profile.py \
    --profile Q-CUDA-MECHANISM --env-name llm_quant_cu128
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from project_runtime import bootstrap_project_root, load_environment_profile


def command_text(parts: list[str]) -> str:
    """Render the simple argument lists used by this installer."""

    return " ".join(parts)


def run(parts: list[str], *, dry_run: bool) -> None:
    """Print every command and execute it unless this is a dry run."""

    print("+", command_text(parts), flush=True)
    if not dry_run:
        subprocess.run(parts, check=True)


def conda_environments(conda: str) -> set[str]:
    """Return absolute prefixes and environment names visible to Conda."""

    completed = subprocess.run(
        [conda, "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefixes = [Path(item) for item in json.loads(completed.stdout).get("envs", [])]
    return {str(path) for path in prefixes} | {path.name for path in prefixes}


def main() -> int:
    parser = argparse.ArgumentParser(description="按教程 Profile 创建隔离 Conda 环境")
    parser.add_argument("--profile", required=True, help="environment-profiles.json 中的 Profile ID")
    parser.add_argument("--env-name", required=True, help="要创建或显式复用的 Conda 环境名")
    parser.add_argument("--python", default="3.12", help="Conda Python 版本，默认 3.12")
    parser.add_argument("--reuse", action="store_true", help="显式允许向已有环境安装登记依赖")
    parser.add_argument("--no-kernel", action="store_true", help="不注册 Jupyter Kernel")
    parser.add_argument("--dry-run", action="store_true", help="只打印安装计划，不执行命令")
    args = parser.parse_args()

    root = bootstrap_project_root()
    profile = load_environment_profile(root, args.profile)
    external_manager = profile.get("external_manager")
    if external_manager:
        documentation = profile.get("documentation", {}).get("topic", "专题 Quickstart")
        print(
            f"Profile {args.profile!r} 由 {external_manager.get('kind', 'external')} 管理，"
            f"不能写入 Conda 通用环境；请按 {documentation} 使用上游锁文件创建隔离环境",
            file=sys.stderr,
        )
        return 2
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("未找到 conda；请先安装 Miniconda/Anaconda，或手动执行专题 Quickstart")

    existing = set() if args.dry_run else conda_environments(conda)
    environment_exists = args.env_name in existing
    if environment_exists and not args.reuse:
        raise RuntimeError(
            f"Conda 环境 {args.env_name!r} 已存在；为避免覆盖，请换一个名称，"
            "或确认环境可复用后显式添加 --reuse"
        )

    print(json.dumps({
        "profile": args.profile,
        "title": profile.get("title"),
        "env_name": args.env_name,
        "requirements": profile.get("requirements", []),
        "constraints": profile.get("constraint_files", []),
        "reuse": args.reuse,
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))

    if not environment_exists:
        run(
            [conda, "create", "-n", args.env_name, f"python={args.python}", "-y"],
            dry_run=args.dry_run,
        )

    isolation = profile.get("isolation", {})
    environment_variables: list[str] = []
    if isolation.get("require_user_site_disabled", False):
        environment_variables.append("PYTHONNOUSERSITE=1")
    if isolation.get("require_clean_sys_path", False):
        environment_variables.append("PYTHONPATH=")
    if environment_variables:
        run(
            [
                conda,
                "env",
                "config",
                "vars",
                "set",
                "-n",
                args.env_name,
                *environment_variables,
            ],
            dry_run=args.dry_run,
        )

    # Stream pip and preflight output instead of letting ``conda run`` buffer
    # long downloads until the child process exits.
    python = [conda, "run", "--no-capture-output", "-n", args.env_name, "python"]
    run([*python, "-m", "pip", "install", "--upgrade", "pip"], dry_run=args.dry_run)

    constraint_args: list[str] = []
    for relative in profile.get("constraint_files", []):
        constraint_args.extend(["-c", str(root / relative)])
    for relative in profile.get("requirements", []):
        run(
            [*python, "-m", "pip", "install", "-r", str(root / relative), *constraint_args],
            dry_run=args.dry_run,
        )

    run(
        [*python, str(root / "tools" / "register_notebook_runtime.py")],
        dry_run=args.dry_run,
    )
    run([*python, "-m", "pip", "check"], dry_run=args.dry_run)
    run(
        [*python, str(root / "tools" / "environment_preflight.py"), "--profile", args.profile],
        dry_run=args.dry_run,
    )
    if not args.no_kernel:
        run(
            [
                *python,
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                args.env_name,
                "--display-name",
                f"Python ({args.env_name})",
            ],
            dry_run=args.dry_run,
        )

    print("环境安装与预检完成。请重启 Notebook Kernel，并选择登记的环境。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
