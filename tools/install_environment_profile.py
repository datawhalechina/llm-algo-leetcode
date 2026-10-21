"""Create an isolated environment from a tutorial Profile registry entry.

Conda profiles use ``--env-name``; externally locked profiles use ``--target``.
The command is intentionally conservative: an existing environment is not
modified unless ``--reuse`` is explicit. It never removes an environment.

Example:
  python tools/install_environment_profile.py \
    --profile Q-CUDA-MECHANISM --env-name llm_quant_cu128
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from project_runtime import bootstrap_project_root, load_environment_profile


def command_text(parts: list[str]) -> str:
    """Render the simple argument lists used by this installer."""

    return " ".join(parts)


def run(
    parts: list[str],
    *,
    dry_run: bool,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Print every command and execute it unless this is a dry run."""

    prefix = f"(cd {cwd} &&) " if cwd is not None else ""
    print("+", prefix + command_text(parts), flush=True)
    if not dry_run:
        subprocess.run(parts, check=True, cwd=cwd, env=env)


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


def clone_locked_repository(
    *,
    git: str,
    repository: str,
    tag: str,
    target: Path,
    dry_run: bool,
    attempts: int = 3,
) -> None:
    """Clone a shallow locked source tree with bounded network retries."""

    command = [
        git,
        "-c",
        "http.version=HTTP/1.1",
        "-c",
        "http.lowSpeedLimit=1",
        "-c",
        "http.lowSpeedTime=600",
        "clone",
        "--branch",
        tag,
        "--depth",
        "1",
        "--single-branch",
        "--filter=blob:none",
        repository,
        str(target),
    ]
    if dry_run:
        run(command, dry_run=True)
        return

    for attempt in range(1, attempts + 1):
        print(f"+ clone attempt {attempt}/{attempts}: {command_text(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return
        if target.exists():
            raise RuntimeError(
                f"Git clone 失败并留下不完整目录：{target}。"
                "请检查目录内容后移走它，再重新执行安装命令"
            )
        if attempt < attempts:
            print("Git clone 网络中断；目标目录未生成，自动重试。", flush=True)
    raise RuntimeError(f"Git clone 连续失败 {attempts} 次；请检查 GitHub 网络后重试")


def check_uv_dependencies(
    *,
    uv: str,
    python: Path,
    profile: dict,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    """Run uv's metadata check and admit only exact registry exceptions."""

    command = [uv, "pip", "check", "--python", str(python)]
    if dry_run:
        run(command, dry_run=True, cwd=cwd, env=env)
        return

    print("+", f"(cd {cwd} &&) " + command_text(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(
            completed.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    if completed.returncode == 0:
        return

    issue_lines = [
        line.strip()
        for line in (*completed.stdout.splitlines(), *completed.stderr.splitlines())
        if line.strip().startswith("The package `")
    ]
    allowed = set(profile.get("allowed_pip_check_issues", []))
    unexpected = [line for line in issue_lines if line not in allowed]
    counts = [int(match.group(1)) for line in (*completed.stdout.splitlines(), *completed.stderr.splitlines())
              if (match := re.fullmatch(r"Found (\d+) incompatibilit(?:y|ies)", line.strip()))]
    if completed.returncode == 1 and issue_lines and not unexpected and counts == [len(issue_lines)]:
        print("仅发现 Profile 登记的上游锁文件元数据例外；继续执行运行时预检。")
        return
    raise RuntimeError(
        "uv pip check 发现未登记的依赖冲突："
        + ("; ".join(unexpected) if unexpected else "无法解析检查输出")
    )


def install_uv_lock_profile(
    *,
    root: Path,
    profile_id: str,
    profile: dict,
    target_value: str | None,
    python_version: str,
    reuse: bool,
    dry_run: bool,
) -> int:
    """Create and validate one repository-owned uv lock environment."""

    manager = profile["external_manager"]
    if manager.get("kind") != "uv_lock":
        raise RuntimeError(
            f"Profile {profile_id!r} 使用尚未支持的外部环境管理器："
            f"{manager.get('kind')!r}"
        )
    if not target_value:
        raise ValueError(f"Profile {profile_id!r} 使用上游锁环境，必须提供 --target")

    repository = manager.get("repository")
    tag = manager.get("tag")
    expected_commit = manager.get("commit")
    locked_python = python_version or manager.get("python", "3.12")
    if not all((repository, tag, expected_commit)):
        raise RuntimeError(f"Profile {profile_id!r} 缺少 repository、tag 或 commit")

    git = shutil.which("git")
    uv = shutil.which("uv")
    if git is None:
        raise RuntimeError("未找到 git，无法创建上游锁环境")
    if uv is None:
        raise RuntimeError("未找到 uv，无法创建上游锁环境")

    target = Path(target_value).expanduser().resolve()
    target_exists = target.exists()
    if target_exists and not reuse:
        raise RuntimeError(
            f"目标目录已经存在：{target}；为避免覆盖，请更换 --target，"
            "或确认它属于同一 Profile 后显式添加 --reuse"
        )
    if target_exists and not (target / ".git").is_dir():
        raise RuntimeError(f"目标目录不是 Git 仓库，不能复用：{target}")

    print(json.dumps({
        "profile": profile_id,
        "title": profile.get("title"),
        "manager": "uv_lock",
        "target": str(target),
        "repository": repository,
        "tag": tag,
        "expected_commit": expected_commit,
        "python": locked_python,
        "extras": manager.get("extras", []),
        "reuse": reuse,
        "dry_run": dry_run,
    }, ensure_ascii=False, indent=2))

    if not target_exists:
        clone_locked_repository(
            git=git,
            repository=repository,
            tag=tag,
            target=target,
            dry_run=dry_run,
        )

    if not dry_run:
        completed = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=target,
            check=True,
            capture_output=True,
            text=True,
        )
        resolved_commit = completed.stdout.strip()
        if resolved_commit != expected_commit:
            raise RuntimeError(
                f"上游源码版本不匹配：期望 {expected_commit}，实际 {resolved_commit}"
            )

    isolated_env = os.environ.copy()
    isolation = profile.get("isolation", {})
    if isolation.get("require_user_site_disabled", False):
        isolated_env["PYTHONNOUSERSITE"] = "1"
    if isolation.get("require_clean_sys_path", False):
        isolated_env["PYTHONPATH"] = ""
    # Keep HTTP(S) proxies, but avoid HTTPX's optional socksio requirement.
    isolated_env.pop("ALL_PROXY", None)
    isolated_env.pop("all_proxy", None)

    sync_command = [
        uv,
        "sync",
        "--frozen",
        "--all-packages",
        "--python",
        locked_python,
    ]
    for extra in manager.get("extras", []):
        sync_command.extend(["--extra", extra])
    run(sync_command, dry_run=dry_run, cwd=target, env=isolated_env)

    external_python = target / ".venv" / "bin" / "python"
    check_uv_dependencies(
        uv=uv,
        python=external_python,
        profile=profile,
        cwd=target,
        env=isolated_env,
        dry_run=dry_run,
    )
    run(
        [
            str(external_python),
            str(root / "tools" / "environment_preflight.py"),
            "--profile",
            profile_id,
        ],
        dry_run=dry_run,
        cwd=root,
        env=isolated_env,
    )
    print("安装计划已列出；未创建或预检环境。" if dry_run else
          "外部锁环境安装与预检完成；该 Profile 由 Notebook 通过独立进程调用。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="按教程 Profile 创建隔离环境")
    parser.add_argument("--profile", required=True, help="environment-profiles.json 中的 Profile ID")
    parser.add_argument("--env-name", help="Conda Profile 要创建或显式复用的环境名")
    parser.add_argument("--target", help="外部锁 Profile 的安装目录")
    parser.add_argument("--python", default="3.12", help="Conda Python 版本，默认 3.12")
    parser.add_argument("--reuse", action="store_true", help="显式允许向已有环境安装登记依赖")
    parser.add_argument("--no-kernel", action="store_true", help="不注册 Jupyter Kernel")
    parser.add_argument("--dry-run", action="store_true", help="只打印安装计划，不执行命令")
    args = parser.parse_args()

    root = bootstrap_project_root()
    profile = load_environment_profile(root, args.profile)
    external_manager = profile.get("external_manager")
    if external_manager:
        if args.env_name:
            parser.error(f"Profile {args.profile!r} 使用 --target，不使用 --env-name")
        return install_uv_lock_profile(
            root=root,
            profile_id=args.profile,
            profile=profile,
            target_value=args.target,
            python_version=args.python,
            reuse=args.reuse,
            dry_run=args.dry_run,
        )
    if args.target:
        parser.error(f"Conda Profile {args.profile!r} 使用 --env-name，不使用 --target")
    if not args.env_name:
        parser.error(f"Conda Profile {args.profile!r} 必须提供 --env-name")
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

    print("安装计划已列出；未创建或预检环境。" if args.dry_run else
          "环境安装与预检完成。请重启 Notebook Kernel，并选择登记的环境。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
