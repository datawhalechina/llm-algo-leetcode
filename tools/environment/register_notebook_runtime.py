"""Register this repository as an import root for tutorial Notebook kernels.

The tutorial keeps reusable runtime helpers in ``tools/``.  JupyterLab started
from the repository root can import them directly, while VS Code commonly sets
the kernel working directory to the Notebook directory.  A small ``.pth`` file
in the active environment makes both entry points resolve the same repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import sysconfig
from pathlib import Path


PTH_FILENAME = "llm_algo_leetcode_repo.pth"


def repository_root() -> Path:
    """Return the repository containing this registration script."""

    return Path(__file__).resolve().parents[2]


def registration_path(site_packages: str | Path | None = None) -> Path:
    """Return the target ``.pth`` path for the active Python environment."""

    directory = Path(site_packages) if site_packages else Path(sysconfig.get_path("purelib"))
    return directory.resolve() / PTH_FILENAME


def registered_roots(path: Path) -> tuple[Path, ...]:
    """Read non-empty repository roots from an existing registration file."""

    if not path.exists():
        return ()
    return tuple(
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def write_registration(root: Path, path: Path) -> None:
    """Write one explicit repository root without altering other environments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{root.resolve()}\n", encoding="utf-8")


def verify_subdirectory_import(root: Path) -> dict[str, object]:
    """Verify imports from the working directory used by Part 02 notebooks."""

    notebook_directory = root / "02_PyTorch_Algorithms"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tools.notebook_runtime import package_version; "
                "from tools.project_runtime import resolve_project_root; "
                "from tools.quantization_runtime import state_bytes; "
                "print(resolve_project_root())"
            ),
        ],
        cwd=notebook_directory,
        capture_output=True,
        text=True,
    )
    return {
        "working_directory": str(notebook_directory),
        "importable": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="为当前 Python/Notebook Kernel 注册教程仓库运行时")
    parser.add_argument("--check", action="store_true", help="只检查现有注册，不写入 .pth 文件")
    parser.add_argument("--site-packages", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = repository_root()
    path = registration_path(args.site_packages)
    before = registered_roots(path)
    if not args.check:
        write_registration(root, path)

    registered = root.resolve() in registered_roots(path)
    verification = verify_subdirectory_import(root) if args.site_packages is None else None
    report = {
        "python": sys.executable,
        "repository_root": str(root),
        "registration_file": str(path),
        "registered": registered,
        "previous_roots": [str(item) for item in before],
        "subdirectory_import": verification,
        "kernel_restart_required": not args.check,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not registered:
        return 2
    if verification is not None and not verification["importable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
