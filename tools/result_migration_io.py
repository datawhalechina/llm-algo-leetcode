"""Write derived migration reports without replacing raw or existing evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_migration_report(
    source: Path, target: Path, report: Any, *, dry_run: bool = False,
) -> Path:
    """Preserve inputs; accept identical output bytes and reject conflicts.

    Preview validates paths and existing content without creating directories.
    Exclusive creation also protects against an output appearing after preview.
    """
    if source.resolve() == target.resolve() or (
        target.exists() and source.samefile(target)
    ):
        raise ValueError(f"转换输出不能是原始输入：{target}")
    if target.is_symlink():
        raise ValueError(f"转换输出不能是符号链接：{target}")
    content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    def check_existing() -> None:
        """Allow a byte-identical rerun, never replace different evidence."""
        if target.read_bytes() != content:
            raise FileExistsError(f"已有不同内容的转换结果，请另选输出路径：{target}")

    if target.exists():
        check_existing()
        return target
    if dry_run:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        check_existing()
    return target
