#!/usr/bin/env python3
"""Sync navigation pages from source parts into docs/.

This script mirrors each Part's intro page and group pages into docs/.
It does not touch notebook part mirrors, which remain the job of
tools/convert_notebook.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


NAV_FILES = {
    "00_Prerequisites": ["intro.md", "0A.md", "0B.md", "0C.md", "0D.md", "0E.md"],
    "01_Hardware_Math_and_Systems": ["intro.md", "1A.md", "1B.md", "1C.md", "1D.md", "1E.md"],
    "02_PyTorch_Algorithms": [
        "intro.md",
        "2_1.md",
        "2_2.md",
        "2_3.md",
        "2_4.md",
        "2_5.md",
        "2_6.md",
        "2_7.md",
        "2_8.md",
        "2_9.md",
    ],
    "03_Triton_Kernels": ["intro.md", "3_1.md", "3_2.md", "3_3.md", "3_4.md", "3_5.md"],
    "04_CUDA_and_System_Optimization": ["intro.md", "4_1.md", "4_2.md", "4_3.md", "4_4.md"],
    "topic_discussion": [
        "intro.md",
        "profiling/intro.md",
        "profiling/casebook.md",
        "compiler_graph_optimization/intro.md",
        "compiler_graph_optimization/casebook.md",
        "inference_optimization/intro.md",
        "inference_optimization/casebook.md",
        "communication_parallel/intro.md",
        "communication_parallel/casebook.md",
        "memory_performance_tuning/intro.md",
        "memory_performance_tuning/casebook.md",
    ],
    "team_study": [
        "intro.md",
        "part2_l1_202606/intro.md",
        "part2_l1_202606/group_topic_1.md",
        "part2_l1_202606/group_topic_2.md",
        "part2_l1_202607/intro.md",
        "part2_l2_202607/intro.md",
    ],
}


def sync_navigation() -> None:
    copied = 0
    for part, files in NAV_FILES.items():
        src_dir = ROOT / part
        dst_dir = DOCS / part
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = src_dir / name
            if not src.exists():
                continue
            dst = dst_dir / name
            text = src.read_text(encoding="utf-8")
            dst.write_text(text.replace(".ipynb)", ".md)"), encoding="utf-8")
            copied += 1
    print(f"Synced {copied} navigation files into docs/")


def main() -> int:
    sync_navigation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
