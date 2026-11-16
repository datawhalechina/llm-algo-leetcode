#!/usr/bin/env python3
"""Preview the historical Part02 question-stub migration (writes are frozen).

The original first-pass rule replaced question-zone function bodies so
validation could classify incomplete exercises as expected failures. It is
retained for read-only inspection: its historical text-based selection does
not establish which cells are safe to edit in current notebooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "02_PyTorch_Algorithms"
TARGETS = [
    "61_Model_Architecture_Exploration.ipynb",
    "62_Instruction_Fine_Tuning_Project.ipynb",
    "63_LoRA_Variants_Benchmark.ipynb",
    "64_SFT_Data_Quality_Project.ipynb",
    "68_Speculative_Decoding_Benchmark.ipynb",
    "69_Prefix_Caching_Benchmark.ipynb",
    "70_Serving_Scheduler_Benchmark.ipynb",
    "76_Activation_Checkpoint_Offload_Benchmark.ipynb",
    "80_MoE_Expert_Parallel_Benchmark.ipynb",
    "81_Distributed_Inference_Project.ipynb",
    "84_DPO_Preference_Project.ipynb",
    "85_GRPO_Groupwise_Alignment_Project.ipynb",
    "86_DPO_Online_Benchmark.ipynb",
]


def _stub_functions(source: str) -> str:
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("def "):
            out.append(line)
            i += 1
            continue

        signature: list[str] = [line]
        i += 1
        if not line.rstrip().endswith(":"):
            while i < len(lines):
                signature.append(lines[i])
                if lines[i].rstrip().endswith(":"):
                    i += 1
                    break
                i += 1

        while i < len(lines):
            current = lines[i]
            if current.startswith("def "):
                break
            if current and not current.startswith((" ", "\t")):
                break
            i += 1

        out.extend(signature)
        out.append('    raise NotImplementedError("请先完成 TODO 代码！")')
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def _rewrite_notebook(path: Path, *, dry_run: bool = True) -> bool:
    """Preview the old rewrite rule without saving any notebook changes."""
    if not dry_run:
        raise ValueError("历史题目改写器已冻结写入；请按当前单元标签逐页修复。")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    in_question_zone = False

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "markdown":
            text = "".join(cell.get("source", []))
            if "## 参考代码与解析" in text:
                break
            continue

        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if "import" in source and not in_question_zone:
            in_question_zone = True
        if not in_question_zone:
            continue
        if "# 测试你的实现" in source:
            continue

        new_source = _stub_functions(source)
        if new_source != source:
            cell["source"] = new_source.splitlines(keepends=True)
            changed = True

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", choices=TARGETS, action="append", help="Preview a historical target by filename")
    parser.add_argument("--dry-run", action="store_true", help="Preview only (already the default)")
    parser.add_argument("--write", action="store_true", help="Unavailable: historical writes are frozen")
    args = parser.parse_args()
    if args.write:
        parser.error("历史题目改写器已冻结写入；预览结果须按当前单元标签人工复核。")
    print("[preview only] Historical text rules; review current cell tags before any manual repair.")
    changed_files = []
    for name in args.file or TARGETS:
        path = NOTEBOOK_DIR / name
        if _rewrite_notebook(path):
            changed_files.append(name)

    print(f"would change {len(changed_files)} notebooks; wrote 0")
    for name in changed_files:
        print(name)


if __name__ == "__main__":
    main()
