#!/usr/bin/env python3
"""Sync navigation pages and topic notebooks into docs/.

This script mirrors each Part's intro page and group pages into docs/.
It does not touch notebook part mirrors, which remain the job of
tools/convert_notebook.py. Cross-Part topic notebooks are also rendered to
Markdown here so topic pages can link to them from the docs site.
"""

from __future__ import annotations

import json
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
        "2_10.md",
    ],
    "03_Triton_Kernels": ["intro.md", "3_1.md", "3_2.md", "3_3.md", "3_4.md", "3_5.md"],
    "04_CUDA_and_System_Optimization": ["intro.md", "4_1.md", "4_2.md", "4_3.md", "4_4.md"],
    "05_Integrated_Projects_and_Decisions": [
        "intro.md",
        "01_Inference_Service_Integrated_Project.md",
        "02_Post_Training_Delivery_Project.md",
    ],
    "topic_discussion": [
        "intro.md",
        "backpropagation_training_mechanism/intro.md",
        "backpropagation_training_mechanism/casebook.md",
        "backpropagation_training_mechanism/walkthrough.md",
        "performance_analysis/intro.md",
        "performance_analysis/casebook.md",
        "performance_analysis/walkthrough.md",
        "inference_optimization/intro.md",
        "inference_optimization/casebook.md",
        "inference_optimization/walkthrough.md",
        "post_training_optimization/intro.md",
        "post_training_optimization/casebook.md",
        "post_training_optimization/walkthrough.md",
        "post_training_optimization/01_sft_data_and_loss.md",
        "post_training_optimization/02_lora_peft_design.md",
        "post_training_optimization/03_training_control.md",
        "post_training_optimization/04_end_to_end_experiment.md",
        "post_training_optimization/05_dpo_and_preference_optimization.md",
        "post_training_optimization/06_rlhf_and_ppo_system_cost.md",
        "post_training_optimization/07_grpo_and_groupwise_alignment.md",
        "post_training_optimization/08_rl_engineering_and_efficiency.md",
        "post_training_optimization/09_preference_data_and_project_decision.md",
        "communication_parallelism/intro.md",
        "communication_parallelism/casebook.md",
        "communication_parallelism/walkthrough.md",
        "memory_performance_tuning/intro.md",
        "llm_architecture_evolution/intro.md",
        "llm_architecture_evolution/casebook.md",
        "llm_architecture_evolution/walkthrough.md",
        # post_training_optimization entries are listed above, including the
        # SFT foundation pages that replaced the old fine_tuning_training route.
        "quantization_compression/intro.md",
        "quantization_compression/casebook.md",
        "quantization_compression/walkthrough.md",
        "operator_optimization/intro.md",
        "operator_optimization/01_why_operator_optimization_matters.md",
        "operator_optimization/02_kernel_semantics_and_memory.md",
        "operator_optimization/03_fusion_and_kernel_composition.md",
        "operator_optimization/04_cuda_execution_and_hardware_constraints.md",
        "operator_optimization/05_cost_model_and_profiling.md",
        "operator_optimization/06_benchmark_and_project_validation.md",
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
        if part == "topic_discussion":
            files = sorted(
                path.relative_to(src_dir).as_posix()
                for path in src_dir.rglob("*.md")
            )
            notebook_mirrors = {
                (dst_dir / path.relative_to(src_dir)).with_suffix(".md")
                for path in src_dir.rglob("*.ipynb")
                if ".ipynb_checkpoints" not in path.parts
            }
            source_files = {dst_dir / name for name in files} | notebook_mirrors
            for stale in dst_dir.rglob("*.md"):
                if stale not in source_files:
                    stale.unlink()
        for name in files:
            src = src_dir / name
            if not src.exists():
                continue
            dst = dst_dir / name
            text = src.read_text(encoding="utf-8")
            mirrored = text.replace(".ipynb)", ".md)")
            # Source topic pages may link to docs/guide.md. Once mirrored
            # inside docs/topic_discussion, guide.md is already at the docs
            # root, so remove the extra docs/ segment in the mirror only.
            mirrored = mirrored.replace("../../docs/guide.md", "../../guide.md")
            mirrored = mirrored.replace("../docs/guide.md", "../guide.md")
            mirrored = mirrored.replace("../docs/", "../")
            mirrored = mirrored.replace(
                "../../tools/",
                "https://github.com/datawhalechina/llm-algo-leetcode/blob/main/tools/",
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(mirrored, encoding="utf-8")
            copied += 1

        if part == "topic_discussion":
            for source in sorted(src_dir.rglob("*.ipynb")):
                if ".ipynb_checkpoints" in source.parts:
                    continue
                relative = source.relative_to(src_dir)
                destination = (dst_dir / relative).with_suffix(".md")
                notebook = json.loads(source.read_text(encoding="utf-8"))
                rendered = []
                for cell in notebook.get("cells", []):
                    cell_source = cell.get("source", "")
                    if isinstance(cell_source, list):
                        cell_source = "".join(cell_source)
                    if cell.get("cell_type") == "markdown" and cell_source.strip():
                        rendered.append(cell_source)
                    elif cell.get("cell_type") == "code" and cell_source.strip():
                        rendered.append(f"\n```python\n{cell_source}\n```\n")
                text = "\n\n".join(rendered)
                text = text.replace(".ipynb)", ".md)")
                text = text.replace("../../docs/guide.md", "../../guide.md")
                text = text.replace("../docs/guide.md", "../guide.md")
                text = text.replace("../docs/", "../")
                text = text.replace(
                    "../../tools/",
                    "https://github.com/datawhalechina/llm-algo-leetcode/blob/main/tools/",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")
                copied += 1
    print(f"Synced {copied} navigation files into docs/")


def main() -> int:
    sync_navigation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
