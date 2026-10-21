"""Build frozen GSM8K workloads for project 88 from the local HF cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset


DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
SOURCE_REVISION = "740312add88f781978c0658806c59bc2815b9866"
BUILDER_VERSION = "gsm8k-stratified-v2"
PROMPT_SUFFIX = "Show your reasoning, then write the final numeric answer after ####."


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache/huggingface/datasets",
    )
    parser.add_argument(
        "--output-workload",
        type=Path,
        default=root / "benchmarks/workloads/distillation_delivery_gsm8k_v1.jsonl",
    )
    parser.add_argument(
        "--output-final",
        type=Path,
        default=root / "benchmarks/workloads/distillation_delivery_gsm8k_final_v1.jsonl",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=root / "benchmarks/workloads/distillation_delivery_gsm8k_v1.manifest.json",
    )
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--train-per-category", type=int, default=32)
    parser.add_argument("--development-per-category", type=int, default=8)
    parser.add_argument("--evaluation-per-category", type=int, default=16)
    parser.add_argument("--final-per-category", type=int, default=16)
    parser.add_argument(
        "--development-offset-per-category",
        type=int,
        default=32,
        help="Keep development rows fixed while the training prefix grows.",
    )
    return parser.parse_args()


def canonical_final_answer(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
    if match is None:
        raise ValueError("GSM8K answer 缺少 #### 数值答案")
    return match.group(1).replace(",", "")


def reasoning_score(answer: str) -> tuple[int, int]:
    rationale = answer.split("####", 1)[0]
    calculation_count = answer.count("<<")
    nonempty_lines = sum(bool(line.strip()) for line in rationale.splitlines())
    return max(calculation_count, nonempty_lines), len(rationale)


def rank_buckets(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ranked = sorted(
        rows,
        key=lambda row: (*reasoning_score(str(row["answer"])), int(row["source_index"])),
    )
    buckets: dict[str, list[dict[str, Any]]] = {}
    for bucket_index in range(4):
        start = len(ranked) * bucket_index // 4
        stop = len(ranked) * (bucket_index + 1) // 4
        buckets[f"reasoning_q{bucket_index + 1}"] = ranked[start:stop]
    return buckets


def select_rows(
    rows: list[dict[str, Any]],
    allocations: list[tuple[str, int]],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    selected = {split: [] for split, _ in allocations}
    for bucket_offset, (category, bucket) in enumerate(rank_buckets(rows).items()):
        shuffled = list(bucket)
        random.Random(seed + bucket_offset).shuffle(shuffled)
        cursor = 0
        for split, count in allocations:
            chosen = shuffled[cursor:cursor + count]
            if len(chosen) != count:
                raise ValueError(f"{category} 不足以分配 {split}={count}")
            for row in chosen:
                selected[split].append({**row, "category": category})
            cursor += count
    for split, _ in allocations:
        random.Random(seed + sum(map(ord, split))).shuffle(selected[split])
    return selected


def select_train_and_development(
    rows: list[dict[str, Any]],
    *,
    train_count: int,
    development_count: int,
    development_offset: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Grow training data around one fixed development slice per category."""

    if min(train_count, development_count, development_offset) < 0:
        raise ValueError("训练/开发规模与 development offset 不能为负")
    selected = {"train": [], "development": []}
    for bucket_offset, (category, bucket) in enumerate(rank_buckets(rows).items()):
        shuffled = list(bucket)
        random.Random(seed + bucket_offset).shuffle(shuffled)
        development = shuffled[
            development_offset:development_offset + development_count
        ]
        train_pool = (
            shuffled[:development_offset]
            + shuffled[development_offset + development_count:]
        )
        train = train_pool[:train_count]
        if len(train) != train_count or len(development) != development_count:
            raise ValueError(f"{category} 不足以分配 train/development")
        selected["train"].extend({**row, "category": category} for row in train)
        selected["development"].extend(
            {**row, "category": category} for row in development
        )
    for split in selected:
        random.Random(seed + sum(map(ord, split))).shuffle(selected[split])
    return selected


def make_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    prompt = f"{str(row['question']).strip()}\n\n{PROMPT_SUFFIX}"
    source_content = json.dumps(
        {"question": row["question"], "answer": row["answer"]},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    record: dict[str, Any] = {
        "record_id": f"gsm8k-{split}-{int(row['source_index']):05d}",
        "split": split,
        "category": row["category"],
        "prompt": prompt,
        "source": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": SOURCE_REVISION,
            "official_split": row["official_split"],
            "source_index": int(row["source_index"]),
            "source_record_sha256": hashlib.sha256(source_content).hexdigest(),
        },
    }
    if split in {"train", "development"}:
        record["reference_answer"] = str(row["answer"]).strip()
    if split in {"development", "evaluation", "final"}:
        record["validator"] = "final_number"
        record["expected_exact"] = [canonical_final_answer(str(row["answer"]))]
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    for field in ("output_workload", "output_final", "output_manifest"):
        path = getattr(args, field)
        if not path.is_absolute():
            setattr(args, field, project_root / path)
    source_root = (
        args.cache_root / "openai___gsm8k/main/0.0.0" / SOURCE_REVISION
    )
    source_paths = {
        "train": source_root / "gsm8k-train.arrow",
        "test": source_root / "gsm8k-test.arrow",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "本地 GSM8K 缓存不完整；先按数据集卡下载固定 revision：" + ", ".join(missing)
        )

    source_rows: dict[str, list[dict[str, Any]]] = {}
    for split, path in source_paths.items():
        dataset = Dataset.from_file(str(path))
        source_rows[split] = [
            {**dict(row), "official_split": split, "source_index": index}
            for index, row in enumerate(dataset)
        ]

    train_side = select_train_and_development(
        source_rows["train"],
        train_count=args.train_per_category,
        development_count=args.development_per_category,
        development_offset=args.development_offset_per_category,
        seed=args.seed,
    )
    test_side = select_rows(
        source_rows["test"],
        [
            ("evaluation", args.evaluation_per_category),
            ("final", args.final_per_category),
        ],
        seed=args.seed + 1000,
    )
    workload_records = [
        make_record(row, split)
        for split in ("train", "development", "evaluation")
        for row in train_side.get(split, test_side.get(split, []))
    ]
    final_records = [make_record(row, "final") for row in test_side["final"]]

    all_records = workload_records + final_records
    ids = [str(record["record_id"]) for record in all_records]
    prompts = [str(record["prompt"]) for record in all_records]
    source_keys = [
        (
            str(record["source"]["official_split"]),
            int(record["source"]["source_index"]),
        )
        for record in all_records
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError("生成结果存在重复 record_id")
    if len(prompts) != len(set(prompts)):
        raise RuntimeError("生成结果存在跨切分重复 prompt")
    if len(source_keys) != len(set(source_keys)):
        raise RuntimeError("生成结果存在跨切分 source 泄漏")

    workload_sha = write_jsonl(args.output_workload, workload_records)
    final_sha = write_jsonl(args.output_final, final_records)
    split_counts = Counter(str(record["split"]) for record in all_records)
    category_counts = {
        split: dict(sorted(Counter(
            str(record["category"])
            for record in all_records
            if record["split"] == split
        ).items()))
        for split in split_counts
    }
    manifest = {
        "schema_version": "distillation-workload-manifest/v1",
        "builder_version": BUILDER_VERSION,
        "seed": args.seed,
        "source": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": SOURCE_REVISION,
            "files": {
                split: {
                    "cache_relative_path": str(path.relative_to(args.cache_root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for split, path in source_paths.items()
            },
        },
        "outputs": {
            "workload": {
                "path": str(args.output_workload.relative_to(project_root)),
                "sha256": workload_sha,
            },
            "final": {
                "path": str(args.output_final.relative_to(project_root)),
                "sha256": final_sha,
            },
        },
        "stratification": {
            "method": "rank quartiles by max(calculation markers, rationale lines), then rationale length",
            "categories": ["reasoning_q1", "reasoning_q2", "reasoning_q3", "reasoning_q4"],
            "allocations_per_category": {
                "train": args.train_per_category,
                "development": args.development_per_category,
                "evaluation": args.evaluation_per_category,
                "final": args.final_per_category,
            },
            "development_offset_per_category": args.development_offset_per_category,
        },
        "quality": {
            "split_counts": dict(sorted(split_counts.items())),
            "category_counts": category_counts,
            "unique_record_ids": len(set(ids)),
            "unique_prompts": len(set(prompts)),
            "unique_source_rows": len(set(source_keys)),
            "cross_split_prompt_overlap": 0,
            "cross_split_source_overlap": 0,
        },
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["quality"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
