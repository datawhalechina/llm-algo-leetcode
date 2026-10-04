"""Aggregate mechanism benchmark JSON files into one project evidence record.

The tool deliberately does not turn a collective smoke result into a full
model benchmark.  Mixed evidence is kept visible so project notebooks can
decide whether a strategy is accepted, needs tuning, or still needs a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from distributed_result_schema import make_record
except ModuleNotFoundError:  # Support both script and package-style execution.
    from tools.distributed_result_schema import make_record


REQUIRED_FIELDS = {"workload", "hardware", "metrics", "evidence_level", "failure", "decision"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=("79", "80", "81", "PAR-MODEL-INTERNAL"), required=True)
    parser.add_argument(
        "--role",
        choices=("training_state_parallel", "model_internal_parallel", "moe_expert_parallel", "distributed_serving"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", action="append", default=[], dest="inputs")
    return parser.parse_args()


def _read(path_text: str) -> dict[str, object]:
    path = Path(path_text)
    result = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS - set(result)
    if missing:
        raise ValueError(f"{path} 缺少统一证据字段：{sorted(missing)}")
    return {"path": str(path), **result}


def _decision(records: list[dict[str, object]]) -> tuple[str, str | None]:
    if not records:
        return "not_started", "尚未提供任何机制或后端结果 JSON。"
    failures = [record for record in records if record["failure"]]
    if failures:
        return "tune", f"{len(failures)} 组运行未完成；先修复环境或配置后再比较。"
    smoke_only = [record for record in records if record["evidence_level"] == "real_multi_gpu_smoke"]
    if smoke_only:
        return "tune", "collective 路径可用，但至少一组仍只有 smoke 证据，不能输出完整策略结论。"
    return "ready_for_review", None


def main() -> int:
    args = _args()
    records: list[dict[str, object]] = []
    unreadable: list[str] = []
    for input_path in args.inputs:
        try:
            records.append(_read(input_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            unreadable.append(f"{input_path}: {type(exc).__name__}: {exc}")

    decision, failure = _decision(records)
    if unreadable:
        decision = "tune"
        failure = "; ".join(unreadable)
    default_roles = {
        "79": "training_state_parallel",
        "80": "moe_expert_parallel",
        "81": "distributed_serving",
        "PAR-MODEL-INTERNAL": "model_internal_parallel",
    }
    result = make_record(
        project=args.project,
        role=args.role or default_roles[args.project],
        workload={"source_result_count": len(records)},
        hardware={"details": "see_each_source_result"},
        metrics={"source_count": len(records)},
        evidence_level="aggregated_mechanism_evidence",
        failure=failure,
        decision={"decision": decision, "reason": failure},
        sources=records,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
