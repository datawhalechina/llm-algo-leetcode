"""Compatibility schema for the training-memory projects 73, 76 and 75.

The original result files are intentionally left untouched.  This module
normalizes their different legacy shapes into one portable representation so
that old measurements remain reproducible while later runs can be compared
with the same fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .result_migration_io import save_migration_report
except ImportError:  # Direct execution from tools/.
    from result_migration_io import save_migration_report


SCHEMA_VERSION = "training-memory-benchmark/v1"


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _candidate(name: str, raw: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    peak = raw.get("peak_memory_mb", raw.get("peak_mem_mb"))
    throughput = raw.get("samples_per_s")
    baseline_peak = baseline.get("peak_memory_mb", baseline.get("peak_mem_mb")) if baseline else None
    baseline_throughput = baseline.get("samples_per_s") if baseline else None

    metrics: dict[str, Any] = {
        "step_time_ms": raw.get("step_time_ms"),
        "samples_per_s": throughput,
        "peak_memory_mb": peak,
        "peak_reserved_mb": raw.get("peak_reserved_mb"),
        "memory_saving_mb": (
            round(baseline_peak - peak, 6)
            if _number(baseline_peak) is not None and _number(peak) is not None
            else None
        ),
        "throughput_ratio": (
            round(throughput / baseline_throughput, 6)
            if _number(throughput) is not None and _number(baseline_throughput) not in (None, 0)
            else None
        ),
        "oom": raw.get("status") == "oom" or raw.get("oom"),
    }
    quality = {
        "loss": raw.get("loss"),
        "eval_loss": raw.get("eval_loss"),
        "val_loss": raw.get("val_loss", raw.get("eval_loss")),
        "quality_status": raw.get("quality_status"),
    }
    return {
        "name": name,
        "status": raw.get("status", "ok"),
        "metrics": metrics,
        "quality": quality,
        "legacy_fields": dict(raw),
    }


def _legacy_candidates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("candidates"), list):
        raw_candidates = [item for item in report["candidates"] if isinstance(item, Mapping)]
    else:
        raw_candidates = []
        for name in ("baseline", "tuned"):
            value = report.get(name)
            if isinstance(value, Mapping):
                raw_candidates.append({"name": name, **value})

    baseline = next(
        (item for item in raw_candidates if item.get("name") == "baseline"),
        None,
    )
    return [
        _candidate(str(item.get("name", f"strategy_{index}")), item, baseline)
        for index, item in enumerate(raw_candidates)
    ]


def normalize_report(report: Mapping[str, Any], *, source: str | None = None) -> dict[str, Any]:
    """Normalize one legacy 73/76/75 report without changing its values."""

    config = report.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    candidates = _legacy_candidates(report)
    baseline = next((item for item in candidates if item["name"] == "baseline"), None)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}

    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": Path(source).stem.split("_")[0] if source else None,
        "stage": report.get("stage"),
        "config": {
            "model": config.get("model_id", config.get("model")),
            "device": config.get("device"),
            "torch": config.get("torch"),
            "cuda": config.get("torch_cuda", config.get("cuda")),
            "dtype": config.get("amp_dtype", config.get("dtype")),
            "batch_size": config.get("batch_size"),
            "seq_len": config.get("seq_len"),
            "warmup": config.get("warmup"),
            "iters": config.get("iters"),
            "workload": config.get("workload"),
        },
        "budget": report.get("budget", {}),
        "quality_floor": report.get("quality_floor", {}),
        "candidates": candidates,
        "summary": dict(summary),
        "decision": report.get("decision", {"decision": "not_evaluated"}),
        "legacy_source": source,
    }
    if report.get("source_baseline") is not None:
        normalized["source_baseline"] = report["source_baseline"]
    return normalized


def normalize_file(
    source: str | Path, output: str | Path | None = None, *, dry_run: bool = False,
) -> Path:
    """Create a legacy sidecar, preserving raw input and conflicting outputs."""
    source_path = Path(source)
    report = json.loads(source_path.read_text(encoding="utf-8"))
    output_path = Path(output) if output else source_path.with_name(f"{source_path.stem}_v1.json")
    return save_migration_report(
        source_path, output_path, normalize_report(report, source=str(source_path)),
        dry_run=dry_run,
    )
