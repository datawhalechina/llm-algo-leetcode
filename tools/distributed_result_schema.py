"""Portable evidence contract for distributed training and inference projects.

The four project roles share one outer envelope while keeping role-specific
measurements under ``metrics`` and ``artifacts``. Raw backend or mechanism JSON
remains the source of truth and can be attached through ``sources``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "distributed-benchmark/v1"
PROJECT_ROLES = (
    "training_state_parallel",
    "model_internal_parallel",
    "moe_expert_parallel",
    "distributed_serving",
)
REQUIRED_FIELDS = {
    "schema_version",
    "project",
    "role",
    "workload",
    "hardware",
    "metrics",
    "evidence_level",
    "failure",
    "decision",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _decision(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value:
        return {"decision": value}
    return {"decision": "not_evaluated"}


def _failure(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {"reason": str(value)}


def make_record(
    *,
    project: str,
    role: str,
    workload: Mapping[str, Any],
    hardware: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evidence_level: str,
    failure: Any = None,
    decision: Any = None,
    artifacts: Mapping[str, Any] | None = None,
    sources: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create one distributed evidence record without flattening source data."""

    if role not in PROJECT_ROLES:
        raise ValueError(f"unknown distributed project role: {role}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "project": str(project),
        "role": role,
        "workload": dict(workload),
        "hardware": dict(hardware),
        "metrics": dict(metrics),
        "artifacts": dict(artifacts or {}),
        "evidence_level": evidence_level,
        "failure": _failure(failure),
        "decision": _decision(decision),
    }
    if sources is not None:
        record["sources"] = [dict(source) for source in sources]
    return record


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Return contract errors while allowing role-specific metric fields."""

    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if record.get("role") not in PROJECT_ROLES:
        errors.append(f"role must be one of {PROJECT_ROLES}")
    for field in ("workload", "hardware", "metrics", "decision"):
        if field in record and not isinstance(record[field], Mapping):
            errors.append(f"{field} must be a mapping")
    return errors


def save_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Save a validated companion record without replacing raw evidence."""

    errors = validate_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
