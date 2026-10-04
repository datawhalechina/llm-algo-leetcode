"""Portable evidence contract for the model-architecture project chain.

Projects 60, 78, 71 and 77 produce local architecture evidence. Project 61
consumes references to those records and produces an audit decision; it must
not copy local benchmark metrics into a second source of truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "architecture-evaluation/v1"
PRODUCER_IDS = {
    "ARCH-BLOCK-STABILITY",
    "ARCH-ATTENTION-ACCESS",
    "ARCH-MLA-KV",
    "ARCH-LONG-SEQUENCE-MEMORY",
}
AUDIT_ID = "ARCH-MODEL-AUDIT"
COMPARISON_MODES = {"controlled", "observational"}
EVIDENCE_LEVELS = {
    "theoretical",
    "cpu_simulation",
    "gpu_smoke",
    "real_benchmark",
    "profiler_trace",
    "not_recorded",
}
STATUSES = {"ok", "failed", "blocked", "skipped"}
DECISIONS = {"accept", "tune", "reject", "pending"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _failure(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"reason": str(value)}


def _decision(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
        status = result.get("status", result.get("decision", "pending"))
        result["status"] = status
    elif isinstance(value, str) and value:
        result = {"status": value}
    else:
        result = {"status": "pending"}
    if result["status"] not in DECISIONS:
        raise ValueError(f"decision status must be one of {sorted(DECISIONS)}")
    return result


def _base_record(
    *,
    semantic_id: str,
    record_type: str,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    workload: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    cost: Mapping[str, Any],
    evidence_level: str,
    status: str,
    failure: Any,
    decision: Any,
) -> dict[str, Any]:
    if not source:
        raise ValueError("source must identify the producer notebook or raw result")
    if not workload:
        raise ValueError("workload must not be empty")
    if not baseline:
        raise ValueError("baseline must not be empty")
    if not candidates:
        raise ValueError("candidates must contain at least one architecture candidate")
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError(f"evidence_level must be one of {sorted(EVIDENCE_LEVELS)}")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    normalized_failure = _failure(failure)
    if status == "failed" and not normalized_failure:
        raise ValueError("failed records must describe the failure")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "semantic_id": semantic_id,
        "source": dict(source),
        "runtime": dict(runtime),
        "workload": dict(workload),
        "baseline": dict(baseline),
        "candidates": [dict(candidate) for candidate in candidates],
        "quality": dict(quality),
        "cost": dict(cost),
        "status": status,
        "failure": normalized_failure,
        "evidence_level": evidence_level,
        "decision": _decision(decision),
    }


def make_producer_record(
    *,
    semantic_id: str,
    comparison_mode: str,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    workload: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    cost: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    evidence_level: str,
    status: str = "ok",
    failure: Any = None,
    decision: Any = None,
) -> dict[str, Any]:
    """Create a local evidence record while preserving mechanism fields."""

    if semantic_id not in PRODUCER_IDS:
        raise ValueError(f"producer semantic_id must be one of {sorted(PRODUCER_IDS)}")
    if comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"comparison_mode must be one of {sorted(COMPARISON_MODES)}")
    if not mechanism:
        raise ValueError("mechanism fields must not be empty")
    record = _base_record(
        semantic_id=semantic_id,
        record_type="architecture_evidence",
        source=source,
        runtime=runtime,
        workload=workload,
        baseline=baseline,
        candidates=candidates,
        quality=quality,
        cost=cost,
        evidence_level=evidence_level,
        status=status,
        failure=failure,
        decision=decision,
    )
    record["role"] = "producer"
    record["comparison_mode"] = comparison_mode
    record["mechanism"] = dict(mechanism)
    return record


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest used by audit references."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_evidence_reference(
    *,
    semantic_id: str,
    result_path: str | Path,
    result_hash: str | None = None,
) -> dict[str, Any]:
    """Reference one producer result without copying its metrics."""

    if semantic_id not in PRODUCER_IDS:
        raise ValueError(f"reference semantic_id must be one of {sorted(PRODUCER_IDS)}")
    path = Path(result_path)
    digest = result_hash or (sha256_file(path) if path.is_file() else None)
    if not digest:
        raise ValueError("result_hash is required when result_path is not readable")
    return {
        "semantic_id": semantic_id,
        "result_path": str(path),
        "sha256": digest,
    }


def verify_evidence_reference(reference: Mapping[str, Any]) -> list[str]:
    """Verify path, digest and producer identity before an audit consumes it."""

    errors: list[str] = []
    semantic_id = reference.get("semantic_id")
    if semantic_id not in PRODUCER_IDS:
        errors.append("reference semantic_id is not a registered producer")
    result_path = reference.get("result_path")
    if not result_path:
        return [*errors, "reference result_path is missing"]
    path = Path(str(result_path))
    if not path.is_file():
        return [*errors, f"reference result_path does not exist: {path}"]
    expected_hash = reference.get("sha256")
    if not expected_hash:
        errors.append("reference sha256 is missing")
    elif sha256_file(path) != expected_hash:
        errors.append("reference sha256 does not match the result file")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [*errors, f"reference result is not readable JSON: {exc}"]
    if record.get("role") != "producer":
        errors.append("reference result must have producer role")
    if record.get("semantic_id") != semantic_id:
        errors.append("reference semantic_id does not match the result record")
    errors.extend(f"producer record: {error}" for error in validate_record(record))
    return errors


def decide_audit(
    *,
    config_decision: Mapping[str, Any],
    required_semantic_id: str | None,
    producer_records: Mapping[str, Mapping[str, Any]],
    reference_issues: Sequence[str] = (),
    external_required: bool = False,
) -> dict[str, str]:
    """Apply the shared gate used by project 61's aggregate decision."""

    config_status = config_decision.get("status", config_decision.get("decision", "pending"))
    producer_decisions = {
        semantic_id: _mapping(record.get("decision")).get("status", "pending")
        for semantic_id, record in producer_records.items()
    }
    required_record = producer_records.get(required_semantic_id) if required_semantic_id else None
    required_decision = producer_decisions.get(required_semantic_id)
    producer_failed = any(record.get("status") != "ok" for record in producer_records.values())
    if config_status == "reject" or producer_failed or required_decision == "reject":
        return {"status": "reject", "reason": "config audit or producer evidence failed"}
    if (
        reference_issues
        or external_required
        or (required_semantic_id is not None and required_record is None)
        or (required_semantic_id is not None and required_decision != "accept")
    ):
        return {"status": "tune", "reason": "required producer evidence is missing or not accepted"}
    return {"status": "accept", "reason": "config change and required producer evidence are accepted"}


def make_audit_record(
    *,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    workload: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    cost: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    evidence_level: str,
    status: str = "ok",
    failure: Any = None,
    decision: Any = None,
) -> dict[str, Any]:
    """Create project-61's aggregate audit record from producer references."""

    if not references:
        raise ValueError("audit records must reference at least one producer result")
    normalized_references = [dict(reference) for reference in references]
    for reference in normalized_references:
        if reference.get("semantic_id") not in PRODUCER_IDS:
            raise ValueError("audit reference has an unknown producer semantic_id")
        if not reference.get("result_path") or not reference.get("sha256"):
            raise ValueError("audit references require result_path and sha256")
        reference_errors = verify_evidence_reference(reference)
        if reference_errors:
            raise ValueError("; ".join(reference_errors))
    if not audit:
        raise ValueError("audit fields must not be empty")
    record = _base_record(
        semantic_id=AUDIT_ID,
        record_type="architecture_audit",
        source=source,
        runtime=runtime,
        workload=workload,
        baseline=baseline,
        candidates=candidates,
        quality=quality,
        cost=cost,
        evidence_level=evidence_level,
        status=status,
        failure=failure,
        decision=decision,
    )
    record["role"] = "audit"
    record["comparison_mode"] = "aggregate"
    record["references"] = normalized_references
    record["audit"] = dict(audit)
    return record


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Return contract errors without rejecting project-specific fields."""

    errors: list[str] = []
    required = {
        "schema_version",
        "record_type",
        "semantic_id",
        "role",
        "source",
        "runtime",
        "workload",
        "baseline",
        "candidates",
        "quality",
        "cost",
        "status",
        "failure",
        "evidence_level",
        "decision",
    }
    missing = required - set(record)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    role = record.get("role")
    semantic_id = record.get("semantic_id")
    if role == "producer":
        if semantic_id not in PRODUCER_IDS:
            errors.append("producer semantic_id is not registered")
        if record.get("comparison_mode") not in COMPARISON_MODES:
            errors.append("producer comparison_mode is invalid")
        if not isinstance(record.get("mechanism"), Mapping):
            errors.append("producer mechanism must be a mapping")
    elif role == "audit":
        if semantic_id != AUDIT_ID:
            errors.append(f"audit semantic_id must be {AUDIT_ID}")
        if not isinstance(record.get("references"), list) or not record.get("references"):
            errors.append("audit references must be a non-empty list")
        if not isinstance(record.get("audit"), Mapping):
            errors.append("audit fields must be a mapping")
    else:
        errors.append("role must be producer or audit")
    for field in ("source", "runtime", "workload", "baseline", "quality", "cost", "failure", "decision"):
        if field in record and not isinstance(record[field], Mapping):
            errors.append(f"{field} must be a mapping")
    if "candidates" in record and not isinstance(record["candidates"], list):
        errors.append("candidates must be a list")
    if record.get("evidence_level") not in EVIDENCE_LEVELS:
        errors.append("evidence_level is invalid")
    if record.get("status") not in STATUSES:
        errors.append("status is invalid")
    return errors


def save_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Persist a validated companion record without replacing raw evidence."""

    errors = validate_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"companion result already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
