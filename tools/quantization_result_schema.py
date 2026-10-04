"""Portable evidence contracts for quantization projects 82, 66, 67 and 83.

The notebooks keep their raw result JSON so learners can inspect backend-specific
fields.  This module creates a compact companion envelope for cross-project
decisions: artifact readiness (82), a matched floating baseline (66), quantized
deployment evidence (67), and profile-specific serving evidence (83).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "quantization-evidence/v1"
EVIDENCE_STAGES = ("artifact_gate", "floating_baseline", "quantized_deployment", "serving_profile")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _failure(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {"status": value.get("status", "recorded"), "reason": value.get("reason")}
    return {"status": "recorded" if value else "none", "reason": value}


def _normalized_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an inference companion record when one is embedded in raw JSON."""

    nested = payload.get("normalized_result")
    return _mapping(nested) or dict(payload)


def _decision(value: Any) -> dict[str, Any]:
    """Keep project decisions structured without inventing an acceptance result."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value:
        return {"decision": value}
    return {"decision": "not_evaluated"}


def artifact_gate_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize raw project-82 output without discarding its manifest details."""

    manifest = _mapping(payload.get("manifest"))
    quality = _mapping(payload.get("quality"))
    compatibility = _mapping(payload.get("compatibility"))
    gate = _mapping(payload.get("gate"))
    model = _mapping(manifest.get("model"))
    quantization = _mapping(manifest.get("quantization"))
    evidence = _mapping(manifest.get("evidence"))
    loader_failure = compatibility.get("failure")
    gate_issues = gate.get("issues", [])
    decision = "accept" if gate.get("ready") else "reject"
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "artifact_gate",
        "artifact": {
            "artifact_id": manifest.get("artifact_id") or manifest.get("id"),
            "model": model.get("id"),
            "model_revision": model.get("revision"),
            "format": quantization.get("format"),
            "bits": quantization.get("bits"),
            "group_size": quantization.get("group_size"),
            "manifest": manifest,
        },
        "quality": {
            "metric": quality.get("metric"),
            "baseline_metric": quality.get("baseline_metric"),
            "threshold": quality.get("threshold"),
            "evaluation_split": quality.get("eval_split"),
        },
        "hardware": _mapping(payload.get("hardware")),
        "backend": {"name": compatibility.get("backend"), "version": compatibility.get("version")},
        "metrics": {"peak_memory_mb": quality.get("peak_memory_mb")},
        "status": "ready" if gate.get("ready") else "failed",
        "evidence_level": gate.get("evidence_level", evidence.get("evidence_level", "not_recorded")),
        "failure": _failure(loader_failure or ("; ".join(gate_issues) if gate_issues else None)),
        "decision": {"decision": decision, "reason": "artifact gate passed" if gate.get("ready") else "; ".join(gate_issues)},
        "raw_payload": dict(payload),
    }


def floating_baseline_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a matched floating-point inference baseline from project 66 or an external run."""

    source = _normalized_payload(payload)
    config = _mapping(source.get("config"))
    model = source.get("model") or config.get("model")
    model_revision = source.get("model_revision") or config.get("model_revision")
    tokenizer = _mapping(source.get("tokenizer")) or _mapping(config.get("tokenizer"))
    quantization_format = source.get("quantization_format") or config.get("quantization_format") or "none"
    if quantization_format not in {"none", "float", "fp16", "bf16", "fp32"}:
        raise ValueError("floating baseline must not use a quantized artifact format")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "floating_baseline",
        "artifact": {
            "artifact_id": source.get("artifact_id") or config.get("artifact_id"),
            "model": model,
            "model_revision": model_revision,
            "format": quantization_format,
        },
        "tokenizer": {
            "id": tokenizer.get("id") or config.get("tokenizer_id") or model,
            "revision": tokenizer.get("revision") or config.get("tokenizer_revision") or model_revision,
            "chat_template": tokenizer.get("chat_template") or config.get("chat_template"),
        },
        "backend": {
            "name": source.get("backend") or config.get("backend"),
            "version": source.get("backend_version") or config.get("backend_version"),
        },
        "hardware": _mapping(source.get("hardware")),
        "workload": _mapping(source.get("workload")) or config,
        "metrics": _mapping(source.get("metrics")),
        "quality": _mapping(source.get("quality")),
        "status": source.get("status", "not_recorded"),
        "evidence_level": source.get("evidence_level", "not_recorded"),
        "failure": _failure(source.get("failure")),
        "decision": _decision(source.get("decision")),
        "raw_payload": dict(payload),
    }


def quantized_deployment_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one quantized backend run while preserving its artifact lineage."""

    source = _normalized_payload(payload)
    config = _mapping(source.get("config"))
    artifact = _mapping(source.get("artifact"))
    tokenizer = _mapping(source.get("tokenizer")) or _mapping(config.get("tokenizer"))
    quantization_format = (
        artifact.get("format")
        or source.get("quantization_format")
        or config.get("quantization_format")
    )
    if not quantization_format or quantization_format in {"none", "float", "fp16", "bf16", "fp32"}:
        raise ValueError("quantized deployment must declare a non-floating quantization format")
    model = artifact.get("model") or source.get("model") or config.get("model")
    model_revision = artifact.get("model_revision") or source.get("model_revision") or config.get("model_revision")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "quantized_deployment",
        "artifact": {
            **artifact,
            "artifact_id": artifact.get("artifact_id") or source.get("artifact_id") or config.get("artifact_id"),
            "model": model,
            "model_revision": model_revision,
            "format": quantization_format,
            "manifest_path": artifact.get("manifest_path") or config.get("artifact_manifest"),
        },
        "tokenizer": {
            "id": tokenizer.get("id") or config.get("tokenizer_id") or model,
            "revision": tokenizer.get("revision") or config.get("tokenizer_revision") or model_revision,
            "chat_template": tokenizer.get("chat_template") or config.get("chat_template"),
        },
        "backend": {
            "name": source.get("backend") or config.get("backend"),
            "version": source.get("backend_version") or config.get("backend_version"),
            "kernel": source.get("kernel") or config.get("kernel"),
            "fallback": source.get("fallback") or config.get("fallback"),
        },
        "hardware": _mapping(source.get("hardware")),
        "workload": _mapping(source.get("workload")) or config,
        "metrics": _mapping(source.get("metrics")),
        "quality": _mapping(source.get("quality")),
        "status": source.get("status", "not_recorded"),
        "evidence_level": source.get("evidence_level", "not_recorded"),
        "failure": _failure(source.get("failure")),
        "decision": _decision(source.get("decision")),
        "raw_payload": dict(payload),
    }


def serving_profile_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one raw project-83 profile record."""

    workload = _mapping(payload.get("workload"))
    metrics = {
        "ttft_p95_ms": payload.get("ttft_p95_ms"),
        "tpot_p95_ms": payload.get("tpot_p95_ms"),
        "throughput_tps": payload.get("throughput_tps"),
        "peak_memory_mb": payload.get("peak_memory_mb"),
        "queue_wait_p95_ms": payload.get("queue_wait_p95_ms"),
        "failure_rate": payload.get("failure_rate"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "serving_profile",
        "artifact": {"artifact_id": payload.get("artifact_id")},
        "backend": {"name": payload.get("backend"), "version": payload.get("backend_version")},
        "hardware": _mapping(payload.get("hardware")),
        "workload": {"id": payload.get("workload_id"), **workload},
        "profile": {"name": payload.get("profile"), "primary_metric": payload.get("primary_metric"), "slo": _mapping(payload.get("slo"))},
        "metrics": metrics,
        "status": payload.get("status", "not_recorded"),
        "evidence_level": payload.get("evidence_level", "not_recorded"),
        "failure": _failure(payload.get("failure")),
        "decision": _mapping(payload.get("decision")) or {"decision": "not_evaluated"},
        "raw_payload": dict(payload),
    }


def validate_profile_inputs(gate: Mapping[str, Any], deployment: Mapping[str, Any], *, artifact_id: str) -> list[str]:
    """Return semantic errors before project 83 consumes 82/67 JSON.

    File existence alone is not enough: the gate must have accepted the same
    artifact and the deployment record must contain a compatible candidate.
    """

    errors: list[str] = []
    normalized_gate = artifact_gate_record(gate) if "gate" in gate else _mapping(gate)
    gate_artifact = _mapping(normalized_gate.get("artifact"))
    gate_decision = _mapping(normalized_gate.get("decision"))
    if normalized_gate.get("stage") != "artifact_gate":
        errors.append("82 result is not an artifact_gate record")
    if gate_decision.get("decision") != "accept":
        errors.append("82 artifact gate is not accepted")
    if artifact_id and gate_artifact.get("artifact_id") not in (None, artifact_id):
        errors.append("82 artifact_id does not match the selected serving artifact")

    raw_deployment = _normalized_payload(deployment)
    try:
        normalized_deployment = (
            raw_deployment
            if raw_deployment.get("stage") == "quantized_deployment"
            else quantized_deployment_record(deployment)
        )
    except ValueError as exc:
        normalized_deployment = raw_deployment
        errors.append(str(exc))
    deployment_artifact = _mapping(normalized_deployment.get("artifact"))
    deployment_status = normalized_deployment.get("status")
    if normalized_deployment.get("stage") != "quantized_deployment":
        errors.append("67 result is not a quantized_deployment record")
    if artifact_id and deployment_artifact.get("artifact_id") not in (None, artifact_id):
        errors.append("67 artifact_id does not match the selected serving artifact")
    if deployment_status in {"failed", "error"}:
        errors.append("67 deployment record is failed")
    if not normalized_deployment:
        errors.append("67 deployment result is empty")
    return errors


def save_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Persist a normalized companion record without replacing raw evidence."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
