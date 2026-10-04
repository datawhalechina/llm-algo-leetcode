"""Shared, backend-neutral contracts for Part 02 project experiments.

The contracts standardize the outer envelope only.  A project keeps its own
mechanism-specific metrics and does not need to consume another notebook's
result file in order to run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "project-evidence/v1"
EVIDENCE_LEVELS = {
    "theoretical",
    "cpu_simulation",
    "gpu_smoke",
    "real_benchmark",
    "profiler_trace",
}
DECISIONS = {"accept", "tune", "reject", "pending"}
STATUSES = {"ok", "failed", "blocked", "skipped"}


def _dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "benchmarks" / "project_registry.json"


def load_project_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the canonical project registry keyed by stable semantic ID."""

    registry_path = Path(path) if path is not None else default_registry_path()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    projects = payload.get("projects")
    if not isinstance(projects, list):
        raise ValueError("project registry 必须包含 projects 列表")
    result: dict[str, dict[str, Any]] = {}
    for project in projects:
        semantic_id = project.get("semantic_id") if isinstance(project, Mapping) else None
        if not semantic_id or semantic_id in result:
            raise ValueError(f"project registry 存在空或重复 semantic_id: {semantic_id!r}")
        result[str(semantic_id)] = dict(project)
    return result


def get_project(semantic_id: str, path: str | Path | None = None) -> dict[str, Any]:
    """Return one registered project or fail with an actionable message."""

    projects = load_project_registry(path)
    try:
        return projects[semantic_id]
    except KeyError as exc:
        raise KeyError(f"未登记的项目 semantic_id: {semantic_id}") from exc


def make_experiment_spec(
    *,
    semantic_id: str,
    experiment_id: str,
    workload: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    hardware: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable input contract shared by project adapters."""

    project = get_project(semantic_id)
    if not experiment_id.strip():
        raise ValueError("experiment_id 不能为空")
    if not workload:
        raise ValueError("workload 不能为空")
    if not baseline:
        raise ValueError("baseline 不能为空")
    if not candidates:
        raise ValueError("candidates 至少包含一个候选")
    return {
        "schema_version": CONTRACT_VERSION,
        "record_type": "experiment_spec",
        "semantic_id": semantic_id,
        "experiment_id": experiment_id,
        "primary_owner": project["primary_owner"],
        "workload": dict(workload),
        "baseline": dict(baseline),
        "candidates": [dict(candidate) for candidate in candidates],
        "hardware": _dict(hardware),
        "runtime": _dict(runtime),
        "metadata": _dict(metadata),
    }


def make_artifact_manifest(
    *,
    artifact_id: str,
    artifact_type: str,
    source: Mapping[str, Any],
    location: str | Path | None = None,
    format_name: str | None = None,
    revision: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a produced or consumed artifact without assuming a backend."""

    if not artifact_id.strip() or not artifact_type.strip():
        raise ValueError("artifact_id 和 artifact_type 不能为空")
    if not source:
        raise ValueError("artifact source 不能为空")
    return {
        "schema_version": CONTRACT_VERSION,
        "record_type": "artifact_manifest",
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "source": dict(source),
        "location": str(location) if location is not None else None,
        "format": format_name,
        "revision": revision,
        "metadata": _dict(metadata),
    }


def make_run_result(
    *,
    spec: Mapping[str, Any],
    status: str,
    evidence_level: str,
    metrics: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
    failure: Mapping[str, Any] | None = None,
    decision: str = "pending",
    decision_reason: str | None = None,
    next_action: str | None = None,
    strategy_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a common result envelope while preserving project-specific metrics."""

    if spec.get("record_type") != "experiment_spec":
        raise ValueError("spec 必须由 make_experiment_spec 生成")
    if status not in STATUSES:
        raise ValueError(f"未知 status: {status}")
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError(f"未知 evidence_level: {evidence_level}")
    if decision not in DECISIONS:
        raise ValueError(f"未知 decision: {decision}")
    if status == "failed" and not failure:
        raise ValueError("failed 结果必须记录 failure")
    return {
        "schema_version": CONTRACT_VERSION,
        "record_type": "run_result",
        "semantic_id": spec["semantic_id"],
        "experiment_id": spec["experiment_id"],
        "primary_owner": spec["primary_owner"],
        "status": status,
        "workload": _dict(spec.get("workload")),
        "hardware": _dict(spec.get("hardware")),
        "runtime": _dict(spec.get("runtime")),
        "metrics": dict(metrics),
        "strategy_metrics": _dict(strategy_metrics),
        "quality": _dict(quality),
        "artifacts": [dict(artifact) for artifact in artifacts],
        "evidence_level": evidence_level,
        "failure": _dict(failure),
        "decision": {
            "status": decision,
            "reason": decision_reason,
            "next_action": next_action,
        },
    }


def save_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Save a companion record; require an explicit opt-in before overwriting."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"结果已存在，若确认替换请设置 overwrite=True: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
