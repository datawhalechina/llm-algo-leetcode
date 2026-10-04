"""Dependency-free smoke tests for the shared project contracts."""

from __future__ import annotations

from pathlib import Path

from tools.project_adapters import CallableExecutionAdapter, execute_with_adapter
from tools.project_contracts import (
    get_project,
    load_project_registry,
    make_artifact_manifest,
    make_experiment_spec,
)


def main() -> None:
    registry = load_project_registry()
    assert len(registry) == 28
    assert len({item["notebook"] for item in registry.values()}) == 28
    root = Path(__file__).resolve().parents[1]
    assert all((root / item["notebook"]).is_file() for item in registry.values())
    assert all(item["primary_owner"] for item in registry.values())
    assert all(item["target_contract"] for item in registry.values())
    assert get_project("INF-FLOAT-BASELINE")["primary_owner"] == "inference_optimization"

    spec = make_experiment_spec(
        semantic_id="INF-FLOAT-BASELINE",
        experiment_id="contract-smoke",
        workload={"prompt_tokens": 16, "output_tokens": 8},
        baseline={"name": "torch-eager"},
        candidates=[{"name": "candidate"}],
        hardware={"device": "cpu"},
        runtime={"backend": "dry-run"},
    )
    artifact = make_artifact_manifest(
        artifact_id="smoke-result",
        artifact_type="json",
        source={"semantic_id": spec["semantic_id"], "experiment_id": spec["experiment_id"]},
    )
    adapter = CallableExecutionAdapter(
        name="dry-run",
        runner=lambda _: {
            "metrics": {"latency_ms": 1.0},
            "artifacts": [artifact],
            "evidence_level": "cpu_simulation",
            "decision": "pending",
        },
    )
    result = execute_with_adapter(adapter, spec, default_evidence_level="cpu_simulation")
    assert result["status"] == "ok"
    assert result["semantic_id"] == "INF-FLOAT-BASELINE"
    assert result["artifacts"][0]["artifact_id"] == "smoke-result"

    failed_adapter = CallableExecutionAdapter(
        name="expected-failure",
        runner=lambda _: (_ for _ in ()).throw(RuntimeError("smoke failure")),
    )
    failed = execute_with_adapter(
        failed_adapter,
        spec,
        default_evidence_level="cpu_simulation",
    )
    assert failed["status"] == "failed"
    assert failed["failure"]["error_type"] == "RuntimeError"
    print("project contract smoke: ok")


if __name__ == "__main__":
    main()
