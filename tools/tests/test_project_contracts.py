"""CPU-only regression tests for shared project contracts and adapters."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.project_adapters import CallableExecutionAdapter, CommandExecutionAdapter, execute_with_adapter
from tools.project_contracts import (
    get_project, load_project_registry, make_artifact_manifest, make_experiment_spec, validate_run_result,
)


class SharedProjectContractTests(unittest.TestCase):
    """Check registry integrity and adapter success/failure envelopes."""

    def test_registry_integrity(self):
        """Verify project identities and existing notebook paths."""
        registry = load_project_registry()
        assert len(registry) == 34
        assert len({item["notebook"] for item in registry.values()}) == 34
        assert sum(item["primary_owner"] == "integrated_projects" for item in registry.values()) == 5
        assert (ROOT / "benchmarks/project_registry/part02.json").is_file()
        assert (ROOT / "benchmarks/project_registry/part05.json").is_file()
        root = ROOT
        assert all((root / item["notebook"]).is_file() for item in registry.values())
        assert all(item["primary_owner"] for item in registry.values())
        assert all(item["target_contract"] for item in registry.values())
        assert get_project("INF-FLOAT-BASELINE")["primary_owner"] == "inference_optimization"

    def test_execution_adapters(self):
        """Keep successful, failed and blocked adapter results distinct."""
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
        assert validate_run_result(result) == []
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
        blocked = execute_with_adapter(
            CommandExecutionAdapter(name="empty", command=(), output_path="unused.json"),
            spec,
            default_evidence_level="real_project_benchmark",
        )
        assert blocked["status"] == "blocked"


if __name__ == "__main__":
    unittest.main()
