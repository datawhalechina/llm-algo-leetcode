from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tools.project_asset_relations import validate_asset_payload
from tools.quantization_result_schema import (
    artifact_gate_record,
    quantized_deployment_record,
    save_record,
    serving_profile_record,
)


def main() -> None:
    gate = artifact_gate_record(
        {
            "manifest": {
                "artifact_id": "qwen-gptq",
                "model": {"id": "Qwen/test", "revision": "abc123"},
                "quantization": {"format": "gptq", "bits": 4, "group_size": 128},
            },
            "quality": {"metric": "ppl", "baseline_metric": 10.0, "threshold": 1.1},
            "compatibility": {"backend": "vllm", "version": "test"},
            "gate": {"ready": True, "evidence_level": "real_project_benchmark"},
        }
    )
    assert gate["semantic_id"] == "QUANT-ARTIFACT-GATE"
    assert validate_asset_payload("quantization.artifact_gate", gate) == []
    for state in ("tune", "pending", "reject"):
        recorded = artifact_gate_record({
            **gate["raw_payload"],
            "gate": {"ready": False, "issues": ["budget_or_evidence"]},
            "decision": {"decision": state, "reason": "original decision"},
        })
        assert recorded["decision"] == {"decision": state, "reason": "original decision"}

    deployment = quantized_deployment_record(
        {
            "artifact": {
                "artifact_id": "qwen-gptq",
                "path": "artifacts/qwen-gptq",
                "model": "Qwen/test",
                "model_revision": "abc123",
                "method": "gptq",
                "format": "gptq",
            },
            "lineage": {
                "artifact_gate_reference": "82_ref.json",
                "artifact_gate_sha256": "abc123",
            },
            "workload": {"id": "fixed-prompts-v1"},
            "metrics": {"ttft_ms": 10.0},
            "status": "ok",
            "decision": {"decision": "accept"},
        }
    )
    assert deployment["semantic_id"] == "QUANT-INFERENCE-DEPLOY"
    assert deployment["artifact"]["path"] == "artifacts/qwen-gptq"
    assert deployment["lineage"]["artifact_gate_reference"] == "82_ref.json"
    assert validate_asset_payload("quantization.deployment_candidate", deployment) == []

    profile = serving_profile_record(
        {
            "artifact_id": "qwen-gptq",
            "profile": "latency",
            "workload_id": "fixed-prompts-v1",
            "status": "ok",
            "lineage": {"deployment_reference": "67_ref.json"},
        }
    )
    assert profile["semantic_id"] == "QUANT-SERVING-PROFILE"
    assert profile["lineage"]["deployment_reference"] == "67_ref.json"

    with TemporaryDirectory() as directory:
        output = Path(directory) / "profile.json"
        save_record(output, profile)
        try:
            save_record(output, profile)
        except FileExistsError:
            pass
        else:
            raise AssertionError("save_record must not overwrite an existing run")
    print("quantization result schema: ok")


if __name__ == "__main__":
    main()
