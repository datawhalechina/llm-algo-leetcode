from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools.project_asset_relations import (
    load_asset_reference,
    load_asset_relations,
    make_asset_reference,
    validate_asset_payload,
)


def main() -> None:
    relations = load_asset_relations()
    assert "quantization.artifact_gate" in relations
    assert relations["quantization.artifact_gate"]["consumers"] == ["QUANT-INFERENCE-DEPLOY"]
    assert relations["quantization.deployment_candidate"]["consumers"] == ["QUANT-SERVING-PROFILE"]
    assert relations["distributed.model_internal"]["producer"] == "PAR-MODEL-INTERNAL"
    assert relations["inference.floating_baseline"]["contract"] == "quantization-evidence/v1"
    assert relations["deployment.single_gpu_handoff"]["consumers"] == ["INF-INTEGRATED-DELIVERY"]
    assert relations["quantization.serving_profile"]["producer"] == "QUANT-SERVING-PROFILE"

    payload = {
        "schema_version": "quantization-evidence/v1",
        "semantic_id": "QUANT-ARTIFACT-GATE",
        "stage": "artifact_gate",
        "artifact": {"artifact_id": "qwen-gptq", "model_revision": "abc123"},
        "status": "ready",
        "decision": {"decision": "accept"},
    }
    assert validate_asset_payload("quantization.artifact_gate", payload) == []

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "gate.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        reference = make_asset_reference("quantization.artifact_gate", result_path)
        assert reference["path_kind"] == "absolute"
        loaded = load_asset_reference(reference, consumer="QUANT-INFERENCE-DEPLOY")
        assert loaded["artifact"]["artifact_id"] == "qwen-gptq"

    with tempfile.TemporaryDirectory(dir=Path("benchmarks")) as tmp:
        result_path = Path(tmp) / "gate.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        reference = make_asset_reference("quantization.artifact_gate", result_path)
        assert reference["path_kind"] == "repository_relative"
        assert not Path(reference["result_path"]).is_absolute()
        loaded = load_asset_reference(reference, consumer="QUANT-INFERENCE-DEPLOY")
        assert loaded["decision"]["decision"] == "accept"

    print("project asset relations: ok")


if __name__ == "__main__":
    main()
