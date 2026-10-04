"""Dependency-free tests for architecture-evaluation/v1."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.architecture_result_schema import (
    SCHEMA_VERSION,
    decide_audit,
    make_audit_record,
    make_evidence_reference,
    make_producer_record,
    save_record,
    validate_record,
    verify_evidence_reference,
)


def _producer() -> dict:
    return make_producer_record(
        semantic_id="ARCH-BLOCK-STABILITY",
        comparison_mode="controlled",
        source={"notebook": "60_Decoder_Block_Stability_Benchmark.ipynb"},
        runtime={"backend": "torch", "device": "cpu"},
        workload={"sequence_length": 32, "batch_size": 2},
        baseline={"name": "pre_norm"},
        candidates=[{"name": "post_norm", "changed_field": "norm_position"}],
        quality={"validation_loss": 1.2},
        cost={"step_time_ms": 4.5, "peak_memory_mb": 128.0},
        mechanism={"changed_field": "norm_position", "gradient_norm": 0.7},
        evidence_level="cpu_simulation",
        decision={"status": "tune", "reason": "GPU evidence is still required"},
    )


def main() -> None:
    producer = _producer()
    assert producer["schema_version"] == SCHEMA_VERSION
    assert producer["role"] == "producer"
    assert validate_record(producer) == []
    missing_gate = decide_audit(
        config_decision={"decision": "tune"},
        required_semantic_id="ARCH-BLOCK-STABILITY",
        producer_records={},
    )
    assert missing_gate["status"] == "tune"
    rejected_producer = {**producer, "status": "failed", "failure": {"reason": "OOM"}}
    rejected_producer["decision"] = {"status": "reject"}
    failed_gate = decide_audit(
        config_decision={"decision": "tune"},
        required_semantic_id="ARCH-BLOCK-STABILITY",
        producer_records={"ARCH-BLOCK-STABILITY": rejected_producer},
    )
    assert failed_gate["status"] == "reject"
    accepted_producer = {**producer, "decision": {"status": "accept"}}
    accepted_gate = decide_audit(
        config_decision={"decision": "tune"},
        required_semantic_id="ARCH-BLOCK-STABILITY",
        producer_records={"ARCH-BLOCK-STABILITY": accepted_producer},
    )
    assert accepted_gate["status"] == "accept"

    try:
        make_producer_record(
            semantic_id="ARCH-MODEL-AUDIT",
            comparison_mode="controlled",
            source={"notebook": "61.ipynb"},
            runtime={},
            workload={"sequence_length": 32},
            baseline={"name": "a"},
            candidates=[{"name": "b"}],
            quality={},
            cost={},
            mechanism={"field": "value"},
            evidence_level="cpu_simulation",
        )
    except ValueError as exc:
        assert "producer semantic_id" in str(exc)
    else:
        raise AssertionError("audit project must not create a producer record")

    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        producer_path = save_record(root / "producer.json", producer)
        reference = make_evidence_reference(
            semantic_id="ARCH-BLOCK-STABILITY",
            result_path=producer_path,
        )
        assert verify_evidence_reference(reference) == []
        audit = make_audit_record(
            source={"notebook": "61_Model_Architecture_Exploration.ipynb"},
            runtime={"mode": "offline_audit"},
            workload={"scenario": "decoder_architecture_selection"},
            baseline={"name": "registered_baseline"},
            candidates=[{"name": "registered_candidate"}],
            quality={"evidence_coverage": 1.0},
            cost={"evidence_coverage": 1.0},
            references=[reference],
            audit={"config_differences": ["norm_position"]},
            evidence_level="cpu_simulation",
            decision={"status": "tune", "reason": "one producer is not enough"},
        )
        assert audit["role"] == "audit"
        assert audit["references"][0]["sha256"]
        assert validate_record(audit) == []
        audit_path = save_record(root / "audit.json", audit)
        loaded = json.loads(audit_path.read_text(encoding="utf-8"))
        assert loaded["semantic_id"] == "ARCH-MODEL-AUDIT"

        producer_path.write_text(producer_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        assert any("sha256" in error for error in verify_evidence_reference(reference))
        try:
            make_audit_record(
                source={"notebook": "61.ipynb"},
                runtime={"mode": "offline_audit"},
                workload={"scenario": "tamper-test"},
                baseline={"name": "a"},
                candidates=[{"name": "b"}],
                quality={},
                cost={},
                references=[reference],
                audit={"status": "invalid"},
                evidence_level="cpu_simulation",
            )
        except ValueError as exc:
            assert "sha256" in str(exc)
        else:
            raise AssertionError("tampered producer evidence must not enter an audit record")

        try:
            save_record(producer_path, producer)
        except FileExistsError:
            pass
        else:
            raise AssertionError("companion records must not be overwritten implicitly")

    invalid = dict(producer)
    invalid.pop("cost")
    assert any("missing fields" in error for error in validate_record(invalid))
    print("architecture result schema: ok")


if __name__ == "__main__":
    main()
