"""CPU smoke tests for the distilled-student project runtime."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from distillation_project_runtime import (
    assess_g0_readiness,
    build_distillation_diagnostics,
    build_development_partition,
    compare_tokenizer_contracts,
    decide_distilled_student,
    estimate_stage_memory_bytes,
    final_number_format_compliant,
    load_distillation_workload,
    model_asset_manifest,
    score_expected_output,
    select_distillation_route,
    select_development_dose,
    select_validation_dose,
    summarize_offline_trace_admission,
    summarize_distilled_candidate,
    validate_distillation_contract,
)


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int], *, eos_token_id: int = 2) -> None:
        self._vocab = vocab
        self.bos_token_id = 1
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id
        self.unk_token_id = 0
        self.chat_template = "{{ messages }}"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._vocab.get(char, self.unk_token_id) for char in text]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    shared_vocab = {char: index + 3 for index, char in enumerate("# 42-.,Solve请给出最终数值答案。")}
    compatible = compare_tokenizer_contracts(
        _FakeTokenizer(shared_vocab), _FakeTokenizer(shared_vocab)
    )
    assert compatible["full_logit_compatible"]
    shifted_vocab = dict(shared_vocab)
    shifted_vocab["4"], shifted_vocab["2"] = shifted_vocab["2"], shifted_vocab["4"]
    incompatible = compare_tokenizer_contracts(
        _FakeTokenizer(shared_vocab), _FakeTokenizer(shifted_vocab)
    )
    assert not incompatible["full_logit_compatible"]
    assert incompatible["recommended_route"] == "offline_trace_distillation"
    route_gates = {
        "min_trace_parse_ratio": 0.80,
        "max_trace_saturation_ratio": 0.25,
        "min_trace_quality_score": 0.70,
        "min_trace_quality_gain": 0.05,
        "min_trace_category_quality": 0.50,
        "max_trace_regressed_categories": 0,
        "min_trace_training_eligible_ratio": 0.70,
        "min_records_per_category": 1,
        "max_trace_mean_output_token_ratio": 3.0,
        "max_trace_mean_latency_ratio": 3.0,
    }
    hybrid_route = select_distillation_route({
        "offline": {
            "trace_available": True,
            "parse_ratio": 0.90,
            "quality_score": 0.75,
            "baseline_quality_score": 0.70,
            "quality_delta": 0.05,
            "minimum_category_count": 1,
            "minimum_category_quality": 0.75,
            "regressed_category_count": 0,
            "training_eligible_ratio": 0.75,
            "training_reuse_allowed": False,
            "saturation_ratio": 0.10,
            "mean_output_token_ratio": 2.0,
            "mean_latency_ratio": 2.0,
        },
        "online": {
            "teacher_ready": True,
            "full_logit_compatible": True,
            "memory_admitted": True,
        },
    }, route_gates)
    assert hybrid_route["route"] == "hybrid"
    assert not hybrid_route["offline_training_data_ready"]
    offline_route = select_distillation_route({
        "offline": {
            "trace_available": True,
            "parse_ratio": 0.90,
            "quality_score": 0.75,
            "baseline_quality_score": 0.70,
            "quality_delta": 0.05,
            "minimum_category_count": 1,
            "minimum_category_quality": 0.75,
            "regressed_category_count": 0,
            "training_eligible_ratio": 0.75,
            "training_reuse_allowed": True,
            "saturation_ratio": 0.10,
            "mean_output_token_ratio": 2.0,
            "mean_latency_ratio": 2.0,
        },
        "online": {
            "teacher_ready": True,
            "full_logit_compatible": False,
            "memory_admitted": True,
        },
    }, route_gates)
    assert offline_route["route"] == "offline"
    blocked_route = select_distillation_route({
        "offline": {
            "trace_available": True,
            "parse_ratio": 0.50,
            "quality_score": 0.60,
            "baseline_quality_score": 0.70,
            "quality_delta": -0.10,
            "minimum_category_count": 1,
            "minimum_category_quality": 0.25,
            "regressed_category_count": 1,
            "training_eligible_ratio": 0.50,
            "training_reuse_allowed": False,
            "saturation_ratio": 0.50,
            "mean_output_token_ratio": 4.0,
            "mean_latency_ratio": 4.0,
        },
        "online": {
            "teacher_ready": False,
            "full_logit_compatible": False,
            "memory_admitted": False,
        },
    }, route_gates)
    assert blocked_route["route"] == "blocked"

    trace_summary = summarize_offline_trace_admission(
        {
            "mean_generated_tokens": 200.0,
            "mean_latency_ms": 2000.0,
            "rows": [
                {
                    "record_id": "a", "category": "q1", "source_split": "evaluation",
                    "correct": True, "parseable": True, "saturated": False,
                    "output": "reasoning\n#### 2",
                },
                {
                    "record_id": "b", "category": "q1", "source_split": "evaluation",
                    "correct": False, "parseable": True, "saturated": False,
                    "output": "<think>wrong</think>\n\\boxed{3}",
                },
            ],
        },
        [
            {"record_id": "a", "passed": False, "generated_tokens": 100, "latency_ms": 1000},
            {"record_id": "b", "passed": True, "generated_tokens": 100, "latency_ms": 1000},
        ],
    )
    assert trace_summary["quality_delta"] == 0.0
    assert trace_summary["paired_transition"]["improved_count"] == 1
    assert trace_summary["paired_transition"]["regressed_count"] == 1
    assert trace_summary["training_eligible_ratio"] == 0.5
    assert trace_summary["raw_student_format_ratio"] == 0.5
    assert not trace_summary["training_reuse_allowed"]
    v1_workload = load_distillation_workload(
        root / "benchmarks/workloads/distillation_delivery.jsonl"
    )
    assert v1_workload["identity"]["train_count"] == 12
    assert v1_workload["identity"]["evaluation_count"] == 8
    workload = load_distillation_workload(
        root / "benchmarks/workloads/distillation_delivery_v2.jsonl"
    )
    assert workload["identity"]["train_count"] == 32
    assert workload["identity"]["evaluation_count"] == 32
    assert workload["identity"]["evaluation_category_counts"] == {
        "basic_accuracy": 8,
        "distillation_contract": 8,
        "serving_systems": 8,
        "training_mechanisms": 8,
    }
    assert workload["identity"]["scoring_contract_sha256"]
    assert workload["identity"]["train_prompt_ids_hash"] != workload["identity"]["evaluation_prompt_ids_hash"]
    v3_workload = load_distillation_workload(
        root / "benchmarks/workloads/distillation_delivery_v3.jsonl"
    )
    assert v3_workload["identity"]["train_count"] == 32
    assert v3_workload["identity"]["evaluation_count"] == 32
    assert v3_workload["identity"]["train_reference_count"] == 32
    assert v3_workload["identity"]["train_reference_answers_hash"]
    development_partition = build_development_partition(
        v3_workload,
        [
            {"record_id": "train-basic-02", "validator": "exact", "expected_exact": ["72"]},
            {"record_id": "train-basic-07", "validator": "exact", "expected_exact": ["jupiter"]},
            {"record_id": "train-serving-03", "expected_groups": [["delay", "latency", "duration"], ["request", "prompt", "submission", "submitting"], ["first generated token", "first token returned", "first output token"]], "forbidden_any": ["first input", "physical token"]},
            {"record_id": "train-serving-05", "expected_groups": [["identical", "same", "shared", "repeated"], ["tokenized prompt prefix", "prompt prefix", "prefix"], ["reuse", "reuses", "reused", "requests"]], "forbidden_any": ["object store", "database", "network"]},
            {"record_id": "train-training-04", "expected_groups": [["evaluation", "validation"], ["separate", "independent", "held out", "outside training"], ["leakage", "contamination", "unbiased", "quality estimate", "generalization", "overfitting"]]},
            {"record_id": "train-training-06", "expected_groups": [["gradient", "gradients"], ["clip", "clipping", "limit", "cap", "maximum"], ["large", "exploding", "magnitude", "norm"], ["stable", "stabilize", "unstable", "prevent"]]},
            {"record_id": "train-distill-05", "expected_groups": [["hash"], ["bytes", "weights"], ["exact", "same", "unchanged", "integrity", "tamper"]]},
            {"record_id": "train-distill-08", "expected_groups": [["gate", "criterion", "criteria", "requirement", "threshold"], ["failed", "unmet", "not passed", "below"], ["not deployment ready", "not deployment-ready", "not deployable", "must not deploy", "no handoff", "cannot hand off"]], "forbidden_any": ["telecommunications", "carrier", "network handoff"]},
        ],
    )
    assert len(development_partition["train"]) == 24
    assert len(development_partition["development"]) == 8
    assert len(development_partition["evaluation"]) == 32
    assert development_partition["identity"]["development_category_counts"] == {
        "basic_accuracy": 2,
        "distillation_contract": 2,
        "serving_systems": 2,
        "training_mechanisms": 2,
    }
    assert (
        development_partition["identity"]["scoring_contract_sha256"]
        == v3_workload["identity"]["scoring_contract_sha256"]
    )
    development_by_id = {
        record["record_id"]: record for record in development_partition["development"]
    }
    assert all(record["reference_answer"] for record in development_by_id.values())
    assert development_partition["identity"]["development_reference_answers_hash"]

    gsm8k_workload = load_distillation_workload(
        root / "benchmarks/workloads/distillation_delivery_gsm8k_v1.jsonl"
    )
    assert gsm8k_workload["identity"]["train_count"] == 128
    assert gsm8k_workload["identity"]["development_count"] == 32
    assert gsm8k_workload["identity"]["evaluation_count"] == 64
    explicit_partition = build_development_partition(gsm8k_workload, [])
    assert len(explicit_partition["train"]) == 128
    assert len(explicit_partition["development"]) == 32
    assert explicit_partition["identity"]["development_category_counts"] == {
        "reasoning_q1": 8,
        "reasoning_q2": 8,
        "reasoning_q3": 8,
        "reasoning_q4": 8,
    }
    assert score_expected_output(
        "The intermediate value is 12. Final answer: #### 1,024",
        validator="final_number",
        expected_exact=("1024",),
    )["passed"]
    assert not score_expected_output(
        "The intermediate value is 1024, but the final answer is #### 12",
        validator="final_number",
        expected_exact=("1024",),
    )["passed"]
    assert final_number_format_compliant("work\n#### 1,024")
    assert final_number_format_compliant("#### -2.5")
    assert not final_number_format_compliant("#### number: 14")
    assert not final_number_format_compliant("#### 14\nextra text")

    def score_canary(record_id: str, output: str) -> bool:
        spec = development_by_id[record_id]
        return score_expected_output(
            output,
            spec.get("expected_any", ()),
            spec.get("expected_all", ()),
            validator=spec.get("validator", "terms"),
            expected_exact=spec.get("expected_exact", ()),
            forbidden_any=spec.get("forbidden_any", ()),
            expected_groups=spec.get("expected_groups", ()),
        )["passed"]

    assert score_canary(
        "dev-serving-03",
        "TTFT is the latency from submitting a request until its first generated token is returned.",
    )
    assert not score_canary(
        "dev-serving-03",
        "Time to first token is the duration until a user provides their first input.",
    )
    assert score_canary(
        "dev-distill-05",
        "The content hash verifies that the saved model weights have the same exact bytes.",
    )
    assert not score_canary(
        "dev-distill-05",
        "The content hash confirms that the model artifact has integrity.",
    )
    baseline_development = {
        "quality_score": 0.5,
        "category_scores": {
            "a": {"quality_score": 0.5},
            "b": {"quality_score": 0.5},
        },
    }
    selected_dose = select_development_dose(
        baseline_development,
        [
            {"step": 8, "development": {"quality_score": 0.5, "category_scores": {"a": {"quality_score": 0.5}, "b": {"quality_score": 0.5}}}},
            {"step": 16, "development": {"quality_score": 0.75, "category_scores": {"a": {"quality_score": 0.5}, "b": {"quality_score": 1.0}}}},
            {"step": 30, "development": {"quality_score": 0.75, "category_scores": {"a": {"quality_score": 0.5}, "b": {"quality_score": 1.0}}}},
        ],
    )
    assert selected_dose["status"] == "selected"
    assert selected_dose["selected_step"] == 16
    nll_tie_break = select_development_dose(
        {
            **baseline_development,
            "reference_completion": {"mean_nll": 2.0},
        },
        [
            {"step": 1, "development": {**baseline_development, "reference_completion": {"mean_nll": 1.8}}},
            {"step": 2, "development": {**baseline_development, "reference_completion": {"mean_nll": 1.5}}},
            {"step": 4, "development": {**baseline_development, "reference_completion": {"mean_nll": 1.5}}},
        ],
    )
    assert nll_tie_break["selected_step"] == 2
    assert nll_tie_break["dose_summaries"][1]["completion_nll_gain"] == 0.5
    diagnostic_dose = select_development_dose(
        baseline_development,
        [{"step": 8, "development": {"quality_score": 0.25, "category_scores": {"a": {"quality_score": 0.0}, "b": {"quality_score": 0.5}}}}],
    )
    assert diagnostic_dose["status"] == "diagnostic_only"
    validation_dose = select_validation_dose(
        baseline_development,
        [
            {"step": 8, "validation": {"quality_score": 0.5, "category_scores": {"a": {"quality_score": 0.5}, "b": {"quality_score": 0.5}}}},
            {"step": 16, "validation": {"quality_score": 0.75, "category_scores": {"a": {"quality_score": 0.5}, "b": {"quality_score": 1.0}}}},
            {"step": 30, "validation": {"quality_score": 0.75, "category_scores": {"a": {"quality_score": 0.5}, "b": {"quality_score": 1.0}}}},
        ],
    )
    assert validation_dose["selected_step"] == 16
    assert validation_dose["selection_role"] == "hyperparameter_validation"
    assert score_expected_output("Prefill followed by decode", expected_all=("prefill", "decode"))["passed"]
    assert not score_expected_output("Only prefill", expected_all=("prefill", "decode"))["passed"]
    assert score_expected_output("The answer is forty two (42).", expected_any=("42",))["passed"]
    assert not score_expected_output("This was computed.", expected_any=("compute",))["passed"]
    assert score_expected_output(
        "No.", validator="exact", expected_exact=("no",)
    )["passed"]
    assert score_expected_output(
        "The answer is no.", validator="choice", expected_any=("no",),
        forbidden_any=("yes",),
    )["passed"]
    assert not score_expected_output(
        "It may be yes or no.", validator="choice", expected_any=("no",),
        forbidden_any=("yes",),
    )["passed"]
    assert score_expected_output(
        "AdamW keeps first moment and variance estimates.",
        expected_groups=(("first moment", "momentum"), ("second moment", "variance")),
    )["passed"]
    assert not score_expected_output(
        "recompute through communication",
        expected_any=("recompute",),
        forbidden_any=("communication",),
    )["passed"]

    ready_g0 = assess_g0_readiness(
        {
            "teacher": {
                "quality_score": 0.875,
                "category_scores": {
                    "a": {"quality_score": 0.75},
                    "b": {"quality_score": 1.0},
                },
                "generated_tokens": {"saturation_ratio": 0.0},
                "rows": [{"record_id": "x", "deterministic_output": True}],
            }
        },
        min_teacher_quality=0.75,
        min_teacher_category_quality=0.625,
        max_teacher_saturation_ratio=0.10,
    )
    assert ready_g0["ready"]
    gap_blocked_g0 = assess_g0_readiness(
        {
            "teacher": {
                "quality_score": 0.50,
                "category_scores": {"a": {"quality_score": 0.50}},
                "generated_tokens": {"saturation_ratio": 0.0},
                "rows": [{"record_id": "x", "deterministic_output": True}],
            },
            "student": {"quality_score": 0.48},
        },
        min_teacher_quality=0.20,
        min_teacher_category_quality=0.10,
        max_teacher_saturation_ratio=0.10,
        min_teacher_student_gap=0.05,
    )
    assert gap_blocked_g0["issues"] == ["teacher_student_gap_below_gate"]
    blocked_g0 = assess_g0_readiness(
        {
            "teacher": {
                "quality_score": 0.5,
                "category_scores": {"a": {"quality_score": 0.5}},
                "generated_tokens": {"saturation_ratio": 0.25},
                "rows": [{"record_id": "x", "deterministic_output": False}],
            }
        },
        min_teacher_quality=0.75,
        min_teacher_category_quality=0.625,
        max_teacher_saturation_ratio=0.10,
    )
    assert not blocked_g0["ready"]
    assert set(blocked_g0["issues"]) == {
        "teacher_quality_below_gate",
        "teacher_category_below_gate",
        "teacher_output_nondeterministic",
        "teacher_output_saturation_above_gate",
    }

    teacher_bytes = 3 * 1024**3
    student_bytes = 1 * 1024**3
    assert estimate_stage_memory_bytes(teacher_bytes, student_bytes, "g2") > estimate_stage_memory_bytes(
        teacher_bytes, student_bytes, "g0"
    )

    diagnostic_baseline = {
        "quality_score": 0.5,
        "category_scores": {"core": {"quality_score": 0.5}},
        "format_compliance": {"compliance_ratio": 0.25},
        "generated_tokens": {"saturation_ratio": 0.25, "mean_per_request": 100.0},
        "mean_latency_ms": 1000.0,
        "mean_ms_per_output_token": 10.0,
        "rows": [
            {"record_id": "a", "passed": True, "format_compliant": False},
            {"record_id": "b", "passed": False, "format_compliant": False},
        ],
    }
    diagnostic_candidate = {
        "quality_score": 0.5,
        "category_scores": {"core": {"quality_score": 0.5}},
        "format_compliance": {"compliance_ratio": 0.5},
        "generated_tokens": {"saturation_ratio": 0.0, "mean_per_request": 110.0},
        "mean_latency_ms": 1050.0,
        "mean_ms_per_output_token": 9.5,
        "rows": [
            {"record_id": "a", "passed": False, "format_compliant": True},
            {"record_id": "b", "passed": True, "format_compliant": False},
        ],
    }
    diagnostics = build_distillation_diagnostics(
        diagnostic_baseline,
        diagnostic_candidate,
        {
            "dose_selection": {
                "dose_summaries": [{"completion_nll_gain": 0.1, "quality_gain": 0.0}]
            },
            "metric_history": [{"step": 1, "loss": 0.5}],
            "trajectory_summary": {"record_count": 1},
            "adapter": {"update_diagnostics": {"delta_l2_norm": 1.0}},
        },
    )
    assert diagnostics["quality_transition"]["improved_count"] == 1
    assert diagnostics["quality_transition"]["regressed_count"] == 1
    assert diagnostics["proxy_alignment"]["proxy_task_mismatch"]
    assert diagnostics["training_observability"] == {
        "metric_history_available": True,
        "trajectory_summary_available": True,
        "adapter_update_available": True,
    }

    contract_spec = {
        "schema_version": "distilled-student-project/v2",
        "teacher": {"model_id": "teacher", "revision": "t1", "frozen": True},
        "student": {"model_id": "student", "revision": "s1"},
        "data": {"train_hash": "train", "evaluation_hash": "evaluation"},
        "groups": {
            "g0": {"student_revision": "s1", "train_steps": 0, "trajectory_mode": "baseline"},
            "g1": {"student_revision": "s1", "train_steps": 10, "trajectory_mode": "fixed"},
            "g2": {"student_revision": "s1", "train_steps": 10, "trajectory_mode": "on_policy"},
        },
    }
    contract = validate_distillation_contract(contract_spec)
    assert contract["ready"]
    teacher = {
        "quality_score": 0.8,
        "artifact_size_mb": 3000.0,
        "latency_ms": 60.0,
        "category_scores": {"core": {"quality_score": 0.8}},
    }
    baseline = {
        "quality_score": 0.55,
        "artifact_size_mb": 1000.0,
        "latency_ms": 24.0,
        "category_scores": {"core": {"quality_score": 0.5}},
    }
    candidate = {
        "name": "g2",
        "quality_score": 0.72,
        "artifact_size_mb": 1000.0,
        "latency_ms": 24.0,
        "artifact_verified": True,
        "load_status": "loaded",
        "category_scores": {"core": {"quality_score": 0.7}},
        "latency_protocol": {"repeats": 3, "sample_count": 24},
    }
    summary = summarize_distilled_candidate(teacher, baseline, candidate)
    assert abs(summary["quality_gain"] - 0.17) < 1e-9
    assert decide_distilled_student(
        contract,
        summary,
        {
            "min_quality_gain": 0.05,
            "min_teacher_retention": 0.85,
            "max_teacher_size_ratio": 0.5,
            "max_teacher_latency_ratio": 0.7,
            "max_baseline_latency_ratio": 1.05,
            "max_regressed_categories": 0,
        },
    )["decision"] == "accept"

    with tempfile.TemporaryDirectory() as directory:
        bad_path = Path(directory) / "bad.jsonl"
        bad_path.write_text(
            json.dumps({"record_id": "x", "split": "evaluation", "prompt": "x"}) + "\n",
            encoding="utf-8",
        )
        try:
            load_distillation_workload(bad_path)
        except ValueError as error:
            assert "expected_any" in str(error)
        else:
            raise AssertionError("缺少 evaluation 目标时应失败")

        model_path = Path(directory) / "model"
        model_path.mkdir()
        (model_path / "config.json").write_text("{}\n", encoding="utf-8")
        (model_path / "model.safetensors").write_bytes(b"weight-bytes")
        manifest = model_asset_manifest(model_path, include_content_hash=True)
        assert manifest["weight_files"][0]["sha256"]
        assert manifest["weights_content_sha256"]
    print("distillation project runtime smoke: ok")


if __name__ == "__main__":
    main()
