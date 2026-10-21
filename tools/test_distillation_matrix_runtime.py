"""CPU tests for distilled-student matrix contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from distillation_matrix_runtime import (
    build_distillation_matrix_plan,
    load_final_holdout,
    rank_validation_candidates,
    resolve_prompt_candidate_id,
    run_g1_screening_matrix,
    select_prompt_candidate,
    summarize_confirmation_runs,
)
from distillation_project_runtime import load_distillation_workload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    workload = load_distillation_workload(
        root / "benchmarks/workloads/distillation_delivery_v3.jsonl"
    )
    forbidden = [
        prompt_hash
        for prompt_hash in (
            workload["identity"]["train_prompt_ids_hash"],
            workload["identity"]["evaluation_prompt_ids_hash"],
        )
    ]
    # Aggregate hashes above cannot detect row-level overlap; use exact row hashes.
    from tools.post_training_project_runtime import stable_text_hash

    forbidden = [
        stable_text_hash((str(record["prompt"]),))
        for record in workload["records"]
    ]
    final = load_final_holdout(
        root / "benchmarks/workloads/distillation_delivery_final_v1.jsonl",
        forbidden_prompt_hashes=forbidden,
    )
    assert final["identity"]["record_count"] == 16
    assert "records" not in final
    assert final["identity"]["category_counts"] == {
        "basic_accuracy": 4,
        "distillation_contract": 4,
        "serving_systems": 4,
        "training_mechanisms": 4,
    }
    revealed_final = load_final_holdout(
        root / "benchmarks/workloads/distillation_delivery_final_v1.jsonl",
        forbidden_prompt_hashes=forbidden,
        reveal_records=True,
    )
    assert len(revealed_final["records"]) == 16
    assert revealed_final["access_purpose"] == "final_acceptance"
    gsm8k_workload = load_distillation_workload(
        root / "benchmarks/workloads/distillation_delivery_gsm8k_v1.jsonl"
    )
    gsm8k_medium = load_distillation_workload(
        root / "benchmarks/workloads/distillation_delivery_gsm8k_train512_v1.jsonl"
    )
    assert gsm8k_medium["identity"]["train_count"] == 512
    assert gsm8k_medium["identity"]["development_count"] == 32
    assert gsm8k_medium["identity"]["evaluation_count"] == 64
    small_train_ids = {
        record["record_id"] for record in gsm8k_workload["train"]
    }
    medium_train_ids = {
        record["record_id"] for record in gsm8k_medium["train"]
    }
    assert small_train_ids < medium_train_ids
    assert [
        record["record_id"] for record in gsm8k_workload["development"]
    ] == [
        record["record_id"] for record in gsm8k_medium["development"]
    ]
    assert [
        record["record_id"] for record in gsm8k_workload["evaluation"]
    ] == [
        record["record_id"] for record in gsm8k_medium["evaluation"]
    ]
    gsm8k_forbidden = [
        stable_text_hash((str(record["prompt"]),))
        for record in gsm8k_workload["records"]
    ]
    gsm8k_final = load_final_holdout(
        root / "benchmarks/workloads/distillation_delivery_gsm8k_final_v1.jsonl",
        forbidden_prompt_hashes=gsm8k_forbidden,
    )
    assert gsm8k_final["identity"]["record_count"] == 64
    assert gsm8k_final["identity"]["category_counts"] == {
        "reasoning_q1": 16,
        "reasoning_q2": 16,
        "reasoning_q3": 16,
        "reasoning_q4": 16,
    }
    gsm8k_manifest = json.loads((
        root / "benchmarks/workloads/distillation_delivery_gsm8k_v1.manifest.json"
    ).read_text(encoding="utf-8"))
    for output_name in ("workload", "final"):
        output = gsm8k_manifest["outputs"][output_name]
        assert hashlib.sha256((root / output["path"]).read_bytes()).hexdigest() == output["sha256"]
    assert gsm8k_manifest["quality"]["cross_split_prompt_overlap"] == 0
    assert gsm8k_manifest["quality"]["cross_split_source_overlap"] == 0
    plan = build_distillation_matrix_plan(
        run_label="matrix_test",
        validation_identity={**workload["identity"], "role": "hyperparameter_validation"},
        final_holdout_identity=final["identity"],
    )
    assert len(plan["families"]["g1_fixed_trajectory"]) == 2
    assert len(plan["families"]["g2_on_policy"]) == 1
    assert plan["families"]["g2_on_policy"][0]["optimizer_name"] == "adamw"
    assert plan["families"]["g2_on_policy"][0]["adapter_config"]["rank"] == 16
    assert plan["plan_hash"]

    baseline = {
        "quality_score": 0.5,
        "category_scores": {
            "a": {"quality_score": 0.5},
            "b": {"quality_score": 0.5},
        },
    }
    ranked = rank_validation_candidates(
        baseline,
        [
            {
                "candidate_id": "fast_tie",
                "validation": baseline,
                "training_time_s": 2.0,
                "peak_reserved_bytes": 20,
            },
            {
                "candidate_id": "regressed",
                "validation": {
                    "quality_score": 0.75,
                    "category_scores": {
                        "a": {"quality_score": 0.25},
                        "b": {"quality_score": 1.0},
                    },
                },
                "training_time_s": 1.0,
                "peak_reserved_bytes": 10,
            },
            {
                "candidate_id": "slow_tie",
                "validation": baseline,
                "training_time_s": 3.0,
                "peak_reserved_bytes": 20,
            },
        ],
        keep=2,
    )
    assert ranked["selected_candidate_ids"] == ["fast_tie", "slow_tie"]
    assert ranked["selection_status"] == "selected"
    assert ranked["ranking"][-1]["candidate_id"] == "regressed"
    assert ranked["selection_role"] == "validation_only"
    only_one_eligible = rank_validation_candidates(
        baseline,
        [
            {"candidate_id": "eligible", "validation": baseline},
            {
                "candidate_id": "regressed",
                "validation": {
                    "quality_score": 0.25,
                    "category_scores": {
                        "a": {"quality_score": 0.0},
                        "b": {"quality_score": 0.5},
                    },
                },
            },
        ],
        keep=2,
    )
    assert only_one_eligible["selected_candidate_ids"] == ["eligible"]
    strict_baseline = {
        **baseline,
        "format_compliance": {"compliance_ratio": 0.5},
        "generated_tokens": {"saturation_ratio": 0.1},
        "mean_latency_ms": 100.0,
    }
    strict = rank_validation_candidates(
        strict_baseline,
        [{
            "candidate_id": "development_failed",
            "development_selection": {
                "selected_step": 8,
                "selected_admissible": False,
            },
            "validation": {
                **strict_baseline,
                "quality_score": 0.75,
                "format_compliance": {"compliance_ratio": 0.9},
                "generated_tokens": {"saturation_ratio": 0.1},
                "mean_latency_ms": 100.0,
            },
        }],
        keep=1,
        min_quality_gain=0.05,
        require_development_admissible=True,
        min_format_compliance=0.8,
        max_saturation_increase=0.05,
        max_baseline_latency_ratio=1.1,
    )
    assert strict["selection_status"] == "no_eligible_candidates"
    assert not strict["ranking"][0]["checks"]["development_admissible"]

    prompt_results = {
        "control": {
            "candidate": {"candidate_id": "control", "role": "control"},
            "teacher": {
                "quality_score": 0.75,
                "passed_count": 24,
                "evaluation_count": 32,
                "generated_tokens": {
                    "saturation_ratio": 0.125,
                    "mean_per_request": 160.0,
                },
                "format_compliance": {"compliance_ratio": 0.5},
            },
            "student": {
                "quality_score": 0.25,
                "passed_count": 8,
                "evaluation_count": 32,
                "generated_tokens": {
                    "saturation_ratio": 0.125,
                    "mean_per_request": 120.0,
                },
                "format_compliance": {"compliance_ratio": 0.25},
            },
        },
        "concise": {
            "candidate": {"candidate_id": "concise", "role": "candidate"},
            "teacher": {
                "quality_score": 0.75,
                "passed_count": 24,
                "evaluation_count": 32,
                "generated_tokens": {
                    "saturation_ratio": 0.0625,
                    "mean_per_request": 120.0,
                },
                "format_compliance": {"compliance_ratio": 0.875},
            },
            "student": {
                "quality_score": 0.3125,
                "passed_count": 10,
                "evaluation_count": 32,
                "generated_tokens": {
                    "saturation_ratio": 0.0625,
                    "mean_per_request": 100.0,
                },
                "format_compliance": {"compliance_ratio": 0.5},
            },
        },
        "answer_only": {
            "candidate": {"candidate_id": "answer_only", "role": "diagnostic"},
            "teacher": {
                "quality_score": 0.8125,
                "passed_count": 26,
                "evaluation_count": 32,
                "generated_tokens": {
                    "saturation_ratio": 0.0,
                    "mean_per_request": 8.0,
                },
                "format_compliance": {"compliance_ratio": 1.0},
            },
            "student": {"quality_score": 0.375},
        },
    }
    prompt_selection = select_prompt_candidate(
        prompt_results,
        control_id="control",
        max_teacher_quality_loss_count=1,
        max_teacher_saturation_ratio=0.15,
        max_teacher_mean_token_ratio=1.0,
        min_teacher_format_compliance=0.35,
        max_student_quality_loss_count=1,
        max_student_saturation_ratio=0.15,
        max_student_mean_token_ratio=1.0,
    )
    assert prompt_selection["selected_candidate_id"] == "concise"
    assert prompt_selection["ranking"][0]["eligible"] is True
    assert next(
        item for item in prompt_selection["ranking"]
        if item["candidate_id"] == "answer_only"
    )["checks"]["distillation_candidate"] is False

    no_improvement_results = json.loads(json.dumps(prompt_results))
    no_improvement_results["concise"]["student"]["passed_count"] = 0
    retained_control = select_prompt_candidate(
        no_improvement_results,
        control_id="control",
        max_teacher_quality_loss_count=1,
        max_teacher_saturation_ratio=0.15,
        max_teacher_mean_token_ratio=1.0,
        min_teacher_format_compliance=0.35,
        max_student_quality_loss_count=1,
        max_student_saturation_ratio=0.15,
        max_student_mean_token_ratio=1.0,
    )
    assert retained_control["status"] == "retain_control"
    assert retained_control["selected_candidate_id"] == "control"
    historical_selection = {
        **retained_control,
        "status": "no_eligible_candidate",
        "selected_candidate_id": None,
    }
    assert resolve_prompt_candidate_id(historical_selection) == "control"

    confirmation_baseline = {
        **baseline,
        "generated_tokens": {"saturation_ratio": 0.25},
        "mean_latency_ms": 1000.0,
    }
    confirmation = summarize_confirmation_runs(
        confirmation_baseline,
        [
            {
                "seed": seed,
                "artifact_verified": True,
                "artifact": {"path": f"student-{seed}"},
                "evaluation": {
                    "quality_score": 0.5625,
                    "category_scores": {
                        "a": {"quality_score": 0.5},
                        "b": {"quality_score": 0.625},
                    },
                    "generated_tokens": {"saturation_ratio": 0.25},
                    "format_compliance": {"compliance_ratio": 0.875},
                    "mean_latency_ms": 1050.0,
                },
            }
            for seed in (42, 43)
        ],
        min_quality_gain=0.05,
        max_regressed_categories=0,
        max_saturation_increase=0.05,
        min_format_compliance=0.80,
        max_baseline_latency_ratio=1.10,
        required_qualified_runs=2,
    )
    assert confirmation["confirmed"]
    assert confirmation["selected_seed"] == 42
    assert confirmation["qualified_run_count"] == 2
    blocked_confirmation = summarize_confirmation_runs(
        confirmation_baseline,
        [{
            "seed": 42,
            "artifact_verified": True,
            "artifact": {"path": "student-42"},
            "evaluation": {
                "quality_score": 0.5625,
                "category_scores": {
                    "a": {"quality_score": 0.5},
                    "b": {"quality_score": 0.625},
                },
                "generated_tokens": {"saturation_ratio": 0.25},
                "format_compliance": {"compliance_ratio": 0.50},
                "mean_latency_ms": 1050.0,
            },
        }],
        min_quality_gain=0.05,
        max_regressed_categories=0,
        max_saturation_increase=0.05,
        min_format_compliance=0.80,
        max_baseline_latency_ratio=1.10,
        required_qualified_runs=1,
    )
    assert not blocked_confirmation["confirmed"]
    assert not blocked_confirmation["runs"][0]["checks"]["format_compliance"]

    calls: list[dict] = []

    def fake_runner(torch_module, **kwargs):
        calls.append(kwargs)
        hard_weight = float(kwargs["hard_label_weight"])
        quality = 0.5625
        return {
            "training": {
                "training_time_s": 1.0,
                "peak_reserved_bytes": 100,
                "dose_selection": {
                    "selected_step": 8,
                    "selected_admissible": True,
                },
            },
            "evaluation": {
                "quality_score": quality,
                "category_scores": {
                    "a": {"quality_score": quality},
                    "b": {"quality_score": quality},
                },
                "format_compliance": {"compliance_ratio": hard_weight},
                "generated_tokens": {
                    "saturation_ratio": 0.25,
                    "mean_per_request": 120.0,
                },
                "mean_latency_ms": 900.0,
            },
        }

    with tempfile.TemporaryDirectory() as directory:
        summary = run_g1_screening_matrix(
            object(),
            plan=plan,
            baseline_validation=baseline,
            common_run_kwargs={"teacher_path": "teacher", "student_path": "student"},
            output_dir=directory,
            _candidate_runner=fake_runner,
        )
        assert len(calls) == 2
        assert summary["selected_candidate_ids"][0] == "g1_lora_lr_1e-4"
        assert calls[0]["dose_steps"] == (8,)
        assert calls[0]["optimizer_name"] == "adamw"
        assert calls[0]["adapter_config"]["rank"] == 16
        resumed = run_g1_screening_matrix(
            object(),
            plan=plan,
            baseline_validation=baseline,
            common_run_kwargs={"teacher_path": "teacher", "student_path": "student"},
            output_dir=directory,
            _candidate_runner=fake_runner,
        )
        assert resumed == summary
        assert len(calls) == 2
    print("distillation matrix runtime smoke: ok")


if __name__ == "__main__":
    main()
