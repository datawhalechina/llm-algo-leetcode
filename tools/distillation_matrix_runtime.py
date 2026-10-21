"""Contracts and ranking helpers for distilled-student parameter matrices.

The module deliberately separates three evidence roles:

* development canaries choose a checkpoint dose inside one training run;
* the repeated V3 evaluation split acts as validation for hyperparameters;
* a separately stored final holdout is opened only after one recipe is frozen.

GPU training remains in ``distillation_project_runtime``.  Keeping matrix
selection here makes the split policy and ranking rules testable on CPU.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.post_training_project_runtime import stable_text_hash
from tools.post_training_project_runtime import save_json_record


G1_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "g1_lora_lr_1e-4",
        "training_method": "lora_adamw",
        "learning_rate": 1e-4,
        "optimizer_name": "adamw",
        "max_grad_norm": 1.0,
        "hard_label_weight": 0.3,
        "adapter_config": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
    },
    {
        "candidate_id": "g1_lora_lr_2e-4",
        "training_method": "lora_adamw",
        "learning_rate": 2e-4,
        "optimizer_name": "adamw",
        "max_grad_norm": 1.0,
        "hard_label_weight": 0.3,
        "adapter_config": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
    },
)

G2_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "g2_lora_on_policy_lr_1e-4_beta_0",
        "training_method": "trl_lora_on_policy_distillation",
        "learning_rate": 1e-4,
        "beta": 0.0,
        "optimizer_name": "adamw",
        "max_grad_norm": 1.0,
        "adapter_config": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
    },
)


def build_distillation_matrix_plan(
    *,
    run_label: str,
    validation_identity: Mapping[str, Any],
    final_holdout_identity: Mapping[str, Any],
    screening_steps: int = 8,
    screening_dose_steps: Sequence[int] | None = None,
    finalist_steps: int = 30,
    finalists_per_family: int = 2,
    min_screening_quality_gain: float = 0.05,
    require_development_admissible: bool = False,
    max_screening_regressed_categories: int = 0,
    min_screening_format_compliance: float = 0.0,
    max_screening_saturation_increase: float | None = None,
    max_screening_baseline_latency_ratio: float | None = None,
) -> dict[str, Any]:
    """Build the frozen two-phase search plan without reading final answers."""

    if screening_steps < 1 or finalist_steps < screening_steps:
        raise ValueError("矩阵步数必须满足 1 <= screening_steps <= finalist_steps")
    if finalists_per_family < 1:
        raise ValueError("每个方法族至少保留一个 finalist")
    if min_screening_quality_gain < 0:
        raise ValueError("筛选质量增益门槛不能为负")
    normalized_doses = sorted({
        int(step) for step in (screening_dose_steps or (screening_steps,))
    })
    if not normalized_doses or normalized_doses[0] < 1 or normalized_doses[-1] != screening_steps:
        raise ValueError("screening_dose_steps 必须为正整数且以 screening_steps 结尾")
    if max_screening_regressed_categories < 0:
        raise ValueError("退化分类数量门槛不能为负")
    if not 0.0 <= min_screening_format_compliance <= 1.0:
        raise ValueError("格式合规率门槛必须位于 [0, 1]")
    if max_screening_saturation_increase is not None and max_screening_saturation_increase < 0:
        raise ValueError("截断率增量门槛不能为负")
    if max_screening_baseline_latency_ratio is not None and max_screening_baseline_latency_ratio <= 0:
        raise ValueError("延迟比门槛必须大于 0")
    g2 = [dict(item) for item in G2_MATRIX]
    plan = {
        "schema_version": "distillation-matrix/v1",
        "run_label": run_label,
        "selection_policy": {
            "screening_steps": screening_steps,
            "screening_dose_steps": normalized_doses,
            "finalist_steps": finalist_steps,
            "finalists_per_family": finalists_per_family,
            "min_quality_gain": min_screening_quality_gain,
            "pre_confirmation_gates": {
                "require_development_admissible": require_development_admissible,
                "max_regressed_categories": max_screening_regressed_categories,
                "min_format_compliance": min_screening_format_compliance,
                "max_saturation_increase": max_screening_saturation_increase,
                "max_baseline_latency_ratio": max_screening_baseline_latency_ratio,
            },
            "development_role": "checkpoint_dose_selection",
            "validation_role": "hyperparameter_selection",
            "final_holdout_role": "single_use_acceptance_only",
        },
        "validation_identity": dict(validation_identity),
        "final_holdout_identity": dict(final_holdout_identity),
        "families": {
            "g1_fixed_trajectory": [dict(item) for item in G1_MATRIX],
            "g2_on_policy": g2,
        },
    }
    plan["plan_hash"] = stable_text_hash((
        json.dumps(plan, sort_keys=True, ensure_ascii=False),
    ))
    return plan


def load_final_holdout(
    path: str | Path,
    *,
    forbidden_prompt_hashes: Sequence[str] = (),
    reveal_records: bool = False,
) -> dict[str, Any]:
    """Validate a final-only workload; hide scored rows outside final acceptance."""

    source = Path(path)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("final holdout 不能为空")
    required = {"record_id", "split", "category", "prompt", "validator"}
    allowed_validators = {"choice", "exact", "terms", "final_number"}
    known_prompt_hashes = set(forbidden_prompt_hashes)
    ids: list[str] = []
    prompt_hashes: list[str] = []
    scoring_contract: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"final holdout 第 {index + 1} 行缺少字段：{sorted(missing)}")
        if record["split"] != "final":
            raise ValueError("final holdout 只能包含 split='final' 的记录")
        validator = str(record["validator"])
        if validator not in allowed_validators:
            raise ValueError(f"未知 final validator：{validator!r}")
        if validator in {"exact", "final_number"} and not record.get("expected_exact"):
            raise ValueError(f"{validator} final 记录必须提供 expected_exact")
        if validator in {"choice", "terms"} and not (
            record.get("expected_any")
            or record.get("expected_all")
            or record.get("expected_groups")
        ):
            raise ValueError(f"{validator} final 记录缺少评分目标")
        if validator == "choice" and not record.get("forbidden_any"):
            raise ValueError("choice final 记录必须提供 forbidden_any")
        record_id = str(record["record_id"])
        prompt_hash = stable_text_hash((str(record["prompt"]),))
        if prompt_hash in known_prompt_hashes:
            raise ValueError(f"final prompt 与开发/验证数据重复：{record_id}")
        ids.append(record_id)
        prompt_hashes.append(prompt_hash)
        scoring_contract.append({
            key: record.get(key, [])
            for key in (
                "record_id", "category", "validator", "expected_exact",
                "expected_any", "expected_all", "expected_groups", "forbidden_any",
            )
        })
    if len(ids) != len(set(ids)):
        raise ValueError("final record_id 必须唯一")
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise ValueError("final prompt 必须唯一")
    category_counts = Counter(str(record["category"]) for record in records)
    if len(set(category_counts.values())) != 1:
        raise ValueError("final holdout 必须按类别均衡")
    result = {
        "source": str(source),
        "identity": {
            "role": "single_use_final_holdout",
            "record_count": len(records),
            "category_counts": dict(sorted(category_counts.items())),
            "record_ids_hash": stable_text_hash(ids),
            "prompt_ids_hash": stable_text_hash(prompt_hashes),
            "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "scoring_contract_sha256": stable_text_hash((
                json.dumps(scoring_contract, sort_keys=True, ensure_ascii=False),
            )),
        },
    }
    if reveal_records:
        result["records"] = records
        result["access_purpose"] = "final_acceptance"
    return result


def rank_validation_candidates(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    keep: int,
    min_quality_gain: float = 0.0,
    require_development_admissible: bool = False,
    max_regressed_categories: int = 0,
    min_format_compliance: float = 0.0,
    max_saturation_increase: float | None = None,
    max_baseline_latency_ratio: float | None = None,
) -> dict[str, Any]:
    """Rank matrix candidates without consulting final-holdout evidence."""

    if keep < 1:
        raise ValueError("keep 必须大于 0")
    if not candidates:
        raise ValueError("candidates 不能为空")
    if min_quality_gain < 0:
        raise ValueError("min_quality_gain 不能为负")
    if max_regressed_categories < 0:
        raise ValueError("max_regressed_categories 不能为负")
    baseline_quality = float(baseline.get("quality_score", 0.0))
    baseline_categories = baseline.get("category_scores", {})
    baseline_saturation = float(
        baseline.get("generated_tokens", {}).get("saturation_ratio", 0.0)
    )
    baseline_latency_ms = float(baseline.get("mean_latency_ms", 0.0))
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ValueError("candidate_id 必须非空且唯一")
        seen.add(candidate_id)
        evaluation = candidate.get("validation", {})
        quality = float(evaluation.get("quality_score", 0.0))
        format_ratio = float(
            evaluation.get("format_compliance", {}).get("compliance_ratio", 0.0)
        )
        saturation_ratio = float(
            evaluation.get("generated_tokens", {}).get("saturation_ratio", 1.0)
        )
        mean_output_tokens = float(
            evaluation.get("generated_tokens", {}).get("mean_per_request", float("inf"))
        )
        mean_latency_ms = float(evaluation.get("mean_latency_ms", float("inf")))
        development_selection = candidate.get("development_selection", {})
        development_admissible = development_selection.get("selected_admissible") is True
        regressed = sorted(
            category
            for category, result in evaluation.get("category_scores", {}).items()
            if float(result.get("quality_score", 0.0))
            < float(baseline_categories.get(category, {}).get("quality_score", 0.0))
        )
        checks = {
            "development_admissible": (
                development_admissible or not require_development_admissible
            ),
            "quality_gain": quality - baseline_quality >= min_quality_gain,
            "category_stability": len(regressed) <= max_regressed_categories,
            "format_compliance": format_ratio >= min_format_compliance,
            "saturation": (
                max_saturation_increase is None
                or saturation_ratio <= baseline_saturation + max_saturation_increase
            ),
            "latency": (
                max_baseline_latency_ratio is None
                or (
                    baseline_latency_ms > 0
                    and mean_latency_ms <= baseline_latency_ms * max_baseline_latency_ratio
                )
            ),
        }
        ranked.append({
            "candidate_id": candidate_id,
            "selected_step": development_selection.get("selected_step"),
            "development_admissible": development_admissible,
            "quality_score": quality,
            "quality_gain": quality - baseline_quality,
            "format_compliance_ratio": format_ratio,
            "saturation_ratio": saturation_ratio,
            "mean_output_tokens": mean_output_tokens,
            "mean_latency_ms": mean_latency_ms,
            "regressed_categories": regressed,
            "regressed_category_count": len(regressed),
            "training_time_s": float(candidate.get("training_time_s", float("inf"))),
            "peak_reserved_bytes": int(candidate.get("peak_reserved_bytes", 0)),
            "checks": checks,
            "eligible": all(checks.values()),
        })
    ranked.sort(key=lambda item: (
        not item["eligible"],
        item["regressed_category_count"],
        -item["format_compliance_ratio"],
        item["saturation_ratio"],
        item["mean_output_tokens"],
        item["mean_latency_ms"],
        -item["quality_score"],
        item["training_time_s"],
        item["peak_reserved_bytes"],
        item["candidate_id"],
    ))
    eligible = [item for item in ranked if item["eligible"]]
    selected = eligible[: min(keep, len(eligible))]
    return {
        "selection_role": (
            "development_and_validation"
            if require_development_admissible else "validation_only"
        ),
        "selection_status": (
            "selected" if selected else "no_eligible_candidates"
        ),
        "selected_candidate_ids": [item["candidate_id"] for item in selected],
        "ranking": ranked,
    }


def summarize_confirmation_runs(
    baseline: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    *,
    min_quality_gain: float,
    max_regressed_categories: int,
    max_saturation_increase: float,
    min_format_compliance: float,
    max_baseline_latency_ratio: float,
    required_qualified_runs: int,
) -> dict[str, Any]:
    """Apply delivery-oriented gates to repeated confirmation runs."""

    if not runs:
        raise ValueError("confirmation runs 不能为空")
    if required_qualified_runs < 1 or required_qualified_runs > len(runs):
        raise ValueError("required_qualified_runs 必须位于有效运行数量范围内")

    baseline_quality = float(baseline.get("quality_score", 0.0))
    baseline_categories = baseline.get("category_scores", {})
    baseline_saturation = float(
        baseline.get("generated_tokens", {}).get("saturation_ratio", 0.0)
    )
    baseline_latency = float(baseline.get("mean_latency_ms", 0.0))
    if baseline_latency <= 0:
        raise ValueError("baseline mean_latency_ms 必须大于 0")

    summaries: list[dict[str, Any]] = []
    for run in runs:
        evaluation = run.get("evaluation", {})
        quality = float(evaluation.get("quality_score", 0.0))
        regressed = sorted(
            category
            for category, result in evaluation.get("category_scores", {}).items()
            if float(result.get("quality_score", 0.0))
            < float(baseline_categories.get(category, {}).get("quality_score", 0.0))
        )
        saturation = float(
            evaluation.get("generated_tokens", {}).get("saturation_ratio", 1.0)
        )
        format_ratio = float(
            evaluation.get("format_compliance", {}).get("compliance_ratio", 0.0)
        )
        latency_ratio = float(evaluation.get("mean_latency_ms", float("inf"))) / baseline_latency
        checks = {
            "artifact_verified": run.get("artifact_verified") is True,
            "quality_gain": quality - baseline_quality >= min_quality_gain,
            "category_regression": len(regressed) <= max_regressed_categories,
            "saturation": saturation - baseline_saturation <= max_saturation_increase,
            "format_compliance": format_ratio >= min_format_compliance,
            "latency": latency_ratio <= max_baseline_latency_ratio,
        }
        summaries.append({
            "seed": int(run["seed"]),
            "quality_score": quality,
            "quality_gain": quality - baseline_quality,
            "regressed_categories": regressed,
            "saturation_ratio": saturation,
            "saturation_increase": saturation - baseline_saturation,
            "format_compliance_ratio": format_ratio,
            "baseline_latency_ratio": latency_ratio,
            "artifact_verified": run.get("artifact_verified") is True,
            "artifact": run.get("artifact"),
            "checks": checks,
            "qualified": all(checks.values()),
        })

    qualified = [item for item in summaries if item["qualified"]]
    qualified.sort(key=lambda item: (
        -item["quality_score"],
        item["saturation_ratio"],
        item["baseline_latency_ratio"],
        item["seed"],
    ))
    quality_scores = [item["quality_score"] for item in summaries]
    confirmed = len(qualified) >= required_qualified_runs
    return {
        "schema_version": "distillation-confirmation-summary/v1",
        "status": "confirmed" if confirmed else "not_confirmed",
        "confirmed": confirmed,
        "required_qualified_runs": required_qualified_runs,
        "qualified_run_count": len(qualified),
        "run_count": len(summaries),
        "baseline": {
            "quality_score": baseline_quality,
            "saturation_ratio": baseline_saturation,
            "mean_latency_ms": baseline_latency,
        },
        "gates": {
            "min_quality_gain": min_quality_gain,
            "max_regressed_categories": max_regressed_categories,
            "max_saturation_increase": max_saturation_increase,
            "min_format_compliance": min_format_compliance,
            "max_baseline_latency_ratio": max_baseline_latency_ratio,
        },
        "mean_quality_score": sum(quality_scores) / len(quality_scores),
        "quality_score_range": max(quality_scores) - min(quality_scores),
        "selected_seed": qualified[0]["seed"] if confirmed else None,
        "selected_artifact": qualified[0]["artifact"] if confirmed else None,
        "runs": summaries,
    }


def select_prompt_candidate(
    results: Mapping[str, Mapping[str, Any]],
    *,
    control_id: str,
    max_teacher_quality_loss_count: int,
    max_teacher_saturation_ratio: float,
    max_teacher_mean_token_ratio: float,
    min_teacher_format_compliance: float,
    max_student_quality_loss_count: int,
    max_student_saturation_ratio: float,
    max_student_mean_token_ratio: float,
) -> dict[str, Any]:
    """Select a distillation prompt from paired development-only evidence."""

    if control_id not in results:
        raise ValueError("prompt canary 缺少 control")
    if min(max_teacher_quality_loss_count, max_student_quality_loss_count) < 0:
        raise ValueError("quality loss count 不能为负")
    control_teacher = results[control_id].get("teacher", {})
    control_count = int(control_teacher.get("evaluation_count", 0))
    control_passed = int(control_teacher.get("passed_count", 0))
    control_mean_tokens = float(
        control_teacher.get("generated_tokens", {}).get("mean_per_request", 0.0)
    )
    if control_count < 1 or control_mean_tokens <= 0:
        raise ValueError("prompt control 缺少有效 Teacher 证据")
    control_student = results[control_id].get("student", {})
    control_student_count = int(control_student.get("evaluation_count", 0))
    control_student_passed = int(control_student.get("passed_count", 0))
    control_student_mean_tokens = float(
        control_student.get("generated_tokens", {}).get("mean_per_request", 0.0)
    )
    if control_student_count != control_count or control_student_mean_tokens <= 0:
        raise ValueError("prompt control 缺少有效 Student 证据")

    ranking: list[dict[str, Any]] = []
    for candidate_id, result in results.items():
        teacher = result.get("teacher", {})
        candidate = result.get("candidate", {})
        if int(teacher.get("evaluation_count", 0)) != control_count:
            raise ValueError("prompt candidates 必须使用相同 development 记录数")
        passed_count = int(teacher.get("passed_count", 0))
        saturation = float(
            teacher.get("generated_tokens", {}).get("saturation_ratio", 1.0)
        )
        mean_tokens = float(
            teacher.get("generated_tokens", {}).get("mean_per_request", float("inf"))
        )
        mean_token_ratio = mean_tokens / control_mean_tokens
        format_ratio = float(
            teacher.get("format_compliance", {}).get("compliance_ratio", 0.0)
        )
        student = result.get("student", {})
        student_evidence = int(student.get("evaluation_count", 0)) == control_count
        student_passed = int(student.get("passed_count", 0))
        student_saturation = (
            float(student.get("generated_tokens", {}).get("saturation_ratio", 1.0))
            if student_evidence else None
        )
        student_mean_tokens = (
            float(student.get("generated_tokens", {}).get("mean_per_request", 0.0))
            if student_evidence else None
        )
        student_mean_token_ratio = (
            student_mean_tokens / control_student_mean_tokens
            if student_mean_tokens is not None else None
        )
        student_format = (
            float(student.get("format_compliance", {}).get("compliance_ratio", 0.0))
            if student_evidence else None
        )
        checks = {
            "distillation_candidate": candidate.get("role") == "candidate",
            "teacher_quality": (
                passed_count >= control_passed - max_teacher_quality_loss_count
            ),
            "teacher_saturation": saturation <= max_teacher_saturation_ratio,
            "teacher_length": mean_token_ratio <= max_teacher_mean_token_ratio,
            "teacher_format": format_ratio >= min_teacher_format_compliance,
            "student_evidence": student_evidence,
            "student_quality": (
                student_evidence
                and student_passed >= control_student_passed - max_student_quality_loss_count
            ),
            "student_saturation": (
                student_evidence
                and student_saturation <= max_student_saturation_ratio
            ),
            "student_length": (
                student_evidence
                and student_mean_token_ratio <= max_student_mean_token_ratio
            ),
        }
        ranking.append({
            "candidate_id": candidate_id,
            "role": candidate.get("role"),
            "teacher_quality_score": float(teacher.get("quality_score", 0.0)),
            "teacher_passed_count": passed_count,
            "teacher_quality_delta_count": passed_count - control_passed,
            "teacher_saturation_ratio": saturation,
            "teacher_mean_tokens": mean_tokens,
            "teacher_mean_token_ratio": mean_token_ratio,
            "teacher_format_compliance": format_ratio,
            "student_quality_score": float(student.get("quality_score", 0.0)),
            "student_passed_count": student_passed,
            "student_quality_delta_count": student_passed - control_student_passed,
            "student_saturation_ratio": student_saturation,
            "student_mean_tokens": student_mean_tokens,
            "student_mean_token_ratio": student_mean_token_ratio,
            "student_format_compliance": student_format,
            "checks": checks,
            "eligible": all(checks.values()),
        })
    ranking.sort(key=lambda item: (
        not item["eligible"],
        -item["teacher_quality_score"],
        item["teacher_saturation_ratio"],
        item["teacher_mean_tokens"],
        -item["teacher_format_compliance"],
        item["candidate_id"],
    ))
    eligible = [item for item in ranking if item["eligible"]]
    selected_id = eligible[0]["candidate_id"] if eligible else None
    status = "selected"
    if selected_id is None:
        control_ranking = next(
            item for item in ranking if item["candidate_id"] == control_id
        )
        control_checks = {
            name: passed
            for name, passed in control_ranking["checks"].items()
            if name != "distillation_candidate"
        }
        if all(control_checks.values()):
            selected_id = control_id
            status = "retain_control"
        else:
            status = "no_safe_prompt"
    return {
        "schema_version": "distillation-prompt-canary-summary/v1",
        "selection_role": "development_only",
        "status": status,
        "selected_candidate_id": selected_id,
        "control_id": control_id,
        "development_count": control_count,
        "gates": {
            "max_teacher_quality_loss_count": max_teacher_quality_loss_count,
            "max_teacher_saturation_ratio": max_teacher_saturation_ratio,
            "max_teacher_mean_token_ratio": max_teacher_mean_token_ratio,
            "min_teacher_format_compliance": min_teacher_format_compliance,
            "max_student_quality_loss_count": max_student_quality_loss_count,
            "max_student_saturation_ratio": max_student_saturation_ratio,
            "max_student_mean_token_ratio": max_student_mean_token_ratio,
        },
        "ranking": ranking,
    }


def resolve_prompt_candidate_id(selection: dict[str, Any]) -> str:
    """Resolve current and historical prompt-canary summaries safely."""

    selected_id = selection.get("selected_candidate_id")
    if isinstance(selected_id, str) and selected_id:
        return selected_id
    if selection.get("status") != "no_eligible_candidate":
        raise ValueError("prompt_check 没有选出安全提示词")
    control_id = selection.get("control_id")
    control_ranking = next(
        (
            item for item in selection.get("ranking", [])
            if item.get("candidate_id") == control_id
        ),
        None,
    )
    if control_ranking is None:
        raise ValueError("历史 prompt_check 缺少 control 证据")
    checks = {
        name: passed
        for name, passed in control_ranking.get("checks", {}).items()
        if name != "distillation_candidate"
    }
    if not checks or not all(checks.values()):
        raise ValueError("历史 prompt_check 的 control 未通过安全门槛")
    return str(control_id)


def run_g1_screening_matrix(
    torch_module: Any,
    *,
    plan: Mapping[str, Any],
    baseline_validation: Mapping[str, Any],
    common_run_kwargs: Mapping[str, Any],
    output_dir: str | Path,
    _candidate_runner: Any | None = None,
) -> dict[str, Any]:
    """Run or resume the G1 screen and persist each candidate immediately."""

    if _candidate_runner is None:
        from tools.distillation_project_runtime import run_fixed_trajectory_kd

        _candidate_runner = run_fixed_trajectory_kd

    if plan.get("schema_version") != "distillation-matrix/v1":
        raise ValueError("不支持的矩阵计划版本")
    plan_hash = str(plan.get("plan_hash", ""))
    run_label = str(plan.get("run_label", ""))
    if not plan_hash or not run_label:
        raise ValueError("矩阵计划缺少 plan_hash 或 run_label")
    selection = plan.get("selection_policy", {})
    screening_steps = int(selection.get("screening_steps", 0))
    screening_dose_steps = tuple(
        int(step) for step in selection.get("screening_dose_steps", (screening_steps,))
    )
    finalists = int(selection.get("finalists_per_family", 0))
    if (
        screening_steps < 1
        or finalists < 1
        or not screening_dose_steps
        or screening_dose_steps[-1] != screening_steps
    ):
        raise ValueError("矩阵计划缺少有效的 screening 配置")
    candidates = list(plan.get("families", {}).get("g1_fixed_trajectory", ()))
    if not candidates:
        raise ValueError("矩阵计划没有 G1 候选")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    forbidden_overrides = {
        "max_steps", "dose_steps", "max_grad_norm", "hard_label_weight",
        "persist_artifact", "artifact_path",
    }
    overlap = forbidden_overrides & set(common_run_kwargs)
    if overlap:
        raise ValueError(f"common_run_kwargs 不得覆盖矩阵变量：{sorted(overlap)}")
    config_hash = stable_text_hash((
        json.dumps(common_run_kwargs, sort_keys=True, ensure_ascii=False, default=str),
    ))
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        result_path = target / f"g1_matrix_{run_label}_{candidate_id}.json"
        if result_path.is_file():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                record.get("plan_hash") != plan_hash
                or record.get("common_config_hash") != config_hash
                or record.get("candidate") != candidate
            ):
                raise RuntimeError(
                    f"已有矩阵结果与当前计划不一致，请更换 run_label：{result_path}"
                )
        else:
            candidate_run_kwargs = dict(common_run_kwargs)
            for key in (
                "learning_rate", "optimizer_name", "max_grad_norm",
                "hard_label_weight", "adapter_config",
            ):
                if key in candidate:
                    candidate_run_kwargs[key] = candidate[key]
            evidence = _candidate_runner(
                torch_module,
                **candidate_run_kwargs,
                artifact_path=target / f"unused_{candidate_id}",
                max_steps=screening_steps,
                dose_steps=screening_dose_steps,
                persist_artifact=False,
            )
            record = {
                "schema_version": "distillation-matrix-candidate/v1",
                "record_type": "g1_validation_screen",
                "run_label": run_label,
                "plan_hash": plan_hash,
                "common_config_hash": config_hash,
                "candidate": dict(candidate),
                "selection_role": "validation_only",
                "evidence": evidence,
            }
            save_json_record(result_path, record)
        records.append(record)
    ranking_inputs = [
        {
            "candidate_id": record["candidate"]["candidate_id"],
            "validation": record["evidence"]["evaluation"],
            "development_selection": record["evidence"]["training"]["dose_selection"],
            "training_time_s": record["evidence"]["training"]["training_time_s"],
            "peak_reserved_bytes": record["evidence"]["training"]["peak_reserved_bytes"],
        }
        for record in records
    ]
    pre_confirmation_gates = selection.get("pre_confirmation_gates", {})
    ranking = rank_validation_candidates(
        baseline_validation,
        ranking_inputs,
        keep=finalists,
        min_quality_gain=float(selection.get("min_quality_gain", 0.0)),
        require_development_admissible=bool(
            pre_confirmation_gates.get("require_development_admissible", False)
        ),
        max_regressed_categories=int(
            pre_confirmation_gates.get("max_regressed_categories", 0)
        ),
        min_format_compliance=float(
            pre_confirmation_gates.get("min_format_compliance", 0.0)
        ),
        max_saturation_increase=pre_confirmation_gates.get("max_saturation_increase"),
        max_baseline_latency_ratio=pre_confirmation_gates.get(
            "max_baseline_latency_ratio"
        ),
    )
    summary = {
        "schema_version": "distillation-matrix-summary/v1",
        "record_type": "g1_validation_screen_summary",
        "run_label": run_label,
        "plan_hash": plan_hash,
        "common_config_hash": config_hash,
        "screening_gates": {
            "min_quality_gain": float(selection.get("min_quality_gain", 0.0)),
            **pre_confirmation_gates,
        },
        "candidate_paths": [
            str(target / f"g1_matrix_{run_label}_{item['candidate_id']}.json")
            for item in candidates
        ],
        **ranking,
    }
    summary_path = target / f"g1_matrix_summary_{run_label}.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise RuntimeError("已有 G1 矩阵摘要与当前结果不一致")
    else:
        save_json_record(summary_path, summary)
    return summary


__all__ = [
    "G1_MATRIX",
    "G2_MATRIX",
    "build_distillation_matrix_plan",
    "load_final_holdout",
    "rank_validation_candidates",
    "resolve_prompt_candidate_id",
    "select_prompt_candidate",
    "summarize_confirmation_runs",
    "run_g1_screening_matrix",
]
