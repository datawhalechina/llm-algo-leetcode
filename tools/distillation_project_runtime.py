"""Runtime helpers for the distilled-student delivery project.

The module owns workload validation, model-asset facts, conservative memory
admission, paired training stages, frozen evaluation and reusable delivery
checks. Project thresholds and evidence assembly remain visible in notebook 88.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tools.model_runtime import resolve_model
from tools.post_training_project_runtime import (
    extract_final_number,
    save_json_record,
    stable_text_hash,
)
from tools.project_runtime import (
    environment_preflight,
    load_environment_profile,
    read_exact_constraints,
)


def load_distillation_workload(path: str | Path) -> dict[str, Any]:
    """Load and validate the fixed train/development/evaluation contract."""

    source = Path(path)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("蒸馏 workload 不能为空")
    required = {"record_id", "split", "prompt"}
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"workload 第 {index + 1} 行缺少字段：{sorted(missing)}")
        if record["split"] not in {"train", "development", "evaluation"}:
            raise ValueError(f"未知 split：{record['split']!r}")
        if not str(record["prompt"]).strip():
            raise ValueError("prompt 不能为空")
        if record["split"] in {"train", "development"} and "reference_answer" in record:
            if not str(record["reference_answer"]).strip():
                raise ValueError(f"{record['split']} reference_answer 不能为空")
        if record["split"] in {"development", "evaluation"}:
            validator = record.get("validator", "terms")
            if validator not in {"choice", "exact", "terms", "final_number"}:
                raise ValueError(f"未知 {record['split']} validator：{validator!r}")
            if validator in {"exact", "final_number"} and not record.get("expected_exact"):
                raise ValueError(
                    f"{validator} {record['split']} 记录必须提供 expected_exact"
                )
            if validator in {"choice", "terms"} and not (
                record.get("expected_any")
                or record.get("expected_all")
                or record.get("expected_groups")
            ):
                raise ValueError(
                    f"{validator} evaluation 记录必须提供 expected_any 或 expected_all"
                )
            if validator == "choice" and not record.get("forbidden_any"):
                raise ValueError("choice evaluation 记录必须提供 forbidden_any")

    record_ids = [str(record["record_id"]) for record in records]
    prompt_hashes = [stable_text_hash((record["prompt"],)) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record_id 必须唯一")
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise ValueError("train/evaluation prompt 不能重复")

    train = [record for record in records if record["split"] == "train"]
    development = [record for record in records if record["split"] == "development"]
    evaluation = [record for record in records if record["split"] == "evaluation"]
    if not train or not evaluation:
        raise ValueError("workload 必须同时包含 train 和 evaluation")
    development_category_counts = Counter(
        str(record.get("category", "uncategorized")) for record in development
    )
    category_counts = Counter(
        str(record.get("category", "uncategorized")) for record in evaluation
    )
    scoring_contract = [
        {
            "record_id": record["record_id"],
            "category": record.get("category", "uncategorized"),
            "validator": record.get("validator", "terms"),
            "expected_exact": record.get("expected_exact", []),
            "expected_any": record.get("expected_any", []),
            "expected_all": record.get("expected_all", []),
            "expected_groups": record.get("expected_groups", []),
            "forbidden_any": record.get("forbidden_any", []),
        }
        for record in evaluation
    ]
    reference_answers = [
        str(record.get("reference_answer", "")) for record in train
    ]
    return {
        "source": str(source),
        "records": records,
        "train": train,
        "development": development,
        "evaluation": evaluation,
        "identity": {
            "record_count": len(records),
            "train_count": len(train),
            "development_count": len(development),
            "evaluation_count": len(evaluation),
            "development_category_counts": dict(sorted(development_category_counts.items())),
            "evaluation_category_counts": dict(sorted(category_counts.items())),
            "record_ids_hash": stable_text_hash(record_ids),
            "prompt_ids_hash": stable_text_hash(prompt_hashes),
            "train_prompt_ids_hash": stable_text_hash(
                stable_text_hash((record["prompt"],)) for record in train
            ),
            "train_reference_count": sum(bool(answer) for answer in reference_answers),
            "train_reference_answers_hash": stable_text_hash(reference_answers),
            "development_prompt_ids_hash": stable_text_hash(
                stable_text_hash((record["prompt"],)) for record in development
            ),
            "evaluation_prompt_ids_hash": stable_text_hash(
                stable_text_hash((record["prompt"],)) for record in evaluation
            ),
            "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "scoring_contract_sha256": stable_text_hash(
                (json.dumps(scoring_contract, sort_keys=True, ensure_ascii=False),)
            ),
        },
    }


def build_development_partition(
    workload: Mapping[str, Any],
    canary_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use an explicit development split or hold selected training canaries out."""

    train_records = [dict(record) for record in workload.get("train", ())]
    explicit_development = [
        dict(record) for record in workload.get("development", ())
    ]
    evaluation_records = [dict(record) for record in workload.get("evaluation", ())]
    if explicit_development:
        if canary_specs:
            raise ValueError("显式 development split 与 canary_specs 不能同时使用")
        if any(not str(record.get("reference_answer", "")).strip() for record in explicit_development):
            raise ValueError("显式 development 记录必须提供 reference_answer")
        category_counts = Counter(
            str(record.get("category", "uncategorized"))
            for record in explicit_development
        )
        if len(set(category_counts.values())) != 1:
            raise ValueError("显式 development split 必须按类别均衡")
        scoring_contract = [
            {key: record.get(key, []) for key in (
                "record_id", "category", "validator", "expected_exact",
                "expected_any", "expected_all", "expected_groups", "forbidden_any",
            )}
            for record in explicit_development
        ]
        identity = dict(workload["identity"])
        identity.update({
            "selected_train_count": len(train_records),
            "development_count": len(explicit_development),
            "development_category_counts": dict(sorted(category_counts.items())),
            "development_source_ids_hash": stable_text_hash(
                str(record["record_id"]) for record in explicit_development
            ),
            "development_scoring_contract_sha256": stable_text_hash((
                json.dumps(scoring_contract, sort_keys=True, ensure_ascii=False),
            )),
            "development_reference_answers_hash": stable_text_hash(
                str(record["reference_answer"]) for record in explicit_development
            ),
            "selected_train_prompt_ids_hash": stable_text_hash(
                stable_text_hash((record["prompt"],)) for record in train_records
            ),
        })
        return {
            "source": workload.get("source"),
            "train": train_records,
            "development": explicit_development,
            "evaluation": evaluation_records,
            "identity": identity,
        }

    by_id = {str(record["record_id"]): record for record in train_records}
    spec_ids = [str(spec.get("record_id", "")) for spec in canary_specs]
    if not spec_ids or any(not record_id for record_id in spec_ids):
        raise ValueError("development canary 必须提供非空 record_id")
    if len(spec_ids) != len(set(spec_ids)):
        raise ValueError("development canary record_id 必须唯一")
    missing = sorted(set(spec_ids) - set(by_id))
    if missing:
        raise ValueError(f"development canary 不在训练集中：{missing}")

    development: list[dict[str, Any]] = []
    for spec in canary_specs:
        record_id = str(spec["record_id"])
        source = by_id[record_id]
        validator = str(spec.get("validator", "terms"))
        scored = {
            "record_id": record_id.replace("train-", "dev-", 1),
            "source_record_id": record_id,
            "split": "development",
            "category": source.get("category", "uncategorized"),
            "prompt": source["prompt"],
            "reference_answer": str(source.get("reference_answer", "")),
            "validator": validator,
            "expected_exact": list(spec.get("expected_exact", ())),
            "expected_any": list(spec.get("expected_any", ())),
            "expected_all": list(spec.get("expected_all", ())),
            "expected_groups": [list(group) for group in spec.get("expected_groups", ())],
            "forbidden_any": list(spec.get("forbidden_any", ())),
        }
        if validator == "exact" and not scored["expected_exact"]:
            raise ValueError(f"{record_id} exact canary 缺少 expected_exact")
        if validator in {"choice", "terms"} and not (
            scored["expected_any"] or scored["expected_all"] or scored["expected_groups"]
        ):
            raise ValueError(f"{record_id} canary 缺少评分目标")
        if validator == "choice" and not scored["forbidden_any"]:
            raise ValueError(f"{record_id} choice canary 缺少 forbidden_any")
        development.append(scored)

    held_out = set(spec_ids)
    selected_train = [
        record for record in train_records if str(record["record_id"]) not in held_out
    ]
    category_counts = Counter(str(record["category"]) for record in development)
    if len(set(category_counts.values())) != 1:
        raise ValueError("development canary 必须按类别均衡抽样")
    scoring_contract = [
        {key: record.get(key, []) for key in (
            "record_id", "source_record_id", "category", "validator",
            "expected_exact", "expected_any", "expected_all",
            "expected_groups", "forbidden_any",
        )}
        for record in development
    ]
    identity = dict(workload["identity"])
    identity.update({
        "selected_train_count": len(selected_train),
        "development_count": len(development),
        "development_category_counts": dict(sorted(category_counts.items())),
        "development_source_ids_hash": stable_text_hash(spec_ids),
        "development_scoring_contract_sha256": stable_text_hash((
            json.dumps(scoring_contract, sort_keys=True, ensure_ascii=False),
        )),
        "development_reference_answers_hash": stable_text_hash(
            str(record["reference_answer"]) for record in development
        ),
        "selected_train_prompt_ids_hash": stable_text_hash(
            stable_text_hash((record["prompt"],)) for record in selected_train
        ),
    })
    return {
        "source": workload.get("source"),
        "train": selected_train,
        "development": development,
        "evaluation": evaluation_records,
        "identity": identity,
    }


def select_development_dose(
    baseline: Mapping[str, Any],
    dose_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one checkpoint without consulting the frozen evaluation split."""

    if not dose_results:
        raise ValueError("dose_results 不能为空")
    baseline_quality = float(baseline.get("quality_score", 0.0))
    baseline_nll = float(
        baseline.get("reference_completion", {}).get("mean_nll", float("inf"))
    )
    baseline_categories = baseline.get("category_scores", {})
    ranked: list[dict[str, Any]] = []
    for result in dose_results:
        development = result.get("development", {})
        regressed = sorted(
            category
            for category, score in development.get("category_scores", {}).items()
            if float(score.get("quality_score", 0.0))
            < float(baseline_categories.get(category, {}).get("quality_score", 0.0))
        )
        quality = float(development.get("quality_score", 0.0))
        completion_nll = float(
            development.get("reference_completion", {}).get("mean_nll", float("inf"))
        )
        ranked.append({
            "step": int(result["step"]),
            "quality_score": quality,
            "quality_gain": quality - baseline_quality,
            "completion_nll": completion_nll,
            "completion_nll_gain": (
                baseline_nll - completion_nll
                if math.isfinite(baseline_nll) and math.isfinite(completion_nll)
                else None
            ),
            "regressed_categories": regressed,
            "regressed_category_count": len(regressed),
            "admissible": quality >= baseline_quality and not regressed,
        })
    admissible = [item for item in ranked if item["admissible"]]
    pool = admissible or ranked
    selected = max(
        pool,
        key=lambda item: (
            item["admissible"],
            -item["regressed_category_count"],
            item["quality_score"],
            -item["completion_nll"],
            -item["step"],
        ),
    )
    return {
        "status": "selected" if selected["admissible"] else "diagnostic_only",
        "selected_step": selected["step"],
        "selected_admissible": selected["admissible"],
        "baseline_quality_score": baseline_quality,
        "baseline_completion_nll": (
            baseline_nll if math.isfinite(baseline_nll) else None
        ),
        "selection_order": [
            "semantic_admissibility",
            "regressed_category_count",
            "quality_score",
            "completion_nll",
            "smaller_step",
        ],
        "dose_summaries": ranked,
    }


def select_validation_dose(
    baseline: Mapping[str, Any],
    dose_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a training dose on the declared hyperparameter-validation split."""

    if not dose_results:
        raise ValueError("dose_results 不能为空")
    baseline_quality = float(baseline.get("quality_score", 0.0))
    baseline_categories = baseline.get("category_scores", {})
    ranked: list[dict[str, Any]] = []
    for result in dose_results:
        validation = result.get("validation", {})
        quality = float(validation.get("quality_score", 0.0))
        regressed = sorted(
            category
            for category, score in validation.get("category_scores", {}).items()
            if float(score.get("quality_score", 0.0))
            < float(baseline_categories.get(category, {}).get("quality_score", 0.0))
        )
        ranked.append({
            "step": int(result["step"]),
            "quality_score": quality,
            "quality_gain": quality - baseline_quality,
            "regressed_categories": regressed,
            "regressed_category_count": len(regressed),
            "admissible": quality >= baseline_quality and not regressed,
        })
    admissible = [item for item in ranked if item["admissible"]]
    pool = admissible or ranked
    selected = max(
        pool,
        key=lambda item: (
            item["admissible"],
            -item["regressed_category_count"],
            item["quality_score"],
            -item["step"],
        ),
    )
    return {
        "status": "selected" if admissible else "diagnostic_only",
        "selection_role": "hyperparameter_validation",
        "selected_step": selected["step"],
        "selected_admissible": selected["admissible"],
        "baseline_quality_score": baseline_quality,
        "selection_order": [
            "semantic_admissibility", "regressed_category_count",
            "quality_score", "smaller_step",
        ],
        "dose_summaries": sorted(ranked, key=lambda item: item["step"]),
    }


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_asset_manifest(
    model_path: str | Path,
    *,
    include_content_hash: bool = False,
) -> dict[str, Any]:
    """Return snapshot identity, byte count and optional weight-content hashes."""

    root = Path(model_path).resolve()
    weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(root.glob("*.bin"))
    if not (root / "config.json").is_file() or not weight_files:
        raise FileNotFoundError(f"模型快照不完整：{root}")
    inventory = []
    for path in weight_files:
        item = {"name": path.name, "size_bytes": path.stat().st_size}
        if include_content_hash:
            item["sha256"] = _file_sha256(path)
        inventory.append(item)
    identity_payload = json.dumps(inventory, sort_keys=True).encode("utf-8")
    manifest = {
        "path": str(root),
        "snapshot": root.name,
        "weight_bytes": sum(item["size_bytes"] for item in inventory),
        "weight_files": inventory,
        "inventory_sha256": hashlib.sha256(identity_payload).hexdigest(),
    }
    if include_content_hash:
        content_payload = json.dumps(
            [{"name": item["name"], "sha256": item["sha256"]} for item in inventory],
            sort_keys=True,
        ).encode("utf-8")
        manifest["weights_content_sha256"] = hashlib.sha256(
            content_payload
        ).hexdigest()
    return manifest


def inspect_distillation_api() -> dict[str, Any]:
    """Check the stable TRL distillation API required by G2."""

    import trl
    from trl import DistillationConfig, DistillationTrainer

    config_fields = {field.name for field in fields(DistillationConfig)}
    trainer_fields = set(inspect.signature(DistillationTrainer).parameters)
    required_config = {
        "teacher_model_name_or_path", "max_completion_length", "temperature",
        "beta", "use_vllm", "optim", "max_grad_norm", "warmup_steps",
        "lr_scheduler_type", "weight_decay",
    }
    required_trainer = {
        "model", "teacher_model", "args", "train_dataset", "peft_config",
    }
    missing = sorted((required_config - config_fields) | (required_trainer - trainer_fields))
    return {
        "trl_version": trl.__version__,
        "ready": not missing,
        "missing_parameters": missing,
        "config_parameters_hash": stable_text_hash(sorted(config_fields)),
        "trainer_parameters_hash": stable_text_hash(sorted(trainer_fields)),
    }


def run_profile_preflight(
    torch_module: Any,
    *,
    project_root: str | Path,
    profile_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply the registered environment Profile inside notebook 88."""

    root = Path(project_root)
    profile = load_environment_profile(root, profile_id)
    gpu = profile.get("gpu", {})
    isolation = profile.get("isolation", {})
    expected_versions = read_exact_constraints(
        root, tuple(profile.get("constraint_files", ()))
    )
    return environment_preflight(
        torch_module,
        required_packages=tuple(profile.get("required_modules", {}).keys()),
        require_gpu=bool(gpu.get("required", False)),
        min_gpu_memory_gb=gpu.get("minimum_memory_gb"),
        require_native_bf16=True,
        output_path=output_path,
        expected_distribution_versions=expected_versions,
        forbidden_distributions=tuple(profile.get("forbidden_distributions", ())),
        expected_torch_cuda=gpu.get("torch_cuda"),
        expected_torch_version=gpu.get("torch_version"),
        minimum_gpu_devices=int(gpu.get("minimum_devices", 1)),
        python_minimum=profile.get("python", {}).get("minimum"),
        python_maximum_exclusive=profile.get("python", {}).get("maximum_exclusive"),
        environment_profile=profile_id,
        require_clean_sys_path=bool(isolation.get("require_clean_sys_path", False)),
        require_user_site_disabled=bool(isolation.get("require_user_site_disabled", False)),
    )
def estimate_stage_memory_bytes(
    teacher_weight_bytes: int,
    student_weight_bytes: int,
    stage: str,
) -> int:
    """Estimate a conservative admission bound before loading either model."""

    if stage in {"preflight", "g0"}:
        # G0 evaluates teacher and student sequentially.
        return int(max(teacher_weight_bytes, student_weight_bytes) * 1.45 + 1024**3)
    if stage in {"g1", "g2"}:
        # Frozen teacher plus student weights, gradients, optimizer and activations.
        return int(teacher_weight_bytes * 1.20 + student_weight_bytes * 5.0 + 1024**3)
    if stage == "close":
        return 0
    raise ValueError(f"未知阶段：{stage}")


def build_memory_admission(
    torch_module: Any,
    *,
    teacher_weight_bytes: int,
    student_weight_bytes: int,
    stage: str,
    minimum_free_margin_bytes: int,
) -> dict[str, Any]:
    """Compare the stage estimate with currently free device memory."""

    if not torch_module.cuda.is_available():
        return {"admitted": False, "reason": "cuda_unavailable"}
    free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    estimated = estimate_stage_memory_bytes(
        teacher_weight_bytes, student_weight_bytes, stage
    )
    required = estimated + minimum_free_margin_bytes
    return {
        "stage": stage,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "estimated_peak_bytes": estimated,
        "minimum_free_margin_bytes": int(minimum_free_margin_bytes),
        "required_free_bytes": required,
        "admitted": int(free_bytes) >= required,
        "reason": None if int(free_bytes) >= required else "insufficient_free_memory",
    }


def score_expected_output(
    output: str,
    expected_any: Sequence[str] = (),
    expected_all: Sequence[str] = (),
    *,
    validator: str = "terms",
    expected_exact: Sequence[str] = (),
    forbidden_any: Sequence[str] = (),
    expected_groups: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Score strict text, final numbers, bounded terms, or exclusive choices."""

    normalized = " ".join(str(output).lower().replace("-", " ").split())
    normalized_exact = normalized.strip(" \t\r\n.,;:!?\"'`，。；：！？")

    def normalize_term(value: Any) -> str:
        return " ".join(str(value).lower().replace("-", " ").split())

    def contains_term(term: str) -> bool:
        if not term:
            return False
        pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
        return re.search(pattern, normalized) is not None

    exact_terms = [normalize_term(item) for item in expected_exact]
    any_terms = [normalize_term(item) for item in expected_any]
    all_terms = [normalize_term(item) for item in expected_all]
    forbidden_terms = [normalize_term(item) for item in forbidden_any]
    term_groups = [
        [normalize_term(item) for item in group] for group in expected_groups
    ]
    matched_exact = [item for item in exact_terms if item == normalized_exact]
    matched_any = [item for item in any_terms if contains_term(item)]
    matched_all = [item for item in all_terms if contains_term(item)]
    matched_forbidden = [item for item in forbidden_terms if contains_term(item)]
    matched_groups = [
        [item for item in group if contains_term(item)] for group in term_groups
    ]
    matched_number: list[str] = []
    if validator == "exact":
        passed = bool(matched_exact) and not matched_forbidden
    elif validator == "final_number":
        generated_number = extract_final_number(output)
        if generated_number is not None:
            try:
                generated_decimal = Decimal(generated_number)
                matched_number = [
                    item
                    for item in exact_terms
                    if Decimal(item.replace(",", "")) == generated_decimal
                ]
            except InvalidOperation:
                matched_number = []
        passed = bool(matched_number) and not matched_forbidden
    elif validator == "choice":
        any_passed = not any_terms or bool(matched_any)
        all_passed = len(matched_all) == len(all_terms)
        groups_passed = all(bool(group) for group in matched_groups)
        passed = any_passed and all_passed and groups_passed and not matched_forbidden
    elif validator == "terms":
        any_passed = not any_terms or bool(matched_any)
        all_passed = len(matched_all) == len(all_terms)
        groups_passed = all(bool(group) for group in matched_groups)
        passed = any_passed and all_passed and groups_passed and not matched_forbidden
    else:
        raise ValueError(f"未知 validator：{validator!r}")
    return {
        "passed": passed,
        "matched": matched_exact + matched_number + matched_any + matched_all + [
            item for group in matched_groups for item in group
        ],
        "matched_groups": matched_groups,
        "forbidden_matched": matched_forbidden,
    }


def final_number_format_compliant(output: str) -> bool:
    """Return whether output ends with one GSM8K-style numeric answer line."""

    return re.search(
        r"(?:^|\n)####\s*[-+]?(?:\d[\d,]*)(?:\.\d+)?\s*$",
        output.strip(),
    ) is not None


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("percentile values 不能为空")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def evaluate_loaded_model(
    torch_module: Any,
    *,
    model: Any,
    tokenizer: Any,
    evaluation_records: Sequence[Mapping[str, Any]],
    max_prompt_length: int,
    max_completion_length: int,
    latency_warmup_prompts: int = 1,
    latency_repeats: int = 3,
    system_prompt: str = "Answer concisely and follow the requested answer format.",
) -> dict[str, Any]:
    """Evaluate an already loaded model without changing its training state."""

    if not evaluation_records:
        raise ValueError("evaluation_records 不能为空")
    was_training = bool(model.training)
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = True
    model.eval()
    if latency_warmup_prompts < 0 or latency_repeats < 1:
        raise ValueError("latency_warmup_prompts 必须非负且 latency_repeats 至少为 1")

    def generate(record: Mapping[str, Any]) -> tuple[str, int, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(record["prompt"])},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        ).to("cuda")
        torch_module.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch_module.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_completion_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch_module.cuda.synchronize()
        elapsed = time.perf_counter() - generation_started
        completion_ids = generated[0, encoded["input_ids"].shape[1]:]
        output = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return output, int(completion_ids.numel()), elapsed

    for record in evaluation_records[:latency_warmup_prompts]:
        generate(record)

    rows: list[dict[str, Any]] = []
    generation_times_ms: list[float] = []
    category_totals: Counter[str] = Counter()
    category_passed: Counter[str] = Counter()
    for record in evaluation_records:
        repeated = [generate(record) for _ in range(latency_repeats)]
        outputs = [item[0] for item in repeated]
        generated_token_counts = [item[1] for item in repeated]
        latency_samples_ms = [item[2] * 1000 for item in repeated]
        output = outputs[0]
        score = score_expected_output(
            output,
            record.get("expected_any", ()),
            record.get("expected_all", ()),
            validator=record.get("validator", "terms"),
            expected_exact=record.get("expected_exact", ()),
            forbidden_any=record.get("forbidden_any", ()),
            expected_groups=record.get("expected_groups", ()),
        )
        format_required = record.get("validator", "terms") == "final_number"
        format_compliant = (
            final_number_format_compliant(output) if format_required else True
        )
        category = str(record.get("category", "uncategorized"))
        category_totals[category] += 1
        category_passed[category] += int(score["passed"])
        generation_times_ms.extend(latency_samples_ms)
        rows.append({
            "record_id": record["record_id"],
            "category": category,
            "validator": record.get("validator", "terms"),
            "output": output,
            "expected_exact": list(record.get("expected_exact", ())),
            "expected_any": list(record.get("expected_any", ())),
            "expected_all": list(record.get("expected_all", ())),
            "expected_groups": list(record.get("expected_groups", ())),
            "forbidden_any": list(record.get("forbidden_any", ())),
            "passed": score["passed"],
            "format_required": format_required,
            "format_compliant": format_compliant,
            "matched": score["matched"],
            "matched_groups": score["matched_groups"],
            "forbidden_matched": score["forbidden_matched"],
            "generated_tokens": generated_token_counts[0],
            "generated_token_samples": generated_token_counts,
            "deterministic_output": len(set(outputs)) == 1,
            "latency_samples_ms": [round(value, 3) for value in latency_samples_ms],
            "latency_ms": round(statistics.median(latency_samples_ms), 3),
        })

    category_scores = {
        category: {
            "passed_count": category_passed[category],
            "evaluation_count": count,
            "quality_score": category_passed[category] / count,
        }
        for category, count in sorted(category_totals.items())
    }
    generated_token_samples = [
        count for row in rows for count in row["generated_token_samples"]
    ]
    total_generation_s = sum(generation_times_ms) / 1000
    saturated_count = sum(
        int(row["generated_tokens"] >= max_completion_length) for row in rows
    )
    format_required_count = sum(int(row["format_required"]) for row in rows)
    format_compliant_count = sum(
        int(row["format_compliant"]) for row in rows if row["format_required"]
    )
    evidence = {
        "status": "ok",
        "quality_score": sum(int(row["passed"]) for row in rows) / len(rows),
        "passed_count": sum(int(row["passed"]) for row in rows),
        "evaluation_count": len(rows),
        "category_scores": category_scores,
        "latency_protocol": {
            "warmup_prompts": latency_warmup_prompts,
            "repeats": latency_repeats,
            "sample_count": len(generation_times_ms),
            "max_completion_length": max_completion_length,
        },
        "mean_latency_ms": round(statistics.mean(generation_times_ms), 3),
        "p50_latency_ms": round(_percentile(generation_times_ms, 0.50), 3),
        "p95_latency_ms": round(_percentile(generation_times_ms, 0.95), 3),
        "generated_tokens": {
            "sample_count": len(generated_token_samples),
            "mean_per_request": round(statistics.mean(generated_token_samples), 3),
            "total": sum(generated_token_samples),
            "saturated_evaluation_count": saturated_count,
            "saturation_ratio": saturated_count / len(rows),
        },
        "format_compliance": {
            "required_count": format_required_count,
            "compliant_count": format_compliant_count,
            "compliance_ratio": (
                format_compliant_count / format_required_count
                if format_required_count else 1.0
            ),
        },
        "output_token_throughput_per_s": round(
            sum(generated_token_samples) / total_generation_s, 3
        ),
        "mean_ms_per_output_token": round(
            sum(generation_times_ms) / sum(generated_token_samples), 3
        ),
        "rows": rows,
    }
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache
    if was_training:
        model.train()
    return evidence


def evaluate_completion_nll(
    torch_module: Any,
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    max_prompt_length: int,
    max_completion_length: int,
    system_prompt: str = "Answer concisely and follow the requested answer format.",
) -> dict[str, Any]:
    """Measure token-weighted NLL on held-out reference completions."""

    if not records:
        raise ValueError("completion NLL records 不能为空")
    was_training = bool(model.training)
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    model.eval()
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_tokens = 0
    for record in records:
        reference_answer = str(record.get("reference_answer", "")).strip()
        if not reference_answer:
            raise ValueError(f"{record.get('record_id')} 缺少 reference_answer")
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(record["prompt"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"][0]
        completion_ids = tokenizer(
            reference_answer,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        if tokenizer.eos_token_id is not None:
            completion_ids = torch_module.cat((
                completion_ids,
                torch_module.tensor([tokenizer.eos_token_id]),
            ))
        completion_ids = completion_ids[:max_completion_length]
        if not int(completion_ids.numel()):
            raise ValueError(f"{record.get('record_id')} 的参考答案没有 token")
        input_ids = torch_module.cat((prompt_ids, completion_ids)).unsqueeze(0).to("cuda")
        prediction_start = max(int(prompt_ids.numel()) - 1, 0)
        prediction_stop = int(input_ids.shape[1]) - 1
        with torch_module.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=torch_module.ones_like(input_ids),
            ).logits[:, prediction_start:prediction_stop, :].float()
        targets = input_ids[:, prediction_start + 1:prediction_stop + 1]
        loss_sum = torch_module.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        token_count = int(targets.numel())
        row_loss = float(loss_sum.detach().cpu())
        total_loss += row_loss
        total_tokens += token_count
        rows.append({
            "record_id": str(record["record_id"]),
            "token_count": token_count,
            "mean_nll": round(row_loss / token_count, 6),
        })
    mean_nll = total_loss / total_tokens
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache
    if was_training:
        model.train()
    return {
        "status": "ok",
        "reduction": "token_weighted_mean",
        "record_count": len(rows),
        "token_count": total_tokens,
        "mean_nll": round(mean_nll, 6),
        "perplexity": round(math.exp(min(mean_nll, 80.0)), 6),
        "rows": rows,
    }


def evaluate_model(
    torch_module: Any,
    *,
    model_path: str,
    evaluation_records: Sequence[Mapping[str, Any]],
    max_prompt_length: int,
    max_completion_length: int,
    dtype_name: str,
    latency_warmup_prompts: int = 1,
    latency_repeats: int = 3,
    reference_records: Sequence[Mapping[str, Any]] = (),
    system_prompt: str = "Answer concisely and follow the requested answer format.",
) -> dict[str, Any]:
    """Load a model and run deterministic quality and latency evaluation."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch_module, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    torch_module.cuda.synchronize()
    load_time_s = time.perf_counter() - started
    evidence = evaluate_loaded_model(
        torch_module,
        model=model,
        tokenizer=tokenizer,
        evaluation_records=evaluation_records,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        latency_warmup_prompts=latency_warmup_prompts,
        latency_repeats=latency_repeats,
        system_prompt=system_prompt,
    )
    if reference_records:
        evidence["reference_completion"] = evaluate_completion_nll(
            torch_module,
            model=model,
            tokenizer=tokenizer,
            records=reference_records,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            system_prompt=system_prompt,
        )
    evidence.update({
        "load_time_s": round(load_time_s, 3),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    })
    del model, tokenizer
    gc.collect()
    torch_module.cuda.empty_cache()
    return evidence


def evaluate_prompt_candidates(
    torch_module: Any,
    *,
    model_path: str,
    evaluation_records: Sequence[Mapping[str, Any]],
    prompt_candidates: Sequence[Mapping[str, Any]],
    max_prompt_length: int,
    max_completion_length: int,
    dtype_name: str,
    latency_warmup_prompts: int = 0,
    latency_repeats: int = 1,
) -> dict[str, dict[str, Any]]:
    """Evaluate several prompts while loading one model only once."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not prompt_candidates:
        raise ValueError("prompt_candidates 不能为空")
    candidate_ids = [str(item.get("candidate_id", "")) for item in prompt_candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("prompt candidate_id 不能为空")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("prompt candidate_id 必须唯一")
    if any(not str(item.get("system_prompt", "")).strip() for item in prompt_candidates):
        raise ValueError("prompt candidate 必须提供 system_prompt")

    dtype = getattr(torch_module, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    torch_module.cuda.synchronize()
    load_time_s = time.perf_counter() - started

    results: dict[str, dict[str, Any]] = {}
    for candidate in prompt_candidates:
        candidate_id = str(candidate["candidate_id"])
        evidence = evaluate_loaded_model(
            torch_module,
            model=model,
            tokenizer=tokenizer,
            evaluation_records=evaluation_records,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            latency_warmup_prompts=latency_warmup_prompts,
            latency_repeats=latency_repeats,
            system_prompt=str(candidate["system_prompt"]),
        )
        evidence["candidate"] = dict(candidate)
        results[candidate_id] = evidence

    metadata = {
        "load_time_s": round(load_time_s, 3),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
    }
    for evidence in results.values():
        evidence["shared_model_load"] = metadata
    del model, tokenizer
    gc.collect()
    torch_module.cuda.empty_cache()
    return results


def assess_g0_readiness(
    evidence: Mapping[str, Any],
    *,
    min_teacher_quality: float,
    min_teacher_category_quality: float,
    max_teacher_saturation_ratio: float,
    min_teacher_student_gap: float = 0.0,
) -> dict[str, Any]:
    """Decide whether the measured teacher is a usable distillation upper bound."""

    teacher = evidence.get("teacher", {})
    issues: list[str] = []
    quality = float(teacher.get("quality_score", 0.0))
    if quality < min_teacher_quality:
        issues.append("teacher_quality_below_gate")
    student_quality = float(evidence.get("student", {}).get("quality_score", 0.0))
    teacher_student_gap = quality - student_quality
    if teacher_student_gap < min_teacher_student_gap:
        issues.append("teacher_student_gap_below_gate")
    category_scores = teacher.get("category_scores", {})
    weak_categories = sorted(
        name
        for name, result in category_scores.items()
        if float(result.get("quality_score", 0.0)) < min_teacher_category_quality
    )
    if weak_categories:
        issues.append("teacher_category_below_gate")
    rows = teacher.get("rows", ())
    nondeterministic = sorted(
        str(row.get("record_id"))
        for row in rows
        if row.get("deterministic_output") is not True
    )
    if nondeterministic:
        issues.append("teacher_output_nondeterministic")
    saturation_ratio = float(
        teacher.get("generated_tokens", {}).get("saturation_ratio", 1.0)
    )
    if saturation_ratio > max_teacher_saturation_ratio:
        issues.append("teacher_output_saturation_above_gate")
    return {
        "ready": not issues,
        "issues": issues,
        "teacher_quality": quality,
        "student_quality": student_quality,
        "teacher_student_gap": teacher_student_gap,
        "weak_categories": weak_categories,
        "nondeterministic_record_ids": nondeterministic,
        "teacher_saturation_ratio": saturation_ratio,
        "gates": {
            "min_teacher_quality": min_teacher_quality,
            "min_teacher_category_quality": min_teacher_category_quality,
            "max_teacher_saturation_ratio": max_teacher_saturation_ratio,
            "min_teacher_student_gap": min_teacher_student_gap,
        },
    }


def _candidate_evidence(
    *,
    name: str,
    artifact_path: str | Path,
    evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine training, artifact and frozen-evaluation evidence."""

    artifact = model_asset_manifest(artifact_path, include_content_hash=True)
    return {
        "name": name,
        "status": "ok",
        "training": dict(training),
        "artifact": artifact,
        "artifact_verified": bool(artifact.get("weights_content_sha256")),
        "load_status": "loaded",
        "evaluation": dict(evaluation),
    }


def _ephemeral_candidate_evidence(
    *,
    name: str,
    evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Return screening evidence without pretending an artifact was delivered."""

    return {
        "name": name,
        "status": "ok",
        "training": dict(training),
        "artifact": {
            "persisted": False,
            "path": None,
            "reason": "matrix_screening_candidate",
        },
        "artifact_verified": False,
        "load_status": "evaluated_in_memory",
        "evaluation": dict(evaluation),
    }


def build_distillation_diagnostics(
    student_baseline: Mapping[str, Any],
    candidate_evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain whether optimization proxies agree with generated-answer quality."""

    baseline_rows = {
        str(row["record_id"]): row for row in student_baseline.get("rows", ())
    }
    candidate_rows = {
        str(row["record_id"]): row for row in candidate_evaluation.get("rows", ())
    }
    if not baseline_rows or set(baseline_rows) != set(candidate_rows):
        raise ValueError("诊断要求基线与候选包含相同的逐题记录")

    improved_ids: list[str] = []
    regressed_ids: list[str] = []
    format_improved_ids: list[str] = []
    format_regressed_ids: list[str] = []
    for record_id in sorted(baseline_rows):
        baseline_row = baseline_rows[record_id]
        candidate_row = candidate_rows[record_id]
        if not baseline_row.get("passed") and candidate_row.get("passed"):
            improved_ids.append(record_id)
        elif baseline_row.get("passed") and not candidate_row.get("passed"):
            regressed_ids.append(record_id)
        if not baseline_row.get("format_compliant") and candidate_row.get("format_compliant"):
            format_improved_ids.append(record_id)
        elif baseline_row.get("format_compliant") and not candidate_row.get("format_compliant"):
            format_regressed_ids.append(record_id)

    baseline_categories = student_baseline.get("category_scores", {})
    candidate_categories = candidate_evaluation.get("category_scores", {})
    category_deltas = {
        category: round(
            float(candidate_categories[category]["quality_score"])
            - float(baseline_categories[category]["quality_score"]),
            6,
        )
        for category in sorted(set(baseline_categories) & set(candidate_categories))
    }

    def metric_delta(path: Sequence[str]) -> float:
        baseline_value: Any = student_baseline
        candidate_value: Any = candidate_evaluation
        for key in path:
            baseline_value = baseline_value[key]
            candidate_value = candidate_value[key]
        return round(float(candidate_value) - float(baseline_value), 6)

    dose_summaries = list(training.get("dose_selection", {}).get("dose_summaries", ()))
    nll_improved = bool(
        dose_summaries
        and float(dose_summaries[-1].get("completion_nll_gain", 0.0)) > 0
    )
    development_quality_improved = bool(
        dose_summaries
        and float(dose_summaries[-1].get("quality_gain", 0.0)) > 0
    )
    validation_quality_delta = metric_delta(("quality_score",))
    return {
        "schema_version": "distillation-diagnostics/v1",
        "record_count": len(baseline_rows),
        "quality_transition": {
            "improved_count": len(improved_ids),
            "regressed_count": len(regressed_ids),
            "unchanged_count": len(baseline_rows) - len(improved_ids) - len(regressed_ids),
            "improved_record_ids": improved_ids,
            "regressed_record_ids": regressed_ids,
            "category_quality_deltas": category_deltas,
        },
        "format_transition": {
            "improved_count": len(format_improved_ids),
            "regressed_count": len(format_regressed_ids),
            "improved_record_ids": format_improved_ids,
            "regressed_record_ids": format_regressed_ids,
        },
        "metric_deltas": {
            "quality_score": validation_quality_delta,
            "format_compliance_ratio": metric_delta(("format_compliance", "compliance_ratio")),
            "saturation_ratio": metric_delta(("generated_tokens", "saturation_ratio")),
            "mean_generated_tokens": metric_delta(("generated_tokens", "mean_per_request")),
            "mean_latency_ms": metric_delta(("mean_latency_ms",)),
            "mean_ms_per_output_token": metric_delta(("mean_ms_per_output_token",)),
        },
        "proxy_alignment": {
            "reference_nll_improved": nll_improved,
            "development_quality_improved": development_quality_improved,
            "validation_quality_improved": validation_quality_delta > 0,
            "proxy_task_mismatch": nll_improved and validation_quality_delta <= 0,
        },
        "training_observability": {
            "metric_history_available": bool(training.get("metric_history")),
            "trajectory_summary_available": bool(training.get("trajectory_summary")),
            "adapter_update_available": bool(
                training.get("adapter", {}).get("update_diagnostics")
            ),
        },
    }


def run_fixed_trajectory_kd(
    torch_module: Any,
    *,
    teacher_path: str,
    student_path: str,
    train_records: Sequence[Mapping[str, Any]],
    development_records: Sequence[Mapping[str, Any]],
    development_baseline: Mapping[str, Any],
    evaluation_records: Sequence[Mapping[str, Any]],
    artifact_path: str | Path,
    max_prompt_length: int,
    max_completion_length: int,
    max_steps: int,
    dose_steps: Sequence[int],
    learning_rate: float,
    temperature: float,
    optimizer_name: str,
    max_grad_norm: float | None,
    warmup_ratio: float,
    weight_decay: float,
    hard_label_weight: float,
    dtype_name: str,
    gradient_checkpointing: bool,
    seed: int,
    latency_warmup_prompts: int = 1,
    latency_repeats: int = 3,
    system_prompt: str = "Answer concisely and follow the requested answer format.",
    persist_artifact: bool = True,
    checkpoint_validation_records: Sequence[Mapping[str, Any]] | None = None,
    checkpoint_validation_baseline: Mapping[str, Any] | None = None,
    adapter_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one student on fixed trajectories with soft and hard targets."""

    from transformers import (
        Adafactor,
        AutoModelForCausalLM,
        AutoTokenizer,
        get_constant_schedule_with_warmup,
    )

    if not train_records:
        raise ValueError("G1 训练记录不能为空")
    if optimizer_name not in {"adafactor", "adamw", "sgd"}:
        raise ValueError(f"G1 不支持 optimizer={optimizer_name!r}")
    if not 0.0 <= hard_label_weight <= 1.0:
        raise ValueError("hard_label_weight 必须位于 [0, 1]")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm 必须大于 0")
    normalized_doses = sorted({int(step) for step in dose_steps})
    if not development_records or not development_baseline:
        raise ValueError("G1 剂量选择需要 development 记录与基线")
    if not normalized_doses or normalized_doses[-1] != max_steps or normalized_doses[0] < 1:
        raise ValueError("dose_steps 必须以 max_steps 结尾且均为正整数")
    if (checkpoint_validation_records is None) != (checkpoint_validation_baseline is None):
        raise ValueError("checkpoint validation records 与 baseline 必须同时提供")
    if adapter_config is not None:
        required_adapter_keys = {"rank", "alpha", "dropout", "target_modules"}
        missing_adapter_keys = required_adapter_keys - set(adapter_config)
        if missing_adapter_keys:
            raise ValueError(f"LoRA 配置缺少字段：{sorted(missing_adapter_keys)}")
        if int(adapter_config["rank"]) < 1 or int(adapter_config["alpha"]) < 1:
            raise ValueError("LoRA rank 与 alpha 必须为正整数")
        if not 0.0 <= float(adapter_config["dropout"]) < 1.0:
            raise ValueError("LoRA dropout 必须位于 [0, 1)")
        if not list(adapter_config["target_modules"]):
            raise ValueError("LoRA target_modules 不能为空")
    output = Path(artifact_path)
    if persist_artifact and output.exists():
        raise FileExistsError(f"G1 学生产物已存在：{output}")

    torch_module.manual_seed(seed)
    dtype = getattr(torch_module, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    ).to("cuda")
    teacher.eval()
    teacher.requires_grad_(False)

    fixed_rows: list[dict[str, Any]] = []
    for record in train_records:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(record["prompt"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            rendered, return_tensors="pt", truncation=True,
            max_length=max_prompt_length,
        ).to("cuda")
        reference_answer = str(record.get("reference_answer", "")).strip()
        reference_token_count = 0
        if reference_answer:
            reference_ids = tokenizer(
                reference_answer,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            if tokenizer.eos_token_id is not None:
                reference_ids = torch_module.cat((
                    reference_ids,
                    torch_module.tensor([tokenizer.eos_token_id]),
                ))
            reference_token_count = int(reference_ids.numel())
        if reference_answer:
            if reference_token_count <= max_completion_length:
                completion_ids = reference_ids.detach().cpu()
                trajectory_source = "reference_answer"
            else:
                final_number = extract_final_number(reference_answer)
                if final_number is None:
                    raise RuntimeError(
                        f"超长参考答案缺少最终数值：{record['record_id']}"
                    )
                suffix_ids = tokenizer(
                    f"\n#### {final_number}",
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]
                if tokenizer.eos_token_id is not None:
                    suffix_ids = torch_module.cat((
                        suffix_ids,
                        torch_module.tensor([tokenizer.eos_token_id]),
                    ))
                rationale = reference_answer.split("####", 1)[0].strip()
                rationale_ids = tokenizer(
                    rationale,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]
                rationale_budget = max_completion_length - int(suffix_ids.numel())
                if rationale_budget < 1:
                    raise RuntimeError("max_completion_length 无法容纳最终答案")
                completion_ids = torch_module.cat((
                    rationale_ids[:rationale_budget], suffix_ids,
                )).detach().cpu()
                trajectory_source = "reference_answer_compacted"
        else:
            with torch_module.inference_mode():
                generated = teacher.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_completion_length,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = generated[
                0, encoded["input_ids"].shape[1]:
            ].detach().cpu()
            trajectory_source = "teacher_generation"
        fixed_rows.append({
            "record_id": str(record["record_id"]),
            "prompt_ids": encoded["input_ids"][0].detach().cpu(),
            "completion_ids": completion_ids,
            "completion_hash": stable_text_hash((completion_ids.tolist(),)),
            "trajectory_source": trajectory_source,
            "reference_token_count": reference_token_count,
        })
        del encoded, completion_ids
    random.Random(seed).shuffle(fixed_rows)

    student = AutoModelForCausalLM.from_pretrained(
        student_path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    ).to("cuda")
    adapter_metadata = None
    if adapter_config is not None:
        from peft import LoraConfig, TaskType, get_peft_model

        student = get_peft_model(
            student,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(adapter_config["rank"]),
                lora_alpha=int(adapter_config["alpha"]),
                lora_dropout=float(adapter_config["dropout"]),
                target_modules=list(adapter_config["target_modules"]),
                bias="none",
            ),
        )
        adapter_metadata = {
            "type": "lora",
            "rank": int(adapter_config["rank"]),
            "alpha": int(adapter_config["alpha"]),
            "dropout": float(adapter_config["dropout"]),
            "target_modules": list(adapter_config["target_modules"]),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in student.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(parameter.numel() for parameter in student.parameters()),
        }
    student.train()
    if gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.config.use_cache = False
        if adapter_config is not None and hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    if optimizer_name == "adafactor":
        optimizer = Adafactor(
            student.parameters(),
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch_module.optim.AdamW(
            (parameter for parameter in student.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch_module.optim.SGD(
            (parameter for parameter in student.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    warmup_steps = round(max_steps * warmup_ratio)
    scheduler = (
        get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        if warmup_steps > 0
        else torch_module.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    )

    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    loss_history: list[dict[str, float | int]] = []
    dose_results: list[dict[str, Any]] = []
    selected_state: dict[str, Any] | None = None
    selected_step: int | None = None
    started = time.perf_counter()
    for step in range(max_steps):
        row = fixed_rows[step % len(fixed_rows)]
        prompt_ids = row["prompt_ids"].to("cuda")
        completion_ids = row["completion_ids"].to("cuda")
        input_ids = torch_module.cat((prompt_ids, completion_ids)).unsqueeze(0)
        attention_mask = torch_module.ones_like(input_ids)
        prediction_start = max(int(prompt_ids.numel()) - 1, 0)
        prediction_stop = int(input_ids.shape[1]) - 1
        if prediction_stop <= prediction_start:
            raise RuntimeError("G1 固定轨迹没有可训练的 completion token")

        optimizer.zero_grad(set_to_none=True)
        with torch_module.inference_mode():
            teacher_logits = teacher(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[:, prediction_start:prediction_stop, :]
        student_logits = student(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits[:, prediction_start:prediction_stop, :]
        teacher_prob = torch_module.softmax(teacher_logits.float() / temperature, dim=-1)
        student_log_prob = torch_module.log_softmax(student_logits.float() / temperature, dim=-1)
        distillation_loss = torch_module.nn.functional.kl_div(
            student_log_prob, teacher_prob, reduction="batchmean"
        ) * (temperature**2) / max(prediction_stop - prediction_start, 1)
        target_ids = input_ids[:, prediction_start + 1:prediction_stop + 1]
        hard_label_loss = (
            torch_module.nn.functional.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                target_ids.reshape(-1),
            )
            if hard_label_weight > 0.0
            else student_logits.new_zeros(())
        )
        loss = (
            (1.0 - hard_label_weight) * distillation_loss
            + hard_label_weight * hard_label_loss
        )
        loss.backward()
        grad_norm = torch_module.nn.utils.clip_grad_norm_(
            student.parameters(),
            max_grad_norm if max_grad_norm is not None else float("inf"),
        )
        optimizer.step()
        scheduler.step()
        loss_history.append({
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "distillation_loss": float(distillation_loss.detach().cpu()),
            "hard_label_loss": float(hard_label_loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        })
        completed_step = step + 1
        if completed_step in normalized_doses:
            development = evaluate_loaded_model(
                torch_module,
                model=student,
                tokenizer=tokenizer,
                evaluation_records=development_records,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                latency_warmup_prompts=0,
                latency_repeats=1,
                system_prompt=system_prompt,
            )
            development["reference_completion"] = evaluate_completion_nll(
                torch_module,
                model=student,
                tokenizer=tokenizer,
                records=development_records,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                system_prompt=system_prompt,
            )
            dose_result = {"step": completed_step, "development": development}
            if checkpoint_validation_records is not None:
                dose_result["validation"] = evaluate_loaded_model(
                    torch_module,
                    model=student,
                    tokenizer=tokenizer,
                    evaluation_records=checkpoint_validation_records,
                    max_prompt_length=max_prompt_length,
                    max_completion_length=max_completion_length,
                    latency_warmup_prompts=0,
                    latency_repeats=1,
                    system_prompt=system_prompt,
                )
            dose_results.append(dose_result)
            current_selection = (
                select_validation_dose(checkpoint_validation_baseline, dose_results)
                if checkpoint_validation_baseline is not None
                else select_development_dose(development_baseline, dose_results)
            )
            if current_selection["selected_step"] != selected_step:
                selected_step = int(current_selection["selected_step"])
                if selected_step == completed_step:
                    selected_state = {
                        name: value.detach().cpu().clone()
                        for name, value in student.state_dict().items()
                    }

    torch_module.cuda.synchronize()
    training_time_s = time.perf_counter() - started
    peak_allocated = int(torch_module.cuda.max_memory_allocated())
    peak_reserved = int(torch_module.cuda.max_memory_reserved())
    dose_selection = (
        select_validation_dose(checkpoint_validation_baseline, dose_results)
        if checkpoint_validation_baseline is not None
        else select_development_dose(development_baseline, dose_results)
    )
    if selected_state is None or selected_step != dose_selection["selected_step"]:
        raise RuntimeError("G1 未保留到选中的剂量快照")
    student.load_state_dict(selected_state)
    student.config.use_cache = True
    if adapter_config is not None:
        student = student.merge_and_unload()
        student.config.use_cache = True
    if persist_artifact:
        output.mkdir(parents=True, exist_ok=False)
        student.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
    training = {
        "method": (
            "fixed_trajectory_lora_blended_kd"
            if adapter_config is not None
            else (
                "fixed_trajectory_forward_kl"
                if hard_label_weight == 0.0
                else "fixed_trajectory_blended_kd"
            )
        ),
        "trajectory_mode": "fixed",
        "optimizer": optimizer_name,
        "train_steps": max_steps,
        "dose_steps": normalized_doses,
        "selected_step": dose_selection["selected_step"],
        "dose_selection": dose_selection,
        "development_results": dose_results,
        "temperature": temperature,
        "beta": 0.0,
        "max_grad_norm": max_grad_norm,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "hard_label_weight": hard_label_weight,
        "adapter": adapter_metadata,
        "loss_history": loss_history,
        "selected_step_loss": loss_history[dose_selection["selected_step"] - 1]["loss"],
        "last_executed_loss": loss_history[-1]["loss"],
        "training_time_s": round(training_time_s, 3),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "trajectory_count": len(fixed_rows),
        "trajectory_source_counts": dict(Counter(
            row["trajectory_source"] for row in fixed_rows
        )),
        "reference_overflow_record_ids": sorted(
            row["record_id"]
            for row in fixed_rows
            if row["trajectory_source"] == "reference_answer_compacted"
        ),
        "trajectory_hash": stable_text_hash(
            [row["record_id"] + ":" + row["completion_hash"] for row in fixed_rows]
        ),
    }
    del selected_state, scheduler, optimizer, teacher
    gc.collect()
    torch_module.cuda.empty_cache()
    if persist_artifact:
        del student, tokenizer
        gc.collect()
        torch_module.cuda.empty_cache()
        evaluation = evaluate_model(
            torch_module,
            model_path=str(output),
            evaluation_records=evaluation_records,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            dtype_name=dtype_name,
            latency_warmup_prompts=latency_warmup_prompts,
            latency_repeats=latency_repeats,
            system_prompt=system_prompt,
        )
        return _candidate_evidence(
            name="g1_fixed_trajectory", artifact_path=output,
            evaluation=evaluation, training=training,
        )
    evaluation = evaluate_loaded_model(
        torch_module,
        model=student,
        tokenizer=tokenizer,
        evaluation_records=evaluation_records,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        latency_warmup_prompts=latency_warmup_prompts,
        latency_repeats=latency_repeats,
        system_prompt=system_prompt,
    )
    del student, tokenizer
    gc.collect()
    torch_module.cuda.empty_cache()
    return _ephemeral_candidate_evidence(
        name="g1_fixed_trajectory", evaluation=evaluation, training=training,
    )


def run_on_policy_distillation(
    torch_module: Any,
    *,
    teacher_path: str,
    student_path: str,
    train_records: Sequence[Mapping[str, Any]],
    development_records: Sequence[Mapping[str, Any]],
    development_baseline: Mapping[str, Any],
    evaluation_records: Sequence[Mapping[str, Any]],
    artifact_path: str | Path,
    max_prompt_length: int,
    max_completion_length: int,
    max_steps: int,
    dose_steps: Sequence[int],
    learning_rate: float,
    temperature: float,
    beta: float,
    optimizer_name: str,
    max_grad_norm: float,
    warmup_ratio: float,
    weight_decay: float,
    dtype_name: str,
    gradient_checkpointing: bool,
    seed: int,
    adapter_config: Mapping[str, Any] | None = None,
    latency_warmup_prompts: int = 1,
    latency_repeats: int = 3,
    system_prompt: str = "Answer concisely and follow the requested answer format.",
) -> dict[str, Any]:
    """Run stable TRL on-policy distillation and evaluate the saved student."""

    from datasets import Dataset
    from transformers import AutoTokenizer, TrainerCallback
    from trl import DistillationConfig, DistillationTrainer

    if not train_records:
        raise ValueError("G2 训练记录不能为空")
    if optimizer_name not in {"adafactor", "adamw", "sgd"}:
        raise ValueError(f"G2 不支持 optimizer={optimizer_name!r}")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm 必须大于 0")
    normalized_doses = sorted({int(step) for step in dose_steps})
    if not development_records or not development_baseline:
        raise ValueError("G2 剂量选择需要 development 记录与基线")
    if not normalized_doses or normalized_doses[-1] != max_steps or normalized_doses[0] < 1:
        raise ValueError("dose_steps 必须以 max_steps 结尾且均为正整数")
    output = Path(artifact_path)
    if output.exists():
        raise FileExistsError(f"G2 学生产物已存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list([
        {
            "record_id": str(record["record_id"]),
            "reference_answer": str(record.get("reference_answer", "")),
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(record["prompt"])},
            ],
        }
        for record in train_records
    ])
    peft_config = None
    if adapter_config is not None:
        from peft import LoraConfig

        required_adapter_keys = {"rank", "alpha", "dropout", "target_modules"}
        missing_adapter_keys = required_adapter_keys - set(adapter_config)
        if missing_adapter_keys:
            raise ValueError(
                f"adapter_config 缺少字段：{sorted(missing_adapter_keys)}"
            )
        if int(adapter_config["rank"]) < 1 or int(adapter_config["alpha"]) < 1:
            raise ValueError("LoRA rank 与 alpha 必须为正整数")
        if not 0.0 <= float(adapter_config["dropout"]) < 1.0:
            raise ValueError("LoRA dropout 必须位于 [0, 1)")
        if not list(adapter_config["target_modules"]):
            raise ValueError("LoRA target_modules 不能为空")
        peft_config = LoraConfig(
            r=int(adapter_config["rank"]),
            lora_alpha=int(adapter_config["alpha"]),
            lora_dropout=float(adapter_config["dropout"]),
            target_modules=list(adapter_config["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
    warmup_steps = round(max_steps * warmup_ratio)
    trainer_optimizer = (
        "adamw_torch_fused" if optimizer_name == "adamw" else optimizer_name
    )
    config_kwargs = dict(
        output_dir=str(output),
        teacher_model_name_or_path=teacher_path,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        learning_rate=learning_rate,
        temperature=temperature,
        beta=beta,
        use_vllm=False,
        bf16=dtype_name == "bfloat16",
        fp16=dtype_name == "float16",
        gradient_checkpointing=gradient_checkpointing,
        optim=trainer_optimizer,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
        model_init_kwargs={
            "dtype": dtype_name,
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
        teacher_model_init_kwargs={
            "dtype": dtype_name,
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
    )
    if warmup_steps > 0:
        config_kwargs["lr_scheduler_type"] = "constant_with_warmup"
    args = DistillationConfig(**config_kwargs)

    class DoseSelectionCallback(TrainerCallback):
        """Evaluate development canaries and retain one CPU checkpoint snapshot."""

        def __init__(self) -> None:
            self.results: list[dict[str, Any]] = []
            self.selected_step: int | None = None
            self.selected_state: dict[str, Any] | None = None

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            completed_step = int(state.global_step)
            if completed_step not in normalized_doses:
                return control
            model = kwargs.get("model")
            if model is None:
                raise RuntimeError("G2 剂量回调未收到模型")
            development = evaluate_loaded_model(
                torch_module,
                model=model,
                tokenizer=tokenizer,
                evaluation_records=development_records,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                latency_warmup_prompts=0,
                latency_repeats=1,
                system_prompt=system_prompt,
            )
            development["reference_completion"] = evaluate_completion_nll(
                torch_module,
                model=model,
                tokenizer=tokenizer,
                records=development_records,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                system_prompt=system_prompt,
            )
            self.results.append({"step": completed_step, "development": development})
            selection = select_development_dose(development_baseline, self.results)
            if selection["selected_step"] != self.selected_step:
                self.selected_step = int(selection["selected_step"])
                if self.selected_step == completed_step:
                    if peft_config is None:
                        self.selected_state = {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        }
                    else:
                        from peft import get_peft_model_state_dict

                        self.selected_state = {
                            name: value.detach().cpu().clone()
                            for name, value in get_peft_model_state_dict(model).items()
                        }
            return control

    dose_callback = DoseSelectionCallback()
    trajectory_rows: list[dict[str, Any]] = []

    class EvidenceDistillationTrainer(DistillationTrainer):
        """Capture bounded, task-level evidence from the actual online trajectories."""

        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            generated = super()._generate_and_score_completions(inputs)
            completion_ids = generated["completion_ids"].detach().cpu()
            completion_mask = generated["completion_mask"].detach().cpu()
            for index, source in enumerate(inputs):
                valid_ids = completion_ids[index][completion_mask[index].bool()].tolist()
                output_text = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
                reference_answer = str(source.get("reference_answer", ""))
                expected_number = extract_final_number(reference_answer)
                generated_number = extract_final_number(output_text)
                trajectory_rows.append({
                    "step": int(self.state.global_step) + 1,
                    "record_id": str(source.get("record_id", "unknown")),
                    "completion_tokens": len(valid_ids),
                    "clipped": len(valid_ids) >= max_completion_length,
                    "format_compliant": final_number_format_compliant(output_text),
                    "answer_correct": (
                        expected_number is not None
                        and generated_number is not None
                        and expected_number == generated_number
                    ),
                    "completion_hash": stable_text_hash((output_text,)),
                    "output_text": output_text,
                })
            return generated

    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    trainer = EvidenceDistillationTrainer(
        model=student_path,
        teacher_model=teacher_path,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    initial_adapter_state = None
    if adapter_config is not None:
        from peft import get_peft_model_state_dict

        initial_adapter_state = {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(trainer.model).items()
        }
    trainer.add_callback(dose_callback)
    started = time.perf_counter()
    result = trainer.train()
    torch_module.cuda.synchronize()
    training_time_s = time.perf_counter() - started
    peak_allocated = int(torch_module.cuda.max_memory_allocated())
    peak_reserved = int(torch_module.cuda.max_memory_reserved())
    dose_selection = select_development_dose(
        development_baseline, dose_callback.results
    )
    if (
        dose_callback.selected_state is None
        or dose_callback.selected_step != dose_selection["selected_step"]
    ):
        raise RuntimeError("G2 未保留到选中的剂量快照")
    adapter_parameter_counts = (
        {
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in trainer.model.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(
                parameter.numel() for parameter in trainer.model.parameters()
            ),
        }
        if adapter_config is not None
        else None
    )
    if peft_config is None:
        trainer.model.load_state_dict(dose_callback.selected_state)
        saved_model = trainer.model
    else:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(trainer.model, dose_callback.selected_state)
        saved_model = trainer.model.merge_and_unload()
    saved_model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metric_keys = {
        "step", "loss", "grad_norm", "learning_rate", "entropy", "num_tokens",
        "completions/mean_length", "completions/min_length", "completions/max_length",
        "completions/clipped_ratio", "completions/mean_terminated_length",
    }
    metric_history = [
        {key: value for key, value in row.items() if key in metric_keys}
        for row in trainer.state.log_history
        if "loss" in row
    ]
    history = [
        {key: value for key, value in row.items() if key in {"step", "loss", "grad_norm", "learning_rate"}}
        for row in metric_history
    ]
    selected_history = [
        row for row in history
        if int(float(row.get("step", -1))) == dose_selection["selected_step"]
    ]
    trajectory_hash = stable_text_hash((
        json.dumps(
            [
                {
                    "step": row["step"],
                    "record_id": row["record_id"],
                    "completion_hash": row["completion_hash"],
                }
                for row in trajectory_rows
            ],
            sort_keys=True,
        ),
    ))
    trajectory_samples = trajectory_rows[:4] + trajectory_rows[-4:]
    trajectory_summary = {
        "record_count": len(trajectory_rows),
        "unique_record_count": len({row["record_id"] for row in trajectory_rows}),
        "answer_correct_count": sum(int(row["answer_correct"]) for row in trajectory_rows),
        "answer_accuracy": (
            sum(int(row["answer_correct"]) for row in trajectory_rows) / len(trajectory_rows)
            if trajectory_rows else None
        ),
        "format_compliant_count": sum(int(row["format_compliant"]) for row in trajectory_rows),
        "format_compliance_ratio": (
            sum(int(row["format_compliant"]) for row in trajectory_rows) / len(trajectory_rows)
            if trajectory_rows else None
        ),
        "clipped_count": sum(int(row["clipped"]) for row in trajectory_rows),
        "clipped_ratio": (
            sum(int(row["clipped"]) for row in trajectory_rows) / len(trajectory_rows)
            if trajectory_rows else None
        ),
        "mean_completion_tokens": (
            round(statistics.mean(row["completion_tokens"] for row in trajectory_rows), 3)
            if trajectory_rows else None
        ),
        "trajectory_hash": trajectory_hash,
        "bounded_samples": trajectory_samples,
    }
    adapter_update_diagnostics = None
    if adapter_config is not None and initial_adapter_state is not None:
        selected_adapter_state = dose_callback.selected_state
        initial_norm_sq = sum(
            float(value.float().pow(2).sum()) for value in initial_adapter_state.values()
        )
        selected_norm_sq = sum(
            float(selected_adapter_state[name].float().pow(2).sum())
            for name in initial_adapter_state
        )
        module_deltas = {
            name: float(
                (selected_adapter_state[name].float() - initial_value.float())
                .pow(2).sum().sqrt()
            )
            for name, initial_value in initial_adapter_state.items()
        }
        delta_norm = math.sqrt(sum(value * value for value in module_deltas.values()))
        adapter_update_diagnostics = {
            "initial_l2_norm": round(math.sqrt(initial_norm_sq), 6),
            "selected_l2_norm": round(math.sqrt(selected_norm_sq), 6),
            "delta_l2_norm": round(delta_norm, 6),
            "relative_delta_l2": (
                round(delta_norm / math.sqrt(initial_norm_sq), 6)
                if initial_norm_sq > 0 else None
            ),
            "largest_module_deltas": [
                {"name": name, "delta_l2_norm": round(value, 6)}
                for name, value in sorted(
                    module_deltas.items(), key=lambda item: item[1], reverse=True
                )[:10]
            ],
        }
    training = {
        "method": (
            "trl_lora_distillation_trainer"
            if adapter_config is not None
            else "trl_distillation_trainer"
        ),
        "trajectory_mode": "on_policy",
        "optimizer": optimizer_name,
        "trainer_optimizer": trainer_optimizer,
        "train_steps": max_steps,
        "dose_steps": normalized_doses,
        "selected_step": dose_selection["selected_step"],
        "dose_selection": dose_selection,
        "development_results": dose_callback.results,
        "temperature": temperature,
        "beta": beta,
        "max_grad_norm": max_grad_norm,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "loss_history": history,
        "metric_history": metric_history,
        "trajectory_summary": trajectory_summary,
        "selected_step_loss": (
            float(selected_history[-1]["loss"]) if selected_history else None
        ),
        "mean_training_loss": float(result.training_loss),
        "training_time_s": round(training_time_s, 3),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "trl_api": inspect_distillation_api(),
        "adapter": (
            {
                "enabled": True,
                "config": dict(adapter_config),
                "saved_artifact": "merged_full_model",
                **adapter_parameter_counts,
                "update_diagnostics": adapter_update_diagnostics,
            }
            if adapter_config is not None
            else {"enabled": False}
        ),
    }
    del trainer, saved_model, tokenizer, dataset
    gc.collect()
    torch_module.cuda.empty_cache()
    evaluation = evaluate_model(
        torch_module,
        model_path=str(output),
        evaluation_records=evaluation_records,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        dtype_name=dtype_name,
        latency_warmup_prompts=latency_warmup_prompts,
        latency_repeats=latency_repeats,
        system_prompt=system_prompt,
    )
    return _candidate_evidence(
        name="g2_on_policy", artifact_path=output,
        evaluation=evaluation, training=training,
    )


def run_g0_baseline(
    torch_module: Any,
    *,
    teacher_path: str,
    student_path: str,
    development_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    max_prompt_length: int,
    max_completion_length: int,
    dtype_name: str,
    latency_warmup_prompts: int = 1,
    latency_repeats: int = 3,
    system_prompt: str = "Answer concisely and follow the requested answer format.",
) -> dict[str, Any]:
    """Evaluate teacher and student sequentially under one frozen protocol."""

    teacher = evaluate_model(
        torch_module,
        model_path=teacher_path,
        evaluation_records=evaluation_records,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        dtype_name=dtype_name,
        latency_warmup_prompts=latency_warmup_prompts,
        latency_repeats=latency_repeats,
        system_prompt=system_prompt,
    )
    student = evaluate_model(
        torch_module,
        model_path=student_path,
        evaluation_records=evaluation_records,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        dtype_name=dtype_name,
        latency_warmup_prompts=latency_warmup_prompts,
        latency_repeats=latency_repeats,
        system_prompt=system_prompt,
    )
    development_teacher = evaluate_model(
        torch_module,
        model_path=teacher_path,
        evaluation_records=development_records,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        dtype_name=dtype_name,
        latency_warmup_prompts=0,
        latency_repeats=1,
        reference_records=development_records,
        system_prompt=system_prompt,
    )
    development_student = evaluate_model(
        torch_module,
        model_path=student_path,
        evaluation_records=development_records,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        dtype_name=dtype_name,
        latency_warmup_prompts=0,
        latency_repeats=1,
        reference_records=development_records,
        system_prompt=system_prompt,
    )
    teacher_score = float(teacher["quality_score"])
    return {
        "status": "ok",
        "teacher": teacher,
        "student": student,
        "development": {
            "teacher": development_teacher,
            "student": development_student,
        },
        "comparison": {
            "quality_gap": float(student["quality_score"]) - teacher_score,
            "teacher_retention": (
                float(student["quality_score"]) / teacher_score
                if teacher_score > 0 else None
            ),
            "latency_ratio": (
                float(student["mean_latency_ms"]) / float(teacher["mean_latency_ms"])
                if float(teacher["mean_latency_ms"]) > 0 else None
            ),
        },
    }


def resolve_distillation_models(
    teacher_model: str,
    student_model: str,
    *,
    source: str,
    cache_dir: str | Path | None = "auto",
) -> dict[str, dict[str, Any]]:
    """Resolve teacher/student snapshots and return immutable asset facts."""

    teacher_path = resolve_model(teacher_model, source=source, cache_dir=cache_dir)
    student_path = resolve_model(student_model, source=source, cache_dir=cache_dir)
    return {
        "teacher": {"model_id": teacher_model, **model_asset_manifest(teacher_path)},
        "student": {"model_id": student_model, **model_asset_manifest(student_path)},
    }


def compare_tokenizer_contracts(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    *,
    probe_texts: Sequence[str] = (
        "#### 42",
        "-2.5 + 1,024",
        "Solve the problem concisely.",
        "请给出最终数值答案。",
    ),
) -> dict[str, Any]:
    """Compare token IDs before allowing full-logit distillation.

    Matching vocabulary sizes are insufficient: dense-logit KD requires every
    token column to have the same meaning in both models. Probe encodings and
    chat templates remain diagnostic and do not override that hard condition.
    """

    student_vocab = {
        str(token): int(token_id)
        for token, token_id in student_tokenizer.get_vocab().items()
    }
    teacher_vocab = {
        str(token): int(token_id)
        for token, token_id in teacher_tokenizer.get_vocab().items()
    }
    shared_tokens = set(student_vocab) & set(teacher_vocab)
    matching_token_ids = sum(
        int(student_vocab[token] == teacher_vocab[token]) for token in shared_tokens
    )
    exact_token_id_mapping = student_vocab == teacher_vocab

    def digest(value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def special_token_ids(tokenizer: Any) -> dict[str, int | None]:
        return {
            name: getattr(tokenizer, f"{name}_token_id", None)
            for name in ("bos", "eos", "pad", "unk")
        }

    probe_rows = []
    for text in probe_texts:
        student_ids = list(student_tokenizer.encode(text, add_special_tokens=False))
        teacher_ids = list(teacher_tokenizer.encode(text, add_special_tokens=False))
        probe_rows.append({
            "text": text,
            "student_ids": student_ids,
            "teacher_ids": teacher_ids,
            "identical": student_ids == teacher_ids,
        })

    student_special = special_token_ids(student_tokenizer)
    teacher_special = special_token_ids(teacher_tokenizer)
    student_template = str(getattr(student_tokenizer, "chat_template", "") or "")
    teacher_template = str(getattr(teacher_tokenizer, "chat_template", "") or "")
    full_logit_compatible = (
        exact_token_id_mapping
        and student_special == teacher_special
        and all(row["identical"] for row in probe_rows)
    )
    return {
        "schema_version": "tokenizer-compatibility/v1",
        "student_vocab_size": len(student_vocab),
        "teacher_vocab_size": len(teacher_vocab),
        "shared_token_count": len(shared_tokens),
        "matching_token_id_count": matching_token_ids,
        "matching_token_id_ratio": (
            matching_token_ids / len(shared_tokens) if shared_tokens else 0.0
        ),
        "student_vocab_sha256": digest(student_vocab),
        "teacher_vocab_sha256": digest(teacher_vocab),
        "exact_token_id_mapping": exact_token_id_mapping,
        "student_special_token_ids": student_special,
        "teacher_special_token_ids": teacher_special,
        "special_token_ids_match": student_special == teacher_special,
        "probe_encodings": probe_rows,
        "probe_encodings_match": all(row["identical"] for row in probe_rows),
        "student_chat_template_sha256": digest(student_template),
        "teacher_chat_template_sha256": digest(teacher_template),
        "chat_templates_match": student_template == teacher_template,
        "full_logit_compatible": full_logit_compatible,
        "offline_trace_compatible": True,
        "recommended_route": (
            "full_logit_gkd_candidate"
            if full_logit_compatible
            else "offline_trace_distillation"
        ),
    }


def inspect_tokenizer_compatibility(
    student_path: str | Path,
    teacher_path: str | Path,
) -> dict[str, Any]:
    """Load two local tokenizers and return their distillation contract."""

    from transformers import AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(
        str(student_path), local_files_only=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_path), local_files_only=True
    )
    return compare_tokenizer_contracts(student_tokenizer, teacher_tokenizer)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Return a Wilson score interval for one observed Bernoulli rate."""

    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson interval 要求 0 <= successes <= total")
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        rate * (1 - rate) / total + z**2 / (4 * total**2)
    ) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def summarize_offline_trace_admission(
    trace_evidence: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare sampled traces with a paired baseline and expose admission risks."""

    trace_rows = list(trace_evidence.get("rows", ()))
    baseline_by_id = {str(row["record_id"]): row for row in baseline_rows}
    trace_by_id = {str(row["record_id"]): row for row in trace_rows}
    if not trace_rows or len(trace_by_id) != len(trace_rows):
        raise ValueError("轨迹记录不能为空且 record_id 必须唯一")
    if set(trace_by_id) != set(baseline_by_id):
        raise ValueError("轨迹与基线必须包含相同的 record_id")

    categories: dict[str, dict[str, Any]] = {}
    improved = 0
    regressed = 0
    both_correct = 0
    both_incorrect = 0
    eligible = 0
    raw_format_compliant = 0
    source_splits: set[str] = set()
    closing_without_opening = 0
    for record_id, trace_row in trace_by_id.items():
        baseline_row = baseline_by_id[record_id]
        category = str(trace_row.get("category", "uncategorized"))
        category_row = categories.setdefault(category, {
            "record_count": 0,
            "trace_correct_count": 0,
            "baseline_correct_count": 0,
        })
        trace_correct = trace_row.get("correct") is True
        baseline_correct = baseline_row.get("passed") is True
        category_row["record_count"] += 1
        category_row["trace_correct_count"] += int(trace_correct)
        category_row["baseline_correct_count"] += int(baseline_correct)
        if trace_correct and not baseline_correct:
            improved += 1
        elif baseline_correct and not trace_correct:
            regressed += 1
        elif trace_correct:
            both_correct += 1
        else:
            both_incorrect += 1
        output = str(trace_row.get("output", ""))
        raw_format_compliant += int(final_number_format_compliant(output))
        eligible += int(
            trace_correct
            and trace_row.get("parseable") is True
            and trace_row.get("saturated") is not True
        )
        source_splits.add(str(trace_row.get("source_split", "unknown")))
        closing_without_opening += int("</think>" in output and "<think>" not in output)

    for category_row in categories.values():
        count = int(category_row["record_count"])
        trace_correct_count = int(category_row["trace_correct_count"])
        baseline_correct_count = int(category_row["baseline_correct_count"])
        trace_quality = trace_correct_count / count
        baseline_quality = baseline_correct_count / count
        category_row.update({
            "trace_quality_score": trace_quality,
            "trace_quality_interval_95": _wilson_interval(trace_correct_count, count),
            "baseline_quality_score": baseline_quality,
            "quality_delta": round(trace_quality - baseline_quality, 6),
        })

    count = len(trace_rows)
    trace_correct_count = sum(int(row.get("correct") is True) for row in trace_rows)
    baseline_correct_count = sum(
        int(row.get("passed") is True) for row in baseline_rows
    )
    discordant = improved + regressed
    paired_one_sided_p = (
        sum(math.comb(discordant, value) for value in range(improved, discordant + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    baseline_mean_tokens = statistics.mean(
        float(row.get("generated_tokens", row.get("output_tokens", 0)))
        for row in baseline_rows
    )
    baseline_mean_latency = statistics.mean(
        float(row.get("latency_ms", 0.0)) for row in baseline_rows
    )
    trace_mean_tokens = float(trace_evidence.get("mean_generated_tokens", 0.0))
    trace_mean_latency = float(trace_evidence.get("mean_latency_ms", 0.0))
    return {
        "record_count": count,
        "trace_quality_score": trace_correct_count / count,
        "trace_quality_interval_95": _wilson_interval(trace_correct_count, count),
        "baseline_quality_score": baseline_correct_count / count,
        "baseline_quality_interval_95": _wilson_interval(
            baseline_correct_count, count
        ),
        "quality_delta": round(
            (trace_correct_count - baseline_correct_count) / count, 6
        ),
        "paired_transition": {
            "improved_count": improved,
            "regressed_count": regressed,
            "both_correct_count": both_correct,
            "both_incorrect_count": both_incorrect,
            "net_gain_count": improved - regressed,
            "discordant_count": discordant,
            "one_sided_sign_test_p": round(paired_one_sided_p, 6),
        },
        "category_scores": dict(sorted(categories.items())),
        "minimum_category_quality": min(
            float(row["trace_quality_score"]) for row in categories.values()
        ),
        "regressed_categories": sorted(
            name for name, row in categories.items()
            if float(row["quality_delta"]) < 0
        ),
        "training_eligible_count": eligible,
        "training_eligible_ratio": eligible / count,
        "raw_student_format_ratio": raw_format_compliant / count,
        "source_splits": sorted(source_splits),
        "training_reuse_allowed": source_splits == {"train"},
        "think_protocol": {
            "closing_without_opening_count": closing_without_opening,
            "closing_without_opening_ratio": closing_without_opening / count,
        },
        "mean_output_token_ratio": (
            trace_mean_tokens / baseline_mean_tokens if baseline_mean_tokens else None
        ),
        "mean_latency_ratio": (
            trace_mean_latency / baseline_mean_latency
            if baseline_mean_latency else None
        ),
    }


def select_distillation_route(
    evidence: Mapping[str, Any],
    gates: Mapping[str, float],
) -> dict[str, Any]:
    """Select offline, online, hybrid, or blocked from measured evidence."""

    offline = evidence.get("offline", {})
    online = evidence.get("online", {})
    offline_checks = {
        "trace_available": offline.get("trace_available") is True,
        "sample_coverage": int(offline.get("minimum_category_count", 0))
        >= int(gates.get("min_records_per_category", 1)),
        "parse_ratio": float(offline.get("parse_ratio", 0.0))
        >= float(gates["min_trace_parse_ratio"]),
        "quality_floor": float(offline.get("quality_score", 0.0))
        >= float(gates.get("min_trace_quality_score", 0.0)),
        "quality_gain": float(offline.get("quality_delta", -1.0))
        >= float(gates.get("min_trace_quality_gain", 0.0)),
        "category_floor": float(offline.get("minimum_category_quality", 0.0))
        >= float(gates.get("min_trace_category_quality", 0.0)),
        "category_stability": int(offline.get("regressed_category_count", 0))
        <= int(gates.get("max_trace_regressed_categories", 0)),
        "training_yield": float(offline.get("training_eligible_ratio", 0.0))
        >= float(gates.get("min_trace_training_eligible_ratio", 0.0)),
        "saturation": float(offline.get("saturation_ratio", 1.0))
        <= float(gates["max_trace_saturation_ratio"]),
        "output_cost": float(offline.get("mean_output_token_ratio", math.inf))
        <= float(gates.get("max_trace_mean_output_token_ratio", math.inf)),
        "latency_cost": float(offline.get("mean_latency_ratio", math.inf))
        <= float(gates.get("max_trace_mean_latency_ratio", math.inf)),
    }
    online_checks = {
        "teacher_ready": online.get("teacher_ready") is True,
        "full_logit_compatible": online.get("full_logit_compatible") is True,
        "memory_admitted": online.get("memory_admitted") is True,
    }
    offline_ready = all(offline_checks.values())
    online_ready = all(online_checks.values())
    if offline_ready and online_ready:
        route = "hybrid"
    elif offline_ready:
        route = "offline"
    elif online_ready:
        route = "online"
    else:
        route = "blocked"
    return {
        "route": route,
        "offline_ready": offline_ready,
        "offline_training_data_ready": (
            offline_ready and offline.get("training_reuse_allowed") is True
        ),
        "online_ready": online_ready,
        "offline_checks": offline_checks,
        "online_checks": online_checks,
        "failed_checks": [
            f"offline.{name}"
            for name, passed in offline_checks.items()
            if not passed
        ] + [
            f"online.{name}"
            for name, passed in online_checks.items()
            if not passed
        ],
    }


def evaluate_offline_trace_candidate(
    torch_module: Any,
    *,
    model_path: str | Path,
    evaluation_records: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    samples_per_category: int,
    max_prompt_length: int,
    max_completion_length: int,
    dtype_name: str,
    temperature: float,
    top_p: float,
    seed: int,
    assistant_prefix: str = "",
) -> dict[str, Any]:
    """Generate a small native-template trace set for offline KD admission."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if samples_per_category < 1:
        raise ValueError("samples_per_category 必须为正整数")
    if not 0.0 < temperature <= 2.0 or not 0.0 < top_p <= 1.0:
        raise ValueError("temperature 与 top_p 超出有效范围")
    selected: list[Mapping[str, Any]] = []
    for category in categories:
        candidates = sorted(
            (
                record
                for record in evaluation_records
                if str(record.get("category")) == str(category)
            ),
            key=lambda record: str(record["record_id"]),
        )
        if len(candidates) < samples_per_category:
            raise ValueError(f"类别 {category!r} 的轨迹样本不足")
        selected.extend(candidates[:samples_per_category])

    dtype = getattr(torch_module, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    rows: list[dict[str, Any]] = []
    elapsed_samples_ms: list[float] = []
    generated_counts: list[int] = []
    for index, record in enumerate(selected):
        question = str(record["prompt"]).split(
            "\n\nShow your reasoning, then write the final numeric answer after ####.",
            1,
        )[0]
        prompt = (
            f"{question}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered = f"{rendered}{assistant_prefix}"
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        ).to("cuda")
        record_seed = seed + index
        torch_module.manual_seed(record_seed)
        torch_module.cuda.manual_seed_all(record_seed)
        torch_module.cuda.synchronize()
        started = time.perf_counter()
        with torch_module.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_completion_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch_module.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        completion_ids = generated[0, encoded["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()
        output = f"{assistant_prefix}{generated_text}".strip()
        final_number = extract_final_number(output)
        expected = [str(item).replace(",", "") for item in record.get("expected_exact", ())]
        correct = final_number is not None and any(
            Decimal(final_number) == Decimal(item) for item in expected
        )
        token_count = int(completion_ids.numel())
        generated_counts.append(token_count)
        elapsed_samples_ms.append(elapsed_ms)
        rows.append({
            "record_id": str(record["record_id"]),
            "category": str(record.get("category", "uncategorized")),
            "source_split": str(record.get("split", "unknown")),
            "seed": record_seed,
            "expected_exact": expected,
            "final_number": final_number,
            "parseable": final_number is not None,
            "correct": correct,
            "generated_tokens": token_count,
            "saturated": token_count >= max_completion_length,
            "latency_ms": round(elapsed_ms, 3),
            "output": output,
        })

    del model, tokenizer
    gc.collect()
    torch_module.cuda.empty_cache()
    record_count = len(rows)
    parsed_count = sum(int(row["parseable"]) for row in rows)
    correct_count = sum(int(row["correct"]) for row in rows)
    saturated_count = sum(int(row["saturated"]) for row in rows)
    return {
        "status": "ok",
        "protocol": {
            "template": "candidate_native_chat_template",
            "prompt_style": "deepseek_math_boxed_answer",
            "assistant_prefix": assistant_prefix,
            "sampling": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "seed": seed,
            "samples_per_category": samples_per_category,
        },
        "record_count": record_count,
        "record_ids_hash": stable_text_hash(row["record_id"] for row in rows),
        "parseable_count": parsed_count,
        "parse_ratio": parsed_count / record_count,
        "correct_count": correct_count,
        "quality_score": correct_count / record_count,
        "saturated_count": saturated_count,
        "saturation_ratio": saturated_count / record_count,
        "mean_generated_tokens": round(statistics.mean(generated_counts), 3),
        "mean_latency_ms": round(statistics.mean(elapsed_samples_ms), 3),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
        "rows": rows,
    }


def validate_distillation_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate that G0/G1/G2 share one comparable distillation contract."""

    issues: list[str] = []
    teacher = spec.get("teacher", {})
    student = spec.get("student", {})
    data = spec.get("data", {})
    groups = spec.get("groups", {})
    required_groups = ("g0", "g1", "g2")

    if spec.get("schema_version") not in {
        "distilled-student-project/v1",
        "distilled-student-project/v2",
        "distilled-student-project/v3",
    }:
        issues.append("schema")
    if not all(
        (
            teacher.get("model_id"),
            teacher.get("revision"),
            student.get("model_id"),
            student.get("revision"),
        )
    ):
        issues.append("model_identity")
    if teacher.get("frozen") is not True:
        issues.append("teacher_not_frozen")
    if (
        not data.get("train_hash")
        or not data.get("evaluation_hash")
        or data.get("train_hash") == data.get("evaluation_hash")
    ):
        issues.append("evaluation_overlap")
    if any(name not in groups for name in required_groups):
        issues.append("missing_group")
    else:
        if any(
            groups[name].get("student_revision") != student.get("revision")
            for name in required_groups
        ):
            issues.append("student_start_mismatch")
        if groups["g1"].get("train_steps") != groups["g2"].get("train_steps"):
            issues.append("training_budget_mismatch")
        if (
            groups["g1"].get("trajectory_mode") != "fixed"
            or groups["g2"].get("trajectory_mode") != "on_policy"
        ):
            issues.append("trajectory_mode")
    return {"ready": not issues, "issues": issues}


def summarize_distilled_candidate(
    teacher: dict[str, Any],
    student_baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | bool | str]:
    """Summarize candidate quality and deployment ratios under one protocol."""

    numeric_fields = ("quality_score", "artifact_size_mb", "latency_ms")
    if any(
        not isinstance(record.get(field), (int, float))
        for record in (teacher, student_baseline, candidate)
        for field in numeric_fields
    ):
        raise ValueError("教师、学生基线和候选缺少可比较数值")
    if teacher["quality_score"] <= 0 or any(
        record[field] <= 0
        for record in (teacher, student_baseline, candidate)
        for field in ("artifact_size_mb", "latency_ms")
    ):
        raise ValueError("教师质量以及三组大小和延迟必须为正")
    teacher_categories = teacher.get("category_scores", {})
    baseline_categories = student_baseline.get("category_scores", {})
    candidate_categories = candidate.get("category_scores", {})
    common_categories = sorted(
        set(teacher_categories) & set(baseline_categories) & set(candidate_categories)
    )
    regressed_categories = [
        category
        for category in common_categories
        if float(candidate_categories[category]["quality_score"])
        < float(baseline_categories[category]["quality_score"])
    ]
    latency_protocol = candidate.get("latency_protocol", {})
    latency_measured = (
        int(latency_protocol.get("repeats", 0)) >= 2
        and int(latency_protocol.get("sample_count", 0)) >= 2
    )
    return {
        "candidate": str(candidate.get("name", "candidate")),
        "quality_gain": candidate["quality_score"]
        - student_baseline["quality_score"],
        "teacher_retention": candidate["quality_score"] / teacher["quality_score"],
        "teacher_size_ratio": candidate["artifact_size_mb"]
        / teacher["artifact_size_mb"],
        "teacher_latency_ratio": candidate["latency_ms"] / teacher["latency_ms"],
        "baseline_size_ratio": candidate["artifact_size_mb"]
        / student_baseline["artifact_size_mb"],
        "baseline_latency_ratio": candidate["latency_ms"]
        / student_baseline["latency_ms"],
        "regressed_categories": regressed_categories,
        "regressed_category_count": len(regressed_categories),
        "artifact_verified": candidate.get("artifact_verified") is True,
        "load_ready": candidate.get("load_status") == "loaded",
        "latency_measured": latency_measured,
    }


def decide_distilled_student(
    contract: dict[str, Any],
    summary: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, str]:
    """Turn contract, quality and deployment evidence into a delivery decision."""

    if not contract.get("ready"):
        return {"decision": "reject", "reason": "; ".join(contract.get("issues", []))}
    if not summary.get("artifact_verified") or not summary.get("load_ready"):
        return {"decision": "reject", "reason": "学生产物未通过完整性或加载检查"}
    if (
        summary["quality_gain"] < gates["min_quality_gain"]
        or summary["teacher_retention"] < gates["min_teacher_retention"]
    ):
        return {"decision": "tune", "reason": "学生增益或教师能力保持率未达到门槛"}
    if not summary.get("latency_measured"):
        return {"decision": "measure", "reason": "缺少同协议教师—学生推理测量"}
    if (
        summary["teacher_size_ratio"] > gates["max_teacher_size_ratio"]
        or summary["teacher_latency_ratio"] > gates["max_teacher_latency_ratio"]
        or summary["baseline_latency_ratio"] > gates["max_baseline_latency_ratio"]
        or summary["regressed_category_count"] > gates["max_regressed_categories"]
    ):
        return {"decision": "tune", "reason": "学生质量通过，但部署收益未达到门槛"}
    return {"decision": "accept", "reason": "质量、产物、大小和延迟门槛全部通过"}


__all__ = [
    "build_memory_admission",
    "build_distillation_diagnostics",
    "assess_g0_readiness",
    "decide_distilled_student",
    "inspect_distillation_api",
    "load_distillation_workload",
    "model_asset_manifest",
    "evaluate_completion_nll",
    "evaluate_model",
    "evaluate_prompt_candidates",
    "final_number_format_compliant",
    "resolve_distillation_models",
    "run_fixed_trajectory_kd",
    "run_g0_baseline",
    "run_on_policy_distillation",
    "run_profile_preflight",
    "select_validation_dose",
    "save_json_record",
    "score_expected_output",
    "summarize_distilled_candidate",
    "validate_distillation_contract",
]
