"""Project-owned single-GPU training benchmark for Part 05 · 03.

The workload is pinned to a local Qwen2.5-0.5B checkpoint and fixed synthetic
token batches. This module does not download assets. CPU tests replace model,
CUDA and profiler helpers with local fixtures to check capture and review flow.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from tools.performance_delivery_runtime import MEASUREMENT_SCHEMA, save_json_record_once, sha256_file, _gate_errors


SEMANTIC_ID = "PERF-INTEGRATED-DELIVERY"
WORKLOAD_ID = "qwen025b_b1_s768_fp32_bf16_adamw_local_tokens_v1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SNAPSHOT = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_RELATIVE_PATH = Path("model_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots") / MODEL_SNAPSHOT
BATCH_SIZE = 1
SEQUENCE_LENGTH = 768
PARAMETER_DTYPE = "float32"
COMPUTE_DTYPE = "bfloat16"
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
SEED = 42
WARMUP_STEPS = 1
MEASURED_STEPS = 3
REPEATS = 3
MIN_FREE_MEMORY_MB = 10_000
MAX_LOSS_REGRESSION_RATIO = 0.02

CANDIDATE_ORDER = ["fused_adamw", "activation_checkpoint"]
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"fused_adamw": False, "activation_checkpoint": False, "components": []},
    "fused_adamw": {"fused_adamw": True, "activation_checkpoint": False, "components": ["fused_adamw"]},
    "activation_checkpoint": {"fused_adamw": False, "activation_checkpoint": True, "components": ["activation_checkpoint"]},
    "fused_checkpoint": {"fused_adamw": True, "activation_checkpoint": True, "components": CANDIDATE_ORDER},
}


def _result_directory(project_root: str | Path, run_label: str) -> Path:
    if not isinstance(run_label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", run_label) or run_label == "replace_with_unique_label":
        raise ValueError("请填写 3–64 位唯一 RUN_LABEL。")
    return Path(project_root).resolve() / "benchmarks/results/part05_performance_delivery" / run_label


def model_path_for(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / MODEL_RELATIVE_PATH


def _model_files_ready(model_path: Path) -> tuple[bool, list[str]]:
    required = ("config.json", "model.safetensors", "tokenizer_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    return not missing, missing


def check_model_assets(project_root: str | Path) -> dict[str, Any]:
    """Perform an offline, filesystem-only check suitable for CPU notebooks."""

    model_path = model_path_for(project_root)
    ready, missing = _model_files_ready(model_path)
    return {
        "ready": ready,
        "model_id": MODEL_ID,
        "revision": MODEL_SNAPSHOT,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "missing_files": missing,
        "weight_bytes": (model_path / "model.safetensors").stat().st_size
        if (model_path / "model.safetensors").is_file() else None,
    }


def validate_project_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Validate the small fixed policy file; workload details stay in code."""

    errors: list[str] = []
    if not isinstance(protocol, Mapping):
        return ["protocol 必须是映射"]
    if not isinstance(protocol.get("outputs"), Mapping):
        return ["outputs 必须是映射"]
    if protocol.get("schema_version") != "performance-optimization-delivery-config/v2":
        errors.append("schema_version 不匹配")
    if protocol.get("semantic_id") != SEMANTIC_ID:
        errors.append("semantic_id 不匹配")
    if protocol.get("environment_profile") not in ("PERF-CUDA-MECHANISM", "PERF-DELIVERY"):
        errors.append("environment_profile 必须是 PERF-CUDA-MECHANISM 或 PERF-DELIVERY")
    gates = protocol.get("gates")
    if not isinstance(gates, Mapping):
        errors.append("gates 必须是映射")
    else:
        errors.extend(_gate_errors(gates))
        required = ("primary_metric", "minimum_primary_improvement_ratio", "maximum_guardrail_regression_ratio", "minimum_samples")
        errors.extend(f"gates 缺少 {key}" for key in required if key not in gates)
        if gates.get("primary_metric") != "step_time_ms" or gates.get("primary_direction") != "lower":
            errors.append("本项目主指标固定为 step_time_ms 越低越好")
        improvement = gates.get("minimum_primary_improvement_ratio")
        if not isinstance(improvement, (int, float)) or not 0 <= improvement < 1:
            errors.append("minimum_primary_improvement_ratio 必须在 [0, 1) 范围内")
        if gates.get("minimum_samples") != REPEATS:
            errors.append(f"minimum_samples 必须与固定 repeats={REPEATS} 一致")
        guardrails = gates.get("maximum_guardrail_regression_ratio")
        if not isinstance(guardrails, Mapping) or set(guardrails) != {"peak_memory_mb", "loss"}:
            errors.append("守护指标固定为 peak_memory_mb 和 loss")
        elif any(not isinstance(value, (int, float)) or value < 0 for value in guardrails.values()):
            errors.append("守护指标回归阈值必须是非负数")
        for key in ("require_composition", "require_reprofile", "require_rollback_ready"):
            if gates.get(key) is not True:
                errors.append(f"gates.{key} 必须保持启用")
    output_dir = protocol.get("outputs", {}).get("directory", "")
    if not isinstance(output_dir, str) or not output_dir.strip():
        errors.append("outputs.directory 不能为空")
    elif Path(output_dir).is_absolute() or ".." in Path(output_dir).parts:
        errors.append("outputs.directory 必须位于仓库相对路径内")
    elif output_dir != "benchmarks/results/part05_performance_delivery":
        errors.append("outputs.directory 必须与本项目固定测量目录一致")
    context_sources = protocol.get("context_sources", [])
    if not isinstance(context_sources, list):
        errors.append("context_sources 必须是列表")
    return errors


def build_workload_identity(torch_module: Any, project_root: str | Path) -> dict[str, Any]:
    """Return the identity shared by every paired measurement record."""

    if not torch_module.cuda.is_available():
        raise RuntimeError("GPU 不可用；本项目的测量阶段要求 CUDA。")
    device = torch_module.cuda.get_device_properties(0)
    model_path = model_path_for(project_root)
    return {
        "workload_id": WORKLOAD_ID,
        "task": "causal_language_model_training_step",
        "model": MODEL_ID,
        "model_revision": MODEL_SNAPSHOT,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "model_file_sha256": {name: sha256_file(model_path / name) for name in ("config.json", "model.safetensors", "tokenizer_config.json")},
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "parameter_dtype": PARAMETER_DTYPE,
        "compute_dtype": COMPUTE_DTYPE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "input_source": "seeded_synthetic_token_ids",
        "warmup_steps": WARMUP_STEPS,
        "measured_steps_per_repeat": MEASURED_STEPS,
        "repeats": REPEATS,
        "hardware": torch_module.cuda.get_device_name(0),
        "device_capability": list(torch_module.cuda.get_device_capability(0)),
        "device_memory_mb": round(device.total_memory / (1024**2), 1),
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
    }


def _cuda_kernel_smoke_test(torch_module: Any) -> dict[str, str]:
    """Verify the CUDA/BF16/fused-AdamW kernels needed by the workload."""

    parameter = optimizer = loss = None
    try:
        parameter = torch_module.nn.Parameter(
            torch_module.ones((32, 32), device="cuda:0", dtype=torch_module.float32)
        )
        optimizer = torch_module.optim.AdamW([parameter], lr=LEARNING_RATE, fused=True)
        with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
            loss = (parameter @ parameter).float().mean()
        loss.backward()
        optimizer.step()
        torch_module.cuda.synchronize(0)
        if not torch_module.isfinite(parameter).all().item():
            raise FloatingPointError("CUDA smoke test produced non-finite parameters")
    except Exception as error:
        raise RuntimeError(
            "CUDA kernel smoke test failed; confirm the active PERF-CUDA-MECHANISM "
            "environment has kernels for this GPU (including fused AdamW)."
        ) from error
    finally:
        del parameter, optimizer, loss
        gc.collect()
        torch_module.cuda.empty_cache()
    return {"cuda_tensor_ops": "passed", "bf16_autocast": "passed", "fused_adamw": "passed"}


def preflight(torch_module: Any, project_root: str | Path) -> dict[str, Any]:
    """Check local model assets and safe CUDA headroom without loading a model."""

    model_path = model_path_for(project_root)
    assets_ready, missing_assets = _model_files_ready(model_path)
    if not assets_ready:
        raise FileNotFoundError(f"本地模型资产不完整：{missing_assets}；路径：{model_path}")
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    weight_bytes = (model_path / "model.safetensors").stat().st_size
    if model_config.get("vocab_size", 0) <= 0 or model_config.get("max_position_embeddings", 0) < SEQUENCE_LENGTH:
        raise ValueError("本地模型配置与固定 token 序列长度不匹配。")
    if weight_bytes < 500_000_000:
        raise ValueError(f"本地权重文件大小异常：{weight_bytes} bytes。")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA 不可用；未加载模型、未启动测量。")
    if torch_module.cuda.device_count() != 1:
        raise RuntimeError(f"本实验固定单卡，检测到 {torch_module.cuda.device_count()} 张 CUDA 卡。")
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError("GPU 不支持 BF16；实验配置固定为 FP32 参数 + BF16 计算。")
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("当前环境缺少 Transformers；请使用 PERF-CUDA-MECHANISM 环境。")
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
    free_mb = free_bytes / (1024**2)
    if free_mb < MIN_FREE_MEMORY_MB:
        raise RuntimeError(
            f"空闲显存 {free_mb:.0f} MiB，低于安全门槛 {MIN_FREE_MEMORY_MB} MiB；"
            "关闭其他 GPU 程序后重新运行 prepare。"
        )
    kernel_smoke = _cuda_kernel_smoke_test(torch_module)
    return {
        "status": "ready",
        "model_path": str(model_path),
        "model_assets_complete": True,
        "model_weight_bytes": weight_bytes,
        "model_vocab_size": model_config["vocab_size"],
        "free_memory_mb": round(free_mb, 1),
        "total_memory_mb": round(total_bytes / (1024**2), 1),
        "minimum_free_memory_mb": MIN_FREE_MEMORY_MB,
        "cuda_kernel_smoke": kernel_smoke,
        "workload_identity": build_workload_identity(torch_module, project_root),
    }


def _seeded_tokens(torch_module: Any, vocab_size: int, seed: int, device: Any) -> Any:
    generator = torch_module.Generator(device="cpu").manual_seed(seed)
    token_ids = torch_module.randint(
        low=0,
        high=vocab_size,
        size=(BATCH_SIZE, SEQUENCE_LENGTH),
        generator=generator,
        dtype=torch_module.long,
    )
    return token_ids.to(device)


def _release_cuda(torch_module: Any) -> None:
    """Collect after the caller has released model, optimizer and token tensors."""
    gc.collect()
    torch_module.cuda.empty_cache()


def _load_train_components(torch_module: Any, model_class: Any, model_path: Path, variant: str) -> tuple[Any, Any]:
    spec = VARIANTS[variant]
    model = model_class.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch_module.float32,
        attn_implementation="sdpa",
        use_safetensors=True,
    )
    model.config.use_cache = False
    model.train()
    if spec["activation_checkpoint"]:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if bool(getattr(model, "is_gradient_checkpointing", False)) != spec["activation_checkpoint"]:
        raise RuntimeError("激活检查点开关与候选配置不一致。")
    model.to(device="cuda")
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=spec["fused_adamw"],
    )
    if bool(optimizer.defaults.get("fused", False)) != spec["fused_adamw"]:
        raise RuntimeError("AdamW fused 执行开关与候选配置不一致。")
    return model, optimizer


def _run_one_repeat(
    torch_module: Any, model_class: Any, model_path: Path, variant: str, seed: int
) -> dict[str, Any]:
    torch_module.manual_seed(seed)
    model, optimizer = _load_train_components(torch_module, model_class, model_path, variant)
    token_ids = _seeded_tokens(torch_module, model.config.vocab_size, seed, torch_module.device("cuda"))

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
            loss = model(input_ids=token_ids, labels=token_ids, use_cache=False).loss
        if not torch_module.isfinite(loss):
            raise FloatingPointError("训练损失出现 NaN 或 Inf。")
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    try:
        for _ in range(WARMUP_STEPS):
            train_step()
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()
        step_times: list[float] = []
        losses: list[float] = []
        for _ in range(MEASURED_STEPS):
            started = time.perf_counter()
            losses.append(train_step())
            torch_module.cuda.synchronize()
            step_times.append((time.perf_counter() - started) * 1000)
        peak_allocated_mb = torch_module.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch_module.cuda.max_memory_reserved() / (1024**2)
        if not all(math.isfinite(value) for value in step_times + losses):
            raise FloatingPointError("测量结果出现非有限数值。")
        return {
            "step_time_ms": statistics.median(step_times),
            "peak_memory_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "loss": losses[-1],
            "first_loss": losses[0],
            "tokens_per_s": BATCH_SIZE * SEQUENCE_LENGTH * 1000 / statistics.mean(step_times),
            "seed": seed,
        }
    finally:
        del token_ids, optimizer, model
        _release_cuda(torch_module)


def _record_for_variant(
    variant: str,
    samples: list[dict[str, Any]],
    identity: Mapping[str, Any],
    model_path: Path,
    *,
    baseline_samples: list[dict[str, Any]] | None = None,
    run_label: str,
) -> dict[str, Any]:
    spec = VARIANTS[variant]
    complete = len(samples) == REPEATS
    relative_loss_errors: list[float] = []
    if baseline_samples is not None and complete and len(baseline_samples) == REPEATS:
        relative_loss_errors = [
            abs(sample["loss"] - base["loss"]) / max(abs(base["loss"]), 1e-12)
            for sample, base in zip(samples, baseline_samples, strict=True)
        ]
    quality_passed = complete and all(
        math.isfinite(sample["loss"]) for sample in samples
    ) and (
        baseline_samples is None
        or (bool(relative_loss_errors) and max(relative_loss_errors) <= MAX_LOSS_REGRESSION_RATIO)
    )
    return {
        "schema_version": MEASUREMENT_SCHEMA,
        "project": "05-03",
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "variant": variant,
        "components": spec["components"],
        "workload_identity": dict(identity),
        "execution_path": {
            "verified": complete,
            "verification_scope": "runtime_switches",
            "optimizer_fused": spec["fused_adamw"],
            "gradient_checkpointing": spec["activation_checkpoint"],
        },
        "correctness": {
            "passed": quality_passed,
            "max_loss_relative_difference_vs_baseline": max(relative_loss_errors) if relative_loss_errors else 0.0,
            "maximum_allowed_loss_relative_difference": MAX_LOSS_REGRESSION_RATIO,
            "evidence_path": "samples[*].metrics.loss",
        },
        "source_evidence": {
            "model_path": MODEL_RELATIVE_PATH.as_posix(),
            "model_revision": MODEL_SNAPSHOT,
            "input_kind": "seeded_synthetic_token_ids",
            "seed_start": SEED,
        },
        "samples": [
            {
                "pair_id": f"pair-{sample['seed'] - SEED + 1:02d}",
                "seed": sample["seed"],
                "metrics": {
                    "step_time_ms": sample["step_time_ms"],
                    "peak_memory_mb": sample["peak_memory_mb"],
                    "peak_reserved_mb": sample["peak_reserved_mb"],
                    "loss": sample["loss"],
                    "tokens_per_s": sample["tokens_per_s"],
                },
            }
            for sample in samples
        ],
    }


def _profile_variant(
    torch_module: Any,
    model_class: Any,
    model_path: Path,
    variant: str,
    seed: int,
    trace_path: Path,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    torch_module.manual_seed(seed)
    model, optimizer = _load_train_components(torch_module, model_class, model_path, variant)
    token_ids = _seeded_tokens(torch_module, model.config.vocab_size, seed, torch_module.device("cuda"))

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
            loss = model(input_ids=token_ids, labels=token_ids, use_cache=False).loss
        loss.backward()
        optimizer.step()

    try:
        train_step()
        torch_module.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            train_step()
        torch_module.cuda.synchronize()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
        events = profiler.key_averages()
        def device_time(event: Any) -> float:
            return float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)))
        top_ops = sorted(events, key=device_time, reverse=True)[:8]
        return {
            "variant": variant,
            "trace_path": str(trace_path),
            "top_device_operators": [
                {"name": event.key, "self_device_time_us": device_time(event)}
                for event in top_ops if device_time(event) > 0
            ],
        }
    finally:
        del token_ids, optimizer, model
        _release_cuda(torch_module)


def run_benchmark(
    torch_module: Any,
    auto_model_class: Any,
    project_root: str | Path,
    run_label: str,
) -> dict[str, Any]:
    """Run fixed paired variants and save immutable measurement evidence."""

    root = Path(project_root).resolve()
    model_path = model_path_for(root)
    result_dir = _result_directory(root, run_label)
    if result_dir.exists():
        raise FileExistsError(f"同标签结果目录已存在，不覆盖：{result_dir}")
    check = preflight(torch_module, project_root)
    result_dir.mkdir(parents=True, exist_ok=False)
    identity = check["workload_identity"]
    samples_by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    failures: list[dict[str, str]] = []
    names = list(VARIANTS)

    for repeat_index in range(REPEATS):
        seed = SEED + repeat_index
        order = names[repeat_index:] + names[:repeat_index]
        for variant in order:
            try:
                samples_by_variant[variant].append(
                    _run_one_repeat(torch_module, auto_model_class, model_path, variant, seed)
                )
            except Exception as error:
                if isinstance(error, torch_module.cuda.OutOfMemoryError):
                    torch_module.cuda.empty_cache()
                failures.append({"variant": variant, "repeat": str(repeat_index + 1), "error": f"{type(error).__name__}: {error}"[:500]})
                if variant == "baseline":
                    break
        if failures and failures[-1]["variant"] == "baseline":
            break

    baseline_samples = samples_by_variant["baseline"]
    records: dict[str, dict[str, Any]] = {}
    for variant, samples in samples_by_variant.items():
        records[variant] = _record_for_variant(
            variant,
            samples,
            identity,
            model_path,
            baseline_samples=baseline_samples if variant != "baseline" else None,
            run_label=run_label,
        )
        path = result_dir / f"{variant}.json"
        save_json_record_once(path, records[variant])

    composition_record = dict(records["fused_checkpoint"])
    composition_record["variant"] = "composition"
    composition_record["components"] = CANDIDATE_ORDER
    save_json_record_once(result_dir / "composition.json", composition_record)

    profile_records: list[dict[str, Any]] = []
    if len(baseline_samples) == REPEATS:
        profile_variants = ["baseline"] + [
            name for name in VARIANTS if name != "baseline" and len(samples_by_variant[name]) == REPEATS
        ]
        for variant in profile_variants:
            try:
                profile_record = _profile_variant(
                    torch_module, auto_model_class, model_path, variant, SEED,
                    result_dir / f"traces/{variant}.json",
                )
                profile_record["trace_path"] = str(Path(profile_record["trace_path"]).relative_to(root))
                profile_records.append(profile_record)
            except Exception as error:
                if isinstance(error, torch_module.cuda.OutOfMemoryError):
                    torch_module.cuda.empty_cache()
                failures.append({"variant": f"reprofile:{variant}", "repeat": "1", "error": f"{type(error).__name__}: {error}"[:500]})

    profile_summary_path = result_dir / "profile_summary.json"
    if len(profile_records) >= 2 and profile_records[0]["variant"] == "baseline":
        profile_summary = {
            "schema_version": "performance-delivery-reprofile/v1",
            "semantic_id": SEMANTIC_ID, "run_label": run_label,
            "workload_identity": identity,
            "profiles": profile_records,
        }
        save_json_record_once(profile_summary_path, profile_summary)
        reprofile_record = {
            "semantic_id": SEMANTIC_ID, "run_label": run_label,
            "workload_identity": identity,
            "bottleneck_before": profile_records[0]["top_device_operators"],
            "bottleneck_after": {
                ("composition" if item["variant"] == "fused_checkpoint" else item["variant"]): item["top_device_operators"] for item in profile_records[1:]
            },
            "evidence_path": str(profile_summary_path.relative_to(root)),
        }
        save_json_record_once(result_dir / "reprofile.json", reprofile_record)

    rollback_record = {
        "semantic_id": SEMANTIC_ID, "run_label": run_label,
        "rollback_ready": check["model_assets_complete"],
        "restore_action": "reload the pinned local base checkpoint and disable fused AdamW / gradient checkpointing",
        "model_revision": MODEL_SNAPSHOT,
        "model_file_sha256": identity["model_file_sha256"],
        "baseline_variant": "baseline",
        "candidate_variants": CANDIDATE_ORDER + ["fused_checkpoint"],
    }
    save_json_record_once(result_dir / "rollback.json", rollback_record)

    manifest = {
        "schema_version": "performance-delivery-capture/v1",
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "status": "complete" if not failures and all(len(rows) == REPEATS for rows in samples_by_variant.values()) else "incomplete",
        "preflight": check,
        "workload_identity": identity,
        "candidate_order": CANDIDATE_ORDER,
        "repeat_orders": [names[index:] + names[:index] for index in range(REPEATS)],
        "variant_status": {name: {"successful_repeats": len(rows), "required_repeats": REPEATS} for name, rows in samples_by_variant.items()},
        "failures": failures,
        "paths": {
            "baseline": str((result_dir / "baseline.json").relative_to(root)),
            "candidates": {
                name: str((result_dir / f"{name}.json").relative_to(root))
                for name in CANDIDATE_ORDER
            },
            "composition": str((result_dir / "composition.json").relative_to(root)),
            "reprofile": str((result_dir / "reprofile.json").relative_to(root)) if (result_dir / "reprofile.json").is_file() else "",
            "rollback": str((result_dir / "rollback.json").relative_to(root)),
        },
    }
    manifest["file_sha256"] = {str(path.relative_to(root)): sha256_file(path) for path in sorted(result_dir.rglob("*")) if path.is_file()}
    save_json_record_once(result_dir / "capture_manifest.json", manifest)
    return manifest


def build_review_config(
    project_protocol: Mapping[str, Any], project_root: str | Path, run_label: str
) -> dict[str, Any]:
    """Construct the decision config from the capture manifest and fixed protocol."""

    errors = validate_project_protocol(project_protocol)
    if errors:
        raise ValueError("项目协议错误：" + "; ".join(errors))
    root = Path(project_root).resolve()
    result_dir = _result_directory(root, run_label)
    manifest_path = result_dir / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少本项目测量清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "performance-delivery-capture/v1" or manifest.get("semantic_id") != SEMANTIC_ID
            or manifest.get("run_label") != run_label or manifest.get("candidate_order") != CANDIDATE_ORDER):
        raise ValueError("测量清单的项目身份、候选顺序或 RUN_LABEL 不匹配")
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("测量清单缺少文件哈希")
    for relative, digest in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(result_dir.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"测量文件缺失、路径超出本轮目录或 SHA256 不匹配：{relative}")
    paths = manifest["paths"]
    expected_paths = {
        "baseline": str((result_dir / "baseline.json").relative_to(root)),
        "candidates": {name: str((result_dir / f"{name}.json").relative_to(root)) for name in CANDIDATE_ORDER},
        "composition": str((result_dir / "composition.json").relative_to(root)),
        "rollback": str((result_dir / "rollback.json").relative_to(root)),
    }
    if any(paths.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("测量清单引用的输入不是本轮固定结果文件")
    required = [expected_paths["baseline"], expected_paths["composition"], expected_paths["rollback"], *expected_paths["candidates"].values()]
    if paths.get("reprofile"):
        if paths["reprofile"] != str((result_dir / "reprofile.json").relative_to(root)):
            raise ValueError("复测记录不在本轮固定路径内")
        required.append(paths["reprofile"])
    if any(path not in hashes for path in required):
        raise ValueError("测量清单遗漏必需输入的 SHA256")
    identity = manifest.get("workload_identity", {})
    if identity.get("workload_id") != WORKLOAD_ID or identity.get("model_revision") != MODEL_SNAPSHOT:
        raise ValueError("测量负载或模型版本不匹配")
    for relative in [expected_paths["baseline"], expected_paths["composition"], *expected_paths["candidates"].values()]:
        record = json.loads((root / relative).read_text(encoding="utf-8"))
        if record.get("semantic_id") != SEMANTIC_ID or record.get("run_label") != run_label or record.get("workload_identity") != identity:
            raise ValueError(f"测量文件的项目身份、标签或负载不匹配：{relative}")
    model_hashes = identity.get("model_file_sha256", {})
    if set(model_hashes) != {"config.json", "model.safetensors", "tokenizer_config.json"} or any(
        not (model_path_for(root) / name).is_file() or sha256_file(model_path_for(root) / name) != digest
        for name, digest in model_hashes.items()
    ):
        raise ValueError("固定基础模型文件与测量时的 SHA256 不一致")
    return {
        "schema_version": project_protocol["schema_version"],
        "semantic_id": project_protocol["semantic_id"],
        "environment_profile": project_protocol["environment_profile"],
        "run_label": run_label,
        "workload_identity": manifest["workload_identity"],
        "baseline_measurement": {"result_path": paths["baseline"]},
        "candidate_order": manifest["candidate_order"],
        "candidates": {
            name: {"result_path": paths["candidates"][name], "conflicts_with": []}
            for name in manifest["candidate_order"]
        },
        "composition": {"result_path": paths["composition"]},
        "reprofile": {"result_path": paths["reprofile"]},
        "rollback": {
            "ready": json.loads((root / paths["rollback"]).read_text(encoding="utf-8")).get("rollback_ready") is True,
            "evidence_path": paths["rollback"],
        },
        "context_sources": project_protocol.get("context_sources", []),
        "gates": project_protocol["gates"],
        "outputs": project_protocol["outputs"],
        "capture_status": manifest["status"],
        "capture_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_file(manifest_path)},
    }
