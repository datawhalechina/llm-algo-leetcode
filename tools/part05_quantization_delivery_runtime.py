"""Runtime helpers for the staged Part 05 quantization delivery project."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_ID = "QUANT-INTEGRATED-DELIVERY"
RESULT_SCHEMA = "part05-quantization-delivery/v1"
RESULT_ROOT = "benchmarks/results/part05_quantization_delivery"


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors = []
    if not isinstance(protocol, Mapping):
        return ["protocol must be a mapping"]
    for section in ("outputs", "model", "quantization", "serving"):
        if not isinstance(protocol.get(section), Mapping):
            errors.append(f"{section} must be a mapping")
    if errors:
        return errors
    if protocol.get("schema_version") != "quantization-delivery-config/v1":
        errors.append("unexpected schema_version")
    if protocol.get("semantic_id") != SEMANTIC_ID:
        errors.append("unexpected semantic_id")
    if protocol.get("quantization_environment") != "Q-CUDA-MECHANISM":
        errors.append("quantization stage must use Q-CUDA-MECHANISM")
    if protocol.get("serving_environment") != "INF-VLLM-011-CU128":
        errors.append("serving stage must use INF-VLLM-011-CU128")
    if protocol.get("outputs", {}).get("directory") != RESULT_ROOT:
        errors.append(f"outputs.directory must be {RESULT_ROOT}")
    model = protocol.get("model", {})
    if not isinstance(model.get("id"), str) or not model["id"].strip() or not isinstance(model.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        errors.append("model.id and immutable 40-character revision are required")
    quant = protocol.get("quantization", {})
    if (quant.get("method") != "GPTQ" or quant.get("format") != "W4A16"
            or quant.get("bits") != 4 or quant.get("group_size") != 128):
        errors.append("quantization must be GPTQ W4A16 group-128")
    serving = protocol.get("serving", {})
    if not serving.get("workload") or serving.get("num_prompts") != 20:
        errors.append("fixed 20-request serving workload is required")
    for key in ("dataset", "dataset_config", "calibration_split", "evaluation_split"):
        if not isinstance(quant.get(key), str) or not quant[key].strip():
            errors.append(f"quantization.{key} must be a nonempty string")
    if not isinstance(quant.get("dataset_revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", quant["dataset_revision"]):
        errors.append("dataset_revision must be an immutable commit")
    if quant.get("calibration_split") == quant.get("evaluation_split"):
        errors.append("calibration and evaluation splits must differ")
    for section, keys in ((quant, ("bits", "group_size", "calibration_samples", "evaluation_samples", "max_sequence_length")), (serving, ("num_prompts", "max_tokens", "concurrency", "repeats"))):
        for key in keys:
            if type(section.get(key)) is not int or section[key] <= 0:
                errors.append(f"{key} must be a positive integer")
    if type(serving.get("warmup")) is not int or serving["warmup"] < 0:
        errors.append("warmup must be a nonnegative integer")
    if not isinstance(serving.get("workload"), str) or not serving["workload"].strip():
        errors.append("serving.workload must be a nonempty path")
    for section, key, positive in (
        (quant, "maximum_nll_ratio_vs_bf16", True),
        (serving, "maximum_ttft_p95_ms", True), (serving, "maximum_tpot_p95_ms", True),
        (serving, "minimum_output_throughput_tps", True), (serving, "maximum_peak_memory_gib", True),
        (serving, "minimum_answer_accuracy", False), (serving, "maximum_answer_accuracy_regression", False),
    ):
        value = _number(section.get(key))
        if value is None or value < 0 or (positive and value == 0):
            errors.append(f"{key} must be a finite {'positive' if positive else 'nonnegative'} number")
        elif key in ("minimum_answer_accuracy", "maximum_answer_accuracy_regression") and value > 1:
            errors.append(f"{key} must be in [0, 1]")
    return errors


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _require_protocol(protocol: Mapping[str, Any]) -> None:
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("invalid quantization protocol: " + "; ".join(errors))


def _check_record(record: Mapping[str, Any], protocol: Mapping[str, Any], label: str, schema: str) -> None:
    if (record.get("schema_version") != schema or record.get("semantic_id") != SEMANTIC_ID
            or record.get("run_label") != label or record.get("protocol") != dict(protocol)):
        raise ValueError("阶段记录的项目身份或协议与当前配置不一致，请使用新 RUN_LABEL。")


def _report_protocol_matches(report: Mapping[str, Any], protocol: Mapping[str, Any], label: str) -> bool:
    workload = report.get("workload", {})
    serving = protocol["serving"]
    return (
        report.get("label") == label
        and all(workload.get(key) == serving[config_key] for key, config_key in (
            ("requests", "num_prompts"), ("max_tokens", "max_tokens"),
            ("concurrency", "concurrency"), ("warmup", "warmup"), ("repeats", "repeats"),
        ))
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _text_rows_hash(dataset: Any, count: int) -> str:
    """Hash the ordered calibration or evaluation texts selected by this run."""
    texts = [row["text"] for row in dataset.select(range(count))]
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _check_inputs(inputs: Mapping[str, Any], protocol: Mapping[str, Any], run_label: str) -> None:
    _check_record(inputs, protocol, run_label, "part05-quantization-inputs/v1")
    model = inputs.get("model", {})
    if any(model.get(key) != protocol["model"][key] for key in ("id", "revision")):
        raise ValueError("准备清单中的模型与固定配置不一致。")
    if not inputs.get("model_sha256") or hash_tree(model["path"]) != inputs["model_sha256"]:
        raise ValueError("基础模型文件缺失或 SHA256 与准备清单不一致。")


def _check_artifact_inputs(record: Mapping[str, Any], run_dir: Path, protocol: Mapping[str, Any], run_label: str) -> None:
    path = run_dir / "inputs.json"
    if not path.is_file() or record.get("inputs_sha256") != sha256_file(path):
        raise ValueError("量化记录与准备清单的 SHA256 不一致。")
    inputs = json.loads(path.read_text(encoding="utf-8"))
    _check_inputs(inputs, protocol, run_label)
    if any(record.get(key) != inputs.get(key) for key in ("model", "dataset", "quantization")):
        raise ValueError("量化记录中的模型、数据或量化配置与准备清单不一致。")


def save_once(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != dict(payload):
            raise FileExistsError(f"同一路径已有不同内容，不覆盖：{destination}")
    else:
        destination.write_text(rendered, encoding="utf-8")
    return destination


def _result_dir(project_root: str | Path, run_label: str) -> Path:
    if not isinstance(run_label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", run_label) or run_label == "replace_with_unique_label":
        raise ValueError("请填写唯一 RUN_LABEL。")
    return Path(project_root).resolve() / RESULT_ROOT / run_label


def prepare_quantization_inputs(
    *, protocol: Mapping[str, Any], project_root: str | Path, run_label: str,
    model_runtime: Any, load_dataset: Any,
) -> Path:
    """Download/resolve only the pinned model and dataset, then record identity."""
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("invalid quantization protocol: " + "; ".join(errors))
    root = Path(project_root).resolve()
    run_dir = _result_dir(root, run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path, model_source = model_runtime.resolve_model_snapshot(
        protocol["model"]["id"], protocol["model"]["revision"],
        cache_dir=root / "model_cache", allow_download=True,
    )
    quant = protocol["quantization"]
    datasets_root = root / "model_cache" / "datasets"
    train = load_dataset(
        quant["dataset"], quant["dataset_config"], split=quant["calibration_split"],
        revision=quant["dataset_revision"], cache_dir=str(datasets_root),
    )
    evaluation = load_dataset(
        quant["dataset"], quant["dataset_config"], split=quant["evaluation_split"],
        revision=quant["dataset_revision"], cache_dir=str(datasets_root),
    )
    train = train.filter(lambda row: bool(str(row.get("text", "")).strip()))
    evaluation = evaluation.filter(lambda row: bool(str(row.get("text", "")).strip()))
    n_cal = quant["calibration_samples"]
    n_eval = quant["evaluation_samples"]
    if len(train) < n_cal or len(evaluation) < n_eval:
        raise ValueError("固定数据切分中有效文本不足以满足样本数。")
    model_hashes = hash_tree(model_path)
    if not model_hashes:
        raise ValueError("固定模型目录没有可核验的文件。")
    manifest = {
        "schema_version": "part05-quantization-inputs/v1",
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "protocol": dict(protocol),
        "model": {"id": protocol["model"]["id"], "revision": protocol["model"]["revision"], "path": str(model_path), "source": model_source},
        "model_sha256": model_hashes,
        "dataset_sha256": {"calibration": _text_rows_hash(train, n_cal), "evaluation": _text_rows_hash(evaluation, n_eval)},
        "dataset": {
            "id": quant["dataset"], "config": quant["dataset_config"], "revision": quant["dataset_revision"],
            "calibration_split": quant["calibration_split"], "calibration_rows": n_cal,
            "evaluation_split": quant["evaluation_split"], "evaluation_rows": n_eval,
            "calibration_indices": list(range(n_cal)), "evaluation_indices": list(range(n_eval)),
        },
        "quantization": {key: quant[key] for key in ("method", "format", "bits", "group_size", "max_sequence_length")},
        "status": "prepared",
    }
    return save_once(run_dir / "inputs.json", manifest)


def _tokenize_rows(dataset: Any, tokenizer: Any, count: int, max_length: int) -> Any:
    selected = dataset.select(range(count))
    tokenized = selected.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False),
        batched=True,
        remove_columns=selected.column_names,
    )
    return tokenized


def _mean_nll(torch: Any, model: Any, tokenized_dataset: Any) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for row in tokenized_dataset:
            token_ids = row["input_ids"]
            if len(token_ids) < 2:
                continue
            input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
            logits = model(input_ids=input_ids).logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            loss_sum = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), labels.view(-1), reduction="sum"
            )
            total_loss += float(loss_sum.item())
            total_tokens += labels.numel()
    if total_tokens == 0:
        raise ValueError("留出集没有可计分 token。")
    return total_loss / total_tokens


def quantize_and_evaluate(
    *, torch: Any, auto_model_class: Any, tokenizer_class: Any, load_dataset: Any,
    protocol: Mapping[str, Any], project_root: str | Path, run_label: str,
) -> dict[str, Any]:
    """Measure BF16 and GPTQ held-out NLL, export compressed weights, hash files."""
    _require_protocol(protocol)
    root = Path(project_root).resolve()
    run_dir = _result_dir(root, run_label)
    manifest_path = run_dir / "inputs.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"先运行 prepare：{manifest_path}")
    inputs = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_inputs(inputs, protocol, run_label)
    existing_record = run_dir / "quantization.json"
    if existing_record.is_file():
        record = json.loads(existing_record.read_text(encoding="utf-8"))
        _check_record(record, protocol, run_label, "part05-quantization-artifact/v1")
        _check_artifact_inputs(record, run_dir, protocol, run_label)
        artifact_dir = root / record.get("artifact_path", "")
        if (record.get("run_label") != run_label or record.get("semantic_id") != SEMANTIC_ID
                or not artifact_dir.is_dir()
                or not record.get("artifact_sha256") or hash_tree(artifact_dir) != record.get("artifact_sha256")):
            raise ValueError("已有量化记录或产物哈希不匹配；请用新 RUN_LABEL 重跑。")
        return record
    quant = protocol["quantization"]
    if (run_dir / "artifacts" / "gptq_w4a16_g128").exists():
        raise FileExistsError("量化目录已有中断产物，请保留现场并使用新 RUN_LABEL。")
    tokenizer = tokenizer_class.from_pretrained(inputs["model"]["path"], local_files_only=True)
    train = load_dataset(
        quant["dataset"], quant["dataset_config"], split=quant["calibration_split"],
        revision=quant["dataset_revision"], cache_dir=str(root / "model_cache" / "datasets"),
    ).filter(lambda row: bool(str(row.get("text", "")).strip()))
    evaluation = load_dataset(
        quant["dataset"], quant["dataset_config"], split=quant["evaluation_split"],
        revision=quant["dataset_revision"], cache_dir=str(root / "model_cache" / "datasets"),
    ).filter(lambda row: bool(str(row.get("text", "")).strip()))
    if inputs.get("dataset_sha256") != {
        "calibration": _text_rows_hash(train, quant["calibration_samples"]),
        "evaluation": _text_rows_hash(evaluation, quant["evaluation_samples"]),
    }:
        raise ValueError("选中的校准或评测文本与准备清单不一致。")
    calibration_rows = _tokenize_rows(train, tokenizer, quant["calibration_samples"], quant["max_sequence_length"])
    evaluation_rows = _tokenize_rows(evaluation, tokenizer, quant["evaluation_samples"], quant["max_sequence_length"])
    model = auto_model_class.from_pretrained(
        inputs["model"]["path"], torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
    )
    baseline_nll = _mean_nll(torch, model, evaluation_rows)
    if _number(baseline_nll) is None or baseline_nll <= 0:
        raise ValueError("BF16 留出集 NLL 须为有限正数。")

    from compressed_tensors.offload import dispatch_model
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType,
    )
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier

    weights = QuantizationArgs(
        num_bits=4, type=QuantizationType.INT, strategy=QuantizationStrategy.GROUP,
        group_size=quant["group_size"], symmetric=True, dynamic=False,
    )
    recipe = GPTQModifier(
        targets="Linear",
        config_groups={"group_0": QuantizationScheme(targets=["Linear"], weights=weights)},
        ignore=["lm_head"],
    )
    oneshot(
        model=model, dataset=calibration_rows, recipe=recipe,
        max_seq_length=quant["max_sequence_length"],
        num_calibration_samples=quant["calibration_samples"],
    )
    dispatch_model(model)
    quantized_nll = _mean_nll(torch, model, evaluation_rows)
    if _number(quantized_nll) is None or quantized_nll < 0:
        raise ValueError("量化模型留出集 NLL 须为有限非负数。")
    artifact_dir = run_dir / "artifacts" / "gptq_w4a16_g128"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(artifact_dir, save_compressed=True)
    tokenizer.save_pretrained(artifact_dir)
    artifact_hashes = hash_tree(artifact_dir)
    if not artifact_hashes:
        raise RuntimeError("量化阶段未生成模型文件。")
    quant_config = getattr(model.config, "quantization_config", None)
    if hasattr(quant_config, "to_dict"):
        quant_config = quant_config.to_dict()
    if not isinstance(quant_config, Mapping) or not quant_config:
        raise RuntimeError("量化模型配置中没有可核验的 quantization_config。")
    artifact_record = {
        "schema_version": "part05-quantization-artifact/v1",
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "protocol": dict(protocol),
        "inputs_sha256": sha256_file(manifest_path),
        "model": inputs["model"],
        "dataset": inputs["dataset"],
        "quantization": inputs["quantization"],
        "quality": {
            "bf16_evaluation_nll": baseline_nll,
            "quantized_evaluation_nll": quantized_nll,
            "nll_ratio_vs_bf16": quantized_nll / baseline_nll if baseline_nll else math.inf,
        },
        "quantization_config": dict(quant_config),
        "artifact_path": str(artifact_dir.relative_to(root)),
        "artifact_sha256": artifact_hashes,
        "status": "complete",
    }
    save_once(run_dir / "quantization.json", artifact_record)
    return artifact_record


def run_service_pair(
    *, protocol: Mapping[str, Any], project_root: str | Path, run_label: str,
    start_server: Any, stop_server: Any, run_benchmark: Any,
) -> dict[str, Any]:
    """Benchmark the pinned BF16 and locally produced quantized model in vLLM."""
    _require_protocol(protocol)
    root = Path(project_root).resolve()
    run_dir = _result_dir(root, run_label)
    artifact_path = run_dir / "quantization.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"先在 Q-CUDA-MECHANISM 环境运行 quantize：{artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    _check_record(artifact, protocol, run_label, "part05-quantization-artifact/v1")
    _check_artifact_inputs(artifact, run_dir, protocol, run_label)
    if artifact.get("run_label") != run_label or artifact.get("semantic_id") != SEMANTIC_ID:
        raise ValueError("量化产物身份不匹配。")
    model_path = Path(artifact["model"]["path"])
    quantized_path = root / artifact["artifact_path"]
    if not artifact.get("artifact_sha256") or hash_tree(quantized_path) != artifact.get("artifact_sha256"):
        raise ValueError("量化产物哈希与 quantization.json 不匹配。")
    serving = protocol["serving"]
    workload_path = root / serving["workload"]
    if not workload_path.is_file():
        raise FileNotFoundError(f"固定请求集不存在：{workload_path}")
    workload_hash = sha256_file(workload_path)
    if len(_expected_request_ids(workload_path)) != serving["num_prompts"]:
        raise ValueError("固定请求集的请求数与配置不一致。")
    output_files = {"bf16": run_dir / "service_bf16.json", "gptq": run_dir / "service_gptq.json"}
    rows = {}
    for role, model in (("bf16", model_path), ("gptq", quantized_path)):
        if output_files[role].is_file():
            saved = json.loads(output_files[role].read_text(encoding="utf-8"))
            expected_path = str(model)
            execution = saved.get("service_execution", {})
            if (saved.get("run_label") != f"{run_label}_{role}"
                    or not _report_protocol_matches(saved, protocol, f"{run_label}_{role}")
                    or execution.get("requested_model_path") != expected_path
                    or execution.get("resolved_model_path") != expected_path
                    or execution.get("protocol") != dict(protocol)
                    or execution.get("workload_sha256") != workload_hash):
                raise ValueError(f"已有 {role} 服务结果与本次模型路径或 RUN_LABEL 不匹配；请用新标签。")
            rows[role] = saved
            continue
        server = log_handle = None
        for name in (f"benchmark_{role}.json", f"vllm_{role}.log"):
            if (run_dir / name).exists():
                raise FileExistsError(f"服务阶段存在中断记录，请使用新 RUN_LABEL：{run_dir / name}")
        try:
            server, log_handle, port, selected_dtype, resolved_path = start_server(
                model_id=str(model), model_source="local", cache_dir=root / "model_cache",
                dtype="bfloat16", max_model_len=512, gpu_memory_utilization=0.45,
                enforce_eager=True, served_model_name=f"part05-quant-{role}",
                enable_prefix_caching=False,
                log_path=run_dir / f"vllm_{role}.log",
            )
            rows[role] = run_benchmark(
                project="part05-quantization-delivery",
                base_url=f"http://127.0.0.1:{port}",
                model=f"part05-quant-{role}",
                label=f"{run_label}_{role}",
                output=run_dir / f"benchmark_{role}.json", workload=serving["workload"],
                num_prompts=serving["num_prompts"], max_tokens=serving["max_tokens"],
                concurrency=serving["concurrency"], warmup=serving["warmup"],
                repeats=serving["repeats"], temperature=0.0, timeout_s=180.0,
                backend="vllm", dtype=selected_dtype, batch=1,
                cache_policy="disabled", role="baseline" if role == "bf16" else "candidate",
                measure_device_peak_memory=True, include_requests=True, quiet=True,
            )
            if not _report_protocol_matches(rows[role], protocol, f"{run_label}_{role}"):
                raise ValueError("服务压测报告中的标签、请求数或测量参数与配置不一致。")
            rows[role]["run_label"] = f"{run_label}_{role}"
            log_path = Path(getattr(log_handle, "name", "")) if log_handle else None
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path and log_path.is_file() else ""
            rows[role]["service_execution"] = {
                "protocol": dict(protocol), "workload_sha256": workload_hash,
                "requested_model_path": str(model), "resolved_model_path": str(resolved_path),
                "log_path": str(log_path) if log_path else None,
                "log_tail": log_text[-6000:], "backend": "vllm", "dtype": selected_dtype,
                "quantization_runtime_observed": (
                    any(marker in log_text.lower() for marker in ("compressed-tensors", "compressed_tensors", "gptq"))
                    if role == "gptq" else None
                ),
            }
            save_once(output_files[role], rows[role])
        finally:
            if server is not None:
                stop_server(server, log_handle)
    pair = {
        "schema_version": "part05-quantization-service-pair/v1",
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "protocol": dict(protocol),
        "artifact_manifest": str(artifact_path.relative_to(root)),
        "artifact_sha256": artifact["artifact_sha256"],
        "workload_sha256": sha256_file(workload_path),
        "members": {role: str(path.relative_to(root)) for role, path in output_files.items()},
        "member_sha256": {role: sha256_file(path) for role, path in output_files.items()},
        "status": "complete",
    }
    save_once(run_dir / "service_pair.json", pair)
    return pair


def _metric(report: Mapping[str, Any], name: str, stat: str = "p95") -> float | None:
    normalized = report.get("normalized_result", report)
    metrics = normalized.get("metrics", {}) if isinstance(normalized, Mapping) else {}
    value = metrics.get(name)
    if isinstance(value, Mapping):
        value = value.get(stat, value.get("mean"))
    result = _number(value)
    return result if result is not None and result >= 0 else None


def _accuracy(report: Mapping[str, Any]) -> float | None:
    normalized = report.get("normalized_result", report)
    quality = normalized.get("quality", {}) if isinstance(normalized, Mapping) else {}
    for key in ("complete_correct_rate", "expected_answer_accuracy"):
        value = quality.get(key)
        if value is None:
            value = report.get("metrics", {}).get(key)
        if value is not None:
            number = _number(value)
            return number if number is not None and 0 <= number <= 1 else None
    return None


def decide_quantization_delivery(project_root: str | Path, run_label: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Verify lineage, paired quality/performance gates, and record final decision."""
    _require_protocol(protocol)
    root = Path(project_root).resolve()
    run_dir = _result_dir(root, run_label)
    paths = {"quantization": run_dir / "quantization.json", "pair": run_dir / "service_pair.json"}
    paths.update({"bf16": run_dir / "service_bf16.json", "gptq": run_dir / "service_gptq.json"})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("阶段证据缺失：" + "; ".join(missing))
    artifact = json.loads(paths["quantization"].read_text(encoding="utf-8"))
    pair = json.loads(paths["pair"].read_text(encoding="utf-8"))
    _check_record(artifact, protocol, run_label, "part05-quantization-artifact/v1")
    _check_artifact_inputs(artifact, run_dir, protocol, run_label)
    _check_record(pair, protocol, run_label, "part05-quantization-service-pair/v1")
    baseline = json.loads(paths["bf16"].read_text(encoding="utf-8"))
    candidate = json.loads(paths["gptq"].read_text(encoding="utf-8"))
    if any(record.get("run_label") != run_label for record in (artifact, pair)):
        raise ValueError("量化产物或服务配对记录的 RUN_LABEL 不匹配。")
    candidate_path = root / artifact["artifact_path"]
    workload_path = root / protocol["serving"]["workload"]
    hash_match = (
        bool(artifact.get("artifact_sha256"))
        and hash_tree(candidate_path) == artifact.get("artifact_sha256") == pair.get("artifact_sha256")
        and pair.get("member_sha256") == {role: sha256_file(paths[role]) for role in ("bf16", "gptq")}
        and pair.get("workload_sha256") == sha256_file(workload_path)
        and pair.get("artifact_manifest") == str(paths["quantization"].relative_to(root))
        and pair.get("semantic_id") == SEMANTIC_ID
        and pair.get("members") == {
            "bf16": str(paths["bf16"].relative_to(root)),
            "gptq": str(paths["gptq"].relative_to(root)),
        }
    )
    expected_ids = _expected_request_ids(root / protocol["serving"]["workload"])
    baseline_ids, candidate_ids = _request_ids(baseline), _request_ids(candidate)
    same_requests = (
        bool(expected_ids)
        and Counter(baseline_ids) == Counter(candidate_ids)
        and set(baseline_ids) == expected_ids
        and set(candidate_ids) == expected_ids
        and all(count == protocol["serving"]["repeats"] for count in Counter(baseline_ids).values())
        and all(count == protocol["serving"]["repeats"] for count in Counter(candidate_ids).values())
    )
    quality = artifact.get("quality", {})
    ratio = _number(quality.get("nll_ratio_vs_bf16"))
    base_nll = _number(quality.get("bf16_evaluation_nll"))
    candidate_nll = _number(quality.get("quantized_evaluation_nll"))
    nll_complete = (
        ratio is not None and ratio >= 0
        and base_nll is not None and base_nll > 0
        and candidate_nll is not None and candidate_nll >= 0
        and math.isclose(ratio, candidate_nll / base_nll, rel_tol=1e-6, abs_tol=1e-9)
    )
    nll_pass = nll_complete and ratio <= protocol["quantization"]["maximum_nll_ratio_vs_bf16"]
    baseline_accuracy, candidate_accuracy = _accuracy(baseline), _accuracy(candidate)
    accuracy_complete = baseline_accuracy is not None and candidate_accuracy is not None
    serving = protocol["serving"]
    ttft = _metric(candidate, "ttft_ms")
    tpot = _metric(candidate, "tpot_ms")
    throughput = _metric(candidate, "throughput_tokens_per_s", stat="mean")
    peak_memory = _metric(candidate, "peak_memory_mb", stat="mean")
    service_metrics_complete = all(value is not None for value in (ttft, tpot, throughput, peak_memory))
    request_counts_complete = all(
        report.get("metrics", {}).get("successful_requests") == serving["num_prompts"]
        and report.get("metrics", {}).get("failed_requests") == 0
        for report in (baseline, candidate)
    )
    accuracy_pass = accuracy_complete and candidate_accuracy >= serving["minimum_answer_accuracy"]
    regression_pass = accuracy_complete and (baseline_accuracy - candidate_accuracy) <= serving["maximum_answer_accuracy_regression"]
    service_pass = service_metrics_complete and (
        ttft <= serving["maximum_ttft_p95_ms"]
        and tpot <= serving["maximum_tpot_p95_ms"]
        and throughput >= serving["minimum_output_throughput_tps"]
        and peak_memory <= serving["maximum_peak_memory_gib"] * 1024
    )
    execution_paths = (
        baseline.get("service_execution", {}).get("resolved_model_path") == artifact["model"]["path"]
        and candidate.get("service_execution", {}).get("resolved_model_path") == str(candidate_path)
        and candidate.get("service_execution", {}).get("quantization_runtime_observed") is True
    )
    reports_match = all(
        report.get("run_label") == f"{run_label}_{role}"
        and _report_protocol_matches(report, protocol, f"{run_label}_{role}")
        and report.get("service_execution", {}).get("protocol") == dict(protocol)
        and report.get("service_execution", {}).get("workload_sha256") == pair.get("workload_sha256")
        for role, report in (("bf16", baseline), ("gptq", candidate))
    )
    complete = hash_match and reports_match and nll_complete and same_requests and accuracy_complete and service_metrics_complete and request_counts_complete
    if not complete:
        decision = "pending"
    elif not execution_paths:
        decision = "reject"
    elif not nll_pass:
        decision = "reject"
    elif accuracy_pass and regression_pass and service_pass:
        decision = "accept"
    else:
        decision = "tune"
    result = {
        "schema_version": RESULT_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": run_label,
        "decision": decision,
        "gates": {
            "artifact_hash_match": hash_match,
            "same_fixed_requests": same_requests,
            "all_requests_succeeded": request_counts_complete,
            "quantized_path_matches_artifact": execution_paths,
            "nll_ratio": nll_pass,
            "answer_accuracy": accuracy_pass,
            "accuracy_regression": regression_pass,
            "service_slo": service_pass,
        },
        "metrics": {
            "nll_ratio_vs_bf16": ratio,
            "bf16_complete_correct_rate": baseline_accuracy,
            "gptq_complete_correct_rate": candidate_accuracy,
            "gptq_ttft_p95_ms": ttft,
            "gptq_tpot_p95_ms": tpot,
            "gptq_output_throughput_tokens_per_s": throughput,
            "gptq_peak_memory_mb": peak_memory,
        },
        "evidence": {key: str(path.relative_to(root)) for key, path in paths.items()},
    }
    save_once(run_dir / "decision.json", result)
    return result


def _request_ids(report: Mapping[str, Any]) -> list[str]:
    runs = report.get("runs", [])
    if runs:
        results = [item for run in runs for item in run.get("results", [])]
    else:
        results = report.get("results", [])
    if not results:
        results = report.get("normalized_result", {}).get("results", [])
    return [str(item.get("request_id")) for item in results if item.get("request_id") is not None]


def _expected_request_ids(path: str | Path) -> set[str]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["request_id"]) for row in rows if row.get("request_id") is not None]
    if not ids or len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError("固定请求集须为每条请求提供唯一 request_id。")
    return set(ids)
