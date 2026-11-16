"""Runtime helpers for the Part 05 MiniMind post-training project."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "minimind-post-training-config/v1"
RESULT_SCHEMA = "minimind-post-training-delivery/v1"
SEMANTIC_ID = "PT-INTEGRATED-DELIVERY"
MINIMIND_LORA_RANK = 16


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hash_errors(path: Any, digest: Any, root: Path, name: str) -> list[str]:
    """Check a recorded input against the file that will actually be read."""
    if not isinstance(path, str) or not path:
        return [f"{name} 缺少文件路径"]
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    if not target.is_file():
        return [f"{name} 文件缺失：{target}"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(target) != digest:
        return [f"{name} 文件 SHA256 不匹配"]
    return []


def _check_prepared_inputs(config: Mapping[str, Any], project_root: str | Path, prepared: Mapping[str, Any]) -> None:
    """Reject changed input files or changed configuration before GPU work."""
    root = Path(project_root).resolve()
    errors = validate_config(config)
    if prepared.get("config") != dict(config):
        errors.append("准备清单与当前配置不一致，请使用新标签重新准备")
    errors.extend(_file_hash_errors(prepared.get("base_weight_path"), prepared.get("base_weight_sha256"), root, "基础权重"))
    for name in ("sft", "dpo", "eval_sft", "eval_dpo"):
        errors.extend(_file_hash_errors(prepared.get("data_paths", {}).get(name), prepared.get("data_sha256", {}).get(name), root, name))
    if errors:
        raise ValueError("准备输入校验失败：" + "; ".join(errors))
    _check_source_identity(Path(prepared["source_path"]), config["source"]["revision"])


def validate_training_evidence(raw: Mapping[str, Any], config: Mapping[str, Any], project_root: str | Path) -> list[str]:
    """Check the complete, persisted evidence bundle before project closeout."""
    errors: list[str] = []
    label = config["run_label"]
    if raw.get("schema_version") != RESULT_SCHEMA or raw.get("semantic_id") != SEMANTIC_ID:
        errors.append("原始结果的 schema 或 semantic_id 不匹配")
    if raw.get("run_label") != label:
        errors.append("原始结果 RUN_LABEL 不匹配")
    if raw.get("environment_profile") != config.get("environment_profile"):
        errors.append("原始结果环境 Profile 与配置不匹配")

    source = raw.get("source", {})
    if config["gates"].get("require_fixed_source_revision") and source.get("revision") != config["source"].get("revision"):
        errors.append("MiniMind 源码提交与固定版本不匹配")
    model = raw.get("model", {})
    base_hash = model.get("weight_sha256")
    if not isinstance(base_hash, str) or len(base_hash) != 64:
        errors.append("缺少有效的基础模型权重 SHA256")

    workload = raw.get("workload", {})
    for name, expected in config["workload"].items():
        if workload.get(name) != expected:
            errors.append(f"训练负载配置与本次结果不一致：{name}")
    for name, expected in config["model"].items():
        if model.get(name) != expected:
            errors.append(f"模型配置与本次结果不一致：{name}")
    expected_thresholds = {"evaluation": dict(config["evaluation"]), "memory": dict(config["gates"])}
    if raw.get("decision_thresholds") != expected_thresholds:
        errors.append("当前收口门槛与训练时登记的门槛不一致")
    if config["evaluation"].get("require_train_eval_disjoint") and workload.get("split_disjoint") is not True:
        errors.append("训练集与评测集没有通过隔离检查")
    expected_counts = {
        "sft": int(config["workload"]["sft_examples"]),
        "dpo": int(config["workload"]["dpo_pairs"]),
        "evaluation": int(config["workload"]["evaluation_examples"]),
    }
    if workload.get("data_counts") != expected_counts:
        errors.append("固定数据集样本数与本次配置不一致")
    data_hashes = workload.get("data_sha256", {})
    data_paths = workload.get("data_paths", {})
    for name in ("sft", "dpo", "eval_sft", "eval_dpo"):
        value = data_hashes.get(name)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"缺少数据切分 {name} 的 SHA256")
        if not isinstance(data_paths.get(name), str) or not data_paths[name]:
            errors.append(f"缺少数据切分 {name} 的文件路径")
        errors.extend(_file_hash_errors(data_paths.get(name), value, Path(project_root).resolve(), name))

    base = raw.get("evaluation", {}).get("base", {})
    errors.extend(_file_hash_errors(base.get("weight_path"), base_hash, Path(project_root).resolve(), "基础权重"))

    probe = raw.get("memory_admission_probe", {})
    if probe.get("status") != "passed" or probe.get("run_label") != label:
        errors.append("显存准入探测缺失、未通过或标签不匹配")
    if probe.get("source_revision") != source.get("revision") or probe.get("base_weight_sha256") != base_hash:
        errors.append("显存探测与训练结果使用的源码或基座权重不一致")
    if probe.get("data_sha256") != data_hashes:
        errors.append("显存探测与训练使用的数据身份不一致")
    if probe.get("gates") != dict(config["gates"]):
        errors.append("显存探测门槛与当前配置不一致")
    phases = probe.get("phases", {})
    for name in ("sft", "sft_lora", "dpo"):
        row = phases.get(name, {})
        peak_bytes = row.get("peak_reserved_bytes")
        if row.get("loss_finite") is not True or not isinstance(peak_bytes, int) or peak_bytes <= 0:
            errors.append(f"缺少有效的 {name} 短步训练与显存证据")

    expected_stages = ("sft", "sft_lora", "dpo")
    logs = raw.get("logs", {})
    project_root = Path(project_root).resolve()
    for stage in expected_stages:
        row = logs.get(stage, {})
        log_path = Path(row.get("path", ""))
        if not log_path.is_absolute():
            log_path = project_root / log_path
        if row.get("returncode") != 0 or not log_path.is_file():
            errors.append(f"{stage} 训练日志缺失或运行失败")
        elapsed = row.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed < 0:
            errors.append(f"{stage} 缺少有效的阶段耗时")
        if source.get("path") and data_paths.get("sft") and data_paths.get("dpo"):
            data_key = "dpo" if stage == "dpo" else "sft"
            expected_command = _train_command(Path(source["path"]), config, stage, data_paths[data_key])
            if row.get("command") != expected_command:
                errors.append(f"{stage} 训练命令与固定配置不一致")

    evaluation = raw.get("evaluation", {})
    if set(evaluation) != {"base", "sft", "sft_lora", "dpo"}:
        errors.append("评测候选必须完整包含 base、sft、sft_lora、dpo")
    required_metrics = (
        "heldout_sft_loss_mean", "heldout_preference_win_rate", "evaluation_examples",
        "preference_pairs", "total_parameter_count", "artifact_bytes",
        "peak_allocated_bytes", "peak_reserved_bytes",
    )
    for name in ("base", "sft", "sft_lora", "dpo"):
        row = evaluation.get(name, {})
        for metric in required_metrics:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{name} 缺少有效指标 {metric}")
        loss = row.get("heldout_sft_loss_mean")
        win_rate = row.get("heldout_preference_win_rate")
        if isinstance(loss, (int, float)) and loss < 0:
            errors.append(f"{name} 留出损失不能为负")
        if isinstance(win_rate, (int, float)) and not 0 <= win_rate <= 1:
            errors.append(f"{name} 偏好胜率必须在 0 到 1 之间")
        for metric in ("evaluation_examples", "preference_pairs", "total_parameter_count", "artifact_bytes", "peak_reserved_bytes"):
            if isinstance(row.get(metric), (int, float)) and row[metric] <= 0:
                errors.append(f"{name} 指标 {metric} 必须大于 0")
        if row.get("evaluation_examples") != int(config["workload"]["evaluation_examples"]):
            errors.append(f"{name} 留出样本数与配置不一致")
        if row.get("preference_pairs") != int(config["workload"]["evaluation_examples"]):
            errors.append(f"{name} 偏好对数量与配置不一致")
        artifact = Path(row.get("weight_path", ""))
        if not artifact.is_absolute():
            artifact = project_root / artifact
        if not artifact.is_file():
            errors.append(f"{name} 模型或 adapter 文件缺失")
        elif row.get("weight_sha256") != sha256_file(artifact) or row.get("artifact_bytes") != artifact.stat().st_size:
            errors.append(f"{name} 模型或 adapter 的哈希/文件大小不匹配")
        adapter_count = row.get("adapter_parameter_count")
        if name == "sft_lora" and (not isinstance(adapter_count, int) or adapter_count <= 0):
            errors.append("LoRA 评测记录缺少可训练 adapter 参数量")

    return errors


def validate_config(config: Mapping[str, Any], *, allow_placeholder: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, Mapping):
        return ["项目配置须为键值映射"]
    if config.get("schema_version") != CONFIG_SCHEMA:
        errors.append(f"schema_version 必须是 {CONFIG_SCHEMA}")
    if config.get("semantic_id") != SEMANTIC_ID:
        errors.append(f"semantic_id 必须是 {SEMANTIC_ID}")
    label = config.get("run_label", "")
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", label) or (label == "replace_with_unique_label" and not allow_placeholder):
        errors.append("run_label 须为 3–64 位字母、数字、点、下划线或连字符")
    if config.get("environment_profile") != "PT-SFT-CUDA":
        errors.append("本项目固定使用 PT-SFT-CUDA")
    for section, keys in {
        "source": ("repository", "revision", "path"),
        "model": ("repository", "revision", "weight", "weight_prefix", "hidden_size", "num_hidden_layers", "use_moe"),
        "workload": ("seed", "max_seq_len", "sft_examples", "dpo_pairs", "evaluation_examples", "batch_size", "gradient_accumulation_steps", "sft_epochs", "dpo_epochs", "sft_learning_rate"),
        "outputs": ("directory",),
        "evaluation": ("protocol_version", "require_train_eval_disjoint", "maximum_sft_loss_regression_vs_base", "maximum_lora_loss_regression_vs_full_sft", "minimum_dpo_preference_win_rate"),
        "gates": ("maximum_peak_memory_gib", "minimum_free_memory_gib_before_load", "require_model_artifact_hash", "require_fixed_source_revision"),
    }.items():
        values = config.get(section)
        if not isinstance(values, Mapping):
            errors.append(f"缺少 {section} 配置")
            continue
        for key in keys:
            if key not in values or values[key] in (None, ""):
                errors.append(f"缺少 {section}.{key}")
    if errors:
        return errors
    workload, model = config["workload"], config["model"]
    if type(workload["seed"]) is not int or workload["seed"] != 42:
        errors.append("当前固定评测协议使用整数 seed=42")
    for key in ("max_seq_len", "sft_examples", "dpo_pairs", "evaluation_examples", "batch_size", "gradient_accumulation_steps", "sft_epochs", "dpo_epochs"):
        if type(workload[key]) is not int or workload[key] <= 0:
            errors.append(f"workload.{key} 须为正整数")
    for key, capacity in (("sft_examples", 990**2), ("dpo_pairs", 9000**2), ("evaluation_examples", 90000**2)):
        if type(workload[key]) is int and workload[key] > capacity:
            errors.append(f"workload.{key} 超过该切分的不同算式数量")
    for key in ("hidden_size", "num_hidden_layers"):
        if type(model[key]) is not int or model[key] <= 0:
            errors.append(f"model.{key} 须为正整数")
    if model["use_moe"] is not False or model["weight_prefix"] != "pretrain" or model["weight"] != f"pretrain_{model['hidden_size']}.pth":
        errors.append("当前训练与评测路径要求稠密模型及 pretrain_<hidden_size>.pth 基础权重")
    for section in ("source", "model"):
        revision = config[section]["revision"]
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            errors.append(f"{section}.revision 须为完整提交哈希")
    numeric_rules = (
        ("workload", "sft_learning_rate", True),
        ("gates", "maximum_peak_memory_gib", True),
        ("gates", "minimum_free_memory_gib_before_load", False),
        ("evaluation", "maximum_sft_loss_regression_vs_base", False),
        ("evaluation", "maximum_lora_loss_regression_vs_full_sft", False),
        ("evaluation", "minimum_dpo_preference_win_rate", False),
    )
    for section, key, positive in numeric_rules:
        value = config[section][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            errors.append(f"{section}.{key} 须为有限{'正' if positive else '非负'}实数")
    win_rate = config["evaluation"]["minimum_dpo_preference_win_rate"]
    if type(win_rate) in (int, float) and win_rate > 1:
        errors.append("偏好胜率门槛须在 0 到 1 之间")
    for section, key in (("gates", "require_model_artifact_hash"), ("gates", "require_fixed_source_revision"), ("evaluation", "require_train_eval_disjoint")):
        if config[section][key] is not True:
            errors.append(f"本协议要求 {section}.{key}=true")
    if config["evaluation"]["protocol_version"] != "minimind-small-task-eval-v1":
        errors.append("评测协议须为 minimind-small-task-eval-v1")
    return errors


def _write_jsonl_once(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    encoded = payload.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"同标签数据已存在且内容不同：{path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def build_fixed_workload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create deterministic, disjoint arithmetic SFT, DPO and held-out records."""
    import random

    errors = validate_config(config, allow_placeholder=True)
    if errors:
        raise ValueError("项目配置错误：" + "; ".join(errors))
    workload = config["workload"]
    rng = random.Random(int(workload["seed"]))

    def make_cases(count: int, low: int, high: int, prefix: str) -> list[dict[str, Any]]:
        rows = []
        seen: set[tuple[int, int]] = set()
        while len(rows) < count:
            a, b = rng.randint(low, high), rng.randint(low, high)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append({"id": f"{prefix}-{len(rows):03d}", "a": a, "b": b, "answer": a + b})
        return rows

    # Separate number ranges keep the held-out examples distinct from training prompts.
    sft_cases = make_cases(int(workload["sft_examples"]), 10, 999, "sft")
    dpo_cases = make_cases(int(workload["dpo_pairs"]), 1000, 9999, "dpo")
    eval_cases = make_cases(int(workload["evaluation_examples"]), 10000, 99999, "eval")

    def user_prompt(case: Mapping[str, Any]) -> str:
        return f"计算 {case['a']} + {case['b']}。只输出最终整数。"

    def sft_message(role: str, content: str) -> dict[str, str]:
        # Match MiniMind SFTDataset's declared Features, including optional fields.
        return {"role": role, "content": content, "reasoning_content": "", "tools": "", "tool_calls": ""}

    sft = [{"conversations": [
        sft_message("user", user_prompt(case)),
        sft_message("assistant", str(case["answer"])),
    ]} for case in sft_cases]
    dpo = []
    for case in dpo_cases:
        dpo.append({
            "chosen": [
                {"role": "user", "content": user_prompt(case)},
                {"role": "assistant", "content": str(case["answer"])},
            ],
            "rejected": [
                {"role": "user", "content": user_prompt(case)},
                {"role": "assistant", "content": str(case["answer"] + 1)},
            ],
        })
    eval_sft = [{"conversations": [
        sft_message("user", user_prompt(case)),
        sft_message("assistant", str(case["answer"])),
    ]} for case in eval_cases]
    eval_dpo = []
    for case in eval_cases:
        eval_dpo.append({
            "chosen": [
                {"role": "user", "content": user_prompt(case)},
                {"role": "assistant", "content": str(case["answer"])},
            ],
            "rejected": [
                {"role": "user", "content": user_prompt(case)},
                {"role": "assistant", "content": str(case["answer"] + 1)},
            ],
        })

    all_train_prompts = {user_prompt(case) for case in sft_cases + dpo_cases}
    eval_prompts = {user_prompt(case) for case in eval_cases}
    if all_train_prompts & eval_prompts:
        raise AssertionError("训练提示与评测提示发生重叠")

    return {
        "sft": sft,
        "dpo": dpo,
        "eval_sft": eval_sft,
        "eval_dpo": eval_dpo,
        "counts": {"sft": len(sft), "dpo": len(dpo), "evaluation": len(eval_sft)},
        "split_disjoint": True,
    }


def _check_source_identity(path: Path, expected: str) -> None:
    """Check the pinned checkout without cloning or altering its files."""
    current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if current != expected:
        raise RuntimeError(f"MiniMind 源码版本不匹配：{current} != {expected}")
    changes = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    additions = subprocess.check_output(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "--", "model", "trainer", "dataset"], text=True,
    ).strip()
    if changes or additions:
        raise RuntimeError("MiniMind 源码存在本地修改或新增代码，请保留当前目录并使用干净的固定版本目录")


def _ensure_source_checkout(source: Mapping[str, Any]) -> Path:
    path = Path(source["path"]).expanduser().resolve()
    expected = str(source["revision"])
    created = False
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", source["repository"], str(path)], check=True)
        created = True
    if not (path / ".git").is_dir():
        raise RuntimeError(f"MiniMind 路径存在但不是 Git 仓库：{path}")
    current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if created and current != expected:
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", expected], check=True)
        current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if current != expected:
        raise RuntimeError(f"MiniMind 源码版本不匹配：{current} != {expected}。请使用项目专用目录重新准备固定版本。")
    required = (
        path / "trainer/train_full_sft.py",
        path / "trainer/train_lora.py",
        path / "trainer/train_dpo.py",
        path / "model/model_minimind.py",
        path / "model/model_lora.py",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"固定版本缺少 MiniMind 训练入口：{missing}")
    _check_source_identity(path, expected)
    return path


def _prepare_weight(config: Mapping[str, Any], source_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    model = config["model"]
    downloaded = Path(hf_hub_download(
        repo_id=model["repository"], filename=model["weight"], revision=model["revision"]
    ))
    output = source_root / "out" / model["weight"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and sha256_file(output) != sha256_file(downloaded):
        raise FileExistsError(f"MiniMind 基础权重已存在且哈希不同：{output}")
    if not output.exists():
        shutil.copy2(downloaded, output)
    return output


def prepare_project(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    errors = validate_config(config)
    if errors:
        raise ValueError("项目配置错误：" + "; ".join(errors))
    source_root = _ensure_source_checkout(config["source"])
    weight_path = _prepare_weight(config, source_root)
    output_root = Path(project_root) / config["outputs"]["directory"] / config["run_label"]
    data_dir = output_root / "data"
    workload = build_fixed_workload(config)
    hashes = {}
    for key in ("sft", "dpo", "eval_sft", "eval_dpo"):
        hashes[key] = _write_jsonl_once(data_dir / f"{key}.jsonl", workload[key])
    prepared = {
        "config": dict(config),
        "source_path": str(source_root),
        "source_revision": config["source"]["revision"],
        "base_weight_path": str(weight_path),
        "base_weight_sha256": sha256_file(weight_path),
        "data_paths": {key: str(data_dir / f"{key}.jsonl") for key in hashes},
        "data_sha256": hashes,
        "data_counts": workload["counts"],
        "split_disjoint": workload["split_disjoint"],
    }
    manifest = output_root / "prepare_manifest.json"
    payload = json.dumps(prepared, ensure_ascii=False, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"准备清单已存在且内容不同，请更换 RUN_LABEL：{manifest}")
    manifest.write_text(payload, encoding="utf-8")
    return prepared


def load_prepared(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root) / config["outputs"]["directory"] / config["run_label"] / "prepare_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"先运行 RUN_STAGE='prepare'：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_revision") != config["source"]["revision"]:
        raise ValueError("准备清单的 MiniMind 版本与配置不一致")
    _check_prepared_inputs(config, project_root, payload)
    return payload


def probe_training_memory(config: Mapping[str, Any], project_root: str | Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Run one full-SFT, LoRA-SFT and DPO optimizer step before full training."""
    _check_prepared_inputs(config, project_root, prepared)
    import gc
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法执行训练显存准入探测")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gates = config["gates"]
    minimum = float(gates["minimum_free_memory_gib_before_load"])
    if free_bytes / 1024**3 < minimum:
        raise RuntimeError(f"探测前可用显存不足：{free_bytes / 1024**3:.2f} GiB < {minimum:.2f} GiB")

    source_root = Path(prepared["source_path"])
    sys.path.insert(0, str(source_root))
    from dataset.lm_dataset import DPODataset, SFTDataset
    from model.model_lora import apply_lora
    from model.model_minimind import MiniMindConfig
    from trainer.train_dpo import dpo_loss, logits_to_log_probs
    from trainer.trainer_utils import init_model

    model_cfg = MiniMindConfig(
        hidden_size=int(config["model"]["hidden_size"]),
        num_hidden_layers=int(config["model"]["num_hidden_layers"]),
        use_moe=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source_root / "model"))
    device = "cuda:0"
    phases: dict[str, Any] = {}

    def load_upstream_model(weight_prefix: str):
        # MiniMind's init_model resolves ../out relative to trainer/.
        previous_cwd = Path.cwd()
        try:
            os.chdir(source_root / "trainer")
            return init_model(model_cfg, weight_prefix, device=device)
        finally:
            os.chdir(previous_cwd)

    # SFT probe includes BF16 forward/backward and AdamW state allocation.
    model, _ = load_upstream_model(config["model"]["weight_prefix"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["workload"]["sft_learning_rate"]))
    batch = next(iter(DataLoader(SFTDataset(
        prepared["data_paths"]["sft"], tokenizer,
        max_length=int(config["workload"]["max_seq_len"]),
    ), batch_size=int(config["workload"]["batch_size"]))))
    input_ids, labels = (value.to(device) for value in batch)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels=labels)
        loss = output.loss + output.aux_loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    phases["sft"] = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "sequence_length": int(input_ids.shape[-1]),
    }
    del model, optimizer, batch, input_ids, labels, output, loss
    gc.collect()
    torch.cuda.empty_cache()

    # LoRA uses MiniMind's native adapter path, initialized from the same pretrain weights.
    model, _ = load_upstream_model(config["model"]["weight_prefix"])
    apply_lora(model, rank=MINIMIND_LORA_RANK)
    lora_parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name
        if parameter.requires_grad:
            lora_parameters.append(parameter)
    if not lora_parameters:
        raise RuntimeError("MiniMind LoRA 注入后没有可训练参数")
    model.train()
    optimizer = torch.optim.AdamW(lora_parameters, lr=float(config["workload"]["sft_learning_rate"]))
    batch = next(iter(DataLoader(SFTDataset(
        prepared["data_paths"]["sft"], tokenizer,
        max_length=int(config["workload"]["max_seq_len"]),
    ), batch_size=int(config["workload"]["batch_size"]))))
    input_ids, labels = (value.to(device) for value in batch)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels=labels)
        loss = output.loss + output.aux_loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    phases["sft_lora"] = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "sequence_length": int(input_ids.shape[-1]),
        "lora_rank": MINIMIND_LORA_RANK,
        "trainable_parameter_count": sum(parameter.numel() for parameter in lora_parameters),
    }
    del model, optimizer, lora_parameters, batch, input_ids, labels, output, loss
    gc.collect()
    torch.cuda.empty_cache()

    # DPO probe mirrors the upstream simultaneous reference/policy model path.
    policy, _ = load_upstream_model(config["model"]["weight_prefix"])
    reference, _ = load_upstream_model(config["model"]["weight_prefix"])
    policy.train()
    reference.eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=4e-8)
    batch = next(iter(DataLoader(DPODataset(
        prepared["data_paths"]["dpo"], tokenizer,
        max_length=int(config["workload"]["max_seq_len"]),
    ), batch_size=int(config["workload"]["batch_size"]))))
    x = torch.cat([batch["x_chosen"], batch["x_rejected"]], dim=0).to(device)
    y = torch.cat([batch["y_chosen"], batch["y_rejected"]], dim=0).to(device)
    mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]], dim=0).to(device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            ref_log_probs = logits_to_log_probs(reference(x).logits, y)
        policy_output = policy(x)
        policy_log_probs = logits_to_log_probs(policy_output.logits, y)
        dpo = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.15)
        loss = dpo + policy_output.aux_loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    phases["dpo"] = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "sequence_length": int(x.shape[-1]),
        "simultaneous_models": 2,
    }
    del policy, reference, optimizer, batch, x, y, mask, ref_log_probs
    del policy_output, policy_log_probs, dpo, loss
    gc.collect()
    torch.cuda.empty_cache()

    maximum = float(gates["maximum_peak_memory_gib"])
    for name, row in phases.items():
        peak_gib = row["peak_reserved_bytes"] / 1024**3
        if not row["loss_finite"]:
            raise RuntimeError(f"{name.upper()} 探测损失不是有限值")
        if peak_gib > maximum:
            raise RuntimeError(f"{name.upper()} 探测峰值 {peak_gib:.2f} GiB 超过 {maximum:.2f} GiB 门槛")

    record = {
        "schema_version": "minimind-memory-admission/v1",
        "run_label": config["run_label"],
        "source_revision": prepared["source_revision"],
        "base_weight_sha256": prepared["base_weight_sha256"],
        "data_sha256": prepared["data_sha256"],
        "device": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(total_bytes),
        "free_memory_before_probe_bytes": int(free_bytes),
        "gates": dict(gates),
        "phases": phases,
        "status": "passed",
    }
    path = Path(project_root) / config["outputs"]["directory"] / config["run_label"] / "memory_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"显存探测记录已存在且不同，请更换 RUN_LABEL：{path}")
    path.write_text(payload, encoding="utf-8")
    return record


def _train_command(source_root: Path, config: Mapping[str, Any], stage: str, data_path: str) -> list[str]:
    if stage not in ("sft", "sft_lora", "dpo"):
        raise ValueError(f"未知训练阶段：{stage}")
    workload, model = config["workload"], config["model"]
    label = config["run_label"]
    if stage == "sft":
        script = "train_full_sft.py"
        args = ["--from_weight", "pretrain", "--save_weight", f"sft_{label}", "--data_path", data_path,
                "--epochs", str(workload["sft_epochs"]), "--batch_size", str(workload["batch_size"]),
                "--accumulation_steps", str(workload["gradient_accumulation_steps"]),
                "--seed", str(workload["seed"]), "--learning_rate", str(workload["sft_learning_rate"])]
    elif stage == "sft_lora":
        script = "train_lora.py"
        args = ["--from_weight", "pretrain", "--lora_name", f"lora_{label}", "--data_path", data_path,
                "--epochs", str(workload["sft_epochs"]), "--batch_size", str(workload["batch_size"]),
                "--accumulation_steps", str(workload["gradient_accumulation_steps"]),
                "--learning_rate", str(workload["sft_learning_rate"])]
    else:
        script = "train_dpo.py"
        args = ["--from_weight", f"sft_{label}", "--save_weight", f"dpo_{label}", "--data_path", data_path,
                "--epochs", str(workload["dpo_epochs"]), "--batch_size", str(workload["batch_size"]),
                "--accumulation_steps", str(workload["gradient_accumulation_steps"])]
    args += ["--hidden_size", str(model["hidden_size"]), "--num_hidden_layers", str(model["num_hidden_layers"]),
             "--use_moe", "0", "--max_seq_len", str(workload["max_seq_len"]), "--dtype", "bfloat16",
             "--num_workers", "0", "--log_interval", "1", "--save_interval", "1000000"]
    return [sys.executable, str(source_root / "trainer" / script), *args]


def _evaluate_checkpoints(config: Mapping[str, Any], source_root: Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    import random
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    sys.path.insert(0, str(source_root))
    from dataset.lm_dataset import DPODataset, SFTDataset
    from model.model_lora import apply_lora, load_lora
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(source_root / "model"))
    model_cfg = MiniMindConfig(
        hidden_size=int(config["model"]["hidden_size"]),
        num_hidden_layers=int(config["model"]["num_hidden_layers"]),
        use_moe=False,
    )
    out_dir = source_root / "out"
    weights = {
        "base": config["model"]["weight_prefix"],
        "sft": f"sft_{config['run_label']}",
        "dpo": f"dpo_{config['run_label']}",
    }
    evaluation: dict[str, Any] = {}
    for name in ("base", "sft", "sft_lora", "dpo"):
        prefix = f"lora_{config['run_label']}" if name == "sft_lora" else weights[name]
        model_path = out_dir / f"{prefix}_{config['model']['hidden_size']}.pth"
        if not model_path.is_file():
            raise FileNotFoundError(f"缺少 {name} 权重：{model_path}")
        model = MiniMindForCausalLM(model_cfg).to(device)
        if name == "sft_lora":
            base_path = out_dir / f"{config['model']['weight_prefix']}_{config['model']['hidden_size']}.pth"
            model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True), strict=True)
            apply_lora(model, rank=MINIMIND_LORA_RANK)
            load_lora(model, str(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        # MiniMind may randomly add a system prompt; reset the seed so each checkpoint
        # receives the same tokenized held-out prompts.
        random.seed(int(config["workload"]["seed"]))
        sft_ds = SFTDataset(prepared["data_paths"]["eval_sft"], tokenizer, max_length=int(config["workload"]["max_seq_len"]))
        losses = []
        with torch.inference_mode():
            for input_ids, labels in DataLoader(sft_ds, batch_size=1, shuffle=False):
                output = model(input_ids.to(device), labels=labels.to(device))
                losses.append(float(output.loss.float().item()))
        dpo_ds = DPODataset(prepared["data_paths"]["eval_dpo"], tokenizer, max_length=int(config["workload"]["max_seq_len"]))
        wins = 0
        with torch.inference_mode():
            for row in DataLoader(dpo_ds, batch_size=1, shuffle=False):
                x = torch.cat([row["x_chosen"], row["x_rejected"]], dim=0).to(device)
                y = torch.cat([row["y_chosen"], row["y_rejected"]], dim=0).to(device)
                mask = torch.cat([row["mask_chosen"], row["mask_rejected"]], dim=0).to(device)
                log_probs = torch.log_softmax(model(x).logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
                scores = (log_probs * mask).sum(dim=-1)
                wins += int(scores[0] > scores[1])
        adapter_parameters = sum(
            parameter.numel() for parameter_name, parameter in model.named_parameters()
            if ".lora." in parameter_name
        ) if name == "sft_lora" else 0
        evaluation[name] = {
            "heldout_sft_loss_mean": sum(losses) / len(losses),
            "heldout_preference_win_rate": wins / len(dpo_ds),
            "evaluation_examples": len(sft_ds),
            "preference_pairs": len(dpo_ds),
            "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "adapter_parameter_count": adapter_parameters,
            "adapter_parameter_ratio": adapter_parameters / max(sum(parameter.numel() for parameter in model.parameters()), 1),
            "adapter_rank": MINIMIND_LORA_RANK if name == "sft_lora" else None,
            "artifact_bytes": model_path.stat().st_size,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "weight_path": str(model_path),
            "weight_sha256": sha256_file(model_path),
        }
        del model
        torch.cuda.empty_cache()
    return evaluation


def train_and_evaluate(config: Mapping[str, Any], project_root: str | Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    _check_prepared_inputs(config, project_root, prepared)
    run_dir = Path(project_root) / config["outputs"]["directory"] / config["run_label"]
    for name in ("raw_result.json", "sft.log", "sft_lora.log", "dpo.log"):
        if (run_dir / name).exists():
            raise FileExistsError(f"训练记录已存在，请更换 RUN_LABEL：{run_dir / name}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止训练")
    gates = config["gates"]
    probe_path = Path(project_root) / config["outputs"]["directory"] / config["run_label"] / "memory_probe.json"
    if not probe_path.is_file():
        raise FileNotFoundError(f"先运行 RUN_STAGE='probe'：{probe_path}")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if (
        probe.get("status") != "passed"
        or probe.get("run_label") != config["run_label"]
        or probe.get("gates") != dict(gates)
        or probe.get("source_revision") != prepared["source_revision"]
        or probe.get("base_weight_sha256") != prepared["base_weight_sha256"]
        or probe.get("data_sha256") != prepared["data_sha256"]
    ):
        raise RuntimeError("同标签显存准入探测未通过，停止完整训练")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes / 1024**3 < float(gates["minimum_free_memory_gib_before_load"]):
        raise RuntimeError(f"加载模型前可用显存不足：{free_bytes / 1024**3:.2f} GiB")
    project_root = Path(project_root)
    preflight = subprocess.run(
        [sys.executable, str(project_root / "tools/environment_preflight.py"), "--profile", "PT-SFT-CUDA"],
        check=False, capture_output=True, text=True,
    )
    if preflight.returncode:
        raise RuntimeError(f"PT-SFT-CUDA 环境预检失败：\n{preflight.stdout}\n{preflight.stderr}")
    source_root = Path(prepared["source_path"])
    for prefix in (f"sft_{config['run_label']}", f"lora_{config['run_label']}", f"dpo_{config['run_label']}"):
        artifact = source_root / "out" / f"{prefix}_{config['model']['hidden_size']}.pth"
        if artifact.exists():
            raise FileExistsError(f"训练产物已存在，不覆盖：{artifact}。请更换 RUN_LABEL。")
    logs = {}
    for stage, key in (("sft", "sft"), ("sft_lora", "sft"), ("dpo", "dpo")):
        log_path = project_root / config["outputs"]["directory"] / config["run_label"] / f"{stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _train_command(source_root, config, stage, prepared["data_paths"][key])
        started = time.perf_counter()
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=source_root / "trainer", stdout=log, stderr=subprocess.STDOUT, check=False)
        logs[stage] = {
            "path": str(log_path), "returncode": completed.returncode, "command": command,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if completed.returncode:
            raise RuntimeError(f"MiniMind {stage.upper()} 失败，查看日志：{log_path}")
    evaluation = _evaluate_checkpoints(config, source_root, prepared)
    memory_probe = json.loads(probe_path.read_text(encoding="utf-8"))
    raw = {
        "schema_version": RESULT_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": config["run_label"],
        "environment_profile": config["environment_profile"],
        "source": {"path": str(source_root), "revision": config["source"]["revision"]},
        "model": {**dict(config["model"]), "weight_sha256": prepared["base_weight_sha256"]},
        "workload": {**dict(config["workload"]), "data_counts": prepared["data_counts"], "data_sha256": prepared["data_sha256"], "data_paths": prepared["data_paths"], "split_disjoint": prepared["split_disjoint"]},
        "decision_thresholds": {"evaluation": dict(config["evaluation"]), "memory": dict(config["gates"])},
        "runtime": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "total_memory_bytes": int(total_bytes), "free_memory_before_load_bytes": int(free_bytes)},
        "logs": logs,
        "memory_admission_probe": memory_probe,
        "evaluation": evaluation,
    }
    raw_path = project_root / config["outputs"]["directory"] / config["run_label"] / "raw_result.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raise FileExistsError(f"原始结果已存在，请换 RUN_LABEL：{raw_path}")
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"raw_result_path": str(raw_path), "evaluation": evaluation}


def close_project(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    errors = validate_config(config)
    if errors:
        raise ValueError("项目配置错误：" + "; ".join(errors))
    root = Path(project_root)
    run_dir = root / config["outputs"]["directory"] / config["run_label"]
    raw_path = run_dir / "raw_result.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"先运行 RUN_STAGE='train'：{raw_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != RESULT_SCHEMA or raw.get("run_label") != config["run_label"]:
        raise ValueError("原始结果契约或 RUN_LABEL 不匹配")
    evidence_errors = validate_training_evidence(raw, config, root)
    if evidence_errors:
        raise RuntimeError("实验记录不完整，不能收口：" + "; ".join(evidence_errors))
    evaluation = raw["evaluation"]
    base, sft, dpo = evaluation["base"], evaluation["sft"], evaluation["dpo"]
    gates = config["evaluation"]
    peak_bytes = [row["peak_reserved_bytes"] for row in evaluation.values()]
    peak_bytes.extend(row["peak_reserved_bytes"] for row in raw["memory_admission_probe"]["phases"].values())
    memory_gib = max(peak_bytes) / 1024**3
    hard_failures = []
    tune_reasons = []
    if not raw["workload"].get("split_disjoint"):
        hard_failures.append("train_eval_overlap")
    if config["gates"].get("require_model_artifact_hash") and any(not row.get("weight_sha256") for row in evaluation.values()):
        hard_failures.append("missing_model_hash")
    if memory_gib > float(config["gates"]["maximum_peak_memory_gib"]):
        tune_reasons.append("peak_memory_over_budget")
    if sft["heldout_sft_loss_mean"] > base["heldout_sft_loss_mean"] * (1 + float(gates["maximum_sft_loss_regression_vs_base"])):
        tune_reasons.append("sft_heldout_loss_regression")
    if evaluation["sft_lora"]["heldout_sft_loss_mean"] > sft["heldout_sft_loss_mean"] * (1 + float(gates["maximum_lora_loss_regression_vs_full_sft"])):
        tune_reasons.append("lora_heldout_loss_regression_vs_full_sft")
    if dpo["heldout_preference_win_rate"] < float(gates["minimum_dpo_preference_win_rate"]):
        tune_reasons.append("dpo_preference_win_rate_below_gate")
    lora = evaluation["sft_lora"]
    full_sft_probe = raw["memory_admission_probe"]["phases"]["sft"]
    lora_probe = raw["memory_admission_probe"]["phases"]["sft_lora"]
    full_sft_seconds = float(raw["logs"]["sft"]["elapsed_seconds"])
    lora_sft_seconds = float(raw["logs"]["sft_lora"]["elapsed_seconds"])
    full_sft_loss = float(sft["heldout_sft_loss_mean"])
    lora_loss = float(lora["heldout_sft_loss_mean"])
    lora_comparison = {
        "heldout_loss_delta_vs_full_sft": lora_loss - full_sft_loss,
        "heldout_loss_relative_change_vs_full_sft": (lora_loss - full_sft_loss) / max(abs(full_sft_loss), 1e-12),
        "trainable_parameter_ratio_vs_full_sft": int(lora["adapter_parameter_count"]) / int(sft["total_parameter_count"]),
        "artifact_size_ratio_vs_full_sft": int(lora["artifact_bytes"]) / int(sft["artifact_bytes"]),
        "training_time_ratio_vs_full_sft": lora_sft_seconds / max(full_sft_seconds, 1e-12),
        "probe_peak_reserved_memory_ratio_vs_full_sft": int(lora_probe["peak_reserved_bytes"]) / int(full_sft_probe["peak_reserved_bytes"]),
    }
    decision = "reject" if hard_failures else "tune" if tune_reasons else "accept"
    report = {
        "schema_version": RESULT_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "run_label": config["run_label"],
        "raw_result": {"path": str(raw_path.relative_to(root)), "sha256": sha256_file(raw_path)},
        "candidate_summary": evaluation,
        "lora_vs_full_sft": lora_comparison,
        "memory_admission_probe": raw["memory_admission_probe"],
        "thresholds": {
            "evaluation": dict(config["evaluation"]),
            "memory": dict(config["gates"]),
        },
        "peak_memory_gib": memory_gib,
        "hard_failures": hard_failures,
        "tune_reasons": tune_reasons,
        "decision": decision,
        "decision_scope": "固定 MiniMind 64M 小规模合成算术任务；不代表通用能力或生产发布结论",
    }
    output = run_dir / "delivery_report.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"交付报告已存在且内容不同：{output}")
    output.write_text(payload, encoding="utf-8")
    return report
