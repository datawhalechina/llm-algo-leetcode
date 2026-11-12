"""Shared fact collection for post-training project notebooks.

The helpers collect training facts, verify saved inputs and publish replay-safe
records. Thresholds and selection rules remain visible in fixed contracts and
notebooks; readers and registration helpers call those supplied rules rather
than inventing an independent project decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def normalized_text(value: object) -> str:
    """Normalize one text field for deterministic identity checks."""

    return " ".join(str(value or "").strip().split())


def stable_text_hash(parts: Iterable[object]) -> str:
    """Return a stable SHA-256 identity for normalized text fields."""

    payload = "\u241f".join(normalized_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_sft_records(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    train_count: int,
) -> dict[str, Any]:
    """Collect SFT admission facts without applying project thresholds.

    Records use ``instruction``, ``input`` and ``output`` fields.  The stable
    input order defines the train/evaluation split so leakage checks are fully
    reproducible from the returned hashes and configuration.
    """

    if max_length < 8:
        raise ValueError("max_length 必须至少为 8")
    if train_count <= 0 or train_count >= len(records):
        raise ValueError("train_count 必须位于 0 与样本总数之间")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须提供 eos_token_id")

    pair_hashes: list[str] = []
    prompt_hashes: list[str] = []
    supervised_tokens: list[int] = []
    empty_instruction_count = 0
    empty_response_count = 0
    truncated_count = 0
    zero_supervision_count = 0

    for record in records:
        instruction = normalized_text(record.get("instruction"))
        context = normalized_text(record.get("input"))
        response = normalized_text(record.get("output"))
        empty_instruction_count += int(not instruction)
        empty_response_count += int(not response)

        prompt = f"### Instruction:\n{instruction}\n"
        if context:
            prompt += f"### Input:\n{context}\n"
        prompt += "### Response:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [tokenizer.eos_token_id]
        available = max(max_length - len(prompt_ids), 0)
        kept_response_tokens = min(len(response_ids), available)
        total_tokens = len(prompt_ids) + len(response_ids)

        pair_hashes.append(stable_text_hash((instruction, context, response)))
        prompt_hashes.append(stable_text_hash((instruction, context)))
        supervised_tokens.append(kept_response_tokens)
        truncated_count += int(total_tokens > max_length)
        zero_supervision_count += int(kept_response_tokens == 0)

    train_prompt_hashes = set(prompt_hashes[:train_count])
    evaluation_prompt_hashes = set(prompt_hashes[train_count:])
    duplicate_count = len(pair_hashes) - len(set(pair_hashes))
    split_leakage_count = len(train_prompt_hashes & evaluation_prompt_hashes)
    sample_count = len(records)

    return {
        "sample_count": sample_count,
        "train_count": train_count,
        "evaluation_count": sample_count - train_count,
        "empty_instruction_count": empty_instruction_count,
        "empty_response_count": empty_response_count,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_count / sample_count if sample_count else 0.0,
        "truncated_count": truncated_count,
        "truncation_ratio": truncated_count / sample_count if sample_count else 0.0,
        "zero_supervision_count": zero_supervision_count,
        "split_leakage_count": split_leakage_count,
        "minimum_supervised_tokens": min(supervised_tokens, default=0),
        "mean_supervised_tokens": (
            sum(supervised_tokens) / len(supervised_tokens) if supervised_tokens else 0.0
        ),
        "sample_ids_hash": stable_text_hash(pair_hashes),
        "train_prompt_ids_hash": stable_text_hash(prompt_hashes[:train_count]),
        "evaluation_prompt_ids_hash": stable_text_hash(prompt_hashes[train_count:]),
    }


def save_json_record(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Save one JSON object without silently replacing an existing run."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"结果已存在，请更换 RUN_LABEL：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def json_record_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize finite JSON deterministically for identity and publication."""
    return (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def publish_json_record(path: str | Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish a complete file; reuse identical bytes, reject conflicts.

    A same-directory temporary file is linked only after flush/fsync. A killed
    writer can leave a temporary file, but never a partially published record.
    Returns whether a new record was published.
    """
    output = Path(path)
    content = json_record_bytes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".pending-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(temporary, output)
            return True
        except FileExistsError:
            if output.read_bytes() != content:
                raise FileExistsError(f"同名资产内容冲突，请更换 RUN_LABEL：{output}")
            return False
    finally:
        temporary.unlink(missing_ok=True)


def validate_sft_admission_record(record, contract, run_label, evaluate):
    """Check a v2 SFT record against its fixed contract and notebook gate rules.

    This validates saved facts and identities, not the original remote dataset.
    ``evaluate`` remains visible in the notebook and recomputes all gate results.
    Historical records without the v2 identity require their original reader.
    """
    for key in ("schema_version", "semantic_id", "measurement_version",
                "environment_profile", "rule_version", "sequence_protocol", "limits"):
        if record[key] != contract[key]:
            raise ValueError(f"合同字段不匹配：{key}")
    expected_digest = hashlib.sha256(json_record_bytes(contract)).hexdigest()
    if record["contract_sha256"] != expected_digest or record["run_label"] != run_label:
        raise ValueError("合同哈希或运行标签不匹配")
    if record["run_mode"] != "real_data":
        raise ValueError("证据必须来自真实数据审计")
    for key in ("dataset", "tokenizer"):
        if any(record[key][field] != contract[key][field]
               for field in ("id", "requested_revision")):
            raise ValueError(f"来源身份不匹配：{key}")
        if not re.fullmatch(r"[0-9a-f]{40}", record[key]["resolved_commit"]):
            raise ValueError(f"来源提交不是不可变哈希：{key}")
    if record["source_revision"] != record["dataset"]["resolved_commit"]:
        raise ValueError("来源血缘与数据提交不匹配")
    if record["workload"] != contract["workload"]:
        raise ValueError("采集工作负载不匹配")
    for audit_key, admission_key in (("raw_audit", "raw_admission"), ("audit", "admission")):
        audit = record[audit_key]
        count_keys = ("sample_count", "train_count", "evaluation_count",
                      "empty_instruction_count", "empty_response_count", "duplicate_count",
                      "truncated_count", "zero_supervision_count", "split_leakage_count",
                      "minimum_supervised_tokens")
        if any(type(audit[key]) is not int or audit[key] < 0 for key in count_keys):
            raise ValueError("审计计数必须为非负整数")
        total = audit["sample_count"]
        if total <= 1 or not 0 < audit["train_count"] < total:
            raise ValueError("训练/评测切分为空")
        if audit["train_count"] + audit["evaluation_count"] != total:
            raise ValueError("切分计数不守恒")
        for key in ("sample_ids_hash", "train_prompt_ids_hash", "evaluation_prompt_ids_hash"):
            if not isinstance(audit[key], str) or not re.fullmatch(r"[0-9a-f]{64}", audit[key]):
                raise ValueError("审计身份哈希不合法")
        for ratio_key, count_key in (("duplicate_ratio", "duplicate_count"),
                                     ("truncation_ratio", "truncated_count")):
            ratio = audit[ratio_key]
            if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError("比例必须有限且位于[0,1]")
            if not math.isclose(ratio, audit[count_key] / total, abs_tol=1e-12):
                raise ValueError("比例与计数不匹配")
        for key in count_keys[3:-1]:
            if audit[key] > total:
                raise ValueError("风险计数超过样本总数")
        mean = audit["mean_supervised_tokens"]
        if (type(mean) not in (int, float) or not math.isfinite(mean)
                or not 0 <= audit["minimum_supervised_tokens"] <= mean <= contract["workload"]["max_length"]):
            raise ValueError("监督词元统计不合法")
        if (audit["minimum_supervised_tokens"] == 0) != (audit["zero_supervision_count"] > 0):
            raise ValueError("最小监督数与零监督计数不一致")
        if evaluate(audit, contract["limits"]) != record[admission_key]:
            raise ValueError("准入状态或失败项无法由审计事实复算")
    audit, raw = record["audit"], record["raw_audit"]
    if raw["sample_count"] != contract["workload"]["sample_count"] or raw["train_count"] != contract["workload"]["train_count"]:
        raise ValueError("原始采集规模不匹配")
    removed = raw["sample_count"] - audit["sample_count"]
    cleaning = record["cleaning"]
    if (removed < 0 or type(cleaning["removed_count"]) is not int
            or cleaning["removed_count"] != removed
            or cleaning["rule"] != "drop_empty_instruction_response_and_exact_duplicates"
            or type(cleaning["retained_ratio"]) not in (int, float)
            or cleaning["retained_ratio"] != audit["sample_count"] / raw["sample_count"]):
        raise ValueError("清洗计数或规则不匹配")
    if audit["train_count"] != min(contract["workload"]["train_count"],
                                    audit["sample_count"] - contract["limits"]["minimum_evaluation_count"]):
        raise ValueError("清洗后的切分与固定规则不匹配")
    snapshot = record["snapshot"]
    all_indices, all_ids = [], []
    for split, count in (("train", audit["train_count"]), ("evaluation", audit["evaluation_count"])):
        indices, ids = snapshot[f"{split}_record_indices"], snapshot[f"{split}_sample_ids"]
        if (not isinstance(indices, list) or not isinstance(ids, list)
                or len(indices) != count or len(ids) != count
                or any(type(i) is not int or not 0 <= i < raw["sample_count"] for i in indices)
                or any(not isinstance(i, str) or not re.fullmatch(r"[0-9a-f]{64}", i) for i in ids)
                or stable_text_hash(ids) != snapshot[f"{split}_sample_ids_hash"]):
            raise ValueError(f"切分成员或哈希不匹配：{split}")
        all_indices.extend(indices)
        all_ids.extend(ids)
    if len(set(all_indices)) != len(all_indices) or len(set(all_ids)) != len(all_ids):
        raise ValueError("快照重复使用源索引或样本身份")
    if stable_text_hash(all_ids) != audit["sample_ids_hash"]:
        raise ValueError("完整快照哈希不匹配")
    if record["decision"] != record["admission"]["decision"] or record["decision_details"] != record["admission"]:
        raise ValueError("顶层决策不一致")
    if record["quality"]["independent_evaluation"] is not record["admission"]["hard_gate_passed"]:
        raise ValueError("评测准入标记不一致")
    return record


def read_sft_admission_record(path, contract, run_label, evaluate):
    """Independently read and validate evidence without mutating it."""
    return validate_sft_admission_record(
        json.loads(Path(path).read_text(encoding="utf-8")), contract, run_label, evaluate,
    )


def close_sft_admission(root, result_path, manifest_path, receipt_path, contract, run_label, evaluate):
    """Publish an accept-only handoff and a recoverable registration receipt."""
    root = Path(root).resolve()
    result_path, manifest_path, receipt_path = map(Path, (result_path, manifest_path, receipt_path))
    relative_paths = [str(path.resolve().relative_to(root)) for path in (result_path, manifest_path, receipt_path)]
    if len(set(relative_paths)) != 3:
        raise ValueError("证据、清单与回执必须使用不同路径")
    content = result_path.read_bytes()
    record = validate_sft_admission_record(json.loads(content), contract, run_label, evaluate)
    source_hash = hashlib.sha256(content).hexdigest()
    publications = []
    files = [{"path": relative_paths[0], "sha256": source_hash}]
    if record["decision"] == "accept":
        manifest = {key: record[key] for key in (
            "schema_version", "semantic_id", "measurement_version", "contract_sha256",
            "environment_profile", "run_mode", "run_label", "decision", "decision_details",
            "source_revision", "rule_version", "sequence_protocol", "dataset", "tokenizer",
            "workload", "snapshot", "audit", "admission",
        )}
        manifest.update({"artifact_role": "training.sft_data_quality",
                         "source_evidence": relative_paths[0], "source_evidence_sha256": source_hash,
                         "sample_ids_hash": record["audit"]["sample_ids_hash"],
                         "next_consumer": ["PT-LORA-VARIANTS", "PT-QLORA-SELECTION"]})
        publications.append((manifest_path, manifest))
        files.append({"path": relative_paths[1], "sha256": hashlib.sha256(json_record_bytes(manifest)).hexdigest()})
    elif manifest_path.exists():
        raise FileExistsError("非accept证据存在同名交接清单，请检查标签")
    receipt = {"schema_version": "project-registration/v1", "semantic_id": record["semantic_id"],
               "run_label": run_label, "evidence_schema": record["schema_version"],
               "measurement_version": record["measurement_version"], "decision": record["decision"],
               "failed_checks": record["admission"]["hard_failures"] + record["admission"]["soft_failures"],
               "handoff": relative_paths[1] if record["decision"] == "accept" else None,
               "files": files}
    publications.append((receipt_path, receipt))
    for path, payload in publications:
        if path.exists() and path.read_bytes() != json_record_bytes(payload):
            raise FileExistsError(f"已有资产内容冲突，请更换 RUN_LABEL：{path}")
    if result_path.read_bytes() != content:
        raise RuntimeError("核对期间原始证据发生变化")
    changed = False
    for path, payload in publications:
        changed = publish_json_record(path, payload) or changed
    for item in files:
        if hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("登记文件哈希不匹配")
    return {"status": "completed" if changed else "already_complete", "registration_receipt": receipt}


def read_lora_data_manifest(root, contract):
    """Read the single pinned SFT input without downloads or output writes."""
    root = Path(root).resolve()
    path = (root / contract["upstream_manifest"]).resolve()
    path.relative_to(root)
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract["upstream_manifest_sha256"]:
        raise ValueError("上游数据清单SHA256不匹配")
    upstream = json.loads(path.read_text(encoding="utf-8"))
    if (upstream["schema_version"] != "fine-tuning-project/v1"
            or upstream["semantic_id"] != "PT-SFT-DATA-QUALITY"
            or upstream["decision"] != "accept"):
        raise ValueError("上游数据没有通过指定项目的准入")
    if (upstream["tokenizer"]["id"] != contract["model"]["id"]
            or upstream["tokenizer"]["resolved_commit"] != contract["model"]["resolved_commit"]
            or upstream["workload"]["max_length"] != contract["workload"]["max_length"]):
        raise ValueError("分词器、模型提交或最大长度与数据审计不一致")
    indices, ids = [], []
    for split in ("train", "evaluation"):
        snapshot = upstream["snapshot"]
        part_indices, part_ids = snapshot[f"{split}_record_indices"], snapshot[f"{split}_sample_ids"]
        if (len(part_indices) != len(part_ids)
                or len(part_indices) < contract["workload"][f"{split}_examples"]
                or any(type(i) is not int or not 0 <= i < upstream["workload"]["sample_count"] for i in part_indices)
                or stable_text_hash(part_ids) != snapshot[f"{split}_sample_ids_hash"]):
            raise ValueError("上游样本成员、数量或哈希不匹配")
        indices.extend(part_indices)
        ids.extend(part_ids)
    if (len(set(indices)) != len(indices) or len(set(ids)) != len(ids)
            or stable_text_hash(ids) != upstream["audit"]["sample_ids_hash"]):
        raise ValueError("上游切分有重复成员或完整哈希不匹配")
    return upstream


def read_lora_selection(root, path, contract, run_label, select):
    """Validate saved scan identity, full matrix and per-seed adapter bindings."""
    root = Path(root).resolve()
    upstream = read_lora_data_manifest(root, contract)
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("schema_version", "semantic_id", "measurement_version", "environment_profile",
                "evaluation_reduction", "sequence_protocol", "model", "workload", "gates"):
        if record[key] != contract[key]:
            raise ValueError(f"扫描结果与固定合同不匹配：{key}")
    digest = hashlib.sha256(json_record_bytes(contract)).hexdigest()
    if (record["run_label"] != run_label or record["run_mode"] != "real_gpu"
            or record["contract_sha256"] != digest
            or record["upstream_manifest"] != contract["upstream_manifest"]
            or record["upstream_manifest_sha256"] != contract["upstream_manifest_sha256"]
            or record["upstream_sample_ids_hash"] != upstream["audit"]["sample_ids_hash"]):
        raise ValueError("运行标签、合同哈希或来源不匹配")
    configs = {item["name"]: item for item in contract["candidates"]}
    seeds = contract["workload"]["run_seeds"]
    expected = {(name, seed) for name in configs for seed in seeds}
    pairs = [(item["name"], item["run_seed"]) for item in record["candidates"]]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError("候选与种子的扫描矩阵不完整或有重复")
    for item in record["candidates"]:
        if item["config"] != configs[item["name"]] or item["seed"] != item["run_seed"]:
            raise ValueError("候选配置或种子身份不匹配")
        artifact = item.get("artifact")
        if item["run_seed"] == seeds[0]:
            expected_path = f"benchmarks/results/63_lora_variants/adapters_{run_label}/{item['name']}"
            if not artifact or artifact["path"] != expected_path:
                raise ValueError("首种子适配器路径不匹配")
            if hash_directory(root / expected_path) != artifact["sha256"]:
                raise ValueError("适配器文件与采集时哈希不匹配")
        elif artifact is not None:
            raise ValueError("未声明保存产物的重复运行不能绑定适配器")
    decision = select(record["candidates"], contract["gates"], list(configs), seeds)
    return record, decision


def close_lora_selection(root, raw_path, report_path, handoff_path, receipt_path,
                         contract, run_label, select):
    """Publish a deterministic report, accept-only handoff and hashed receipt."""
    root = Path(root).resolve()
    paths = [Path(p).resolve() for p in (raw_path, report_path, handoff_path, receipt_path)]
    relative = [p.relative_to(root).as_posix() for p in paths]
    if len(set(relative)) != 4:
        raise ValueError("原始证据、报告、交接和回执路径必须不同")
    record, choice = read_lora_selection(root, paths[0], contract, run_label, select)
    raw_hash = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    report = {"schema_version": record["schema_version"], "semantic_id": record["semantic_id"],
              "measurement_version": record["measurement_version"], "run_label": run_label,
              "contract_sha256": record["contract_sha256"],
              "decision": choice["decision"], "decision_details": choice,
              "source_evidence": relative[0], "source_evidence_sha256": raw_hash}
    publications = [(paths[1], report)]
    if choice["decision"] == "accept":
        name = choice["selected"]["name"]
        selected = next(item for item in record["candidates"]
                        if item["name"] == name and item["run_seed"] == contract["workload"]["run_seeds"][0])
        handoff = {**report, "selected_config": selected["config"], "model": record["model"],
                   "workload": record["workload"], "repeat_seeds": contract["workload"]["run_seeds"],
                   "upstream_manifest": record["upstream_manifest"],
                   "upstream_manifest_sha256": record["upstream_manifest_sha256"],
                   "selection_evidence": relative[0], "candidates": record["candidates"],
                   "repeat_results": [row for row in record["candidates"] if row["name"] == name],
                   "artifact": {"role": "training.lora_selection", "adapter_path": selected["artifact"]["path"],
                                "sha256": selected["artifact"]["sha256"], "seed": selected["run_seed"]},
                   "next_consumer": "PT-SFT-DELIVERY"}
        publications.append((paths[2], handoff))
    elif paths[2].exists():
        raise FileExistsError("未接受的扫描存在同名交接文件")
    files = [{"path": relative[0], "sha256": raw_hash},
             {"path": contract["upstream_manifest"], "sha256": contract["upstream_manifest_sha256"]}]
    files.extend({"path": p.relative_to(root).as_posix(),
                  "sha256": hashlib.sha256(json_record_bytes(data)).hexdigest()} for p, data in publications)
    receipt = {"schema_version": "project-registration/v1", "semantic_id": record["semantic_id"],
               "run_label": run_label, "measurement_version": record["measurement_version"],
               "evidence_schema": record["schema_version"], "decision": choice["decision"],
               "handoff": relative[2] if choice["decision"] == "accept" else None, "files": files,
               "artifacts": [row["artifact"] for row in record["candidates"] if row.get("artifact")]}
    publications.append((paths[3], receipt))
    for p, data in publications:
        if p.exists() and p.read_bytes() != json_record_bytes(data):
            raise FileExistsError(f"同名资产冲突，请更换标签：{p}")
    if hashlib.sha256(paths[0].read_bytes()).hexdigest() != raw_hash:
        raise ValueError("核对期间原始证据发生变化")
    changed = False
    for p, data in publications:
        changed = publish_json_record(p, data) or changed
    for item in files:
        if hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("登记文件SHA256不匹配")
    for artifact in receipt["artifacts"]:
        if hash_directory(root / artifact["path"]) != artifact["sha256"]:
            raise ValueError("登记适配器SHA256不匹配")
    return {"status": "completed" if changed else "already_complete", "registration_receipt": receipt}


def read_lora_delivery_inputs(root, path, contract):
    """Resolve and verify a v3 selection handoff before delivery training.

    The caller still applies its visible selection policy. This reader checks
    the saved scan, selected rows, adapter and data snapshot without training.
    """
    root = Path(root).resolve()

    def contained_path(value):
        candidate = (root / value).resolve()
        candidate.relative_to(root)
        return candidate

    manifest_path = contained_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("measurement_version") != "lora-selection/v3"
            or manifest.get("semantic_id") != "PT-LORA-VARIANTS"
            or manifest.get("decision") != "accept"
            or manifest.get("next_consumer") != "PT-SFT-DELIVERY"):
        raise ValueError("需要通过选型的 v3 LoRA 交接清单")
    scan_path = contained_path(manifest["selection_evidence"])
    scan_bytes = scan_path.read_bytes()
    if (manifest["source_evidence"] != manifest["selection_evidence"]
            or hashlib.sha256(scan_bytes).hexdigest() != manifest["source_evidence_sha256"]):
        raise ValueError("选型扫描路径或SHA256不匹配")
    scan = json.loads(scan_bytes)
    for key in ("model", "workload", "candidates", "upstream_manifest", "upstream_manifest_sha256",
                "run_label", "measurement_version", "contract_sha256", "schema_version", "semantic_id"):
        if manifest[key] != scan[key]:
            raise ValueError(f"交接清单与扫描记录不匹配：{key}")
    if (manifest["contract_sha256"] != hashlib.sha256(json_record_bytes(contract)).hexdigest()
            or scan["run_mode"] != "real_gpu"):
        raise ValueError("选型合同哈希或运行模式不匹配")
    for key in ("model", "workload", "gates", "measurement_version", "schema_version", "semantic_id",
                "environment_profile", "evaluation_reduction", "sequence_protocol",
                "upstream_manifest", "upstream_manifest_sha256"):
        if scan[key] != contract[key]:
            raise ValueError(f"选型扫描与合同不匹配：{key}")
    upstream = read_lora_data_manifest(root, contract)
    if scan["upstream_sample_ids_hash"] != upstream["audit"]["sample_ids_hash"]:
        raise ValueError("选型扫描的数据成员哈希不匹配")
    configs = {item["name"]: item for item in contract["candidates"]}
    seeds = contract["workload"]["run_seeds"]
    rows = scan["candidates"]
    pairs = [(row["name"], row["run_seed"]) for row in rows]
    if (len(pairs) != len(set(pairs))
            or set(pairs) != {(name, seed) for name in configs for seed in seeds}
            or any(row["config"] != configs[row["name"]] or row["seed"] != row["run_seed"] for row in rows)):
        raise ValueError("选型扫描的配置或重复矩阵不匹配")
    selected = manifest["selected_config"]
    selected_rows = [row for row in rows if row["name"] == selected["name"]]
    if (selected != configs.get(selected["name"])
            or manifest["repeat_seeds"] != seeds
            or manifest["repeat_results"] != selected_rows):
        raise ValueError("选中配置或重复结果与扫描不匹配")
    first = next(row for row in selected_rows if row["run_seed"] == seeds[0])
    artifact = manifest["artifact"]
    expected_path = f"benchmarks/results/63_lora_variants/adapters_{manifest['run_label']}/{selected['name']}"
    if (first["artifact"] != {"path": expected_path, "sha256": artifact["sha256"]}
            or artifact != {"role": "training.lora_selection", "adapter_path": expected_path,
                            "sha256": first["artifact"]["sha256"], "seed": seeds[0]}
            or hash_directory(contained_path(expected_path)) != artifact["sha256"]):
        raise ValueError("选中适配器路径、种子或SHA256不匹配")
    return manifest, scan, upstream


def build_causal_sft_examples(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, list[int]]]:
    """Build fixed-length causal-LM examples with prompt labels masked."""

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer 必须提供 pad_token_id 或 eos_token_id")

    examples: list[dict[str, list[int]]] = []
    for record in records:
        instruction = normalized_text(record.get("instruction"))
        context = normalized_text(record.get("input"))
        response = normalized_text(record.get("output"))
        prompt = f"### Instruction:\n{instruction}\n"
        if context:
            prompt += f"### Input:\n{context}\n"
        prompt += "### Response:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[:max_length]
        supervised = max(len(input_ids) - min(len(prompt_ids), max_length), 0)
        if supervised == 0:
            raise ValueError("样本在当前 max_length 下没有有效监督 Token")
        labels = [-100] * (len(input_ids) - supervised) + input_ids[-supervised:]
        attention_mask = [1] * len(input_ids)
        padding = max_length - len(input_ids)
        examples.append(
            {
                "input_ids": input_ids + [pad_token_id] * padding,
                "attention_mask": attention_mask + [0] * padding,
                "labels": labels + [-100] * padding,
            }
        )
    return examples


def run_lora_candidate(
    *,
    model_id: str,
    revision: str,
    tokenizer: Any,
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    max_length: int,
    update_steps: int,
    learning_rate: float,
    seed: int,
    dtype_name: str = "bfloat16",
    save_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run one real LoRA training candidate and return project-level evidence.

    The function reloads the base model for every candidate so peak memory,
    initial loss and optimizer state are comparable.  Candidate selection stays
    in the notebook and is never hidden in this runtime helper.
    """

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("真实 LoRA 候选实验需要 CUDA")
    if update_steps <= 0:
        raise ValueError("update_steps 必须大于 0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16，请改用 float16")

    train_examples = build_causal_sft_examples(
        train_records, tokenizer, max_length=max_length
    )
    evaluation_examples = build_causal_sft_examples(
        evaluation_records, tokenizer, max_length=max_length
    )
    device = torch.device("cuda")

    def tensor_batch(example: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        return {
            key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
            for key, value in example.items()
        }

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(candidate["rank"]),
        lora_alpha=int(candidate["alpha"]),
        lora_dropout=float(candidate["dropout"]),
        target_modules=list(candidate["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for example in evaluation_examples:
                losses.append(float(model(**tensor_batch(example)).loss.detach().float().item()))
        return statistics.fmean(losses)

    initial_eval_loss = evaluate()
    torch.cuda.reset_peak_memory_stats()
    step_times_ms: list[float] = []
    train_losses: list[float] = []
    supervised_tokens = 0
    model.train()
    for step in range(update_steps):
        example = train_examples[step % len(train_examples)]
        batch = tensor_batch(example)
        supervised_tokens += sum(token != -100 for token in example["labels"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - started) * 1000.0)
        train_losses.append(float(output.loss.detach().float().item()))

    final_eval_loss = evaluate()
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    elapsed_seconds = sum(step_times_ms) / 1000.0
    if save_directory is not None:
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(output_dir)

    return {
        "name": str(candidate["name"]),
        "config": dict(candidate),
        "seed": seed,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "eval_loss_improvement": initial_eval_loss - final_eval_loss,
        "initial_train_loss": train_losses[0],
        "final_train_loss": train_losses[-1],
        "median_step_time_ms": statistics.median(step_times_ms),
        "p95_step_time_ms": sorted(step_times_ms)[max(0, int(0.95 * len(step_times_ms)) - 1)],
        "supervised_tokens_per_second": supervised_tokens / elapsed_seconds,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "status": "ok",
    }


def hash_directory(directory: str | Path) -> str:
    """Hash relative names and bytes for every file in one artifact directory."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def causal_supervised_token_count(labels):
    """Count labels actually used by a causal-LM loss, excluding shifted position 0."""
    return int(labels[..., 1:].ne(-100).sum().item())


def read_qlora_selection(root, path, contract, run_label, summarize, compare):
    """Verify a paired QLoRA run, real token counts and local adapter bytes."""
    root = Path(root).resolve()
    data = read_lora_data_manifest(root, contract)
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("schema_version", "semantic_id", "measurement_version", "environment_profile",
                "evaluation_reduction", "timing_scope", "sequence_protocol", "model", "workload",
                "adapter", "preparation", "capacity_probe", "gates", "upstream_manifest", "upstream_manifest_sha256"):
        if record[key] != contract[key]:
            raise ValueError(f"QLoRA记录与合同不匹配：{key}")
    if (record["run_label"] != run_label or record["run_mode"] != "real_gpu"
            or record["contract_sha256"] != hashlib.sha256(json_record_bytes(contract)).hexdigest()
            or record["data_sample_ids_hash"] != data["audit"]["sample_ids_hash"]):
        raise ValueError("QLoRA标签、来源或合同哈希不匹配")
    json_record_bytes(record)
    if record.get("status") == "failed":
        if not record.get("failure"):
            raise ValueError("失败记录缺少原因")
        return record, {"decision": "tune", "selection_scope": "none", "failed_checks": ["execution_failed"]}
    if record.get("status") != "ok":
        raise ValueError("QLoRA执行状态不完整")
    seeds = contract["workload"]["seeds"]
    groups = record["paired_reports"]
    if set(groups) != {"lora_baseline", "qlora_candidate"}:
        raise ValueError("必须有唯一baseline与candidate")
    def check_member(row, label, micro_batch, steps, saved):
        if (row["label"] != label or row["model_id"] != contract["model"]["id"]
                or row["steps"] != steps or row["fixed_micro_batch"] != micro_batch
                or row["train_examples"] != contract["workload"]["train_examples"]
                or row["evaluation_examples"] != contract["workload"]["evaluation_examples"]
                or row["low_bit_active"] is not label.startswith("qlora_candidate")
                or row["gradient_checkpointing"] is not False):
            raise ValueError("配对成员的模型、预算或执行状态不匹配")
        expected_preparation = "kbit_no_gradient_checkpointing" if label.startswith("qlora_candidate") else "bf16_default"
        if row["preparation"] != expected_preparation:
            raise ValueError("配对成员的训练准备策略不匹配")
        for key in ("elapsed_training_s", "step_time_s", "supervised_tokens_per_s", "peak_memory_mb", "val_loss"):
            if (type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                    or row[key] < 0 or (key != "val_loss" and row[key] == 0)):
                raise ValueError(f"非法QLoRA指标：{key}")
        tokens = row["supervised_tokens_seen"]
        if (type(tokens) is not int or not 0 < tokens <= micro_batch * steps * (contract["workload"]["max_length"] - 1)
                or not math.isclose(row["supervised_tokens_per_s"], tokens / row["elapsed_training_s"], rel_tol=1e-9)
                or not math.isclose(row["step_time_s"], row["elapsed_training_s"] / steps, rel_tol=1e-9)):
            raise ValueError("实际监督词元计数、耗时或吞吐不匹配")
        if saved:
            expected = (root / f"benchmarks/results/65_qlora_selection/{run_label}_{label}_seed{row['seed']}_adapter").resolve()
            expected.relative_to(root)
            if (root / row["artifact_path"]).resolve() != expected or hash_directory(expected) != row["artifact_hash"]:
                raise ValueError("QLoRA适配器路径或SHA256不匹配")
    for label, rows in groups.items():
        if [row["seed"] for row in rows] != seeds:
            raise ValueError("配对种子缺失、重复或错序")
        for row in rows:
            check_member(row, label, contract["workload"]["fixed_micro_batch"], contract["workload"]["steps"], True)
        expected_summary = summarize(rows)
        if record["baseline" if label == "lora_baseline" else "candidate"] != expected_summary:
            raise ValueError("QLoRA汇总与逐种子记录不匹配")
    probe = record["scale_probe"]
    if contract["capacity_probe"]["enabled"]:
        if probe["status"] != "completed" or len(probe["reports"]) != 2 or {r["label"] for r in probe["reports"]} != set(groups):
            raise ValueError("容量阶梯缺少两侧完整记录")
        for group in probe["reports"]:
            attempts = group["attempts"]
            sizes = [row["fixed_micro_batch"] for row in attempts]
            planned = contract["capacity_probe"]["micro_batches"]
            if not sizes or sizes != planned[:len(sizes)] or (len(sizes) != len(planned) and attempts[-1].get("status") != "oom"):
                raise ValueError("容量阶梯未按计划完成或记录OOM")
            successful = []
            for index, row in enumerate(attempts):
                if row.get("status") == "oom":
                    if index != len(attempts) - 1 or not row.get("error"):
                        raise ValueError("容量OOM必须是最后一次尝试且保留原因")
                else:
                    if row["seed"] != contract["capacity_probe"]["seed"]:
                        raise ValueError("容量探测种子不匹配")
                    check_member(row, f"{group['label']}_mb{sizes[index]}", sizes[index], contract["capacity_probe"]["steps"], False)
                    successful.append(sizes[index])
            if group["max_micro_batch"] != max(successful, default=0):
                raise ValueError("容量上限与逐档尝试不匹配")
    return record, compare(record["baseline"], record["candidate"], probe)


def close_qlora_selection(root, raw_path, report_path, manifest_path, receipt_path,
                          contract, run_label, summarize, compare, build_manifest):
    """Register a QLoRA decision without retraining or overwriting other evidence."""
    root = Path(root).resolve()
    paths = [Path(p).resolve() for p in (raw_path, report_path, manifest_path, receipt_path)]
    names = [p.relative_to(root).as_posix() for p in paths]
    if len(set(names)) != 4:
        raise ValueError("QLoRA原始记录、报告、来源清单和回执必须分开")
    digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    record, decision = read_qlora_selection(root, paths[0], contract, run_label, summarize, compare)
    if hashlib.sha256(paths[0].read_bytes()).hexdigest() != digest:
        raise ValueError("读取期间QLoRA记录发生变化")
    report = {**record, "comparison": decision, "decision": decision["decision"],
              "source_evidence": names[0], "source_evidence_sha256": digest}
    publications = [(paths[1], report)]
    if decision["decision"] == "accept":
        manifest = build_manifest(report)
        manifest.update(decision=decision["decision"], comparison=decision,
                        source_evidence=names[0], source_evidence_sha256=digest)
        publications.append((paths[2], manifest))
    elif paths[2].exists():
        raise FileExistsError("未通过选型的运行存在同名来源清单")
    files = [{"path": names[0], "sha256": digest},
             {"path": contract["upstream_manifest"], "sha256": contract["upstream_manifest_sha256"]}]
    files.extend({"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(json_record_bytes(payload)).hexdigest()}
                 for p, payload in publications)
    artifacts = [{"path": (root / row["artifact_path"]).resolve().relative_to(root).as_posix(), "sha256": row["artifact_hash"]}
                 for rows in record.get("paired_reports", {}).values() for row in rows] if record["status"] == "ok" else []
    receipt = {"schema_version": "project-registration/v1", "semantic_id": contract["semantic_id"],
               "measurement_version": contract["measurement_version"], "contract_sha256": record["contract_sha256"],
               "run_label": run_label, "decision": decision["decision"], "files": files, "artifacts": artifacts,
               "handoff": names[2] if decision["decision"] == "accept" else None}
    publications.append((paths[3], receipt))
    for path, payload in publications:
        if path.exists() and path.read_bytes() != json_record_bytes(payload):
            raise FileExistsError(f"同名QLoRA资产内容冲突：{path}")
    changed = False
    for path, payload in publications:
        changed = publish_json_record(path, payload) or changed
    for item in files:
        if hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("登记文件SHA256不匹配")
    for item in artifacts:
        if hash_directory(root / item["path"]) != item["sha256"]:
            raise ValueError("登记adapter SHA256不匹配")
    return {"status": "completed" if changed else "already_complete", "registration_receipt": receipt}


def read_sft_delivery_inputs(root, contract):
    """Read one pinned selection and its data lineage without remote access.

    Historical absolute references are matched against the pinned relative
    suffix, never opened outside the current repository. No adapter is loaded:
    delivery retrains the selected configuration from the base model.
    """
    root = Path(root).resolve()
    inputs = []
    for key in ("selection_manifest", "selection_evidence", "upstream_manifest"):
        path = (root / contract[key]).resolve()
        path.relative_to(root)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != contract[f"{key}_sha256"]:
            raise ValueError(f"输入文件SHA256不匹配：{key}")
        inputs.append(json.loads(content))
    selection, scan, data = inputs
    for key in ("selection_evidence", "upstream_manifest"):
        value = selection[key]
        if value != contract[key] and not (Path(value).is_absolute() and value.endswith("/" + contract[key])):
            raise ValueError(f"选型清单的来源引用不匹配：{key}")
    if (selection["schema_version"] != "fine-tuning-project/v1"
            or selection["semantic_id"] != "PT-LORA-VARIANTS" or selection["decision"] != "accept"
            or selection["next_consumer"] != "PT-SFT-DELIVERY"):
        raise ValueError("选型清单未通过指定项目")
    for record in (selection, scan):
        if any(record["model"][key] != value for key, value in contract["model"].items()):
            raise ValueError("选型模型与交付合同不匹配")
    if (selection["candidates"] != scan["candidates"]
            or selection["workload"] != scan["workload"]
            or scan["run_label"] != selection["run_label"]):
        raise ValueError("选型清单与原始扫描不匹配")
    if any(selection["workload"][key] != contract["workload"][key]
           for key in ("max_length", "train_examples", "evaluation_examples")):
        raise ValueError("交付样本范围与选型不匹配")
    # Reuse the data member/hash checks; the data snapshot remains the only
    # dataset input even when the selection itself uses a historical protocol.
    read_lora_data_manifest(root, contract)
    return selection, scan, data


def read_sft_delivery(root, path, contract, run_label, decide):
    """Validate saved delivery identity and local assets before a decision."""
    root = Path(root).resolve()
    selection, _, data = read_sft_delivery_inputs(root, contract)
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("schema_version", "semantic_id", "measurement_version", "environment_profile",
                "evaluation_reduction", "sequence_protocol", "model", "workload", "gates"):
        if record[key] != contract[key]:
            raise ValueError(f"交付记录与固定合同不匹配：{key}")
    if (record["contract_sha256"] != hashlib.sha256(json_record_bytes(contract)).hexdigest()
            or record["run_label"] != run_label or record["run_mode"] != "real_gpu"
            or record["selection_manifest"] != contract["selection_manifest"]
            or record["selection_manifest_sha256"] != contract["selection_manifest_sha256"]
            or record["selected_config"] != selection["selected_config"]
            or record["data"] != {"manifest": contract["upstream_manifest"],
                                  "sample_ids_hash": data["audit"]["sample_ids_hash"]}):
        raise ValueError("交付标签、合同或输入来源不匹配")
    evidence = record["evidence"]
    if evidence["measurement_version"] != contract["measurement_version"]:
        raise ValueError("恢复测量版本不匹配")
    checkpoint = evidence["checkpoint"]
    identity = checkpoint["training_identity"]
    workload = contract["workload"]
    expected_identity = {"model_id": contract["model"]["id"], "model_commit": contract["model"]["resolved_commit"],
                         "lora_config": selection["selected_config"], "train_example_count": workload["train_examples"],
                         **{key: workload[key] for key in ("max_length", "total_steps", "checkpoint_step",
                                                          "learning_rate", "seed", "dtype")},
                         "optimizer": "AdamW_default", "scheduler": "constant_lambda_1",
                         "sample_order": "sequential_step_mod_count"}
    for key in ("train_examples_sha256", "evaluation_examples_sha256"):
        if not isinstance(identity.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key]):
            raise ValueError("缺少实际分词后样本摘要")
        expected_identity[key] = identity[key]
    if identity != expected_identity:
        raise ValueError("检查点身份与交付合同不匹配")
    for item, prefix in ((checkpoint, "checkpoint"), (evidence["artifact"], "adapter")):
        expected = (root / f"benchmarks/results/62_sft_delivery/{prefix}_{run_label}").resolve()
        expected.relative_to(root)
        if (root / item["path"]).resolve() != expected or hash_directory(expected) != item["sha256"]:
            raise ValueError("交付产物路径或SHA256不匹配")
    progress = read_lora_checkpoint(root / checkpoint["path"], checkpoint["sha256"], expected_identity)
    if (checkpoint["saved_step"] != progress["global_step"]
            or checkpoint["next_sample_index"] != progress["next_sample_index"]
            or checkpoint["recovery_scope"] != contract["recovery_scope"]):
        raise ValueError("检查点进度与结果摘要不匹配")
    # Finite-only serialization rejects unhandled NaN/Inf anywhere in a record.
    json_record_bytes(record)
    return record, decide(evidence, contract["gates"], workload["total_steps"])


def close_sft_delivery(root, raw_path, report_path, manifest_path, receipt_path, contract, run_label, decide):
    """Publish report, accept-only baseline and receipt with replay-safe writes."""
    root = Path(root).resolve()
    paths = [Path(path).resolve() for path in (raw_path, report_path, manifest_path, receipt_path)]
    names = [path.relative_to(root).as_posix() for path in paths]
    if len(set(names)) != 4:
        raise ValueError("原始记录、报告、清单和回执路径必须不同")
    raw_hash = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    record, decision = read_sft_delivery(root, paths[0], contract, run_label, decide)
    if hashlib.sha256(paths[0].read_bytes()).hexdigest() != raw_hash:
        raise ValueError("读取期间原始训练记录发生变化")
    report = {"schema_version": record["schema_version"], "semantic_id": record["semantic_id"],
              "measurement_version": record["measurement_version"], "run_label": run_label,
              "contract_sha256": record["contract_sha256"], "decision": decision["decision"],
              "decision_details": decision, "source_evidence": names[0], "source_evidence_sha256": raw_hash}
    publications = [(paths[1], report)]
    if decision["decision"] == "accept":
        manifest = {**report, "model": record["model"], "data": record["data"],
                    "selected_config": record["selected_config"], "checkpoint": record["evidence"]["checkpoint"],
                    "artifacts": {"adapter": {"role": "training.sft_baseline", **record["evidence"]["artifact"]}},
                    "quality": {key: record["evidence"][key] for key in ("final_eval_loss", "reload_loss_abs_diff")},
                    "next_consumer": ["PT-DPO-OFFLINE", "PT-GRPO-GROUPWISE"]}
        publications.append((paths[2], manifest))
    elif paths[2].exists():
        raise FileExistsError("未通过交付的运行存在同名基线清单")
    files = [{"path": names[0], "sha256": raw_hash}]
    files.extend({"path": contract[key], "sha256": contract[f"{key}_sha256"]}
                 for key in ("selection_manifest", "selection_evidence", "upstream_manifest"))
    files.extend({"path": path.relative_to(root).as_posix(),
                  "sha256": hashlib.sha256(json_record_bytes(payload)).hexdigest()}
                 for path, payload in publications)
    artifacts = [{"path": (root / record["evidence"][key]["path"]).resolve().relative_to(root).as_posix(),
                  "sha256": record["evidence"][key]["sha256"]} for key in ("checkpoint", "artifact")]
    receipt = {"schema_version": "project-registration/v1", "semantic_id": record["semantic_id"],
               "run_label": run_label, "measurement_version": record["measurement_version"],
               "evidence_schema": record["schema_version"], "contract_sha256": record["contract_sha256"],
               "decision": decision["decision"], "handoff": names[2] if decision["decision"] == "accept" else None,
               "files": files, "artifacts": artifacts}
    publications.append((paths[3], receipt))
    for path, payload in publications:
        if path.exists() and path.read_bytes() != json_record_bytes(payload):
            raise FileExistsError(f"同名资产内容冲突：{path}")
    changed = False
    for path, payload in publications:
        changed = publish_json_record(path, payload) or changed
    for item in files:
        if hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("登记文件SHA256不匹配")
    for item in artifacts:
        if hash_directory(root / item["path"]) != item["sha256"]:
            raise ValueError("登记产物SHA256不匹配")
    return {"status": "completed" if changed else "already_complete", "registration_receipt": receipt}


def read_lora_checkpoint(directory, expected_sha256, expected_identity):
    """Check checkpoint bytes, training identity and sequential sample cursor.

    The expected hash is retained outside the checkpoint by the caller. This
    validates a same-process rebuild, not process-crash recovery or a signature.
    """
    directory = Path(directory)
    if hash_directory(directory) != expected_sha256:
        raise ValueError("检查点文件SHA256与保存后记录不一致")
    if not (directory / "adapter").is_dir() or not any((directory / "adapter").rglob("*")):
        raise ValueError("检查点缺少适配器")
    for name in ("optimizer.pt", "scheduler.pt", "rng.pt", "progress.json"):
        if not (directory / name).is_file():
            raise ValueError(f"检查点缺少状态文件：{name}")
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    if (progress.get("schema_version") != "lora-checkpoint/v1"
            or progress.get("training_identity") != expected_identity):
        raise ValueError("检查点模型、样本或训练配置不匹配")
    step, count = progress.get("global_step"), expected_identity.get("train_example_count")
    total = expected_identity.get("total_steps")
    if (type(step) is not int or type(total) is not int or type(count) is not int
            or count <= 0 or not 0 < step < total
            or step != expected_identity.get("checkpoint_step")
            or type(progress.get("next_sample_index")) is not int
            or progress["next_sample_index"] != step % count):
        raise ValueError("检查点步数或下一条样本位置不匹配")
    return progress


def run_lora_delivery(
    *,
    model_id: str,
    revision: str,
    tokenizer: Any,
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    lora_config: Mapping[str, Any],
    max_length: int,
    total_steps: int,
    checkpoint_step: int,
    learning_rate: float,
    seed: int,
    dtype_name: str,
    checkpoint_directory: str | Path,
    artifact_directory: str | Path,
) -> dict[str, Any]:
    """Train, checkpoint, resume, save and reload one selected LoRA adapter."""

    import gc

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    if not 0 < checkpoint_step < total_steps:
        raise ValueError("checkpoint_step 必须位于 0 与 total_steps 之间")
    checkpoint_root = Path(checkpoint_directory)
    artifact_root = Path(artifact_directory)
    if checkpoint_root.exists() or artifact_root.exists():
        raise FileExistsError("checkpoint 或 artifact 目录已存在，请修改 RUN_LABEL")
    checkpoint_root.mkdir(parents=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    train_examples = build_causal_sft_examples(train_records, tokenizer, max_length=max_length)
    evaluation_examples = build_causal_sft_examples(
        evaluation_records, tokenizer, max_length=max_length
    )
    if not train_examples or not evaluation_examples:
        raise ValueError("训练与评测样本均不能为空")
    training_identity = {
        "model_id": model_id, "model_commit": revision,
        "train_examples_sha256": hashlib.sha256(json_record_bytes({"examples": train_examples})).hexdigest(),
        "evaluation_examples_sha256": hashlib.sha256(json_record_bytes({"examples": evaluation_examples})).hexdigest(),
        "train_example_count": len(train_examples), "lora_config": dict(lora_config),
        "max_length": max_length, "total_steps": total_steps, "checkpoint_step": checkpoint_step,
        "learning_rate": learning_rate, "seed": seed, "dtype": dtype_name,
        "optimizer": "AdamW_default", "scheduler": "constant_lambda_1",
        "sample_order": "sequential_step_mod_count",
    }
    device = torch.device("cuda")

    def tensor_batch(example: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        return {
            key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
            for key, value in example.items()
        }

    def load_base():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        ).to(device)
        model.config.use_cache = False
        return model

    def evaluate(model: Any) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for example in evaluation_examples:
                losses.append(float(model(**tensor_batch(example)).loss.detach().float().item()))
        return statistics.fmean(losses)

    base = load_base()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(base, peft_config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    probe_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    initial_eval_loss = evaluate(model)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    times_ms: list[float] = []
    gradient_norms: list[float] = []
    parameter_delta_norms: list[float] = []
    learning_rates: list[float] = []
    finite_flags: list[bool] = []

    def train_range(start: int, stop: int) -> None:
        model.train()
        for step in range(start, stop):
            batch = tensor_batch(train_examples[step % len(train_examples)])
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            gradient_norm = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient_norm += float(parameter.grad.detach().float().pow(2).sum().item())
            gradient_norm = gradient_norm**0.5
            probe_before = probe_parameter.detach().float().reshape(-1)[:4096].cpu().clone()
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - started) * 1000.0)
            losses.append(float(output.loss.detach().float().item()))
            gradient_norms.append(gradient_norm)
            parameter_delta_norms.append(
                float(
                    (probe_parameter.detach().float().reshape(-1)[:4096].cpu() - probe_before)
                    .pow(2)
                    .sum()
                    .sqrt()
                    .item()
                )
            )
            finite_flags.append(
                bool(torch.isfinite(output.loss).item()) and math.isfinite(gradient_norm)
                and math.isfinite(parameter_delta_norms[-1])
                and all(bool(torch.isfinite(parameter).all().item())
                        for parameter in model.parameters() if parameter.requires_grad)
            )

    train_range(0, checkpoint_step)
    checkpoint_adapter = checkpoint_root / "adapter"
    model.save_pretrained(checkpoint_adapter)
    torch.save(optimizer.state_dict(), checkpoint_root / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_root / "scheduler.pt")
    torch.save(
        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)},
        checkpoint_root / "rng.pt",
    )
    (checkpoint_root / "progress.json").write_text(
        json.dumps({"schema_version": "lora-checkpoint/v1", "global_step": checkpoint_step,
                    "training_identity": training_identity,
                    "next_sample_index": checkpoint_step % len(train_examples)}, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_hash = hash_directory(checkpoint_root)

    del scheduler, optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()

    progress = read_lora_checkpoint(checkpoint_root, checkpoint_hash, training_identity)
    base = load_base()
    model = PeftModel.from_pretrained(base, checkpoint_adapter, is_trainable=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    optimizer.load_state_dict(
        torch.load(checkpoint_root / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(torch.load(checkpoint_root / "scheduler.pt", weights_only=True))
    rng_state = torch.load(checkpoint_root / "rng.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng_state["cpu"])
    torch.cuda.set_rng_state(rng_state["cuda"], device)
    probe_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    resumed_from_step = progress["global_step"]
    train_range(resumed_from_step, total_steps)
    final_eval_loss = evaluate(model)
    model.save_pretrained(artifact_root)
    tokenizer.save_pretrained(artifact_root)
    artifact_hash = hash_directory(artifact_root)
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

    del scheduler, optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base = load_base()
    reloaded = PeftModel.from_pretrained(reloaded_base, artifact_root)
    reloaded_eval_loss = evaluate(reloaded)

    return {
        "measurement_version": "sft-delivery-recovery/v4",
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "reloaded_eval_loss": reloaded_eval_loss,
        "reload_loss_abs_diff": abs(reloaded_eval_loss - final_eval_loss),
        "initial_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "median_step_time_ms": statistics.median(times_ms),
        "update_dynamics": {
            "finite_all": all(finite_flags),
            "gradient_norm_median": statistics.median(gradient_norms),
            "parameter_delta_norm_median": statistics.median(parameter_delta_norms),
            "learning_rate_start": learning_rates[0],
            "learning_rate_end": learning_rates[-1],
        },
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "checkpoint": {
            "path": str(checkpoint_root),
            "sha256": checkpoint_hash,
            "saved_step": checkpoint_step,
            "resumed_from_step": resumed_from_step,
            "completed_step": total_steps,
            "training_update_count": len(losses),
            "optimizer_state_present": True,
            "scheduler_state_present": True,
            "rng_state_present": True,
            "identity_verified": True,
            "files_sha256_verified": True,
            "sample_cursor_verified": True,
            "next_sample_index": progress["next_sample_index"],
            "training_identity": training_identity,
            "recovery_scope": "same_process_model_rebuild",
        },
        "artifact": {"path": str(artifact_root), "sha256": artifact_hash},
        "status": "ok",
    }


def preference_record_to_text(record: Mapping[str, Any]) -> tuple[str, str, str]:
    """Extract one shared prompt and two responses from a chat preference pair."""

    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not isinstance(chosen, Sequence) or not isinstance(rejected, Sequence):
        raise ValueError("chosen/rejected 必须是消息列表")
    if len(chosen) < 2 or len(rejected) < 2:
        raise ValueError("偏好对必须包含 prompt 与 response")
    chosen_prompt = normalized_text(chosen[0].get("content"))
    rejected_prompt = normalized_text(rejected[0].get("content"))
    if chosen_prompt != rejected_prompt:
        raise ValueError("chosen/rejected 的 prompt 不一致")
    chosen_response = normalized_text(chosen[-1].get("content"))
    rejected_response = normalized_text(rejected[-1].get("content"))
    if not chosen_prompt or not chosen_response or not rejected_response:
        raise ValueError("偏好对存在空字段")
    return chosen_prompt, chosen_response, rejected_response


def audit_preference_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect pair validity, identity and duplicate facts without thresholds."""

    valid_pairs: list[tuple[str, str, str]] = []
    invalid_count = 0
    identical_response_count = 0
    for record in records:
        try:
            pair = preference_record_to_text(record)
        except (TypeError, ValueError, AttributeError):
            invalid_count += 1
            continue
        identical_response_count += int(pair[1] == pair[2])
        valid_pairs.append(pair)
    pair_hashes = [stable_text_hash(pair) for pair in valid_pairs]
    prompt_hashes = [stable_text_hash((pair[0],)) for pair in valid_pairs]
    return {
        "sample_count": len(records),
        "valid_count": len(valid_pairs),
        "valid_ratio": len(valid_pairs) / len(records) if records else 0.0,
        "invalid_count": invalid_count,
        "identical_response_count": identical_response_count,
        "duplicate_count": len(pair_hashes) - len(set(pair_hashes)),
        "pair_ids_hash": stable_text_hash(pair_hashes),
        "prompt_ids_hash": stable_text_hash(prompt_hashes),
    }


def build_preference_pair_sequences(
    tokenizer: Any,
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Encode one preference pair while preserving a shared prompt boundary.

    Long prompts are truncated from the left.  Chosen and rejected use the
    same retained prompt, and each side keeps at least one response token.
    """

    if max_length < 2:
        raise ValueError("max_length 必须至少为 2")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须提供 eos_token_id")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]
    rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]
    if not prompt_ids or not chosen_ids or not rejected_ids:
        raise ValueError("偏好样本的 prompt、chosen 和 rejected 均必须产生 Token")

    maximum_response_tokens = max_length - 1
    chosen_kept = chosen_ids[:maximum_response_tokens]
    rejected_kept = rejected_ids[:maximum_response_tokens]
    shared_prompt_budget = max_length - max(len(chosen_kept), len(rejected_kept))
    shared_prompt = prompt_ids[-shared_prompt_budget:]
    if not shared_prompt:
        raise ValueError("偏好样本没有保留 prompt Token")

    def sequence(response_ids: Sequence[int], kept: Sequence[int]) -> dict[str, Any]:
        input_ids = list(shared_prompt) + list(kept)
        return {
            "input_ids": input_ids,
            "response_start": len(shared_prompt),
            "response_tokens_total": len(response_ids),
            "response_tokens_retained": len(kept),
            "response_truncated": len(kept) < len(response_ids),
        }

    return {
        "chosen": sequence(chosen_ids, chosen_kept),
        "rejected": sequence(rejected_ids, rejected_kept),
        "prompt_tokens_total": len(prompt_ids),
        "prompt_tokens_retained": len(shared_prompt),
        "prompt_truncated": len(shared_prompt) < len(prompt_ids),
    }


def run_dpo_delivery(
    *,
    model_id: str,
    revision: str,
    sft_adapter_path: str | Path,
    tokenizer: Any,
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    max_length: int,
    update_steps: int,
    beta: float,
    learning_rate: float,
    seed: int,
    dtype_name: str,
    artifact_directory: str | Path,
) -> dict[str, Any]:
    """Run a fixed DPO candidate against a frozen reference and save its adapter."""

    import gc

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("真实 DPO 项目需要 CUDA")
    if beta <= 0 or update_steps <= 0:
        raise ValueError("beta 和 update_steps 必须大于 0")
    artifact_root = Path(artifact_directory)
    if artifact_root.exists():
        raise FileExistsError(f"artifact 已存在：{artifact_root}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda")

    def load_policy(trainable: bool):
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        ).to(device)
        base.config.use_cache = False
        return PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=trainable)

    policy = load_policy(True)
    reference = load_policy(False)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    def response_logp(model: Any, sequence: Mapping[str, Any], *, grad: bool) -> Any:
        input_ids = sequence["input_ids"]
        response_start = int(sequence["response_start"])
        if not 0 < response_start < len(input_ids):
            raise ValueError("偏好序列必须同时保留 prompt 与 response Token")
        tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.ones_like(tensor)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = model(input_ids=tensor, attention_mask=mask).logits[:, :-1]
            targets = tensor[:, 1:]
            token_logps = functional.log_softmax(logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            start = max(response_start - 1, 0)
            return token_logps[:, start:].mean()

    def encode_record(record: Mapping[str, Any]) -> dict[str, Any]:
        prompt, chosen, rejected = preference_record_to_text(record)
        return build_preference_pair_sequences(
            tokenizer, prompt, chosen, rejected, max_length=max_length
        )

    train_pairs = [encode_record(record) for record in train_records]
    evaluation_pairs = [encode_record(record) for record in evaluation_records]

    def evaluate(model: Any) -> dict[str, float]:
        margins = []
        for pair in evaluation_pairs:
            chosen_logp = response_logp(model, pair["chosen"], grad=False)
            rejected_logp = response_logp(model, pair["rejected"], grad=False)
            margins.append(float((chosen_logp - rejected_logp).item()))
        return {
            "preference_accuracy": sum(margin > 0 for margin in margins) / len(margins),
            "margin_mean": statistics.fmean(margins),
            "margin_min": min(margins),
        }

    baseline = evaluate(policy)
    losses: list[float] = []
    times_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    policy.train()
    for step in range(update_steps):
        pair = train_pairs[step % len(train_pairs)]
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        policy_chosen = response_logp(policy, pair["chosen"], grad=True)
        policy_rejected = response_logp(policy, pair["rejected"], grad=True)
        reference_chosen = response_logp(reference, pair["chosen"], grad=False)
        reference_rejected = response_logp(reference, pair["rejected"], grad=False)
        logits = beta * (
            (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        )
        loss = -functional.logsigmoid(logits)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().item()))
        times_ms.append((time.perf_counter() - started) * 1000.0)

    candidate = evaluate(policy)
    policy.save_pretrained(artifact_root)
    tokenizer.save_pretrained(artifact_root)
    artifact_hash = hash_directory(artifact_root)
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "preference_accuracy_gain": candidate["preference_accuracy"] - baseline["preference_accuracy"],
            "margin_mean_gain": candidate["margin_mean"] - baseline["margin_mean"],
            "margin_floor_gain": candidate["margin_min"] - baseline["margin_min"],
        },
        "training": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "median_step_time_ms": statistics.median(times_ms),
            "update_steps": update_steps,
        },
        "sequence_contract": {
            "max_length": max_length,
            "train_pair_count": len(train_pairs),
            "evaluation_pair_count": len(evaluation_pairs),
            "train_prompt_truncated_count": sum(pair["prompt_truncated"] for pair in train_pairs),
            "evaluation_prompt_truncated_count": sum(pair["prompt_truncated"] for pair in evaluation_pairs),
            "train_response_truncated_count": sum(
                pair[side]["response_truncated"]
                for pair in train_pairs for side in ("chosen", "rejected")
            ),
            "evaluation_response_truncated_count": sum(
                pair[side]["response_truncated"]
                for pair in evaluation_pairs for side in ("chosen", "rejected")
            ),
            "prompt_truncation_ratio": (
                sum(pair["prompt_truncated"] for pair in train_pairs + evaluation_pairs)
                / (len(train_pairs) + len(evaluation_pairs))
            ),
            "response_truncation_ratio": (
                sum(
                    pair[side]["response_truncated"]
                    for pair in train_pairs + evaluation_pairs
                    for side in ("chosen", "rejected")
                )
                / (2 * (len(train_pairs) + len(evaluation_pairs)))
            ),
        },
        "resources": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "artifact": {"path": str(artifact_root), "sha256": artifact_hash},
        "status": "ok",
    }
    del optimizer, policy, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def extract_final_number(text: str) -> str | None:
    """Extract the final normalized numeric answer from generated text."""

    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(text))
    return matches[-1].replace(",", "") if matches else None


def run_grpo_delivery(
    *,
    model_id: str,
    revision: str,
    sft_adapter_path: str | Path,
    tokenizer: Any,
    train_tasks: Sequence[Mapping[str, Any]],
    evaluation_tasks: Sequence[Mapping[str, Any]],
    group_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    clip_epsilon: float,
    kl_coefficient: float,
    learning_rate: float,
    seed: int,
    dtype_name: str,
    artifact_directory: str | Path,
    source_policy_version: str,
    verifier_version: str,
) -> dict[str, Any]:
    """Measure grouped rollouts and finite response-level proxy updates.

    source_policy_version identifies the initial adapter; behavior_policy_version
    identifies its run-local state after behavior_update_step completed updates,
    not a persisted checkpoint hash. Measurement v2 requires a new run label.
    """

    import gc

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if group_size < 2:
        raise ValueError("group_size 必须至少为 2")
    if not torch.cuda.is_available():
        raise RuntimeError("真实 GRPO 项目需要 CUDA")
    artifact_root = Path(artifact_directory)
    if artifact_root.exists():
        raise FileExistsError(f"artifact 已存在：{artifact_root}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_policy(trainable: bool):
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        ).to(device)
        base.config.use_cache = True
        return PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=trainable)

    policy = load_policy(True)
    reference = load_policy(False)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    def gold_answer(task: Mapping[str, Any]) -> str | None:
        return extract_final_number(str(task["answer"]).split("####")[-1])

    def generate_group(model: Any, prompt: str, generation_seed: int) -> tuple[list[str], list[int]]:
        torch.manual_seed(generation_seed)
        torch.cuda.manual_seed_all(generation_seed)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=group_size,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        generated_tokens = generated[:, prompt_tokens:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        response_lengths = []
        for row in generated_tokens:
            token_ids = row.tolist()
            if tokenizer.eos_token_id in token_ids:
                response_lengths.append(token_ids.index(tokenizer.eos_token_id) + 1)
            else:
                response_lengths.append(len(token_ids))
        return responses, response_lengths

    def response_logp(model: Any, prompt: str, response: str, *, grad: bool) -> Any:
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_length
        )["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [
            tokenizer.eos_token_id
        ]
        input_ids = torch.tensor(
            prompt_ids + response_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = model(input_ids=input_ids).logits[:, :-1]
            targets = input_ids[:, 1:]
            token_logps = functional.log_softmax(logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            return token_logps[:, max(len(prompt_ids) - 1, 0) :].mean()

    def rollout_summary(model: Any, tasks: Sequence[Mapping[str, Any]], seed_offset: int):
        rewards: list[float] = []
        pass_at_group: list[float] = []
        response_lengths: list[int] = []
        for index, task in enumerate(tasks):
            responses, lengths = generate_group(
                model, str(task["question"]), seed + seed_offset + index
            )
            gold = gold_answer(task)
            group_rewards = [float(extract_final_number(response) == gold) for response in responses]
            rewards.extend(group_rewards)
            response_lengths.extend(lengths)
            pass_at_group.append(float(any(group_rewards)))
        return {
            "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
            "pass_at_group": statistics.fmean(pass_at_group) if pass_at_group else 0.0,
            "response_count": len(rewards),
            "response_length_mean": (
                statistics.fmean(response_lengths) if response_lengths else 0.0
            ),
            "response_length_p95": (
                sorted(response_lengths)[
                    min(len(response_lengths) - 1, int(0.95 * len(response_lengths)))
                ]
                if response_lengths
                else 0
            ),
            "max_length_hit_ratio": (
                sum(length >= max_new_tokens for length in response_lengths)
                / len(response_lengths)
                if response_lengths
                else 0.0
            ),
        }

    evaluation_seed_offset = 10_000
    baseline = rollout_summary(policy, evaluation_tasks, evaluation_seed_offset)
    valid_groups = 0
    total_groups = 0
    update_losses: list[float] = []
    rollout_seconds = 0.0
    update_seconds = 0.0
    trajectory_records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    training_loop_started = time.perf_counter()

    for task_index, task in enumerate(train_tasks):
        total_groups += 1
        behavior_update_step = len(update_losses)
        behavior_policy_version = stable_text_hash((source_policy_version, behavior_update_step))
        prompt = str(task["question"])
        started = time.perf_counter()
        responses, response_lengths = generate_group(policy, prompt, seed + task_index)
        rollout_seconds += time.perf_counter() - started
        gold = gold_answer(task)
        rewards = [float(extract_final_number(response) == gold) for response in responses]
        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.pstdev(rewards)
        prompt_id = stable_text_hash((prompt,))
        trace_ids = [
            stable_text_hash(
                (
                    source_policy_version,
                    behavior_update_step,
                    verifier_version,
                    prompt_id,
                    task_index,
                    sample_index,
                    seed,
                )
            )
            for sample_index in range(group_size)
        ]
        trajectory_records.append(
            {
                "prompt_id": prompt_id,
                "trace_ids": trace_ids,
                "source_policy_version": source_policy_version,
                "behavior_policy_version": behavior_policy_version,
                "behavior_update_step": behavior_update_step,
                "verifier_version": verifier_version,
                "rewards": rewards,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "response_length_mean": statistics.fmean(response_lengths),
                "max_length_hit_ratio": sum(
                    length >= max_new_tokens for length in response_lengths
                )
                / len(response_lengths),
            }
        )
        if reward_std == 0.0:
            continue
        valid_groups += 1
        advantages = [(reward - reward_mean) / reward_std for reward in rewards]
        old_logps = [
            response_logp(policy, prompt, response, grad=False).detach()
            for response in responses
        ]
        reference_logps = [
            response_logp(reference, prompt, response, grad=False).detach()
            for response in responses
        ]
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        terms = []
        started = time.perf_counter()
        for response, advantage, old_logp, reference_logp in zip(
            responses, advantages, old_logps, reference_logps
        ):
            current_logp = response_logp(policy, prompt, response, grad=True)
            ratio = torch.exp(current_logp - old_logp)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_term = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
            kl_proxy = (current_logp - reference_logp).pow(2)
            terms.append(policy_term + kl_coefficient * kl_proxy)
        loss = torch.stack(terms).mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        update_seconds += time.perf_counter() - started
        update_losses.append(float(loss.detach().item()))

    torch.cuda.synchronize()
    training_loop_seconds = time.perf_counter() - training_loop_started
    candidate = rollout_summary(policy, evaluation_tasks, evaluation_seed_offset)
    policy.save_pretrained(artifact_root)
    tokenizer.save_pretrained(artifact_root)
    artifact_hash = hash_directory(artifact_root)
    valid_trajectory_count = valid_groups * group_size
    result = {
        "measurement_version": "grpo-delivery-measurement/v2",
        "objective_kind": "response_mean_logprob_clipped_surrogate_with_squared_reference_score_penalty",
        "timing_scope": "complete_training_loop_including_generation_reward_old_reference_scoring_and_updates_excluding_evaluation_and_artifact_io",
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "reward_mean_gain": candidate["reward_mean"] - baseline["reward_mean"],
            "pass_at_group_gain": candidate["pass_at_group"] - baseline["pass_at_group"],
            "paired_evaluation_seed_offset": evaluation_seed_offset,
        },
        "rollout": {
            "group_size": group_size,
            "group_count": total_groups,
            "valid_group_count": valid_groups,
            "valid_group_ratio": valid_groups / total_groups if total_groups else 0.0,
            "trajectory_records": trajectory_records,
            "rollout_seconds": rollout_seconds,
        },
        "update": {
            "update_count": len(update_losses),
            "initial_loss": update_losses[0] if update_losses else None,
            "final_loss": update_losses[-1] if update_losses else None,
            "update_seconds": update_seconds,
            "training_loop_seconds": training_loop_seconds,
            "effective_trajectories_per_second": (
                valid_trajectory_count / training_loop_seconds
                if training_loop_seconds > 0
                else 0.0
            ),
        },
        "resources": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "artifact": {"path": str(artifact_root), "sha256": artifact_hash},
        "status": "ok" if valid_groups else "no_effective_update",
    }
    del optimizer, policy, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result
