"""Static contracts for the Part 05 MiniMind post-training workload."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from tools.minimind_post_training_runtime import (
    build_fixed_workload,
    validate_config,
    _write_jsonl_once,
    _train_command,
    sha256_file,
    validate_training_evidence,
    close_project,
    load_prepared,
    train_and_evaluate,
    _check_source_identity,
    RESULT_SCHEMA,
    SEMANTIC_ID,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery/project_config.example.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert validate_config(config, allow_placeholder=True) == []
    assert validate_config(config)
    for section, key, value in (
        (None, "run_label", None),
        ("workload", "seed", "invalid"),
        ("workload", "batch_size", True),
        ("workload", "sft_examples", 990**2 + 1),
        ("workload", "sft_learning_rate", float("nan")),
        ("model", "use_moe", True),
        ("model", "weight_prefix", "sft"),
        ("gates", "maximum_peak_memory_gib", -1),
        ("gates", "require_fixed_source_revision", False),
        ("evaluation", "minimum_dpo_preference_win_rate", 1.1),
    ):
        invalid = deepcopy(config)
        invalid["run_label"] = "config_test_v1"
        (invalid if section is None else invalid[section])[key] = value
        assert validate_config(invalid), (section, key, value)
    assert validate_config(None)
    assert validate_config({})

    first = build_fixed_workload(config)
    second = build_fixed_workload(config)
    assert first == second
    assert first["counts"] == {"sft": 48, "dpo": 24, "evaluation": 32}
    assert first["split_disjoint"] is True
    assert len(first["dpo"]) == 24
    assert first["dpo"][0]["chosen"] != first["dpo"][0]["rejected"]
    train_prompts = {row["conversations"][0]["content"] for row in first["sft"]}
    train_prompts.update(row["chosen"][0]["content"] for row in first["dpo"])
    eval_prompts = {row["conversations"][0]["content"] for row in first["eval_sft"]}
    eval_prompts.update(row["chosen"][0]["content"] for row in first["eval_dpo"])
    assert train_prompts.isdisjoint(eval_prompts)
    chosen = int(first["dpo"][0]["chosen"][1]["content"])
    rejected = int(first["dpo"][0]["rejected"][1]["content"])
    operands = [int(value) for value in re.findall(r"\d+", first["dpo"][0]["chosen"][0]["content"])]
    assert chosen == sum(operands) and rejected == chosen + 1

    # The controlled SFT comparison keeps data, budget and learning rate fixed;
    # DPO starts from the full-SFT checkpoint, not the LoRA adapter branch.
    source_root = Path("/tmp/minimind-source")
    full_sft = _train_command(source_root, config, "sft", "sft.jsonl")
    lora_sft = _train_command(source_root, config, "sft_lora", "sft.jsonl")
    dpo = _train_command(source_root, config, "dpo", "dpo.jsonl")
    assert full_sft[1].endswith("train_full_sft.py")
    assert lora_sft[1].endswith("train_lora.py")
    assert dpo[1].endswith("train_dpo.py")
    assert "--learning_rate" in full_sft and "--learning_rate" in lora_sft
    assert full_sft[full_sft.index("--learning_rate") + 1] == lora_sft[lora_sft.index("--learning_rate") + 1]
    assert dpo[dpo.index("--from_weight") + 1] == f"sft_{config['run_label']}"
    assert f"lora_{config['run_label']}" in lora_sft
    try:
        _train_command(source_root, config, "typo", "data.jsonl")
    except ValueError:
        pass
    else:
        raise AssertionError("未知阶段不能自动执行 DPO")

    evidence_config = deepcopy(config)
    evidence_config["run_label"] = "minimind_test_v1"
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        run_dir = root / config["outputs"]["directory"] / evidence_config["run_label"]
        run_dir.mkdir(parents=True)
        source_root = root / "minimind-source"
        source_root.mkdir()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(source_root), *args], text=True).strip()
        git("init", "--quiet")
        source_file = source_root / "source.py"
        source_file.write_text("# fixture\n", encoding="utf-8")
        git("add", "--", "source.py")
        git("-c", "user.name=Runtime Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
        evidence_config["source"]["revision"] = git("rev-parse", "HEAD")
        evidence_config["source"]["path"] = str(source_root)
        _check_source_identity(source_root, evidence_config["source"]["revision"])
        source_file.write_text("# modified\n", encoding="utf-8")
        try:
            _check_source_identity(source_root, evidence_config["source"]["revision"])
        except RuntimeError:
            pass
        else:
            raise AssertionError("固定提交下的未提交源码修改必须停止")
        source_file.write_text("# fixture\n", encoding="utf-8")
        try:
            _check_source_identity(source_root, "0" * 40)
        except RuntimeError:
            pass
        else:
            raise AssertionError("源码提交与配置不一致必须停止")
        added_code = source_root / "model" / "extra.py"
        added_code.parent.mkdir()
        added_code.write_text("# new module\n", encoding="utf-8")
        try:
            _check_source_identity(source_root, evidence_config["source"]["revision"])
        except RuntimeError:
            pass
        else:
            raise AssertionError("模型目录新增未跟踪代码必须停止")
        added_code.unlink()
        data_paths = {name: str(run_dir / f"{name}.jsonl") for name in ("sft", "dpo", "eval_sft", "eval_dpo")}
        artifact_paths = {}
        for name in ("base", "sft", "sft_lora", "dpo"):
            artifact = run_dir / f"{name}.pth"
            artifact.write_bytes(f"artifact:{name}".encode())
            artifact_paths[name] = artifact
        log_paths = {}
        for name in ("sft", "sft_lora", "dpo"):
            log_path = run_dir / f"{name}.log"
            log_path.write_text("training complete\n", encoding="utf-8")
            log_paths[name] = log_path
        data_hashes = {name: _write_jsonl_once(Path(path), first[name]) for name, path in data_paths.items()}
        base_hash = sha256_file(artifact_paths["base"])
        prepared = {
            "config": evidence_config,
            "source_path": str(source_root),
            "source_revision": evidence_config["source"]["revision"],
            "base_weight_path": str(artifact_paths["base"]),
            "base_weight_sha256": base_hash,
            "data_paths": data_paths,
            "data_sha256": data_hashes,
        }
        manifest = run_dir / "prepare_manifest.json"
        manifest.write_text(json.dumps(prepared), encoding="utf-8")
        assert load_prepared(evidence_config, root) == prepared
        changed_config = deepcopy(evidence_config)
        changed_config["workload"]["max_seq_len"] *= 2
        try:
            load_prepared(changed_config, root)
        except ValueError:
            pass
        else:
            raise AssertionError("修改配置后必须重新准备输入")
        raw = {
            "schema_version": RESULT_SCHEMA,
            "semantic_id": SEMANTIC_ID,
            "run_label": evidence_config["run_label"],
            "environment_profile": evidence_config["environment_profile"],
            "model": {**evidence_config["model"], "weight_sha256": base_hash},
            "workload": {
                **evidence_config["workload"], "split_disjoint": True,
                "data_sha256": data_hashes,
                "data_paths": data_paths,
                "data_counts": {"sft": 48, "dpo": 24, "evaluation": 32},
            },
            "source": {"path": str(source_root), "revision": evidence_config["source"]["revision"]},
            "decision_thresholds": {
                "evaluation": dict(evidence_config["evaluation"]),
                "memory": dict(evidence_config["gates"]),
            },
            "memory_admission_probe": {
                "status": "passed", "run_label": evidence_config["run_label"],
                "source_revision": evidence_config["source"]["revision"],
                "base_weight_sha256": base_hash, "data_sha256": data_hashes,
                "gates": dict(evidence_config["gates"]),
                "phases": {name: {"loss_finite": True, "peak_reserved_bytes": 1024} for name in ("sft", "sft_lora", "dpo")},
            },
            "logs": {
                name: {
                    "returncode": 0, "path": str(log_paths[name]), "elapsed_seconds": 4.0,
                    "command": _train_command(source_root, evidence_config, name, data_paths["dpo" if name == "dpo" else "sft"]),
                }
                for name in log_paths
            },
            "evaluation": {},
        }
        for name, artifact in artifact_paths.items():
            row = {
                "heldout_sft_loss_mean": 1.0,
                "heldout_preference_win_rate": 0.5,
                "evaluation_examples": 32,
                "preference_pairs": 32,
                "total_parameter_count": 1000,
                "artifact_bytes": artifact.stat().st_size,
                "peak_allocated_bytes": 512,
                "peak_reserved_bytes": 1024,
                "weight_path": str(artifact),
                "weight_sha256": sha256_file(artifact),
                "adapter_parameter_count": 10 if name == "sft_lora" else 0,
            }
            raw["evaluation"][name] = row
        assert validate_training_evidence(raw, evidence_config, root) == []
        raw_path = run_dir / "raw_result.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        report = close_project(evidence_config, root)
        assert report["decision"] == "accept"
        assert report["lora_vs_full_sft"]["trainable_parameter_ratio_vs_full_sft"] == 0.01
        assert close_project(evidence_config, root) == report
        try:
            train_and_evaluate(evidence_config, root, prepared)
        except FileExistsError:
            pass
        else:
            raise AssertionError("已有记录必须在导入 torch 和训练前停止")
        raw_path.unlink()
        try:
            train_and_evaluate(evidence_config, root, prepared)
        except FileExistsError:
            pass
        else:
            raise AssertionError("中断后保留的阶段日志也必须阻止覆盖")
        input_path = Path(data_paths["sft"])
        original = input_path.read_bytes()
        input_path.write_bytes(original + b"\n")
        assert any("sft 文件 SHA256 不匹配" in item for item in validate_training_evidence(raw, evidence_config, root))
        try:
            load_prepared(evidence_config, root)
        except ValueError:
            pass
        else:
            raise AssertionError("准备后数据变化必须停止")
        input_path.unlink()
        assert any("sft 文件缺失" in item for item in validate_training_evidence(raw, evidence_config, root))
        input_path.write_bytes(original)
        original_weight = artifact_paths["base"].read_bytes()
        artifact_paths["base"].write_bytes(b"changed-base")
        assert any("基础权重 文件 SHA256 不匹配" in item for item in validate_training_evidence(raw, evidence_config, root))
        try:
            load_prepared(evidence_config, root)
        except ValueError:
            pass
        else:
            raise AssertionError("准备后基础权重变化必须停止")
        artifact_paths["base"].write_bytes(original_weight)
        raw["evaluation"]["sft_lora"]["weight_sha256"] = "0" * 64
        assert any("哈希/文件大小不匹配" in item for item in validate_training_evidence(raw, evidence_config, root))

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "records.jsonl"
        rows = [{"text": "same"}]
        digest = _write_jsonl_once(path, rows)
        assert len(digest) == 64
        assert _write_jsonl_once(path, rows) == digest
        try:
            _write_jsonl_once(path, [{"text": "different"}])
        except FileExistsError:
            pass
        else:
            raise AssertionError("同一路径的不同数据必须停止而不是覆盖")

    print("MiniMind post-training project contracts: ok")


class MiniMindRuntimeContracts(unittest.TestCase):
    """Expose the CPU-only contract suite to standard test discovery."""

    def test_runtime_contracts(self) -> None:
        """Check configuration, inputs and closeout using temporary files."""
        main()


if __name__ == "__main__":
    main()
