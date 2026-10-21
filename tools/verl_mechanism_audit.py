"""Read-only API and evidence audit for the two single-card verl mechanisms.

Finding a method is an interface inspection, not proof of a working experiment.
The default audit never imports CUDA packages. The optional loss check uses
CPU tensors only; neither path initializes Ray or starts training workers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from tools.verl_system_mechanism_runtime import VERL_COMMIT


INTERFACES = (
    ("worker_set_loss", "verl/workers/engine_workers.py", "ActorRolloutRefWorker.set_loss_fn"),
    ("worker_update", "verl/workers/engine_workers.py", "ActorRolloutRefWorker.update_actor"),
    ("worker_score", "verl/workers/engine_workers.py", "ActorRolloutRefWorker.compute_log_prob"),
    ("worker_sync", "verl/workers/engine_workers.py", "ActorRolloutRefWorker.update_weights"),
    ("supervised_loss", "verl/workers/utils/losses.py", "sft_loss"),
    ("sft_wiring", "verl/trainer/sft_trainer.py", "SFTTrainer._build_engine"),
    ("sync_after_step", "verl/trainer/ppo/v1/trainer_sync.py", "PPOTrainerSync.on_step_end"),
    ("sleep_after_sample", "verl/trainer/ppo/v1/trainer_sync.py", "PPOTrainerSync.on_sample_end"),
    ("engine_sleep", "verl/workers/rollout/vllm_rollout/vllm_async_server.py", "vLLMHttpServer.sleep"),
    ("engine_release", "verl/workers/rollout/vllm_rollout/vllm_async_server.py", "vLLMHttpServer.release_kv_cache"),
    ("engine_resume", "verl/workers/rollout/vllm_rollout/vllm_async_server.py", "vLLMHttpServer.resume_kv_cache"),
)


def inspect_symbol(text: str, symbol: str) -> dict:
    """Inspect an exact class/function path without importing upstream modules."""
    nodes = ast.parse(text).body
    node = None
    for component in symbol.split("."):
        node = next((n for n in nodes if getattr(n, "name", None) == component), None)
        if node is None:
            return {"found": False}
        nodes = getattr(node, "body", [])
    is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return {
        "found": is_function,
        "line": node.lineno,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "parameters": [a.arg for a in node.args.args] if is_function else [],
        "source_sha256": hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest(),
    }


def inspect_api(root: Path) -> dict:
    """Pin the inspected source and report unsupported wiring explicitly."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
    )
    resolved = commit.stdout.strip() if commit.returncode == 0 else None
    interfaces = {}
    for name, relative, symbol in INTERFACES:
        path = root / relative
        interfaces[name] = {
            "path": relative, "symbol": symbol,
            **(inspect_symbol(path.read_text(), symbol) if path.is_file() else {"found": False}),
        }
    source_ok = resolved == VERL_COMMIT
    return {
        "schema_version": "verl-mechanism-api-audit/v1",
        "expected_commit": VERL_COMMIT, "resolved_commit": resolved,
        "source_identity_matched": source_ok, "interfaces": interfaces,
        "inspection_passed": source_ok and all(i["found"] for i in interfaces.values()),
        "evidence_level": "source_interface_inspection",
        "gpu_recipe_ready": False,
        "remaining_checks": [
            "把监督损失接入实际 actor 更新路径，核对批次掩码和监督位置",
            "采集同名同 dtype 参数的更新前后指纹与差值",
            "冻结轨迹并在更新前后重评，先核对同版本概率容差",
            "记录实际权重同步确认与生成侧概率，不用版本号字符串代替权重证据",
            "把引擎休眠、权重恢复和 KV 恢复事件与资源采样关联",
        ],
        "scope": "接口存在不等于调用链可运行；不证明 GPU 更新、同步或机制通过",
    }


def audit_history(record: dict) -> dict:
    """Describe what the old shared run actually proves, without rewriting it."""
    candidates = record.get("candidates", {})
    release = candidates.get("async_release_metrics", {})
    reject = candidates.get("async_release_reject", {})
    retain = candidates.get("async_retain_metrics", {})
    release_metrics = release.get("metrics", {}).get("mean", {})
    reject_metrics = reject.get("metrics", {}).get("mean", {})
    calls_ok = bool(candidates) and all(c.get("status") == "ok" for c in candidates.values())
    correction_present = all(k in release_metrics for k in (
        "rollout_corr/kl", "rollout_corr/log_ppl_abs_diff", "rollout_corr/chi2_token",
    )) and "rollout_corr/rollout_rs_masked_fraction" in reject_metrics
    resources_present = all(
        c.get("status") == "ok" and c.get("gpu_memory", {}).get("peak_used_mb") is not None
        and all(k in c.get("metrics", {}).get("mean", {}) for k in (
            "timing_s/gen", "timing_s/update_actor", "timing_s/step",
        )) for c in (release, retain)
    )
    return {
        "run_label": record.get("run_label"),
        "schema_version": "verl-mechanism-history-audit/v1",
        "rollout_lifecycle": {
            "status": "partial" if calls_ok and correction_present else "incomplete",
            "observed": ["verl 阶段调用", "概率差异指标", "Token 过滤路径"]
            if correction_present else [],
            "not_proven": ["真实参数变化", "更新前后同一轨迹重评", "权重身份与同步后概率核对"],
        },
        "rl_system": {
            "status": "partial" if resources_present else "incomplete",
            "observed": ["整段显存与主机内存", "逐步阶段耗时", "引擎释放/保留配置对照"]
            if resources_present else [],
            "not_proven": ["逐阶段状态驻留", "有效更新吞吐", "稳态加速收益", "跨池通信与队列容量"],
        },
        "scope": "历史指标齐全不等于机制全部通过；缺失证据不能从零梯度或配置开关推断",
    }


def check_sft_loss_cpu() -> dict:
    """Check real verl loss/mask alignment on CPU, without any worker setup."""
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.workers.utils.losses import sft_loss

    values = torch.tensor([[-0.2, -0.4, -9.0], [-0.6, -9.0, -9.0]], requires_grad=True)
    data = TensorDict({"response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)}, [2])
    tu.assign_non_tensor(data, pad_mode=DatasetPadMode.LEFT_RIGHT, dp_size=1, batch_num_tokens=3)
    loss, _ = sft_loss(None, {"log_probs": values}, data)
    loss.backward()
    assert torch.allclose(loss.detach(), torch.tensor(0.4))
    assert torch.allclose(values.grad, torch.tensor([[-1 / 3, -1 / 3, 0], [-1 / 3, 0, 0]]))

    flat = torch.tensor([-9.0, -0.2, -0.4, -9.0, -0.6, -9.0], requires_grad=True)
    offsets = torch.tensor([0, 4, 6])
    log_probs = torch.nested.nested_tensor_from_jagged(flat, offsets)
    mask = torch.nested.nested_tensor_from_jagged(torch.tensor([0, 0, 1, 1, 0, 1]), offsets)
    data = TensorDict({"loss_mask": mask}, [2])
    tu.assign_non_tensor(data, pad_mode=DatasetPadMode.NO_PADDING, dp_size=1, batch_num_tokens=3)
    loss, _ = sft_loss(None, {"log_probs": log_probs}, data)
    loss.backward()
    assert torch.allclose(loss.detach(), torch.tensor(0.4))
    assert torch.allclose(flat.grad, torch.tensor([0, -1 / 3, -1 / 3, 0, -1 / 3, 0]))
    return {
        "status": "passed", "evidence_level": "cpu_loss_contract",
        "checks": ["padded_valid_token_reduction", "nested_shifted_action_positions"],
        "gpu_recipe_ready": False,
        "scope": "仅验证实际监督损失与掩码位置，不验证 actor 更新或权重同步",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="只读核对 verl 机制接口与历史证据")
    parser.add_argument("--verl-root", required=True, type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--check-loss", action="store_true", help="在现有 verl 环境核对实际 CPU 损失")
    args = parser.parse_args()
    root = args.verl_root.expanduser().resolve()
    result = {"api": inspect_api(root)}
    if args.history:
        result["history"] = audit_history(json.loads(args.history.read_text()))
    if args.check_loss:
        result["cpu_loss_contract"] = check_sft_loss_cpu()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["api"]["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
