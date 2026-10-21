# 86. DPO Online Benchmark | 在线后训练生命周期与恢复

**难度：** Hard | **环境：** CPU + 真实产物 | **标签：** `策略版本`, `恢复`, `回滚`, `交接状态` | **目标人群：** 后训练交付学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/86_DPO_Online_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节接收 84 的 DPO 候选或 85 的 GRPO/RLVR 候选，检查候选是否来自当前 SFT 基线、反馈或轨迹是否仍然有效、各阶段失败后能否恢复，并把通过生命周期检查的候选交给 Part 05 独立评测。

**关键词：** `策略版本`、`证据新鲜度`、`恢复`、`回滚`、`交接状态`

---
## 前置阅读

- [Part 02 · 51 在线 DPO 与反馈窗口](./51_Online_DPO.md)
- [Part 02 · PT-RL-SYSTEM Rollout、训练与奖励系统](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)
- [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹生命周期与策略新鲜度](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)
- [Part 02 · 84 DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [Part 02 · 85 GRPO 组内奖励优化项目](./85_GRPO_Groupwise_Alignment_Project.md)

---
### Step 1：把并列候选送入生命周期交接流程

DPO 与 GRPO/RLVR 是从同一 SFT 基线派生的并列候选，而不是连续训练阶段。项目读取各分支已经形成的候选清单，核对候选血缘、证据新鲜度、产物完整性和回滚基线，并在失败时选择最晚可信恢复点。输出是 `ready_for_evaluation`、`tune` 或 `reject` 交接状态，不在本节比较不同任务指标，也不形成最终采用结论。

| 阶段 | 输入 | 关键动作 | 输出或失败处理 |
|:---|:---|:---|:---|
| 分支状态 | DPO 或 GRPO/RLVR 候选清单 | 核对上游状态、候选产物和分支结论 | `reject` 排除；`tune` 保留调优；`accept` 进入生命周期检查 |
| 身份核对 | 当前 SFT 基线、评测版本和候选血缘 | 比较源策略、评测与产物身份 | 输出新鲜度报告；身份失配则 `reject` |
| 产物检查 | 候选适配器与 SFT 回滚基线 | 重新计算哈希并核对磁盘产物 | 产物或回滚失配则 `reject` |
| 恢复与交接 | 失败阶段和已保存状态 | 选择最晚可信恢复点并汇总检查结果 | 输出恢复动作与交接状态 |

![后训练候选的生命周期交接流程](../public/02_PyTorch_Algorithms/86_post_training_adoption_flow.svg)

### Step 2：用证据身份判断候选是否仍然有效

候选证据至少由源基线版本、候选策略版本、数据或反馈版本、评测版本和产物哈希共同标识。当前激活基线、评测集或验证规则发生变化时，旧结果可以保留为历史记录，却不能直接交给当前独立评测。在线候选还要检查轨迹窗口和源策略版本；离线候选则要确认偏好证据仍对应当前基线。

| 发生变化的身份 | DPO 证据受到的影响 | GRPO / RLVR 证据受到的影响 |
|:---|:---|:---|
| 当前 SFT 基线 | 相对参考与候选血缘失配 | 轨迹来源与候选血缘失配 |
| 偏好数据或问题集 | 需要重新检查隔离与质量门槛 | 需要重新采样并验证轨迹 |
| 参考模型或验证器 | 相对边际的参照发生变化 | 奖励含义和组内优势发生变化 |
| 评测版本 | 原质量结论不能直接沿用 | 原新轨迹结论不能直接沿用 |
| 产物哈希 | 无法证明审核对象仍是原候选 | 无法证明审核对象仍是原候选 |

### Step 3：从最晚可信状态恢复并形成交接状态

恢复动作由失败发生在哪个阶段以及此前保存了什么可信状态共同决定。恢复完成后仍要重新检查候选血缘、新鲜度、产物和回滚条件。上游已拒绝、证据失效、产物损坏或回滚不可用时输出 `reject`；上游仍需调优时输出 `tune`；生命周期条件全部通过时输出 `ready_for_evaluation`，由 Part 05 再检查质量、安全、鲁棒性和成本。

| 事件或检查 | 可复用的最晚可信状态 | 动作与候选状态 |
|:---|:---|:---|
| 采样失败 | 问题队列 | 从队列重试；缺失时重建队列 |
| 奖励失败 | 已保存轨迹 | 重新计算奖励；缺失时重新采样 |
| 更新失败 | 更新前检查点 | 恢复检查点；缺失时回滚基线 |
| 产物校验失败 | 已验证基线 | 回滚并阻断发布 |
| 血缘、产物或回滚失败 | 无可接受替代证据 | `reject` |
| 上游候选仍需调优 | 候选及现有证据仍可复用 | `tune` |
| 生命周期条件全部通过 | 已验证候选与回滚基线 | `ready_for_evaluation` |

![后训练失败阶段与恢复点](../public/02_PyTorch_Algorithms/86_recovery_state_flow.svg)

### Step 4：实现新鲜度、恢复与交接判断

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：检查候选血缘与分支证据有效期 | 候选类型、源策略、评测身份及分支证据 | DPO 保留偏好证据身份；GRPO/RLVR 还满足轨迹窗口与验证器条件 | 基线推进、在线窗口过期、评测或分支身份缺失 |
| TODO 2：按失败阶段选择恢复点 | 失败阶段和现有可信状态 | 指出要复用的状态，并选择最小重试或回退动作 | 采样、奖励、更新和产物校验失败 |
| TODO 3：传播上游状态并形成交接判断 | 上游状态、新鲜度、产物和回滚 | 上游拒绝或生命周期硬失败则拒绝；上游调优则调优；其余进入独立评测 | 上游三态、证据失效、产物失配与回滚缺失 |

```python
from typing import Any, Dict
```


```python
# ==========================================
# TODO 1：检查候选策略血缘与证据新鲜度
# ==========================================
def check_candidate_freshness(candidate: Dict[str, Any], active_policy_version: str, maximum_window_age: int) -> Dict[str, Any]:
    """按离线偏好或在线轨迹分支检查候选证据。"""
    candidate_kind = candidate.get("candidate_kind")
    common_checks = {
        "candidate_kind": candidate.get("candidate_kind") in {"dpo", "grpo", "rlvr"},
        "source_policy": candidate.get("source_policy_version") == active_policy_version,
        "candidate_identity": bool(candidate.get("policy_version")),
        "evaluation_identity": bool(candidate.get("evaluation_version")),
    }
    # branch_evidence_fresh = ???  # DPO 要有偏好证据身份；在线候选还要有验证器身份且窗口未过期。
    # fresh = ???  # 共同血缘与分支证据必须同时有效。
    raise NotImplementedError("TODO 1：请判断分支证据与共同血缘是否新鲜")
    checks = {**common_checks, "branch_evidence": branch_evidence_fresh}
    return {"fresh": fresh, "failed_checks": [name for name, passed in checks.items() if not passed]}


# ==========================================
# TODO 2：根据失败阶段选择最晚可信恢复点
# ==========================================
def choose_recovery_action(failure: Dict[str, Any]) -> Dict[str, str]:
    """依据失败阶段和已保存证据选择恢复动作。"""
    stage = failure["stage"]
    available = failure["available"]
    recovery_candidates_by_stage = {
        "sampling": ["prompt_queue"],
        "reward": ["trajectories", "prompt_queue"],
        "update": ["pre_update_checkpoint", "verified_baseline"],
        "artifact_validation": ["verified_baseline"],
    }
    action_by_stage_and_state = {
        ("sampling", "prompt_queue"): "retry_sampling_from_prompt_queue",
        ("reward", "trajectories"): "resume_reward_from_saved_trajectories",
        ("reward", "prompt_queue"): "restart_sampling_from_prompt_queue",
        ("update", "pre_update_checkpoint"): "restore_pre_update_checkpoint",
        ("update", "verified_baseline"): "rollback_to_verified_baseline",
        ("artifact_validation", "verified_baseline"): "rollback_to_verified_baseline",
    }
    fallback_action_by_stage = {
        "sampling": "rebuild_prompt_queue",
        "reward": "restart_sampling",
        "update": "block_update",
        "artifact_validation": "block_release",
    }
    if stage not in recovery_candidates_by_stage:
        raise ValueError(f"未知失败阶段：{stage}")
    # recovery_state = ???  # 按候选顺序选择第一个当前可用的可信状态；都不可用时为 None。
    # action = ???  # 有可信状态时执行对应的最小恢复动作，否则执行阶段回退动作。
    raise NotImplementedError("TODO 2：请搜索最晚可信恢复点并选择对应动作")
    return {"stage": stage, "recovery_state": recovery_state, "reused_state": recovery_state is not None, "action": action}


# ==========================================
# TODO 3：形成生命周期交接状态
# ==========================================
def decide_lifecycle_handoff(upstream_decision: str, freshness: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """传播上游状态，并检查产物与回滚条件。"""
    hard_checks = {
        "freshness": freshness["fresh"],
        "rollback": evidence["rollback_ready"],
        "artifact": evidence["artifact_verified"],
    }

    # handoff_status = ???  # 含义：reject、tune 或 ready_for_evaluation。
    raise NotImplementedError("TODO 3：请形成生命周期交接状态")
    return {
        "handoff_status": handoff_status,
        "hard_failures": [name for name, passed in hard_checks.items() if not passed],
    }
```


```python
def test_candidate_freshness():
    """TODO 1：验证候选血缘和证据窗口。"""
    candidate = {"candidate_kind": "dpo", "decision": "accept", "source_policy_version": "sft-v1", "policy_version": "dpo-v1", "evaluation_version": "pref-eval-v1", "preference_evidence_version": "pref-train-v1", "window_age": 99}
    freshness = check_candidate_freshness(candidate, "sft-v1", 2)
    assert freshness["fresh"]
    assert check_candidate_freshness({**candidate, "decision": "tune"}, "sft-v1", 2)["fresh"]
    assert not check_candidate_freshness({**candidate, "source_policy_version": "sft-v0"}, "sft-v1", 2)["fresh"]
    assert not check_candidate_freshness({**candidate, "preference_evidence_version": ""}, "sft-v1", 2)["fresh"]
    online = {**candidate, "candidate_kind": "grpo", "verifier_version": "math-v1", "window_age": 2}
    assert check_candidate_freshness(online, "sft-v1", 2)["fresh"]
    assert not check_candidate_freshness({**online, "window_age": 3}, "sft-v1", 2)["fresh"]
    assert not check_candidate_freshness({**online, "verifier_version": ""}, "sft-v1", 2)["fresh"]
    assert not check_candidate_freshness({**candidate, "candidate_kind": "unknown"}, "sft-v1", 2)["fresh"]
    assert not check_candidate_freshness({**candidate, "evaluation_version": ""}, "sft-v1", 2)["fresh"]


def test_recovery_actions():
    """TODO 2：验证各失败阶段的恢复与回退动作。"""
    assert choose_recovery_action({"stage": "sampling", "available": {"prompt_queue": True}})["action"] == "retry_sampling_from_prompt_queue"
    reward_recovery = choose_recovery_action({"stage": "reward", "available": {"trajectories": True}})
    assert reward_recovery["action"] == "resume_reward_from_saved_trajectories"
    assert reward_recovery["recovery_state"] == "trajectories" and reward_recovery["reused_state"]
    assert choose_recovery_action({"stage": "update", "available": {"pre_update_checkpoint": True}})["action"] == "restore_pre_update_checkpoint"
    assert choose_recovery_action({"stage": "artifact_validation", "available": {"verified_baseline": True}})["action"] == "rollback_to_verified_baseline"
    reward_fallback = choose_recovery_action({"stage": "reward", "available": {"trajectories": False, "prompt_queue": True}})
    assert reward_fallback["recovery_state"] == "prompt_queue"
    assert reward_fallback["action"] == "restart_sampling_from_prompt_queue"
    assert choose_recovery_action({"stage": "reward", "available": {}})["action"] == "restart_sampling"


def test_lifecycle_handoff():
    """TODO 3：验证拒绝、调优和进入独立评测三种结果。"""
    freshness = {"fresh": True}
    evidence = {"rollback_ready": True, "artifact_verified": True}
    assert decide_lifecycle_handoff("accept", freshness, evidence)["handoff_status"] == "ready_for_evaluation"
    assert decide_lifecycle_handoff("tune", freshness, evidence)["handoff_status"] == "tune"
    rejected = decide_lifecycle_handoff("reject", freshness, evidence)
    assert rejected["handoff_status"] == "reject" and "upstream_reject" in rejected["hard_failures"]
    assert decide_lifecycle_handoff("accept", {"fresh": False}, evidence)["handoff_status"] == "reject"
    assert decide_lifecycle_handoff("accept", freshness, {**evidence, "rollback_ready": False})["handoff_status"] == "reject"


test_candidate_freshness()
test_recovery_actions()
test_lifecycle_handoff()
print("所有测试通过！")
```

🛑 **STOP HERE** 🛑
## 参考代码与解析

### 代码

```python
# TODO 1：检查候选策略血缘与证据新鲜度
def check_candidate_freshness(candidate: Dict[str, Any], active_policy_version: str, maximum_window_age: int) -> Dict[str, Any]:
    """按离线偏好或在线轨迹分支检查候选证据。"""
    candidate_kind = candidate.get("candidate_kind")
    common_checks = {
        "candidate_kind": candidate.get("candidate_kind") in {"dpo", "grpo", "rlvr"},
        "source_policy": candidate.get("source_policy_version") == active_policy_version,
        "candidate_identity": bool(candidate.get("policy_version")),
        "evaluation_identity": bool(candidate.get("evaluation_version")),
    }
    if candidate_kind == "dpo":
        branch_evidence_fresh = bool(candidate.get("preference_evidence_version"))
    else:
        branch_evidence_fresh = bool(candidate.get("verifier_version")) and 0 <= candidate.get("window_age", -1) <= maximum_window_age
    checks = {**common_checks, "branch_evidence": branch_evidence_fresh}
    fresh = all(checks.values())
    return {"fresh": fresh, "failed_checks": [name for name, passed in checks.items() if not passed]}


# TODO 2：根据失败阶段选择最晚可信恢复点
def choose_recovery_action(failure: Dict[str, Any]) -> Dict[str, str]:
    """依据失败阶段和现有可信状态选择最小恢复范围。"""
    stage = failure["stage"]
    available = failure["available"]
    recovery_candidates_by_stage = {
        "sampling": ["prompt_queue"],
        "reward": ["trajectories", "prompt_queue"],
        "update": ["pre_update_checkpoint", "verified_baseline"],
        "artifact_validation": ["verified_baseline"],
    }
    action_by_stage_and_state = {
        ("sampling", "prompt_queue"): "retry_sampling_from_prompt_queue",
        ("reward", "trajectories"): "resume_reward_from_saved_trajectories",
        ("reward", "prompt_queue"): "restart_sampling_from_prompt_queue",
        ("update", "pre_update_checkpoint"): "restore_pre_update_checkpoint",
        ("update", "verified_baseline"): "rollback_to_verified_baseline",
        ("artifact_validation", "verified_baseline"): "rollback_to_verified_baseline",
    }
    fallback_action_by_stage = {
        "sampling": "rebuild_prompt_queue",
        "reward": "restart_sampling",
        "update": "block_update",
        "artifact_validation": "block_release",
    }
    if stage not in recovery_candidates_by_stage:
        raise ValueError(f"未知失败阶段：{stage}")
    recovery_state = next((state for state in recovery_candidates_by_stage[stage] if available.get(state)), None)
    action = action_by_stage_and_state[(stage, recovery_state)] if recovery_state is not None else fallback_action_by_stage[stage]
    return {"stage": stage, "recovery_state": recovery_state, "reused_state": recovery_state is not None, "action": action}


# TODO 3：形成生命周期交接状态
def decide_lifecycle_handoff(upstream_decision: str, freshness: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """传播上游状态，并按生命周期硬条件形成三态交接。"""
    hard_checks = {
        "freshness": freshness["fresh"],
        "rollback": evidence["rollback_ready"],
        "artifact": evidence["artifact_verified"],
    }
    hard_failures = [name for name, passed in hard_checks.items() if not passed]
    if upstream_decision not in {"accept", "tune", "reject"}:
        hard_failures.append("upstream_state")
    elif upstream_decision == "reject":
        hard_failures.append("upstream_reject")
    handoff_status = "reject" if hard_failures else "tune" if upstream_decision == "tune" else "ready_for_evaluation"
    return {
        "handoff_status": handoff_status,
        "upstream_decision": upstream_decision,
        "hard_failures": hard_failures,
    }
```

### 解析

#### TODO 1：检查候选新鲜度

- **判断依据**：所有候选都必须来自当前基线并声明独立评测身份；DPO 还要保留偏好证据版本，GRPO/RLVR 则要检查验证器身份和轨迹窗口。上游采用状态不属于新鲜度。
- **实现动作**：先检查共同血缘，再按候选分支检查偏好证据或在线轨迹证据。
- **测试证据**：测试区分 DPO 的偏好证据缺失与在线候选的窗口过期、验证器缺失。

#### TODO 2：选择恢复动作

- **判断依据**：恢复应复用最晚可信证据，避免奖励失败后重新采样，也避免更新失败后重复消费轨迹。
- **实现动作**：按失败阶段列出由近到远的可信状态候选，选择第一个可用状态；没有状态可复用时才执行阶段回退。
- **测试证据**：测试覆盖轨迹可复用、轨迹缺失但问题队列仍可用，以及无可信状态三种恢复范围。

#### TODO 3：形成交接状态

- **判断依据**：上游 `reject`、证据失效、回滚失败或产物损坏会阻断候选；上游 `tune` 继续保留调优状态；其余候选只获得进入独立评测的资格。
- **实现动作**：先传播上游三态，再检查血缘、产物和回滚硬条件，输出 `reject`、`tune` 或 `ready_for_evaluation`。
- **测试证据**：测试分别覆盖上游拒绝、上游调优、证据失效、回滚缺失和进入独立评测路径。
## Step 5：真实产物生命周期与故障恢复演练

本实验读取 62、84、85 的 JSON 证据，重新计算基线与候选适配器目录哈希，并验证候选进入独立评测前的血缘、新鲜度、恢复和回滚契约。该阶段消费既有训练产物，因此使用 CPU 即可完成。

| 对照层 | 输入 | 主要证据 | 输出与证据边界 |
|:---|:---|:---|:---|
| CPU 状态机 | 人工构造的候选、失败阶段与可用状态 | 新鲜度、恢复动作、交接状态 | 只确认生命周期逻辑，不证明真实产物可恢复 |
| 真实产物审核 | SFT 基线与 DPO/GRPO 候选清单、磁盘产物 | 血缘、哈希与回滚路径 | 输出交接清单；质量、安全、鲁棒性和成本由 Part 05 复测 |

### 5.1 环境、输入资产与固定条件

本页不启动新的训练进程，使用能够导入仓库运行时的 CPU 环境读取 62、84、85 的已保存清单与适配器目录。固定输入为 SFT 回退基线、DPO 候选和 GRPO / RLVR 候选；修改候选集合或新鲜度窗口时必须使用新的结果标签。

```python
from pathlib import Path
import json

from tools.project_runtime import resolve_project_root

RUN_MODE = "dry_run"  # 默认值 dry_run：只检查配置；改为 real_artifact 审核真实产物。
RUN_LABEL = "post_training_lifecycle_v6"  # v5 暴露了窗口年龄方向错误；v6 仍交接 84 v5 与 85 v7。
SEMANTIC_ID = "PT-POST-TRAINING-LIFECYCLE"
SCHEMA_VERSION = "post-training-lifecycle/v1"

PROJECT_ROOT = resolve_project_root()
SFT_MANIFEST = PROJECT_ROOT / "benchmarks/results/62_sft_delivery/sft_delivery_manifest_sft_delivery_v2_1.json"
CANDIDATE_MANIFESTS = [
    PROJECT_ROOT / "benchmarks/results/84_dpo_preference/dpo_manifest_dpo_offline_len1024_v5.json",
    PROJECT_ROOT / "benchmarks/results/85_grpo_alignment/grpo_manifest_grpo_rlvr_len384_v7.json",
]
MAXIMUM_WINDOW_AGE = 2
ACTIVE_WINDOW_INDEX = 1  # 当前交接窗口；85 v7 在窗口 1 生成，年龄为当前窗口减证据窗口。
RESULTS_DIR = PROJECT_ROOT / "benchmarks/results/86_post_training_lifecycle"
EVIDENCE_PATH = RESULTS_DIR / f"lifecycle_evidence_{RUN_LABEL}.json"
HANDOFF_PATH = RESULTS_DIR / f"lifecycle_handoff_{RUN_LABEL}.json"
print({"run_mode": RUN_MODE, "candidate_manifests": [str(path) for path in CANDIDATE_MANIFESTS]})
```

### 5.2 预检基线与候选清单

`real_artifact` 模式至少需要一个真实候选清单；尚未生成的并列分支记录为未完成，不参与生命周期交接。

```python
from tools.post_training_project_runtime import hash_directory, save_json_record
from tools.project_runtime import require_input_file

if RUN_MODE == "real_artifact" and (EVIDENCE_PATH.exists() or HANDOFF_PATH.exists()):
    raise FileExistsError("结果已存在，请修改 RUN_LABEL")
if RUN_MODE == "real_artifact":
    sft_path = require_input_file(SFT_MANIFEST, "62 节 SFT manifest")
    sft_manifest = json.loads(sft_path.read_text(encoding="utf-8"))
    available_candidates = [path for path in CANDIDATE_MANIFESTS if path.exists()]
    if not available_candidates:
        raise FileNotFoundError("84/85 尚无候选 manifest")
    print({"baseline": str(sft_path), "available_candidates": [str(path) for path in available_candidates]})
else:
    print({"run_mode": RUN_MODE, "next_action": "完成 84 或 85 后改为 real_artifact"})
```

### 5.3 固定新鲜度窗口与故障场景

故障演练覆盖采样、奖励、更新和产物校验四个阶段；每个场景只改变失败阶段和可用证据。

```python
def check_candidate_freshness_for_run(candidate, active_policy_version, maximum_window_age):
    semantic_id = candidate.get("semantic_id")
    common_checks = {
        "candidate_kind": semantic_id in {"PT-DPO-OFFLINE", "PT-GRPO-GROUPWISE"},
        "source_policy": candidate.get("source_policy_version") == active_policy_version,
        "candidate_identity": bool(candidate.get("policy_version")),
        "evaluation_identity": bool(candidate.get("evaluation_version")),
    }
    if semantic_id == "PT-DPO-OFFLINE":
        branch_evidence_fresh = bool(candidate.get("source_evidence"))
    else:
        verifier_version = candidate.get("trajectory_identity", {}).get("verifier_version")
        branch_evidence_fresh = bool(verifier_version) and 0 <= candidate.get("window_age", -1) <= maximum_window_age
    checks = {**common_checks, "branch_evidence": branch_evidence_fresh}
    return {"fresh": all(checks.values()), "failed_checks": [name for name, passed in checks.items() if not passed]}

def choose_recovery_action_for_run(failure):
    stage, available = failure["stage"], failure["available"]
    required_state_by_stage = {
        "sampling": "prompt_queue", "reward": "trajectories",
        "update": "pre_update_checkpoint", "artifact_validation": "verified_baseline",
    }
    recovery_action_by_stage = {
        "sampling": "retry_sampling_from_prompt_queue", "reward": "resume_reward_from_saved_trajectories",
        "update": "restore_pre_update_checkpoint", "artifact_validation": "rollback_to_verified_baseline",
    }
    fallback_action_by_stage = {
        "sampling": "rebuild_prompt_queue", "reward": "restart_sampling",
        "update": "rollback_to_verified_baseline", "artifact_validation": "block_release",
    }
    if stage not in required_state_by_stage:
        raise ValueError(f"未知失败阶段：{stage}")
    required_state = required_state_by_stage[stage]
    reuse_trusted_state = bool(available.get(required_state))
    action = recovery_action_by_stage[stage] if reuse_trusted_state else fallback_action_by_stage[stage]
    return {"stage": stage, "required_state": required_state, "reused_state": reuse_trusted_state, "action": action}

def decide_lifecycle_handoff_for_run(upstream_decision, freshness, evidence):
    hard_checks = {
        "freshness": freshness["fresh"],
        "rollback": evidence["rollback_ready"],
        "artifact": evidence["artifact_verified"],
    }
    hard_failures = [name for name, passed in hard_checks.items() if not passed]
    if upstream_decision not in {"accept", "tune", "reject"}:
        hard_failures.append("upstream_state")
    elif upstream_decision == "reject":
        hard_failures.append("upstream_reject")
    handoff_status = "reject" if hard_failures else "tune" if upstream_decision == "tune" else "ready_for_evaluation"
    return {"handoff_status": handoff_status, "upstream_decision": upstream_decision, "hard_failures": hard_failures}

FAILURE_SCENARIOS = [
    {"stage": "sampling", "available": {"prompt_queue": True}},
    {"stage": "reward", "available": {"trajectories": True}},
    {"stage": "update", "available": {"pre_update_checkpoint": True}},
    {"stage": "artifact_validation", "available": {"verified_baseline": True}},
]
print([choose_recovery_action_for_run(scenario) for scenario in FAILURE_SCENARIOS])
```

### 5.4 执行真实候选清单审核与恢复演练

真实审核重新计算候选与回滚产物哈希，并核对候选是否仍来自当前 SFT 基线。它不比较 DPO、GRPO 与 SFT 的质量或资源指标；这些分支使用不同训练负载，统一比较留给 Part 05 的冻结评测。

```python
if RUN_MODE == "real_artifact":
    active_policy_version = sft_manifest["artifacts"]["adapter"]["sha256"]
    candidate_records = []
    for path in available_candidates:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        candidate["window_age"] = ACTIVE_WINDOW_INDEX - candidate.get("evidence_window_index", -1)
        freshness = check_candidate_freshness_for_run(candidate, active_policy_version, MAXIMUM_WINDOW_AGE)
        candidate_adapter = candidate.get("artifacts", {}).get("adapter", {})
        candidate_artifact_path = Path(candidate_adapter.get("path", ""))
        expected_candidate_hash = candidate_adapter.get("sha256")
        actual_candidate_hash = hash_directory(candidate_artifact_path) if candidate_artifact_path.is_dir() else None
        baseline_adapter = sft_manifest["artifacts"]["adapter"]
        baseline_artifact_path = Path(baseline_adapter["path"])
        expected_baseline_hash = baseline_adapter.get("sha256")
        actual_baseline_hash = hash_directory(baseline_artifact_path) if baseline_artifact_path.is_dir() else None
        evidence = {
            "rollback_ready": bool(expected_baseline_hash) and actual_baseline_hash == expected_baseline_hash,
            "artifact_verified": bool(expected_candidate_hash) and actual_candidate_hash == expected_candidate_hash,
            "artifact_check": {"expected": expected_candidate_hash, "actual": actual_candidate_hash},
            "rollback_check": {"expected": expected_baseline_hash, "actual": actual_baseline_hash},
        }
        upstream_decision = candidate.get("decision")
        handoff = decide_lifecycle_handoff_for_run(upstream_decision, freshness, evidence)
        candidate_records.append({"path": str(path), "semantic_id": candidate["semantic_id"], "upstream_decision": upstream_decision, "freshness": freshness, "artifact": candidate_adapter, "evidence": evidence, **handoff})
    recovery = [choose_recovery_action_for_run(scenario) for scenario in FAILURE_SCENARIOS]
    record = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "run_mode": RUN_MODE, "run_label": RUN_LABEL,
        "active_policy_version": active_policy_version,
        "candidate_records": candidate_records, "recovery_scenarios": recovery,
        "missing_candidate_manifests": [str(path) for path in CANDIDATE_MANIFESTS if not path.exists()],
    }
    save_json_record(EVIDENCE_PATH, record)
    print({"evidence_path": str(EVIDENCE_PATH), "candidate_count": len(candidate_records)})
```

### 5.5 独立读取生命周期证据

本节只列出可进入独立评测的候选，不在 DPO 偏好准确率与 GRPO `pass@group` 之间强行排序。多个候选通过时全部交给 Part 05，在同一任务与解码协议下复测。

```python
if EVIDENCE_PATH.exists():
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    ready = [item for item in record["candidate_records"] if item["handoff_status"] == "ready_for_evaluation"]
    print({"ready_for_evaluation": [item["semantic_id"] for item in ready], "candidate_status": {item["semantic_id"]: item["handoff_status"] for item in record["candidate_records"]}})
```

### 5.6 保存生命周期交接清单

交接清单引用候选清单和已验证基线，只保存 Part 05 所需的身份、产物、恢复与回滚证据。

| 候选或恢复项 | 实测证据 | 结果 | 后续动作 |
|:---|:---|:---|:---|
| DPO 候选 | 84 v5 为 `accept`；血缘、产物哈希与回滚基线通过 | `ready_for_evaluation` | 交给 Part 05 冻结评测 |
| GRPO / RLVR 候选 | 85 v7 为 `accept`；验证器、窗口、产物哈希与回滚基线通过 | `ready_for_evaluation` | 交给 Part 05 冻结评测 |
| 故障恢复 | 采样、奖励、更新和产物校验均可复用可信状态 | 通过 | 保留四类恢复记录 |
| Part 05 交接 | 两个候选均通过生命周期核对 | `ready` | 在统一任务、解码协议与成本口径下比较 |

`post_training_lifecycle_v6` 交接 84 的 DPO v5 与 85 的 GRPO / RLVR v7。两个候选都来自当前 SFT 基线，分支证据、适配器哈希和 SFT 回滚基线均通过核对；GRPO / RLVR 的证据窗口与当前窗口一致。交接清单状态为 `ready`，表示两个候选可以进入 Part 05 的统一冻结评测，而不是已经比较出算法优劣或完成最终采用。质量、安全、鲁棒性和成本仍由 Part 05 决定。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 86 节后训练生命周期交接；`post_training_lifecycle_v6` |
| 运行条件 | 当前 SFT 基线；84 DPO v5 与 85 GRPO / RLVR v7 并列候选 |
| 原始证据 | 由生命周期证据引用两条候选分支、恢复演练和回滚基线 |
| 正式证据 | [生命周期证据](../../benchmarks/results/86_post_training_lifecycle/lifecycle_evidence_post_training_lifecycle_v6.json) |
| 清单与交接 | [生命周期交接](../../benchmarks/results/86_post_training_lifecycle/lifecycle_handoff_post_training_lifecycle_v6.json) |
| 结论 | `ready`：允许进入 Part 05 冻结评测；尚未完成 DPO 与 GRPO / RLVR 的最终采用比较 |

```python
if RUN_MODE == "real_artifact":
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    ready = [item for item in record["candidate_records"] if item["handoff_status"] == "ready_for_evaluation"]
    handoff_record = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "run_label": RUN_LABEL,
        "handoff_status": "ready" if ready else "no_ready_candidate",
        "baseline_manifest": str(SFT_MANIFEST),
        "candidate_records": record["candidate_records"],
        "source_evidence": str(EVIDENCE_PATH),
        "recovery_scenarios": record["recovery_scenarios"],
    }
    save_json_record(HANDOFF_PATH, handoff_record)
    print({"handoff_path": str(HANDOFF_PATH), "handoff_status": handoff_record["handoff_status"], "ready_count": len(ready)})
```

## 相关阅读

- [Part 02 · PT-RL-SYSTEM Rollout、训练与奖励系统](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)
- [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹生命周期与策略新鲜度](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)
- [Part 02 · 51 在线 DPO 与反馈窗口](./51_Online_DPO.md)
