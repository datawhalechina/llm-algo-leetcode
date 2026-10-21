# 85. GRPO Groupwise Alignment Project | GRPO 组内奖励优化项目

**难度：** Hard | **环境：** CPU + 单卡 GPU | **标签：** `RLVR`, `GRPO`, `真实轨迹采样` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节与 84 节并列读取 62 节的 SFT 基线。模型对同一可验证问题生成一组回答，规则验证器产生奖励，只有奖励存在组内差异时才形成有效优势并更新。项目比较更新前后的平均奖励与组内至少一次通过率，同时记录有效组比例、轨迹采样与更新耗时以及适配器产物。

**关键词：** `RLVR`、`GRPO`、`轨迹采样`、`组内优势`、`有效更新吞吐`

---
## 前置阅读

- [Part 02 · 18 GRPO、RLVR 与组内优势](./18_GRPO_Loss_Tutorial.md)
- [Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md)
- [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹生命周期与策略新鲜度](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)
- [Part 02 · 62 SFT 训练交付项目](./62_Instruction_Fine_Tuning_Project.md)

---
### Step 1：从可验证问题完成一次在线奖励优化交付

项目从已交付的 SFT 基线、可验证问题集和固定验证器开始。候选策略先为每个训练问题生成一组真实回答，验证器返回奖励，奖励存在组内差异的轨迹才进入更新。完成更新后，候选必须在隔离问题上重新生成一批回答；训练轨迹只说明更新发生过，不能兼任独立质量证据。最终交付包括适配器、策略与验证器身份、训练轨迹摘要和新轨迹评测。

| 阶段 | 输入 | 关键动作 | 输出或失败处理 |
|:---|:---|:---|:---|
| 输入准入 | SFT 基线、训练/评测问题和验证器 | 核对问题隔离、基线与验证器身份 | 通过后固定输入身份；失败则 `reject` |
| 训练采样 | 固定策略版本和组规模 | 生成带问题、策略和验证器身份的轨迹组 | 输出完整轨迹组；身份或完整性失败则 `reject` |
| 奖励与更新 | 真实轨迹和验证器奖励 | 识别奖励可区分的组并执行参数更新 | 输出有效组和更新记录；无有效更新则 `reject` |
| 独立复测 | 更新前策略、候选和隔离问题 | 在固定随机条件下重新生成并配对比较 | 输出新轨迹质量；未达门槛则 `tune` |
| 产物交付 | 候选、资源记录和复测结果 | 核对适配器、哈希、回滚来源和资源 | 输出 GRPO/RLVR 候选清单；产物缺失则 `reject` |

![GRPO 真实轨迹生命周期](../public/02_PyTorch_Algorithms/85_grpo_trajectory_flow.svg)

### Step 2：把采样成本换算成有效更新量

生成完成的轨迹不一定都能产生更新。只有问题身份完整、来自当前策略、通过验证且组内奖励可区分的完整组，才计入有效组；有效轨迹数等于有效组数乘以固定组规模。再用有效轨迹数除以采样、验证和更新的总耗时，得到有效轨迹吞吐。这个量把昂贵的生成成本与真正进入更新的训练信号联系起来，比原始生成吞吐更适合判断项目是否值得继续。

| 项目量 | 计算或记录方式 | 需要防止的误判 |
|:---|:---|:---|
| 总轨迹数 | 问题数乘以固定组规模 | 只代表已支付的生成与验证成本 |
| 有效组比例 | 奖励可区分的完整组数除以全部组数 | 不能把零方差组计入更新覆盖率 |
| 有效轨迹数 | 有效组数乘以组规模 | 不能用总轨迹数替代 |
| 有效轨迹吞吐 | 有效轨迹数除以采样、验证和更新总耗时 | 不能只计算生成阶段耗时 |
| 最大长度命中率 | 达到生成上限的回答数除以总回答数 | 高比例说明奖励可能基于截断回答 |
| 更新约束记录 | 概率比限制、参考约束和实际更新次数 | 奖励提高不能证明更新幅度安全 |

### Step 3：用独立新轨迹验证策略变化

训练轨迹既参与了优势计算又参与了参数更新，不能继续充当独立质量证据。候选模型必须在固定评测问题和随机条件下重新采样，与更新前策略配对比较平均奖励、组内至少一次通过率和长度分布。资源证据则分开记录采样、验证和更新阶段，避免把较高的生成吞吐误认为较低的训练成本。

| 证据层 | 需要回答的问题 | 失败后的状态 |
|:---|:---|:---|
| 轨迹契约 | 问题、策略、验证器和轨迹身份是否一致 | `reject` |
| 有效更新 | 是否存在有效组并实际完成参数更新 | `reject` |
| 独立新轨迹 | 平均奖励与组内通过率是否改善 | `tune` |
| 长度与资源 | 截断、显存及各阶段耗时是否通过 | `tune` |
| 适配器产物 | 路径、哈希和源策略版本是否完整 | `reject` |
| 全部证据 | 上述检查同时成立 | `accept` |

![GRPO 候选的独立评测与交付](../public/02_PyTorch_Algorithms/85_grpo_delivery_gate.svg)

### Step 4：实现轨迹准入、候选门槛和交付决策

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：检查实际轨迹契约 | 每组的问题、轨迹、策略与验证器身份 | 真实生成记录同源、完整且不重复 | 旧策略混入、空问题、重复轨迹与组不完整 |
| TODO 2：由有效组形成更新与质量门槛 | 组大小、有效组、阶段耗时、更新次数、奖励、截断和资源 | 从非零方差组计算有效轨迹吞吐，再检查质量与资源 | 全零奖励、虚假高吞吐、截断过高与显存越界 |
| TODO 3：区分拒绝、调优与交付 | 轨迹、候选和适配器证据 | 轨迹或产物失败时拒绝，候选未达标时调优 | 缺少适配器、轨迹失效与质量不足 |

```python
from typing import Any, Dict, List
```


```python
# ==========================================
# TODO 1：检查轨迹组的身份与完整性
# ==========================================
def check_rollout_contract(groups: List[Dict[str, Any]], expected_policy_version: str, expected_verifier_version: str, expected_group_size: int) -> Dict[str, Any]:
    """检查实际轨迹的任务身份、版本、组大小和唯一性。"""
    trace_ids = [trace_id for group in groups for trace_id in group["trace_ids"]]
    # checks = ???  # 分别检查问题身份、策略版本、验证器版本、组大小和轨迹唯一性。
    # admitted = ???  # 非空轨迹组必须通过全部检查。
    raise NotImplementedError("TODO 1：请构造轨迹组准入检查并形成结论")
    return {"admitted": admitted, "failed_checks": [name for name, passed in checks.items() if not passed]}


# ==========================================
# TODO 2：检查 GRPO 候选的质量、有效更新与资源门槛
# ==========================================
def evaluate_grpo_candidate(evidence: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """使用有效更新、新轨迹质量、截断和资源证据判断候选资格。"""
    rollout = evidence["rollout"]
    update = evidence["update"]
    elapsed_seconds = rollout["rollout_seconds"] + update["update_seconds"]
    reward_groups = rollout["reward_groups"]
    group_size = rollout["group_size"]
    # complete_groups = ???  # 只保留奖励数量等于固定组规模的组。
    # valid_groups = ???  # 只保留组内奖励存在差异、能够形成相对优势的完整组。
    # valid_group_ratio = ???
    # effective_trajectory_count = ???
    # effective_trajectories_per_second = ???  # 分母覆盖采样、验证和更新。
    raise NotImplementedError("TODO 2：请从逐组奖励形成有效更新与吞吐证据")
    checks = {
        "execution": evidence["status"] == "ok",
        "effective_groups": valid_group_ratio >= gates["minimum_valid_group_ratio"],
        "effective_throughput": effective_trajectories_per_second >= gates["minimum_effective_trajectories_per_second"],
        "updates": update["update_count"] >= gates["minimum_update_count"],
        "reward": evidence["comparison"]["reward_mean_gain"] >= gates["minimum_reward_gain"],
        "pass_at_group": evidence["comparison"]["pass_at_group_gain"] >= gates["minimum_pass_at_group_gain"],
        "truncation": rollout["max_length_hit_ratio"] <= gates["maximum_max_length_hit_ratio"],
        "memory": evidence["resources"]["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
    }
    eligible = all(checks.values())
    return {"eligible": eligible, "valid_group_ratio": valid_group_ratio, "effective_trajectory_count": effective_trajectory_count, "effective_trajectories_per_second": effective_trajectories_per_second, "failed_checks": [name for name, passed in checks.items() if not passed]}


# ==========================================
# TODO 3：形成 GRPO 项目交付决策
# ==========================================
def decide_grpo_delivery(rollout: Dict[str, Any], candidate: Dict[str, Any], artifact: Dict[str, Any]) -> Dict[str, Any]:
    """依据轨迹、候选和适配器证据形成三态交付决策。"""
    checks = {
        "rollout": rollout["admitted"],
        "candidate": candidate["eligible"],
        "artifact": bool(artifact.get("path")) and bool(artifact.get("sha256")),
    }
    # 轨迹或产物失败时拒绝，候选未达标时调优，其余情况才可交付。
    # decision = ???  # 含义：reject、tune 或 accept
    raise NotImplementedError("TODO 3：请形成 GRPO 交付决策")
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}
```


```python
def _grpo_candidate_fixture():
    """构造候选检查共用的证据与门槛。"""
    gates = {
        "minimum_valid_group_ratio": 0.25,
        "minimum_update_count": 1,
        "minimum_effective_trajectories_per_second": 1.0,
        "minimum_reward_gain": 0.0,
        "minimum_pass_at_group_gain": 0.0,
        "maximum_peak_allocated_bytes": 10_000,
        "maximum_max_length_hit_ratio": 0.10,
    }
    evidence = {
        "status": "ok",
        "rollout": {"group_size": 4, "reward_groups": [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], "rollout_seconds": 4.0, "max_length_hit_ratio": 0.05},
        "update": {"update_count": 2, "update_seconds": 2.0},
        "comparison": {"reward_mean_gain": 0.1, "pass_at_group_gain": 0.1},
        "resources": {"peak_allocated_bytes": 8_000},
    }
    return evidence, gates


def test_rollout_contract():
    """TODO 1：验证轨迹身份、来源版本、组大小和唯一性。"""
    groups = [
        {"prompt_id": "math-1", "source_policy_version": "p1", "verifier_version": "v1", "trace_ids": ["a1", "a2", "a3", "a4"]},
        {"prompt_id": "math-2", "source_policy_version": "p1", "verifier_version": "v1", "trace_ids": ["b1", "b2", "b3", "b4"]},
    ]
    rollout = check_rollout_contract(groups, "p1", "v1", 4)
    assert rollout["admitted"]
    assert not check_rollout_contract([{**groups[0], "source_policy_version": "p0"}], "p1", "v1", 4)["admitted"]
    assert not check_rollout_contract([{**groups[0], "verifier_version": "v0"}], "p1", "v1", 4)["admitted"]
    duplicate = [groups[0], {**groups[1], "trace_ids": ["a1", "b2", "b3", "b4"]}]
    assert check_rollout_contract(duplicate, "p1", "v1", 4)["failed_checks"] == ["unique_traces"]


def test_grpo_candidate_gates():
    """TODO 2：验证有效更新、截断与资源门槛。"""
    evidence, gates = _grpo_candidate_fixture()
    candidate = evaluate_grpo_candidate(evidence, gates)
    assert candidate["eligible"]
    no_effective_update = {**evidence, "status": "no_effective_update", "rollout": {**evidence["rollout"], "reward_groups": [[0.0] * 4 for _ in range(4)]}, "update": {**evidence["update"], "update_count": 0}}
    assert not evaluate_grpo_candidate(no_effective_update, gates)["eligible"]
    incomplete_groups = {**evidence, "rollout": {**evidence["rollout"], "reward_groups": [[0.0, 1.0], [1.0, 1.0]]}}
    assert evaluate_grpo_candidate(incomplete_groups, gates)["valid_group_ratio"] == 0.0
    assert not evaluate_grpo_candidate({**evidence, "rollout": {**evidence["rollout"], "max_length_hit_ratio": 0.50}}, gates)["eligible"]
    assert not evaluate_grpo_candidate({**evidence, "resources": {"peak_allocated_bytes": 20_000}}, gates)["eligible"]


def test_grpo_delivery_decision():
    """TODO 3：验证拒绝、调优和交付三种结果。"""
    evidence, gates = _grpo_candidate_fixture()
    rollout = {"admitted": True}
    candidate = evaluate_grpo_candidate(evidence, gates)
    assert decide_grpo_delivery(rollout, candidate, {"path": "adapter", "sha256": "abc"})["decision"] == "accept"
    assert decide_grpo_delivery(rollout, candidate, {})["decision"] == "reject"
    assert decide_grpo_delivery(rollout, {"eligible": False}, {"path": "adapter", "sha256": "abc"})["decision"] == "tune"


test_rollout_contract()
test_grpo_candidate_gates()
test_grpo_delivery_decision()
print("所有测试通过！")
```

🛑 **STOP HERE** 🛑
## 参考代码与解析

### 代码

```python
# TODO 1：检查轨迹组的身份与完整性
def check_rollout_contract(groups: List[Dict[str, Any]], expected_policy_version: str, expected_verifier_version: str, expected_group_size: int) -> Dict[str, Any]:
    """检查实际轨迹的任务身份、版本、组大小和唯一性。"""
    trace_ids = [trace_id for group in groups for trace_id in group["trace_ids"]]
    checks = {
        "prompt_identity": all(bool(group.get("prompt_id")) for group in groups),
        "policy_version": all(group["source_policy_version"] == expected_policy_version for group in groups),
        "verifier_version": all(group["verifier_version"] == expected_verifier_version for group in groups),
        "group_size": all(len(group["trace_ids"]) == expected_group_size for group in groups),
        "unique_traces": len(trace_ids) == len(set(trace_ids)),
    }
    admitted = bool(groups) and all(checks.values())
    return {"admitted": admitted, "failed_checks": [name for name, passed in checks.items() if not passed]}


# TODO 2：检查 GRPO 候选的质量、有效更新与资源门槛
def evaluate_grpo_candidate(evidence: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """检查有效更新、新轨迹质量、截断和资源证据。"""
    rollout = evidence["rollout"]
    update = evidence["update"]
    elapsed_seconds = rollout["rollout_seconds"] + update["update_seconds"]
    reward_groups = rollout["reward_groups"]
    group_size = rollout["group_size"]
    complete_groups = [rewards for rewards in reward_groups if len(rewards) == group_size]
    valid_groups = [rewards for rewards in complete_groups if max(rewards) > min(rewards)]
    valid_group_ratio = len(valid_groups) / len(reward_groups) if reward_groups else 0.0
    effective_trajectory_count = len(valid_groups) * group_size
    effective_trajectories_per_second = effective_trajectory_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
    checks = {
        "execution": evidence["status"] == "ok",
        "effective_groups": valid_group_ratio >= gates["minimum_valid_group_ratio"],
        "effective_throughput": effective_trajectories_per_second >= gates["minimum_effective_trajectories_per_second"],
        "updates": update["update_count"] >= gates["minimum_update_count"],
        "reward": evidence["comparison"]["reward_mean_gain"] >= gates["minimum_reward_gain"],
        "pass_at_group": evidence["comparison"]["pass_at_group_gain"] >= gates["minimum_pass_at_group_gain"],
        "truncation": rollout["max_length_hit_ratio"] <= gates["maximum_max_length_hit_ratio"],
        "memory": evidence["resources"]["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
    }
    eligible = all(checks.values())
    return {"eligible": eligible, "valid_group_ratio": valid_group_ratio, "effective_trajectory_count": effective_trajectory_count, "effective_trajectories_per_second": effective_trajectories_per_second, "failed_checks": [name for name, passed in checks.items() if not passed]}


# TODO 3：形成 GRPO 项目交付决策
def decide_grpo_delivery(rollout: Dict[str, Any], candidate: Dict[str, Any], artifact: Dict[str, Any]) -> Dict[str, Any]:
    """区分轨迹或产物失败、候选待调优和可以交付。"""
    checks = {
        "rollout": rollout["admitted"],
        "candidate": candidate["eligible"],
        "artifact": bool(artifact.get("path")) and bool(artifact.get("sha256")),
    }
    if not checks["rollout"] or not checks["artifact"]:
        decision = "reject"
    elif not checks["candidate"]:
        decision = "tune"
    else:
        decision = "accept"
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}
```

### 解析

#### TODO 1：检查轨迹采样契约

- **判断依据**：组内优势只能比较有明确问题身份、同一策略、同一验证器和完整组的真实轨迹；重复轨迹会重复计权。
- **实现动作**：由问题身份、策略版本、验证器版本、组大小和轨迹唯一性构造具名检查，再形成准入结果。
- **测试证据**：测试覆盖合法组、旧策略、旧验证器和重复轨迹。

#### TODO 2：检查候选门槛

- **判断依据**：原始生成吞吐不代表有效更新；零方差组必须从有效组和有效轨迹吞吐中排除。
- **实现动作**：从逐组奖励中先排除不完整组与零方差组，再计算有效组比例、有效轨迹数和覆盖采样、验证、更新的吞吐，最后检查独立质量、截断与显存。
- **测试证据**：测试覆盖零方差组、不完整组、长度超限和显存越界；完成生成但没有相对奖励信号时不能通过。

#### TODO 3：形成交付决策

- **判断依据**：轨迹契约、候选收益和可追溯适配器是独立硬门槛。
- **实现动作**：轨迹或产物失败时 `reject`，候选门槛未通过时 `tune`，三项完整时 `accept`。
- **测试证据**：测试覆盖缺少产物与候选仍需调优两类路径。
## Step 5：真实轨迹采样、GRPO 更新与新轨迹复测

CPU 题目区验证轨迹身份、有效组和候选决策；真实实验执行候选生成、验证器奖励、有限 GRPO 更新和独立新轨迹复测。

| 对照层 | 输入 | 主要证据 | 输出与证据边界 |
|:---|:---|:---|:---|
| CPU 生命周期 | 人工构造的轨迹组和候选摘要 | 策略版本、新鲜度、有效组与采用状态 | 只确认轨迹与决策逻辑，不证明真实质量 |
| 真实 GRPO / RLVR | 固定 SFT 策略、题集、验证器和生成预算 | 实际轨迹身份、奖励方差、更新、复测、截断与资源 | 输出候选清单；不外推到其他任务 |

### 5.1 环境、输入资产与固定条件

按[后训练优化 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-rl-cuda-quickstart)准备 `PT-RL-CUDA` 环境。12 GB 单卡保持 Qwen2.5-0.5B、每组 8 条轨迹、24 道训练题和 12 道独立评测题不变；v7 只将生成上限从 256 提高到 384 词元，并使用新的轨迹窗口。v6 的峰值分配显存约 6.49 GiB、峰值保留显存约 10.10 GiB，故本轮需重新记录实际峰值与触顶率。

```python
from pathlib import Path
import json

from tools.project_runtime import resolve_project_root

RUN_MODE = "dry_run"  # 默认只检查配置；改为 real_gpu 执行新的真实轨迹与更新。
RUN_LABEL = "grpo_rlvr_len384_v7"  # v6 的生成触顶率超限；v7 只扩大生成预算。
PROFILE_ID = "PT-RL-CUDA"
SEMANTIC_ID = "PT-GRPO-GROUPWISE"
SCHEMA_VERSION = "model-evaluation/v1"

PROJECT_ROOT = resolve_project_root()
SFT_MANIFEST = PROJECT_ROOT / "benchmarks/results/62_sft_delivery/sft_delivery_manifest_sft_delivery_v2_1.json"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "main"
TRAIN_PROMPTS = 24  # 扩大更新窗口，避免单个有效组支配结论。
EVALUATION_PROMPTS = 12  # 降低小评测集造成的 pass@group 跳变。
GROUP_SIZE = 8  # v1 的 group=4 仅产生 1/6 个有效更新组。
MAX_PROMPT_LENGTH = 192
MAX_NEW_TOKENS = 384  # v6 的基线触顶率为 51.04%；本轮只扩大生成预算。
CLIP_EPSILON = 0.2
KL_COEFFICIENT = 0.02
LEARNING_RATE = 1e-5
DTYPE = "bfloat16"
SEED = 42
EVIDENCE_WINDOW_INDEX = 1  # 新生成的轨迹窗口；不复用 v6 的在线采样证据。
GATES = {
    "minimum_valid_group_ratio": 0.20,
    "minimum_update_count": 1,
    "minimum_effective_trajectories_per_second": 1.0,
    "minimum_reward_gain": 0.0,
    "minimum_pass_at_group_gain": 0.0,
    "maximum_max_length_hit_ratio": 0.50,
    "maximum_peak_allocated_bytes": 10 * 1024**3,
}

RESULTS_DIR = PROJECT_ROOT / "benchmarks/results/85_grpo_alignment"
EVIDENCE_PATH = RESULTS_DIR / f"grpo_evidence_{RUN_LABEL}.json"
MANIFEST_PATH = RESULTS_DIR / f"grpo_manifest_{RUN_LABEL}.json"
ARTIFACT_DIR = RESULTS_DIR / f"adapter_{RUN_LABEL}"
print({"run_mode": RUN_MODE, "group_size": GROUP_SIZE, "evidence_path": str(EVIDENCE_PATH)})
```

### 5.2 预检 SFT 基线、环境与输出路径

```python
import torch
from tools.project_runtime import environment_preflight, require_input_file
from tools.post_training_project_runtime import run_grpo_delivery, save_json_record, stable_text_hash

preflight = environment_preflight(
    torch, required_packages=("transformers", "peft", "datasets", "huggingface_hub"),
    require_gpu=RUN_MODE == "real_gpu", output_path=EVIDENCE_PATH,
    environment_profile=PROFILE_ID,
)
if not preflight["ready"]:
    raise RuntimeError(preflight["reasons"])
sft_path = require_input_file(SFT_MANIFEST, "62 节 SFT manifest")
sft_manifest = json.loads(sft_path.read_text(encoding="utf-8"))
if sft_manifest["decision"] != "accept":
    raise RuntimeError("SFT baseline 未通过交付")
for path in (EVIDENCE_PATH, MANIFEST_PATH, ARTIFACT_DIR):
    if RUN_MODE == "real_gpu" and path.exists():
        raise FileExistsError(f"输出已存在，请修改 RUN_LABEL：{path}")
print({"preflight": preflight["status"], "sft_policy_version": sft_manifest["artifacts"]["adapter"]["sha256"]})
```

### 5.3 固定问题集、策略版本与验证器

训练和评测来自 GSM8K 的不同数据划分；规则验证器提取回答中的最后一个数值，与标准答案 `####` 后的数值比较。

```python
def check_rollout_contract_for_run(groups, expected_policy_version, expected_verifier_version, expected_group_size):
    """核对实际生成轨迹的身份、版本和组大小。"""
    trace_ids = [trace_id for group in groups for trace_id in group["trace_ids"]]
    checks = {
        "policy_version": all(group["source_policy_version"] == expected_policy_version for group in groups),
        "verifier_version": all(group["verifier_version"] == expected_verifier_version for group in groups),
        "group_size": all(len(group["trace_ids"]) == expected_group_size for group in groups),
        "prompt_identity": all(bool(group.get("prompt_id")) for group in groups),
        "unique_traces": len(trace_ids) == len(set(trace_ids)),
    }
    return {"admitted": bool(groups) and all(checks.values()), "failed_checks": [name for name, passed in checks.items() if not passed]}

if RUN_MODE == "real_gpu":
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset_commit = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION).sha
    train_tasks = [dict(row) for row in load_dataset(DATASET_ID, DATASET_CONFIG, split=f"train[:{TRAIN_PROMPTS}]", revision=dataset_commit)]
    evaluation_tasks = [dict(row) for row in load_dataset(DATASET_ID, DATASET_CONFIG, split=f"test[:{EVALUATION_PROMPTS}]", revision=dataset_commit)]
    policy_version = sft_manifest["artifacts"]["adapter"]["sha256"]
    verifier_version = "gsm8k-final-number-v1"
    train_prompt_ids = [stable_text_hash((row["question"],)) for row in train_tasks]
    evaluation_prompt_ids = [stable_text_hash((row["question"],)) for row in evaluation_tasks]
    if set(train_prompt_ids) & set(evaluation_prompt_ids):
        raise RuntimeError("训练题与独立评测题发生 prompt 身份重合")
    print({"dataset_commit": dataset_commit, "train_prompts": len(train_prompt_ids), "evaluation_prompts": len(evaluation_prompt_ids), "prompt_overlap": 0})
```

### 5.4 执行真实轨迹采样、验证器奖励和有限更新

零方差组会保存为成本证据，但不会进入优势更新；更新后使用评测问题重新生成，不复用训练轨迹。实际轨迹必须携带任务输入 ID、轨迹 ID、源策略版本和验证器版本。v5 有 12/24 个训练组产生非零方差，但基线与候选分别有 73.96% 和 72.92% 的回答触及 128 词元上限；本轮只扩大生成预算，检验质量持平是否由截断造成。

```python
if RUN_MODE == "real_gpu":
    from transformers import AutoTokenizer

    model_id = sft_manifest["model"]["id"]
    model_commit = sft_manifest["model"]["resolved_commit"]
    sft_adapter = sft_manifest["artifacts"]["adapter"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_commit)
    run_evidence = run_grpo_delivery(
        model_id=model_id, revision=model_commit, sft_adapter_path=sft_adapter,
        tokenizer=tokenizer, train_tasks=train_tasks, evaluation_tasks=evaluation_tasks,
        group_size=GROUP_SIZE, max_prompt_length=MAX_PROMPT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS, clip_epsilon=CLIP_EPSILON,
        kl_coefficient=KL_COEFFICIENT, learning_rate=LEARNING_RATE,
        seed=SEED, dtype_name=DTYPE, artifact_directory=ARTIFACT_DIR,
        source_policy_version=policy_version, verifier_version=verifier_version,
    )
    rollout_contract = check_rollout_contract_for_run(
        run_evidence["rollout"]["trajectory_records"], policy_version, verifier_version, GROUP_SIZE
    )
    if not rollout_contract["admitted"]:
        raise RuntimeError(f"实际轨迹契约失败：{rollout_contract['failed_checks']}")
    record = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "environment_profile": PROFILE_ID, "run_mode": RUN_MODE, "run_label": RUN_LABEL,
        "sft_manifest": str(SFT_MANIFEST), "policy_version": policy_version,
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "resolved_commit": dataset_commit},
        "verifier_version": verifier_version, "rollout_contract": rollout_contract,
        "workload": {"train_prompts": TRAIN_PROMPTS, "evaluation_prompts": EVALUATION_PROMPTS, "group_size": GROUP_SIZE, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED},
        "gates": GATES, "evidence": run_evidence,
    }
    save_json_record(EVIDENCE_PATH, record)
    print({"evidence_path": str(EVIDENCE_PATH), "status": run_evidence["status"], "comparison": run_evidence["comparison"]})
else:
    print({"run_mode": RUN_MODE, "next_action": "确认问题集和门槛后改为 real_gpu"})
```

### 5.5 独立读取结果并判断候选

```python
def decide_grpo_candidate_for_run(evidence, gates, rollout_contract):
    """以轨迹硬契约和候选质量、截断、资源门槛形成三级决策。"""
    checks = {
        "rollout": rollout_contract["admitted"],
        "execution": evidence["status"] == "ok",
        "effective_groups": evidence["rollout"]["valid_group_ratio"] >= gates["minimum_valid_group_ratio"],
        "updates": evidence["update"]["update_count"] >= gates["minimum_update_count"],
        "effective_throughput": evidence["update"]["effective_trajectories_per_second"] >= gates["minimum_effective_trajectories_per_second"],
        "reward": evidence["comparison"]["reward_mean_gain"] >= gates["minimum_reward_gain"],
        "pass_at_group": evidence["comparison"]["pass_at_group_gain"] >= gates["minimum_pass_at_group_gain"],
        "length_budget": max(evidence["baseline"]["max_length_hit_ratio"], evidence["candidate"]["max_length_hit_ratio"]) <= gates["maximum_max_length_hit_ratio"],
        "memory": evidence["resources"]["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
        "artifact": bool(evidence.get("artifact", {}).get("path")) and bool(evidence.get("artifact", {}).get("sha256")),
    }
    if not checks["rollout"] or not checks["execution"] or not checks["artifact"]:
        decision = "reject"
    elif not all(checks.values()):
        decision = "tune"
    else:
        decision = "accept"
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}

if EVIDENCE_PATH.exists():
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    decision = decide_grpo_candidate_for_run(record["evidence"], record["gates"], record["rollout_contract"])
    print({"decision": decision, "baseline": record["evidence"]["baseline"], "candidate": record["evidence"]["candidate"], "rollout": {key: value for key, value in record["evidence"]["rollout"].items() if key != "trajectory_records"}, "update": record["evidence"]["update"]})
```

### 5.6 形成 GRPO / RLVR 候选清单

该候选清单与 84 的 DPO 候选清单并列交给 86；若没有有效组，则保留失败证据并输出 `reject`，不能把完成生成等同于完成策略更新。

| 决策项 | 实测证据 | 结果 | 后续动作 |
|:---|:---|:---|:---|
| 有效更新 | 24 个组中 18 个产生有效优势，完成 18 次更新 | 通过 | 保留轨迹身份与更新记录 |
| 独立复测 | 平均奖励 0.2708 → 0.3229；`pass@group` 0.5833 → 0.6667 | 通过 | 保留独立新轨迹结果 |
| 长度预算 | 更新前策略与候选的触顶率为 11.46% / 18.75% | 通过 | 固定 384 词元上限 |
| 项目决策 | 轨迹、资源、独立复测和适配器证据完整 | `accept` | 交给 86 进行生命周期交接 |

`grpo_rlvr_len384_v7` 保持题集、组规模、策略与验证器不变，只把生成上限提高到 384，并使用新的轨迹窗口。平均奖励从 0.2708 提高到 0.3229，`pass@group` 从 0.5833 提高到 0.6667；24 个组中 18 个形成有效优势并完成更新。基线与候选触顶率分别为 11.46% 与 18.75%，峰值分配显存约 7.00 GiB、峰值保留显存约 9.50 GiB，全部门槛通过，结论为 `accept`。该结果仅适用于当前题集、组规模、策略版本与 384 词元预算。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 85 节 GRPO / RLVR；`grpo_rlvr_len384_v7` |
| 运行条件 | 当前 SFT 基线；固定题集、组规模、策略与验证器；生成上限 384 |
| 原始证据 | [GRPO 证据](../../benchmarks/results/85_grpo_alignment/grpo_evidence_grpo_rlvr_len384_v7.json) |
| 正式证据 | GRPO 证据内保存轨迹、奖励、有效更新、资源和独立复测指标 |
| 清单与交接 | [候选清单](../../benchmarks/results/85_grpo_alignment/grpo_manifest_grpo_rlvr_len384_v7.json)；本地 adapter 由清单记录 |
| 结论 | `accept`：候选交给 86 节核对生命周期，不表示已完成最终算法采用 |

```python
if RUN_MODE == "real_gpu":
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    decision = decide_grpo_candidate_for_run(record["evidence"], record["gates"], record["rollout_contract"])
    trajectory_records = record["evidence"]["rollout"]["trajectory_records"]
    trace_ids = [trace_id for group in trajectory_records for trace_id in group["trace_ids"]]
    manifest = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "run_label": RUN_LABEL, "decision": decision["decision"],
        "decision_details": decision,
        "source_evidence": str(EVIDENCE_PATH), "sft_manifest": str(SFT_MANIFEST),
        "evidence_window_index": EVIDENCE_WINDOW_INDEX,
        "source_policy_version": sft_manifest["artifacts"]["adapter"]["sha256"],
        "evaluation_version": record["dataset"]["resolved_commit"],
        "policy_version": record["evidence"]["artifact"]["sha256"],
        "quality": {"baseline": record["evidence"]["baseline"], "candidate": record["evidence"]["candidate"], "comparison": record["evidence"]["comparison"]},
        "rollout": {key: value for key, value in record["evidence"]["rollout"].items() if key != "trajectory_records"},
        "rollout_contract": record["rollout_contract"],
        "trajectory_identity": {"group_count": len(trajectory_records), "trace_count": len(trace_ids), "trace_ids_hash": stable_text_hash(trace_ids), "verifier_version": record["verifier_version"]},
        "resources": record["evidence"]["resources"],
        "artifacts": {"adapter": record["evidence"]["artifact"]},
        "next_consumer": "PT-POST-TRAINING-LIFECYCLE",
    }
    save_json_record(MANIFEST_PATH, manifest)
    print({"manifest_path": str(MANIFEST_PATH), "decision": decision["decision"]})
```

## 相关阅读

- [DeepSeekMath：GRPO](https://arxiv.org/abs/2402.03300)
- [TRL GRPO Trainer 文档](https://huggingface.co/docs/trl/grpo_trainer)
- [OpenAI GSM8K 数据集卡](https://huggingface.co/datasets/openai/gsm8k)
- [Part 02 · 86 在线后训练生命周期与恢复](./86_DPO_Online_Benchmark.md)
