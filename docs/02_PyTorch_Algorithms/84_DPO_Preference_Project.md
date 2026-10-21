# 84. DPO Preference Project | DPO 偏好优化项目

**难度：** Hard | **环境：** CPU + 单卡 GPU | **标签：** `离线偏好`, `DPO`, `独立评测` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节从 62 节交付的 SFT 基线出发，在固定偏好数据、冻结参考模型、固定 `beta` 和更新预算下训练一个 DPO 候选。项目结论同时检查训练—评测隔离、独立偏好准确率、边际改善、资源上限和适配器产物。

**关键词：** `DPO`、`偏好对`、`参考模型`、`独立评测`、`适配器`

---
## 前置阅读

- [Part 02 · 17 DPO 损失与固定 Reference](./17_DPO_Loss_Tutorial.md)
- [Part 02 · 50 偏好数据与评测](./50_Preference_Data_and_Evaluation.md)
- [Part 02 · 62 SFT 训练交付项目](./62_Instruction_Fine_Tuning_Project.md)

---
### Step 1：从 SFT 基线完成一次可复核的 DPO 交付

项目从一个已交付的 SFT 基线和两组相互隔离的偏好数据开始。训练分支复制该基线得到可更新策略，同时冻结一份参考模型；评测分支保留未参与更新的问题。训练完成后，项目不直接采用最低损失的检查点，而是用同一组独立评测对比较更新前后的偏好表现，再把适配器、模型血缘和评测结果一起交付。

| 阶段 | 输入 | 关键动作 | 输出或失败处理 |
|:---|:---|:---|:---|
| 输入准入 | SFT 交付清单、训练偏好对、独立评测对 | 核对基线身份、偏好对有效性和数据隔离 | 通过后固定数据身份；失败则 `reject` |
| 候选训练 | 同源策略、冻结参考模型、固定训练预算 | 执行 DPO 更新并记录参考模型身份 | 输出候选与训练记录；执行失败则 `reject` |
| 独立复测 | 更新前策略、候选和固定评测对 | 配对比较偏好准确率、边际和截断 | 输出质量变化；未达门槛则 `tune` |
| 产物交付 | 候选、资源记录和评测结果 | 核对适配器路径、哈希与资源上限 | 输出 DPO 候选清单；产物缺失则 `reject` |

![离线偏好优化的对象与数据流](../public/02_PyTorch_Algorithms/84_dpo_project_flow.svg)

### Step 2：固定评测工作负载，比较更新前后的独立表现

更新前策略与更新后候选必须使用同一模型起点、分词器、评测对、最大长度和评分方法。训练阶段只允许策略参数发生变化；参考模型、数据版本、偏好约束强度、更新步数和随机种子都写入运行记录。这样，偏好准确率和边际变化才能归因于本轮候选，而不是评测工作负载发生变化。

| 比较项 | 固定条件或变化量 | 项目判断 |
|:---|:---|:---|
| 评测工作负载 | 相同问题、偏好回答、分词器与序列上限 | 不一致时停止比较 |
| 偏好准确率变化 | 候选减去更新前基线 | 判断更多评测对是否转向优选回答 |
| 平均边际变化 | 候选减去更新前基线 | 判断整体偏好强度是否提高 |
| 最差边际变化 | 候选减去更新前基线 | 防止均值改善掩盖尾部退化 |
| 问题与回答截断率 | 两次评测采用同一长度限制 | 截断超限时不使用质量变化作交付依据 |
| 训练资源 | 更新耗时与峰值显存 | 质量通过但预算越界时进入调优 |

### Step 3：按证据优先级形成候选状态

交付判断不是把指标加权成一个总分，而是按依赖顺序检查。数据泄漏、执行失败或适配器缺失会使候选不可复现，应直接拒绝；独立质量、截断或资源门槛未达到时，已有训练证据仍可用于调参，因此进入 `tune`；只有数据、更新、独立质量、资源和产物全部成立时，候选才进入 86 的生命周期交接，最终采用判断留给 Part 05。

| 证据层 | 必须确认的事实 | 失败后的状态 |
|:---|:---|:---|
| 数据 | 偏好对有效，问题与样本均无跨集合泄漏 | `reject` |
| 执行与产物 | 更新成功，适配器和身份哈希完整 | `reject` |
| 独立质量 | 准确率、平均边际和最差边际通过 | `tune` |
| 序列与资源 | 截断率、峰值显存和更新时间通过 | `tune` |
| 全部证据 | 上述检查同时成立 | `accept` |

![DPO 候选交付证据链](../public/02_PyTorch_Algorithms/84_dpo_delivery_gate.svg)

### Step 4：实现数据隔离、候选门槛和交付决策

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：检查样本与任务输入隔离 | 训练/评测的偏好对和任务输入身份 | 两个层次都无交集，偏好对本身有效 | 同偏好对泄漏、同问题换回答泄漏与无效对 |
| TODO 2：由更新前后评测形成候选门槛 | 更新前后胜率、平均/最差边际、截断和资源记录 | 正确计算增益，且无尾部退化、截断超限或资源越界 | 均值改善但尾部退化、截断过高与显存越界 |
| TODO 3：区分拒绝、调优与交付 | 数据、候选和产物检查 | 数据或产物失败时拒绝，候选未达标时调优 | 缺产物、数据泄漏与候选质量不足 |

```python
from typing import Any, Dict, List
```


```python
# ==========================================
# TODO 1：检查训练—评测隔离与偏好数据准入
# ==========================================
def check_preference_partition(
    train_pair_ids: List[str],
    evaluation_pair_ids: List[str],
    train_prompt_ids: List[str],
    evaluation_prompt_ids: List[str],
    audit: Dict[str, Any],
    minimum_valid_ratio: float,
) -> Dict[str, Any]:
    """检查偏好对、任务输入和样本内容能否支持独立评测。"""
    pair_overlap = sorted(set(train_pair_ids) & set(evaluation_pair_ids))
    prompt_overlap = sorted(set(train_prompt_ids) & set(evaluation_prompt_ids))
    # checks = ???  # 分别判断偏好对隔离、任务输入隔离、有效率和回答差异。
    # admitted = ???  # 四项检查必须全部通过。
    raise NotImplementedError("TODO 1：请构造偏好数据准入检查并形成结论")
    return {"admitted": admitted, "pair_overlap": pair_overlap, "prompt_overlap": prompt_overlap, "failed_checks": [name for name, passed in checks.items() if not passed]}


# ==========================================
# TODO 2：检查 DPO 候选的质量与资源门槛
# ==========================================
def evaluate_dpo_candidate(evidence: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """逐项检查独立偏好评测、尾部边际、显存和执行状态。"""
    baseline_margins = evidence["baseline_margins"]
    candidate_margins = evidence["candidate_margins"]
    if not baseline_margins or len(baseline_margins) != len(candidate_margins):
        raise ValueError("更新前后必须使用同一组非空偏好对")
    # baseline = ???  # 由逐偏好对边际汇总偏好准确率、平均边际和最差边际。
    # candidate = ???
    # comparison = ???  # 对齐计算三项更新前后变化量。
    raise NotImplementedError("TODO 2：请从逐偏好对边际形成候选证据")
    checks = {
        "execution": evidence["status"] == "ok",
        "accuracy_gain": comparison["preference_accuracy_gain"] >= gates["minimum_accuracy_gain"],
        "margin_gain": comparison["margin_mean_gain"] >= gates["minimum_margin_gain"],
        "margin_floor": comparison["margin_floor_gain"] >= gates["minimum_margin_floor_gain"],
        "truncation": evidence["evaluation"]["truncation_ratio"] <= gates["maximum_truncation_ratio"],
        "memory": evidence["resources"]["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
    }
    eligible = all(checks.values())
    return {"eligible": eligible, "comparison": comparison, "failed_checks": [name for name, passed in checks.items() if not passed]}


# ==========================================
# TODO 3：形成 DPO 项目交付决策
# ==========================================
def decide_dpo_delivery(partition: Dict[str, Any], candidate: Dict[str, Any], artifact: Dict[str, Any]) -> Dict[str, Any]:
    """只有数据、候选与适配器哈希同时通过时才交付。"""
    checks = {
        "partition": partition["admitted"],
        "candidate": candidate["eligible"],
        "artifact": bool(artifact.get("path")) and bool(artifact.get("sha256")),
    }
    # 数据或产物失败时拒绝，候选未达标时调优，其余情况才可交付。
    # decision = ???  # 数据或产物失败应拒绝；候选未达标应调优。
    raise NotImplementedError("TODO 3：请形成 DPO 交付决策")
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}
```


```python
def test_preference_partition_contract():
    """验证偏好对和任务输入两个层次的训练—评测隔离。"""
    audit = {"valid_ratio": 1.0, "identical_response_count": 0}
    partition = check_preference_partition(["a", "b"], ["c"], ["p1", "p2"], ["p3"], audit, 0.95)
    assert partition["admitted"]
    assert not check_preference_partition(["a"], ["a"], ["p1"], ["p2"], audit, 0.95)["admitted"]
    assert not check_preference_partition(["a"], ["b"], ["p1"], ["p1"], audit, 0.95)["admitted"]
    assert not check_preference_partition(["a"], ["b"], ["p1"], ["p2"], {**audit, "identical_response_count": 1}, 0.95)["admitted"]


def _dpo_candidate_fixture():
    """构造满足独立质量、截断和资源门槛的候选证据。"""
    gates = {
        "minimum_accuracy_gain": 0.05,
        "minimum_margin_gain": 0.02,
        "minimum_margin_floor_gain": 0.0,
        "maximum_peak_allocated_bytes": 10_000,
        "maximum_truncation_ratio": 0.05,
    }
    evidence = {
        "status": "ok",
        "baseline_margins": [-0.20, 0.00, 0.20, 0.40],
        "candidate_margins": [-0.19, 0.05, 0.25, 0.45],
        "evaluation": {"truncation_ratio": 0.01},
        "resources": {"peak_allocated_bytes": 8_000},
    }
    return evidence, gates


def test_dpo_candidate_gates():
    """验证尾部边际退化不能被平均收益抵消。"""
    evidence, gates = _dpo_candidate_fixture()
    candidate = evaluate_dpo_candidate(evidence, gates)
    assert candidate["eligible"]
    tail_regression = {**evidence, "candidate_margins": [-0.21, 0.05, 0.25, 0.47]}
    assert not evaluate_dpo_candidate(tail_regression, gates)["eligible"]
    assert not evaluate_dpo_candidate({**evidence, "evaluation": {"truncation_ratio": 0.20}}, gates)["eligible"]
    try:
        evaluate_dpo_candidate({**evidence, "candidate_margins": [0.10]}, gates)
    except ValueError:
        pass
    else:
        raise AssertionError("未拒绝无法配对的评测边际")


def test_dpo_delivery_decision():
    """验证数据、候选和产物对三态交付决策的不同影响。"""
    audit = {"valid_ratio": 1.0, "identical_response_count": 0}
    partition = check_preference_partition(["a"], ["b"], ["p1"], ["p2"], audit, 0.95)
    evidence, gates = _dpo_candidate_fixture()
    candidate = evaluate_dpo_candidate(evidence, gates)
    assert decide_dpo_delivery(partition, candidate, {"path": "adapter", "sha256": "abc"})["decision"] == "accept"
    assert decide_dpo_delivery(partition, candidate, {})["decision"] == "reject"
    assert decide_dpo_delivery(partition, {"eligible": False}, {"path": "adapter", "sha256": "abc"})["decision"] == "tune"


test_preference_partition_contract()
test_dpo_candidate_gates()
test_dpo_delivery_decision()
print("所有测试通过！")
```

🛑 **STOP HERE** 🛑
## 参考代码与解析

### 代码

```python
# TODO 1：检查训练—评测隔离与偏好数据准入
def check_preference_partition(train_pair_ids: List[str], evaluation_pair_ids: List[str], train_prompt_ids: List[str], evaluation_prompt_ids: List[str], audit: Dict[str, Any], minimum_valid_ratio: float) -> Dict[str, Any]:
    """检查偏好对、任务输入和样本内容能否支持独立评测。"""
    pair_overlap = sorted(set(train_pair_ids) & set(evaluation_pair_ids))
    prompt_overlap = sorted(set(train_prompt_ids) & set(evaluation_prompt_ids))
    checks = {
        "pair_isolation": not pair_overlap,
        "prompt_isolation": not prompt_overlap,
        "valid_ratio": audit["valid_ratio"] >= minimum_valid_ratio,
        "distinct_responses": audit["identical_response_count"] == 0,
    }
    admitted = all(checks.values())
    return {"admitted": admitted, "pair_overlap": pair_overlap, "prompt_overlap": prompt_overlap, "failed_checks": [name for name, passed in checks.items() if not passed]}


# TODO 2：检查 DPO 候选的质量与资源门槛
def evaluate_dpo_candidate(evidence: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """检查独立质量、截断和资源证据。"""
    baseline_margins = evidence["baseline_margins"]
    candidate_margins = evidence["candidate_margins"]
    if not baseline_margins or len(baseline_margins) != len(candidate_margins):
        raise ValueError("更新前后必须使用同一组非空偏好对")
    def summarize(margins: List[float]) -> Dict[str, float]:
        return {
            "preference_accuracy": sum(margin > 0 for margin in margins) / len(margins),
            "margin_mean": sum(margins) / len(margins),
            "margin_min": min(margins),
        }
    baseline = summarize(baseline_margins)
    candidate = summarize(candidate_margins)
    comparison = {
        "preference_accuracy_gain": candidate["preference_accuracy"] - baseline["preference_accuracy"],
        "margin_mean_gain": candidate["margin_mean"] - baseline["margin_mean"],
        "margin_floor_gain": candidate["margin_min"] - baseline["margin_min"],
    }
    checks = {
        "execution": evidence["status"] == "ok",
        "accuracy_gain": comparison["preference_accuracy_gain"] >= gates["minimum_accuracy_gain"],
        "margin_gain": comparison["margin_mean_gain"] >= gates["minimum_margin_gain"],
        "margin_floor": comparison["margin_floor_gain"] >= gates["minimum_margin_floor_gain"],
        "truncation": evidence["evaluation"]["truncation_ratio"] <= gates["maximum_truncation_ratio"],
        "memory": evidence["resources"]["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
    }
    eligible = all(checks.values())
    return {"eligible": eligible, "comparison": comparison, "failed_checks": [name for name, passed in checks.items() if not passed]}


# TODO 3：形成 DPO 项目交付决策
def decide_dpo_delivery(partition: Dict[str, Any], candidate: Dict[str, Any], artifact: Dict[str, Any]) -> Dict[str, Any]:
    """区分不可交付、需要调优和可以交付三种状态。"""
    checks = {
        "partition": partition["admitted"],
        "candidate": candidate["eligible"],
        "artifact": bool(artifact.get("path")) and bool(artifact.get("sha256")),
    }
    if not checks["partition"] or not checks["artifact"]:
        decision = "reject"
    elif not checks["candidate"]:
        decision = "tune"
    else:
        decision = "accept"
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}
```

### 解析

#### TODO 1：检查偏好数据隔离

- **判断依据**：同一偏好对或同一任务输入跨集合出现都会污染独立评测；相同回答则使偏好关系失效。
- **实现动作**：由偏好对重合、任务输入重合、有效率和回答差异构造具名检查，再汇总准入结论。
- **测试证据**：测试分别覆盖偏好对重合与同问题更换回答后的隐性泄漏。

#### TODO 2：检查候选门槛

- **判断依据**：平均边际改善不能掩盖最差边际退化，质量收益也不能抵消显存越界。
- **实现动作**：先从同一组偏好对的逐样本边际分别汇总偏好准确率、平均边际和最差边际，再计算更新前后变化并检查截断率与显存。
- **测试证据**：测试覆盖样本无法配对、均值改善但尾部退化，以及截断超限三种失败。

#### TODO 3：形成交付决策

- **判断依据**：数据、训练证据和可追溯适配器是三个独立硬门槛。
- **实现动作**：数据或产物失败时 `reject`，候选门槛未通过时 `tune`，三类证据完整时 `accept`。
- **测试证据**：测试覆盖缺少产物与候选仍需调优两类路径。
## Step 5：真实 DPO 候选训练与交付

CPU 题目区验证偏好对隔离、序列覆盖和三级决策；真实实验从固定 SFT 基线出发，执行有限 DPO 更新并保存独立偏好评测、资源与适配器证据。

| 对照层 | 输入 | 主要证据 | 输出与证据边界 |
|:---|:---|:---|:---|
| CPU 偏好门禁 | 人工构造的偏好对和指标 | 偏好对与任务输入隔离、截断、三级决策 | 只确认门禁逻辑，不证明真实 DPO 更新 |
| 真实 DPO 候选 | 固定 SFT 适配器、偏好数据快照和训练预算 | 独立偏好边际、序列覆盖、资源与适配器 | 输出候选清单；安全和通用能力仍需独立评测 |

### 5.1 环境、输入资产与固定条件

按[后训练优化 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-rl-cuda-quickstart)准备 `PT-RL-CUDA` 环境。12 GB 单卡使用 Qwen2.5-0.5B 的 SFT 适配器、策略模型与参考模型；v5 将统一序列上限设为 1024，训练对、评测对、更新步数与门槛保持不变。v4 在 768 上限下的峰值分配显存约 5.48 GiB、峰值保留显存约 6.62 GiB，因此本轮重新记录序列覆盖与实际峰值。

```python
from pathlib import Path
import json

from tools.project_runtime import resolve_project_root

RUN_MODE = "dry_run"  # 默认只检查配置；改为 real_gpu 执行新的真实 DPO 更新。
RUN_LABEL = "dpo_offline_len1024_v5"  # v4 的输入截断率超限；v5 只扩大统一序列上限。
PROFILE_ID = "PT-RL-CUDA"
SEMANTIC_ID = "PT-DPO-OFFLINE"
SCHEMA_VERSION = "model-evaluation/v1"

PROJECT_ROOT = resolve_project_root()
SFT_MANIFEST = PROJECT_ROOT / "benchmarks/results/62_sft_delivery/sft_delivery_manifest_sft_delivery_v2_1.json"
PREFERENCE_DATASET_ID = "trl-lib/ultrafeedback_binarized"
PREFERENCE_DATASET_REVISION = "main"
TRAIN_PAIRS = 24
EVALUATION_PAIRS = 16
TRAIN_POOL_SIZE = 256  # 从固定训练池按 seed 选择，不依赖数据集当前顺序的前 N 条。
EVALUATION_POOL_SIZE = 128  # 评测池来自独立 split。
MAX_LENGTH = 1024  # v4 在 768 Token 下输入截断率为 27.5%；本轮只扩大序列覆盖。
UPDATE_STEPS = 12
BETA = 0.1
LEARNING_RATE = 1e-5
DTYPE = "bfloat16"
SEED = 42
EVIDENCE_WINDOW_INDEX = 0  # 当前偏好数据窗口编号；窗口推进时递增。
GATES = {
    "minimum_valid_ratio": 0.95,
    "minimum_accuracy_gain": 0.0,
    "minimum_margin_gain": 0.0,
    "minimum_margin_floor_gain": -0.05,
    "maximum_prompt_truncation_ratio": 0.25,
    "maximum_response_truncation_ratio": 0.10,
    "maximum_peak_allocated_bytes": 10 * 1024**3,
}

RESULTS_DIR = PROJECT_ROOT / "benchmarks/results/84_dpo_preference"
EVIDENCE_PATH = RESULTS_DIR / f"dpo_evidence_{RUN_LABEL}.json"
MANIFEST_PATH = RESULTS_DIR / f"dpo_manifest_{RUN_LABEL}.json"
ARTIFACT_DIR = RESULTS_DIR / f"adapter_{RUN_LABEL}"
print({"run_mode": RUN_MODE, "evidence_path": str(EVIDENCE_PATH), "beta": BETA})
```

### 5.2 预检 SFT 基线、环境与输出路径

```python
import torch
from tools.project_runtime import environment_preflight, require_input_file
from tools.post_training_project_runtime import audit_preference_records, preference_record_to_text, run_dpo_delivery, save_json_record, stable_text_hash

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
print({"preflight": preflight["status"], "sft_artifact": sft_manifest["artifacts"]["adapter"]["path"]})
```

### 5.3 固定偏好数据与独立评测

训练使用数据集的 `train` 划分，评测使用 `test` 划分；优选回答/拒绝回答的共享任务输入、字段有效率和偏好对 ID 均在启动训练前保存。

```python
def check_preference_partition_for_run(train_pair_ids, evaluation_pair_ids, train_prompt_ids, evaluation_prompt_ids, audit, minimum_valid_ratio):
    """同时检查完整偏好对与共享 prompt 的跨切分隔离。"""
    pair_overlap = sorted(set(train_pair_ids) & set(evaluation_pair_ids))
    prompt_overlap = sorted(set(train_prompt_ids) & set(evaluation_prompt_ids))
    checks = {"no_pair_overlap": not pair_overlap, "no_prompt_overlap": not prompt_overlap, "valid_ratio": audit["valid_ratio"] >= minimum_valid_ratio, "distinct_responses": audit["identical_response_count"] == 0}
    return {"admitted": all(checks.values()), "pair_overlap": pair_overlap, "prompt_overlap": prompt_overlap, "train_pair_ids": train_pair_ids, "evaluation_pair_ids": evaluation_pair_ids, "train_prompt_ids": train_prompt_ids, "evaluation_prompt_ids": evaluation_prompt_ids, "failed_checks": [name for name, passed in checks.items() if not passed]}

if RUN_MODE == "real_gpu":
    from datasets import load_dataset
    from huggingface_hub import HfApi

    import random

    dataset_commit = HfApi().dataset_info(PREFERENCE_DATASET_ID, revision=PREFERENCE_DATASET_REVISION).sha
    train_pool = load_dataset(PREFERENCE_DATASET_ID, split=f"train[:{TRAIN_POOL_SIZE}]", revision=dataset_commit)
    evaluation_pool = load_dataset(PREFERENCE_DATASET_ID, split=f"test[:{EVALUATION_POOL_SIZE}]", revision=dataset_commit)
    selector = random.Random(SEED)
    train_indices = sorted(selector.sample(range(len(train_pool)), TRAIN_PAIRS))
    evaluation_indices = sorted(selector.sample(range(len(evaluation_pool)), EVALUATION_PAIRS))
    train_records = [dict(train_pool[index]) for index in train_indices]
    evaluation_records = [dict(evaluation_pool[index]) for index in evaluation_indices]
    all_audit = audit_preference_records(train_records + evaluation_records)
    train_pair_ids = [stable_text_hash(preference_record_to_text(row)) for row in train_records]
    evaluation_pair_ids = [stable_text_hash(preference_record_to_text(row)) for row in evaluation_records]
    train_prompt_ids = [stable_text_hash((preference_record_to_text(row)[0],)) for row in train_records]
    evaluation_prompt_ids = [stable_text_hash((preference_record_to_text(row)[0],)) for row in evaluation_records]
    partition = check_preference_partition_for_run(train_pair_ids, evaluation_pair_ids, train_prompt_ids, evaluation_prompt_ids, all_audit, GATES["minimum_valid_ratio"])
    if not partition["admitted"]:
        raise RuntimeError(f"偏好数据未通过准入：{partition['failed_checks']}")
    print({"dataset_commit": dataset_commit, "audit": all_audit, "partition": partition})
```

### 5.4 执行固定 DPO 候选并保存完整证据

策略模型与参考模型从同一 SFT 适配器初始化；参考模型冻结，策略模型在固定 `beta` 和步数下更新。

```python
if RUN_MODE == "real_gpu":
    from transformers import AutoTokenizer

    model_id = sft_manifest["model"]["id"]
    model_commit = sft_manifest["model"]["resolved_commit"]
    sft_adapter = sft_manifest["artifacts"]["adapter"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_commit)
    run_evidence = run_dpo_delivery(
        model_id=model_id, revision=model_commit, sft_adapter_path=sft_adapter,
        tokenizer=tokenizer, train_records=train_records, evaluation_records=evaluation_records,
        max_length=MAX_LENGTH, update_steps=UPDATE_STEPS, beta=BETA,
        learning_rate=LEARNING_RATE, seed=SEED, dtype_name=DTYPE,
        artifact_directory=ARTIFACT_DIR,
    )
    record = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "environment_profile": PROFILE_ID, "run_mode": RUN_MODE, "run_label": RUN_LABEL,
        "sft_manifest": str(SFT_MANIFEST),
        "dataset": {"id": PREFERENCE_DATASET_ID, "requested_revision": PREFERENCE_DATASET_REVISION, "resolved_commit": dataset_commit},
        "partition": partition, "audit": all_audit,
        "workload": {"train_pairs": TRAIN_PAIRS, "evaluation_pairs": EVALUATION_PAIRS, "train_pool_size": TRAIN_POOL_SIZE, "evaluation_pool_size": EVALUATION_POOL_SIZE, "train_indices": train_indices, "evaluation_indices": evaluation_indices, "max_length": MAX_LENGTH, "update_steps": UPDATE_STEPS, "beta": BETA, "seed": SEED},
        "gates": GATES, "evidence": run_evidence,
    }
    save_json_record(EVIDENCE_PATH, record)
    print({"evidence_path": str(EVIDENCE_PATH), "comparison": run_evidence["comparison"]})
else:
    print({"run_mode": RUN_MODE, "next_action": "确认数据和门槛后改为 real_gpu"})
```

### 5.5 独立读取结果并判断候选

```python
def decide_dpo_candidate_for_run(evidence, gates, partition):
    comparison = evidence["comparison"]
    sequence_contract = evidence["sequence_contract"]
    checks = {
        "partition": partition["admitted"],
        "execution": evidence["status"] == "ok",
        "accuracy_gain": comparison["preference_accuracy_gain"] >= gates["minimum_accuracy_gain"],
        "margin_gain": comparison["margin_mean_gain"] >= gates["minimum_margin_gain"],
        "margin_floor": comparison["margin_floor_gain"] >= gates["minimum_margin_floor_gain"],
        "prompt_coverage": sequence_contract["prompt_truncation_ratio"] <= gates["maximum_prompt_truncation_ratio"],
        "response_coverage": sequence_contract["response_truncation_ratio"] <= gates["maximum_response_truncation_ratio"],
        "memory": evidence["resources"]["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
        "artifact": bool(evidence.get("artifact", {}).get("path")) and bool(evidence.get("artifact", {}).get("sha256")),
    }
    decision = "accept" if all(checks.values()) else "reject"
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}

if EVIDENCE_PATH.exists():
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    decision = decide_dpo_candidate_for_run(record["evidence"], record["gates"], record["partition"])
    print({"decision": decision, "baseline": record["evidence"]["baseline"], "candidate": record["evidence"]["candidate"], "resources": record["evidence"]["resources"]})
```

### 5.6 形成离线 DPO 候选清单

84 与 85 是并列候选分支。该候选清单交给 86 核对生命周期、恢复与交接状态，不作为 85 的前置输入；最终采用判断由 Part 05 在统一的质量、安全、鲁棒性和成本评测后完成。

| 决策项 | 实测证据 | 结果 | 后续动作 |
|:---|:---|:---|:---|
| 数据与隔离 | 40 组偏好对全部有效；训练与评测任务输入无重合 | 通过 | 保留当前快照身份 |
| 偏好质量 | 偏好准确率保持 0.5625；平均边际 +0.0022，最差边际 -0.0036 | 通过质量下限 | 不据此自动采用 |
| 序列覆盖 | 1024 上限下任务输入截断率 5%，回答截断率 0 | 通过 | 固定当前长度与数据快照 |
| 项目决策 | 适配器、资源与独立偏好证据完整 | `accept` | 交给 86 进行生命周期交接 |

`dpo_offline_len1024_v5` 完成 12 次更新。偏好准确率保持 0.5625，平均偏好边际提高 0.0022，最差边际下降 0.0036；峰值分配显存约 6.49 GiB、峰值保留显存约 10.13 GiB。训练与评测隔离保持通过，任务输入截断率从 v4 的 27.5% 降至 5%，回答截断率为 0，因此全部候选门槛通过，结论为 `accept`。该结论只适用于当前 SFT 基线、偏好快照、12 次更新和 1024 序列上限。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 84 节离线 DPO；`dpo_offline_len1024_v5` |
| 运行条件 | 当前 SFT 基线；40 组偏好对；12 次更新；序列上限 1024 |
| 原始证据 | [DPO 证据](../../benchmarks/results/84_dpo_preference/dpo_evidence_dpo_offline_len1024_v5.json) |
| 正式证据 | DPO 证据内保存偏好质量、资源、隔离和截断指标 |
| 清单与交接 | [候选清单](../../benchmarks/results/84_dpo_preference/dpo_manifest_dpo_offline_len1024_v5.json)；本地 adapter 由清单记录 |
| 结论 | `accept`：候选交给 86 节核对生命周期，不表示已完成最终算法采用 |

```python
if RUN_MODE == "real_gpu":
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    decision = decide_dpo_candidate_for_run(record["evidence"], record["gates"], record["partition"])
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
        "resources": record["evidence"]["resources"],
        "artifacts": {"adapter": record["evidence"]["artifact"]},
        "next_consumer": "PT-POST-TRAINING-LIFECYCLE",
    }
    save_json_record(MANIFEST_PATH, manifest)
    print({"manifest_path": str(MANIFEST_PATH), "decision": decision["decision"]})
```

## 相关阅读

- [DPO 原论文](https://arxiv.org/abs/2305.18290)
- [TRL DPO Trainer 文档](https://huggingface.co/docs/trl/dpo_trainer)
- [UltraFeedback Binarized 数据集卡](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)
- [Part 02 · 86 在线后训练生命周期与恢复](./86_DPO_Online_Benchmark.md)
