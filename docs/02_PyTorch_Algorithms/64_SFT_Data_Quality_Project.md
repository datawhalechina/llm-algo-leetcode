# 64. SFT Data Quality Project | SFT 数据质量项目

**难度：** Hard | **环境：** CPU + 真实数据审计 | **标签：** `监督微调`, `数据准入`, `独立评测` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节判断一个固定版本的 SFT 数据能否进入参数更新。学习者将检查字段、重复、切分泄漏、截断和有效监督词元（Token），形成可复查的准入结论，并输出供 LoRA 或 QLoRA 选型读取的数据清单。

**关键词：** `SFT`, `数据准入`, `切分泄漏`, `有效监督词元`, `数据准入清单`

---
## 前置阅读

**导语：** 先理解 SFT 监督目标、长样本截断和数据准入，再处理完整数据版本。

- [Part 02 · 13 SFT 训练循环](./13_SFT_Training_Loop.md)
- [Part 02 · 30 长上下文微调机制](./30_Long_Context_Fine_Tuning.md)
- [Part 02 · 32 SFT 数据准入与长度审计](./32_Data_Engineering_for_SFT.md)

---
### Step 1：固定数据身份、切分与准入输出

一次数据准入对应一个可追溯的数据版本。原始样本依次经过身份登记、稳定切分、监督范围检查和风险审计；通过准入后，训练与评测样本清单才可以进入参数更新。

| 阶段 | 输入 | 关键问题 | 输出 |
|:---|:---|:---|:---|
| 身份登记 | 数据来源与固定版本 | 本次审计究竟对应哪批样本 | 来源版本、样本身份和规则版本 |
| 稳定切分 | 已登记的样本 | 训练集与评测集能否独立复现 | 训练/评测样本清单 |
| 监督审计 | 模板化与分词后的样本 | 回答是否有效、监督词元是否保留 | 字段、截断和监督范围事实 |
| 准入交付 | 审计事实与固定阈值 | 应拒绝、清洗后复查，还是允许训练 | 数据准入清单与训练快照 |

![SFT 数据质量项目关系图](../public/02_PyTorch_Algorithms/64_sft_data_quality_flow.svg)

### Step 2：先固定身份与切分，再解释质量指标

同一来源中的样本必须用稳定身份选取并切分，不能依赖文件当前顺序。数据版本、分词器版本、选样规则、训练/评测样本清单和最大长度共同决定一次审计；其中任何一项变化，都应形成新的审计结果。完整样本重复与同任务输入跨切分重复是两类风险：前者会改变样本权重，后者会破坏独立评测。

| 审计指标 | 计算对象 | 对准入结论的影响 |
|:---|:---|:---|
| 空回答或零监督词元 | 模板化并分词后的样本 | 直接阻断训练准入 |
| 训练/评测输入交集 | 规范化任务输入的稳定哈希 | 直接阻断独立评测 |
| 完整样本重复率 | 任务输入与回答的联合哈希 | 超过阈值时进入清洗 |
| 截断率 | 固定分词器与最大长度 | 超过阈值时调整数据或长度策略 |
| 评测覆盖 | 固定任务类别与最小样例数 | 缺少核心任务时不能交付 |

### Step 3：让失败项决定修复动作与版本去向

准入状态由失败级别决定，而不是由指标总分决定。空指令、空回答、零监督词元（Token）、跨切分泄漏或评测缺失会破坏训练或独立评测，因此进入 `reject`（拒绝）；重复率或截断率超限属于可修复风险，进入 `tune`（调优）。所有门槛通过后进入 `accept`（接受）并冻结训练快照。每个状态同时记录失败项和下一步动作，使“完成审计”和“允许训练”保持明确区别。

| 证据状态 | 触发条件 | 是否允许进入训练 | 下一动作与版本去向 |
|:---|:---|:---|:---|
| `reject` | 存在空指令、空回答、零监督词元、切分泄漏或评测缺失 | 不允许 | 修复阻断项并生成新的审计结果 |
| `tune` | 无硬阻断，但重复率或截断率超过门槛 | 不允许 | 清洗重复样本、调整模板或长度策略后重新审计 |
| `accept` | 硬门槛与可修复门槛全部通过 | 允许 | 冻结训练/评测清单和数据身份，交给更新方案选型 |
| 清洗版本晋级 | 来源与规则一致、评测覆盖不下降、保留率达标、旧风险解除且未引入新阻断项 | 仅新版本为 `accept` 时允许 | 新快照替换旧快照；否则保留旧版本并继续修复 |
| 无可交付快照 | 没有同时满足 `accept` 与清单完整性的版本 | 不允许 | 停止交接，补齐清单或重新审计候选版本 |

![SFT 数据质量准入决策](../public/02_PyTorch_Algorithms/64_sft_data_quality_decision.svg)

### Step 4：实现数据准入状态机

CPU 题目使用审计指标、新旧数据版本和多个快照候选。学习者需要根据失败项生成准入状态与修复动作，比较新旧版本的数据来源与处理规则（数据血缘），再从已经准入且清单完整的候选中确定训练快照。测试覆盖硬阻断、可修复风险、数据血缘变化、评测覆盖下降、保留率边界和无可用候选。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：由失败项生成状态与修复动作 | 审计事实、硬门槛和软门槛 | 状态优先级正确，并指出下一步动作 | 空监督、泄漏、重复和截断边界 |
| TODO 2：判断清洗版本能否替换原版本 | 新旧版本的数据血缘、覆盖、保留量和状态 | 只允许同源且风险解除的版本晋级 | 来源或规则变化、覆盖下降和保留率边界 |
| TODO 3：选择可交付的训练快照 | 多个版本的准入状态、完整性和样本量 | 排除未准入版本并稳定处理并列 | 无可用候选与输入顺序变化 |


```python
from typing import Any, Dict, List
```


```python
# ==========================================
# TODO 1：根据硬门槛与软门槛决定数据版本状态
# ==========================================
def decide_data_admission(audit: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """根据固定审计字段返回 reject、tune 或 accept。"""
    hard_failures = {
        "empty_instruction": audit["empty_instruction_count"] > 0,
        "empty_response": audit["empty_response_count"] > 0,
        "zero_supervision": audit["zero_supervision_count"] > 0,
        "split_leakage": audit["split_leakage_count"] > 0,
        "missing_evaluation": audit["evaluation_count"] < limits["minimum_evaluation_count"],
    }
    soft_failures = {
        "duplicate_ratio": audit["duplicate_ratio"] > limits["maximum_duplicate_ratio"],
        "truncation_ratio": audit["truncation_ratio"] > limits["maximum_truncation_ratio"],
    }

    hard_failure_names = [name for name, failed in hard_failures.items() if failed]
    soft_failure_names = [name for name, failed in soft_failures.items() if failed]
    # 先按失败级别决定状态，再给出与状态一致的修复动作。
    # 用失败级别一次性确定状态和后续动作，避免两个字段彼此矛盾。
    # decision = ???  # 硬失败优先于软失败；无失败才允许训练。
    # next_action = ???  # 动作必须与 decision 对应。
    raise NotImplementedError("TODO 1：请形成数据准入状态与动作")

    return {
        "decision": decision,
        "next_action": next_action,
        "hard_failures": hard_failure_names,
        "soft_failures": soft_failure_names,
    }


# ==========================================
# TODO 2：判断清洗后的版本能否替换原版本
# ==========================================
def check_cleaned_snapshot_transition(
    before: Dict[str, Any],
    after: Dict[str, Any],
    minimum_retained_ratio: float,
) -> Dict[str, Any]:
    """检查数据血缘、评测覆盖、保留率和准入状态是否允许版本晋级。"""
    same_lineage = before["source_revision"] == after["source_revision"]
    same_rules = before["rule_version"] == after["rule_version"]
    evaluation_preserved = after["evaluation_count"] >= before["evaluation_count"]
    retained_ratio = after["sample_count"] / before["sample_count"]

    resolved_failures = set(before.get("hard_failures", [])) - set(after.get("hard_failures", []))
    introduced_failures = set(after.get("hard_failures", [])) - set(before.get("hard_failures", []))
    # 新版本既要保持血缘与覆盖，也要真正解除风险而不引入新的阻断项。
    # eligible = ???  # 血缘、规则、覆盖、保留率、风险解除和新风险必须共同检查。
    raise NotImplementedError("TODO 2：请判断清洗版本能否晋级")

    return {
        "eligible": eligible,
        "same_lineage": same_lineage,
        "same_rules": same_rules,
        "evaluation_preserved": evaluation_preserved,
        "retained_ratio": retained_ratio,
        "resolved_failures": sorted(resolved_failures),
        "introduced_failures": sorted(introduced_failures),
    }


# ==========================================
# TODO 3：从已准入版本中选择训练快照
# ==========================================
def select_training_snapshot(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从清单完整的 accept 候选中选择保留样本最多的稳定版本。"""
    eligible = [
        candidate
        for candidate in candidates
        if candidate["decision"] == "accept" and candidate["manifest_complete"]
    ]
    if not eligible:
        raise ValueError("没有通过准入且清单完整的数据版本")

    # 样本数相同时使用 revision 做稳定排序，避免依赖输入顺序。
    # selected = ???  # 先最大化保留样本数，并用 revision 稳定处理并列。
    raise NotImplementedError("TODO 3：请选择稳定训练快照")
    return selected
```


```python
def _data_admission_fixture():
    """构造无阻断项的数据审计与固定门槛。"""
    limits = {
        "minimum_evaluation_count": 2,
        "maximum_duplicate_ratio": 0.02,
        "maximum_truncation_ratio": 0.10,
    }
    clean = {
        "empty_instruction_count": 0,
        "empty_response_count": 0,
        "zero_supervision_count": 0,
        "split_leakage_count": 0,
        "evaluation_count": 16,
        "duplicate_ratio": 0.01,
        "truncation_ratio": 0.05,
    }
    return clean, limits


def test_data_admission_priority():
    """验证硬失败、软失败和通过状态的优先级。"""
    clean, limits = _data_admission_fixture()
    clean_result = decide_data_admission(clean, limits)
    assert (clean_result["decision"], clean_result["next_action"]) == ("accept", "freeze_training_snapshot")

    duplicate_risk = {**clean, "duplicate_ratio": 0.03}
    assert decide_data_admission(duplicate_risk, limits)["next_action"] == "clean_and_reaudit"
    assert decide_data_admission({**clean, "zero_supervision_count": 1}, limits)["decision"] == "reject"
    assert decide_data_admission({**clean, "split_leakage_count": 1}, limits)["decision"] == "reject"



def test_cleaned_snapshot_transition():
    """验证清洗版本保持血缘、覆盖和最低保留率。"""
    before = {
        "source_revision": "source-a",
        "rule_version": "rules-v1",
        "decision": "tune",
        "sample_count": 100,
        "evaluation_count": 20,
        "hard_failures": ["split_leakage"],
    }
    after = {**before, "decision": "accept", "sample_count": 92, "hard_failures": []}
    assert check_cleaned_snapshot_transition(before, after, 0.90)["eligible"]
    assert not check_cleaned_snapshot_transition(before, {**after, "source_revision": "source-b"}, 0.90)["eligible"]
    assert not check_cleaned_snapshot_transition(before, {**after, "evaluation_count": 19}, 0.90)["eligible"]
    assert not check_cleaned_snapshot_transition(before, {**after, "sample_count": 89}, 0.90)["eligible"]
    assert not check_cleaned_snapshot_transition(before, {**after, "hard_failures": ["zero_supervision"]}, 0.90)["eligible"]



def test_training_snapshot_selection():
    """验证仅从完整准入版本选择，并稳定处理并列。"""
    selected = select_training_snapshot([
        {"revision": "v2", "decision": "accept", "manifest_complete": True, "sample_count": 90},
        {"revision": "v1", "decision": "accept", "manifest_complete": True, "sample_count": 90},
        {"revision": "v3", "decision": "tune", "manifest_complete": True, "sample_count": 120},
    ])
    assert selected["revision"] == "v1"

    try:
        select_training_snapshot([{"revision": "v0", "decision": "reject", "manifest_complete": True, "sample_count": 100}])
    except ValueError:
        pass
    else:
        raise AssertionError("无 accept 候选时必须失败")



test_data_admission_priority()
test_cleaned_snapshot_transition()
test_training_snapshot_selection()
print("所有测试通过！")
```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1：根据硬门槛与软门槛决定数据版本状态
def decide_data_admission(audit: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """根据固定审计字段返回 reject、tune 或 accept。"""
    hard_failures = {
        "empty_instruction": audit["empty_instruction_count"] > 0,
        "empty_response": audit["empty_response_count"] > 0,
        "zero_supervision": audit["zero_supervision_count"] > 0,
        "split_leakage": audit["split_leakage_count"] > 0,
        "missing_evaluation": audit["evaluation_count"] < limits["minimum_evaluation_count"],
    }
    soft_failures = {
        "duplicate_ratio": audit["duplicate_ratio"] > limits["maximum_duplicate_ratio"],
        "truncation_ratio": audit["truncation_ratio"] > limits["maximum_truncation_ratio"],
    }
    hard_failure_names = [name for name, failed in hard_failures.items() if failed]
    soft_failure_names = [name for name, failed in soft_failures.items() if failed]
    if hard_failure_names:
        decision, next_action = "reject", "repair_blockers_and_reaudit"
    elif soft_failure_names:
        decision, next_action = "tune", "clean_and_reaudit"
    else:
        decision, next_action = "accept", "freeze_training_snapshot"
    return {
        "decision": decision,
        "next_action": next_action,
        "hard_failures": hard_failure_names,
        "soft_failures": soft_failure_names,
    }


# TODO 2：判断清洗后的版本能否替换原版本
def check_cleaned_snapshot_transition(
    before: Dict[str, Any],
    after: Dict[str, Any],
    minimum_retained_ratio: float,
) -> Dict[str, Any]:
    """检查数据血缘、评测覆盖、保留率和准入状态是否允许版本晋级。"""
    same_lineage = before["source_revision"] == after["source_revision"]
    same_rules = before["rule_version"] == after["rule_version"]
    evaluation_preserved = after["evaluation_count"] >= before["evaluation_count"]
    retained_ratio = after["sample_count"] / before["sample_count"]
    resolved_failures = set(before.get("hard_failures", [])) - set(after.get("hard_failures", []))
    introduced_failures = set(after.get("hard_failures", [])) - set(before.get("hard_failures", []))
    eligible = (
        same_lineage
        and same_rules
        and evaluation_preserved
        and retained_ratio >= minimum_retained_ratio
        and bool(resolved_failures)
        and not introduced_failures
        and before["decision"] != "accept"
        and after["decision"] == "accept"
    )
    return {
        "eligible": eligible,
        "same_lineage": same_lineage,
        "same_rules": same_rules,
        "evaluation_preserved": evaluation_preserved,
        "retained_ratio": retained_ratio,
        "resolved_failures": sorted(resolved_failures),
        "introduced_failures": sorted(introduced_failures),
    }


# TODO 3：从已准入版本中选择训练快照
def select_training_snapshot(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从清单完整的 accept 候选中选择保留样本最多的稳定版本。"""
    eligible = [
        candidate
        for candidate in candidates
        if candidate["decision"] == "accept" and candidate["manifest_complete"]
    ]
    if not eligible:
        raise ValueError("没有通过准入且清单完整的数据版本")
    selected = sorted(eligible, key=lambda item: (-item["sample_count"], item["revision"]))[0]
    return selected
```

### 解析

#### TODO 1：判断数据准入状态

- **判断依据**：空回答、零监督词元、训练—评测泄漏和评测缺失会破坏训练或独立评测，属于硬门槛；重复率和截断率属于可修复门槛。
- **实现动作**：先保留具体失败项，再按阻断项、可修复项、全部通过的优先级生成状态和下一步动作。
- **测试证据**：测试分别覆盖干净数据、重复率越界、零监督词元和切分泄漏。

#### TODO 2：检查版本替换

- **判断依据**：清洗版本只有在数据血缘与规则不变、评测覆盖不下降、样本保留率达标且状态确实晋级时，才能替换原版本。
- **实现动作**：比较新旧失败集合，并同时检查血缘、规则、评测覆盖和样本保留率。
- **测试证据**：血缘漂移、评测覆盖下降和保留率越界都会阻止替换。

#### TODO 3：选择训练快照

- **判断依据**：参数更新只接受状态为 `accept` 且准入清单完整的数据版本；样本量更大不能抵消未通过的质量门槛。
- **实现动作**：过滤不合格候选，按保留样本数降序、revision 升序稳定选择。
- **测试证据**：测试验证未准入的大版本不会胜出，并覆盖无合格候选的失败路径。
## Step 5：真实数据准入实验

CPU 题目区验证准入状态机；真实数据实验固定数据来源、分词器版本、切分规则和最大长度，记录重复率、跨切分泄漏、截断率、有效监督词元和样本身份，并生成供 LoRA 或 QLoRA 选型使用的数据准入清单。

| 对照层 | 输入 | 主要证据 | 输出与证据边界 |
|:---|:---|:---|:---|
| CPU 状态机 | 人工构造的审计摘要 | 门槛优先级、版本替换、相同输入下的确定性选择 | 只确认准入逻辑，不判断真实数据 |
| 真实数据准入 | 固定数据与分词器快照 | 重复、泄漏、截断、监督词元、身份哈希 | 输出数据准入清单；不证明训练后模型质量 |

### 5.1 环境、输入资产与固定条件

先按[后训练优化 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart)准备 `PT-SFT-CUDA` 环境。实验读取真实数据并执行分词、切分和审计，不更新模型参数；后续 LoRA、QLoRA 和 SFT 训练可继续使用该环境。

```python
from pathlib import Path
import json

from tools.project_runtime import resolve_project_root

RUN_MODE = "dry_run"  # 默认只检查配置；改为 real_data 执行新的真实数据审计。
RUN_LABEL = "alpaca_qwen25_15b_v1"  # 1.5B QLoRA 容量交叉实验的上游数据快照。
PROFILE_ID = "PT-SFT-CUDA"  # 后训练 SFT 项目共用环境。
SEMANTIC_ID = "PT-SFT-DATA-QUALITY"  # 项目稳定语义身份。
SCHEMA_VERSION = "fine-tuning-project/v1"  # 与项目资产关系契约一致。

DATASET_ID = "tatsu-lab/alpaca"  # 固定公开 SFT 数据来源。
DATASET_REVISION = "main"  # 运行时解析为不可变 commit。
MODEL_ID = "Qwen/Qwen2.5-1.5B"  # 为 65 的 1.5B 低资源分支重新核对长度与监督范围。
MODEL_REVISION = "main"  # 运行时解析为不可变 commit。
LOCAL_FILES_ONLY = False  # 缓存完整后可改为 True，避免网络请求。

SAMPLE_COUNT = 512  # 审计规模；包含训练与评测切分。
TRAIN_COUNT = 448  # 前 448 条作为训练候选，其余作为独立评测候选。
MAX_LENGTH = 256  # tokenizer 后的统一最大长度。
SEED = 42  # 固定抽样和切分顺序。
LIMITS = {
    "minimum_evaluation_count": 32,
    "maximum_duplicate_ratio": 0.02,
    "maximum_truncation_ratio": 0.10,
}

PROJECT_ROOT = resolve_project_root()
RESULTS_DIR = PROJECT_ROOT / "benchmarks/results/64_sft_data_quality"
RESULT_PATH = RESULTS_DIR / f"sft_data_admission_{RUN_LABEL}.json"
MANIFEST_PATH = RESULTS_DIR / f"sft_data_manifest_{RUN_LABEL}.json"
print({"run_mode": RUN_MODE, "result_path": str(RESULT_PATH), "manifest_path": str(MANIFEST_PATH), "limits": LIMITS})
```

### 5.2 预检依赖与数据身份

预检确认运行库、仓库导入和结果路径。进入 `real_data` 模式后，再将数据集与分词器的请求版本解析为不可变提交哈希。

```python
import importlib.util

required_packages = ["datasets", "transformers", "huggingface_hub"]
missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
if missing_packages:
    raise ModuleNotFoundError(f"缺少依赖：{missing_packages}；请执行 PT-SFT-CUDA Quickstart。")

from tools.post_training_project_runtime import (
    audit_sft_records, normalized_text, save_json_record, stable_text_hash,
)

for path in (RESULT_PATH, MANIFEST_PATH):
    if RUN_MODE == "real_data" and path.exists():
        raise FileExistsError(f"输出已存在，请修改 RUN_LABEL：{path}")
print({"profile_id": PROFILE_ID, "packages": "ready", "output_paths_available": True})
```

### 5.3 固定准入门槛

| 门槛 | 条件 | 失败状态 |
|:---|:---|:---|
| 有效监督 | 空回答数、零监督词元数均为 0 | `reject` |
| 独立评测 | 跨切分输入重合为 0，评测样本数达标 | `reject` |
| 重复风险 | 完整样本重复率不超过阈值 | `tune` |
| 长度风险 | 分词器截断率不超过阈值 | `tune` |
### 5.4 执行真实数据审计并保存完整证据

`real_data` 模式从本地缓存或远程仓库读取数据。清洗后使用固定随机种子打乱保留样本，再形成训练与评测切分；结果同时保存源索引、样本 ID 与切分哈希。

```python
def apply_data_admission_limits(audit, limits):
    """将真实审计事实映射为固定的 reject、tune 或 accept。"""
    hard_failures = {
        "empty_instruction": audit["empty_instruction_count"] > 0,
        "empty_response": audit["empty_response_count"] > 0,
        "zero_supervision": audit["zero_supervision_count"] > 0,
        "split_leakage": audit["split_leakage_count"] > 0,
        "missing_evaluation": audit["evaluation_count"] < limits["minimum_evaluation_count"],
    }
    soft_failures = {
        "duplicate_ratio": audit["duplicate_ratio"] > limits["maximum_duplicate_ratio"],
        "truncation_ratio": audit["truncation_ratio"] > limits["maximum_truncation_ratio"],
    }
    hard_gate_passed = not any(hard_failures.values())
    soft_gate_passed = not any(soft_failures.values())
    decision = "reject" if not hard_gate_passed else "accept" if soft_gate_passed else "tune"
    return {
        "decision": decision,
        "hard_gate_passed": hard_gate_passed,
        "soft_gate_passed": soft_gate_passed,
        "hard_failures": [name for name, failed in hard_failures.items() if failed],
        "soft_failures": [name for name, failed in soft_failures.items() if failed],
    }

if RUN_MODE == "real_data":
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    dataset_commit = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION).sha
    model_commit = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    dataset = load_dataset(DATASET_ID, split=f"train[:{SAMPLE_COUNT}]", revision=dataset_commit)
    records = [dict(row) for row in dataset]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=model_commit,
        local_files_only=LOCAL_FILES_ONLY,
    )
    raw_audit = audit_sft_records(records, tokenizer, max_length=MAX_LENGTH, train_count=TRAIN_COUNT)
    raw_admission = apply_data_admission_limits(raw_audit, LIMITS)

    import random

    accepted_rows = []
    seen_sample_ids = set()
    for source_index, row in enumerate(records):
        instruction = normalized_text(row.get("instruction"))
        context = normalized_text(row.get("input"))
        response = normalized_text(row.get("output"))
        sample_id = stable_text_hash((instruction, context, response))
        if not instruction or not response or sample_id in seen_sample_ids:
            continue
        seen_sample_ids.add(sample_id)
        accepted_rows.append((source_index, row, sample_id))

    random.Random(SEED).shuffle(accepted_rows)
    cleaned_records = [row for _, row, _ in accepted_rows]
    cleaned_train_count = min(
        TRAIN_COUNT, len(cleaned_records) - int(LIMITS["minimum_evaluation_count"]),
    )
    if cleaned_train_count <= 0:
        raise RuntimeError("清洗后样本不足，无法保留最低评测覆盖")
    audit = audit_sft_records(
        cleaned_records, tokenizer, max_length=MAX_LENGTH, train_count=cleaned_train_count,
    )
    admission = apply_data_admission_limits(audit, LIMITS)
    accepted_indices = [source_index for source_index, _, _ in accepted_rows]
    accepted_sample_ids = [sample_id for _, _, sample_id in accepted_rows]
    train_sample_ids = accepted_sample_ids[:cleaned_train_count]
    evaluation_sample_ids = accepted_sample_ids[cleaned_train_count:]
    record = {
        "schema_version": SCHEMA_VERSION,
        "semantic_id": SEMANTIC_ID,
        "environment_profile": PROFILE_ID,
        "run_mode": RUN_MODE,
        "run_label": RUN_LABEL,
        "source_revision": dataset_commit,
        "rule_version": "sft-data-admission-v1",
        "dataset": {"id": DATASET_ID, "requested_revision": DATASET_REVISION, "resolved_commit": dataset_commit},
        "tokenizer": {"id": MODEL_ID, "requested_revision": MODEL_REVISION, "resolved_commit": model_commit},
        "workload": {"sample_count": SAMPLE_COUNT, "train_count": cleaned_train_count, "max_length": MAX_LENGTH, "seed": SEED},
        "limits": LIMITS,
        "raw_audit": raw_audit,
        "raw_admission": raw_admission,
        "cleaning": {
            "rule": "drop_empty_instruction_response_and_exact_duplicates",
            "removed_count": len(records) - len(cleaned_records),
            "retained_ratio": len(cleaned_records) / len(records),
        },
        "snapshot": {
            "train_record_indices": accepted_indices[:cleaned_train_count],
            "evaluation_record_indices": accepted_indices[cleaned_train_count:],
            "train_sample_ids": train_sample_ids,
            "evaluation_sample_ids": evaluation_sample_ids,
            "train_sample_ids_hash": stable_text_hash(train_sample_ids),
            "evaluation_sample_ids_hash": stable_text_hash(evaluation_sample_ids),
        },
        "audit": audit,
        "quality": {"independent_evaluation": admission["hard_gate_passed"]},
        "decision": admission["decision"],
        "decision_details": admission,
        "admission": admission,
    }
    save_json_record(RESULT_PATH, record)
    print({"result_path": str(RESULT_PATH), "decision": admission["decision"]})
else:
    print({"run_mode": RUN_MODE, "next_action": "检查配置后改为 real_data"})
```

### 5.5 独立读取结果并核对证据

从保存的 JSON 重新加载审计结果，并核对数据身份、提交哈希、硬门槛和可修复门槛。

```python
from tools.post_training_project_runtime import stable_text_hash

def recompute_data_admission_for_read(audit, limits):
    """仅从结果文件中的审计事实重新计算准入状态。"""
    hard_failed = (
        audit["empty_instruction_count"] > 0
        or audit["empty_response_count"] > 0
        or audit["zero_supervision_count"] > 0
        or audit["split_leakage_count"] > 0
        or audit["evaluation_count"] < limits["minimum_evaluation_count"]
    )
    soft_failed = (
        audit["duplicate_ratio"] > limits["maximum_duplicate_ratio"]
        or audit["truncation_ratio"] > limits["maximum_truncation_ratio"]
    )
    return "reject" if hard_failed else "tune" if soft_failed else "accept"

if RESULT_PATH.exists():
    evidence = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["semantic_id"] == SEMANTIC_ID
    assert evidence["dataset"]["resolved_commit"]
    assert evidence["tokenizer"]["resolved_commit"]
    assert len(evidence["snapshot"]["train_record_indices"]) == evidence["audit"]["train_count"]
    assert len(evidence["snapshot"]["evaluation_record_indices"]) == evidence["audit"]["evaluation_count"]
    assert len(evidence["snapshot"]["train_sample_ids"]) == evidence["audit"]["train_count"]
    assert not set(evidence["snapshot"]["train_sample_ids"]) & set(evidence["snapshot"]["evaluation_sample_ids"])
    assert stable_text_hash(evidence["snapshot"]["train_sample_ids"]) == evidence["snapshot"]["train_sample_ids_hash"]
    assert stable_text_hash(evidence["snapshot"]["evaluation_sample_ids"]) == evidence["snapshot"]["evaluation_sample_ids_hash"]
    recomputed_decision = recompute_data_admission_for_read(evidence["audit"], evidence["limits"])
    assert recomputed_decision == evidence["admission"]["decision"] == evidence["decision"]
    print({
        "raw_decision": evidence["raw_admission"]["decision"],
        "decision": recomputed_decision,
        "failed_checks": evidence["admission"]["hard_failures"] + evidence["admission"]["soft_failures"],
        "removed_count": evidence["cleaning"]["removed_count"],
        "sample_count": evidence["audit"]["sample_count"],
        "duplicate_ratio": evidence["audit"]["duplicate_ratio"],
        "truncation_ratio": evidence["audit"]["truncation_ratio"],
        "split_leakage_count": evidence["audit"]["split_leakage_count"],
        "minimum_supervised_tokens": evidence["audit"]["minimum_supervised_tokens"],
    })
```

### 5.6 生成数据准入清单

`accept` 表示该数据快照可以进入 LoRA 或 QLoRA 选型；`tune` 表示需要先清洗重复样本或调整长度策略；`reject` 表示需要先修复监督数据或训练—评测隔离。数据准入清单保存采用状态与快照身份，完整审计指标保存在独立证据文件中。

| 决策项 | 实测证据 | 结果 | 后续动作 |
|:---|:---|:---|:---|
| 有效监督 | 移除 1 条空回答后保留 511 条；零监督样本为 0 | 通过 | 使用清洗后的快照 |
| 切分隔离 | 448 条训练、63 条评测；跨切分重合为 0 | 通过 | 固定切分哈希 |
| 重复与长度 | 重复率 0；截断率 1.17% | 通过 | 保留当前长度策略 |
| 项目决策 | 样本 ID、源索引、随机种子与切分哈希完整 | `accept` | 交给 65 的 1.5B QLoRA 容量分支 |

`alpaca_qwen25_15b_v1` 使用 Qwen2.5-1.5B 的分词器重新登记同一数据快照：从 512 条原始记录中移除 1 条空回答，形成 448 条训练记录和 63 条独立评测记录。重复率为 0，截断率为 1.17%，跨切分任务输入重合和零监督样本均为 0；随机种子、源索引、样本 ID 与切分哈希已经登记。全部准入门槛通过，结论为 `accept`。63 的 LoRA 选型继续使用已归档的 0.5B 数据快照，不与本次 1.5B 容量分支混用。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 64 节 SFT 数据准入；`alpaca_qwen25_15b_v1` |
| 运行条件 | Qwen2.5-1.5B tokenizer；512 条原始记录；固定训练/评测切分 |
| 原始证据 | [数据审计](../../benchmarks/results/64_sft_data_quality/sft_data_admission_alpaca_qwen25_15b_v1.json) |
| 正式证据 | 数据审计内保存重复、截断、监督和跨切分隔离指标 |
| 清单与交接 | [数据清单](../../benchmarks/results/64_sft_data_quality/sft_data_manifest_alpaca_qwen25_15b_v1.json) |
| 结论 | `accept`：允许进入 65 节 1.5B QLoRA 容量分支 |

```python
if RUN_MODE == "real_data":
    evidence = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    project_output = {
        "schema_version": evidence["schema_version"],
        "semantic_id": evidence["semantic_id"],
        "environment_profile": evidence["environment_profile"],
        "run_mode": evidence["run_mode"],
        "run_label": evidence["run_label"],
        "decision": evidence["decision"],
        "decision_details": evidence["decision_details"],
        "artifact_role": "training.sft_data_quality",
        "source_evidence": str(RESULT_PATH),
        "source_revision": evidence["source_revision"],
        "dataset": evidence["dataset"],
        "tokenizer": evidence["tokenizer"],
        "workload": evidence["workload"],
        "snapshot": evidence["snapshot"],
        "audit": evidence["audit"],
        "admission": evidence["admission"],
        "sample_ids_hash": evidence["audit"]["sample_ids_hash"],
        "next_consumer": ["PT-LORA-VARIANTS", "PT-QLORA-SELECTION"] if evidence["decision"] == "accept" else [],
    }
    save_json_record(MANIFEST_PATH, project_output)
    print({"manifest_path": str(MANIFEST_PATH), "decision": project_output["decision"], "source_evidence": str(RESULT_PATH)})
```

## 相关阅读

- [Hugging Face Datasets 官方文档](https://huggingface.co/docs/datasets)
- [LIMA：少量高质量数据的监督微调](https://arxiv.org/abs/2305.11206)
- [Part 02 · 63 LoRA 变体选型](./63_LoRA_Variants_Benchmark.md)
- [Part 02 · 65 QLoRA 低资源选型](./65_QLoRA_Selection_Project.md)
