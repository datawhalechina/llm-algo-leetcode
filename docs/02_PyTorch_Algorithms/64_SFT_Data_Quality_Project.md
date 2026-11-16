# 64. SFT Data Quality Project | SFT 数据质量项目

**难度：** Hard | **环境：** [CPU 主线](#runtime-environment) | **标签：** `监督微调`, `数据检查`, `训练与验证切分` | **目标人群：** 理解监督微调、需要检查训练数据的读者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

一条空回答追加结束标记后，也可能得到非零的监督词元数；同一个问题换了回答，又可能躲过完整样本去重。如果只检查“能加载、有标签”，这些记录仍会进入训练，甚至同时出现在训练和验证两侧。

本节固定数据版本、分词器、文本模板和长度，先检查原始记录，再清洗、切分并重新检查。通过后保存样本编号、切分成员和处理配置，让 LoRA 或 QLoRA 选型能使用同一批数据；模型训练后的质量在后续项目中评测。

**关键词：** `SFT`, `空回答`, `样本重复`, `任务重合`, `有效监督词元`

---

<a id="runtime-environment"></a>

## 运行环境

- **CPU练习：** 使用 [BASE-CPU](../guide.md#base-cpu)，运行Step 4与测试。
- **真实数据审计：** 复用 [PT-SFT-CUDA](../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart) 的数据与分词依赖；本实验不加载模型权重、不调用GPU，环境与输入配置见Step 5。

---

## 前置阅读

先确定哪些词元参与损失计算，再理解窗口截断和训练／评测切分。

- [13 SFT 监督边界与损失](./13_SFT_Training_Loop.md)：区分提示、回答、结束标记和填充标签。
- [30 长上下文微调机制](./30_Long_Context_Fine_Tuning.md)：观察长度上限怎样影响有效监督。
- [32 SFT 数据契约与泄漏审计](./32_Data_Engineering_for_SFT.md)：区分样本编号、完整样本重复和评测任务重合。

## Step 1：从原始记录到固定训练样本集合

SFT 数据中的指令和上下文组成输入，示范回答提供训练目标。检查时先记录数据集提交版本及原始行号，再按固定模板、分词器和最大长度，统计空内容、重复、截断及可用监督词元。清洗移除空指令、空回答和完整重复记录，再打乱保留样本、划分训练与验证，并重新检查两侧是否含有同一任务。

通过后固定的样本集合也称数据快照。数据清单保存来源提交、原始行号、样本编号、分词配置及两侧成员；下游据此重新读取相同记录。原始检查与清洗后检查分开保存，便于看清删掉了什么、最终还剩哪些问题。

| 阶段 | 输入 | 检查或处理 | 保存内容 |
|:---|:---|:---|:---|
| 记录来源 | 固定版本与原始记录 | 对应哪批数据、哪一行 | 数据提交与原始行号 |
| 原始检查 | 模板、分词器和长度 | 空内容、监督数量、重复和截断 | 原始计数及比例 |
| 清洗与切分 | 保留记录与固定种子 | 去重后打乱，分出训练和验证 | 删除数量、两侧成员 |
| 重新检查 | 清洗后两侧集合 | 监督是否可用、任务是否重合、比例是否达标 | 最终结果、问题清单与数据清单 |

![SFT 数据检查、清洗与切分](../public/02_PyTorch_Algorithms/64_sft_data_quality_flow.svg)

## Step 2：按实际序列统计监督，分别检查样本和任务重复

本项目把提示与回答分别分词，给回答追加结束标记 EOS，拼接后从右侧截断到最大长度，再用 PAD 补齐短序列。提示和填充位置不参与损失，保留下来的回答与 EOS 参与监督。图中 $p_1,p_2$ 表示提示词元， $a_1,a_2$ 表示回答词元：空回答可能仍有一个 EOS 标签；提示占满窗口时，则连 EOS 也无法保留。切换对话模板或分词方式后，应按新序列重新统计。

样本编号由内容摘要生成：本页合并连续空白、保留大小写，再计算哈希。完整样本比较指令、上下文与回答；任务比较只看指令和上下文。例如同一问题换一个回答，是两条不同示范，却仍是同一输入任务，分到训练和验证两侧时会被标记为重合。这里检查规范化文本完全相同的输入；改写和语义近似需要额外筛查。

| 检查量 | 计算对象 | 对准入的影响 |
|:---|:---|:---|
| 空指令、空回答 | 规范化后的字段 | 任一存在时需修复；EOS 不能补足缺失的示范回答 |
| 零监督词元 | 截断后保留的回答及 EOS | 为零时无法按该序列训练 |
| 两侧任务重合数 | 训练与验证的任务哈希集合交集大小 | 存在相同任务时重新检查切分 |
| 完整样本重复率 | 重复条数除以总条数，首份不计重复 | 超限时去重后重查 |
| 截断率 | 完整序列超出长度上限的样本占比 | 超限时检查示范长度或调整窗口 |
| 验证样本数 | 切分后验证侧的记录数量 | 达到数量下限；任务类型覆盖还需另看分布 |

![正常回答、空回答与零监督的区别](../public/02_PyTorch_Algorithms/64_sft_data_quality_decision.svg)

## Step 3：区分首次准备数据与修复已有版本

有空指令、空回答、零监督词元、两侧任务重合，或验证样本数不足时，结果为 `reject`，先修复这些问题；只有重复率或截断率超限时，结果为 `tune`，清洗或调整长度后重新检查。各项通过才得到 `accept`。同时存在空回答和重复率超限时，结果仍为拒绝，但两项原因都保留。

首次准备按最低验证数量判断；修复已有未通过版本时，还要求来源、处理规则、分词器和长度相同，原有问题全部消除、样本保留率达标且验证数量不减少。例如 100 条中保留 90 条、验证仍为 20 条，可能满足 90% 保留率；验证降为 19 条则不满足这个修复条件。数量相同仍可能换了成员，比较模型质量时还要核对验证成员清单。

| 情形 | 检查条件 | 结果与下一步 |
|:---|:---|:---|
| 存在空内容、零监督、任务重合或验证不足 | 任一项未通过 | `reject`，修复后重新检查 |
| 仅重复率或截断率超限 | 其他检查均通过 | `tune`，去重或调整长度 |
| 首次准备完成 | 清洗后各项通过，验证数量达到下限 | `accept`，保存最终成员与处理配置 |
| 修复已有未通过版本 | 相同来源与处理配置、问题消除、保留率达标、验证数量不减 | 采用修复后的版本；否则继续修复 |
| 选择可用于训练的版本 | 结果为接受、清单完整且问题列表为空 | 再按保留样本量和版本标识选择 |

## Step 4：实现准入判断与数据版本选择

补全三个函数：根据检查结果决定数据能否继续使用，判断清洗是否修复了原有问题，再筛选清单完整的训练数据版本。骨架负责输入校验、问题列表和候选排序。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：判断数据能否用于训练 | 空内容、零监督、重合、验证数量、重复率与截断率 | 优先处理需拒绝的问题，保留全部原因 | 同时出现空回答与重复超限、阈值等号及非法比例 |
| TODO 2：判断清洗后能否采用 | 清洗前后来源、处理配置、数量及问题列表 | 原先未通过、现在通过；来源与配置相同，数量条件达标且问题消除 | 只去重的成功情况；分词器变化、验证减少及新问题 |
| TODO 3：选出可训练版本 | 结果、清单完整性、问题列表及样本量 | 前三项均允许使用，才进入固定排序 | 样本更多但仍有问题、清单缺失、版本重复及输入顺序变化 |


```python
import math
from typing import Any, Dict, List

```


```python
def _validate_admission_inputs(audit: Dict[str, Any], limits: Dict[str, float]) -> None:
    """在比较门槛前拒绝非法计数、非有限比例及无效阈值。"""
    for name in ("empty_instruction_count", "empty_response_count", "zero_supervision_count", "split_leakage_count", "evaluation_count"):
        if type(audit[name]) is not int or audit[name] < 0:
            raise ValueError(f"{name} 必须为非负整数")
    for name in ("duplicate_ratio", "truncation_ratio"):
        if not math.isfinite(audit[name]) or not 0 <= audit[name] <= 1:
            raise ValueError(f"{name} 必须为[0, 1]内的有限比例")
    count = limits["minimum_evaluation_count"]
    if type(count) is not int or count <= 0:
        raise ValueError("最低评测样本数必须为正整数")
    for name in ("maximum_duplicate_ratio", "maximum_truncation_ratio"):
        if not math.isfinite(limits[name]) or not 0 <= limits[name] <= 1:
            raise ValueError(f"{name} 必须为[0, 1]内的有限比例")


def decide_data_admission(audit: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """根据审计事实决定能否冻结训练快照，并保留全部失败原因。"""
    _validate_admission_inputs(audit, limits)
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
    # ==========================================
    # TODO 1：判断数据准入状态
    # ==========================================
    # decision = ???  # reject/tune/accept；阻断项优先，所有失败都解除后才允许训练。
    raise NotImplementedError("TODO 1：判断数据准入状态")
    actions = {
        "reject": "repair_blockers_and_reaudit",
        "tune": "clean_and_reaudit",
        "accept": "freeze_training_snapshot",
    }
    return {
        "decision": decision, "next_action": actions[decision],
        "hard_failures": hard_failure_names, "soft_failures": soft_failure_names,
    }


def check_cleaned_snapshot_transition(
    before: Dict[str, Any], after: Dict[str, Any], minimum_retained_ratio: float,
) -> Dict[str, Any]:
    """检查同一来源和序列协议下，风险是否解除且保留了足够数据。"""
    if not math.isfinite(minimum_retained_ratio) or not 0 <= minimum_retained_ratio <= 1:
        raise ValueError("最低保留率必须为[0, 1]内的有限数")
    for snapshot in (before, after):
        for name in ("sample_count", "evaluation_count", "max_length"):
            if type(snapshot[name]) is not int or snapshot[name] < 0:
                raise ValueError(f"{name} 必须为非负整数")
        if snapshot["sample_count"] == 0 or snapshot["max_length"] == 0:
            raise ValueError("样本总数与最大长度必须大于零")
        if snapshot["evaluation_count"] > snapshot["sample_count"]:
            raise ValueError("评测样本数不能大于总样本数")
        if snapshot["decision"] not in {"reject", "tune", "accept"}:
            raise ValueError("未知准入状态")
        if any(not snapshot[name] for name in ("source_revision", "rule_version", "tokenizer_revision")):
            raise ValueError("数据来源、规则与分词器版本必须完整")
    if after["sample_count"] > before["sample_count"]:
        raise ValueError("清洗不能增加来源样本，扩充数据应另建版本")
    same_lineage = before["source_revision"] == after["source_revision"]
    same_rules = (
        before["rule_version"] == after["rule_version"]
        and before["tokenizer_revision"] == after["tokenizer_revision"]
        and before["max_length"] == after["max_length"]
    )
    evaluation_preserved = after["evaluation_count"] >= before["evaluation_count"]
    retained_ratio = after["sample_count"] / before["sample_count"]
    before_failures = set(before.get("hard_failures", [])) | set(before.get("soft_failures", []))
    after_failures = set(after.get("hard_failures", [])) | set(after.get("soft_failures", []))
    resolved_failures = before_failures - after_failures
    introduced_failures = after_failures - before_failures
    stage_transition = before["decision"] != "accept" and after["decision"] == "accept"
    # ==========================================
    # TODO 2：判断清洗版本能否替换
    # ==========================================
    # identity_compatible = ???  # bool；数据来源、处理规则、分词器版本和最大长度相同。
    # risks_repaired = ???  # bool；确实解除旧风险，而且新版本没有剩余风险。
    # eligible = ???  # bool；来源与分词配置相同、旧问题已解决、评测样本数不减且保留率达标，新版本通过检查。
    raise NotImplementedError("TODO 2：判断清洗版本能否替换")
    return {
        "eligible": eligible, "same_lineage": same_lineage, "same_rules": same_rules,
        "evaluation_preserved": evaluation_preserved, "retained_ratio": retained_ratio,
        "resolved_failures": sorted(resolved_failures),
        "introduced_failures": sorted(introduced_failures),
    }


def select_training_snapshot(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """筛选有完整清单且无失败项的版本，再按样本量及版本稳定选择。"""
    eligible = []
    seen_revisions = set()
    for candidate in candidates:
        if not isinstance(candidate["revision"], str) or not candidate["revision"].strip():
            raise ValueError("快照版本标识必须非空")
        if candidate["revision"] in seen_revisions:
            raise ValueError("候选版本标识重复，无法唯一交接")
        seen_revisions.add(candidate["revision"])
        if type(candidate["manifest_complete"]) is not bool:
            raise ValueError("清单完整性必须为布尔值")
        if type(candidate["sample_count"]) is not int or candidate["sample_count"] <= 0:
            raise ValueError("快照样本数必须为正整数")
        failures = candidate.get("hard_failures", []) + candidate.get("soft_failures", [])
        # ==========================================
        # TODO 3：筛选可用于训练的数据版本
        # ==========================================
        # handoff_allowed = ???  # bool；准入状态、清单完整性与失败清单一致。
        raise NotImplementedError("TODO 3：筛选可用于训练的数据版本")
        if handoff_allowed:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("没有通过准入且清单完整的数据版本")
    return sorted(eligible, key=lambda item: (-item["sample_count"], item["revision"]))[0]

```

## 测试

分别改变空回答、重复比例、来源配置、样本保留量或清单完整性，核对哪一项会改变最终结果。测试也检查比例恰好等于阈值及候选输入顺序变化。


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
    """验证清洗前后来源相同、评测样本数不减且样本保留率达标。"""
    before = {
        "source_revision": "source-a",
        "rule_version": "rules-v1",
        "decision": "reject",
        "tokenizer_revision": "tokenizer-a",
        "max_length": 256,
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

def _assert_value_error(callback):
    """确认无效事实或无法交付的版本被显式拒绝。"""
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("预期ValueError")


def test_soft_only_cleaning_can_advance():
    """重复或截断风险解除后也能晋级，不要求旧版本存在阻断项。"""
    before = {
        "source_revision": "source-a", "rule_version": "rules-v1",
        "tokenizer_revision": "tokenizer-a", "max_length": 256,
        "decision": "tune", "sample_count": 100, "evaluation_count": 20,
        "hard_failures": [], "soft_failures": ["duplicate_ratio"],
    }
    after = {**before, "decision": "accept", "sample_count": 90, "soft_failures": []}
    assert check_cleaned_snapshot_transition(before, after, 0.9)["eligible"]
    assert not check_cleaned_snapshot_transition(before, {**after, "tokenizer_revision": "tokenizer-b"}, 0.9)["eligible"]
    assert not check_cleaned_snapshot_transition(before, {**after, "max_length": 512}, 0.9)["eligible"]
    assert not check_cleaned_snapshot_transition(before, {**after, "soft_failures": ["truncation_ratio"]}, 0.9)["eligible"]
    _assert_value_error(lambda: check_cleaned_snapshot_transition({**before, "sample_count": 0}, after, 0.9))
    _assert_value_error(lambda: check_cleaned_snapshot_transition(before, after, float("nan")))


def test_all_failures_and_threshold_boundaries():
    """门槛等号通过，同时发生阻断与可修复风险时保留两类原因。"""
    clean, limits = _data_admission_fixture()
    boundary = {**clean, "evaluation_count": 2, "duplicate_ratio": 0.02, "truncation_ratio": 0.10}
    assert decide_data_admission(boundary, limits)["decision"] == "accept"
    result = decide_data_admission({**clean, "empty_response_count": 1, "duplicate_ratio": 0.03}, limits)
    assert result["decision"] == "reject"
    assert result["hard_failures"] == ["empty_response"]
    assert result["soft_failures"] == ["duplicate_ratio"]
    for ratio in (float("nan"), float("inf"), -0.1, 1.1):
        _assert_value_error(lambda: decide_data_admission({**clean, "duplicate_ratio": ratio}, limits))
    _assert_value_error(lambda: decide_data_admission({**clean, "evaluation_count": 1.5}, limits))


def test_handoff_excludes_inconsistent_and_incomplete_candidates():
    """更大的候选不能用样本量抵消失败项或清单缺失。"""
    valid = {"revision": "v1", "decision": "accept", "manifest_complete": True, "sample_count": 90}
    bad = {**valid, "revision": "v2", "sample_count": 120, "soft_failures": ["duplicate_ratio"]}
    incomplete = {**valid, "revision": "v3", "sample_count": 130, "manifest_complete": False}
    for pool in ([valid, bad, incomplete], [incomplete, bad, valid]):
        assert select_training_snapshot(pool)["revision"] == "v1"
    _assert_value_error(lambda: select_training_snapshot([bad, incomplete]))
    _assert_value_error(lambda: select_training_snapshot([valid, dict(valid)]))


test_soft_only_cleaning_can_advance()
test_all_failures_and_threshold_boundaries()
test_handoff_excludes_inconsistent_and_incomplete_candidates()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
def _validate_admission_inputs(audit: Dict[str, Any], limits: Dict[str, float]) -> None:
    """在比较门槛前拒绝非法计数、非有限比例及无效阈值。"""
    for name in ("empty_instruction_count", "empty_response_count", "zero_supervision_count", "split_leakage_count", "evaluation_count"):
        if type(audit[name]) is not int or audit[name] < 0:
            raise ValueError(f"{name} 必须为非负整数")
    for name in ("duplicate_ratio", "truncation_ratio"):
        if not math.isfinite(audit[name]) or not 0 <= audit[name] <= 1:
            raise ValueError(f"{name} 必须为[0, 1]内的有限比例")
    count = limits["minimum_evaluation_count"]
    if type(count) is not int or count <= 0:
        raise ValueError("最低评测样本数必须为正整数")
    for name in ("maximum_duplicate_ratio", "maximum_truncation_ratio"):
        if not math.isfinite(limits[name]) or not 0 <= limits[name] <= 1:
            raise ValueError(f"{name} 必须为[0, 1]内的有限比例")


def decide_data_admission(audit: Dict[str, Any], limits: Dict[str, float]) -> Dict[str, Any]:
    """根据审计事实决定能否冻结训练快照，并保留全部失败原因。"""
    _validate_admission_inputs(audit, limits)
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
    # TODO 1：判断数据准入状态
    decision = "reject" if hard_failure_names else "tune" if soft_failure_names else "accept"
    actions = {
        "reject": "repair_blockers_and_reaudit",
        "tune": "clean_and_reaudit",
        "accept": "freeze_training_snapshot",
    }
    return {
        "decision": decision, "next_action": actions[decision],
        "hard_failures": hard_failure_names, "soft_failures": soft_failure_names,
    }


def check_cleaned_snapshot_transition(
    before: Dict[str, Any], after: Dict[str, Any], minimum_retained_ratio: float,
) -> Dict[str, Any]:
    """检查同一来源和序列协议下，风险是否解除且保留了足够数据。"""
    if not math.isfinite(minimum_retained_ratio) or not 0 <= minimum_retained_ratio <= 1:
        raise ValueError("最低保留率必须为[0, 1]内的有限数")
    for snapshot in (before, after):
        for name in ("sample_count", "evaluation_count", "max_length"):
            if type(snapshot[name]) is not int or snapshot[name] < 0:
                raise ValueError(f"{name} 必须为非负整数")
        if snapshot["sample_count"] == 0 or snapshot["max_length"] == 0:
            raise ValueError("样本总数与最大长度必须大于零")
        if snapshot["evaluation_count"] > snapshot["sample_count"]:
            raise ValueError("评测样本数不能大于总样本数")
        if snapshot["decision"] not in {"reject", "tune", "accept"}:
            raise ValueError("未知准入状态")
        if any(not snapshot[name] for name in ("source_revision", "rule_version", "tokenizer_revision")):
            raise ValueError("数据来源、规则与分词器版本必须完整")
    if after["sample_count"] > before["sample_count"]:
        raise ValueError("清洗不能增加来源样本，扩充数据应另建版本")
    same_lineage = before["source_revision"] == after["source_revision"]
    same_rules = (
        before["rule_version"] == after["rule_version"]
        and before["tokenizer_revision"] == after["tokenizer_revision"]
        and before["max_length"] == after["max_length"]
    )
    evaluation_preserved = after["evaluation_count"] >= before["evaluation_count"]
    retained_ratio = after["sample_count"] / before["sample_count"]
    before_failures = set(before.get("hard_failures", [])) | set(before.get("soft_failures", []))
    after_failures = set(after.get("hard_failures", [])) | set(after.get("soft_failures", []))
    resolved_failures = before_failures - after_failures
    introduced_failures = after_failures - before_failures
    stage_transition = before["decision"] != "accept" and after["decision"] == "accept"
    # TODO 2：判断清洗版本能否替换
    identity_compatible = same_lineage and same_rules
    risks_repaired = bool(resolved_failures) and not after_failures
    eligible = (
        identity_compatible and risks_repaired and evaluation_preserved
        and retained_ratio >= minimum_retained_ratio and stage_transition
    )
    return {
        "eligible": eligible, "same_lineage": same_lineage, "same_rules": same_rules,
        "evaluation_preserved": evaluation_preserved, "retained_ratio": retained_ratio,
        "resolved_failures": sorted(resolved_failures),
        "introduced_failures": sorted(introduced_failures),
    }


def select_training_snapshot(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """筛选有完整清单且无失败项的版本，再按样本量及版本稳定选择。"""
    eligible = []
    seen_revisions = set()
    for candidate in candidates:
        if not isinstance(candidate["revision"], str) or not candidate["revision"].strip():
            raise ValueError("快照版本标识必须非空")
        if candidate["revision"] in seen_revisions:
            raise ValueError("候选版本标识重复，无法唯一交接")
        seen_revisions.add(candidate["revision"])
        if type(candidate["manifest_complete"]) is not bool:
            raise ValueError("清单完整性必须为布尔值")
        if type(candidate["sample_count"]) is not int or candidate["sample_count"] <= 0:
            raise ValueError("快照样本数必须为正整数")
        failures = candidate.get("hard_failures", []) + candidate.get("soft_failures", [])
        # TODO 3：筛选可用于训练的数据版本
        handoff_allowed = candidate["decision"] == "accept" and candidate["manifest_complete"] and not failures
        if handoff_allowed:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("没有通过准入且清单完整的数据版本")
    return sorted(eligible, key=lambda item: (-item["sample_count"], item["revision"]))[0]

```

### 解析

#### TODO 1：按问题类型选择结果

- 空内容、零监督、任务重合和验证数量不足需要先处理，重复与截断超限则进入待修复状态。两类问题同时出现时，结果为拒绝，问题列表保留两类原因。
- 根据两组问题列表是否为空选择三种状态。测试同时增加空回答和重复比例，检查最终拒绝且两项原因都存在；等号用例确认比例上限允许取等。

#### TODO 2：检查清洗前后的变化

- 采用修复版本要求原版本未通过、新版本通过，并保持来源与序列处理配置相同；原有问题全部消除、保留率达标，验证数量也要满足条件。
- 对各项取合取。仅去重的成功用例，与修改分词器、减少验证数量或新增截断问题的失败用例，分别检查这些条件。

#### TODO 3：筛选后再排序

- 只有接受状态、完整清单和空问题列表同时成立，候选才可用于训练；通过后的样本量用于选择，而非弥补前面的失败。
- 先筛选，再由骨架按样本量降序、版本标识升序排列。测试混入更大但不完整的候选，并反转输入顺序，确认仍选择同一可用版本。

## Step 5：真实数据准入实验

真实数据实验在 CPU 上读取固定规模的记录和分词器，保存原始检查、清洗后检查及最终训练／验证成员；通过后生成供 LoRA 或 QLoRA 选型读取的数据清单。这里采用首次准备规则，清洗后验证数量达到下限即可，未调用题目中的修复版本替换函数。

| 对照层 | 输入 | 主要证据 | 输出与证据边界 |
|:---|:---|:---|:---|
| CPU 状态机 | 人工构造的审计摘要 | 门槛优先级、版本替换、相同输入下的确定性选择 | 确认摘要输入下的判断结果 |
| 真实数据准入 | 固定数据与分词器提交 | 重复、泄漏、截断、监督词元、身份哈希 | 输出数据清单，后续训练再评测模型质量 |

合法阶段为`prepare → collect → close`：准备只读合同与依赖状态，采集保存事实，交接重新读取并校验。修改5.1后按六层顺序运行；准备阶段可在基础CPU环境执行。

### 5.1 运行意图与配置

先按[后训练优化 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart)准备 `PT-SFT-CUDA` 环境。实验读取真实数据并执行分词、切分和审计，不更新模型参数；后续 LoRA、QLoRA 和 SFT 训练可继续使用该环境。

```python
RUN_LABEL = "replace_with_unique_label"  # 新审计使用唯一标签；历史标签只用于登记与读取。
RUN_STAGE = "prepare"  # prepare：只检查；collect：采集数据事实；close：校验并交接。

```

### 5.2 预检依赖与数据身份

准备与读取阶段使用CPU；采集阶段复用 [PT-SFT-CUDA](../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart) 的数据集与分词器依赖，不加载模型权重。固定合同从项目注册表读取；数据集和分词器的请求版本在采集时解析为不可变提交。


```python
import hashlib
import importlib
import importlib.util
import json
import re
from pathlib import Path

from tools.project_runtime import bootstrap_project_root
import tools.post_training_project_runtime as data_runtime

data_runtime = importlib.reload(data_runtime)
PROJECT_ROOT = bootstrap_project_root()
registry = json.loads((PROJECT_ROOT / "benchmarks/project_registry/part02.json").read_text(encoding="utf-8"))
project_entry = next(item for item in registry["projects"] if item["semantic_id"] == "PT-SFT-DATA-QUALITY")
CONTRACT = json.loads((PROJECT_ROOT / project_entry["default_experiment_contract"]).read_text(encoding="utf-8"))
if RUN_STAGE not in {"prepare", "collect", "close"}:
    raise ValueError("RUN_STAGE 必须为 prepare、collect 或 close")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", RUN_LABEL):
    raise ValueError("标签只允许字母、数字、下划线与连字符，最多96个字符")
if RUN_STAGE != "prepare" and RUN_LABEL == "replace_with_unique_label":
    raise ValueError("采集或交接前请设置唯一 RUN_LABEL")
RESULTS_DIR = PROJECT_ROOT / "benchmarks/results/64_sft_data_quality"
RESULT_PATH = RESULTS_DIR / f"sft_data_admission_{RUN_LABEL}.json"
MANIFEST_PATH = RESULTS_DIR / f"sft_data_manifest_{RUN_LABEL}.json"
RECEIPT_PATH = RESULTS_DIR / f"sft_data_registration_{RUN_LABEL}.json"
required_packages = ("datasets", "transformers", "huggingface_hub")
missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
if RUN_STAGE == "collect":
    if missing_packages:
        raise ModuleNotFoundError(f"缺少依赖：{missing_packages}；请执行 PT-SFT-CUDA Quickstart")
    if any(path.exists() for path in (RESULT_PATH, MANIFEST_PATH, RECEIPT_PATH)):
        raise FileExistsError("同名资产已存在；完成采集后使用close，重新采集需更换标签")
if RUN_STAGE == "close" and not RESULT_PATH.is_file():
    raise FileNotFoundError(f"缺少本次采集证据：{RESULT_PATH}")
print({"stage": RUN_STAGE, "profile": CONTRACT["environment_profile"],
       "collection_dependencies_missing": missing_packages, "contract": project_entry["default_experiment_contract"],
       "result_path": str(RESULT_PATH), "formal_assets_written": False})

```

### 5.3 固定准入门槛

先检查内容与监督、两侧任务重合和验证数量，再检查重复率与截断率。全部未通过项随结果保存；阈值从固定配置文件读取。

| 门槛 | 条件 | 未通过时 |
|:---|:---|:---|
| 内容与监督 | 空指令、空回答、零监督词元数均为 0 | `reject` |
| 训练／验证切分 | 相同输入任务的重合数为 0，验证样本数达标 | `reject` |
| 重复比例 | 完整样本重复率不超过阈值 | `tune`，与需拒绝的问题并存时优先拒绝 |
| 长度比例 | 截断率不超过阈值 | `tune`，调整策略后重新统计 |


```python
DATASET_ID = CONTRACT["dataset"]["id"]
DATASET_REVISION = CONTRACT["dataset"]["requested_revision"]
MODEL_ID = CONTRACT["tokenizer"]["id"]
MODEL_REVISION = CONTRACT["tokenizer"]["requested_revision"]
SAMPLE_COUNT = CONTRACT["workload"]["sample_count"]
TRAIN_COUNT = CONTRACT["workload"]["train_count"]
MAX_LENGTH = CONTRACT["workload"]["max_length"]
SEED = CONTRACT["workload"]["seed"]
LIMITS = CONTRACT["limits"]
SCHEMA_VERSION = CONTRACT["schema_version"]
SEMANTIC_ID = CONTRACT["semantic_id"]
PROFILE_ID = CONTRACT["environment_profile"]
CONTRACT_SHA256 = hashlib.sha256(data_runtime.json_record_bytes(CONTRACT)).hexdigest()
if not (0 < TRAIN_COUNT < SAMPLE_COUNT and MAX_LENGTH >= 8
        and 0 < LIMITS["minimum_evaluation_count"] <= SAMPLE_COUNT - TRAIN_COUNT
        and all(0 <= LIMITS[key] <= 1 for key in ("maximum_duplicate_ratio", "maximum_truncation_ratio"))):
    raise ValueError("固定合同中的规模或门槛无效")

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

print({"workload": CONTRACT["workload"], "limits": LIMITS,
       "sequence_protocol": CONTRACT["sequence_protocol"], "contract_sha256": CONTRACT_SHA256})

```

### 5.4 执行真实数据审计并保存原始证据

`collect` 从缓存或远程仓库读取固定规模的数据与分词器。清洗后使用固定种子打乱保留样本，再切分训练与评测集合；原始审计与清洗后审计一并保存，包含源索引、样本身份和切分哈希。此阶段只保存采集证据。


```python
if RUN_STAGE == "collect":
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
        local_files_only=False,
    )
    raw_audit = data_runtime.audit_sft_records(records, tokenizer, max_length=MAX_LENGTH, train_count=TRAIN_COUNT)
    raw_admission = apply_data_admission_limits(raw_audit, LIMITS)

    import random

    accepted_rows = []
    seen_sample_ids = set()
    for source_index, row in enumerate(records):
        instruction = data_runtime.normalized_text(row.get("instruction"))
        context = data_runtime.normalized_text(row.get("input"))
        response = data_runtime.normalized_text(row.get("output"))
        sample_id = data_runtime.stable_text_hash((instruction, context, response))
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
        raise RuntimeError("清洗后样本不足，无法同时保留训练样本和最低数量的评测样本")
    audit = data_runtime.audit_sft_records(
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
        "run_mode": "real_data",
        "run_label": RUN_LABEL,
        "source_revision": dataset_commit,
        "rule_version": CONTRACT["rule_version"],
        "measurement_version": CONTRACT["measurement_version"],
        "sequence_protocol": CONTRACT["sequence_protocol"],
        "contract_sha256": CONTRACT_SHA256,
        "dataset": {"id": DATASET_ID, "requested_revision": DATASET_REVISION, "resolved_commit": dataset_commit},
        "tokenizer": {"id": MODEL_ID, "requested_revision": MODEL_REVISION, "resolved_commit": model_commit},
        "workload": dict(CONTRACT["workload"]),
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
            "train_sample_ids_hash": data_runtime.stable_text_hash(train_sample_ids),
            "evaluation_sample_ids_hash": data_runtime.stable_text_hash(evaluation_sample_ids),
        },
        "audit": audit,
        "quality": {"independent_evaluation": admission["hard_gate_passed"]},
        "decision": admission["decision"],
        "decision_details": admission,
        "admission": admission,
    }
    data_runtime.validate_sft_admission_record(record, CONTRACT, RUN_LABEL, apply_data_admission_limits)
    data_runtime.publish_json_record(RESULT_PATH, record)
    print({"result_path": str(RESULT_PATH), "decision": admission["decision"]})
else:
    print({"stage": RUN_STAGE, "collection": "skipped"})
```

### 5.5 独立读取结果并核对证据

重新读取原始JSON，按固定合同核对运行标签、来源提交、序列协议、计数与比例、切分成员和哈希，再从保存的事实复算准入状态与全部失败项。`close` 可以在新Kernel中按5.1–5.6顺序运行，无需复用采集单元的变量。


```python
if RUN_STAGE in {"collect", "close"}:
    evidence = data_runtime.read_sft_admission_record(
        RESULT_PATH, CONTRACT, RUN_LABEL, apply_data_admission_limits,
    )
    print({"run_label": evidence["run_label"], "decision": evidence["decision"],
           "failed_checks": evidence["admission"]["hard_failures"] + evidence["admission"]["soft_failures"],
           "raw_sample_count": evidence["raw_audit"]["sample_count"],
           "cleaned_sample_count": evidence["audit"]["sample_count"],
           "evaluation_count": evidence["audit"]["evaluation_count"],
           "truncation_ratio": evidence["audit"]["truncation_ratio"],
           "minimum_supervised_tokens": evidence["audit"]["minimum_supervised_tokens"]})
else:
    print({"stage": RUN_STAGE, "evidence_read": "skipped"})

```

### 5.6 形成决策、登记结果与交接

`accept` 表示这批数据可以进入 LoRA 或 QLoRA 选型；`tune` 表示需要先清洗重复样本或调整长度策略；`reject` 表示需要先修复监督数据或训练—评测隔离。数据准入清单保存采用状态与样本成员，完整审计指标保存在独立证据文件中。

下表为v1协议的归档结果，使用手工指令模板与分段分词协议；v2合同的真实采集结果尚未登记。63节继续读取已归档的0.5B清单，65节容量分支读取下表1.5B清单；二者不能混为同一比较身份。

只有`close`写登记回执，`accept`才生成下游清单。回执绑定证据与清单的SHA256；同一证据重复交接复用文件，中断后可补齐缺失清单或回执，同名不同内容停止。


| 决策项 | 实测证据 | 结果 | 后续动作 |
|:---|:---|:---|:---|
| 有效监督 | 移除 1 条空回答后保留 511 条；零监督样本为 0 | 通过 | 使用清洗后的样本集合 |
| 切分隔离 | 448 条训练、63 条评测；跨切分重合为 0 | 通过 | 固定切分哈希 |
| 重复与长度 | 重复率 0；截断率 1.17% | 通过 | 保留当前长度策略 |
| 项目决策 | 样本 ID、源索引、随机种子与切分哈希完整 | `accept` | 交给 65 的 1.5B QLoRA 容量分支 |

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
if RUN_STAGE == "close":
    registration = data_runtime.close_sft_admission(
        PROJECT_ROOT, RESULT_PATH, MANIFEST_PATH, RECEIPT_PATH,
        CONTRACT, RUN_LABEL, apply_data_admission_limits,
    )
    print(registration)
else:
    print({"stage": RUN_STAGE, "registration": "skipped"})

```

## 相关阅读

固定数据清单后，分别比较 LoRA 配置与低显存训练方案。

- [63 LoRA 变体选型](./63_LoRA_Variants_Benchmark.md)：沿用数据成员，比较秩配置的验证损失与资源成本。
- [65 QLoRA 选型](./65_QLoRA_Selection_Project.md)：观察同一任务下的质量、速度和可行批次。
- [Hugging Face Datasets 文档](https://huggingface.co/docs/datasets)：核对数据版本读取、行选择与切分接口。
