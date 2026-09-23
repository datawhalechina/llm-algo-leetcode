# 85. GRPO Groupwise Alignment Project | GRPO 组内对齐项目
**难度：** Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `GRPO`, `项目评估` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 本节导读

本节要求你评估 GRPO 在分组奖励口径下是否值得继续训练。先固定分组方式、奖励计算和评测窗口，再比较 baseline 与 candidate 的平均奖励、组内波动、覆盖率和最差组表现。最终判断平均收益是否可靠，并给出保留、调参或停止训练的建议。

**关键词：** `groupwise reward`, `stability`, `advantage`, `alignment decision`
## 前置阅读

**导语：** 先把 GRPO 损失、偏好优化主线、偏好数据评测和离线 DPO 项目理顺，再进入这个项目；本节默认你已经知道 group-wise 优化的基本对象，重点转向 GRPO 方案是否值得继续训练。
- [16. GRPO Loss Tutorial | GRPO 损失教程](./16_GRPO_Loss_Tutorial.md)
- [2.4](./2_4.md)
- [50. Preference Data and Evaluation | 偏好数据与评测](./50_Preference_Data_and_Evaluation.md)
- [84. DPO Preference Project | DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [后训练与对齐专题入口](../topic_discussion/post_training_optimization/intro.md)
- [06 Project Decision and Delivery | 项目决策与交付](../topic_discussion/post_training_optimization/06_project_decision_and_delivery.md)

## 相关阅读

**导语：** 做完 group-wise 对齐项目后，最自然的下一步是把离线结论推进到在线基准，或回到专题收口页统一看 adopt / tune / reject 的交付口径。
- [86. DPO Online Benchmark | DPO 在线基准](./86_DPO_Online_Benchmark.md)
- [06 Project Decision and Delivery | 项目决策与交付](../topic_discussion/post_training_optimization/06_project_decision_and_delivery.md)
### Step 1: 定义组内对齐项目目标

- 固定 baseline、group 组织方式、奖励口径和评测窗口。
- 明确 candidate 想回答的是“组内优势是否更稳、奖励波动是否更低、group 覆盖是否足够”。
- 输出的是是否进入下一轮 GRPO 验证，而不是只给一段统计代码。
### Step 2: baseline 和 group 覆盖先要合法

- 每组至少要有足够候选，不能把单样本噪声当成稳定优势。
- baseline 先要给出可比较的 reward mean、reward std 和 advantage margin。
- 如果 group 覆盖本身不够，candidate 再高的均值也不能直接 adopt。
### Step 3: 用统一口径比较收益与代价

- GRPO 项目必须同时看 reward mean、组内波动和最差组表现，不能只挑平均奖励下结论。
- `reward_mean` 回答整体奖励是否变好。
- `avg_std` 回答组内波动是否压住。
- `worst_group_mean` 回答最差组有没有明显拖后腿，避免只看平均数。
### Step 4: 输出项目结论

- GRPO 项目最终不是输出“平均奖励有没有涨”，而是输出这套 group-wise 对齐方案在当前分组口径下是否值得继续保留、微调或放弃。
- 最终建议统一为 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回奖励设计、候选生成和分组策略，而不是直接扩大裁剪范围。
#### 图解：16-50-84 如何收束到 85 GRPO 项目

```text
16 GRPO loss -> 50 preference/eval -> 84 offline alignment baseline -> 85 groupwise project
```
项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | reward_mean、avg_std、worst_group_mean、avg_group_size | 保证比较口径合法 |
| candidate | group 统计、优势稳定性、最差组表现 | 解释组内收益来源 |
| 对比 | 均值增益、波动变化、覆盖风险 | 判断是否值得继续训练 |
| 决策 | accept / tune / reject | 输出项目结论 |

```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：组内统计、baseline 对比、项目决策
# 目标：把 group-wise 对齐整理成 baseline -> candidate -> decision 闭环

def summarize_grpo_runs(runs: List[Dict[str, float]]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_grpo_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_grpo_project_decision(
    baseline: Dict[str, float], candidate: Dict[str, float], min_reward_gain: float, min_group_size: float
) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_grpo_project_template():
    runs = [
        {"group_size": 3, "reward_mean": 0.45, "reward_std": 0.22},
        {"group_size": 4, "reward_mean": 0.52, "reward_std": 0.18},
        {"group_size": 3, "reward_mean": 0.39, "reward_std": 0.20},
    ]
    summary = summarize_grpo_runs(runs)
    assert summary["group_count"] == 3
    assert round(summary["avg_group_size"], 4) == 3.3333
    assert round(summary["reward_mean"], 4) == 0.4533
    assert round(summary["worst_group_mean"], 4) == 0.39

    baseline = {"avg_group_size": 3.0, "reward_mean": 0.40, "avg_std": 0.28, "worst_group_mean": 0.30}
    comparison = compare_grpo_to_baseline(baseline, summary)
    assert round(comparison["reward_gain"], 4) == 0.0533
    assert round(comparison["std_delta"], 4) == -0.08
    assert round(comparison["worst_group_gain"], 4) == 0.09

    decision = recommend_grpo_project_decision(baseline, summary, min_reward_gain=0.04, min_group_size=3.0)
    assert decision["decision"] == "accept"
    assert decision["next_action"] == "promote_to_groupwise_eval"

    weak_candidate = {"avg_group_size": 3.0, "reward_mean": 0.43, "avg_std": 0.30, "worst_group_mean": 0.31}
    weak_decision = recommend_grpo_project_decision(baseline, weak_candidate, min_reward_gain=0.04, min_group_size=3.0)
    assert weak_decision["decision"] == "tune"

    bad_candidate = {"avg_group_size": 2.0, "reward_mean": 0.34, "avg_std": 0.45, "worst_group_mean": 0.10}
    bad_decision = recommend_grpo_project_decision(baseline, bad_candidate, min_reward_gain=0.04, min_group_size=3.0)
    assert bad_decision["decision"] == "reject"


test_grpo_project_template()
print("测试通过：GRPO 组内项目模板可以工作。")
```

🛑 **STOP HERE** 🛑

请先尝试自己完成代码并跑通测试。如果你在 Colab 中运行，并且暂时没有思路，再继续看下面的参考答案。
## 参考代码与解析

```python
from typing import Dict, List


def summarize_grpo_runs(runs: List[Dict[str, float]]) -> Dict[str, float]:
    group_count = len(runs)
    avg_group_size = sum(item.get("group_size", 0.0) for item in runs) / group_count if runs else 0.0
    reward_mean = sum(item.get("reward_mean", 0.0) for item in runs) / group_count if runs else 0.0
    avg_std = sum(item.get("reward_std", 0.0) for item in runs) / group_count if runs else 0.0
    worst_group_mean = min((item.get("reward_mean", 0.0) for item in runs), default=0.0)
    return {
        "group_count": group_count,
        "avg_group_size": round(avg_group_size, 4),
        "reward_mean": round(reward_mean, 4),
        "avg_std": round(avg_std, 4),
        "worst_group_mean": round(worst_group_mean, 4),
    }


def compare_grpo_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    return {
        "reward_gain": round(candidate.get("reward_mean", 0.0) - baseline.get("reward_mean", 0.0), 4),
        "std_delta": round(candidate.get("avg_std", 0.0) - baseline.get("avg_std", 0.0), 4),
        "worst_group_gain": round(candidate.get("worst_group_mean", 0.0) - baseline.get("worst_group_mean", 0.0), 4),
        "group_size_delta": round(candidate.get("avg_group_size", 0.0) - baseline.get("avg_group_size", 0.0), 4),
    }


def recommend_grpo_project_decision(
    baseline: Dict[str, float], candidate: Dict[str, float], min_reward_gain: float, min_group_size: float
) -> Dict[str, object]:
    comparison = compare_grpo_to_baseline(baseline, candidate)
    if candidate.get("avg_group_size", 0.0) < min_group_size:
        return {
            "decision": "reject",
            "reason": "group 覆盖不足，当前组内结论不可靠",
            "next_action": "expand_candidate_generation",
        }
    if (
        comparison["reward_gain"] >= min_reward_gain
        and comparison["std_delta"] <= -0.05
        and comparison["worst_group_gain"] >= 0.05
    ):
        return {
            "decision": "accept",
            "reason": "candidate 同时改善了组内均值、波动和最差组表现",
            "next_action": "promote_to_groupwise_eval",
        }
    if comparison["reward_gain"] >= 0.0 and comparison["worst_group_gain"] >= 0.0:
        return {
            "decision": "tune",
            "reason": "奖励已有改善，但组内波动还不够稳",
            "next_action": "refine_reward_design_or_sampling",
        }
    return {
        "decision": "reject",
        "reason": "candidate 没有形成稳定的组内收益",
        "next_action": "fallback_to_reward_audit",
    }
```

### 解析

这页现在按 `summarize -> compare -> decide` 的最小 GRPO 项目闭环组织，不再只看平均奖励。

#### TODO 1

- 实现方式：统一汇总 group 数量、平均组大小、平均奖励、平均波动和最差组表现。
- 关键点：这里同时记录均值和最差组，避免只看平均奖励掩盖尾部风险。
- 项目意义：先把 group-wise 候选摘要做平，后面才能判断这套对齐策略是否真的稳定。

#### TODO 2

- 实现方式：统一计算 reward、波动、最差组和组覆盖的变化。
- 关键点：这些增量都按 candidate - baseline 计算，方向必须一致，才能正确判断收益和风险。
- 项目意义：这一步把 GRPO 从“单看 reward mean”转成“均值、波动和最差组能否一起改善”的项目比较。

#### TODO 3

- 实现方式：先看 group 覆盖，再按 reward gain、波动压缩和最差组改善输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应奖励已有改善，但组内波动还没有一起收稳；覆盖不足则直接 `reject`。
- 项目意义：GRPO 项目最后要回答的是“这套 group-wise 对齐是否值得继续放大训练”，而不是只看一组 reward 平均值。
