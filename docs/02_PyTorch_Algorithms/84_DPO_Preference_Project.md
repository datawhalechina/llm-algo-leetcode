# 84. DPO Preference Project | DPO 偏好优化项目
**难度：** Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `DPO`, `项目评估` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 本节导读

本节围绕一个离线 DPO 方案是否值得继续训练展开。你需要先检查偏好数据和 chosen / rejected 样本的合法性，再比较 baseline 与 candidate 的 loss、margin 和有效样本比例。最后判断当前收益是否足以覆盖数据风险，并给出继续训练、调整数据或停止实验的建议。

**关键词：** `preference pair`, `margin`, `baseline`, `alignment decision`
## 前置阅读

**导语：** 先把 DPO 损失、偏好优化主线、偏好数据评测和在线 DPO 背景理顺，再进入这个项目；本节默认你已经知道离线偏好优化的基本对象，重点转向 DPO 方案是否值得继续训练。
- [15. DPO Loss Tutorial | DPO 损失教程](./15_DPO_Loss_Tutorial.md)
- [2.4](./2_4.md)
- [50. Preference Data and Evaluation | 偏好数据与评测](./50_Preference_Data_and_Evaluation.md)
- [51. Online DPO | 在线 DPO](./51_Online_DPO.md)
- [后训练与对齐专题入口](../topic_discussion/post_training_optimization/intro.md)
- [06 Project Decision and Delivery | 项目决策与交付](../topic_discussion/post_training_optimization/06_project_decision_and_delivery.md)

## 相关阅读

**导语：** 做完离线 DPO 项目后，最自然的下一步是继续看 group-wise 对齐，或把离线结论推进到在线基准。
- [85. GRPO Groupwise Alignment Project | GRPO 组内对齐项目](./85_GRPO_Groupwise_Alignment_Project.md)
- [86. DPO Online Benchmark | DPO 在线基准](./86_DPO_Online_Benchmark.md)
### Step 1: 定义离线偏好项目目标

- 固定 baseline、偏好数据模板、beta 和评测口径。
- 明确 candidate 想回答的是“margin 是否更稳、loss 是否更低、chosen 是否更 consistently 胜出”。
- 输出的不是公式推导，而是一个可以继续训练还是应该先停下修数据的项目结论。
### Step 2: baseline 和数据合法性先要成立

- 偏好对至少要有足够样本数，且 chosen / rejected 字段不能缺失。
- baseline 先要给出可比较的 loss 和 margin 口径。
- 如果数据本身不合法，后面的 candidate 再好也不能直接 adopt。
### Step 3: 用统一口径比较收益与代价

- DPO 项目必须同时看 loss、margin 和 valid_ratio，不能只挑单项 loss 收益下结论。
- `loss` 回答 candidate 是否更容易优化。
- `margin_mean` 和 `margin_min` 回答 chosen 是否真的被稳定拉开。
- `valid_ratio` 回答偏好数据是否足够干净，避免拿脏数据做错结论。
### Step 4: 输出项目结论

- DPO 项目最终不是输出“loss 有没有降”，而是输出这套离线偏好优化方案在当前数据口径下是否值得继续保留、微调或放弃。
- 最终建议统一为 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回偏好数据、模板一致性和 beta，而不是直接加训练步数。
#### 图解：15-50-51 如何收束到 84 DPO 项目

```text
15 DPO loss -> 50 preference data/eval -> 51 online DPO context -> 84 offline DPO project
```
项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | loss、margin、valid_ratio | 保证比较口径合法 |
| candidate | beta、loss、margin、样本覆盖 | 解释离线收益来源 |
| 对比 | loss 改善、margin 改善、coverage 风险 | 判断是否值得继续训练 |
| 决策 | accept / tune / reject | 输出项目结论 |

```python
from typing import Dict, List

```


```python
# 4 个核心 TODO：样本审计、候选汇总、baseline 对比、项目决策
# 目标：把离线偏好优化整理成 baseline -> candidate -> decision 闭环

def audit_preference_batch(samples: List[Dict[str, object]]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_dpo_candidate(losses: List[float], margins: List[float], audit: Dict[str, float], beta: float) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_dpo_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_dpo_project_decision(
    baseline: Dict[str, float], candidate: Dict[str, float], min_margin_gain: float, min_valid_ratio: float
) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_dpo_preference_project_template():
    samples = [
        {"prompt": "p1", "chosen": "a", "rejected": "b"},
        {"prompt": "p2", "chosen": "c", "rejected": "d"},
        {"prompt": "p3", "chosen": "e", "rejected": ""},
    ]
    audit = audit_preference_batch(samples)
    assert audit["sample_count"] == 3
    assert audit["valid_count"] == 2
    assert round(audit["valid_ratio"], 4) == 0.6667

    baseline = {"loss_mean": 0.72, "margin_mean": 0.12, "margin_min": -0.08, "valid_ratio": 0.95}
    candidate = summarize_dpo_candidate([0.60, 0.58, 0.57], [0.24, 0.21, 0.18], {"valid_ratio": 0.97}, beta=0.1)
    assert round(candidate["loss_mean"], 4) == 0.5833
    assert candidate["margin_min"] == 0.18
    comparison = compare_dpo_to_baseline(baseline, candidate)
    assert round(comparison["loss_delta"], 4) == -0.1367
    assert round(comparison["margin_gain"], 4) == 0.09
    decision = recommend_dpo_project_decision(baseline, candidate, min_margin_gain=0.05, min_valid_ratio=0.95)
    assert decision["decision"] == "accept"
    assert decision["next_action"] == "promote_to_longer_dpo_run"

    weak_candidate = {"loss_mean": 0.64, "margin_mean": 0.15, "margin_min": -0.05, "valid_ratio": 0.96}
    weak_decision = recommend_dpo_project_decision(baseline, weak_candidate, min_margin_gain=0.05, min_valid_ratio=0.95)
    assert weak_decision["decision"] == "tune"

    bad_candidate = {"loss_mean": 0.80, "margin_mean": 0.05, "margin_min": -0.20, "valid_ratio": 0.80}
    bad_decision = recommend_dpo_project_decision(baseline, bad_candidate, min_margin_gain=0.05, min_valid_ratio=0.95)
    assert bad_decision["decision"] == "reject"


test_dpo_preference_project_template()
print("测试通过：DPO 偏好项目模板可以工作。")
```

🛑 **STOP HERE** 🛑

请先尝试自己完成代码并跑通测试。如果你在 Colab 中运行，并且暂时没有思路，再继续看下面的参考答案。
## 参考代码与解析

```python
from typing import Dict, List


def audit_preference_batch(samples: List[Dict[str, object]]) -> Dict[str, float]:
    valid_count = 0
    for sample in samples:
        prompt = str(sample.get("prompt", "")).strip()
        chosen = str(sample.get("chosen", "")).strip()
        rejected = str(sample.get("rejected", "")).strip()
        if prompt and chosen and rejected:
            valid_count += 1
    sample_count = len(samples)
    valid_ratio = valid_count / sample_count if sample_count else 0.0
    return {"sample_count": sample_count, "valid_count": valid_count, "valid_ratio": round(valid_ratio, 4)}


def summarize_dpo_candidate(losses: List[float], margins: List[float], audit: Dict[str, float], beta: float) -> Dict[str, float]:
    loss_mean = sum(losses) / len(losses) if losses else 0.0
    margin_mean = sum(margins) / len(margins) if margins else 0.0
    margin_min = min(margins) if margins else 0.0
    return {
        "beta": beta,
        "loss_mean": round(loss_mean, 4),
        "margin_mean": round(margin_mean, 4),
        "margin_min": round(margin_min, 4),
        "valid_ratio": float(audit.get("valid_ratio", 0.0)),
    }


def compare_dpo_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    return {
        "loss_delta": round(candidate.get("loss_mean", 0.0) - baseline.get("loss_mean", 0.0), 4),
        "margin_gain": round(candidate.get("margin_mean", 0.0) - baseline.get("margin_mean", 0.0), 4),
        "margin_floor_delta": round(candidate.get("margin_min", 0.0) - baseline.get("margin_min", 0.0), 4),
        "valid_ratio_delta": round(candidate.get("valid_ratio", 0.0) - baseline.get("valid_ratio", 0.0), 4),
    }


def recommend_dpo_project_decision(
    baseline: Dict[str, float], candidate: Dict[str, float], min_margin_gain: float, min_valid_ratio: float
) -> Dict[str, object]:
    comparison = compare_dpo_to_baseline(baseline, candidate)
    if candidate.get("valid_ratio", 0.0) < min_valid_ratio:
        return {
            "decision": "reject",
            "reason": "偏好数据覆盖不达标，当前比较口径不可靠",
            "next_action": "fix_preference_data_pipeline",
        }
    if (
        comparison["loss_delta"] < 0.0
        and comparison["margin_gain"] >= min_margin_gain
        and comparison["margin_floor_delta"] >= 0.05
    ):
        return {
            "decision": "accept",
            "reason": "candidate 同时改善了 loss、平均 margin 和最差样本 margin",
            "next_action": "promote_to_longer_dpo_run",
        }
    if comparison["loss_delta"] < 0.0 and candidate.get("margin_mean", 0.0) >= baseline.get("margin_mean", 0.0):
        return {
            "decision": "tune",
            "reason": "loss 已改善，但 margin 增益还不够稳",
            "next_action": "refine_beta_or_prompt_template",
        }
    return {
        "decision": "reject",
        "reason": "candidate 没有稳定改善 DPO 核心指标",
        "next_action": "fallback_to_data_audit",
    }
```

### 解析

这页现在按 `audit -> summarize -> compare -> decide` 的最小离线 DPO 项目闭环组织，不再只看 loss 变化。

#### TODO 1

- 实现方式：逐个样本检查 prompt、chosen 和 rejected 是否都非空，再汇总 valid_ratio。
- 关键点：偏好数据 coverage 不足时，后面的 loss 和 margin 比较都没有解释力。
- 项目意义：先把样本合法性收进项目结论，避免离线 DPO 退回成只看训练曲线的页面。

#### TODO 2

- 实现方式：统一计算 candidate 的 `loss_mean`、`margin_mean` 和 `margin_min`，并保留 beta 与 valid_ratio。
- 关键点：这里同时记录平均 margin 和最差样本 margin，避免只看均值掩盖尾部风险。
- 项目意义：这一步把 DPO candidate 从单次训练结果，转成可与 baseline 同口径比较的项目摘要。

#### TODO 3

- 实现方式：统一计算 loss、平均 margin、最差样本 margin 和 coverage 的变化。
- 关键点：loss_delta 用 candidate - baseline，越小越好；其余增益越大越好，方向必须一致。
- 项目意义：这一步把离线 DPO 从“单一 loss 改善”转成“收益与覆盖风险能否一起成立”的项目比较。

#### TODO 4

- 实现方式：先看 valid_ratio 是否达标，再按 loss、平均 margin 和最差样本 margin 输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应 loss 已改善，但 margin 增益还不够稳；coverage 不达标则直接 `reject`。
- 项目意义：离线 DPO 项目最后要回答的是“值不值得继续训练”，而不是只看某条 loss 曲线是否下降。
