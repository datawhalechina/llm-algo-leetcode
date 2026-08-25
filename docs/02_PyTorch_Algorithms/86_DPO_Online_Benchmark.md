# 86. DPO Online Benchmark | DPO 在线基准
**难度：** Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `在线对齐`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/86_DPO_Online_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节要求你在固定反馈流、更新窗口和安全阈值后，评估在线 DPO 是否具备上线条件。比较 baseline 与 online candidate 的胜率变化、更新时延、训练稳定性和安全指标，并检查收益是否能在允许的更新成本内持续复现。最终输出上线、继续灰度或暂停更新的建议。

**关键词：** `online update`, `win rate`, `stability`, `safety gate`

---
## 前置阅读

**导语：** 先把 DPO / GRPO 损失、偏好数据评测、在线 DPO 背景和离线对齐项目理顺，再进入这个 benchmark；本节默认你已经知道后训练对齐的基本对象，重点转向在线方案是否值得上线。
- [15. DPO Loss Tutorial | DPO 损失教程](./15_DPO_Loss_Tutorial.md)
- [16. GRPO Loss Tutorial | GRPO 损失教程](./16_GRPO_Loss_Tutorial.md)
- [50. Preference Data and Evaluation | 偏好数据与评测](./50_Preference_Data_and_Evaluation.md)
- [51. Online DPO | 在线 DPO](./51_Online_DPO.md)
- [84. DPO Preference Project | DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [85. GRPO Groupwise Alignment Project | GRPO 组内对齐项目](./85_GRPO_Groupwise_Alignment_Project.md)

## 相关阅读

**导语：** 做完在线 DPO benchmark 后，回到对齐专题和项目决策页，把离线和在线项目统一放进交付决策闭环。
- [后训练与对齐专题入口](../topic_discussion/post_training_alignment/intro.md)
- [06 Project Decision and Delivery | 项目决策与交付](../topic_discussion/post_training_alignment/06_project_decision_and_delivery.md)
### Step 1: 定义在线 benchmark 目标

- 固定初始模型、偏好流、更新频率、batch size 和评估窗口。
- 明确 candidate 的在线更新规则、采样策略和安全阈值。
- 统一记录偏好胜率、更新步时、波动性和成本。

### Step 2: 先确认 baseline 和安全口径稳定

- baseline 先要有稳定的 win rate 和安全分数。
- 如果 baseline 本身波动很大，在线收益很容易被噪声放大。
- 上线前必须显式记录最低稳定性 / 安全阈值。

### Step 3: 用统一口径比较收益与代价

- 在线 DPO 项目必须同时看胜率增量、更新时延、稳定性和安全阈值，不能只挑单项 win rate 收益下结论。
- 胜率增量回答“值不值得更好”。
- 更新时延回答“代价能不能接受”。
- 稳定性和安全阈值回答“能不能上线”。

### Step 4: 输出 benchmark 结论

- 在线 DPO 最终不是输出“胜率有没有涨”，而是输出这套在线更新方案在当前反馈流下是否值得继续保留、微调或上线。
- 最终建议统一为 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回更新频率、反馈流质量和安全阈值，而不是只盯住 win rate。 
#### 图解：50-51-84-85 如何收束到 86 在线基准

```text
50 Preference -> 51 Online DPO -> 84/85 offline projects -> 86 Online benchmark
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | win rate、stability、安全阈值 | 保证上线前比较合法 |
| candidate | 更新频率、update cost、增益 | 解释在线收益来源 |
| 对比 | 胜率增量、稳定性变化、更新时间 | 判断是否值得 adopt |
| 决策 | accept / tune / reject | 输出 benchmark 结论 |


```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：run 汇总、baseline 对比、项目判断
# 目标：把在线偏好更新整理成 benchmark 报告

def summarize_online_dpo_runs(runs: List[Dict[str, float]]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_online_dpo_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_online_dpo_run(baseline: Dict[str, float], candidate: Dict[str, float], min_win_rate_gain: float, min_safety_score: float) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_online_dpo_benchmark_template():
    baseline = {'name': 'baseline', 'win_rate': 0.52, 'update_ms': 90, 'stability': 0.80, 'safety_score': 0.83}
    candidate = {'name': 'online_dpo', 'win_rate': 0.60, 'update_ms': 105, 'stability': 0.78, 'safety_score': 0.81}
    summary = summarize_online_dpo_runs([baseline, candidate])
    assert summary['run_count'] == 2
    assert summary['best_win_rate_run'] == 'online_dpo'
    comparison = compare_online_dpo_to_baseline(baseline, candidate)
    assert comparison['win_rate_gain'] == 0.08
    assert comparison['update_delta_ms'] == 15
    assert comparison['safety_delta'] == -0.02
    decision = recommend_online_dpo_run(baseline, candidate, min_win_rate_gain=0.05, min_safety_score=0.8)
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_online_eval'


test_online_dpo_benchmark_template()
print('测试通过：DPO 在线基准模板可以工作。')

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
def summarize_online_dpo_runs(runs: List[Dict[str, float]]) -> Dict[str, object]:
    best = max(runs, key=lambda item: item.get('win_rate', 0.0))
    avg_stability = sum(item.get('stability', 0.0) for item in runs) / len(runs) if runs else 0.0
    return {'run_count': len(runs), 'best_win_rate_run': best.get('name', 'run'), 'avg_stability': avg_stability}


def compare_online_dpo_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    return {
        'win_rate_gain': round(candidate.get('win_rate', 0.0) - baseline.get('win_rate', 0.0), 4),
        'update_delta_ms': round(candidate.get('update_ms', 0.0) - baseline.get('update_ms', 0.0), 4),
        'stability_delta': round(candidate.get('stability', 0.0) - baseline.get('stability', 0.0), 4),
        'safety_delta': round(candidate.get('safety_score', 0.0) - baseline.get('safety_score', 0.0), 4),
    }


def recommend_online_dpo_run(baseline: Dict[str, float], candidate: Dict[str, float], min_win_rate_gain: float, min_safety_score: float) -> Dict[str, object]:
    comparison = compare_online_dpo_to_baseline(baseline, candidate)
    if comparison['win_rate_gain'] >= min_win_rate_gain and candidate.get('safety_score', 0.0) >= min_safety_score and comparison['stability_delta'] >= -0.05:
        return {'decision': 'accept', 'reason': '胜率收益、安全阈值和稳定性都达标', 'next_action': 'promote_to_online_eval'}
    if comparison['win_rate_gain'] >= min_win_rate_gain and candidate.get('safety_score', 0.0) >= min_safety_score:
        return {'decision': 'tune', 'reason': '胜率收益可用，但稳定性仍需压波动', 'next_action': 'refine_update_frequency'}
    return {'decision': 'reject', 'reason': '胜率收益不足或安全阈值不达标', 'next_action': 'fallback_to_offline_alignment'}

```

### 解析

这页现在按 `summarize -> compare -> decide` 的最小在线 DPO 项目闭环组织，不再只看 win rate 变化。

#### TODO 1

- 实现方式：先汇总 run 数量和平均稳定性，再找出 win rate 最高的 run。
- 关键点：`best_win_rate_run` 只是帮助定位最值得回看的 candidate，不等于最终项目结论。
- 项目意义：先把 benchmark run 摘要做平，后面才能比较胜率收益和在线系统代价。

#### TODO 2

- 实现方式：统一计算 win rate、更新时间、稳定性和安全分的变化。
- 关键点：这些变化都按 candidate - baseline 计算，才能在同一方向上判断收益和风险。
- 项目意义：这一步把在线 DPO 从“单看 win rate”转成“收益、安全和系统代价能否一起成立”的项目比较。

#### TODO 3

- 实现方式：先看 win rate gain 是否达标，再按安全阈值和稳定性输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应胜率收益可用，但更新频率、反馈流质量或稳定性还没有一起收稳。
- 项目意义：在线 DPO 项目最后要回答的是“值不值得推进到在线评估”，而不是只看胜率曲线有没有上涨。
