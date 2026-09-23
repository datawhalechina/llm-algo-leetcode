# 51. Online DPO | 在线 DPO
**难度：** Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `DPO`, `在线优化` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/51_Online_DPO.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线 DPO 的关键，不是把 DPO 简单换成流式输入，而是把反馈流、更新频率、评测回路和稳定性边界一起管住。`50` 先回答“偏好数据和评测口径是否可信”，`15` 回答“离线 pairwise preference 如何优化”，而 `51` 再往前推进一步：当新反馈持续进来时，要不要立刻更新、怎么更新、更新后会不会把模型带偏。

这一节不做完整线上系统，也不复现工业级在线训练平台，而是先用一个最小教学模拟把三件事串起来：在线反馈流怎么汇总，online update 相比 offline run 到底带来了多少收益/代价，以及什么情况下才值得采用在线 DPO。学完后，你应该能看清“反馈进入 -> 在线更新 -> 稳定性复核 -> 是否上线”这条最小决策链。

**关键词：** `online feedback`, `update cadence`, `win-rate gain`, `stability boundary`

---

## 前置阅读

**导语：** 这一节同时承接离线偏好优化和对齐评测两条线：先知道 DPO / GRPO 在优化什么，再回来看在线反馈为什么会把训练闭环变得更敏感。
- [15. DPO Loss Tutorial | DPO 损失教程](./15_DPO_Loss_Tutorial.md)
- [16. GRPO Loss Tutorial | GRPO 损失教程](./16_GRPO_Loss_Tutorial.md)
- [50. Preference Data and Evaluation | 偏好数据与评测](./50_Preference_Data_and_Evaluation.md)


## 相关阅读

**导语：** 学完在线 DPO 后，下一步重点不是继续背更新公式，而是看它怎样进入阈值判断、项目验证和 benchmark 闭环，确认“更新更快”是否真的等于“对齐更好”。
- [52. Reserved 52 | 通用预留](./52_Reserved_52.md)
- [84. DPO Preference Project | DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [86. DPO Online Benchmark | DPO 在线 Benchmark](./86_DPO_Online_Benchmark.md)

---

### Step 1: 定义在线 DPO 要解决的问题

- 固定初始模型、反馈流、更新频率、batch size 和评估窗口。
- 明确在线样本来源、过滤规则和是否允许重复反馈。
- 统一比较口径：win rate、loss 波动、更新时间和安全阈值。

在线闭环应明确记录：

`online request/feedback → filter/deduplicate/risk check → new preference pair or reward sample → policy update / checkpoint → independent eval and safety gate → accept/defer/rollback`

与离线 DPO 相比，在线 DPO 还要关注 `policy freshness`（反馈产生时使用的 policy 版本与当前 policy 的距离）、reference 更新策略、反馈分布漂移和更新期间的回滚点。

#### 图解：50-15-16-84-85 如何收束到 51 在线 DPO

```text
50 Preference -> 15 DPO -> 16 GRPO -> 84/85 offline projects -> 51 online update
```

### 提示

- `TODO 1` 先统计总反馈、接受反馈和重复反馈，再计算 `accept_rate`。
- `TODO 2` 先比较 `win_rate / stability / update_cost` 的变化，再输出三个增量指标。
- `TODO 3` 先拿到 comparison，再结合 `min_win_rate_gain` 和稳定性边界判断是否采用在线方案。


```python
from typing import Dict, List

```


```python
def summarize_online_preferences(feedback_stream: List[Dict[str, object]]) -> Dict[str, float]:
    """
    TODO 1: 汇总在线反馈流。
    """
    # 提示：先建立 seen，再统计 total_feedback、accepted_feedback、duplicate_count 和 accept_rate。
    # total_feedback = ???
    # accepted_feedback = ???
    # duplicate_count = ???
    # accept_rate = ???
    raise NotImplementedError


def compare_online_update(offline_run: Dict[str, float], online_run: Dict[str, float]) -> Dict[str, float]:
    """
    TODO 2: 比较 offline 和 online 更新结果。
    """
    # 提示：先分别计算 win_rate_gain、stability_delta 和 cost_delta，再返回字典。
    # win_rate_gain = ???
    # stability_delta = ???
    # cost_delta = ???
    raise NotImplementedError


def recommend_online_dpo(offline_run: Dict[str, float], online_run: Dict[str, float], min_win_rate_gain: float) -> Dict[str, object]:
    """
    TODO 3: 输出是否采用在线 DPO。
    """
    # 提示：先拿到 comparison，再判断 adopt_online，最后给出 reason。
    # comparison = ???
    # adopt_online = ???
    # reason = ???
    raise NotImplementedError

```


```python
def test_online_dpo_template():
    try:
        feedback_stream = [
            {'id': 'a', 'accepted': True},
            {'id': 'b', 'accepted': False},
            {'id': 'a', 'accepted': True},
        ]
        summary = summarize_online_preferences(feedback_stream)
        assert summary['total_feedback'] == 3
        assert summary['accepted_feedback'] == 2
        assert summary['duplicate_count'] == 1
        assert abs(summary['accept_rate'] - 2 / 3) < 1e-8
        offline = {'name': 'offline', 'win_rate': 0.52, 'stability': 0.82, 'update_cost': 1.0}
        online = {'name': 'online', 'win_rate': 0.59, 'stability': 0.79, 'update_cost': 1.2}
        comparison = compare_online_update(offline, online)
        assert comparison['win_rate_gain'] == 0.07
        decision = recommend_online_dpo(offline, online, min_win_rate_gain=0.05)
        assert decision['adopt_online'] is True
        print('测试通过：在线 DPO 模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_online_dpo_template()

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
def summarize_online_preferences(feedback_stream: List[Dict[str, object]]) -> Dict[str, float]:
    """
    TODO 1: 汇总在线反馈流。
    """
    # 提示：先建立 seen，再统计 total_feedback、accepted_feedback、duplicate_count 和 accept_rate。
    total_feedback = len(feedback_stream)
    accepted_feedback = 0
    duplicate_count = 0
    seen = set()
    for item in feedback_stream:
        feedback_id = item.get('id')
        if item.get('accepted', False):
            accepted_feedback += 1
        if feedback_id in seen:
            duplicate_count += 1
        else:
            seen.add(feedback_id)
    accept_rate = accepted_feedback / total_feedback if total_feedback else 0.0
    return {
        'total_feedback': total_feedback,
        'accepted_feedback': accepted_feedback,
        'duplicate_count': duplicate_count,
        'accept_rate': accept_rate,
    }


def compare_online_update(offline_run: Dict[str, float], online_run: Dict[str, float]) -> Dict[str, float]:
    """
    TODO 2: 比较 offline 和 online 更新结果。
    """
    # 提示：先分别计算 win_rate_gain、stability_delta 和 cost_delta，再返回字典。
    win_rate_gain = round(online_run.get('win_rate', 0.0) - offline_run.get('win_rate', 0.0), 4)
    stability_delta = round(online_run.get('stability', 0.0) - offline_run.get('stability', 0.0), 4)
    cost_delta = round(online_run.get('update_cost', 0.0) - offline_run.get('update_cost', 0.0), 4)
    return {
        'win_rate_gain': win_rate_gain,
        'stability_delta': stability_delta,
        'cost_delta': cost_delta,
    }


def recommend_online_dpo(offline_run: Dict[str, float], online_run: Dict[str, float], min_win_rate_gain: float) -> Dict[str, object]:
    """
    TODO 3: 输出是否采用在线 DPO。
    """
    # 提示：先拿到 comparison，再判断 adopt_online，最后给出 reason。
    comparison = compare_online_update(offline_run, online_run)
    adopt_online = comparison['win_rate_gain'] >= min_win_rate_gain and comparison['stability_delta'] >= -0.05
    reason = '收益覆盖稳定性代价' if adopt_online else '收益不足或稳定性下降过大'
    return {'adopt_online': adopt_online, 'reason': reason}

```

### 解析

TODO 1：`summarize_online_preferences` 先扫描反馈流，统计总条数、接受条数、重复反馈数，再计算 `accept_rate`，用于判断在线偏好数据是否干净且足够稳定。

TODO 2：`compare_online_update` 对比 offline 与 online 两次运行的核心指标，重点看 `win_rate_gain`、`stability_delta` 和 `cost_delta`，把收益与代价拆开表达。

TODO 3：`recommend_online_dpo` 基于比较结果做上线建议。只有当胜率提升达到阈值，且稳定性下降没有超过容忍边界时，才建议采用在线 DPO。
