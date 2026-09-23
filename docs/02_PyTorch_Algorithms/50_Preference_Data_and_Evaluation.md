# 50. Preference Data and Evaluation | 偏好数据与对齐评测

**难度：** Medium | **环境：** CPU-first | **标签：** `后训练对齐`, `偏好数据`, `评测` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

很多人一提到对齐，就立刻想到 DPO、GRPO 或 reward model，但真正决定后续实验是否可信的，往往不是优化公式本身，而是前面那层更朴素的基础设施：偏好数据到底干不干净、评测口径到底稳不稳定、不同实验之间是不是在用同一套判断标准。

这是一节**机制原理节**：它和 `15`、`16` 是前后承接关系。`15` 主讲 pairwise preference 下的 DPO 损失，`16` 主讲 group-wise 比较下的 GRPO 损失；而 `50` 先退回到更上游的一层，回答“这些对齐方法到底建立在什么样的数据口径和最小评测口径之上”。

这一节不做完整 judge 系统或复杂评测平台，而是先用最小汇总器把“样本有没有空值、prompt 是否重复聚集、pair 长度是否离谱、win-rate 怎么统一计算”这条链路固定下来。一个实用判断可以先保持简单：如果偏好数据本身就脏，或者评测口径来回变动，那么后面就算 DPO / GRPO 指标看起来上涨，也很难说明优化真的有效。

**关键词：** `preference data`, `win-rate`, `pairwise evaluation`

---
## 前置阅读

**导语：** 先把 pairwise 偏好、group-wise 比较和对齐组页的位置补齐，再进入这一节，会更容易把“优化目标”与“数据/评测口径”区分开。

- [15. DPO Loss Tutorial | DPO 损失教程](./15_DPO_Loss_Tutorial.md)
- [16. GRPO Loss Tutorial | GRPO 损失教程](./16_GRPO_Loss_Tutorial.md)
- [2.4](./2_4.md)
## 相关阅读

**导语：** 学完这页后，下一步重点不是继续背数据字段，而是看这些最小口径如何接到在线更新、偏好项目和 benchmark 验证里，确认“模型变好了”到底是不是建立在可复现的数据与评测标准上。

- [51. Online DPO | 在线 DPO](./51_Online_DPO.md)
- [84. DPO Preference Project | DPO 偏好项目](./84_DPO_Preference_Project.md)
- [86. DPO Online Benchmark | DPO 在线 Benchmark](./86_DPO_Online_Benchmark.md)
### Step 1: 偏好数据长什么样

典型偏好样本至少包含 `prompt`、`chosen`、`rejected`，有些任务还会带 `source`、`judge`、`score` 和 `group_id`。

偏好数据进入训练前，先固定三类契约：

| 契约 | 要检查什么 | 不满足时的风险 |
|:---|:---|:---|
| 样本契约 | prompt 非空；chosen/rejected 都存在；两条回复针对同一问题 | DPO/GRPO 学到格式差异，而不是偏好差异 |
| 分组契约 | 同一 prompt 的候选有稳定的 `group_id`；训练、验证、测试不交叉泄漏 | win-rate 或组内优势被重复样本虚高 |
| 来源契约 | 记录数据来源、judge/人工标注方式和版本 | 不同数据源混合后无法解释结果 |

这张表不是额外的清洗平台，而是判断一条偏好样本能否进入训练和评测的最小准入标准。
### Step 2: 对齐评测看什么

评测先固定比较对象、样本切分和判分规则，再计算指标：

| 指标 | 计算对象 | 适合回答的问题 | 主要陷阱 |
|:---|:---|:---|:---|
| `win-rate` | candidate 与 baseline 的成对胜负 | 新模型是否更常被判为更好 | 受 prompt 分布和 tie 处理影响 |
| `pairwise accuracy` | chosen/rejected 的偏好方向 | 数据或 judge 是否复现已知偏好 | 不能单独代表真实任务质量 |
| `judge score` | 每条回答的标量评分 | 质量变化幅度和维度差异 | judge 版本变化会破坏可比性 |
| `safety / style` | 独立约束集 | 是否出现安全或格式回归 | 不能用平均总分掩盖严重回归 |

每次比较至少保留 baseline、candidate、评测集版本、judge 版本、样本数、tie 规则和失败样本；同一批偏好训练数据不能直接充当最终评测集。
### Step 3: 从样本到指标

偏好对齐最怕两件事：数据本身不干净，以及评测口径不稳定。这里先把数据检查和最小指标汇总函数做出来。

推荐的最小评测流程是：

```text
数据准入 → 固定 train/validation/test 切分 → 保存 baseline 输出
        → 运行 candidate → 成对比较与安全检查 → 记录失败案例
        → 综合质量、稳定性和成本，作出 accept / tune / reject
```

如果只有 win-rate 上升、但安全集或独立任务集下降，结论应是 `tune`，不能直接写成对齐成功。
### Step 4: 动手实战

实现一个最小偏好数据汇总器，检查空样本、重复样本和 win-rate 口径；再用同一套 baseline/candidate 约定解释结果。
### 提示

- 先把 `TODO 1` 当成一次最小数据质检：先确认样本总量，再看有没有空值、prompt 是否重复聚集、pair 长度是否异常。
- `empty` 统计的是 `prompt / chosen / rejected` 中任意一项为空的样本数；这类样本如果不先清掉，后面的偏好训练和评测结论会很不稳。
- `duplicate_prompt_groups` 统计的是 prompt 重复出现的组数，不是重复样本总数。这里想看的是“某些 prompt 是否被反复采样”，而不是简单数重复条目。
- `TODO 2` 先统一 win-rate 口径：先校验长度，再统计 `chosen_score > rejected_score` 的次数，最后返回胜出比例。先把评测口径固定，后面接 DPO / GRPO 才有可比性。

```python
from collections import Counter
from dataclasses import dataclass

```


```python
@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


def summarize_preference_dataset(samples):
    """
    TODO 1: 汇总偏好数据的最小统计信息。

    你需要返回：
    - total: 样本总数
    - empty: 空 prompt / chosen / rejected 的样本数
    - duplicate_prompt_groups: prompt 重复的组数
    - avg_pair_tokens: chosen + rejected 的平均 token 近似数（用 split 近似即可）
    """
    # 提示：先算 total，再补 empty、duplicate_prompt_groups、avg_pair_tokens。
    # total = ???
    # empty = ???
    # duplicate_prompt_groups = ???
    # avg_pair_tokens = ???
    raise NotImplementedError


def compute_win_rate(chosen_scores, rejected_scores):
    """
    TODO 2: 计算最小 win-rate。

    规则：chosen_score > rejected_score 记为一次胜出。
    """
    # 提示：先校验长度，再统计 wins，最后返回比例。
    # wins = ???
    # return ???
    raise NotImplementedError
```

### 测试

运行下面的测试，检查你的偏好数据汇总和 win-rate 计算是否正确。

```python
def test_preference_summary():
    try:
        samples = [
            PreferenceSample("Explain LoRA", "LoRA adds adapters.", "LoRA changes all weights."),
            PreferenceSample("Explain DPO", "DPO uses preference pairs.", "DPO needs no reward model at inference."),
            PreferenceSample("Explain DPO", "Short answer.", ""),
            PreferenceSample("", "missing prompt", "missing rejected"),
        ]
        stats = summarize_preference_dataset(samples)
        assert stats["total"] == 4
        assert stats["empty"] == 2
        assert stats["duplicate_prompt_groups"] == 1
        assert stats["avg_pair_tokens"] > 0
        print("preference summary passed")


        win_rate = compute_win_rate([0.8, 0.7, 0.9], [0.2, 0.5, 0.95])
        assert 0.6 < win_rate < 0.8
        assert compute_win_rate([1.0, 0.9], [0.2, 0.1]) == 1.0
        assert compute_win_rate([0.1, 0.2], [0.3, 0.4]) == 0.0
        try:
            compute_win_rate([0.5], [0.2, 0.1])
        except ValueError:
            pass
        else:
            raise AssertionError("length mismatch should raise ValueError")
        print("win rate passed")
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e


test_preference_summary()
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
from collections import Counter
from dataclasses import dataclass

@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


def summarize_preference_dataset(samples):
    """
    TODO 1: 汇总偏好数据的最小统计信息。

    你需要返回：
    - total: 样本总数
    - empty: 空 prompt / chosen / rejected 的样本数
    - duplicate_prompt_groups: prompt 重复的组数
    - avg_pair_tokens: chosen + rejected 的平均 token 近似数（用 split 近似即可）
    """
    # 提示 1: 先算 total，再统计 prompt / chosen / rejected 任意一项为空的样本数。
    # 提示 2: duplicate_prompt_groups 统计的是“重复 prompt 的组数”，不是重复样本总数。
    # 提示 3: avg_pair_tokens 用 split() 近似即可；如果 total == 0，返回 0.0。
    total = len(samples)

    empty = sum(
        1
        for s in samples
        if not s.prompt.strip() or not s.chosen.strip() or not s.rejected.strip()
    )

    duplicate_prompt_groups = sum(
        v > 1 for v in Counter(s.prompt for s in samples).values()
    )

    avg_pair_tokens = (
        sum(len(s.chosen.split()) + len(s.rejected.split()) for s in samples) / total
        if total else 0.0
    )

    return {
        "total": total,
        "empty": empty,
        "duplicate_prompt_groups": duplicate_prompt_groups,
        "avg_pair_tokens": round(avg_pair_tokens, 2),
    }


def compute_win_rate(chosen_scores, rejected_scores):
    """
    TODO 2: 计算最小 win-rate。

    规则：chosen_score > rejected_score 记为一次胜出。
    """
    # 提示 1: 先校验两个 score 列表长度一致，否则直接报错。
    # 提示 2: 只统计 chosen_score > rejected_score 的次数，再除以总样本数。
    # 提示 3: 如果输入为空列表，返回 0.0。
    if len(chosen_scores) != len(rejected_scores):
        raise ValueError("score length mismatch")

    wins = sum(c > r for c, r in zip(chosen_scores, rejected_scores))

    return wins / len(chosen_scores) if chosen_scores else 0.0
```

### 解析

**1. TODO 1：汇总偏好数据的最小统计信息**
- `total` 先确认数据规模；`empty` 用来快速发现 prompt / chosen / rejected 是否有脏样本。
- `duplicate_prompt_groups` 关注的是“同一个 prompt 是否反复出现”，帮助你识别一题多答或采样聚集。
- `avg_pair_tokens` 用最小口径观察 chosen / rejected 的长度，避免样本分布极端失衡却毫无感知。
- 这一组统计不是为了做完整数据平台，而是为了先把“这批偏好数据是否值得继续训练”这件事讲清楚。

**2. TODO 2：计算最小 win-rate**
- 先校验长度一致，避免 `chosen_scores` 和 `rejected_scores` 比较错位。
- 再统计 `chosen_score > rejected_score` 的胜出次数，并返回胜出比例。
- 这不是完整对齐评测体系，而是最轻的一条判断线：模型偏好是否至少在 pairwise 比较里更常赢过 rejected。
- 先把这条最小口径固定下来，后面再接 judge score、安全性或更复杂的多维评测，实验结果才不会因为口径漂移而失真。

**3. 这页的定位**
- 这里先固定偏好数据和评测的最小口径。
- 如果后面要接 DPO、GRPO 或更复杂的 judge 评测，再往上扩展即可。