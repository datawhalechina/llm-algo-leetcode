# 16. GRPO Loss Tutorial | 群体相对策略优化损失教程

**难度：** Medium-Hard | **环境：** CPU-first | **标签：** `对齐`, `RL`, `GRPO` | **目标人群：** 模型对齐与训练工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


GRPO (Group Relative Policy Optimization) 可以看作面向组内比较的策略优化方法。它通常不依赖显式 Critic，而是把同一组样本的奖励做相对归一化，再结合策略比率限制更新幅度。本页提供一个简化版的 GRPO Loss 实现，用来把 `RLHF -> DPO -> GRPO` 这条对齐链路补齐。

**关键词：** `GRPO`, `group relative`, `reward`
## 前置阅读

**导语：** 先补齐训练闭环和性能分析基础，再看 GRPO 的组内相对优化思想会更顺。

- [13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [20. Profiling and Memory Ledger | 性能分析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)


## 相关阅读

**导语：** 完成对齐链路后，可以继续看性能分析与通信基础。

- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)

### Step 1: 核心思想

> **为什么需要 GRPO？**
> GRPO 关注的是同一组样本内部的相对优劣，而不是把每个样本都单独拉到一个绝对奖励空间里。
> 这种做法的好处是：
> - 训练目标更稳，减少极端奖励对更新方向的冲击。
> - 可以和策略比率裁剪一起使用，限制一次更新的幅度。
> - 在某些场景下可以减少对显式 Critic 的依赖。

### Step 2: 数学形式

给定同一组中的奖励 $r_i$，先计算组内均值和标准差：

$$
\bar r = \frac{1}{N} \sum_{i=1}^{N} r_i, \quad\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (r_i - \bar r)^2 + \epsilon}
$$

然后把相对优势定义为：

$$
A_i = \frac{r_i - \bar r}{\sigma}
$$

最后再代入类似 PPO 的 clipped objective：

$$
L = -\mathbb{E}[\min(ratio \cdot A, clip(ratio) \cdot A)]
$$
这一节的实现链路就是先做组内归一化，再构造 clipped surrogate，最后汇总成 GRPO loss。


```python
import torch

```


```python
def compute_grpo_loss(log_probs_new, log_probs_old, rewards, group_ids, clip_range=0.2, eps=1e-6):
    """
    简化版 GRPO Loss。
    rewards/group_ids 允许把同一 prompt 下的多个候选答案分到一组。
    """
    # ==========================================
    # TODO 1: 计算组内相对优势
    # ==========================================
    # advantages = ???
    
    # ==========================================
    # TODO 2: 计算策略比率与两个 surrogate 目标
    # ==========================================
    # ratio = ???
    # surr1 = ???
    # surr2 = ???
    
    # ==========================================
    # TODO 3: 计算最终 loss 并返回
    # ==========================================
    # loss = ???
    return loss, advantages

```


```python
# 运行此单元格以测试你的实现
def test_grpo_loss():
    try:
        log_new = torch.tensor([-1.0, -0.5, -1.5, -0.2], requires_grad=True)
        log_old = torch.tensor([-1.1, -0.4, -1.6, -0.3])
        rewards = torch.tensor([1.0, 2.0, 0.5, 1.5])
        group_ids = torch.tensor([0, 0, 1, 1])
        loss, adv = compute_grpo_loss(log_new, log_old, rewards, group_ids)
        assert loss.ndim == 0, "Loss 应该是标量"
        assert torch.isfinite(loss), "Loss 不能是 NaN/Inf"
        assert torch.allclose(adv[group_ids == 0].mean(), torch.tensor(0.0), atol=1e-6), "组内优势均值应接近 0"
        loss.backward()
        assert log_new.grad is not None, "梯度没有回传到策略分数"
        print("✅ 测试通过！GRPO 简化版 Loss 可运行。")
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_grpo_loss()

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
import torch

def compute_grpo_loss(log_probs_new, log_probs_old, rewards, group_ids, clip_range=0.2, eps=1e-6):
    # ==========================================
    # TODO 1: 计算组内相对优势
    # ==========================================
    # advantages = ???
    advantages = torch.zeros_like(rewards)
    for gid in group_ids.unique(sorted=True):
        mask = group_ids == gid
        group_rewards = rewards[mask]
        centered = group_rewards - group_rewards.mean()
        denom = group_rewards.std(unbiased=False).clamp_min(eps)
        advantages[mask] = centered / denom

    # ==========================================
    # TODO 2: 计算策略比率与两个 surrogate 目标
    # ==========================================
    # ratio = ???
    ratio = torch.exp(log_probs_new - log_probs_old)
    # surr1 = ???
    surr1 = ratio * advantages
    # surr2 = ???
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages

    # ==========================================
    # TODO 3: 计算最终 loss 并返回
    # ==========================================
    # loss = ???
    loss = -torch.min(surr1, surr2).mean()
    return loss, advantages

```

### 解析

**1. TODO 1: 计算组内相对优势**

- **实现方式**：按 `group_ids` 把同组奖励聚合，再做去均值和标准差归一化。
- **代码核心**：`advantages[mask] = centered / denom`
- **数学含义**：这里的优势是“组内相对值”，不是单样本的绝对奖励。
- **工程意义**：把同一 prompt 下多个候选答案放在一起比较，可以减少不同样本尺度差异。

**2. TODO 2: 计算策略比率与两个 surrogate 目标**

- **实现方式**：先算 `ratio = exp(log_probs_new - log_probs_old)`，再构造 `surr1` 和 `surr2`。
- **代码核心**：`surr1 = ratio * advantages`，`surr2 = clamp(ratio) * advantages`
- **数学含义**：这一步沿用了 PPO 的 clipped objective 思路，用两个 surrogate 限制单步更新幅度。
- **工程意义**：既允许模型朝更好的组内排序移动，又避免策略比率变化过大。

**3. TODO 3: 计算最终 loss 并返回**

- **实现方式**：`loss = -torch.min(surr1, surr2).mean()`
- **代码核心**：`torch.min` 取更保守的 surrogate 估计，再对 batch 求均值。
- **数学含义**：这是一个偏悲观的优化目标，避免模型过度相信单次更新带来的收益。
- **工程意义**：把组内相对优势和策略裁剪结合起来，形成一个稳定的简化版 GRPO loss。

**进阶思考**

- 为什么 GRPO 通常不需要显式 Critic？
- 如果把组内归一化换成全局归一化，会发生什么？
- 这个实现和 PPO 的 clipped surrogate 有哪些本质相同与不同？
