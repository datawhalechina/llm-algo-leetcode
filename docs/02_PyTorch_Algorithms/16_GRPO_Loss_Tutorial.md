# 16. GRPO Loss Tutorial | 群体相对策略优化损失教程

**难度：** Medium-Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `GRPO`, `奖励优化` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在偏好优化里，模型常常会针对同一个 prompt 生成多个候选答案。此时我们不只关心某个答案的绝对奖励，更关心它在同一组候选里相对好还是相对差。

GRPO 的重点就是把这种“组内比较”变成训练信号：先把同组奖励归一化成相对优势，再用类似 PPO 的裁剪目标限制策略更新幅度，从而减少对显式 Critic 的依赖。本节实现一个简化版 GRPO Loss。完成后，你应该能看懂 `group_ids`、相对优势和策略比率裁剪各自解决什么问题，并把 `RLHF -> DPO -> GRPO` 这条对齐链路串起来。

**关键词：** `GRPO`, `group relative`, `reward`

---
## 前置阅读

**导语：** 先理解训练闭环、偏好优化和显存账本，再看 GRPO 如何用组内相对优势稳定策略更新。

- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [P0: 20. Profiling and Memory Ledger | 性能分析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)
- [15. DPO Loss Tutorial | 直接偏好优化损失教程](../02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.md)


---
### Step 1：GRPO 的组内比较

> **为什么需要 GRPO？**
> GRPO 关注的是同一组样本内部的相对优劣，而不是把每个样本都单独拉到一个绝对奖励空间里。
> 这种做法的好处是：
> - 训练目标更稳，减少极端奖励对更新方向的冲击。
> - 可以和策略比率裁剪一起使用，限制一次更新的幅度。
> - 在某些场景下可以减少对显式 Critic 的依赖。

如果任务有可验证的结果，例如数学答案、代码测试或格式约束，可以把规则验证器产生的结果作为 RLVR（Reinforcement Learning from Verifiable Rewards）信号。此时仍然要区分“验证器分数”和“真实任务质量”：验证器只覆盖它写出的规则。

![GRPO 从候选组到策略更新的总览](/02_PyTorch_Algorithms/16_grpo_overview_cn.svg)

#### 组内比较为什么更稳

GRPO 的关键不是“再造一个 PPO 变体”，而是把一个 prompt 下的多个候选答案放在同一个组里比较，从而降低奖励尺度波动带来的训练不稳定。

| 维度 | 单样本 reward | 组内相对 reward |
|:---|:---|:---|
| 评价方式 | 直接看绝对奖励 | 看同组里谁更好 |
| 稳定性 | 容易受奖励尺度影响 | 先中心化再标准化，方差更小 |
| 数据组织 | 一条样本就能训练 | 需要同一 prompt 下的多个候选 |
| 训练直觉 | 绝对分数高就更新 | 组内相对更优的样本获得更强信号 |

#### 一个最小例子

假设同一个 prompt 生成了 4 个候选：

- `group 0`: reward = `[1.0, 2.0]`
- `group 1`: reward = `[0.5, 1.5]`

GRPO 不直接拿这 4 个分数去做全局比较，而是分别在各自组内做均值和标准差归一化。这样做的结果是：

- 组内平均值被拉回 0，避免组与组之间的绝对奖励尺度干扰更新。
- 高于组均值的候选得到正优势，低于组均值的候选得到负优势。
- 策略更新更关注“同一个 prompt 下谁更好”，而不是“不同 prompt 的 reward 绝对值有多大”。

这也是 GRPO 适合做组内排序、候选比较和生成优化的原因。
### Step 2：组内优势与裁剪目标

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
这条机制链先把奖励变成组内相对优势，再用策略比率和 clipped objective 限制更新幅度。代码实现留到 Step 4。

![GRPO 组内优势机制](/02_PyTorch_Algorithms/16_grpo_groupwise_mechanism_cn.svg)

### Step 3：从奖励分组到策略更新

把同一组候选的奖励做归一化后，正负优势会决定更新方向；策略比率和 clipped objective 再限制一次更新的幅度。这里先沿着机制链理解每个量的作用，代码实现留到 Step 4。

#### 机制顺序

1. 同一组候选共享一个比较基准：高于组均值的样本得到正优势，低于组均值的样本得到负优势。
2. 策略比率表示新旧策略对同一响应的偏移程度，clipped objective 用于限制过大的偏移。
3. 最终目标同时保留组内排序信号和更新幅度约束。

#### 观察重点

- 如果组内中心化错了，奖励尺度会重新干扰更新方向。
- 如果策略比率偏移过大，单次更新可能破坏已有策略。
- 如果缺少保守的 clipped objective，组内排序信号就可能被过激更新放大。

#### RLVR 与 reward hacking

| 情况 | 表面现象 | 应检查什么 |
|:---|:---|:---|
| 验证器过窄 | reward 上升，但答案只是在迎合格式 | 独立题集、人工抽检和反例 |
| 奖励项失衡 | 长答案或固定模板获得高分 | 各奖励项分布、长度相关性和消融结果 |
| 组内投机 | 同组候选分数拉开，但绝对质量没有提高 | baseline 质量、组外评测和失败案例 |
| 验证器泄漏 | 训练集上 reward 很高，换规则后崩溃 | 独立 verifier、隐藏测试和数据隔离 |

因此 GRPO 的 `advantage` 只说明候选在当前组内相对更好，不等于模型已经获得真实能力。RLVR 训练至少需要保留 verifier 版本、奖励组成、独立评测和失败样本，避免把 reward 上升直接当作能力提升。

### Step 4：实现并验证 GRPO Loss

请补全下方 `compute_grpo_loss` 函数，并运行测试，确认组内优势、策略比率、裁剪目标和梯度回传均符合前面定义的机制。


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
def _legacy_test_grpo_loss():
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

# 机制测试入口：组内归一化、策略比率、clipping 和梯度分别验证。
def _build_grpo_loss_case():
    log_new = torch.tensor([-1.0, -0.5, -1.5, -0.2], requires_grad=True)
    log_old = torch.tensor([-1.1, -0.4, -1.6, -0.3])
    rewards = torch.tensor([1.0, 2.0, 0.5, 1.5])
    group_ids = torch.tensor([0, 0, 1, 1])
    return log_new, log_old, rewards, group_ids

def _assert_group_normalization(advantages, group_ids):
    for group_id in group_ids.unique(sorted=True):
        group_advantage = advantages[group_ids == group_id]
        assert torch.allclose(group_advantage.mean(), torch.tensor(0.0), atol=1e-6), "组内优势均值不为 0"

def _assert_grpo_loss_and_ratio(log_new, log_old, rewards, group_ids):
    loss, advantages = compute_grpo_loss(log_new, log_old, rewards, group_ids)
    assert loss.ndim == 0 and torch.isfinite(loss), "Loss 必须是有限标量"
    ratio = torch.exp(log_new.detach() - log_old)
    assert torch.all(ratio > 0), "策略比率必须为正"
    return loss, advantages

def _assert_grpo_gradient(loss, log_new):
    loss.backward()
    assert log_new.grad is not None and torch.isfinite(log_new.grad).all(), "梯度没有回传到新策略"

def test_grpo_loss():
    try:
        log_new, log_old, rewards, group_ids = _build_grpo_loss_case()
        loss, advantages = _assert_grpo_loss_and_ratio(log_new, log_old, rewards, group_ids)
        _assert_group_normalization(advantages, group_ids)
        _assert_grpo_gradient(loss, log_new)
        print("✅ GRPO 组内归一化、策略比率、clipping 和梯度测试通过。")
    except (AttributeError, NameError, TypeError, ValueError) as e:
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e

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

## 相关阅读

完成 GRPO Loss 后，可以继续阅读 GRPO/RLVR 的论文与训练框架实现。

- [GRPO 原论文](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/grpo_trainer)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)