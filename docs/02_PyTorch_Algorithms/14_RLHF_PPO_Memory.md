# 14. RLHF PPO Memory | RLHF / PPO 资源桥接与显存流转

**难度：** Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `RLHF`, `PPO` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/14_RLHF_PPO_Memory.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

完成 SFT 以后，模型已经能模仿数据里的回答，但还没有真正学会“哪个回答更符合人类偏好”。RLHF/PPO 要解决的就是这个问题：让模型根据奖励信号继续调整生成策略；代价是训练链路一下变重，Actor、Reference、Reward Model 和 Critic 会同时进入显存账本。

本节承担 2.4 与 2.5 之间的资源桥接：先看 PPO Actor Loss 如何把“提高高奖励行为、限制策略偏移”落实为 ratio、advantage 和 clip，再看 Actor、Reference、Reward Model 和 Critic 的流转为什么会带来明显显存压力。也就是说，本节把一个对齐目标映射到训练对象、前向/反向路径和驻留状态的资源代价。完成后，你应该能把“对齐目标更复杂”与“训练资源代价更高”联系起来，也能理解后面 DPO / GRPO 为什么会尝试把这条链路做轻。

**关键词：** `PPO`, `ratio`, `advantage`

---
## 前置阅读

**导语：** 先补齐训练闭环和优化器基础，再看 RLHF / PPO 如何在 SFT 之后继续做偏好对齐会更顺。
- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)

---
### Step 1：PPO 目标与 Actor Loss

PPO 的核心思想是：**让 Actor 往 Reward 高的方向更新，但步子不能迈得太大。**
为此，PPO 引入了优势函数 (Advantage, $A_t$) 和重要性采样比率 (Ratio, $r_t$)。

1. **重要性采样比率 r_t**：
   r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
   也就是当前模型生成某个 Token 的概率，除以旧模型生成该 Token 的概率。
   
2. **Clip 截断目标函数 (Actor Loss)**：
   L^CLIP(theta) = - min(r_t * A_t, clip(r_t, 1-epsilon, 1+epsilon) * A_t)
   如果优势 A_t > 0（这个 Token 很好），我们要提高它的概率，但如果 r_t 已经超过 1+epsilon（涨得太多了），就停止给予梯度奖励，防止模型“发疯”。

PPO 的一次更新可以先按下面的数据流理解：

```text
prompt → Actor rollout → response
                    ├→ Reward Model / 规则奖励 → reward_t
                    ├→ Critic value_t
                    └→ old_log_prob_t

reward_t + value_t → advantage_t → ratio_t + clip → Actor update
reference log_prob → KL 约束，限制新策略偏离 SFT / reference 基线
```

在简化说明中，可以把优势理解为“当前奖励相对价值基线的改进幅度”：`A_t ≈ reward_t - V(s_t)`；完整 PPO 还会结合折扣回报和 GAE。优势决定更新方向，ratio/clip 决定更新幅度，二者不是同一个量。

![PPO 训练与资源路径总览](/02_PyTorch_Algorithms/14_rlhf_ppo_overview_cn.svg)

### Step 2：RLHF 资源流转与显存账本

RLHF / PPO 的难点不只在公式，而在训练链路里同时存在多个状态对象：

```text
prompt -> rollout / sampling -> reward model -> advantage -> PPO update
         \-> reference model (frozen)
         \-> critic / value baseline
```

这意味着一次更新里，除了要算 Actor 的新旧策略比率，还要保存参考模型、奖励模型和价值模型相关的中间结果。相比 SFT，PPO 的显存压力不是来自“单个 loss 更复杂”，而是来自“训练对象更多、流转步骤更长、每一步都要保留更多状态”。
其中 `reward` 是对 response 的评价信号，`value` 是 Critic 对当前状态未来回报的估计；`advantage` 不是另一个模型输出，而是由二者组合出来的更新信号。若 reward 与 value 的时间步、mask 或 response 范围没有对齐，Actor Loss 即使数值正常，也可能沿错误方向更新。

#### 四模型显存账本

| 模型 / 状态 | 主要作用 | 资源含义 |
|:---|:---|:---|
| Actor | 生成当前策略并接受梯度更新 | 需要反向传播，保存激活与梯度 |
| Reference | 提供稳定参照 | 冻结，但仍会占 forward 显存 |
| Reward Model | 给生成结果打分 | 额外前向开销，常与 rollout 配合 |
| Critic / Value Model | 估计 advantage 的基线 | 额外 forward / backward 路径 |

#### 为什么 PPO 比 SFT 重

- **对象更多**：SFT 主要是一个模型 + 一个 loss；PPO 往往是策略、参考、奖励和价值模型一起出现。
- **状态更多**：PPO 需要保存 rollout、log prob、reward 和 advantage 的中间量。
- **更新更谨慎**：clip、ratio 和 advantage 让每一步都要额外检查更新幅度。

#### 和 DPO / GRPO 的关系

- `DPO` 试图把偏好对齐收敛成更轻的 logprob 差值，不再维护完整 PPO 流程。
- `GRPO` 继续保留组内比较，但进一步弱化对显式 Critic 的依赖。
- 所以这节的角色不是“再造一个 RLHF 系统”，而是先把 PPO 为什么重讲清楚，后面的轻量方法才有对照意义。

![PPO 单步显存账本](/02_PyTorch_Algorithms/14_ppo_memory_ledger_cn.svg)

### Step 3：对齐目标与资源代价

Step 1 的 ratio、advantage 和 clip 说明单次策略更新如何受约束；Step 2 的四类模型和 rollout 状态说明一次 PPO 更新为什么需要更多资源。把两者放在一起，可以得到一条从目标到代价的判断链。

| 对齐目标 | 额外状态或计算 | 资源代价 | 后续轻量化方向 |
| --- | --- | --- | --- |
| 提高高优势行为的概率 | 新旧策略 log probability、advantage | rollout 与策略状态需要保留 | DPO 用离线偏好对替代在线 PPO 更新 |
| 限制策略单步偏移 | ratio、clip 区间、参考策略 | 需要比较当前策略与旧/参考策略 | 通过目标函数减少显式模型或更新路径 |
| 让奖励信号可用于更新 | Reward Model、Critic / value、reward 与 advantage | 额外模型前向或反向、更多激活状态 | GRPO 弱化显式 Critic，减少资源驻留 |

这张表用于解释 PPO 为什么重，不是完整 RLHF 训练器的实现清单。代码区只验证 Actor Loss 的数学机制；四模型驻留、rollout 和真实吞吐需要在项目级实验中单独测量。
因此 PPO 的关键不是“把 reward 直接乘进 loss”，而是先完成 `response → reward/value → advantage → clipped policy update` 的时间和 token 对齐。

#### SFT update 与 PPO update 的区别

| 维度 | SFT | PPO / RLHF |
|:---|:---|:---|
| 训练信号 | 固定数据中的 target token | rollout 后得到的 reward 和 advantage |
| 更新对象 | 直接最小化监督交叉熵 | 最大化 clipped surrogate，并约束策略偏移 |
| 数据来源 | 训练集样本 | 当前 policy 生成的新 response |
| 额外状态 | labels、mask | old/reference log-prob、reward、value、advantage 和 KL |

如果已有高质量离线偏好对，DPO 通常比 PPO 更容易控制资源和训练稳定性；只有当任务需要在线探索、环境反馈或可验证奖励时，才值得承担 PPO 的额外模型和 rollout 代价。


### Step 4：实现并验证 Actor Loss

完成下方的 `compute_actor_loss`。你将接收到新旧模型的 log probability，以及用于控制更新方向的优势 `advantage`。实现主线分三步：计算新旧策略概率比 `ratio`，构造 clipped surrogate，再汇总得到 PPO actor loss。

本题只验证 Actor Loss 的张量形状、截断边界和梯度路径，不实现 rollout、Reward Model 或 Critic。


```python
import torch
import torch.nn.functional as F

```


```python
def compute_actor_loss(log_probs_new, log_probs_old, advantages, clip_range=0.2):
    """
    计算 PPO 的 Actor Clip Loss。
    
    Args:
        log_probs_new: 当前 Actor 模型在采样的 Token 上的对数概率, shape [batch_size, seq_len]
        log_probs_old: 采样时(旧) Actor 模型在对应 Token 上的对数概率, shape [batch_size, seq_len]
        advantages: 优势函数 (Reward - Critic Value), shape [batch_size, seq_len]
        clip_range: 截断范围 epsilon，默认 0.2
        
    Returns:
        actor_loss: 标量
    """
    if log_probs_new.shape != log_probs_old.shape or log_probs_new.shape != advantages.shape:
        raise ValueError("log_probs_new / log_probs_old / advantages 的形状必须一致")

    
    # ==========================================
    # TODO 1: 计算概率比率 ratio (r_t)
    # ==========================================
    # ratio = ???
    
    # ==========================================
    # TODO 2: 计算无截断的 surrogate 目标
    # ==========================================
    # surr1 = ???
    
    # ==========================================
    # TODO 3: 计算截断后的 surrogate 目标
    # ==========================================
    # surr2 = ???
    
    # ==========================================
    # TODO 4: 计算最终的 Loss 
    # ==========================================
    # loss = ???
    
    return loss

```


```python
# 测试
def _legacy_test_ppo_actor_loss():
    try:
        torch.manual_seed(42)
        log_p_new = torch.tensor([[-2.0, -1.5], [-1.0, -0.5]], requires_grad=True)
        log_p_old = torch.tensor([[-2.1, -1.4], [-1.0, -0.6]])
        # 假设第一句生成很好(优势全正), 第二句生成很差(优势全负)
        adv = torch.tensor([[1.0, 0.5], [-1.0, -0.5]])
        
        loss = compute_actor_loss(log_p_new, log_p_old, adv, clip_range=0.2)
        
        # 解析解计算
        ratio = torch.exp(log_p_new - log_p_old)
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 0.8, 1.2) * adv
        expected_loss = -torch.min(s1, s2).mean()
        
        assert torch.allclose(loss, expected_loss), "Loss 计算不正确！"

        # 验证 ratio 超出区间时，正优势和负优势都按 PPO 规则被截断。
        high_ratio = torch.tensor([[2.0]])
        low_ratio = torch.tensor([[0.5]])
        positive_adv = torch.tensor([[1.0]])
        negative_adv = torch.tensor([[-1.0]])
        high_loss = compute_actor_loss(torch.log(high_ratio), torch.zeros_like(high_ratio), positive_adv)
        low_loss = compute_actor_loss(torch.log(low_ratio), torch.zeros_like(low_ratio), negative_adv)
        assert torch.allclose(high_loss, torch.tensor(-1.2)), "正优势的上界截断错误"
        assert torch.allclose(low_loss, torch.tensor(0.8)), "负优势的下界截断错误"

        try:
            compute_actor_loss(torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(1, 2))
        except ValueError:
            pass
        else:
            raise AssertionError("形状不一致时应主动报错")

        print("✅ 测试通过！PPO Actor Loss 实现、截断边界和输入契约均通过测试。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        else:
            print("代码可能未完成，导致了张量维度错误")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise

# 机制测试入口：ratio、clipping、梯度路径和输入契约分别验证。
def _build_ppo_loss_case():
    new = torch.tensor([[-2.0, -1.5], [-1.0, -0.5]], requires_grad=True)
    old = torch.tensor([[-2.1, -1.4], [-1.0, -0.6]])
    advantage = torch.tensor([[1.0, 0.5], [-1.0, -0.5]])
    return new, old, advantage

def _assert_ppo_ratio_and_loss(new, old, advantage):
    loss = compute_actor_loss(new, old, advantage, clip_range=0.2)
    ratio = torch.exp(new - old)
    expected = -torch.minimum(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage).mean()
    assert torch.allclose(loss, expected), "PPO Actor Loss 计算错误"

def _assert_ppo_clipping():
    high = compute_actor_loss(torch.tensor([[2.0]]).log(), torch.zeros(1, 1), torch.ones(1, 1))
    low = compute_actor_loss(torch.tensor([[0.5]]).log(), torch.zeros(1, 1), -torch.ones(1, 1))
    assert torch.allclose(high, torch.tensor(-1.2)), "正优势上界截断错误"
    assert torch.allclose(low, torch.tensor(0.8)), "负优势下界截断错误"

def _assert_ppo_gradient(new, old, advantage):
    compute_actor_loss(new, old, advantage).backward()
    assert new.grad is not None and torch.isfinite(new.grad).all(), "ratio 没有形成有效梯度"

def _assert_ppo_input_contract():
    try:
        compute_actor_loss(torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(1, 2))
    except ValueError:
        return
    raise AssertionError("形状不一致时应主动报错")

def test_ppo_actor_loss():
    try:
        new, old, advantage = _build_ppo_loss_case()
        _assert_ppo_ratio_and_loss(new, old, advantage)
        _assert_ppo_clipping()
        _assert_ppo_gradient(new, old, advantage)
        _assert_ppo_input_contract()
        print("✅ PPO ratio、clipping、梯度和输入契约测试通过。")
    except (AttributeError, NameError, TypeError, ValueError) as e:
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e

test_ppo_actor_loss()

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
def compute_actor_loss(log_probs_new, log_probs_old, advantages, clip_range=0.2):
    """
    计算 PPO 的 Actor Clip Loss。
    
    Args:
        log_probs_new: 当前 Actor 模型在采样的 Token 上的对数概率, shape [batch_size, seq_len]
        log_probs_old: 采样时(旧) Actor 模型在对应 Token 上的对数概率, shape [batch_size, seq_len]
        advantages: 优势函数 (Reward - Critic Value), shape [batch_size, seq_len]
        clip_range: 截断范围 epsilon，默认 0.2
        
    Returns:
        actor_loss: 标量
    """
    if log_probs_new.shape != log_probs_old.shape or log_probs_new.shape != advantages.shape:
        raise ValueError("log_probs_new / log_probs_old / advantages 的形状必须一致")

    
    # TODO 1: 计算概率比率 ratio
    ratio = torch.exp(log_probs_new - log_probs_old)
    
    # TODO 2: 计算无截断的 surrogate 目标
    surr1 = ratio * advantages
    
    # TODO 3: 计算截断后的 surrogate 目标
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    
    # TODO 4: 计算最终的 Loss
    loss = -torch.min(surr1, surr2).mean()
    
    return loss

```

### 解析

**1. TODO 1: 计算概率比率 (Importance Sampling Ratio)**

- **实现方式**：`ratio = torch.exp(log_probs_new - log_probs_old)`
- **数学原理**：$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)} = \exp(\log \pi_\theta - \log \pi_{old})$
- **物理含义**：衡量新策略相对于旧策略的变化幅度。如果 `ratio > 1`，说明新策略更倾向于选择该 Token；如果 `ratio < 1`，说明新策略不太喜欢该 Token。
- **数值稳定性**：使用对数概率相减再取指数，避免直接计算概率比值时的数值下溢问题。

**2. TODO 2: 计算无截断的 surrogate 目标**

- **实现方式**：`surr1 = ratio * advantages`
- **数学含义**：$L^{CPI}(\theta) = r_t \cdot A_t$（Conservative Policy Iteration 的目标函数）
- **优化方向**：
  - 当 `advantage > 0`（该动作好于平均水平）且 `ratio > 1`（新策略更倾向选择该动作）时，`surr1` 为正，梯度会进一步增大该动作的概率。
  - 当 `advantage < 0`（该动作差于平均水平）且 `ratio < 1`（新策略不太选择该动作）时，`surr1` 为正，梯度会进一步减小该动作的概率。

**3. TODO 3: 计算截断后的 surrogate 目标**

- **实现方式**：`surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages`
- **截断机制**：将 `ratio` 限制在 `[1-ε, 1+ε]` 区间内（通常 ε=0.2）
- **防止过度更新**：
  - 当 `advantage > 0` 且 `ratio > 1+ε` 时，截断为 `1+ε`，防止新策略过度偏向该动作。
  - 当 `advantage < 0` 且 `ratio < 1-ε` 时，截断为 `1-ε`，防止新策略过度远离该动作。
- **核心思想**：允许策略改进，但限制单步更新幅度，确保训练稳定性。

**4. TODO 4: 计算最终的 Loss**

- **实现方式**：`loss = -torch.min(surr1, surr2).mean()`
- **取最小值的原因**：悲观估计（Pessimistic Bound）
  - 当 `advantage > 0` 时，取 `min(surr1, surr2)` 限制了过度乐观的更新。
  - 当 `advantage < 0` 时，取 `min(surr1, surr2)` 同样限制了过度悲观的更新。
- **负号的作用**：PPO 的目标是最大化期望回报，而 PyTorch 的优化器是最小化 Loss，因此需要加负号。
- **均值聚合**：对所有 Token 的 Loss 取平均，得到 batch 级别的标量 Loss。

**工程要点**

- **四模型架构**：RLHF 训练需要同时维护 Actor（策略模型）、Critic（价值模型）、Reward Model（奖励模型）、Reference Model（参考模型），显存开销巨大。
- **KL 散度惩罚**：实际训练中通常会加入 KL 散度项 `KL(π_new || π_ref)`，防止新策略过度偏离参考策略，避免输出不可控的内容。
- **优势函数计算**：`advantage = reward + γ * V(s_{t+1}) - V(s_t)`，其中 `V` 由 Critic 模型估计。
- **与 DPO 对比**：DPO 通过隐式奖励建模绕过了显式的 Reward Model 和 Critic Model，显存效率更高，但 PPO 在复杂任务上通常表现更好。

## 相关阅读

完成 PPO Actor Loss 后，可以继续比较 DPO/GRPO，并回到资源与性能证据。

- [15. DPO Loss Tutorial | 直接偏好优化损失教程](../02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.md)
- [16. GRPO Loss Tutorial | 群体相对策略优化损失教程](../02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.md)
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [TRL PPOTrainer 文档](https://huggingface.co/docs/trl/ppo_trainer)