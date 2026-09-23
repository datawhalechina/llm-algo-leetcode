# 15. DPO Loss Tutorial | 直接偏好优化损失教程

**难度：** Hard | **环境：** CPU-first | **标签：** `后训练对齐`, `DPO`, `偏好优化` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

RLHF 能把模型往人类偏好方向拉，但 PPO 训练链路很重：要维护策略模型、参考模型、奖励模型和价值模型，还要处理采样、打分和策略更新的稳定性问题。

DPO 的切入点更直接：既然偏好数据本身已经告诉我们“哪个回答更好”，就把 chosen / rejected 的对比关系写成一个可优化的损失函数，让策略模型相对参考模型更偏向 chosen。本节只实现 DPO Loss 的核心计算。完成后，你应该能看懂它如何把偏好比较转成 log probability 的差值，并理解为什么 DPO 是从复杂 RLHF 训练闭环到轻量偏好优化的重要过渡。

**关键词：** `DPO`, `preference`, `loss`

---
## 前置阅读

**导语：** 先补齐训练接口、损失函数和 RLHF/PPO 的系统开销，再看 DPO 为什么要把偏好优化收敛成一个更轻的 loss。

- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [14. RLHF PPO Memory | RLHF/PPO 显存拆解](../02_PyTorch_Algorithms/14_RLHF_PPO_Memory.md)


---
### Step 1：DPO 的动机与偏好数据

![DPO 从偏好数据到策略更新的总览](/02_PyTorch_Algorithms/15_dpo_overview_cn.svg)

> **RLHF 的问题：**
> 标准的 RLHF（如 PPO）需要训练 4 个模型：Actor（你要训练的模型）、Reference（参考模型，防止跑偏）、Reward Model（根据偏好训练的打分模型）和 Value Model（Critic）。训练非常不稳定，显存占用极大。
> **DPO 的本质：**
> 作者通过数学推导证明了：Reward Model 的奖励得分 $r(x,y)$ 可以隐式地由语言模型的概率 $P_	heta(y|x)$ 表达出来。因此，我们只需要**Actor 模型和 Reference 模型**，直接在一对一比较的样本上（Chosen vs Rejected）最大化对数似然差值。

#### DPO 和 PPO 的差异

DPO 不是把 PPO 换个名字，而是把偏好对齐的训练对象、训练信号和工程复杂度都压缩掉一层。

| 维度 | PPO / RLHF | DPO |
|:---|:---|:---|
| 需要的模型 | Actor + Reference + Reward Model + Critic | 主要是 Policy + Reference |
| 训练信号 | reward / advantage / rollout | chosen vs rejected 的偏好对比 |
| 工程复杂度 | 高，链路长，状态多 | 低，loss 直接、实现更轻 |
| 显存压力 | 大，多个模型和状态同时驻留 | 小很多，更适合工程落地 |
| 适合场景 | 需要完整 RLHF 语义时 | 已有偏好数据，希望快速做对齐时 |

#### `beta` 和数据构造

- `beta` 控制 policy 相对 reference 偏移的力度；太小会学得慢，太大容易偏离参考分布。
- chosen / rejected 不是任意两条回复，而是同一个 prompt 下的偏好对。
- 所以 DPO 的关键不是“算一个 loss”，而是先把偏好数据配成正确的 pair，再把 pair 变成稳定的 logprob 差值。
### Step 2：偏好对与隐式奖励
先确认同一个 prompt 下的 chosen / rejected 是成对样本，再分别取得 Policy 和 Reference 对两条回复的 log probability。两组 policy-reference 差值就是后续的隐式奖励输入；本 Step 只负责对齐数据来源和符号，不展开最终 loss。

一条偏好对至少要通过三项检查：chosen 与 rejected 回答的是同一个 prompt；差异确实来自质量、正确性或安全性，而不是长度、格式或是否泄漏答案；标注规则和来源可以追溯。若两条回复都正确、都错误，或只是长短不同，应标为不确定或移出训练集。

Reference Model 通常从已经完成 SFT 的 checkpoint 初始化：SFT 先提供可用的格式和任务能力，DPO 再在此基础上调整偏好。若 reference 仍是未经 SFT 的 base model，log-probability 差值会同时混入格式学习和偏好学习，难以解释。

Reference Model 通常从已经完成 SFT 的 checkpoint 初始化：SFT 先提供可用的格式和任务能力，DPO 再在此基础上调整偏好。若 reference 仍是未经 SFT 的 base model，log-probability 差值会同时混入格式学习和偏好学习，难以解释。
### Step 3：DPO logits、β 与损失

给定一段 Prompt $x$，模型生成了两个回复：好的回复 $y_w$ (Chosen/Win) 和差的回复 $y_l$ (Rejected/Lose)。

1. **计算策略比率的对数差 (Log Prob Ratios)：**
   $ \pi_{\theta}(y|x) $
   代表当前训练模型（Actor）对生成的 Token 的对数概率。
   $ \pi_{ref}(y|x) $ 
   代表冻结的参考模型（Reference）对生成的 Token 的对数概率。
   
   计算差距：
   $$ \hat{r}(x,y) = \beta \log \frac{\pi_	heta(y|x)}{\pi_{ref}(y|x)} = \beta (\log \pi_	heta(y|x) - \log \pi_{ref}(y|x)) $$
   *这实际上隐式地代表了该回复获得的 Reward 分数。*

2. **DPO Loss (二元交叉熵变体)：**
   我们要最大化 Chosen 和 Rejected 之间的 Reward 差，即最小化其负对数 Sigmoid：
   $$ L_{DPO} = -\log \sigma \left( \hat{r}(x, y_w) - \hat{r}(x, y_l) \right) $$

   其中 $\beta$ 是控制偏离 Reference Model 程度的温度参数（如 `0.1`）。$\beta$ 越大，偏离 reference 的代价越强，更新通常更保守；$\beta$ 越小，偏好差异的推动更激进，但更容易放大噪声。

DPO 依赖已经收集好的离线偏好对，不会像 PPO/GRPO 那样通过 rollout 主动探索新回答。因此它适合先利用稳定的 chosen/rejected 数据做偏好迁移；如果任务需要环境反馈、可验证奖励或持续发现新策略，就需要进入 GRPO/PPO 或在线对齐路线。

![DPO 损失计算路径](/02_PyTorch_Algorithms/15_dpo_loss_flow_cn.svg)

### Step 4：实现并验证 DPO Loss

**要求**：请补全下方 `dpo_loss` 函数。
为了简化代码，我们假设你已经通过前向传播拿到了 Chosen 和 Rejected 样本的 `Log Probs`（对数概率和）。


```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    直接偏好优化 (DPO) 的损失函数实现

    Args:
        policy_chosen_logps: 当前模型在 chosen 样本上的对数概率和，形状 [batch_size]
        policy_rejected_logps: 当前模型在 rejected 样本上的对数概率和，形状 [batch_size]
        reference_chosen_logps: 参考模型在 chosen 样本上的对数概率和，形状 [batch_size]
        reference_rejected_logps: 参考模型在 rejected 样本上的对数概率和，形状 [batch_size]
        beta: 偏好系数 (温度超参)，控制偏离参考模型的程度

    Returns:
        losses: DPO 损失，形状 [batch_size]
        chosen_rewards: 隐式的 chosen 奖励，形状 [batch_size]
        rejected_rewards: 隐式的 rejected 奖励，形状 [batch_size]
    """
    if not (
        policy_chosen_logps.shape == policy_rejected_logps.shape
        == reference_chosen_logps.shape == reference_rejected_logps.shape
    ):
        raise ValueError("policy / reference 的 chosen-rejected logps 形状必须一致")
    if beta <= 0:
        raise ValueError("beta 必须是正数")

    # chosen / rejected 各自相对 reference 的隐式奖励
    pi_logratios_chosen = policy_chosen_logps - reference_chosen_logps
    pi_logratios_rejected = policy_rejected_logps - reference_rejected_logps

    chosen_rewards = beta * pi_logratios_chosen
    rejected_rewards = beta * pi_logratios_rejected

    # chosen 比 rejected 更好时，logits 应该更大，loss 才会更小
    logits = chosen_rewards - rejected_rewards
    losses = -F.logsigmoid(logits)

    return losses, chosen_rewards, rejected_rewards
```


```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    直接偏好优化 (DPO) 的损失函数实现
    
    Args:
        policy_chosen_logps: 当前模型在 chosen 样本上的对数概率和，形状 [batch_size]
        policy_rejected_logps: 当前模型在 rejected 样本上的对数概率和，形状 [batch_size]
        reference_chosen_logps: 参考模型在 chosen 样本上的对数概率和，形状 [batch_size]
        reference_rejected_logps: 参考模型在 rejected 样本上的对数概率和，形状 [batch_size]
        beta: 偏好系数 (温度超参)，控制偏离参考模型的程度
    
    Returns:
        losses: DPO 损失，形状 [batch_size]
        chosen_rewards: 隐式的 chosen 奖励，形状 [batch_size]
        rejected_rewards: 隐式的 rejected 奖励，形状 [batch_size]
    """
    
    # ==========================================
    # TODO 1: 计算 chosen 样本的隐式奖励分数 (Logits 差值)
    # ==========================================
    # pi_logratios_chosen = ???
    
    # ==========================================
    # TODO 2: 计算 rejected 样本的隐式奖励分数
    # ==========================================
    # pi_logratios_rejected = ???
    
    # 乘以 beta (控制项)
    chosen_rewards = beta * pi_logratios_chosen
    rejected_rewards = beta * pi_logratios_rejected
    
    # ==========================================
    # TODO 3: 计算 logits 差值并放入 sigmoid 交叉熵中
    # ==========================================
    # logits = ???
    # losses = ???
    
    return losses, chosen_rewards, rejected_rewards

```


```python
# 运行此单元格以测试你的实现
def test_dpo_loss():
    try:
        batch_size = 4
        
        # 模拟模型输出：假设我们想要 policy 表现更好，
        # 所以它对 chosen 的概率更高，对 rejected 的概率更低
        policy_chosen = torch.tensor([-1.0, -1.2, -0.8, -1.5]) 
        policy_rejected = torch.tensor([-3.0, -2.5, -4.0, -2.8])
        
        ref_chosen = torch.tensor([-2.0, -2.0, -2.0, -2.0])
        ref_rejected = torch.tensor([-2.0, -2.0, -2.0, -2.0])
        
        losses, c_rewards, r_rewards = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )
        
        # 测试返回的形状
        assert losses.shape == (batch_size,)
        assert c_rewards.shape == (batch_size,)
        assert r_rewards.shape == (batch_size,)
        
        # 测试隐式奖励的计算 (policy - ref) * beta
        assert torch.allclose(c_rewards[0], torch.tensor(0.1)), f"Chosen Reward 错误: {c_rewards[0]}"
        assert torch.allclose(r_rewards[0], torch.tensor(-0.1)), f"Rejected Reward 错误: {r_rewards[0]}"
        
        # 测试 DPO 损失数值
        # log_sigmoid(0.1 - (-0.1)) = log_sigmoid(0.2)
        expected_loss_0 = -F.logsigmoid(torch.tensor(0.2))
        assert torch.allclose(losses[0], expected_loss_0), f"Loss 计算错误: 期望 {expected_loss_0}, 实际 {losses[0]}"
        
        print("\n✅ All Tests Passed! DPO 损失函数实现通过测试。")
        
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
        print(f"\n❌ 测试失败: {e}")
        raise

test_dpo_loss()
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
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    直接偏好优化 (DPO) 的损失函数实现
    
    Args:
        policy_chosen_logps: 当前模型在 chosen 样本上的对数概率和，形状 [batch_size]
        policy_rejected_logps: 当前模型在 rejected 样本上的对数概率和，形状 [batch_size]
        reference_chosen_logps: 参考模型在 chosen 样本上的对数概率和，形状 [batch_size]
        reference_rejected_logps: 参考模型在 rejected 样本上的对数概率和，形状 [batch_size]
        beta: 偏好系数 (温度超参)，控制偏离参考模型的程度
    
    Returns:
        losses: DPO 损失，形状 [batch_size]
        chosen_rewards: 隐式的 chosen 奖励，形状 [batch_size]
        rejected_rewards: 隐式的 rejected 奖励，形状 [batch_size]
    """
    
    # TODO 1: 计算 chosen 样本的隐式奖励分数
    pi_logratios_chosen = policy_chosen_logps - reference_chosen_logps
    
    # TODO 2: 计算 rejected 样本的隐式奖励分数
    pi_logratios_rejected = policy_rejected_logps - reference_rejected_logps
    
    # 乘以 beta 得到隐式奖励
    chosen_rewards = beta * pi_logratios_chosen
    rejected_rewards = beta * pi_logratios_rejected
    
    # TODO 3: 计算 DPO 损失
    logits = chosen_rewards - rejected_rewards
    losses = -F.logsigmoid(logits)
    
    return losses, chosen_rewards, rejected_rewards

```

### 解析

**1. TODO 1 & 2: 计算隐式奖励分数 (Implicit Reward)**

- **实现方式**：
  ```python
  pi_logratios_chosen = policy_chosen_logps - reference_chosen_logps
  pi_logratios_rejected = policy_rejected_logps - reference_rejected_logps
  ```
- **数学原理**：DPO 的核心洞察是将 Reward Model 的奖励函数隐式地表达为策略模型与参考模型的对数概率比：
  $$\hat{r}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} = \beta (\log \pi_\theta(y|x) - \log \pi_{ref}(y|x))$$
- **物理含义**：
  - 如果策略模型对某个回复的概率高于参考模型（`policy_logps > reference_logps`），则隐式奖励为正，表示该回复被认为是好的。
  - 如果策略模型对某个回复的概率低于参考模型（`policy_logps < reference_logps`），则隐式奖励为负，表示该回复被认为是差的。
- **beta 的作用**：控制策略模型偏离参考模型的程度。较大的 beta 会更严格地约束策略模型不要偏离参考模型太远。

**2. TODO 3: 计算 DPO 损失**

- **实现方式**：
  ```python
  logits = chosen_rewards - rejected_rewards
  losses = -F.logsigmoid(logits)
  ```
- **数学公式**：
  $$L_{DPO} = -\log \sigma(\hat{r}(x, y_w) - \hat{r}(x, y_l))$$
  其中 $\sigma$ 是 Sigmoid 函数，$y_w$ 是 chosen 样本，$y_l$ 是 rejected 样本。
- **优化目标**：最大化 chosen 样本的隐式奖励与 rejected 样本的隐式奖励之间的差距。
  - 当 `chosen_rewards > rejected_rewards` 时，`logits > 0`，`sigmoid(logits) > 0.5`，损失较小。
  - 当 `chosen_rewards < rejected_rewards` 时，`logits < 0`，`sigmoid(logits) < 0.5`，损失较大，梯度会推动模型增大 chosen 的概率，减小 rejected 的概率。
- **数值稳定性**：使用 `F.logsigmoid` 而非 `torch.log(torch.sigmoid())`，避免数值下溢问题。

**工程要点**

- **与 RLHF 对比**：
  - RLHF (PPO) 需要 4 个模型：Actor、Critic、Reward Model、Reference Model，显存开销巨大。
  - DPO 只需要 2 个模型：Policy Model（训练中）、Reference Model（冻结），显存效率提升 50%。
- **数据格式**：DPO 需要成对的偏好数据 `(prompt, chosen_response, rejected_response)`，通常来自人类标注或 AI 反馈。
- **训练稳定性**：DPO 避免了强化学习的不稳定性（如 PPO 的 Clip 机制、Value 函数估计误差），训练过程更加稳定。
- **超参数调优**：
  - `beta`：通常设为 0.1-0.5，控制偏离参考模型的程度。
  - Reference Model：通常使用 SFT 后的模型作为参考模型，确保策略模型不会偏离有监督微调的分布太远。
- **实际应用**：DPO 已被广泛应用于开源模型的对齐训练，如 Zephyr、Mistral-Instruct 等，是目前最流行的 RLHF 替代方案。

## 相关阅读

完成 DPO 后，可以继续看 GRPO 以及偏好优化在真实框架中的实现。

- [16. GRPO Loss Tutorial | 群体相对策略优化损失教程](../02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.md)
- [DPO 原论文](https://arxiv.org/abs/2305.18290)
- [TRL DPO Trainer 官方文档](https://huggingface.co/docs/trl/dpo_trainer)