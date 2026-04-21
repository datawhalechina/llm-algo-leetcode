# 13. DPO Loss Tutorial | 直接偏好优化 Loss 源码解析与实现 (DPO)

**难度：** Hard | **标签：** `微调`, `RLHF`, `Loss Function` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_DPO_Loss_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将实现目前大语言模型领域最炙手可热的对齐算法：**DPO (Direct Preference Optimization)**。相比于 PPO，DPO 将强化学习的 Reward Model 省略，直接利用交叉熵的变体去拟合人类偏好，非常优雅且高效。

### Step 1: 核心思想与痛点

> **RLHF 的问题：**
> 标准的 RLHF（如 PPO）需要训练 4 个模型：Actor（你要训练的模型）、Reference（参考模型，防止跑偏）、Reward Model（根据偏好训练的打分模型）和 Value Model（Critic）。训练非常不稳定，显存占用极大。
> **DPO 的本质：**
> 作者通过数学推导证明了：Reward Model 的奖励得分 $r(x,y)$ 可以隐式地由语言模型的概率 $P_	heta(y|x)$ 表达出来。因此，我们只需要**Actor 模型和 Reference 模型**，直接在一对一比较的样本上（Chosen vs Rejected）最大化对数似然差值。

### Step 2: DPO 损失代码框架
需要四组对齐的数据：选中的 Logprobs ($y_w$) 和拒绝的 Logprobs ($y_l$)，分别来自当前策略模型（Policy）和冻结的参考模型（Reference）。计算它们之间的隐式奖励差，并将其送入 `-F.logsigmoid()` 以获得最终梯度损失。

###  Step 3: 核心公式

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

   其中 $\beta$ 是控制偏离 Reference Model 程度的温度参数（如 `0.1`）。

###  Step 4: 动手实战

**要求**：请补全下方 `dpo_loss` 函数。
为了简化代码，我们假设你已经通过前向传播拿到了 Chosen 和 Rejected 样本的 `Log Probs`（对数概率和）。


```python
import torch
import torch.nn.functional as F
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
    pi_logratios_chosen = torch.zeros_like(policy_chosen_logps)  # 占位初始化
    
    # ==========================================
    # TODO 2: 计算 rejected 样本的隐式奖励分数
    # ==========================================
    # pi_logratios_rejected = ???
    pi_logratios_rejected = torch.zeros_like(policy_rejected_logps)  # 占位初始化
    
    # 乘以 beta (控制项)
    chosen_rewards = beta * pi_logratios_chosen
    rejected_rewards = beta * pi_logratios_rejected
    
    # ==========================================
    # TODO 3: 计算 logits 差值并放入 sigmoid 交叉熵中
    # ==========================================
    # logits = ???
    # losses = ???
    logits = torch.zeros_like(chosen_rewards)  # 占位初始化
    losses = torch.zeros_like(chosen_rewards)  # 占位初始化
    
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
    except TypeError as e:
        print("代码未完成，导致解包 None 错误")
        raise e
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise e

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
