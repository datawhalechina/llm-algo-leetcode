# 12. RLHF PPO Memory | RLHF 对齐：PPO 算法的核心 Loss 与显存流转 (RLHF PPO)

**难度：** Hard | **标签：** `RLHF`, `PPO`, `Alignment` | **目标人群：** 模型微调与对齐算法工程师

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/12_RLHF_PPO_Memory.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在下一节我们要学习的 DPO (直接偏好优化) 虽然因为非常节省显存而广受欢迎，但业界顶尖的大模型（如 OpenAI 系列、Llama 3）的终极对齐手段依然是 **RLHF (基于人类反馈的强化学习)**。

为什么 RLHF 很难？因为它的核心算法 **PPO (近端策略优化)** 在训练时需要**同时在显存中周转 4 个大模型**：
1. **Actor Model (策略模型)**：你要训练的那个模型，负责生成回答。
2. **Critic Model (价值模型)**：负责给 Actor 生成的每一个 Token 打分（评估当前状态的价值）。
3. **Reward Model (奖励模型)**：预先训练好的裁判，负责给整句回答打出一个最终的分数。
4. **Reference Model (参考模型)**：Actor 的冻结克隆版。用来防止 Actor 为了讨好 Reward Model 而输出人类看不懂的乱码（提供 KL 散度惩罚）。

本节我们将手写 PPO 中最核心的 **Actor Clip Loss**，并梳理这 4 个模型的运转逻辑。

### Step 1: PPO 算法与 Actor Loss 公式
    
PPO 的核心思想是：**让 Actor 往 Reward 高的方向更新，但步子不能迈得太大。**
为此，PPO 引入了优势函数 (Advantage, $A_t$) 和重要性采样比率 (Ratio, $r_t$)。

1. **重要性采样比率 $r_t$**：
   $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$
   也就是当前模型生成某个 Token 的概率，除以旧模型生成该 Token 的概率。
   
2. **Clip 截断目标函数 (Actor Loss)**：
   $L^{CLIP}(\theta) = - \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$
   如果优势 $A_t > 0$（这个 Token 很好），我们要提高它的概率，但如果 $r_t$ 已经超过 $1+\epsilon$（涨得太多了），就停止给予梯度奖励，防止模型“发疯”。

### Step 2: 动手实战
    
**要求**：完成下方的 `compute_actor_loss`。你将接收到新旧模型的概率比率 `ratio` 和优势 `advantage`。


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
    
    # ==========================================
    # TODO 1: 计算概率比率 ratio (r_t)
    # ==========================================
    # ratio = ???
    ratio = torch.ones_like(log_probs_new)  # 占位初始化
    
    # ==========================================
    # TODO 2: 计算无截断的 surrogate 目标
    # ==========================================
    # surr1 = ???
    surr1 = torch.zeros_like(advantages)  # 占位初始化
    
    # ==========================================
    # TODO 3: 计算截断后的 surrogate 目标
    # ==========================================
    # surr2 = ???
    surr2 = torch.zeros_like(advantages)  # 占位初始化
    
    # ==========================================
    # TODO 4: 计算最终的 Loss 
    # ==========================================
    # loss = ???
    loss = torch.tensor(0.0)  # 占位初始化
    
    return loss

```


```python
# 测试
def test_ppo_actor_loss():
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
        print("✅ 测试通过！PPO Actor Loss 实现通过测试。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e  # 将错误抛给测试脚本

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
