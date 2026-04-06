# 13 DPO Loss Tutorial

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_DPO_Loss_Tutorial.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 13. 直接偏好优化 Loss 源码解析与实现 (DPO)

**难度：** Hard | **标签：** `微调`, `RLHF`, `Loss Function` | **目标人群：** 模型微调与工程部署

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
    # 提示: policy_chosen_logps - reference_chosen_logps
    # ==========================================
    # pi_logratios_chosen = ???
    
    # ==========================================
    # TODO 2: 计算 rejected 样本的隐式奖励分数
    # ==========================================
    # pi_logratios_rejected = ???
    
    # 乘以 beta (控制项)
    # chosen_rewards = beta * pi_logratios_chosen
    # rejected_rewards = beta * pi_logratios_rejected
    
    # ==========================================
    # TODO 3: 计算 logits 差值并放入 sigmoid 交叉熵中
    # 目标：让 chosen 的 reward 尽可能比 rejected 的 reward 大
    # 公式：-log(sigmoid(chosen_rewards - rejected_rewards))
    # 提示：可以用 F.logsigmoid 保证数值稳定性
    # ==========================================
    # logits = ???
    # losses = ???
    
    # return losses, chosen_rewards, rejected_rewards
    pass

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
        
        print("\n✅ All Tests Passed! 恭喜你，工业级 DPO 损失函数算法实现成功！(这是 TRL 库的核心实现！)")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except TypeError as e:
        print("代码未完成，导致解包 None 错误")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

test_dpo_loss()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

直接偏好优化（DPO）损失旨在绕过强化学习阶段，通过对比正向样本和负向样本的隐式奖励来优化策略。代码计算了参考模型和策略模型对被选中和被拒绝样本的对数概率比，通过 Sigmoid 函数形成交叉熵损失。

```python
def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, 
                     reference_chosen_logps, reference_rejected_logps, 
                     beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    losses = -F.logsigmoid(beta * logits)
    rewards_chosen = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rewards_rejected = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    
    return losses.mean(), rewards_chosen.mean(), rewards_rejected.mean()
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
