# 12 RLHF PPO Memory

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/12_RLHF_PPO_Memory.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 12. RLHF 对齐：PPO 算法的核心 Loss 与显存流转 (RLHF PPO)

**难度：** Hard | **标签：** `RLHF`, `PPO`, `Alignment` | **目标人群：** 模型微调与对齐算法工程师

在下一节我们要学习的 DPO (直接偏好优化) 虽然因为极其节省显存而广受欢迎，但业界顶尖的大模型（如 OpenAI 系列、Llama 3）的终极对齐手段依然是 **RLHF (基于人类反馈的强化学习)**。

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
    # 提示: log(A) - log(B) = log(A/B) -> exp() 后即为 A/B
    # ==========================================
    # ratio = ???
    
    # ==========================================
    # TODO 2: 计算无截断的 surrogate 目标
    # ==========================================
    # surr1 = ???
    
    # ==========================================
    # TODO 3: 计算截断后的 surrogate 目标
    # 提示: 使用 torch.clamp
    # ==========================================
    # surr2 = ???
    
    # ==========================================
    # TODO 4: 计算最终的 Loss 
    # PPO 的目标是最大化期望奖励，所以 Loss 要加负号 (取均值)
    # ==========================================
    # loss = ???
    
    pass


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
        print("✅ 测试通过！成功掌握 PPO 强化学习的最核心微积分逻辑。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_ppo_actor_loss()


```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---


::: details 💡 点击查看官方解析与参考代码

RLHF 的核心在于通过 PPO 算法安全地更新模型参数。通过重要性采样 `ratio = exp(new_logp - old_logp)` 获取策略变化幅度，再结合 `advantage` 并利用 `torch.clamp` 进行截断（Clip），有效防止了策略模型（Actor）在追求高分时过度偏离原有分布，这是大语言模型对齐（Alignment）稳定收敛的基石。


```python
def compute_actor_loss(log_probs_new, log_probs_old, advantages, clip_range=0.2):
    # 1. 概率比率
    ratio = torch.exp(log_probs_new - log_probs_old)
    
    # 2. 两个 surrogate 函数
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    
    # 3. 最终 Loss
    loss = -torch.min(surr1, surr2).mean()
    return loss


```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
