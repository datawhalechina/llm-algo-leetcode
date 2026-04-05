# 06 MoE Load Balancing Loss

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/06_MoE_Load_Balancing_Loss.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 06. MoE 进阶：负载均衡损失函数 (Load Balancing Loss)

**难度：** Hard | **标签：** `MoE`, `Loss Function`, `Mixtral` | **目标人群：** 核心 Infra 与算子开发

在上一节 `05_MoE_Router` 中，我们实现了 Top-K 路由。但在真实的 MoE 模型（如 Mixtral 8x7B, DeepSeek）训练中，会遇到一个非常致命的问题：**路由崩塌 (Router Collapse)**。
即门控网络“偷懒”，把所有的 Token 都发给了第 0 号和第 1 号专家，导致其他专家被饿死（闲置），不仅失去了 MoE 的意义，还会导致算力极度不均衡（OOM）。
因此，面试官极度爱考：**如何用代码实现 MoE 的辅助损失函数 (Auxiliary Loss) 来强制负载均衡？**


### Step 1: 核心数学公式

为了让 $N$ 个 Token 均匀地分配给 $E$ 个专家，我们需要设计一个惩罚项，加到总的 CrossEntropy Loss 里。
Mixtral / Switch Transformer 使用的经典公式：
$$ L_{aux} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i $$

*   $E$: 专家总数。
*   $f_i$: 专家 $i$ 被路由到的 **Token 比例** (即选了专家 $i$ 的 token 数 / 总 token 数)。
*   $P_i$: 专家 $i$ 在所有 Token 上的 **平均路由概率得分** (Softmax 之后的原始得分的均值)。
*   $\alpha$: 辅助损失的权重系数 (通常很小，如 0.01)。

**为什么这个公式有效？**
根据均值不等式，给定总和为 1 的 $f$ 和 $P$，当且仅当所有的 $f_i = 1/E$ 且 $P_i = 1/E$ 时（即绝对均匀分配），它们的内积（点积）之和最小。优化器为了降低这个 Loss，会拼命把 Token 往不同的专家那里赶！


### Step 2: 负载均衡损失的数学原理
为了防止所有 Token 都涌向少数几个专家（导致计算瓶颈和崩溃），我们需要引入辅助损失函数：
$$ L_{aux} = N \sum_{i=1}^E f_i \cdot P_i $$
其中 $f_i$ 是当前 Batch 内被路由到专家 $i$ 的 Token 比例，$P_i$ 是路由网络输出给专家 $i$ 的平均概率。目标是最小化两者的乘积，迫使分配更加均匀。


### Step 3: 代码实现框架
你需要统计在当前批次中每个专家实际被选中的次数（形成频率分布），同时求出门控概率的均值分布。将这两个分布点乘并乘以专家总数和超参数 $\alpha$，即可得到最终的 Load Balancing Loss。


###  Step 4: 动手实战

**要求**：请补全下方 `compute_load_balancing_loss` 的逻辑。
（注：这里为了简化，我们假设 Top-K 的 $K=1$，即每个 Token 只选 1 个专家）


```python
import torch
import torch.nn.functional as F

def compute_load_balancing_loss(router_logits: torch.Tensor, num_experts: int, alpha: float = 0.01):
    """
    计算 MoE 的负载均衡辅助损失 (假设 Top-1 路由)
    
    Args:
        router_logits: [batch_size * seq_len, num_experts]，门控网络输出的原始打分
        num_experts: 专家数量 E
        alpha: 损失权重系数
    """
    num_tokens = router_logits.size(0)
    
    # 1. 计算每个 token 对各个专家的概率分布 P
    # 形状: [num_tokens, num_experts]
    routing_probs = F.softmax(router_logits, dim=-1)
    
    # 2. 找出每个 token 实际选择的专家索引 (Top-1)
    # 形状: [num_tokens]
    _, selected_experts = torch.max(routing_probs, dim=-1)
    
    # ==========================================
    # TODO 1: 计算 P_i (每个专家的平均路由概率得分)
    # 提示: 沿着 token 维度 (dim=0) 求均值
    # 形状应为: [num_experts]
    # ==========================================
    # P_i = ???
    
    # ==========================================
    # TODO 2: 计算 f_i (每个专家实际分到的 Token 比例)
    # 1. 使用 F.one_hot 把 selected_experts 变成 [num_tokens, num_experts] 的 0-1 矩阵
    # 2. 沿着 token 维度求均值，得到比例
    # 形状应为: [num_experts]
    # ==========================================
    # expert_mask = F.one_hot(???, num_classes=num_experts).float()
    # f_i = ???
    
    # ==========================================
    # TODO 3: 计算最终的 auxiliary loss
    # 公式: alpha * num_experts * sum(f_i * P_i)
    # ==========================================
    # aux_loss = ???
    
    # return aux_loss
    pass

```

```python
# 测试你的实现
def test_aux_loss():
    try:
        torch.manual_seed(42)
        num_experts = 4
        num_tokens = 1000
        
        # 1. 模拟一个极度不均衡的 logits (所有 token 都倾向于去专家 0)
        bad_logits = torch.randn(num_tokens, num_experts)
        bad_logits[:, 0] += 5.0  # 强行给专家 0 加分
        
        loss_bad = compute_load_balancing_loss(bad_logits, num_experts)
        
        # 2. 模拟一个绝对均匀的 logits
        good_logits = torch.zeros(num_tokens, num_experts) 
        # (为了让 argmax 均匀分布，我们构造一下)
        for i in range(num_tokens):
            good_logits[i, i % num_experts] = 10.0
            
        loss_good = compute_load_balancing_loss(good_logits, num_experts)
        
        print(f"极度不均衡的 Loss: {loss_bad.item():.4f}")
        print(f"绝对均匀的 Loss  : {loss_good.item():.4f}")
        
        # 均匀分配时的理论最小 Loss 应该是 alpha (0.01)
        assert torch.allclose(loss_good, torch.tensor(0.01), atol=1e-4), "理论最小 Loss 计算错误！"
        assert loss_bad > loss_good * 2, "惩罚项没有对不均衡分布产生足够大的 Loss！"
        
        print("\n✅ All Tests Passed! 你成功掌握了 Mixtral / DeepSeek 的防崩塌核心技术！")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except TypeError:
        print("代码未完成导致返回 None 错误。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_aux_loss()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

混合专家模型（MoE）的负载均衡损失函数（Load Balancing Loss）旨在解决专家不平衡问题。它通过计算各专家被路由的平均概率与平均分配比例的乘积，并求和，确保所有的 Token 能够均匀地分配给不同的专家，防止少数专家过载而大部分专家空闲。

```python
def compute_load_balancing_loss(routing_weights, selected_experts, num_experts):
    batch_size_x_seq_len, top_k = selected_experts.shape
    total_tokens = batch_size_x_seq_len
    
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
    tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
    f_i = tokens_per_expert / (total_tokens * top_k)
    
    P_i = routing_weights.mean(dim=0)
    
    loss = (f_i * P_i).sum() * num_experts
    return loss
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
