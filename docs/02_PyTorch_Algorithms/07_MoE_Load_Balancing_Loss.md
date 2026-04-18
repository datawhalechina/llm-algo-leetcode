# 07. MoE Load Balancing Loss | MoE 进阶：负载均衡损失函数 (Load Balancing Loss)

**难度：** Hard | **标签：** `MoE`, `Loss Function`, `Mixtral` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在上一节 `06_MoE_Router` 中，我们实现了 Top-K 路由。但在真实的 MoE 模型（如 Mixtral 8x7B, DeepSeek）训练中，会遇到一个非常严重的问题：**路由崩塌 (Router Collapse)**。
即门控网络“偷懒”，把所有的 Token 都发给了第 0 号和第 1 号专家，导致其他专家被饿死（闲置），不仅失去了 MoE 的意义，还会导致算力非常不均衡（OOM）。
因此，面试官非常爱考：**如何用代码实现 MoE 的辅助损失函数 (Auxiliary Loss) 来强制负载均衡？**

### Step 1: 核心数学公式

为了让 $N$ 个 Token 均匀地分配给 $E$ 个专家，我们需要设计一个惩罚项，加到总的 CrossEntropy Loss 里。
Mixtral / Switch Transformer 使用的经典公式：
$$ L_{aux} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i $$

- $E$: 专家总数。
- $f_i$: 专家 $i$ 被路由到的 **Token 比例**（即选了专家 $i$ 的 token 数 / 总 token 数）。
- $P_i$: 专家 $i$ 在所有 Token 上的 **平均路由概率得分**（Softmax 之后的概率的均值）。
- $\alpha$: 辅助损失的权重系数（通常很小，如 0.01）。

**为什么这个公式有效？**
根据均值不等式，给定总和为 1 的 $f$ 和 $P$，当且仅当所有的 $f_i = 1/E$ 且 $P_i = 1/E$ 时（即绝对均匀分配），它们的内积（点积）之和最小。优化器为了降低这个 Loss，会拼命把 Token 往不同的专家那里赶！

### Step 2: 代码实现框架

你需要统计在当前批次中每个专家实际被选中的次数（形成频率分布 $f_i$），同时求出门控概率的均值分布（$P_i$）。将这两个分布点乘并乘以专家总数 $E$ 和超参数 $\alpha$，即可得到最终的 Load Balancing Loss。

**关键点**：本实现支持 Top-K 路由（不仅限于 Top-1），通过 `top_k` 参数控制每个 Token 选择的专家数量。

### Step 3: 动手实战

**要求**：请补全下方 `compute_load_balancing_loss` 的逻辑。

**注意**：本实现支持 Top-K 路由，即每个 Token 可以选择 K 个专家（通常 K=2）。


```python
import torch
import torch.nn.functional as F

```


```python
def compute_load_balancing_loss(
    routing_weights: torch.Tensor, 
    selected_experts: torch.Tensor, 
    num_experts: int, 
    top_k: int,
    alpha: float = 0.01
):
    """
    计算 MoE 的负载均衡辅助损失（支持 Top-K 路由）
    
    Args:
        routing_weights: [batch_size * seq_len, top_k]，每个 token 选中的 K 个专家的权重（已归一化）
        selected_experts: [batch_size * seq_len, top_k]，每个 token 选中的 K 个专家的索引
        num_experts: 专家总数 E
        top_k: 每个 token 选择的专家数量 K
        alpha: 损失权重系数
    
    Returns:
        aux_loss: 标量，负载均衡损失
    """
    batch_size_x_seq_len, _ = selected_experts.shape
    total_tokens = batch_size_x_seq_len
    
    # ==========================================
    # TODO 1: 计算 P_i（每个专家的平均路由概率得分）
    # 提示: routing_weights 包含了每个 token 对选中专家的权重
    # 需要统计每个专家在所有 token 上的平均权重
    # 形状应为: [num_experts]
    # ==========================================
    # P_i = ???
    
    # ==========================================
    # TODO 2: 计算 f_i（每个专家实际分到的 Token 比例）
    # 1. 使用 F.one_hot 把 selected_experts 变成 [batch_size_x_seq_len, top_k, num_experts] 的 0-1 矩阵
    # 2. 统计每个专家被选中的总次数
    # 3. 除以 (total_tokens * top_k) 得到比例
    # 形状应为: [num_experts]
    # ==========================================
    # expert_mask = ???
    # tokens_per_expert = ???
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
        num_experts = 8
        top_k = 2
        num_tokens = 1000
        alpha = 0.01
        
        # 模拟路由结果
        # 1. 极度不均衡：所有 token 都选专家 0 和 1
        bad_selected = torch.zeros(num_tokens, top_k, dtype=torch.long)
        bad_selected[:, 0] = 0
        bad_selected[:, 1] = 1
        bad_weights = torch.ones(num_tokens, top_k) / top_k
        
        loss_bad = compute_load_balancing_loss(bad_weights, bad_selected, num_experts, top_k, alpha)
        
        # 2. 绝对均匀：token 均匀分配给所有专家
        good_selected = torch.zeros(num_tokens, top_k, dtype=torch.long)
        for i in range(num_tokens):
            good_selected[i, 0] = (i * 2) % num_experts
            good_selected[i, 1] = (i * 2 + 1) % num_experts
        good_weights = torch.ones(num_tokens, top_k) / top_k
        
        loss_good = compute_load_balancing_loss(good_weights, good_selected, num_experts, top_k, alpha)
        
        print(f"极度不均衡的 Loss: {loss_bad.item():.4f}")
        print(f"绝对均匀的 Loss  : {loss_good.item():.4f}")
        
        # 理论最小值：当完全均匀时，P_i = 1/(E*top_k), f_i = 1/E
        # sum(f_i * P_i) = E * (1/E) * (1/(E*top_k)) = 1/(E*top_k)
        # aux_loss = alpha * E * 1/(E*top_k) = alpha / top_k
        expected_min = alpha / top_k  # 0.01 / 2 = 0.005
        assert torch.allclose(loss_good, torch.tensor(expected_min), atol=1e-4), f"理论最小 Loss 计算错误！期望 {expected_min:.4f}，实际 {loss_good.item():.4f}"
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
## 参考代码与解析

### 代码


```python
def compute_load_balancing_loss(
    routing_weights: torch.Tensor, 
    selected_experts: torch.Tensor, 
    num_experts: int, 
    top_k: int,
    alpha: float = 0.01
):
    batch_size_x_seq_len, _ = selected_experts.shape
    total_tokens = batch_size_x_seq_len
    
    # TODO 1: 计算 P_i（每个专家的平均路由概率得分）
    P_i = torch.zeros(num_experts, dtype=routing_weights.dtype, device=routing_weights.device)
    P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
    P_i = P_i / (total_tokens * top_k)
    
    # TODO 2: 计算 f_i（每个专家实际分到的 Token 比例）
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
    f_i = tokens_per_expert / (total_tokens * top_k)
    
    # TODO 3: 计算最终的 auxiliary loss
    aux_loss = alpha * num_experts * (f_i * P_i).sum()
    
    return aux_loss

```

### 解析

**1. TODO 1: 计算 P_i（平均路由概率）**

- **实现方式**：
  ```python
  P_i = torch.zeros(num_experts, dtype=routing_weights.dtype, device=routing_weights.device)
  P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
  P_i = P_i / (total_tokens * top_k)
  ```
- **核心逻辑**：使用 `scatter_add_` 将每个 token 对选中专家的权重累加到对应专家的位置。
- **归一化**：除以总的选择次数 `(total_tokens * top_k)` 得到平均权重。
- **物理含义**：$P_i$ 表示专家 $i$ 在所有 token 上的平均被选中概率。

**2. TODO 2: 计算 f_i（Token 分配比例）**

- **实现方式**：
  ```python
  expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
  tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
  f_i = tokens_per_expert / (total_tokens * top_k)
  ```
- **核心逻辑**：`F.one_hot` 将专家索引转换为 one-hot 编码，形状为 `[batch_size_x_seq_len, top_k, num_experts]`。
- **统计方法**：沿前两个维度求和，统计每个专家被选中的总次数。
- **归一化**：除以总的选择次数得到比例。
- **物理含义**：$f_i$ 表示专家 $i$ 实际分到的 token 比例。

**3. TODO 3: 计算辅助损失**

- **实现方式**：`aux_loss = alpha * num_experts * (f_i * P_i).sum()`
- **数学公式**：$L_{aux} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$
- **最小值分析**：根据均值不等式，当 $f_i = P_i = 1/E$ 时（完全均匀），损失最小。对于 Top-K 路由，理论最小值为 $\alpha / K$。
- **优化目标**：优化器为了降低这个 Loss，会强制将 Token 均匀分配给所有专家，防止路由崩塌。

**工程要点**

- **Top-K 兼容性**：代码支持任意 K 值，通过 `(total_tokens * top_k)` 归一化确保比例计算正确。
- **数值稳定性**：使用 `scatter_add_` 而非循环累加，提升计算效率和数值稳定性。
- **超参数调优**：$\alpha$ 通常设为 0.01，过大会影响主任务性能，过小则无法有效平衡负载。
- **与主损失结合**：在实际训练中，将 `aux_loss` 加到 CrossEntropy Loss 上：`total_loss = ce_loss + aux_loss`。
