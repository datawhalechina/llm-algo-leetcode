# 06. MoE Router | 混合专家架构: 稀疏路由与负载均衡 (MoE)

**难度：** Medium | **标签：** `模型架构`, `MoE`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/06_MoE_Router.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将解析目前最火爆的模型架构：**MoE (Mixture of Experts)**。这也是 Mixtral、Grok、DeepSeek 等顶级开源模型背后的核心技术。
面试中最常考的并不是专家的内部结构，而是那个“交通警察”——**路由机制 (Router) 和专家权重计算**。

### Step 1: 核心思想与痛点

> **Dense (稠密) 模型的痛点：**
> 在标准的 Transformer 中，每一个 Token 都必须经过全网络的所有参数（比如 70B 的 LLaMA）。这导致随着模型变大，推理和训练的计算量呈线性爆炸。
> 
> **MoE 的破局：稀疏激活 (Sparse Activation)**
> 将原来大规模的 MLP 层，切分成 $N$ 个小型的独立 MLP（称为 Expert，专家）。
> 对于每一个输入的 Token，通过一个非常轻量的 Router (门控网络) 决定它该去请教哪 $K$ 个专家（通常 $K=2$）。
> 这样，即便总参数量有 8x7B=56B，实际每个 Token 只激活了 2x7B=14B 的参数。**计算量骤降，而知识容量剧增。**

### Step 2: 代码实现框架
在门控网络中，首先计算输入对所有专家的打分矩阵（logits）。**关键陷阱**：必须先在全维度（num_experts）上进行 Softmax 将打分转为概率分布，然后再通过 `torch.topk` 获取最大的 K 个概率及其对应的专家索引。最后，为了保证加权和仍为 1，必须对截取出的 K 个概率值进行重归一化（Re-normalize）。

###  Step 3: 核心数学机制：Top-K Routing

**1. 门控网络 (Gating / Router)：**
给定输入 Token 的特征 $x \in \mathbb{R}^d$，我们用一个线性层将其映射到各个专家的打分：
$$ h = x W_{gate} \quad (h \in \mathbb{R}^E) $$
其中 $E$ 是专家总数（如 8）。

**2. 全局归一化与 Top-K 选择 (The Softmax Trap)：**
传统初学者容易犯的错误是先选 Top-K 的 Logits，再做 Softmax。正确的工业级做法（如 Mixtral 8x7B）必须是先做全局 Softmax：
$$ p = \text{Softmax}(h) \quad (p \in \mathbb{R}^E) $$
为了保持稀疏性，我们提取其中概率最高的 $K$ 个专家：
$$ p_{topk}, idx_{topk} = \text{TopK}(p, K) $$

**3. 局部重归一化 (Re-normalize)：**
由于截取了部分概率，剩下的 $K$ 个概率之和不再为 1。为了稳定梯度的尺度，必须按比例将其重新归一化：
$$ w_i = \frac{p_i}{\sum_{j \in TopK} p_j} $$

**4. 最终输出融合：**
Token 经过这 $K$ 个专家的计算后，按最新权重加权求和：
$$ y = \sum_{i \in TopK} w_i \cdot \text{Expert}_{idx_i}(x) $$

###  Step 4: 动手实战

**要求**：请补全下方 `TopKRouter` 函数。
这也是面试中非常经典的 `torch.topk`、`scatter` 和 `gather` 等高级张量操作的考察点。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 定义门控层，将隐藏状态映射到专家数量的得分
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            routing_weights: 形状 [batch_size * seq_len, top_k]，表示选中的专家的权重 (重归一化后)
            selected_experts: 形状 [batch_size * seq_len, top_k]，表示选中的专家索引
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # 展平输入
        hidden_states = hidden_states.view(-1, hidden_size)
        
        # 1. 计算 logits 得分
        router_logits = self.gate(hidden_states)
        
        # ==========================================
        # TODO 1: 对全量 Logits 进行 Softmax 获取所有专家的概率分布
        # 提示: 强制使用 FP32 以防止精度溢出 (router_logits.float())
        # ==========================================
        # routing_probs = ???
        
        # ==========================================
        # TODO 2: 从概率分布中截取 Top-K 最大的概率 (routing_weights) 及其索引 (selected_experts)
        # ==========================================
        # routing_weights, selected_experts = ???
        
        # ==========================================
        # TODO 3: 对截取后的 routing_weights 进行重归一化 (Re-normalize)
        # 提示: 让这 K 个专家的概率按比例放大，使其加和等于 1
        # ==========================================
        # routing_weights = ???
                                                                                                    
        routing_weights = torch.zeros((hidden_states.shape[0], self.top_k), dtype=torch.float32, device=hidden_states.device)  # 占位初始化                                                    
        selected_experts = torch.zeros((hidden_states.shape[0], self.top_k), dtype=torch.long, device=hidden_states.device) # 占位初始化   
        
        # 恢复到原始数据类型
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        return routing_weights, selected_experts

# 为了验证 Router 能正确工作，我们写一个极简的 MoE 聚合层
class SparseMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        # 极简模拟 Expert (真实的 Expert 通常是 SwiGLU MLP)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        
    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        routing_weights, selected_experts = self.router(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_size), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        flat_hidden_states = hidden_states.view(-1, hidden_size)
        
        # 工业界(vLLM/Megatron)会通过 Token Sorting (索引排序) 汇聚同专家的Token，
        # 这里为便于理解核心算法逻辑，使用 For 循环遍历被选中的 Expert
        for expert_idx, expert in enumerate(self.experts):
            token_idx, kth_expert = torch.where(selected_experts == expert_idx)
            if token_idx.shape[0] > 0:
                current_state = flat_hidden_states[token_idx]
                current_output = expert(current_state)
                current_weight = routing_weights[token_idx, kth_expert].unsqueeze(-1)
                final_hidden_states[token_idx] += current_output * current_weight
                
        return final_hidden_states.view(batch_size, seq_len, hidden_size)

```


```python
# 运行此单元格以测试你的实现
def test_moe_router():
    try:
        torch.manual_seed(42)
        batch_size, seq_len, hidden_size = 2, 4, 16
        num_experts, top_k = 8, 2
        
        moe = SparseMoEBlock(hidden_size, num_experts, top_k)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # 1. 验证输出形状
        out = moe(x)
        assert out.shape == x.shape, "MoE 聚合后的输出形状不匹配！"
        
        # 2. 验证 Router 行为
        weights, indices = moe.router(x)
        assert weights.shape == (batch_size * seq_len, top_k), "权重形状不等于 [num_tokens, top_k]！"
        assert indices.shape == (batch_size * seq_len, top_k), "索引形状不等于 [num_tokens, top_k]！"
        
        # 验证重归一化是否正确 (每一行的和应非常接近 1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size * seq_len, dtype=weights.dtype)), "重归一化失败：Top-K 权重之和不等于 1！"
        
        # 3. 验证专家索引合法性
        assert torch.all((indices >= 0) & (indices < num_experts)), "挑选的专家索引越界！"
        
        print("\n✅ All Tests Passed! MoE Top-K Router 和稀疏聚合逻辑验证通过。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except TypeError as e:
        print("代码可能未完成，导致变量为 NoneType。")
        raise e  # 将错误抛给测试脚本
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise e  # 将错误抛给测试脚本

test_moe_router()

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
class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        
        router_logits = self.gate(hidden_states)
        
        # TODO 1: 全局 Softmax 转换为概率分布
        routing_probs = F.softmax(router_logits.float(), dim=-1)
        
        # TODO 2: 截取概率最大的 Top-K 专家
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # TODO 3: 重归一化
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)
        return routing_weights, selected_experts

class SparseMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        
    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        routing_weights, selected_experts = self.router(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_size), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        flat_hidden_states = hidden_states.view(-1, hidden_size)
        
        for expert_idx, expert in enumerate(self.experts):
            token_idx, kth_expert = torch.where(selected_experts == expert_idx)
            if token_idx.shape[0] > 0:
                current_state = flat_hidden_states[token_idx]
                current_output = expert(current_state)
                current_weight = routing_weights[token_idx, kth_expert].unsqueeze(-1)
                final_hidden_states[token_idx] += current_output * current_weight
                
        return final_hidden_states.view(batch_size, seq_len, hidden_size)

```

### 解析

**1. TODO 1: 全局 Softmax 转换**

- **实现方式**：`routing_probs = F.softmax(router_logits.float(), dim=-1)`
- **关键点**：必须在全维度（`num_experts`）上进行 Softmax，将原始打分转换为概率分布。
- **精度控制**：强制使用 `.float()` 转为 FP32 精度，防止 FP16/BF16 下的数值溢出。Softmax 对数值精度极其敏感，低精度会导致概率分布崩塌。
- **核心陷阱**：新手容易犯的错误是先截取 Top-K 的 Logits 再做 Softmax（局部归一化），这会失去全局相对置信度。正确做法是先全局 Softmax，再截取 Top-K 概率。

**2. TODO 2: Top-K 截取**

- **实现方式**：`routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)`
- **关键点**：从全局概率分布中提取最大的 K 个概率值及其对应的专家索引。
- **本质区别**：这里截取的是**概率**而非原始 logits，这是与错误做法的核心差异。
- **工业实践**：Mixtral 8x7B、DeepSeek 等主流 MoE 模型均采用此方法。

**3. TODO 3: 重归一化**

- **实现方式**：`routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)`
- **必要性**：截取后的 K 个概率之和不再为 1，需要按比例放大使其重新归一化，以稳定梯度的尺度。
- **技术细节**：`keepdim=True` 保持维度以支持广播除法。
- **数学原理**：根据均值不等式，当所有专家的 $f_i = P_i = 1/E$ 时（完全均匀），损失最小。

**工程优化要点**

- **稀疏激活的本质**：MoE 的核心价值在于"大容量、低激活"。通过 Top-K 路由，每个 Token 只激活少数专家（通常 K=2），使得 56B 参数的模型实际计算量仅相当于 14B 的稠密模型。
- **高效聚合**：代码中的 `SparseMoEBlock` 使用 For 循环遍历专家是为了便于理解。工业界框架（vLLM、Megatron-LM）会使用 Token Sorting（按专家索引排序）将去往同一专家的 Token 汇聚成批次，一次性计算以提升 GPU 利用率。
