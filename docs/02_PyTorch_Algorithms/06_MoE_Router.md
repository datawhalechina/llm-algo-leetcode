# 06 MoE Router

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/06_MoE_Router.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 06. 混合专家架构: 稀疏路由与负载均衡 (MoE)

**难度：** Medium | **标签：** `模型架构`, `MoE`, `PyTorch` | **目标人群：** 模型微调与工程部署

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
在门控网络中，我们首先计算输入对各个专家的打分矩阵（logits），然后通过 `torch.topk` 获取最大的 K 个分数及其对应的专家索引。最后对这 K 个分数应用 Softmax 进行归一化，作为专家输出的加权系数。


###  Step 3: 核心数学机制：Top-K Routing

**1. 门控网络 (Gating / Router)：**
给定输入 Token 的特征 $x \in \mathbb{R}^d$，我们用一个线性层将其映射到各个专家的打分：
$$ h = x W_{gate} \quad (h \in \mathbb{R}^E) $$
其中 $E$ 是专家总数（如 8）。

**2. 归一化与 Top-K 选择：**
传统的 Softmax 会让所有专家的权重都不为 0：
$$ p = \text{Softmax}(h) $$
MoE 需要的是**稀疏性**，因此我们只保留得分最高的 $K$ 个专家，将其余专家的权重强制置为 0，并对保留的权重重新进行 Softmax 归一化：
$$ p_{topk} = \text{Softmax}(\text{TopK}(h, K)) $$

**3. 最终输出融合：**
Token 经过这 $K$ 个专家的计算后，按权重加权求和：
$$ y = \sum_{i \in \text{TopK}} p_i \cdot \text{Expert}_i(x) $$


###  Step 4: 动手实战

**要求**：请补全下方 `TopKRouter` 函数。
这也是面试中非常经典的 `torch.topk`、`scatter` 和 `gather` 等高级张量操作的考察点。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            routing_weights: 形状 [batch_size * seq_len, top_k]，表示选中的专家的权重
            selected_experts: 形状 [batch_size * seq_len, top_k]，表示选中的专家索引
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # 将张量展平，把所有的 Token 一视同仁地排队
        # 形状: [batch_size * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_size)
        
        # 1. 计算每个 token 对于每个专家的原始 logits 得分
        # 形状: [batch_size * seq_len, num_experts]
        router_logits = self.gate(hidden_states)
        
        # ==========================================
        # TODO 1: 选出得分最高的 Top-K 个专家的值 (routing_weights) 和索引 (selected_experts)
        # 提示: 使用 torch.topk 函数，在最后一维上操作
        # ==========================================
        # routing_weights, selected_experts = ???
        
        # ==========================================
        # TODO 2: 对选中的 Top-K 权重进行 Softmax 归一化
        # 提示: 让这 K 个专家的权重加起来等于 1
        # ==========================================
        # routing_weights = ???
        
        # (通常我们需要转换为 FP32 来保证精度的稳定性)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        return routing_weights, selected_experts

# 为了验证 Router 能正确工作，我们写一个极简的 MoE 聚合层
class SparseMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        # 这里用极简的 Linear 模拟 Expert (真实的 Expert 可能是 SwiGLU MLP)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        
    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 获取路由权重和索引
        routing_weights, selected_experts = self.router(hidden_states)
        
        # 准备一个空张量存放最终输出
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_size), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        
        # 展平输入
        flat_hidden_states = hidden_states.view(-1, hidden_size)
        
        # 遍历所有被选中的 expert
        # 注意：工业界 (如 vLLM/Megatron) 会通过高效的索引排序或 Triton 算子，
        # 将去往同一个专家的 token 汇聚成一个批次一次性计算。
        # 这里为了便于理解核心逻辑，我们使用 for 循环遍历每一个 Token 选择的专家
        
        for expert_idx, expert in enumerate(self.experts):
            # 找到哪些 token (在 top_k 的哪个位置) 选择了当前这个专家
            # selected_experts 形状: [num_tokens, top_k]
            token_idx, kth_expert = torch.where(selected_experts == expert_idx)
            
            if token_idx.shape[0] > 0:
                # 把这些 token 抽出来送进专家计算
                current_state = flat_hidden_states[token_idx]
                current_output = expert(current_state)
                
                # 乘以该专家对应的路由权重
                # routing_weights[token_idx, kth_expert] 获取了一维的权重列表，需要 unsqueeze 适配形状
                current_weight = routing_weights[token_idx, kth_expert].unsqueeze(-1)
                
                # 累加到最终结果里
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
        
        # 2. 验证 Router 行为 (只提取路由函数单独测)
        weights, indices = moe.router(x)
        
        assert weights.shape == (batch_size * seq_len, top_k), "权重形状不等于 [num_tokens, top_k]！"
        assert indices.shape == (batch_size * seq_len, top_k), "索引形状不等于 [num_tokens, top_k]！"
        
        # 验证 Top-K Softmax 归一化是否正确 (每一行的和应非常接近 1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size * seq_len)), "Top-K 权重之和不等于 1，Softmax 归一化失败！"
        
        # 3. 验证专家索引是否合法
        assert torch.all((indices >= 0) & (indices < num_experts)), "挑选的专家索引越界！"
        
        print("\n✅ All Tests Passed! 恭喜你，最前沿的 MoE Top-K Router 和稀疏聚合逻辑已被你攻克！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except TypeError as e:
        print("代码可能未完成，导致变量为 NoneType。")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_moe_router()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

混合专家模型（MoE）的灵魂在于稀疏路由机制。输入特征首先通过一个门控全连接层映射到所有专家的得分。接着利用 torch.topk 选取得分最高的 K 个专家（保留分数与索引），对这 K 个分数做局部 Softmax 归一化。在聚合阶段，不同于常规稠密计算，只有被选中的专家才会对特定 Token 的计算进行加权求和，从而以极小的激活参数量扩展了模型的总容量。

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
        
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        return routing_weights, selected_experts
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
