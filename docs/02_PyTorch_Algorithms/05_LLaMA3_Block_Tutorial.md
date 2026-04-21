# 05. LLaMA3 Block Tutorial | 经典模型搭建: LLaMA-3 Transformer Block

**难度：** Medium | **标签：** `模型架构`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将进入激动人心的“组装阶段”！我们将把之前实现的 **RMSNorm**、**RoPE** 和 **GQA (Grouped-Query Attention)** 拼装在一起，外加一个 **SwiGLU** 激活函数的 MLP 层，构建一个真正的 **LLaMA-3 Decoder Layer**。这也是面试中常考的全局架构题。

### Step 1: 核心思想与痛点

> **LLaMA 架构 vs 传统 Transformer (如 GPT-2)**
> 1. **归一化位置 (Pre-Norm vs Post-Norm)**：LLaMA 使用 Pre-Norm（在 Attention 和 MLP **之前**进行归一化），这让深层网络的训练更加稳定；而早期模型多用 Post-Norm。
> 2. **归一化算法**：将 LayerNorm 替换为无偏置、不减均值的 **RMSNorm**，提升计算效率。
> 3. **激活函数**：将 ReLU/GELU 替换为 **SwiGLU**，通过门控机制（Gating）显著提升了模型的表达能力。
> 4. **位置编码**：彻底抛弃绝对位置编码，拥抱 **RoPE**。
> 5. **注意力机制**：从 LLaMA-2 开始，为了优化推理时的 KV Cache，将标准 MHA 升级为 **GQA**。

### Step 2: 模块集成框架
LLaMA-3 的 Decoder 层采用了 Pre-RMSNorm 结构。前向传播的具体流程为：
1. 输入经过 Attention 层的 RMSNorm。
2. 执行带 KV Cache 的 GQA 注意力机制（内含 RoPE 旋转）。
3. 将残差相加：`x = x + attn_out`。
4. 经过 MLP 层的 RMSNorm。
5. 执行 SwiGLU 前馈网络并再次加上残差。

###  Step 3: 核心公式与架构

**1. SwiGLU MLP:**
$$ \text{SwiGLU}(x) = (\text{Swish}(x W_{\text{gate}}) \otimes (x W_{\text{up}})) W_{\text{down}} $$
其中 $\text{Swish}(z) = z \cdot \sigma(z)$ (在 PyTorch 中对应 `F.silu`)。注意，为了保持参数量与传统 MLP 一致，LLaMA 中的隐藏层维度通常设置为 $\frac{8}{3} d$ 并向上取整。

**2. Decoder Layer 残差连接 (Residual Connections):**
$$ h = x + \text{Attention}(\text{RMSNorm}(x)) $$
$$ \text{out} = h + \text{MLP}(\text{RMSNorm}(h)) $$
*注意：这里的 Attention 内部包含了 RoPE 旋转位置编码。*

###  Step 4: 动手实战

**要求**：请补全下方 `LlamaMLP` 和 `LlamaDecoderLayer`。
为了让你直接上手核心逻辑，我们假设 `RMSNorm` 和 `Attention` 模块已经由你之前的代码提供（这里我们用 Dummy Class 占位模拟）。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
# ---------------------------------------------------------
# 以下是我们之前实现的组件 (此处用极简占位符代替，以保持代码整洁)
# ---------------------------------------------------------
class DummyRMSNorm(nn.Module):
    def __init__(self, dim): super().__init__(); self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.w

class DummyAttention(nn.Module):
    def __init__(self, dim): super().__init__(); self.proj = nn.Linear(dim, dim)
    def forward(self, x): return self.proj(x) # 假装它做了 RoPE 和 GQA
# ---------------------------------------------------------

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # ==========================================
        # TODO 1: 定义 SwiGLU 所需的三个线性层 (无 bias)
        # ==========================================
        # self.gate_proj = ???
        # self.up_proj = ???
        # self.down_proj = ???
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 2: 实现 SwiGLU 的前向传播
        # ==========================================
        # return ???
        pass

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. 注意力模块与它的前置 LayerNorm
        self.input_layernorm = DummyRMSNorm(hidden_size)
        self.self_attn = DummyAttention(hidden_size)
        
        # 2. MLP 模块与它的前置 LayerNorm
        self.post_attention_layernorm = DummyRMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 3: 实现 LLaMA 的 Pre-Norm 残差连接
        # ==========================================
        
        # --- Attention Block ---
        # residual = ???
        # h = ???
        
        # --- MLP Block ---
        # residual = ???
        # out = ???
        
        # return out
        pass

```


```python
# 运行此单元格以测试你的实现
def test_llama_block():
    try:
        batch_size, seq_len, hidden_size = 2, 16, 512
        # LLaMA 通常设置 intermediate_size 为 8/3 * hidden_size，并向 multiple_of 取整
        intermediate_size = 1376 
        
        layer = LlamaDecoderLayer(hidden_size, intermediate_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        out = layer(x)
        
        assert out.shape == (batch_size, seq_len, hidden_size), "输出形状错误！"
        
        # 简单验证一下计算图是否连通 (是否包含所有的参数)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"参数 {name} 没有接收到梯度，请检查前向传播连接！"
            
        print("\n✅ All Tests Passed! LLaMA-3 Transformer Block 组装完成，所有测试通过。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AttributeError as e:
        print(f"代码未完成: {e}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

test_llama_block()


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
# 补充前置依赖，确保答案区代码可独立运行
class DummyRMSNorm(nn.Module):
    def __init__(self, dim): super().__init__(); self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.w

class DummyAttention(nn.Module):
    def __init__(self, dim): super().__init__(); self.proj = nn.Linear(dim, dim)
    def forward(self, x): return self.proj(x)

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # TODO 1: 定义 SwiGLU 的三个线性层
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2: 实现 SwiGLU 前向传播
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_layernorm = DummyRMSNorm(hidden_size)
        self.self_attn = DummyAttention(hidden_size)

        self.post_attention_layernorm = DummyRMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO 3: 实现 Pre-Norm 残差连接

        # Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

```

### 解析

**1. TODO 1 (SwiGLU 的三个线性层定义)**

- **gate_proj（门控投影）：** 将输入从 `hidden_size` 映射到 `intermediate_size`，用于生成门控信号。这是 SwiGLU 的核心创新，通过门控机制动态调节信息流。
- **up_proj（上投影）：** 同样将输入映射到 `intermediate_size`，生成要被门控的特征。与 gate_proj 并行计算。
- **down_proj（下投影）：** 将 `intermediate_size` 映射回 `hidden_size`，完成前馈网络的降维。
- **无偏置设计：** `bias=False` 是 LLaMA 的标准配置，减少参数量并提升训练稳定性。在大规模模型中，偏置项的作用相对较小，去除后可以节省显存和计算。
- **工程细节：** 为什么需要三个线性层？传统 MLP 只有两层（up + down），而 SwiGLU 引入了门控机制，需要额外的 gate_proj 来生成门控信号。这使得模型能够动态选择哪些特征需要被激活。

**2. TODO 2 (SwiGLU 前向传播)**

- **门控计算：** `F.silu(self.gate_proj(x))` 使用 SiLU (Swish) 激活函数处理门控投影。SiLU 定义为 $f(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid 函数。
- **逐元素乘法：** `* self.up_proj(x)` 将门控信号与上投影特征进行逐元素相乘（Hadamard 积），实现动态特征选择。
- **降维输出：** `self.down_proj(...)` 将中间特征映射回原始维度。
- **数学公式：** $\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$
- **为什么用 SiLU 而不是 ReLU？** SiLU 是平滑的非线性函数，梯度更加稳定，在深层网络中表现更好。ReLU 在负值区域梯度为 0，容易导致神经元死亡。
- **进阶思考：** 为什么 `intermediate_size` 通常是 $\frac{8}{3} \times \text{hidden\_size}$？这是为了在引入门控机制后，保持与传统 MLP（通常是 $4 \times \text{hidden\_size}$）相近的参数量。由于 SwiGLU 需要两个投影（gate + up），所以每个投影的维度相应减小。

**3. TODO 3 (Pre-Norm 残差连接)**

- **Pre-Norm 架构：** 在每个子层（Attention 或 MLP）**之前**进行归一化，这是 LLaMA 相对于早期 Transformer（如 GPT-2）的重要改进。
- **Attention Block 流程：**
  1. 保存残差：`residual = hidden_states`
  2. 归一化：`hidden_states = self.input_layernorm(hidden_states)`
  3. 注意力计算：`hidden_states = self.self_attn(hidden_states)`（内部包含 RoPE 和 GQA）
  4. 残差连接：`hidden_states = residual + hidden_states`
- **MLP Block 流程：** 与 Attention Block 完全对称，只是将 self_attn 替换为 mlp。
- **为什么 Pre-Norm 更好？** Post-Norm（先计算再归一化）在深层网络中容易出现梯度爆炸或消失。Pre-Norm 将归一化放在前面，使得每个子层的输入都是归一化的，梯度更加稳定，训练更容易收敛。
- **残差连接的作用：** 提供梯度的"高速公路"，使得梯度可以直接从输出层反向传播到输入层，缓解深层网络的梯度消失问题。这是训练超过 100 层 Transformer 的关键技术。

**进阶思考：LLaMA 架构的五大创新**

1. **Pre-Norm 拓扑：** 相比 Post-Norm，训练更稳定，支持更深的网络。
2. **RMSNorm 替代 LayerNorm：** 去除均值计算和偏置，速度提升约 10-15%，且在大规模训练中表现相当。
3. **SwiGLU 激活函数：** 门控机制带来更强的表达能力，在多个基准测试中优于 GELU 和 ReLU。
4. **RoPE 位置编码：** 相对位置编码，支持长度外推，是当前大模型的标配。
5. **GQA 注意力：** 在 LLaMA-2/3 中引入，大幅减少 KV Cache 显存占用（相比 MHA 减少 8 倍），同时保持接近 MHA 的性能。

**工程实践：**
- **LLaMA-3 8B**：32 层 Decoder Layer，hidden_size=4096，32 个 Query 头，8 个 KV 头（GQA 比例 4:1）。
- **LLaMA-3 70B**：80 层 Decoder Layer，hidden_size=8192，64 个 Query 头，8 个 KV 头（GQA 比例 8:1）。
- **训练技巧：** 使用 BF16 混合精度训练，梯度裁剪（clip_grad_norm=1.0），AdamW 优化器（β1=0.9, β2=0.95），余弦学习率衰减。
