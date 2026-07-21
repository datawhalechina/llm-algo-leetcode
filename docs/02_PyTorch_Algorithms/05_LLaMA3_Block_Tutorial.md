# 05. LLaMA3 Block Tutorial | LLaMA3 Block 教程

**难度：** Medium | **环境：** CPU-first | **标签：** `模型架构`, `Transformer`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


前面的 `00-04` 不是零散的预热，而是在为这一节铺底：先把 Python、数据入口、对象封装和张量思维立住，再把这些底座接到真正的 LLaMA-3 Decoder Layer 上。到了这里，读者不再只是在看单个 API，而是在看一个完整 Block 怎样从组件拼成主干。

本节进入激动人心的“组装阶段”！我们会把之前实现的 **RMSNorm**、**RoPE** 和 **GQA (Grouped-Query Attention)** 拼装在一起，外加一个 **SwiGLU** 激活函数的 MLP 层，构建一个真正的 **LLaMA-3 Decoder Layer**。可以先把 Decoder Layer 记成一条固定流水线：先归一化，再做 Attention / MLP 变换，最后通过残差连接保留原始输入，并和新特征相加。

**关键词：** `LLaMA3`, `Transformer Block`, `Decoder Layer`
## 前置阅读

**导语：** 如果还没把组成 Block 的关键组件理顺，先看下面几页再进入 LLaMA-3 Block 会更顺。

- [01. RMSNorm Tutorial | RMSNorm 教程](../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.md)
- [02. SwiGLU Activation | SwiGLU 激活](../02_PyTorch_Algorithms/02_SwiGLU_Activation.md)
- [03. RoPE Tutorial | 旋转位置编码教程](../02_PyTorch_Algorithms/03_RoPE_Tutorial.md)
- [04. Attention MHA GQA | 多头注意力](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)


## 相关阅读

**导语：** 本节先把 Block 组装讲清楚；如果想继续补实现封装和硬件、精度、融合优化背景，再看下面几页。

- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

### Step 1: 核心思想与痛点

LLaMA 系列模型在 Transformer 基础上做了多项关键改进。这些改进共同构成了 LLaMA 的架构基础，也是后续实现 Decoder 层的核心依据。

**LLaMA 架构 vs 传统 Transformer (如 GPT-2)**

1. **归一化位置 (Pre-Norm vs Post-Norm)**：LLaMA 使用 Pre-Norm（在 Attention 和 MLP **之前**进行归一化），这让深层网络的训练更加稳定；而早期模型多用 Post-Norm。
2. **归一化算法**：将 LayerNorm 替换为无偏置、不减均值的 **RMSNorm**，提升计算效率。
3. **激活函数**：将 ReLU/GELU 替换为 **SwiGLU**，通过门控机制（Gating）显著提升了模型的表达能力。
4. **位置编码**：彻底抛弃绝对位置编码，拥抱 **RoPE**。
5. **注意力机制**：从 LLaMA-2 开始，为了优化推理时的 KV Cache，将标准 MHA 升级为 **GQA**（通过减少 KV 头数压缩 Cache 大小，详见 04 Step1 的架构对比）。

### Step 2: 模块集成框架

LLaMA-3 的单个 Decoder 层遵循 Pre-Norm 架构，沿着一条固定的前向流水线处理输入：先归一化，再通过 Attention/MLP 做变换，最后通过残差连接与原始输入相加。
1. 输入经过 Attention 层的 **RMSNorm**。
2. 执行带 KV Cache 的 GQA 注意力机制（Q/K 投影后应用 RoPE 旋转）。
3. 将残差相加：`x = x + attn_out`。
4. 经过 MLP 层的 **RMSNorm**。
5. 执行 SwiGLU 前馈网络并再次加上残差。`x = x + mlp_out`


### Step 3: 核心公式与架构

前面的模块已经拼起来了，这一步先把 Decoder Layer 的公式骨架和残差关系画清楚。

**1. SwiGLU MLP:**
$$ \text{SwiGLU}(x) = (\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}} $$
其中 $\text{Swish}(z) = z \cdot \sigma(z)$ (在 PyTorch 中对应 `F.silu`)，其中 `⊙` 表示逐元素乘（Hadamard product）。注意，为了保持参数量与传统 MLP 一致，LLaMA 中的隐藏层维度通常设置为 $\frac{8}{3} d$ 并向上取整（约 2.67 倍），其中 d 为 hidden_dim。

**2. Decoder Layer 残差连接 (Residual Connections):**
$$ h = x + \text{Attention}(\text{RMSNorm}(x)) $$
$$ \text{out} = h + \text{MLP}(\text{RMSNorm}(h)) $$
*注意：这里的 Attention 内部包含了 RoPE 旋转位置编码（作用于 Q 和 K 投影后）*

### Step 4: 动手实战

骨架画清楚之后，这里开始把 RMSNorm、Attention、SwiGLU 和残差连接逐个落到代码里，真正拼成一个完整的 Decoder Layer。

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
    """占位 RMSNorm，仅用于验证结构，不做真实归一化。"""
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
        # 提示: gate_proj / up_proj 将 hidden_size 映射到 intermediate_size
        #      down_proj 将 intermediate_size 映射回 hidden_size
        # ==========================================
        # self.gate_proj = ???
        # self.up_proj = ???
        # self.down_proj = ???
        raise NotImplementedError("请先完成 TODO 部分的代码！")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 2: 实现 SwiGLU 的前向传播
        # 提示: gate 分支先过 F.silu，再和 up 分支逐元素相乘，最后过 down_proj
        # ==========================================
        # hidden_states = ???
        # output = ???
        return output

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
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # ==========================================
        # TODO 3: 实现 LLaMA 的 Pre-Norm 残差连接
        # 提示: 先做 Attention residual，再做 MLP residual
        # ==========================================
        # --- Attention Block ---
        # residual = ???
        # hidden_states = ???
        # hidden_states = ???
        # hidden_states = ???
        
        # --- MLP Block ---
        # residual = ???
        # hidden_states = ???
        # hidden_states = ???
        # hidden_states = ???
        
        return hidden_states

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
        raise
    except (AttributeError, NameError, TypeError) as e:
        print(f"代码可能未完成: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise

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
    """占位 RMSNorm，仅用于验证结构，不做真实归一化。"""
    def __init__(self, dim): super().__init__(); self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.w

class DummyAttention(nn.Module):
    def __init__(self, dim): super().__init__(); self.proj = nn.Linear(dim, dim)
    def forward(self, x): return self.proj(x)

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # gate 和 up 分支并行投影，再交给 down_proj 还原维度。
        # TODO 1: 定义 SwiGLU 的三个线性层
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先门控，再逐元素相乘，最后降维回 hidden_size。
        # TODO 2: 实现 SwiGLU 前向传播
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Pre-Norm 的核心是：先规范输入，再让子层做变换，最后用残差绕回主干。

        self.input_layernorm = DummyRMSNorm(hidden_size)
        self.self_attn = DummyAttention(hidden_size)

        self.post_attention_layernorm = DummyRMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
            逐层执行 Pre-Norm 残差连接。

        Args:
            hidden_states: 输入张量，形状为 [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
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

### 答案与直觉

- **这一节要解决什么：** 把前面单独实现的归一化、注意力和 MLP 组合成一个完整 Decoder Layer。
- **为什么这样拼：** Pre-Norm + Residual 是 LLaMA 稳定训练的关键，模块顺序不能随意交换。
- **带走的直觉：** Block 不是把模块机械相加，而是把“归一化 - 变换 - 残差”串成一条稳定链路。

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
- **数学公式：** $\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$，其中 ⊙ 表示逐元素乘（Hadamard product），`W_gate` 为门控投影权重，`W_up` 为上投影权重，`W_down` 为下投影权重。
- **为什么用 SiLU 而不是 ReLU？** SiLU 是平滑的非线性函数，梯度更加稳定，在深层网络中表现更好。ReLU 在负值区域梯度为 0，容易导致神经元死亡。
- **进阶思考：** 为什么 `intermediate_size` 通常是 $\frac{8}{3} \times \text{hidden\_size}$？传统 MLP 的 FFN 扩张比例为 4 倍（`4 × hidden_size`），参数量为 `hidden_size × 4 × hidden_size`。SwiGLU 使用了 gate 和 up 两个投影，若保持相同的参数量，`intermediate_size` 应约为 `2 × hidden_size`。实际 LLaMA 使用 `8/3` 倍（约 2.67 倍），略高于 2 倍，是在保持参数量与表达能力之间经过实验调优后的折中选择。"

**3. TODO 3 (Pre-Norm 残差连接)**

- **Pre-Norm 架构：** 在每个子层（Attention 或 MLP）**之前**进行归一化，这是 LLaMA 相对于早期 Transformer（如 GPT-2）的重要改进。
- **Attention Block 流程：**
  1. 保存残差：`residual = hidden_states`
  2. 归一化：`hidden_states = self.input_layernorm(hidden_states)`
  3. 注意力计算：`hidden_states = self.self_attn(hidden_states)`（内部包含 RoPE 和 GQA）
  4. 残差连接：`hidden_states = residual + hidden_states`
- **MLP Block 流程：** 与 Attention Block 完全对称，只是将 self_attn 替换为 mlp。
- **残差连接公式**
 $$
h = x + \text{Attention}(\text{RMSNorm}(x))
$$
$$
\text{out} = h + \text{MLP}(\text{RMSNorm}(h))
$$
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

### 扩展：最小数据逻辑与训练循环

这一段是可选扩展，主要把前面的 Block 组装继续接到最小训练闭环。它不影响主线，但能帮助你把“模块怎么拼”进一步落到“模型怎么跑起来”。

```python
from typing import Optional
class TinyLlamaLM(nn.Module):
    """
        最小 LLaMA 语言模型，仅用于验证 DecoderLayer 结构正确性。

        结构：Embedding → 1 × DecoderLayer → LM Head
        注意：此实现仅为教学验证用途，非生产级模型。
    """
    def __init__(self, vocab_size: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        # 最小 LM 只保留三层：Embedding -> Decoder Layer -> LM Head
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.decoder = LlamaDecoderLayer(hidden_size, intermediate_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 1) 把 token id 映射成 hidden states
        hidden_states = self.embed_tokens(input_ids)
        # 2) 经过前面已经组装好的 LLaMA-3 Decoder Layer
        hidden_states = self.decoder(hidden_states)
        # 3) 投影到词表维度，得到 next-token logits
        logits = self.lm_head(hidden_states)
        return logits


def build_toy_batch(
    batch_size: int = 2,
    seq_len: int = 4,
    vocab_size: int = 16,
    device: torch.device = None,
    ):
    """
    构造一个 toy batch，用于最小训练闭环验证。

    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        vocab_size: 词表大小，生成的 token id 在 [0, vocab_size-1] 范围内
        device: 张量所在设备

    Returns:
        input_ids: [batch_size, seq_len]
        labels: [batch_size, seq_len]
    """
    # 一个极小的 next-token toy batch，用来演示最小训练闭环。
    # input_ids 是模型输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # labels 是 input_ids 的简单右移版本
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    # 将最后一个位置的标签设为 0（避免越界，实际训练中会 mask 掉）
    labels[:, -1] = 0
    return input_ids, labels


def train_one_step(
    model: nn.Module, 
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    )-> tuple[torch.Tensor, torch.Tensor]:
    """
    执行一步训练。

    Args:
        model: 要训练的模型
        batch: (input_ids, labels)
        optimizer: 优化器

    Returns:
        loss: 当前步的 loss
        logits: 模型输出 logits
    """
    input_ids, labels = batch
    # 前向：得到每个位置上的 next-token logits
    logits = model(input_ids)

    # 预测 t+1 token（因果语言建模）
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)

    # 只保留 next-token 监督关系：预测 t+1 的 token
    loss = F.cross_entropy(shift_logits, shift_labels)

    # 标准训练三连：清梯度 -> 反传 -> 更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, logits


def run_minimal_validation(
    vocab_size: int = 16,
    hidden_size: int = 32,
    intermediate_size: int = 80,
    batch_size: int = 2,
    seq_len: int = 4,
    num_steps: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> None:
    """
    运行最小训练闭环验证。

    Args:
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        intermediate_size: SwiGLU 中间层维度
        batch_size: 批次大小
        seq_len: 序列长度
        num_steps: 训练步数
        lr: 学习率
        device: 张量所在设备
    """
    print("=== 最小训练闭环验证 ===")
    print(f"配置: vocab_size={vocab_size}, hidden_size={hidden_size}, "
          f"intermediate_size={intermediate_size}, batch_size={batch_size}, "
          f"seq_len={seq_len}, num_steps={num_steps}")

    torch.manual_seed(42)

    model = TinyLlamaLM(vocab_size, hidden_size, intermediate_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch = build_toy_batch(batch_size, seq_len, vocab_size)

    losses = []

    for step in range(num_steps):
        loss, logits = train_one_step(model, batch, optimizer)
        losses.append(float(loss))
        if step % 2 == 0 or step == num_steps - 1:
            print(f"  Step {step+1:2d}: loss = {loss.item():.6f}")

    # 验证
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected logits shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    # 验证 loss 在下降
    if losses[0] > losses[-1]:
        print(f"✅ Loss 下降: {losses[0]:.6f} → {losses[-1]:.6f}")
    else:
        print(f"⚠️ Loss 未明显下降: {losses[0]:.6f} → {losses[-1]:.6f}")

    print("✅ 最小数据逻辑与训练循环通过\n")

if __name__ == "__main__":
    # 运行验证
    run_minimal_validation()
```
