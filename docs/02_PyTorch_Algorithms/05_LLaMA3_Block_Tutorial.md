# 05 LLaMA3 Block Tutorial

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 05. 经典模型搭建: LLaMA-3 Transformer Block

**难度：** Medium | **标签：** `模型架构`, `PyTorch` | **目标人群：** 模型微调与工程部署

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
        # gate_proj: hidden_size -> intermediate_size
        # up_proj: hidden_size -> intermediate_size
        # down_proj: intermediate_size -> hidden_size
        # ==========================================
        # self.gate_proj = ???
        # self.up_proj = ???
        # self.down_proj = ???
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 2: 实现 SwiGLU 的前向传播
        # 提示: down_proj( F.silu(gate_proj(x)) * up_proj(x) )
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
        # 步骤:
        # 1. 对 hidden_states 存一个残差备份 (residual = hidden_states)
        # 2. RMSNorm 归一化 -> 丢给 Attention 计算
        # 3. 将 Attention 结果加上 residual，得到中间隐藏层 h
        # 4. 对 h 存一个残差备份 (residual = h)
        # 5. RMSNorm 归一化 -> 丢给 MLP 计算
        # 6. 将 MLP 结果加上 residual，得到最终输出
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
            
        print("\n✅ All Tests Passed! 恭喜你，LLaMA-3 Transformer Block 组装成功！你已经掌握了当今大模型的核心架构！")
        
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

::: details 💡 点击查看官方解析与参考代码

LLaMA 3的架构创新之一在于引入了更高效的Transformer模块结构。它通常采用RMSNorm、SwiGLU和GQA等组件。参考代码展示了如何利用这些组件拼接出完整的LLaMA 3 Decoder层，并保证残差连接和归一化处理。 

```python
class LLaMA3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLUMLP(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, freqs_cis=None, mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, freqs_cis, mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
