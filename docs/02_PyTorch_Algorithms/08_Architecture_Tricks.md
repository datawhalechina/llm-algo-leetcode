# 08. Architecture Tricks | 经典架构变体：Qwen 与 Gemma 的核心机制 (Architecture Tricks)

**难度：** Easy | **标签：** `模型架构`, `Qwen`, `Gemma` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在 `06_LLaMA3_Block_Tutorial` 中我们搭建了 LLaMA 的骨架。但如果你去面试阿里云（通义千问团队）或者谷歌，他们必然会问自家模型与 LLaMA 的区别。
本节我们将以“打补丁”的方式，在 PyTorch 中快速实现 **Qwen 的 Tie Word Embeddings** 以及 **Gemma 的带偏置 RMSNorm**。

### Step 1: 核心差异与机制

> **Trick 1: Tie Word Embeddings (权重绑定) - Qwen 系列 / GPT-2**
> *   **做法**：在绝大多数模型（如 LLaMA）中，最开始的 `Token Embedding` 矩阵（把 ID 变向量）和最后的 `LM Head` 矩阵（把向量变概率）是两个独立的权重矩阵。但在 Qwen 中，**这两个矩阵共享同一份物理内存的参数！**
> *   **意义**：极大减少了参数量（词表动辄 15 万，非常占参数），并且在训练时能让 Embedding 获得更直接的梯度更新。

> **Trick 2: RMSNorm 的 "+1 缩放" - Gemma 系列**
> *   **做法**：标准的 RMSNorm 公式是 $y = \frac{x}{RMS} \cdot w$。而 Google 的 Gemma 把它改成了 $y = \frac{x}{RMS} \cdot (1 + w)$。
> *   **意义**：在 PyTorch 中，权重的默认初始化通常是 0（或者很小的值）。Gemma 加上 1，使得在训练的极早期（$w pprox 0$ 时），RMSNorm 直接等价于一个不做任何缩放的纯归一化层，**这带来了非常平滑的梯度和非常稳定的早期训练！**

### Step 2: Weight Tying 与偏置项的权衡
Weight Tying（权重绑定）强制 Embedding 层和最终的 LM Head 线性层共享同一个权重矩阵。这种方法在早期的模型中很流行，因为它大幅减少了参数量。但在现代极大规模 LLM 中，解绑通常能获得更好的容量表达。此外，取消大部分 Linear 和 Norm 层中的 Bias 项，可以略微提高计算效率并防止显存浪费。

### Step 3: 代码实现框架
要实现权重绑定，只需在网络初始化时将 LM Head 的 `weight` 引用直接指向 Embedding 层的 `weight`。注意，这意味着隐藏层维度必须与词表维度兼容（或者存在中间投影层）。

###  Step 4: 动手实战

**要求**：
1. 补全 `GemmaRMSNorm` 的公式。
2. 补全 `QwenTieEmbeddings` 中的参数共享逻辑。


```python
import torch
import torch.nn as nn
```


```python
# --- Trick 1: Gemma 风格的 RMSNorm ---
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # weight 初始化为全 0
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        
        # ==========================================
        # TODO 1: 实现 Gemma 的 +1 缩放
        # 注意类型转换回 x.dtype
        # ==========================================
        # output = ???

        # 占位初始化（返回错误值，确保数值测试失败）                                                                                                                                 
        output = torch.zeros_like(x)                                                                                                                                                 
  
        return output     
        



# --- Trick 2: Qwen 风格的权重绑定 ---
class QwenTieEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # 1. 定义标准的 Embedding 层
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # 2. 定义最后的 LM Head 预测层，注意不要 bias
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # ==========================================
        # TODO 2: 将 lm_head 的权重在内存级别绑定到 embed_tokens 上
        # 提示: 在 PyTorch 中，可以直接赋值 nn.Parameter 或是底层 tensor
        # self.lm_head.weight = ???
        # ==========================================
        # ???
        
    def forward_embed(self, input_ids):
        return self.embed_tokens(input_ids)
        
    def forward_lm_head(self, hidden_states):
        return self.lm_head(hidden_states)

```


```python
# 测试你的实现
def test_tricks():
    try:
        hidden_size = 64
        vocab_size = 1000
        
        # 1. 测试 Gemma RMSNorm
        gemma_norm = GemmaRMSNorm(hidden_size)
        x = torch.randn(2, 10, hidden_size)
        out = gemma_norm(x)
        
        # 验证初始化时 (weight=0)，输出等价于无缩放的 norm
        variance = x.float().pow(2).mean(-1, keepdim=True)
        expected = (x.float() * torch.rsqrt(variance + 1e-6)).to(x.dtype)
        
        assert torch.allclose(out, expected, atol=1e-4), "Gemma 的 1+w 缩放机制实现错误！"
        print("✅ Gemma RMSNorm (+1 trick) 测试通过！")
        
        # 2. 测试 Qwen 权重绑定
        qwen_model = QwenTieEmbeddings(vocab_size, hidden_size)
        
        # 检查物理内存地址是否相同
        ptr_embed = qwen_model.embed_tokens.weight.data_ptr()
        ptr_head = qwen_model.lm_head.weight.data_ptr()
        assert ptr_embed == ptr_head, "权重未在物理内存级别绑定！"
        
        # 模拟训练更新一次 Embedding
        qwen_model.embed_tokens.weight.data += 1.0
        
        # 验证 LM Head 的权重也跟着变了 (因为它们是同一个指针)
        assert qwen_model.lm_head.weight.data[0, 0] == qwen_model.embed_tokens.weight.data[0, 0], "权重更新未同步！"
        
        print("✅ Qwen Tie Word Embeddings 权重绑定测试通过！")
        print("\n架构变体技巧测试通过。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except AttributeError:
        print("代码未完成导致变量属性错误。")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise e

test_tricks()


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
# --- Trick 1: Gemma 风格的 RMSNorm ---
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # weight 初始化为全 0
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        
        # TODO 1: 实现 Gemma 的 +1 缩放
        output = x_norm * (1 + self.weight)
        
        return output.type_as(x)


# --- Trick 2: Qwen 风格的权重绑定 ---
class QwenTieEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # 1. 定义标准的 Embedding 层
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # 2. 定义最后的 LM Head 预测层，注意不要 bias
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # TODO 2: 将 lm_head 的权重在内存级别绑定到 embed_tokens 上
        # 物理指针级共享：直接让 lm_head.weight 指向 embed_tokens.weight
        self.lm_head.weight = self.embed_tokens.weight
        
    def forward_embed(self, input_ids):
        return self.embed_tokens(input_ids)
        
    def forward_lm_head(self, hidden_states):
        return self.lm_head(hidden_states)

```

### 解析

**1. TODO 1: Gemma 的 +1 缩放机制**

- **实现方式**：`output = x_norm * (1 + self.weight)`
- **核心思想**：在标准 RMSNorm 的基础上，将缩放因子从 `w` 改为 `(1 + w)`。
- **初始化优势**：权重初始化为 0 时，`(1 + 0) = 1`，此时 RMSNorm 等价于纯归一化层（无缩放），梯度非常平滑。
- **训练稳定性**：在训练早期（权重接近 0），避免了因权重过小导致的梯度消失问题。随着训练进行，权重逐渐学习到合适的缩放值。
- **工程细节**：必须先转换为 FP32 计算（`x.float()`），最后再转回原始精度（`type_as(x)`），防止 FP16/BF16 下的数值不稳定。

**2. TODO 2: Qwen 的权重绑定（Weight Tying）**

- **实现方式**：`self.lm_head.weight = self.embed_tokens.weight`
- **物理指针级共享**：这不是复制权重，而是让两个模块的 `weight` 参数指向同一块内存。修改其中一个，另一个自动同步。
- **参数量优势**：词表通常很大（15万+），绑定后可以节省一半的参数量。例如，词表 150k、隐藏层 4096 的模型，可以节省 150k × 4096 × 4 bytes ≈ 2.4GB 显存。
- **梯度更新**：训练时，Embedding 层和 LM Head 的梯度会累加到同一个权重上，使得 Embedding 获得更直接的监督信号。
- **适用场景**：Qwen、GPT-2 等模型使用此技巧。但在超大规模模型（如 LLaMA 70B）中，解绑通常能获得更好的表达能力。

**工程要点**

- **内存验证**：可以通过 `data_ptr()` 检查两个权重是否指向同一内存地址。
- **训练同步**：由于是物理指针共享，更新 Embedding 权重时，LM Head 权重会自动同步，无需手动处理。
- **架构权衡**：权重绑定减少参数但可能限制表达能力；+1 缩放提升训练稳定性但增加计算量（需要额外的加法）。
