# 08. Architecture Tricks | 架构技巧

**难度：** Easy | **环境：** CPU-first | **标签：** `模型架构`, `架构技巧`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在 `05_LLaMA3_Block_Tutorial` 中我们搭建了 LLaMA 的骨架。但如果你去面试阿里云（通义千问团队）或者谷歌，他们必然会问自家模型与 LLaMA 的区别。
本节我们将以“打补丁”的方式，在 PyTorch 中快速实现 **Qwen 的 Tie Word Embeddings** 以及 **Gemma 的 +1 缩放 RMSNorm**。
可以先把这两个 trick 记成：一个是输入和输出层共享同一份词表参数，另一个是在归一化缩放上加一个更平滑的偏移。

**关键词：** `Qwen`, `Gemma`, `Tie Embeddings`
## 前置阅读

**导语：** 如果还没把 Block、Router 和负载均衡主线理顺，先看下面几页再进入结构变体会更顺。

- [01. RMSNorm Tutorial | RMSNorm 教程](../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.md)
- [05. LLaMA3 Block Tutorial | LLaMA3 Block 教程](../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)
- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)


## 相关阅读

**导语：** 本节先把结构变体讲清楚；如果想继续看训练与微调主线，再顺着看后面的页面。

- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [P1: 08. Programming Models CUDA Triton | 编程模型演进](../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md)
- [P1: 09. AI Compilers and Graph Optimization | AI 编译器与计算图优化](../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.md)

### Step 1: 核心差异与机制

本节对比 Qwen 和 Gemma 在架构设计上的两项关键改动及其设计动机。

> **Trick 1: Tie Word Embeddings (权重绑定) - Qwen 系列 / GPT-2**
> *   **做法**：在绝大多数模型（如 LLaMA）中，最开始的 `Token Embedding` 矩阵（把 ID 变向量）和最后的 `LM Head` 矩阵（把向量变概率）是两个独立的权重矩阵。但在 Qwen 中，**这两个矩阵共享同一份物理内存的参数！**
> *   **意义**：极大减少了参数量（词表动辄 15 万，非常占参数），并且在训练时能让 Embedding 获得更直接的梯度更新。

> **Trick 2: RMSNorm 的 "+1 缩放" - Gemma 系列**
> *   **做法**：标准的 RMSNorm 公式是 $y = \frac{x}{\mathrm{RMS}(x)} \cdot w$，其中 $\mathrm{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$。而 Google 的 Gemma 把它改成了 $y = \frac{x}{\mathrm{RMS}(x)} \cdot (1 + w)$。
> *   **意义**：在 PyTorch 中，权重的默认初始化通常是 0（或者很小的值）。Gemma 加上 1，使得在训练的极早期（缩放参数 $w \approx 0$ 时），RMSNorm 直接等价于一个不做任何缩放的纯归一化层，**这带来了非常平滑的梯度和非常稳定的早期训练！**

**快速对照：**

| 模型 | Embedding / LM Head | Norm | 备注 |
| --- | --- | --- | --- |
| GPT-2 | 共享 `embed_tokens` 和 `lm_head` 的权重 | 标准 LayerNorm / RMSNorm 变体 | 经典的权重绑定 baseline，便于对照 |
| LLaMA3 | 通常不绑定 | RMSNorm | 现代主流参考，结构较简洁 |
| Qwen | 共享 `embed_tokens` 和 `lm_head` 的权重 | RMSNorm | 减少参数量，输入输出监督更一致 |
| Gemma | 通常不绑定 | `1 + w` 的 RMSNorm | 初始阶段更平滑，训练更稳 |

一句话总结：Qwen 通过共享输入/输出权重压缩参数，Gemma 通过 `1 + w` 缩放提升早期训练稳定性。

### Step 2: Weight Tying 与偏置项设计

Weight Tying 让 Embedding 层和输出层（LM Head）共享同一份参数；LM Head 通常不加 bias，是这类实现的常见配置。

- **收益**：共享权重可以直接减少参数量，输入端和输出端也更容易保持表示一致。
- **取舍**：绑定权重会牺牲一部分独立表达自由度，但通常能换来更好的参数效率。
- **工程动机**：在大模型里，这类改动不是“省参数”这么简单，而是围绕训练稳定性、表达能力和内存布局做的取舍。

设共享权重矩阵为 $W \in \mathbb{R}^{V \times d}$，其中 $V$ 为词表大小，$d$ 为隐藏维度。Embedding 查表和 LM Head 投影都围绕同一份词表参数展开。

### Step 3: 代码实现框架

> **实现方式：Weight Tying 的内存级共享**
> * **做法**：在 `__init__` 里让 `lm_head.weight` 直接指向 `embed_tokens.weight`。
> * **代码**：
>   ```python
>   self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
>   self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
>   self.lm_head.weight = self.embed_tokens.weight
>   ```
> * **补充**：这里需要 `bias=False`，绑定后不要再调用 `reset_parameters()`。

Step 2 已经解释了 Weight Tying 的意义，这里只保留代码落点：Embedding 和 LM Head 共享同一份参数。

### Step 4: 动手实战

请补全下面两个模块的实现：
1. `GemmaRMSNorm`：完成 `y = x / RMS * (1 + w)` 的前向传播。
2. `QwenTieEmbeddings`：实现 `Embedding` 与 `LM Head` 的权重共享。


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
        """
        Gemma 风格的 RMSNorm
        公式: output = x / RMS(x) * (1 + weight)
        其中 weight 初始为 0，实现恒等映射
        """
        # 计算均方根（使用 FP32 保证数值稳定性）
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        
        # ==========================================
        # TODO 1: 实现 Gemma 的 +1 缩放
        # 注意类型转换回 x.dtype
        # Gemma 公式: output = normalized * (1 + weight)
        # 注意: weight 初始为 0，确保类型转换回 x.dtype
        # ==========================================
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
        
        # ==========================================
        # TODO 2: 将 lm_head 的权重在内存级别绑定到 embed_tokens 上
        # 提示: 使用赋值让 lm_head.weight 指向 embed_tokens.weight
        # 验证: 可用 .data_ptr() 检查内存地址是否一致
        # ==========================================
        raise NotImplementedError("请先完成 TODO 代码！")
        
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
        
        # =====1. 测试 Gemma RMSNorm=====
        print("\n[1/2] 测试 Gemma RMSNorm...")
        gemma_norm = GemmaRMSNorm(hidden_size)
        x = torch.randn(2, 10, hidden_size)
        out = gemma_norm(x)
        
        # 验证初始化时 (weight=0)，输出等价于无缩放的 norm
        variance = x.float().pow(2).mean(-1, keepdim=True)
        expected = (x.float() * torch.rsqrt(variance + gemma_norm.eps)).to(x.dtype)
        
        # assert torch.allclose(out, expected, atol=1e-4, rtol=1e-3), "Gemma 的 1+w 缩放机制实现错误！"
        # print("✅ Gemma RMSNorm (+1 trick) 测试通过！")
        assert torch.allclose(out, expected, atol=1e-4, rtol=1e-3), \
            "Gemma RMSNorm: weight=0 时输出与预期不符"

        # 补充测试: weight 非零时缩放生效
        with torch.no_grad():
            gemma_norm.weight.data = torch.randn(hidden_size) * 0.1
        out2 = gemma_norm(x)
        assert not torch.allclose(out, out2, atol=1e-4), \
            "Gemma RMSNorm: weight 非零时应产生不同输出"

        # 补充测试: FP16 类型转换
        x_fp16 = x.half()
        out_fp16 = gemma_norm(x_fp16)
        assert out_fp16.dtype == torch.float16, \
            "Gemma RMSNorm: FP16 输入应保持 FP16 输出"
        
        print("✅ Gemma RMSNorm 测试通过！")
        
        # =====2. 测试 Qwen 权重绑定=====
        print("\n[2/2] 测试 Qwen 权重绑定...")
        qwen_model = QwenTieEmbeddings(vocab_size, hidden_size)
        
        # 检查物理内存地址是否相同
        ptr_embed = qwen_model.embed_tokens.weight.data_ptr()
        ptr_head = qwen_model.lm_head.weight.data_ptr()
        assert ptr_embed == ptr_head, "权重未在物理内存级别绑定！"
        
        # 模拟训练更新一次 Embedding
        qwen_model.embed_tokens.weight.data += 1.0
        
        # 验证 LM Head 的权重也跟着变了 (因为它们是同一个指针)
        assert torch.allclose(
            qwen_model.lm_head.weight.data[0, 0],
            qwen_model.embed_tokens.weight.data[0, 0]
        ), "权重更新未同步！"

        # 补充测试: 梯度共享
        qwen_model.embed_tokens.weight.requires_grad_(True)
        dummy_ids = torch.randint(0, vocab_size, (2, 5))
        hidden = qwen_model.forward_embed(dummy_ids)
        loss = qwen_model.forward_lm_head(hidden).sum()
        loss.backward()
        
        assert qwen_model.embed_tokens.weight.grad is not None, \
            "Embedding 应有梯度"
        assert qwen_model.lm_head.weight.grad is not None, \
            "LM Head 应有梯度"

        grad_embed_ptr = qwen_model.embed_tokens.weight.grad.data_ptr()
        grad_head_ptr = qwen_model.lm_head.weight.grad.data_ptr()
        assert grad_embed_ptr == grad_head_ptr, \
            "梯度应共享同一内存"


        print("✅ Qwen Tie Word Embeddings 权重绑定测试通过！")
        print("\n架构变体技巧测试通过。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except AttributeError as e:
        print(f"❌ 属性错误: {e}")
        raise
    except TypeError as e:
        print(f"❌ 类型错误: {e}")
        raise
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise

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
        # weight 初始化为全 0（实现恒等映射）
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

### 答案与直觉

- **核心目标**：把 Gemma 的 RMSNorm 变体和 Qwen 的权重绑定机制落到可运行代码。
- **设计哲学**：这两个技巧看似简单，实则是围绕训练稳定性、参数效率和表达能力做出的工程取舍。
- **核心认知**：架构 Trick 不是"花活"——它们解决的是大模型训练中的实际问题，是经过实践检验的工程选择。

**1. TODO 1: Gemma 的 +1 缩放机制**

- **实现方式**：`output = x_norm * (1 + self.weight)`
- **核心思想**：在标准 RMSNorm 的基础上，将缩放因子从 `w` 改为 `(1 + w)`。
- **初始化优势**：权重初始化为 0 时，`(1 + 0) = 1`，此时 RMSNorm 等价于纯归一化层（无缩放），梯度非常平滑。
- **训练稳定性**：权重初始化为 0 时，`(1 + 0) = 1`，使得 RMSNorm 在训练初期表现为"纯归一化"（无额外缩放）。这避免了因随机初始化导致的输出幅度异常波动，让梯度在早期更稳定。随着训练进行，权重从 0 开始逐步学习，微调缩放因子，保持训练过程的平滑。
- **工程细节**：必须先转换为 FP32 计算（`x.float()`），最后再转回原始精度（`type_as(x)`），防止 FP16/BF16 下的数值不稳定。
- **与标准 RMSNorm 对比**：
  - 标准 RMSNorm：`output = x_norm * weight`
  - Gemma RMSNorm：`output = x_norm * (1 + weight)`
  - 区别：Gemma 在初始化时 `weight=0` 实现恒等映射，而标准 RMSNorm 通常 `weight=1`

**2. TODO 2: Qwen 的权重绑定（Weight Tying）**

- **实现方式**：`self.lm_head.weight = self.embed_tokens.weight`
- **物理指针级共享**：这不是复制权重，而是让两个模块的 `weight` 参数指向同一块内存。修改其中一个，另一个自动同步。
- **参数量优势**：词表通常很大（150k+），绑定后可以节省一半的参数量。
  - 以 Qwen-7B 为例：词表 151,936，隐藏层 4,096
  - 节省参数：151,936 × 4,096 ≈ 6.22 亿参数
  - 节省显存（FP16）：6.22e8 × 2 bytes ≈ 1.24 GB
  - 与原计算（FP32）略有差异，建议根据实际精度说明
- **梯度更新**：由于两个模块共享同一参数，反向传播时：
  - Embedding 层从语言模型损失获得梯度
  - LM Head 也从语言模型损失获得梯度
  - 两者梯度**累加**到同一个 `weight.grad` 上
  - 优化器更新一次，同时影响两个模块
  - 这使得 Embedding 不仅接收来自输入侧的监督，也接收来自输出侧的监督
- **适用场景**：Qwen、GPT-2 等模型都可以使用此技巧。是否绑定取决于模型规模、训练目标和工程约束；有些实现会选择解绑，以保留更强的独立表达自由度。

**工程要点**

- **内存验证**：可以通过 `data_ptr()` 检查两个权重是否指向同一内存地址。
   - assert qwen_model.embed_tokens.weight.data_ptr() == qwen_model.lm_head.weight.data_ptr()
- **训练同步**：由于是物理指针共享，更新 Embedding 权重时，LM Head 权重会自动同步，无需手动处理。
- **架构权衡**：权重绑定减少参数但可能限制表达能力；+1 缩放提升训练稳定性但增加计算量（需要额外的加法）。
