# 01. RMSNorm Tutorial | 均方根层归一化 (RMSNorm)

**难度：** Easy | **标签：** `基础架构`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将实现大语言模型（如 LLaMA、Gemma）中最常用的归一化技术：**RMSNorm (Root Mean Square Normalization)**。相比于传统的 LayerNorm，它能带来可观的训练加速，同时几乎不损失模型表现。

> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [03_Triton_Fused_RMSNorm](../03_CUDA_and_Triton_Kernels/03_Triton_Fused_RMSNorm.md)
### Step 1: 核心思想与痛点

> **为什么抛弃了 LayerNorm？**
> 标准的 LayerNorm 需要计算均值（Mean）和方差（Variance）。
> **RMSNorm 的本质：**
> 假设输入的均值已经接近 0（在大型网络中通常成立），那么我们**直接去掉减去均值的操作**，只用均方根（RMS）去归一化特征。这减少了同步开销，显著提升了前向和反向传播的计算速度。

### Step 2: 核心公式与张量维度

给定输入向量 $x \in \mathbb{R}^d$，RMSNorm 的输出 $y$ 为：

1. **计算均方根 (RMS)：**
   $$ \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon} $$
   其中 $\epsilon$ 是为了防止除以 0 的极小值（如 `1e-6`）。

2. **归一化并缩放 (Scale)：**
   $$ y = \frac{x}{\text{RMS}(x)} \odot \gamma $$
   其中 $\gamma \in \mathbb{R}^d$ 是可学习的权重参数（Weight）。**RMSNorm 没有偏置项 (Bias)**。

### Step 3: 代码实现与混合精度 (AMP) 陷阱

在 PyTorch 中，我们需要通过 `torch.mean` 计算均方，加上一个极小的 `eps` 防止除以零，最后乘以可学习的参数 `weight`。

在代码实现时，有一个非常关键的工程细节需要处理：**数值溢出 (Numerical Overflow)**。

> **工程经验：为什么要强制转换精度？**
> 现代大模型训练与推理几乎都会使用混合精度 (AMP) 或半精度格式 (`FP16`) 以节省显存。但我们需要注意，`FP16` 的最大安全数值仅为 `65504`。
> 
> 在计算 RMSNorm 时，第一步是求输入张量的平方 ($x^2$)。如果输入特征中某个值大于 $256$（由于 $256^2 = 65536 > 65504$），该位置计算后就会溢出变为 `inf`（无穷大），进而导致损失函数出现 `NaN`，引发训练崩溃。
> 
> **标准处理方案 (Upcasting)：** 
> 无论模型输入是什么精度格式，在执行平方和均值操作前，通常需要显式地将其转换为 `float32` 计算。待归一化计算完毕后，再将结果转换回原有精度。这是深度学习框架中处理该算子的标准做法。
### Step 4: 动手实战

**要求**：请补全下方 `RMSNorm` 的 `forward` 方法。
**注意：**
1. 确保在浮点数精度较高的情况下计算 RMS，以防止半精度（FP16/BF16）溢出。即：强制转换 `x` 为 `float32` 计算 `pow(2).mean()`。

```python
import torch
import torch.nn as nn
```


```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # ==========================================
        # TODO 1: 定义可学习参数 weight，并初始化为全 1
        # 形状: [hidden_size]
        # 提示: 使用 nn.Parameter 包装张量使其可学习
        # ==========================================
        # self.weight = ???
        pass
        

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 2: 实现 RMSNorm 核心计算逻辑
        # 提示: 
        # 1. 为防止 FP16 溢出，需要在高精度下计算
        # 2. 计算输入的均方值（平方后求均值），注意保持维度以便广播
        # 3. 使用均方根的倒数进行归一化，torch.rsqrt 比 1/sqrt 更快
        # 4. 返回归一化后的结果（保持高精度，便于后续操作）
        # ==========================================
        # variance = ???
        # return ???
        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 3: 组合归一化与权重缩放
        # 提示: 调用 _norm 进行归一化，乘以可学习的 weight，最后转回输入精度
        # ==========================================
        # output = ???
        # return ???
        pass
```


```python
# 运行此单元格以测试你的实现
def test_rmsnorm():
    try:
        # 构造输入
        hidden_size = 512
        x = torch.randn(2, 16, hidden_size, dtype=torch.float16)  # FP16 输入模拟大模型
        
        # 测试你的实现
        my_norm = RMSNorm(hidden_size)
        # 将模型参数也转换为 FP16，对齐真实的工业半精度运行环境，防止发生隐式的 Type Promotion
        my_norm.to(x.dtype)
        my_out = my_norm(x)
        
        assert my_out.dtype == torch.float16, "输出类型必须与输入一致 (FP16)"
        assert my_out.shape == x.shape, "输出形状改变了！"
        
        # LLaMA 原版实现作为标准答案 (HuggingFace 提取)
        def hf_rmsnorm(hidden_states, weight, eps):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            return weight.to(torch.float32) * hidden_states.to(input_dtype)
            
        hf_out = hf_rmsnorm(x, my_norm.weight, my_norm.eps)
        
        # 检查容差
        assert torch.allclose(my_out.float(), hf_out.float(), rtol=1e-3, atol=1e-4), "计算结果与 HuggingFace 不一致！"
        
        print("\n✅ All Tests Passed! RMSNorm 实现通过测试。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AttributeError:
        print("代码未完成，无法找到 Parameter")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

test_rmsnorm()

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
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO 1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2
        x_fp32 = x if x.dtype == torch.float32 else x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return x_fp32 * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3
        weight = self.weight.to(x.dtype)
        return (weight * self._norm(x)).to(x.dtype)
```

### 解析

**1. TODO 1 (可学习参数)**

- **参数定义：** RMSNorm 的 `weight`（论文中称为 $\gamma$）是逐元素乘以归一化结果的，形状应与特征维度 `hidden_size` 一致，初始化为全 1。

**2. TODO 2 (核心计算逻辑)**

- **防溢出：** 大模型特征的平方和极易越界（超过 FP16 的 `65504` 上限），因此在计算均方值前，必须将输入强制转换为 `float32`。
- **张量广播：** 使用 `.mean(dim=-1, keepdim=True)` 保留维度数量（形状变为 `(batch_size, seq_len, 1)`），以便与 `x_fp32` 正确广播相乘。
- **指令优化：** 使用 `torch.rsqrt(x)`（相当于 $1/\sqrt{x}$）而非 `1.0 / torch.sqrt(x)`，前者直接映射为 CUDA 快速倒数平方根指令，速度更快且数值更稳定。
- **精度保持：** 返回 `float32` 结果，不要急着转换精度。

**3. TODO 3 (类型恢复与权重缩放)**

- **精度一致性：** 必须确保最终输出的精度与输入一致。在经过 FP32 的归一化计算后，将其与 `weight` 相乘，最后统一通过 `.to(x.dtype)` 转换回原生精度（如 `float16`）。
- **进阶思考：** 为什么最后的乘法敢在低精度做（真实场景下 weight 也是低精度），不怕溢出吗？因为 `_norm(x)` 计算完毕后，数值的均方根为 1，绝大多数值落在 [-3, 3] 区间（3σ 原则），乘以 weight（通常接近 1）后仍远低于 FP16 的溢出上限 65504。而 `weight`（初始为 1）通常在 `[0.5, 2.0]` 附近波动。两者的乘积一般在 `[-6, 6]` 之间，距离 FP16 的溢出红线 `65504` 差了一万倍，因此发生溢出的概率极低。