# 01. RMSNorm Tutorial | RMSNorm 教程

**难度：** Easy | **环境：** CPU-first | **标签：** `基础架构`, `PyTorch`, `归一化` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将实现大语言模型（如 LLaMA、Gemma）中最常用的归一化技术：**RMSNorm (Root Mean Square Normalization)**。相比于传统的 LayerNorm，它能带来可观的训练加速，同时几乎不损失模型表现。本节聚焦 RMSNorm 的公式、精度实现和可运行代码上，让你快速上手。

**关键词：** `RMSNorm`, `LayerNorm`, `normalization`
## 前置阅读

**导语：** 如果还没把张量基础和归一化直觉理顺，先看下面几页会更顺。

- [P0: 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [P0: 15. Normalization Techniques | 归一化技术](../00_Prerequisites/15_Normalization_Techniques.md)

## 相关阅读

**导语：** 本节先用纯 PyTorch 讲清 RMSNorm 的归一化逻辑与数值稳定性；如果想看同一算子在更高吞吐实现里的做法，再看硬件与融合优化相关页面。

- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

### Step 1: 核心思想与痛点

RMSNorm 的核心洞察很朴素：既然大模型的中间层均值通常已接近0，何不干脆省掉"减去均值"这一步，只用均方根做归一化？

> **为什么抛弃了 LayerNorm？**
> 标准的 LayerNorm 需要计算均值（Mean）和方差（Variance）。
> **RMSNorm 的本质：**
> 假设输入的均值已经接近 0（在大型网络中通常成立），那么我们**直接去掉减去均值的操作**，只用均方根（RMS）去归一化特征。这减少了同步开销，显著提升了前向和反向传播的计算速度。

### Step 2: 核心公式与张量维度

明确了 RMSNorm 的设计取舍后，我们把它的数学公式摆出来。输入输出维度先理清楚，后面代码实现只是在把这组公式翻译成张量操作。

给定输入向量 $x \in \mathbb{R}^d$，RMSNorm 的输出 $y$ 为：

1. **计算均方根 (RMS)：**
   $$ \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon} $$
   其中 $\epsilon$ 是为了防止除以 0 的极小值（如 `1e-6`）。

2. **归一化并缩放 (Scale)：**
   $$ y = \frac{x}{\text{RMS}(x)} \odot \gamma $$
   其中 $\gamma \in \mathbb{R}^d$ 是可学习的权重参数（Weight）。**RMSNorm 没有偏置项 (Bias)**。

### Step 3: 代码实现与混合精度 (AMP) 陷阱

数学公式翻译成代码并不难，真正需要小心的是混合精度训练时的数值稳定性——FP16 下平方运算极易溢出，这里给出标准处理方案。

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

这里开始把前面的公式和精度约束落到最小可运行代码里，重点看每一步为什么存在。

**要求**：请补全下方 `RMSNorm` 的 `forward` 方法。
**注意：**
1. 先在 `float32` 下完成归一化，再乘以可学习的 `weight`，最后转回输入精度。
2. 确保在浮点数精度较高的情况下计算 RMS，以防止半精度（FP16/BF16）溢出。即：强制转换 `x` 为 `float32` 计算 `pow(2).mean()`。


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
        x_fp32 = x if x.dtype == torch.float32 else x.float()
        return x_fp32 * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 3: 先归一化，再缩放并转回输入精度
        # 提示: 调用 _norm 进行归一化后，乘以可学习的 weight，最后转回输入精度
        # ==========================================
        # weight = ???
        return (weight * self._norm(x)).to(x.dtype)

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
        raise
    except (AttributeError, NameError, TypeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到 Parameter")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        else:
            print("代码可能未完成，导致了类型错误")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
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
        # 先定义逐元素缩放参数，形状要和特征维一致。
        # TODO 1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 先把输入提升到 FP32，再做平方均值和倒数平方根。
        # TODO 2
        x_fp32 = x if x.dtype == torch.float32 else x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return x_fp32 * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先按输入精度对齐 weight，再把归一化结果乘回原始 dtype。
        # TODO 3
        weight = self.weight.to(x.dtype)
        return (weight * self._norm(x)).to(x.dtype)
```

### 答案与直觉

**1. TODO 1 (可学习参数)**

- **这一题要解决什么：** 给归一化后的特征加一个逐元素缩放参数，让模型能恢复部分表达能力。
- **为什么这样定义：** `weight` 就是论文里的 $\gamma$，形状要和特征维度一致，初始化为全 1 才不会干扰初始归一化结果。

**2. TODO 2 (核心计算逻辑)**

- **这一题要解决什么：** 在保证数值稳定的前提下，算出 RMSNorm 的归一化分量。
- **为什么先转 FP32：** 输入平方很容易在 FP16 下溢出，所以要先升级精度再算均方值。
- **为什么保留最后一维：** `.mean(dim=-1, keepdim=True)` 是为了让结果能和原输入广播相乘。
- **带走的直觉：** 这里的关键不是公式本身，而是“计算前先升精度”这个工程习惯。

**3. TODO 3 (类型恢复与权重缩放)**

- **这一题要解决什么：** 把归一化结果乘回可学习参数，并恢复到输入时的 dtype。
- **为什么最后再转回原精度：** 前面的归一化已经用 FP32 算完了，最后只需要让输出和输入保持一致即可。
- **带走的直觉：** 先保数值稳定，再做 dtype 对齐，是很多训练算子的通用写法。