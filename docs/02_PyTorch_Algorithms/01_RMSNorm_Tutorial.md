# 01. RMSNorm Tutorial | 均方根层归一化 (RMSNorm)

**难度：** Easy | **标签：** `基础架构`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将实现大语言模型（如 LLaMA、Gemma）中最常用的归一化技术：**RMSNorm (Root Mean Square Normalization)**。相比于传统的 LayerNorm，它能带来可观的训练加速，同时几乎不损失模型表现。

> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_CUDA_and_Triton_Kernels/03_Triton_Fused_RMSNorm.ipynb`](../03_CUDA_and_Triton_Kernels/03_Triton_Fused_RMSNorm.md)

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
### Step 3: 代码实现框架
在 PyTorch 中，我们需要通过 `torch.mean` 计算方差，加上一个极小的 `eps` 防止除以零，最后乘以可学习的权重参数 `weight`。注意在计算方差时，应当保持数据类型为 `float32` 以防止 FP16 下的数值溢出。
### Step 4: 动手实战

**要求**：请补全下方 `RMSNorm` 的 `forward` 方法。
**注意：**
1. 确保在浮点数精度较高的情况下计算 RMS，以防止半精度（FP16/BF16）溢出。即：强制转换 `x` 为 `float32` 计算 `pow(2).mean()`。

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # ==========================================
        # TODO 1: 定义可学习参数 weight，并初始化为全 1
        # 形状: [hidden_size]
        # ==========================================
        # self.weight = nn.Parameter(???)
        pass

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 2: 实现 RMSNorm 核心计算逻辑
        # 1. 强制转 float32 以防溢出 (x.to(torch.float32))
        # 2. 计算均方值: x^2 的均值，注意 keepdim=True 保持广播形状
        # 3. x_norm = x * (均方 + eps)^(-1/2)  # 使用 torch.rsqrt 更快
        # ==========================================
        # variance = ???
        # return ???
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 3: 调用 _norm，乘以 weight，并转换回原精度
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
        my_out = my_norm(x)
        
        assert my_out.dtype == torch.float16, "输出类型必须与输入一致 (FP16)"
        assert my_out.shape == x.shape, "输出形状改变了！"
        
        # LLaMA 原版实现作为标准答案 (HuggingFace 提取)
        def hf_rmsnorm(hidden_states, weight, eps):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            return weight * hidden_states.to(input_dtype)
            
        hf_out = hf_rmsnorm(x, my_norm.weight, my_norm.eps)
        
        # 检查容差
        assert torch.allclose(my_out.float(), hf_out.float(), atol=1e-4), "计算结果与 HuggingFace 不一致！"
        
        print("\n✅ All Tests Passed! 恭喜你，工业级防溢出 RMSNorm 实现成功！")
        
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

## 官方解析与参考代码

**解析：**
1. **TODO 1 (可学习参数)：** RMSNorm 的 `weight` (通常论文中称为 $\gamma$) 是逐元素乘以归一化结果的，所以它的形状应该和特征维度 `hidden_size` 一致，并且初始化为全 1。
2. **TODO 2 (核心计算)：** 这是工业级实现的最核心部分，主要包含三个关键技巧：
   - **防溢出 (Upcasting)**：大模型特征的平方和极易越界（超过 FP16 的 `65504` 上限），因此在计算均方值前，必须将输入强制转换为 `float32`。
   - **张量广播 (Broadcasting)**：计算均方值时使用 `.mean(dim=-1, keepdim=True)`，不仅是对最后一个特征维度求均值，更是为了保留维度数量（形状变为 `(batch_size, seq_len, 1)`），以便在最后一步能与 `x_fp32` 完美广播相乘。
   - **指令优化 (Fast Math)**：强烈推荐使用 `torch.rsqrt(x)`（相当于 $1/\sqrt{x}$）而不是 `1.0 / torch.sqrt(x)`。前者在底层会直接映射为专门的 CUDA 快速倒数平方根指令，速度更快且数值更稳定。最后返回 `float32` 结果，在此处不要急着转换精度。
3. **TODO 3 (类型恢复与权重缩放)：**
    **正确顺序**：必须先将 float32 的 norm 结果转回原生精度（如 `float16`），再乘以 `weight`。即：`x_norm.to(x.dtype) * self.weight`。
    **防坑指南 (Type Promotion)**：在真实大模型中，`weight` 会被转化为与输入相同的低精度格式。如果写成 `(x_norm_fp32 * weight_fp16).to(x.dtype)`，PyTorch 会在底层将 `weight` 隐式提升为 FP32 做乘法，带来额外的转换开销。
    **进阶思考：为什么最后的乘法敢在低精度做，不怕溢出吗？**
    因为 `_norm(x)` 计算完毕后，特征已被归一化，其绝大多数数值被强行压缩到了 `[-3, 3]` 的极小区间内。而 `weight`（初始为 1）通常在 `[0.5, 2.0]` 附近波动。两者的乘积一般在 `[-6, 6]` 之间，距离 FP16 的溢出红线 `65504` 差了一万倍，因此发生溢出的概率极低。


```python
class RMSNormSolution(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO 1: 定义可学习参数 weight
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2: 实现 RMSNorm 核心计算逻辑
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return x_fp32 * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3: 类型回退并应用权重缩放
        # 正确做法：先将 norm 结果转回 x.dtype，再与 weight 相乘
        return self._norm(x).to(x.dtype) * self.weight

```
