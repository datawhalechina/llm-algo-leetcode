# 01. RMSNorm Tutorial | RMSNorm 教程

**难度：** Easy | **环境：** CPU-first | **标签：** `基础实现`, `归一化`, `RMSNorm` | **目标人群：** 基础实现学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

Transformer 层数一深，中间激活的尺度就会不断变化。如果不做归一化，后面的矩阵乘法和激活函数很容易面对不稳定的输入；但标准 LayerNorm 又要同时计算均值和方差，在大模型里会带来额外开销。

RMSNorm 选择更直接的做法：不再减均值，只用均方根控制特征尺度。本节会实现一个最小 RMSNorm，重点看清归一化维度、`eps` 的数值稳定作用，以及为什么许多 LLaMA 系模型会采用这种更轻的归一化层。完成后，你可以把它接到后面的 MLP、注意力和完整 Transformer 组件里。

**关键词：** `RMSNorm`, `LayerNorm`, `normalization`

---
## 前置阅读

**导语：** 先确认输入张量的归一化维度和训练接口，再比较 LayerNorm 与 RMSNorm 如何处理特征尺度。

- [P0: 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md)
- [P0: 13. Transformer Block Dataflow and Residual Stream | Transformer Block 数据流与残差通路](../00_Prerequisites/13_Transformer_Block_Dataflow_and_Residual_Stream.md)
- [P0: 15. Normalization Techniques | 归一化技术](../00_Prerequisites/15_Normalization_Techniques.md)

---
### Step 1：从 LayerNorm 到 RMSNorm 的结构选择

归一化要解决的是残差流在多层变换后尺度漂移的问题。LayerNorm 同时执行重新居中与尺度归一；RMSNorm 只保留尺度归一。现代解码器通常把归一化放在子层之前，让残差主路径可以跨层直接传递，再用 RMSNorm 为注意力或 MLP 提供尺度受控的输入。

从几何上看，LayerNorm 的减均值会移除向量在全 1 方向上的分量，随后再按方差缩放；RMSNorm 不做这个投影，只按向量的均方根缩放。它因此不是 LayerNorm 的数值近似，而是主动取消“重新居中”这一约束。是否适用仍由模型结构、初始化和训练结果共同决定，不能只凭算子更短就推导质量等价。

| 结构 | 统计动作 | 在 Block 中的位置 | 主要取舍 |
|---|---|---|---|
| Post-LayerNorm | 居中并缩放 | 子层与残差合并之后 | 深层训练更依赖稳定技巧 |
| Pre-LayerNorm | 居中并缩放 | 子层输入之前 | 残差主干更直接，但仍计算均值与方差 |
| Pre-RMSNorm | 只按均方根缩放 | 子层输入之前 | 统计路径更短，不保证输出零均值 |

![RMSNorm 在 Block 里的位置](../public/02_PyTorch_Algorithms/01_rmsnorm_diagram.svg)

### Step 2：公式、归一化轴与统计路径

对每个 token 的隐藏向量 $x\in\mathbb{R}^d$，RMSNorm 沿最后一维计算

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\qquad
y=\frac{x}{\mathrm{RMS}(x)}\odot\gamma.
$$

批次维和序列维保持不变，只有每个 token 内部的通道共同参与统计。可学习参数 $\gamma\in\mathbb{R}^d$ 恢复逐通道缩放能力， $\epsilon$ 防止零向量或极小输入导致除零。与 LayerNorm 对照时，应同时检查统计轴、是否减均值、可学习参数和输出数据类型，不能只比较公式中少了一项运算。

| 检查对象 | RMSNorm 契约 | 常见错误 |
|---|---|---|
| 统计轴 | 隐藏状态最后一维 | 跨批次或序列混合统计 |
| 中心化 | 不减均值 | 误写成 LayerNorm |
| 可学习状态 | 一份逐通道缩放参数 | 使用标量导致通道能力丢失 |
| 形状 | 输入输出完全一致 | 统计轴被消除后无法正确广播 |

### Step 3：混合精度下的数值路径

RMSNorm 的平方、均值和倒平方根属于数值敏感的归约路径。FP16/BF16 输入可以降低存储和矩阵计算成本，但较大激活在平方时可能溢出，较小激活也可能损失统计精度。将归约统计提升到 FP32，再把结果恢复为模型主状态的数据类型，可以在接口不变的条件下保护统计精度。单层数值一致性只能验证算子契约；深层稳定性还需要在固定模型、数据和训练预算下比较。

| 阶段 | 建议计算类型 | 原因 |
|---|---|---|
| 输入与模型主状态 | 模型 dtype | 保持模型接口与存储口径 |
| 平方、均值、`rsqrt` | FP32 | 降低溢出和累计误差 |
| 归一化与通道缩放 | FP32 或实现规定的安全路径 | 避免过早舍入 |
| 输出 | 恢复输入数据类型 | 保持后续子层契约 |

![RMSNorm 混合精度统计路径](../public/02_PyTorch_Algorithms/01_rmsnorm_statistics.svg)

### Step 4：代码设计与 RMSNorm 契约

题目区以一个最小模块承载三个机制责任：逐特征缩放参数、沿最后一维的 RMS 统计，以及 dtype 对齐。固定骨架处理模块接口；学习者只补全决定数值路径的变量。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|---|---|---|---|
| TODO 1：创建逐特征缩放参数<br>`RMSNorm.__init__` | 创建可学习的逐特征缩放参数 | 形状为 `[D]`、初值为 1、参与反向传播 | 参数形状、初值和梯度存在性正确 |
| TODO 2：计算 RMS 归一化<br>`RMSNorm._norm` | 沿最后一维在 FP32 中计算均方根并归一化 | 保留统计轴，零输入结果有限 | 与显式 FP32 参考结果一致 |
| TODO 3：恢复输出契约<br>`RMSNorm.forward` | 应用逐特征缩放并恢复输入数据类型 | 输出形状和数据类型与输入一致 | FP16 路径的数值、形状和梯度正确 |

```python
import torch
import torch.nn as nn
```


```python
class RMSNorm(nn.Module):
    """沿最后一维归一化，并保留逐特征缩放参数。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError('hidden_size 必须为正数')
        self.eps = eps
        # ==========================================
        # TODO 1：创建逐特征缩放参数
        # ==========================================
        # self.weight = ???  # [D]；初值全为 1，并参与梯度更新。
        raise NotImplementedError('TODO 1：请创建 RMSNorm 的缩放参数')

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """在 FP32 路径中计算 RMS 归一化，返回 FP32 结果。"""
        if x.ndim < 1:
            raise ValueError('x 至少需要包含一个特征维')
        # ==========================================
        # TODO 2：计算 RMS 归一化
        # ==========================================
        # x_fp32 = ???  # 与 x 同形；统计路径使用 FP32。
        # mean_square = ???  # [..., 1]；沿最后一维统计并保留该轴。
        # normalized = ???  # 与 x 同形；零输入必须保持有限。
        raise NotImplementedError('TODO 2：请完成 RMS 统计与归一化')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """缩放归一化结果，并恢复输入 dtype。"""
        # ==========================================
        # TODO 3：恢复输出契约
        # ==========================================
        # weight = ???  # [D]；与输入数据类型对齐。
        # normalized = ???  # FP32 归一化结果。
        # output = ???  # 与 x 的形状和数据类型一致。
        raise NotImplementedError('TODO 3：请完成缩放与 dtype 对齐')

```


```python
# 测试设计：分别定位参数契约、RMS 数值路径和 dtype/梯度对齐。

def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """使用显式 FP32 统计构造参考结果。"""
    x_fp32 = x.float()
    mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(mean_square + eps) * weight.float()).to(x.dtype)


def test_rmsnorm_parameter_contract():
    """验证缩放参数的类型、形状和单位初值。"""
    norm = RMSNorm(hidden_size=4)
    assert isinstance(norm.weight, nn.Parameter)
    assert norm.weight.shape == (4,)
    assert torch.equal(norm.weight.detach(), torch.ones(4))


def test_rmsnorm_numeric_contract():
    """验证普通输入与零输入的 RMS 数值路径。"""
    x = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    norm = RMSNorm(hidden_size=2, eps=1e-6)
    output = norm(x)
    expected = _rmsnorm_reference(x, norm.weight, norm.eps)
    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.isfinite(output).all()


def test_rmsnorm_dtype_and_gradient_contract():
    """验证低精度输入的输出契约和两类梯度。"""
    x = torch.randn(2, 3, 4, dtype=torch.float16, requires_grad=True)
    norm = RMSNorm(hidden_size=4).to(dtype=torch.float16)
    output = norm(x)
    expected = _rmsnorm_reference(x, norm.weight, norm.eps)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.allclose(output.float(), expected.float(), rtol=1e-3, atol=1e-4)
    output.float().sum().backward()
    assert norm.weight.grad is not None
    assert x.grad is not None


def run_rmsnorm_tests():
    """依次运行参数、数值和数据类型/梯度契约测试。"""
    test_rmsnorm_parameter_contract()
    test_rmsnorm_numeric_contract()
    test_rmsnorm_dtype_and_gradient_contract()
    print('✅ RMSNorm：参数、数值路径与 dtype/梯度契约测试通过')


run_rmsnorm_tests()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难，可以继续向下查看参考代码。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码


```python
class RMSNorm(nn.Module):
    """沿最后一维归一化，并保留逐特征缩放参数。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError('hidden_size 必须为正数')
        self.eps = eps
        # TODO 1：创建逐特征缩放参数
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """在 FP32 路径中计算 RMS 归一化，返回 FP32 结果。"""
        if x.ndim < 1:
            raise ValueError('x 至少需要包含一个特征维')
        # TODO 2：计算 RMS 归一化
        x_fp32 = x.float()
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(mean_square + self.eps)
        return normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """缩放归一化结果，并恢复输入 dtype。"""
        # TODO 3：恢复输出契约
        weight = self.weight.to(dtype=x.dtype)
        normalized = self._norm(x)
        output = (normalized * weight).to(dtype=x.dtype)
        return output

```

### 解析

#### TODO 1：创建逐特征缩放参数

- 每个特征通道需要独立恢复缩放能力，因此参数形状必须是 `[D]`，单位初值保持初始归一化尺度。
- 答案将全 1 张量注册为可学习参数；参数契约测试检查类型、形状和初值，梯度测试确认它参与反向传播。

#### TODO 2：计算 RMS 归一化

- 统计只沿最后一维进行，并保留单例轴用于广播；FP32 归约降低低精度平方和累计的数值风险。
- 答案用均方值与 $\epsilon$ 形成倒均方根；数值测试与显式 FP32 参考结果比较，并检查零输入保持有限。

#### TODO 3：恢复输出契约

- 逐特征参数与归一化结果相乘后，输出必须恢复输入的数据类型和形状。
- 答案显式形成输出并转换数据类型；测试同时验证数值、形状、数据类型以及输入和参数梯度。
## 相关阅读

想把本节的公式与真实模型实现对照时，可以先读原论文，再查看开源 Decoder 实现中的归一化位置。

- [RMSNorm 原论文：Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Transformers 中的 LLaMA 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [02. SwiGLU 激活](../02_PyTorch_Algorithms/02_SwiGLU_Activation.md)
- [04. 多头注意力与 GQA](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
- [P1: 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
