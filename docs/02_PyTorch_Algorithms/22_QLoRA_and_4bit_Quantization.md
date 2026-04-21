# 22. QLoRA and 4bit Quantization | QLoRA 与 4-bit NormalFloat 量化核心机制 (QLoRA & 4bit)

**难度：** Hard | **标签：** `微调`, `QLoRA`, `量化` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/22_QLoRA_and_4bit_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


QLoRA 是 2023-2024 年微调界最具重要的论文。它通过引入 **4-bit NormalFloat (NF4)** 数据类型和 **双重量化 (Double Quantization)**，让算法工程师在一张非常廉价的 24GB 显卡上微调高达 33B 的大语言模型成为了现实。
本节我们将实现模拟 QLoRA 的训练过程：冻结低精度的基础权重，在计算前向/反向时动态反量化，只更新高精度的 LoRA 参数。

### Step 1: 核心机制

> **为什么普通的 INT4 量化不能用来微调大模型？**
> 神经网络的权重通常服从正态分布（钟形曲线），中间多，两头少。但普通的 INT4 整数（从 0 到 15）是均匀分布的。这会导致大量的精度浪费。
> **NF4 (NormalFloat 4-bit) 的本质：**
> 我们预先根据标准正态分布的面积，计算出 16 个分位点（Quantiles）。这 16 个值虽然在内存里用 4 个 bit 存储（代表索引 0 到 15），但它们对应的真实浮点数值是非常精确的、密度集中在 0 附近的浮点数。

> **QLoRA 的训练流：**
> 1. 基础权重 (Base Weights) 被非常压缩为 NF4 (冻结不更新)。
> 2. 前向传播时，读取 NF4 的索引 -> 查表得到高精度 BF16 值 -> 和输入相乘。
> 3. 旁边挂载的 LoRA 矩阵 A 和 B 是高精度的 BF16/FP32，并且 `requires_grad=True`。
> 4. 反向传播时，梯度从输出流向 LoRA（更新参数），也流向基础权重（但不更新它，仅仅为了传递梯度）。

### Step 2: 4-bit NormalFloat (NF4) 原理
QLoRA 的核心在于 NF4 数据类型。由于神经网络的权重通常服从均值为 0 的正态分布，NF4 根据正态分布的累积概率函数，将信息密度高的地方划分更密集的量化区间。配合双重分块量化（Double Quantization），能够把底座模型的显存消耗压榨到极限的 4 bits 每参数。

### Step 3: 代码实现框架
本节我们将模拟一个 16 个元素的 NF4 查表（Lookup Table）。在实际的 QLoRA 层中，权重的类型是 `torch.uint8`，但在前向传播的那一刻，我们利用这个查表将它瞬间恢复为 FP16，然后用 FP16 与输入特征做矩阵乘法。这个过程虽然比原生的 FP16 慢，但却能在一张消费级显卡上微调几百亿参数的模型。

###  Step 4: 动手实战

**要求**：请补全下方 `QLoRALinearSim` 类。为了不引入复杂的 C++ BitsAndBytes 底层实现，我们将用纯 PyTorch 模拟查表反量化和前向传播。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
def create_nf4_lookup_table() -> torch.Tensor:
    """
    创建 4-bit NormalFloat (NF4) 的查表 (共 16 个离散的浮点值)。
    为了教学，这里提供论文中给出的标准 NF4 分位点数值的近似版本。
    """
    nf4_values = [
        -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
        0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0
    ]
    return torch.tensor(nf4_values)

class QLoRALinearSim(nn.Module):
    """
    模拟 QLoRA 的 Linear 层。
    真实的 QLoRA 会把 weight 存为 uint8，两个 4-bit 挤在一个字节里。
    为了只演示原理，我们这里用 torch.int8 存储 0-15 的索引。
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        
        # 1. 冻结的低精度基础权重 (保存 0~15 的索引)
        self.register_buffer("weight_nf4_indices", torch.randint(0, 16, (out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(1.0)) # 简化的单缩放因子
        self.register_buffer("nf4_table", create_nf4_lookup_table())
        
        # 2. 活跃的高精度 LoRA 适配器
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 1: 基础权重反量化 (Dequantization)
        # 1. 将 weight_nf4_indices 转换为长整型 (long)，以作为查表的索引
        # 2. 从 nf4_table 中取出对应的浮点数值
        # 3. 乘以 weight_scale 恢复范围
        # ==========================================
        # indices = ???
        # dequantized_base_weight = ???
        indices = torch.zeros_like(self.weight_nf4_indices, dtype=torch.long)  # 占位初始化
        dequantized_base_weight = torch.zeros_like(self.weight_nf4_indices, dtype=x.dtype)  # 占位初始化
        
        # ==========================================
        # TODO 2: 分别计算基础前向和 LoRA 旁路前向
        # ==========================================
        # base_out = ???
        # lora_out = ???
        # 占位初始化：使用错误的维度计算确保梯度流通但结果错误
        base_out = F.linear(x, dequantized_base_weight)  # 占位初始化：使用全零权重，结果错误
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * (self.scaling * 0.5)  # 占位初始化： 错误的scaling因子
        
        return base_out + lora_out

```


```python
# 测试你的实现
def test_qlora():
    try:
        torch.manual_seed(42)
        batch, seq, in_dim, out_dim = 2, 8, 64, 128
        x = torch.randn(batch, seq, in_dim, requires_grad=True)
        
        # 初始化 QLoRA 层
        layer = QLoRALinearSim(in_features=in_dim, out_features=out_dim)
        
        # 1. 验证前向传播 (必须能跑通且形状对齐)
        out = layer(x)
        assert out.shape == (batch, seq, out_dim), "输出形状不正确！"
        
        # 🚀 新增：验证 NF4 查表反量化的数值正确性
        # 检查 dequantized_base_weight 是否正确使用了 NF4 查表
        indices_ref = layer.weight_nf4_indices.long()
        dequantized_ref = layer.nf4_table[indices_ref] * layer.weight_scale
        
        # 手动计算参考输出
        base_out_ref = F.linear(x, dequantized_ref)
        lora_out_ref = (x @ layer.lora_A.T) @ layer.lora_B.T * layer.scaling
        out_ref = base_out_ref + lora_out_ref
        
        # 验证输出是否与参考实现接近
        assert torch.allclose(out, out_ref, atol=1e-5), f"输出数值不正确！查表反量化或 LoRA 计算有误。\n期望输出范围: [{out_ref.min():.4f}, {out_ref.max():.4f}]\n实际输出范围: [{out.min():.4f}, {out.max():.4f}]"
        
        # 2. 验证反向传播时的梯度断点机制 (QLoRA 的灵魂)
        out.sum().backward()
        
        # 检查输入 x 是否有梯度 (因为我们要微调底层，必须允许梯度回传)
        assert x.grad is not None, "输入 x 没有获得梯度！"
        
        # 检查 LoRA A 和 B 是否有梯度 (必须有)
        assert layer.lora_A.grad is not None, "LoRA_A 没有更新梯度！"
        assert layer.lora_B.grad is not None, "LoRA_B 没有更新梯度！"
        
        # 检查基础权重不能有梯度 (冻结状态)
        assert not layer.weight_nf4_indices.requires_grad, "基础权重的索引不应该有梯度！"
        
        print("✅ 查表反量化逻辑正确！")
        print("✅ 梯度流向正确：低精度冻结，高精度更新！")
        print("\n QLoRA 核心模拟测试准确通过！你已经掌握了如何在 24G 显卡上微调百亿大模型的密码。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except TypeError as e:
        print("代码可能未完成，导致了操作错误。")
        raise e 
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e

test_qlora()
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
def create_nf4_lookup_table() -> torch.Tensor:
    """
    创建 4-bit NormalFloat (NF4) 的查表 (共 16 个离散的浮点值)。
    """
    nf4_values = [
        -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
        0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0
    ]
    return torch.tensor(nf4_values)

class QLoRALinearSim(nn.Module):
    """
    模拟 QLoRA 的 Linear 层。
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        
        # 1. 冻结的低精度基础权重 (保存 0~15 的索引)
        self.register_buffer("weight_nf4_indices", torch.randint(0, 16, (out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("nf4_table", create_nf4_lookup_table())
        
        # 2. 活跃的高精度 LoRA 适配器
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1: 基础权重反量化 (Dequantization)
        # 1. 将 weight_nf4_indices 转换为长整型 (long)，以作为查表的索引
        indices = self.weight_nf4_indices.long()
        
        # 2. 从 nf4_table 中取出对应的浮点数值
        # 3. 乘以 weight_scale 恢复范围
        dequantized_base_weight = self.nf4_table[indices] * self.weight_scale
        
        # TODO 2: 分别计算基础前向和 LoRA 旁路前向
        base_out = F.linear(x, dequantized_base_weight)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        
        return base_out + lora_out
```

### 解析

**1. TODO 1: 基础权重反量化**
- **实现方式**：`indices = self.weight_nf4_indices.long()`，`dequantized_base_weight = self.nf4_table[indices] * self.weight_scale`
- **关键点**：通过查表将 4-bit 索引（0-15）映射到 NF4 浮点值
- **技术细节**：NF4 查表包含 16 个根据正态分布分位点设计的浮点值，密度集中在 0 附近

**2. TODO 2: 分别计算基础前向和 LoRA 旁路**
- **实现方式**：`base_out = F.linear(x, dequantized_base_weight)`，`lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling`
- **关键点**：基础权重冻结（不更新梯度），LoRA 权重可训练
- **技术细节**：LoRA 输出需要乘以 scaling 因子（alpha / r）来平衡贡献

**NF4 量化原理**
- **标准量化问题**：INT4 均匀分布，但神经网络权重服从正态分布，导致精度浪费
- **NF4 解决方案**：根据标准正态分布的累积分布函数（CDF）计算 16 个分位点
- **信息密度**：在 0 附近分配更多的量化点，在尾部分配更少的点
- **查表机制**：4-bit 索引 → NF4 浮点值 → 乘以 scale 恢复原始范围

**工程优化要点**
- **显存节省**：基础权重从 FP16（2 bytes）降至 NF4（0.5 bytes），节省 75% 显存
- **双重量化**：对 scale 参数本身也进行量化，进一步节省显存
- **分块量化**：每 64 或 128 个参数共享一个 scale，平衡精度和显存
- **梯度流向**：基础权重冻结，梯度只更新 LoRA 参数，避免量化误差累积
- **训练效率**：虽然反量化增加计算开销，但显存节省允许更大的 batch size
- **工业实践**：QLoRA 使 33B 模型可在单张 24GB 显卡上微调，65B 模型可在单张 48GB 显卡上微调