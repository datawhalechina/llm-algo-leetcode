# 26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化

**难度：** Hard | **环境：** GPU required | **标签：** `微调`, `QLoRA`, `量化` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


QLoRA 是 2023-2024 年微调界最具重要的论文。它通过引入 **4-bit NormalFloat (NF4)** 数据类型和 **双重量化 (Double Quantization)**，让算法工程师在一张非常廉价的 24GB 显卡上微调高达 33B 的大语言模型成为了现实。
本节我们将实现模拟 QLoRA 的训练过程：冻结低精度的基础权重，在计算前向/反向时动态反量化，只更新高精度的 LoRA 参数。

**关键词：** `QLoRA`, `NF4`, `LoRA`

## 前置阅读

**导语：** 这一节先把显存、量化和低秩适配的基础概念补齐，再看 QLoRA 会更顺。
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)

## 相关阅读

**导语：** 如果你想继续往 Triton 量化和多租户 LoRA 路由延伸，可以接着看这些页面。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 24. SRAM Optimization Techniques | SRAM 优化技术](../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

### Step 1: 核心机制

> **为什么普通的 INT4 量化不能用来微调大模型？**
> 神经网络的权重通常服从正态分布（钟形曲线），中间多，两头少。但普通的 INT4 整数（从 0 到 15）是均匀分布的。这会导致大量的精度浪费。
> **NF4 (NormalFloat 4-bit) 的本质：**
> 我们预先根据标准正态分布的面积，计算出 16 个分位点（Quantiles）。这 16 个值虽然在内存里用 4 个 bit 存储（代表索引 0 到 15），但它们对应的真实浮点数值是非常精确的、密度集中在 0 附近的浮点数。

> **QLoRA 的训练流：**
> 1. 基础权重 (Base Weights) 被压缩为 NF4 并冻结，不参与更新。
> 2. 前向传播时，先查表把 NF4 索引还原成高精度权重，再交给线性层计算。
> 3. 旁边挂载的 LoRA 矩阵 A 和 B 保持高精度，并且 `requires_grad=True`。
> 4. 反向传播时，梯度主要更新 LoRA；底座权重只负责提供稳定的量化存储。
> **一句话总结：** QLoRA 不是把一切都量化，而是在冻结底座权重的同时保留可训练的 LoRA 旁路，用 NF4 查表把显存压下来，再用高精度适配器把微调能力保住。

### Step 2: 4-bit NormalFloat (NF4) 原理
QLoRA 的核心在于 NF4 数据类型。由于神经网络的权重通常服从均值为 0 的正态分布，NF4 会按照正态分布的累积概率函数，把 16 个量化点更密地放在 0 附近、把更少的点放在尾部。这样它比均匀分布的 INT4 更贴近权重统计特性，也更适合作为冻结底座的存储格式。配合双重分块量化（Double Quantization），还能进一步压缩 scale 本身，把底座模型的显存消耗压榨到极限的 4 bits 每参数。

### Step 3: 代码实现框架
本节我们将模拟一个 16 个元素的 NF4 查表（Lookup Table）。在实际的 QLoRA 层中，权重通常以低精度索引的形式存放，但在前向传播时会先通过查表把它恢复成高精度权重，再交给线性层完成计算。下面的代码会把这条链路拆成两步：先做 NF4 反量化，再把基础分支和 LoRA 旁路合在一起，这样就能把原理和实现一一对上。

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
        # TODO 1: 基础权重反量化（查表还原）
        # ==========================================
        # indices = ???
        # dequantized_base_weight = ???

        # ==========================================
        # TODO 2: 计算基础分支和 LoRA 旁路分支
        # ==========================================
        # base_out = ???
        # lora_out = ???

        return base_out + lora_out

```


```python
# 测试你的实现
def test_qlora():
    try:
        torch.manual_seed(42)

        # 使用一个更小、可精确对照的配置，直接验证 NF4 查表 + LoRA 旁路的公式链路
        batch, seq, in_dim, out_dim, r = 1, 2, 4, 3, 2
        x = torch.tensor([[[0.1, -0.2, 0.3, -0.4], [0.5, 0.6, -0.7, 0.8]]], requires_grad=True)
        layer = QLoRALinearSim(in_features=in_dim, out_features=out_dim, r=r, alpha=8.0)

        with torch.no_grad():
            layer.weight_nf4_indices.copy_(torch.tensor([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ], dtype=torch.int8))
            layer.weight_scale.copy_(torch.tensor(0.5))
            layer.lora_A.copy_(torch.tensor([
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ], dtype=layer.lora_A.dtype))
            layer.lora_B.copy_(torch.tensor([
                [0.9, -0.1],
                [0.2, 0.3],
                [-0.4, 0.7],
            ], dtype=layer.lora_B.dtype))

        out = layer(x)
        assert out.shape == (batch, seq, out_dim), "输出形状不正确！"

        indices_ref = layer.weight_nf4_indices.long()
        dequantized_ref = layer.nf4_table[indices_ref] * layer.weight_scale
        base_out_ref = F.linear(x, dequantized_ref)
        lora_out_ref = (x @ layer.lora_A.T) @ layer.lora_B.T * layer.scaling
        out_ref = base_out_ref + lora_out_ref
        assert torch.allclose(out, out_ref, atol=1e-5), "输出数值不正确！查表反量化或 LoRA 计算有误。"
        assert not torch.allclose(out, base_out_ref, atol=1e-6), "LoRA 旁路应该参与输出，不能退化为纯基础分支！"

        # 2. 验证反向传播时的梯度断点机制 (QLoRA 的灵魂)
        out.sum().backward()
        assert x.grad is not None, "输入 x 没有获得梯度！"
        assert layer.lora_A.grad is not None, "LoRA_A 没有更新梯度！"
        assert layer.lora_B.grad is not None, "LoRA_B 没有更新梯度！"
        assert not layer.weight_nf4_indices.requires_grad, "基础权重的索引不应该有梯度！"
        assert layer.weight_nf4_indices.grad is None, "冻结的基础权重不应该产生梯度！"
        assert layer.weight_scale.grad is None, "冻结的缩放因子不应该产生梯度！"

        print("✅ 查表反量化逻辑正确！")
        print("✅ 梯度流向正确：低精度冻结，高精度更新！")
        print("\n QLoRA 核心模拟测试准确通过！你已经掌握了如何在 24G 显卡上微调百亿大模型的密码。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


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
        # TODO 1: 基础权重反量化（查表还原）
        # 1. 将 weight_nf4_indices 转换为长整型 (long)，以作为查表的索引
        indices = self.weight_nf4_indices.long()
        
        # 2. 从 nf4_table 中取出对应的浮点数值
        # 3. 乘以 weight_scale 恢复范围
        dequantized_base_weight = self.nf4_table[indices] * self.weight_scale
        
        # TODO 2: 分别计算基础分支和 LoRA 旁路分支
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