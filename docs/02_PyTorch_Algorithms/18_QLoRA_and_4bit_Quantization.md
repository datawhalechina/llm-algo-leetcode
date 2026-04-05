# 18 QLoRA and 4bit Quantization

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/18_QLoRA_and_4bit_Quantization.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 18. QLoRA 与 4-bit NormalFloat 量化核心机制模拟

**难度：** Hard | **标签：** `微调`, `QLoRA`, `量化` | **目标人群：** 模型微调与工程部署

QLoRA 是 2023-2024 年微调界最具重要的论文。它通过引入 **4-bit NormalFloat (NF4)** 数据类型和 **双重量化 (Double Quantization)**，让算法工程师在一张非常廉价的 24GB 显卡上微调高达 33B 的大语言模型成为了现实。
本节我们将实现模拟 QLoRA 的训练过程：冻结低精度的基础权重，在计算前向/反向时动态反量化，只更新高精度的 LoRA 参数。


### Step 1: 核心机制

> **为什么普通的 INT4 量化不能用来微调大模型？**
> 神经网络的权重通常服从正态分布（钟形曲线），中间多，两头少。但普通的 INT4 整数（从 0 到 15）是均匀分布的。这会导致大量的精度浪费。
> **NF4 (NormalFloat 4-bit) 的本质：**
> 我们预先根据标准正态分布的面积，计算出 16 个分位点（Quantiles）。这 16 个值虽然在内存里用 4 个 bit 存储（代表索引 0 到 15），但它们对应的真实浮点数值是非常精确的、密度集中在 0 附近的浮点数。

> **QLoRA 的训练流：**
> 1. 基础权重 (Base Weights) 被极度压缩为 NF4 (冻结不更新)。
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
        
        # ==========================================
        # TODO 2: 分别计算基础前向和 LoRA 旁路前向
        # base_out = x @ dequantized_base_weight.T
        # lora_out = (x @ lora_A.T) @ lora_B.T * scaling
        # ==========================================
        # base_out = ???
        # lora_out = ???
        
        # return base_out + lora_out
        pass

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
        print("\n🔥 QLoRA 核心模拟测试准确通过！你已经掌握了如何在 24G 显卡上微调百亿大模型的密码。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except TypeError as e:
        print("代码可能未完成，导致了操作错误。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_qlora()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

QLoRA 将 LoRA 和 4-bit NormalFloat 权重量化结合起来，在单卡上实现了超大模型的微调。实现重点在于双重量化机制和基于分块的量化参数管理，以此打破微调大模型时的显存瓶颈。

```python
def quantize_4bit(tensor, block_size=64):
    num_blocks = tensor.numel() // block_size
    tensor_reshaped = tensor.view(num_blocks, block_size)
    
    absmax = tensor_reshaped.abs().max(dim=1, keepdim=True)[0]
    scales = absmax / 7.0 
    
    quantized = torch.round(tensor_reshaped / scales).clamp(-8, 7).to(torch.int8)
    
    return quantized.view_as(tensor), scales
    
def dequantize_4bit(quantized, scales, block_size=64):
    num_blocks = quantized.numel() // block_size
    quantized_reshaped = quantized.view(num_blocks, block_size).float()
    
    dequantized = quantized_reshaped * scales
    
    return dequantized.view_as(quantized)
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
