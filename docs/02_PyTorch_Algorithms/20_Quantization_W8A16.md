# 20 Quantization W8A16

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/20_Quantization_W8A16.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 20. 模型量化基础: INT8 绝对最大值量化与反量化 (Quantization)

**难度：** Medium | **标签：** `推理优化`, `量化`, `PTQ` | **目标人群：** 模型微调与工程部署


大模型动辄百亿参数，显存占用极大。为了让 7B 模型能跑在消费级显卡（如 RTX 4090 / 3090）甚至手机端，**模型量化 (Quantization)** 是不可或缺的技术。
本节我们将实现最基础、也是面试中最常考的 **INT8 对称量化 (Symmetric Quantization)**，理解量化缩放因子 (Scale) 的计算，以及如何在进行矩阵乘法前进行反量化 (Dequantization)。


> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_Triton_Kernels/10_Triton_Quantization_Support.ipynb`](../03_Triton_Kernels/10_Triton_Quantization_Support.ipynb)


### Step 1: 核心思想与概念

> **什么是量化？**
> 将高精度（如 FP32/FP16，占用 4/2 个字节）的浮点数，映射到低精度（如 INT8，占用 1 个字节）的整数上。这样不仅显存占用直接缩小 2-4 倍，还能利用硬件的整数计算单元（如 INT8 Tensor Core）加速计算。

> **PTQ 与 QAT 的区别：**
> - **PTQ (Post-Training Quantization，训练后量化)**：模型已经训练好了。我们只需要拿一小批校准数据（Calibration Data）跑一遍，统计一下激活值的分布，算出缩放因子（Scale），直接对权重转换。本节我们实现的就是 PTQ。
> - **QAT (Quantization-Aware Training，量化感知训练)**：在训练时，正向传播模拟量化的误差，反向传播用“直通估计器 (STE)”更新原始的高精度权重。成本极高，但精度损失最小。


### Step 2: 代码实现框架
我们需要实现 `quantize` 和 `dequantize` 两个函数。在量化时，先计算 Scale，然后将张量除以 Scale，接着进行 `torch.round`，最后通过 `torch.clamp` 限制在 $[-128, 127]$ 区间并转为 `torch.int8` 数据类型。反量化则是简单的乘以 Scale。


###  Step 3: 数学公式：绝对最大值量化

这是对称量化最常用的方法。假设我们有一个浮点张量 $X$，我们要把它映射到 INT8 的范围 $[-127, 127]$ 内。

1. **计算绝对最大值 (Absmax)**：
   找到张量中绝对值最大的元素：$m = \max(|X|)$。

2. **计算缩放因子 (Scale)**：
   S = 
\frac{127}{m}$。这个 $S$ 就代表了“1个单位的 INT8 等于多少个单位的 FP16”。

3. **量化 (Quantize)**：
   将张量乘以缩放因子，然后四舍五入 (Round) 变成整数，并截断 (Clamp) 到 INT8 范围内，防止异常值越界：
   $X_{int8} = 	ext{Clamp}(	ext{Round}(X \times S), -128, 127)$

4. **反量化 (Dequantize)**：
   在真正做矩阵乘法前（如果是 W8A16 这种 Weight-only 量化），需要把 INT8 恢复成 FP16 参与计算：
   $X_{fp16} = \frac{X_{int8}}{S}$


###  Step 4: 动手实战

**要求**：
1. 补全 `absmax_quantize` 函数，实现权重的 INT8 转换并返回 `scale`。
2. 补全 `W8A16Linear` 的 `forward` 方法。W8A16 意味着权重 (We\r\right) 是 INT8，但激活值 (Activation/Input) 保持 FP16。计算时需要实时反量化。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def absmax_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将浮点张量 X 量化为 INT8，并返回缩放因子。
    
    Args:
        x: 浮点类型的张量
    Returns:
        x_quant: dtype 为 torch.int8 的量化张量
        scale: float 类型的缩放因子
    """
    # ==========================================
    # TODO 1: 计算张量的绝对最大值 absmax
    # 提示: 使用 torch.abs 和 torch.max
    # ==========================================
    # absmax = ???
    
    # 避免除以 0 的情况
    # if absmax == 0:
    #     absmax = 1e-8
        
    # ==========================================
    # TODO 2: 计算缩放因子 scale (映射到 [-127, 127])
    # ==========================================
    # scale = ???
    
    # ==========================================
    # TODO 3: 量化过程
    # 1. 乘以 scale
    # 2. 四舍五入 (torch.round)
    # 3. 截断到 [-128, 127] (torch.clamp)
    # 4. 转换数据类型为 torch.int8 (x.to(torch.int8))
    # ==========================================
    # x_scaled = ???
    # x_quant = ???
    
    # return x_quant, scale
    pass

class W8A16Linear(nn.Module):
    """
    Weight-only INT8 量化线性层。
    在内存中，我们存储的是非常微小的 INT8 权重。
    在计算时，我们将权重反量化回 FP16，与同样是 FP16 的输入进行矩阵乘法。
    这种方式虽然没有加速计算，但极大地缓解了从内存读取权重的 Memory-bound (带宽高了 2 倍)。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 内存中存储的是 int8 格式
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def from_float(self, linear_layer: nn.Linear):
        """
        从高精度的 Linear 层中吸收权重并进行 PTQ 量化
        """
        w_quant, scale = absmax_quantize(linear_layer.weight.data)
        self.weight_int8.copy_(w_quant)
        self.scale.copy_(scale)
        if linear_layer.bias is not None:
            self.bias.data.copy_(linear_layer.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 4: 反量化与前向传播
        # 1. 将 weight_int8 转换回与输入 x 相同的类型 (如 float32/float16)
        # 2. 除以 self.scale 恢复其数值范围
        # 3. 使用 F.linear 进行标准的矩阵乘法
        # ==========================================
        
        # w_fp = ???
        # w_dequant = ???
        
        # out = ???
        # return out
        pass

```

```python
# 测试你的实现
def test_quantization():
    try:
        torch.manual_seed(42)
        
        # 1. 测试 absmax_quantize
        x_fp = torch.tensor([-0.8, 1.5, -3.0, 2.5, 0.0])
        # 绝对最大值是 3.0。Scale = 127 / 3.0 = 42.333
        # 2.5 * 42.333 = 105.8 -> 106
        x_q, scale = absmax_quantize(x_fp)
        
        assert x_q.dtype == torch.int8, "量化后的张量必须是 int8 类型！"
        assert torch.allclose(scale, torch.tensor(127.0 / 3.0)), "Scale 计算不正确！"
        assert x_q[3].item() == 106, "量化后的四舍五入数值计算不正确！"
        print("✅ absmax_quantize 核心算法测试通过！")
        
        # 2. 测试 W8A16 线性层
        in_dim, out_dim = 128, 64
        batch, seq = 2, 10
        
        # 构建一个原始的 FP32 Linear 层
        fp_linear = nn.Linear(in_dim, out_dim)
        
        # 构建我们的 INT8 量化层并吸入权重
        q_linear = W8A16Linear(in_dim, out_dim)
        q_linear.from_float(fp_linear)
        
        # 验证显存占用 (理论上应该小 4 倍，因为 FP32 是 4 字节，INT8 是 1 字节)
        fp_bytes = fp_linear.weight.element_size() * fp_linear.weight.numel()
        q_bytes = q_linear.weight_int8.element_size() * q_linear.weight_int8.numel()
        assert q_bytes == fp_bytes // 4, "INT8 权重的内存占用必须是 FP32 的四分之一！"
        
        # 验证前向传播结果的误差 (量化必然带来微小误差)
        x_input = torch.randn(batch, seq, in_dim)
        out_fp = fp_linear(x_input)
        out_q = q_linear(x_input)
        
        # 计算余弦相似度，因为经过量化反量化，数值不可能完全一致，只要相似度极高就算成功
        cos_sim = F.cosine_similarity(out_fp.flatten(), out_q.flatten(), dim=0)
        assert cos_sim > 0.99, f"反量化计算出的张量与原始张量差异过大，相似度仅为: {cos_sim.item():.4f}"
        
        print(f"✅ W8A16Linear 测试通过！输出相似度极高 (Cosine Sim: {cos_sim.item():.4f})，且权重内存成功缩小 4 倍！")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except AttributeError:
        print("代码未完成导致变量属性错误。")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_quantization()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

权重量化（W8A16）是一种能够在不明显损失精度的前提下，将模型权重的显存占用减半的技术。在实现中，我们将 FP16 权重转换为 INT8，并在推理计算前进行反量化，或者在特定的自定义算子中直接计算。

```python
def quantize_to_int8(tensor):
    scale = tensor.abs().max() / 127.0
    
    quantized_tensor = torch.round(tensor / scale).to(torch.int8)
    
    return quantized_tensor, scale
    
def dequantize_from_int8(quantized_tensor, scale, dtype=torch.float16):
    return (quantized_tensor.to(dtype) * scale)
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
