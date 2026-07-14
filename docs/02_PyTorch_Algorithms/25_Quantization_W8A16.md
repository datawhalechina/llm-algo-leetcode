# 25. Quantization W8A16 | W8A16 量化
**难度：** Medium | **环境：** CPU-first | **标签：** `量化`, `推理优化`, `Linear` | **目标人群：** 推理优化与模型压缩

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把推理链路看完，再看量化会更容易理解它为什么能省显存和提速。

**关键词：** `W8A16`, `INT8`, `quantization`


## 前置阅读

**导语：** 先把推理链路看完，再看量化会更容易理解它为什么能省显存和提速。

- [P1: 01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)


## 相关阅读

**导语：** 量化之后，建议继续看分布式并行策略。

- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [P1: 24. SRAM Optimization Techniques | SRAM 优化技术](../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)

### Step 1: 核心思想与概念

> **什么是量化？**
> 将高精度（如 FP32/FP16，占用 4/2 个字节）的浮点数，映射到低精度（如 INT8，占用 1 个字节）的整数上。这样不仅显存占用直接缩小 2-4 倍，还能利用硬件的整数计算单元（如 INT8 Tensor Core）加速计算。

> **为什么本节只做 Weight-only Quantization？**
> 推理时，显存大头通常来自权重本身。把权重量化到 INT8，就能立刻把静态参数显存压到原来的 1/4。相比之下，激活值往往是动态变化的，是否量化要看具体场景，所以这里先聚焦最稳定、收益最直接的权重量化。

> **PTQ 与 QAT 的区别：**
> - **PTQ (Post-Training Quantization，训练后量化)**：模型已经训练好了。我们只需要拿一小批校准数据（Calibration Data）跑一遍，统计一下激活值的分布，算出缩放因子（Scale），直接对权重转换。本节我们实现的就是 PTQ。
> - **QAT (Quantization-Aware Training，量化感知训练)**：在训练时，正向传播模拟量化的误差，反向传播用“直通估计器 (STE)”更新原始的高精度权重。成本极高，但精度损失最小。

> **量化与反量化的闭环**
> 量化时先根据 `absmax` 计算 `scale = 127 / absmax`，再把张量映射到 INT8 区间；反量化时再把 INT8 乘回 `scale`，近似恢复原始浮点数范围。

### Step 2: 代码实现框架
我们需要实现 `quantize` 和 `dequantize` 两个函数。代码链路非常固定：先算 `absmax`，再算 `scale = 127 / absmax`，然后 `x * scale -> round -> clamp -> int8` 完成量化；反量化则是把 `int8` 转回浮点并除以 `scale`，恢复近似数值范围。

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
```


```python
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
    # ==========================================
    # x_scaled = ???
    # x_quant = ???
    return x_quant, scale

class W8A16Linear(nn.Module):
    """
    Weight-only INT8 量化线性层。
    在内存中，我们存储的是非常微小的 INT8 权重。
    在计算时，我们将权重反量化回 FP16，与同样是 FP16 的输入进行矩阵乘法。
    这种方式虽然没有加速计算，但极大地缓解了从内存读取权重的 Memory-bound (带宽高了 2 倍)。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
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
        return out

```


```python
# 测试你的实现
def test_quantization():
    try:
        torch.manual_seed(42)

        # 1. 测试 absmax_quantize 的基础边界
        zero_q, zero_scale = absmax_quantize(torch.zeros(5))
        assert zero_q.dtype == torch.int8, "量化后的张量必须是 int8 类型！"
        assert torch.count_nonzero(zero_q) == 0, "全 0 张量量化后仍应保持全 0！"
        assert torch.isfinite(torch.as_tensor(zero_scale)).item(), "Scale 不能是 NaN/Inf！"

        # 继续沿用带符号样本验证 scale 和 round 行为
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

        fp_linear = nn.Linear(in_dim, out_dim)
        q_linear = W8A16Linear(in_dim, out_dim)
        q_linear.from_float(fp_linear)

        fp_bytes = fp_linear.weight.element_size() * fp_linear.weight.numel()
        q_bytes = q_linear.weight_int8.element_size() * q_linear.weight_int8.numel()
        assert q_bytes == fp_bytes // 4, "INT8 权重的内存占用必须是 FP32 的四分之一！"

        x_input = torch.randn(batch, seq, in_dim)
        out_fp = fp_linear(x_input)
        out_q = q_linear(x_input)
        cos_sim = F.cosine_similarity(out_fp.flatten(), out_q.flatten(), dim=0)
        assert cos_sim > 0.99, f"反量化计算出的张量与原始张量差异过大，相似度仅为: {cos_sim.item():.4f}"

        # 3. 用一个确定性小矩阵，直接验证“量化权重 -> 反量化 -> 线性层”的公式链路
        fp_linear_small = nn.Linear(4, 3)
        with torch.no_grad():
            fp_linear_small.weight.copy_(torch.tensor([
                [1.0, -2.0, 3.0, -4.0],
                [0.5, 0.25, -0.75, 1.5],
                [-1.0, 0.0, 1.0, -2.0],
            ]))
            fp_linear_small.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

        q_linear_small = W8A16Linear(4, 3)
        q_linear_small.from_float(fp_linear_small)
        x_small = torch.tensor([[1.0, -1.0, 0.5, 2.0], [0.0, 1.0, -1.0, 3.0]])
        out_small = q_linear_small(x_small)
        w_dequant = q_linear_small.weight_int8.to(x_small.dtype) / q_linear_small.scale
        out_ref = F.linear(x_small, w_dequant, q_linear_small.bias)
        assert torch.allclose(out_small, out_ref, atol=1e-6), "小矩阵下的反量化前向公式不正确！"

        print(f"✅ W8A16Linear 测试通过！输出相似度极高 (Cosine Sim: {cos_sim.item():.4f})，且权重内存缩小 4 倍。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成导致变量属性错误。")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义。")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误。")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误。")
        elif isinstance(e, AssertionError):
            print(f"❌ 测试失败: {e}")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误。")
        else:
            print("代码可能未完成，导致了断言失败。")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_quantization()

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
def absmax_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将浮点张量 X 量化为 INT8，并返回缩放因子。
    """
    # TODO 1: 计算张量的绝对最大值 absmax
    absmax = torch.max(torch.abs(x))
    
    # 避免除以 0 的情况
    if absmax == 0:
        absmax = 1e-8
        
    # TODO 2: 计算缩放因子 scale (映射到 [-127, 127])
    scale = 127.0 / absmax
    
    # TODO 3: 量化过程
    x_scaled = x * scale
    x_quant = torch.clamp(torch.round(x_scaled), -128, 127).to(torch.int8)
    
    return x_quant, scale

class W8A16Linear(nn.Module):
    """
    Weight-only INT8 量化线性层。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
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
        # TODO 4: 反量化与前向传播
        # 1. 将 weight_int8 转换回与输入 x 相同的类型
        w_fp = self.weight_int8.to(x.dtype)
        
        # 2. 除以 self.scale 恢复其数值范围
        w_dequant = w_fp / self.scale
        
        # 3. 使用 F.linear 进行标准的矩阵乘法
        out = F.linear(x, w_dequant, self.bias)
        return out
```

### 解析

**1. TODO 1（计算绝对最大值）**
- `absmax = torch.max(torch.abs(x))` 找到张量中最“极端”的值，用它来确定量化动态范围。
- 如果 `absmax` 为 0，需要先做保护，避免除零。

**2. TODO 2（计算缩放因子）**
- `scale = 127.0 / absmax` 将浮点范围映射到 INT8 的对称区间。
- 使用 127 而不是 128，是为了保留对称量化的稳定性，避免额外依赖 `-128` 边界。

**3. TODO 3（量化过程）**
- 先执行 `x_scaled = x * scale`，再 `torch.round`，最后 `torch.clamp` 到可用区间。
- 这一步的本质是把连续浮点数离散化成有限的 INT8 取值。
- `torch.int8` 是最终存储格式，能直接把权重显存压到更低。

**4. TODO 4（反量化与前向传播）**
- 反量化时先把 `weight_int8` 转回与输入一致的数据类型。
- 再除以 `scale` 恢复近似的浮点值范围。
- 最后用 `F.linear` 完成标准前向传播。

**5. 进阶思考**
- 本页实现的是 `per-tensor` 量化，工业界常见更细粒度的 `per-channel` 量化。
- `W8A16` 主要压缩的是权重显存，激活仍保持高精度，以平衡收益与精度。
- 量化带来的收益通常更偏向显存与带宽，而不是把所有计算都变成纯 INT8。
