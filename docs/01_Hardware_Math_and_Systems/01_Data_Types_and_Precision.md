# 01. Data Types and Precision | 大模型的数据格式与混合精度

**难度：** Easy | **环境：** CPU-first | **标签：** `数值基础`, `数据类型`, `混合精度` | **目标人群：** 基础概念补齐者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在计算任何大模型的显存或算力之前，先把“数据”在 GPU 中的表示方式搞清楚。这是硬件推导和量化算法的基础。本节主线从 bit / byte 换算、权重占用和混合精度开始；A100、H100、TF32 与 FP8 作为后续硬件与低精度扩展，不作为前置假设。

这一页是 `Part 01` 的数值与精度基础，主要服务训练稳定性、显存账本和量化前的 dtype 判断。它与 `监督微调路线`、`量化与压缩专题` 共享数值表示基础。

**关键词：** `FP16`, `BF16`, `INT8`

## 证据边界与显存路线映射
本节是 **CPU-first；GPU 用于扩展验证**：CPU 练习可以验证 bit / byte 换算、dtype 的理论字节数和简单账本关系；它不能证明真实 kernel 吞吐、CUDA workspace、allocator reserved 或具体显存节省比例。

对应显存优化路线的 Task1（显存与性能认知底座）和 Task5（量化优化）。只有当 dtype 选择进入真实训练或推理 workload 时，才需要在 73 / 76 或 66 / 67 中验证；本节本身不要求进入 73–76。

---

## 前置阅读
**导语：** 先复习张量和自动求导，再用本页的 dtype 账本解释 BF16、AMP 和 QLoRA 的资源与稳定性取舍。

- [Group 0B: PyTorch Tensors and Autograd | 0B: PyTorch 张量与自动求导](../00_Prerequisites/0B.md)
- [Group 0E: Debugging and Performance | 0E: 调试与性能](../00_Prerequisites/0E.md)
- [02. LLM Params and FLOPs | 大模型参数量与算力推导](./02_LLM_Params_and_FLOPs.md)

## 相关阅读
**导语：** 如果想把 `dtype` 选择继续接到显存预算、量化理论和低比特微调判断上，可以沿这条主线继续往下看。
- [02. LLM Params and FLOPs | 大模型参数量与算力推导](./02_LLM_Params_and_FLOPs.md)
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)

---
## Q1：基础认知——常见的数据格式分别占用多大内存空间？

<details>
<summary>点击展开查看解析</summary>

在计算机底层，1 Byte（字节）= 8 bits（位）。下面先计算理论存储下界；真实模型还可能包含 scale、zero-point、packing、padding 和 runtime workspace。

- **FP32 (单精度浮点数)**: 32 bits = **4 Bytes**
- **FP16 (半精度浮点数)**: 16 bits = **2 Bytes**
- **BF16 (BFloat16)**: 16 bits = **2 Bytes**
- **INT8 (8位整型)**: 8 bits = **1 Byte**
- **INT4 (4位整型)**: 4 bits = **0.5 Byte** (通常用于极度压缩的量化如 AWQ/GPTQ)

**实战估算：**
做权重下界估算时，可以把参数量乘以每参数字节数。比如一个 7B（70亿）参数的模型，如果采用 FP16/BF16 加载，纯权重占用约为：$7 \times 10^9 \times 2 \text{ Bytes} \approx 14 \text{ GB}$。这不是完整的推理或训练峰值显存。
</details>
### Q1小验证：基础显存计算

实现一个函数，计算给定参数量和数据格式的模型显存占用。


```python
def calculate_model_memory(num_params_b, dtype):
    """
    计算模型参数的显存占用
    
    Args:
        num_params_b: 参数量（单位：B，即十亿）
        dtype: 数据类型，可选 'fp32', 'fp16', 'bf16', 'int8', 'int4'
    
    Returns:
        memory_gb: 理论权重占用（十进制 GB，不包含量化元数据和运行时缓冲）
    
    示例:
        >>> calculate_model_memory(7, 'fp16')
        14.0
        >>> calculate_model_memory(7, 'int8')
        7.0
    """
    if num_params_b < 0:
        raise ValueError("num_params_b must be non-negative")

    # 每种数据类型占用的理论字节数
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    memory_gb = num_params_b * bytes_per_param[dtype]
    return memory_gb
```


```python
# 测试函数
def test_calculate_model_memory():
    try:
        # 测试用例 1: LLaMA-7B FP16
        result = calculate_model_memory(7, 'fp16')
        assert result == 14, f"错误：LLaMA-7B FP16 应该是 14 GB，实际 {result} GB"
        
        # 测试用例 2: LLaMA-7B INT8
        result = calculate_model_memory(7, 'int8')
        assert result == 7, f"错误：LLaMA-7B INT8 应该是 7 GB，实际 {result} GB"
        
        # 测试用例 3: LLaMA-13B FP16
        result = calculate_model_memory(13, 'fp16')
        assert result == 26, f"错误：LLaMA-13B FP16 应该是 26 GB，实际 {result} GB"
        
        # 测试用例 4: LLaMA-70B INT4
        result = calculate_model_memory(70, 'int4')
        assert result == 35, f"错误：LLaMA-70B INT4 应该是 35 GB，实际 {result} GB"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_model_memory()
```

### Q1扩展验证：对比不同数据格式

使用上面的函数，对比 LLaMA-7B 在不同数据格式下的显存占用。


```python
# 对比 LLaMA-7B 在不同格式下的显存占用
model_name = "LLaMA-7B"
num_params = 7
dtypes = ['fp32', 'fp16', 'bf16', 'int8', 'int4']

print(f"{model_name} 显存占用对比：")
print("-" * 40)
for dtype in dtypes:
    memory = calculate_model_memory(num_params, dtype)
    print(f"{dtype.upper():<8} {memory:>6.1f} GB")
```

## Q2：底层原理——同样是 16-bit，FP16 和 BF16 的底层位分布有什么本质区别？

<details>
<summary>点击展开查看解析</summary>

这涉及浮点数在底层的位分布设计：一个浮点数由 **符号位 (Sign)** + **指数位 (Exponent)** + **尾数位/精度位 (Mantissa/Fraction)** 组成。
核心法则是：**指数位决定了数值的范围大小，尾数位决定了数值的精确度。**

1. **FP16 的结构**：1 位符号 + **5 位指数** + 10 位尾数。
   - 5 位指数意味着它能表示的最大数值只有 **65504**。
   - 尾数长，所以它对小数部分的表示非常“精细”。

2. **BF16 (Brain Float 16) 的结构**：1 位符号 + **8 位指数** + 7 位尾数。
   - 它是 Google Brain 专门为深度学习发明的。它其实就是直接把 FP32（8位指数）砍掉了后面的 16 位尾数！
   - 因为拥有 8 位指数，BF16 能表示的最大数值范围和 FP32 一模一样（高达 $3.4 \times 10^{38}$），**极难发生数值溢出**。代价是尾数位从 10 降到了 7，牺牲了一点数值的“精确度”。
</details>

### Q2小验证：混合精度训练显存计算

在本题的混合精度训练近似中，显存账本只统计参数、梯度和 Adam/SGD 训练状态：
- 模型参数（FP16/BF16）：2Φ
- 梯度（FP16/BF16）：2Φ
- 优化器状态（FP32）：
  - FP32 主权重：4Φ
  - 一阶动量（Adam）：4Φ
  - 二阶动量（Adam）：4Φ
  - 总计：12Φ

**训练状态近似 = 2Φ + 2Φ + 12Φ = 16Φ**

> 这里假设参数和梯度使用 16-bit，Adam 的 FP32 master weights、一次动量和二次动量各占 4 bytes；不包含 activation、通信 buffer、allocator reserved memory 和临时 workspace。不同 optimizer、实现或 offload 配置会改变这个账本。


```python
def calculate_training_memory(num_params_b, model_dtype='fp16', optimizer='adam'):
    """
    计算训练状态显存的理论近似，不代表完整的 peak memory。
    
    Args:
        num_params_b: 参数量（单位：B）
        model_dtype: 模型数据类型（'fp16' 或 'bf16'）
        optimizer: 优化器类型（'adam' 或 'sgd'）
    
    Returns:
        total_memory_gb: 训练状态近似占用（十进制 GB）
    
    示例:
        >>> calculate_training_memory(7, 'fp16', 'adam')
        112.0
    """
    if num_params_b < 0:
        raise ValueError("num_params_b must be non-negative")
    if optimizer not in {'adam', 'sgd'}:
        raise ValueError("optimizer must be 'adam' or 'sgd'")

    model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
    gradient_bytes = model_bytes
    optimizer_bytes = 12 if optimizer == 'adam' else 4
    total_memory_gb = num_params_b * (model_bytes + gradient_bytes + optimizer_bytes)
    return total_memory_gb
```


```python
# 测试函数
def test_calculate_training_memory():
    try:
        # 测试用例 1: LLaMA-7B + Adam
        result = calculate_training_memory(7, 'fp16', 'adam')
        assert result == 112, f"错误：应该是 112 GB，实际 {result} GB"
        
        # 测试用例 2: LLaMA-7B + SGD
        result = calculate_training_memory(7, 'fp16', 'sgd')
        assert result == 56, f"错误：应该是 56 GB，实际 {result} GB"
        
        # 测试用例 3: LLaMA-13B + Adam
        result = calculate_training_memory(13, 'bf16', 'adam')
        assert result == 208, f"错误：应该是 208 GB，实际 {result} GB"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_training_memory()
```


```python
# 分析 LLaMA-7B 训练时的显存分布
num_params = 7

model_params = num_params * 2  # FP16 模型参数
gradients = num_params * 2     # FP16 梯度
optimizer_states = num_params * 12  # FP32 优化器状态
total = model_params + gradients + optimizer_states

print(f"LLaMA-7B 混合精度训练显存分布：")
print("-" * 50)
print(f"模型参数 (FP16):      {model_params:>6.1f} GB ({model_params/total*100:>5.1f}%)")
print(f"梯度 (FP16):          {gradients:>6.1f} GB ({gradients/total*100:>5.1f}%)")
print(f"优化器状态 (FP32):    {optimizer_states:>6.1f} GB ({optimizer_states/total*100:>5.1f}%)")
print("-" * 50)
print(f"总计:                 {total:>6.1f} GB")
print("\n结论：优化器状态占据了大部分显存！")
```

## Q3：训练中为什么常把 BF16 与 FP16 放在一起评估？

<details>
<summary>点击展开查看解析</summary>

选择 BF16 还是 FP16，通常要同时看数值范围、硬件支持、吞吐和 loss-scaling 配置。

**1. FP16 的核心挑战：动态范围受限，上溢风险高**

FP16 的动态范围（最大值约 65504）远窄于 FP32（约 $3.4 \times 10^{38}$）。在大模型训练中：

- **上溢风险**：Attention 的 logits、未归一化激活或梯度在某些 workload 中可能超过 65504，产生 `Inf/NaN`，因此 FP16 训练通常需要更仔细地监控数值。
  
- **下溢问题（次要）**：反向传播中的小梯度（如 $10^{-7}$ 量级）可能因精度不足被截断，影响参数更新。

**Loss Scaling 的权宜之计**：

FP16 训练中常见的配套措施是 **Loss Scaling（损失缩放）**：
- **原理**：在反向传播前放大损失值（如乘以 1024），将小梯度放大到 FP16 可表示范围，计算完成后再缩小回来，从而缓解下溢。
- **局限**：只能解决梯度的下溢，无法解决前向传播中的上溢；且缩放因子需要动态调整（检测溢出后减小，长期无溢出时增大），增加了工程复杂度。
- **历史成功**：早期的 BERT、GPT-2 等模型在 V100（仅支持 FP16 Tensor Core）上通过精心调优仍实现了稳定训练。

**2. BF16 的三大优势：稳定、简单、硬件支持**

- **更大的动态范围**：BF16 继承了 FP32 的 8 位指数，最大值约为 $3.4 \times 10^{38}$；在许多训练 workload 中，它比 FP16 更不容易因为范围不足而上溢，但仍需进行数值监控。
  
- **配置相对简单**：许多 BF16 训练配置不需要 FP16 那样的动态 loss scaling，但是否稳定仍取决于模型、算子和训练设置。
  
- **硬件支持**：A100（Ampere）及更新架构提供 BF16 Tensor Core 路径；实际吞吐仍取决于 GPU、算子和框架实现。

**3. 精度 vs 范围：为什么神经网络更需要范围？**

虽然 BF16 的尾数位从 10 位降到 7 位（损失约 3 位精度），但：

- **神经网络对精度损失鲁棒**：训练是长期累积的统计过程，单步的微小舍入误差会被后续更新”平滑”掉。
- **上溢是灾难性的**：一旦出现 `NaN`，会立即传播到整个模型，导致训练不可恢复地崩溃。

因此，BF16 常被作为大模型训练的候选格式；最终仍应结合 loss、吞吐和硬件实测选择。

**4. 数值对比与应用场景**

| 格式 | 指数位 | 尾数位 | 最大值 | 训练稳定性 | 主要应用 |
|------|--------|--------|--------|-----------|---------|
| FP32 | 8 | 23 | $10^{38}$ | 最稳定 | 基准/调试 |
| FP16 | 5 | 10 | $6.5 \times 10^4$ | 需 Loss Scaling | 推理优化 |
| BF16 | 8 | 7 | $10^{38}$ | 极稳定 | **大模型训练** ✅ |

**补充说明**：FP16 并未被完全”抛弃”——在推理场景中，由于不涉及梯度计算，数值范围需求较小，FP16 的更高精度（10 位尾数）反而能带来更好的输出质量，许多推理框架（如 TensorRT、vLLM）仍优先使用 FP16。

**总结**：BF16 的优势主要来自较大的指数范围和较广的硬件支持，但它不是脱离 workload 的固定答案。
</details>

### Q3小验证：量化显存节省计算


```python
def calculate_quantization_savings(num_params_b, from_dtype, to_dtype):
    """
    计算量化后的显存节省
    
    Args:
        num_params_b: 参数量（单位：B）
        from_dtype: 原始数据类型
        to_dtype: 量化后的数据类型
    
    Returns:
        savings_gb: 节省的显存（单位：GB）
        savings_percent: 节省的百分比
    
    示例:
        >>> calculate_quantization_savings(7, 'fp16', 'int8')
        (7.0, 50.0)
    """
    original_memory = calculate_model_memory(num_params_b, from_dtype)
    quantized_memory = calculate_model_memory(num_params_b, to_dtype)
    savings_gb = original_memory - quantized_memory
    savings_percent = savings_gb / original_memory * 100
    return savings_gb, savings_percent
```


```python
# 测试函数
def test_calculate_quantization_savings():
    try:
        # 测试用例 1: FP16 -> INT8
        savings_gb, savings_percent = calculate_quantization_savings(7, 'fp16', 'int8')
        assert savings_gb == 7, f"错误：应该节省 7 GB，实际 {savings_gb} GB"
        assert savings_percent == 50, f"错误：应该节省 50%，实际 {savings_percent}%"
        
        # 测试用例 2: FP16 -> INT4
        savings_gb, savings_percent = calculate_quantization_savings(7, 'fp16', 'int4')
        assert savings_gb == 10.5, f"错误：应该节省 10.5 GB，实际 {savings_gb} GB"
        assert savings_percent == 75, f"错误：应该节省 75%，实际 {savings_percent}%"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_quantization_savings()
```

## Q4：在一些 BF16 混合精度训练实现中，为什么优化器仍会保留 FP32 主权重？

<details>
<summary>点击展开查看解析</summary>

在一种常见的混合精度训练配置中，前向和反向的部分计算使用 16-bit，以降低存储和计算成本；实际是否节省 50%、是否使用 Tensor Core，取决于参数、梯度和算子实现。

但问题出在**参数更新（Optimizer Step）**这一步：
$$ W_{new} = W_{old} - \text{Learning\_Rate} \times \text{Gradient} $$

在大模型训练后期，学习率（LR）通常非常小（例如 $10^{-5}$），计算出的梯度通常也很小。两者的乘积是一个极其微小的更新量 $\Delta W$。
如果 $W_{old}$ 也只用 16-bit 格式保存，有限的尾数精度可能让一部分很小的更新在舍入时消失（例如 $1.0 + 0.0000001$ 在低精度下可能仍接近 $1.0$）。因此，一些实现会在 FP32 副本上累积更新，再把结果转换到前向计算使用的 dtype。

因此，在一些全参数混合精度训练实现中，优化器会保留一份 **FP32 的 Master Weights**。每次反向传播算出梯度后，在高精度副本上更新，再把结果转换到前向计算使用的 dtype。是否需要、如何保存以及由哪个组件管理，取决于 optimizer、框架和 AMP 实现；这也是训练状态显存可能明显高于静态权重的原因之一。
</details>
### Q4小验证：实际场景应用

给定 GPU 显存容量，计算能加载多大的模型。


```python
def max_model_size(gpu_memory_gb, dtype, overhead_ratio=0.2):
    """
    计算给定 GPU 显存能加载的最大模型参数量
    
    Args:
        gpu_memory_gb: GPU 显存容量（单位：GB）
        dtype: 数据类型
        overhead_ratio: 预留给 KV Cache 和激活值的显存比例（默认 20%）
    
    Returns:
        max_params_b: 最大参数量（单位：B）
    
    示例:
        >>> max_model_size(80, 'fp16', 0.2)
        32.0
    """
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5,
    }
    available_memory = gpu_memory_gb * (1 - overhead_ratio)
    max_params_b = available_memory / bytes_per_param[dtype]
    return max_params_b
```


```python
# 测试不同 GPU 能加载的最大模型
gpus = [
    ('RTX 3090', 24),
    ('RTX 4090', 24),
    ('A100 40GB', 40),
    ('A100 80GB', 80),
    ('H100 80GB', 80),
]

print("不同 GPU 能加载的最大模型参数量（FP16，预留 20% 显存）：")
print("-" * 60)
print(f"{'GPU':<15} {'显存':<10} {'最大模型 (FP16)':<20} {'最大模型 (INT8)'}")
print("-" * 60)

for gpu_name, memory in gpus:
    max_fp16 = max_model_size(memory, 'fp16', 0.2)
    max_int8 = max_model_size(memory, 'int8', 0.2)
    print(f"{gpu_name:<15} {memory:>4} GB     {max_fp16:>6.1f}B              {max_int8:>6.1f}B")
```

## Q5：A100（Ampere）在数据精度支持上带来了哪些变化？

<details>
<summary>点击展开查看解析</summary>

A100 的一个重要变化是提供了 BF16 Tensor Core 路径，并引入 TF32 作为 FP32 矩阵乘法的加速路径。具体收益取决于算子、库和配置。

**1. 原生支持 BF16 Tensor Core**
- 在 A100 之前的 V100（Volta 架构）时期，Tensor Core **只支持 FP16** 乘法。这就是为什么早期研究人员在训练模型时深受溢出之苦。
- A100 的第三代 Tensor Core 支持 BF16 乘加。在特定矩阵规模和库实现下，BF16 Tensor Core 吞吐会明显高于 FP32 路径；它改善了范围不足问题，但不代表训练自动稳定。

**2. 引入了神兵利器：TF32 (TensorFloat-32)**
- NVIDIA 为了让开发者在不改动任何祖传代码（继续写纯 FP32 代码）的情况下也能用上 Tensor Core 的加速，发明了 TF32 格式。
- TF32 是一种精妙的混合态格式：它拥有 **FP32 的指数位（8位，保证不溢出）** 和 **FP16 的尾数位（10位，保证精度）**，总共占用 19 个 bit 的信息，但在显存中依然按 32位 存储。
- **底层机制**：当你在 PyTorch 中设置 `torch.backends.cuda.matmul.allow_tf32 = True`（在 A100 及更新的架构上这是默认开启的）时，如果你向 GPU 丢了两个 FP32 矩阵相乘，A100 会在内部的 Tensor Core 里将其**截断为 TF32 更快算完**，然后再转回 FP32 输出。这让“看似是单精度”的矩阵乘法获得数倍级的性能提升。
</details>
### Q5小验证：A100 的精度路径

比较 A100 在 FP32 / TF32 / BF16 / FP16 下的计算路径和张量核心利用方式。


```python
def a100_precision_path(requested_dtype, allow_tf32=True):
    table = {
        'fp32': {'storage_bits': 32, 'tensor_core': allow_tf32, 'path': 'tf32' if allow_tf32 else 'fp32', 'speed_hint': 1.0 if not allow_tf32 else 4.0},
        'tf32': {'storage_bits': 32, 'tensor_core': True, 'path': 'tf32', 'speed_hint': 4.0},
        'bf16': {'storage_bits': 16, 'tensor_core': True, 'path': 'bf16', 'speed_hint': 2.0},
        'fp16': {'storage_bits': 16, 'tensor_core': True, 'path': 'fp16', 'speed_hint': 2.0},
    }
    return table[requested_dtype]

for dtype in ['fp32', 'tf32', 'bf16', 'fp16']:
    print(dtype, '->', a100_precision_path(dtype))
print('A100 makes BF16 and TF32 first-class paths for Tensor Core acceleration')

```

## Q6：前沿演进——NVIDIA 在 H100（Hopper 架构）中引入的原生 FP8 格式，有什么专门针对 AI 的设计？

<details>
<summary>点击展开查看解析</summary>

随着模型规模和吞吐要求增加，FP8 提供了比 16-bit 更低的存储和计算精度。H100 为 FP8 提供了原生计算路径。

8-bit 格式需要在指数范围和尾数精度之间取舍，因此常见 FP8 格式包括两种侧重点不同的变体：

1. **E4M3 格式** (4位指数 + 3位尾数)：
   - **侧重：精度**。
   - **用途**：常用于前向传播和激活值，但实际选择取决于模型、量化策略和框架实现。
2. **E5M2 格式** (5位指数 + 2位尾数)：
   - **侧重：动态范围**。
   - **用途**：常用于梯度等需要更大动态范围的张量；实际映射也取决于训练实现和数值校准。

FP8 能降低张量存储和搬运量，并在支持的算子上提供更高吞吐；是否带来端到端收益，需要结合模型质量、校准和实际 workload 测量。
</details>
### Q6小验证：H100 的 FP8 变体选择

对比 E4M3 和 E5M2，看看前向和反向为什么要分开设计。


```python
def fp8_variant_policy(stage):
    variants = {
        'forward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision', 'stage': 'forward / activations'},
        'backward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range', 'stage': 'backward / gradients'},
    }
    return variants[stage]

for stage in ['forward', 'backward']:
    print(stage, '->', fp8_variant_policy(stage))
print('H100 splits FP8 into two variants to balance precision and dynamic range')

```
