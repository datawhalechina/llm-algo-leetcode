# 01. Data Types and Precision | 大模型的数据格式与混合精度

**难度：** Easy | **环境：** CPU-first | **标签：** `数值基础`, `数据类型`, `混合精度` | **目标人群：** 需要理解数据格式与数值稳定性的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

模型里的参数、梯度、激活值和缓存都要以特定数据类型（dtype）存储和计算。相同张量采用 FP32、BF16、FP16 或 INT8 时，存储字节数、数值范围、误差风险和可用硬件路径都会变化。
本节建立“数值表示 → 状态字节数 → 精度选择”的关系。完成后，你应能计算模型状态的理论容量，并结合数值范围判断一种精度是否值得进入真实硬件验证。

**关键词：** `FP16`, `BF16`, `INT8`

![本节概念关系](../public/01_Hardware_Math_and_Systems/01_dtype_precision_map.svg)

---
## 前置阅读
**导语：** 进入本节前，先回顾张量和自动求导；随后用 dtype 账本理解混合精度、低比特表示与训练显存之间的关系。

- [Group 0B: PyTorch Tensors and Autograd | 0B: PyTorch 张量与自动求导](../00_Prerequisites/0B.md)

---
## Q1：基础认知——常见的数据格式分别占用多大内存空间？


在计算机底层，1 Byte（字节）= 8 bits（位）。下面先计算理论存储下界；真实模型还可能包含 scale、zero-point、packing、padding 和 runtime workspace。

- **FP32 (单精度浮点数)**: 32 bits = **4 Bytes**
- **FP16 (半精度浮点数)**: 16 bits = **2 Bytes**
- **BF16 (BFloat16)**: 16 bits = **2 Bytes**
- **INT8 (8位整型)**: 8 bits = **1 Byte**
- **INT4 (4位整型)**: 4 bits = **0.5 Byte** (通常用于极度压缩的量化如 AWQ/GPTQ)

**实战估算：**
做权重下界估算时，可以把参数量乘以每参数字节数。比如一个 7B（70亿）参数的模型，如果采用 FP16/BF16 加载，纯权重占用约为：$7 \times 10^9 \times 2 \text{ Bytes} \approx 14 \text{ GB}$。这不是完整的推理或训练峰值显存。

### Q1 机制计算器：从位宽换算理论权重容量

先预测 7B 模型在不同格式下的容量，再运行计算器。随后只修改 `MODEL_PARAMS_B` 或 `MODEL_DTYPE`，检查容量是否按参数量和位宽线性变化。


```python
DTYPE_BITS = {
    'fp32': 32,
    'fp16': 16,
    'bf16': 16,
    'int8': 8,
    'int4': 4,
}
BITS_PER_BYTE = 8
PARAMS_PER_BILLION = 1_000_000_000
BYTES_PER_DECIMAL_GB = 1_000_000_000


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

    try:
        bits_per_element = DTYPE_BITS[dtype]
    except KeyError as exc:
        raise ValueError(f'unsupported dtype: {dtype}') from exc
    num_params = num_params_b * PARAMS_PER_BILLION
    total_bits = num_params * bits_per_element
    total_bytes = total_bits / BITS_PER_BYTE
    memory_gb = total_bytes / BYTES_PER_DECIMAL_GB
    return memory_gb
```


```python
def test_calculate_model_memory():
    """验证位宽比例、参数规模比例与非法输入。"""
    assert calculate_model_memory(7, 'fp16') == 14
    assert calculate_model_memory(7, 'int8') == 7
    assert calculate_model_memory(13, 'fp16') == 26
    assert calculate_model_memory(70, 'int4') == 35
    assert calculate_model_memory(14, 'fp16') == 2 * calculate_model_memory(7, 'fp16')
    try:
        calculate_model_memory(7, 'unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('未知 dtype 必须报错')

test_calculate_model_memory()
```

### Q1 对照验证：固定模型规模，只改变数据格式

下面是本节的显式可修改输入。保持模型规模不变时，容量差异只来自每元素位宽；这仍是理论权重下界。


```python
MODEL_NAME = '7B 教学模型'  # 可修改：只用于输出标签。
MODEL_PARAMS_B = 7          # 可修改：模型参数量，单位 B。
DTYPE_CANDIDATES = ['fp32', 'fp16', 'bf16', 'int8', 'int4']  # 可修改：候选格式。

print(f"{MODEL_NAME} 的理论权重容量：")
print("-" * 40)
for dtype in DTYPE_CANDIDATES:
    memory = calculate_model_memory(MODEL_PARAMS_B, dtype)
    print(f"{dtype.upper():<8} {memory:>6.1f} GB")
```

## Q2：同样是 16-bit，FP16 和 BF16 的位分布如何影响范围与精度？


这涉及浮点数在底层的位分布设计：一个浮点数由 **符号位 (Sign)** + **指数位 (Exponent)** + **尾数位/精度位 (Mantissa/Fraction)** 组成。
核心法则是：**指数位决定了数值的范围大小，尾数位决定了数值的精确度。**

1. **FP16 的结构**：1 位符号 + **5 位指数** + 10 位尾数。
   - 5 位指数意味着它能表示的最大数值只有 **65504**。
   - 尾数长，所以它对小数部分的表示非常“精细”。

2. **BF16 (Brain Float 16) 的结构**：1 位符号 + **8 位指数** + 7 位尾数。
   - 它是 Google Brain 专门为深度学习发明的。它其实就是直接把 FP32（8位指数）砍掉了后面的 16 位尾数！
   - 因为拥有 8 位指数，BF16 的指数范围与 FP32 相同（最大有限值约为 $3.4 \times 10^{38}$），在许多训练 workload 中比 FP16 更不容易因范围不足而溢出；这不等于不会出现 Inf/NaN。代价是尾数位从 10 降到了 7，舍入精度相对较低。

位分布先解释数值范围与精度；训练状态账本、优化器状态和 Master Weights 放到后面的 Q4 与显存账本小节中。

### Q2 对照验证：真实转换后的范围与舍入误差

使用同一组 FP32 数值分别转换为 FP16 和 BF16，观察最大有限值、机器精度、有限值数量和转换误差。唯一变化变量是目标 dtype。


```python
import torch

TORCH_FLOAT_DTYPES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def cast_error_report(values, dtype_name):
    """将同一 FP32 输入转换到目标 dtype，并报告范围与舍入误差。"""
    if dtype_name not in TORCH_FLOAT_DTYPES:
        raise ValueError(f'unsupported float dtype: {dtype_name}')
    source = torch.tensor(values, dtype=torch.float32)
    target_dtype = TORCH_FLOAT_DTYPES[dtype_name]
    converted = source.to(target_dtype)
    restored = converted.to(torch.float32)
    info = torch.finfo(target_dtype)
    finite_mask = torch.isfinite(converted)
    return {
        'dtype': dtype_name,
        'max_finite': info.max,
        'eps_near_one': info.eps,
        'finite_count': int(finite_mask.sum().item()),
        'total_count': source.numel(),
        'max_abs_error': float((restored[finite_mask] - source[finite_mask]).abs().max().item()),
        'restored': restored.tolist(),
    }
```


```python
def test_cast_error_report():
    """验证同为 16-bit 时，FP16 与 BF16 的范围—精度取舍。"""
    values = [1.0, 1.001, 60_000.0, 70_000.0]
    fp16 = cast_error_report(values, 'fp16')
    bf16 = cast_error_report(values, 'bf16')
    assert fp16['finite_count'] == 3
    assert bf16['finite_count'] == 4
    assert fp16['eps_near_one'] < bf16['eps_near_one']
    try:
        cast_error_report(values, 'int8')
    except ValueError:
        pass
    else:
        raise AssertionError('非浮点 dtype 必须报错')

test_cast_error_report()
```


```python
PROBE_VALUES = [1.0, 1.001, 60_000.0, 70_000.0]  # 可修改：同时包含精度和范围压力。
for dtype_name in ['fp16', 'bf16']:
    report = cast_error_report(PROBE_VALUES, dtype_name)
    print(dtype_name, '->', report)
```

## Q3：损失缩放能解决 FP16 的哪些问题？

FP16 的范围较窄：过大的前向值可能变成 `Inf`，过小的梯度可能在转换时变成零。损失缩放先放大梯度，在低精度转换后再缩小，主要缓解小梯度下溢；它不能修复前向阶段已经发生的上溢。

BF16 的指数范围更大，通常较少依赖损失缩放，但尾数更短，舍入误差可能高于 FP16。两者都需要用固定输入检查有限值、零值比例和误差，不能只根据格式名称判断稳定性。

| 观察项 | FP16 | BF16 |
|:---|:---|:---|
| 范围风险 | 上溢与小梯度下溢 | 范围接近 FP32 |
| 分辨率 | 尾数较长 | 尾数较短 |
| 常见训练控制 | 动态损失缩放 | 通常不需要损失缩放，但仍需有限值检查 |

### Q3 对照验证：固定梯度，只改变 dtype 与缩放因子


```python
def gradient_survival_report(gradients, dtype_name, loss_scale=1.0):
    """模拟缩放、低精度转换与反缩放，统计仍可参与更新的梯度。"""
    if loss_scale <= 0:
        raise ValueError('loss_scale must be positive')
    if dtype_name not in {'fp16', 'bf16'}:
        raise ValueError('dtype_name must be fp16 or bf16')
    source = torch.tensor(gradients, dtype=torch.float32)
    target_dtype = TORCH_FLOAT_DTYPES[dtype_name]
    low_precision = (source * loss_scale).to(target_dtype)
    restored = low_precision.to(torch.float32) / loss_scale
    return {
        'dtype': dtype_name,
        'loss_scale': loss_scale,
        'nonzero_count': int((restored != 0).sum().item()),
        'finite_count': int(torch.isfinite(restored).sum().item()),
        'restored': restored.tolist(),
    }

```


```python
TINY_GRADIENTS = [1e-4, 1e-6, 1e-8, 1e-10]  # 可修改：观察更小梯度何时消失。
fp16_unscaled = gradient_survival_report(TINY_GRADIENTS, 'fp16', loss_scale=1.0)
fp16_scaled = gradient_survival_report(TINY_GRADIENTS, 'fp16', loss_scale=1024.0)
bf16_unscaled = gradient_survival_report(TINY_GRADIENTS, 'bf16', loss_scale=1.0)

assert fp16_scaled['nonzero_count'] >= fp16_unscaled['nonzero_count']
assert bf16_unscaled['finite_count'] == len(TINY_GRADIENTS)
print('FP16 without scaling:', fp16_unscaled)
print('FP16 with scaling:   ', fp16_scaled)
print('BF16 without scaling:', bf16_unscaled)
```

## Q4：为什么混合精度训练仍可能保留 FP32 Master Weights？


在一种常见的混合精度训练配置中，前向和反向的部分计算使用 16-bit，但参数更新可能继续使用 FP32 状态。是否保留 Master Weights，取决于 optimizer、框架和 AMP 实现。

但问题出在**参数更新（Optimizer Step）**这一步：
$$ W_{new} = W_{old} - \text{Learning\_Rate} \times \text{Gradient} $$

在大模型训练后期，学习率（LR）通常非常小（例如 $10^{-5}$），计算出的梯度通常也很小。两者的乘积是一个极其微小的更新量 $\Delta W$。
如果 $W_{old}$ 也只用 16-bit 格式保存，有限的尾数精度可能让一部分很小的更新在舍入时消失（例如 $1.0 + 0.0000001$ 在低精度下可能仍接近 $1.0$）。因此，一些实现会在 FP32 副本上累积更新，再把结果转换到前向计算使用的 dtype。

因此，一些全参数混合精度训练实现会保留一份 **FP32 Master Weights**：梯度在高精度副本上参与更新，再把结果转换到前向计算使用的 dtype。它是训练状态账本的一部分，不应与静态模型权重占用混为一谈。
下面的容量估算只用于观察 Master Weights 的理论增量和静态权重容量；它不模拟 activation、通信 buffer、allocator peak 或真实 OOM 边界。

### Q4 机制计算器：逐项建立训练状态账本

把参数、梯度、Adam 一阶矩、Adam 二阶矩和可选 Master Weights 分开计算。显式组件可以避免把 `12 bytes/parameter` 与 Master Weights 再次相加。


```python
def training_state_ledger(
    num_params_b,
    parameter_dtype='fp16',
    gradient_dtype='fp16',
    moment_dtype='fp32',
    keep_master_weights=True,
):
    """返回静态训练状态账本；不包含激活、临时缓冲和分配器预留。"""
    if num_params_b <= 0:
        raise ValueError('num_params_b must be positive')
    for dtype_name in (parameter_dtype, gradient_dtype, moment_dtype):
        if dtype_name not in DTYPE_BITS:
            raise ValueError(f'unsupported dtype: {dtype_name}')

    def component_gb(dtype_name, copies=1):
        return calculate_model_memory(num_params_b, dtype_name) * copies

    components = {
        'parameters_gb': component_gb(parameter_dtype),
        'gradients_gb': component_gb(gradient_dtype),
        'adam_first_moment_gb': component_gb(moment_dtype),
        'adam_second_moment_gb': component_gb(moment_dtype),
        'master_weights_gb': component_gb('fp32') if keep_master_weights else 0.0,
    }
    return {
        'components': components,
        'total_gb': sum(components.values()),
        'scope': 'static_training_states_only',
    }


def test_training_state_ledger():
    """验证每个训练状态只计数一次，并检查 Master Weights 增量。"""
    with_master = training_state_ledger(7, keep_master_weights=True)
    without_master = training_state_ledger(7, keep_master_weights=False)
    assert with_master['components']['parameters_gb'] == 14
    assert with_master['components']['adam_first_moment_gb'] == 28
    assert with_master['components']['adam_second_moment_gb'] == 28
    assert with_master['components']['master_weights_gb'] == 28
    assert with_master['total_gb'] - without_master['total_gb'] == 28

test_training_state_ledger()
```


```python
TRAINING_PARAMS_B = 7               # 可修改：参数量，单位 B。
KEEP_MASTER_WEIGHTS = True          # 可修改：是否额外保留 FP32 参数副本。
ledger = training_state_ledger(
    TRAINING_PARAMS_B,
    parameter_dtype='fp16',
    gradient_dtype='fp16',
    moment_dtype='fp32',
    keep_master_weights=KEEP_MASTER_WEIGHTS,
)
for name, size_gb in ledger['components'].items():
    print(f'{name:<28} {size_gb:>6.1f} GB')
print('-' * 40)
print(f"{'total_gb':<28} {ledger['total_gb']:>6.1f} GB")
```

## Q5：A100 为什么同时引入 BF16 Tensor Core 和 TF32 路径？


A100 的关键变化不是简单增加一种存储 dtype，而是同时提供了两条不同的计算路径：BF16 可以作为 16-bit Tensor Core 输入，TF32 则让 FP32 输入有机会进入 Tensor Core 矩阵乘法。先区分“张量如何存储”和“矩阵乘法如何执行”，再讨论吞吐、数值误差和配置。

**1. 原生支持 BF16 Tensor Core**
- 在 V100 的 Tensor Core 矩阵乘加路径上，主要使用 FP16 输入；这不等于 V100 的所有计算都只能使用 FP16。
- A100 的 Tensor Core 增加了 BF16 乘加路径。BF16 保留与 FP32 相同的指数位宽，通常比 FP16 有更大的动态范围；是否获得收益仍取决于矩阵规模、算子实现和 workload。

**2. TF32 是 FP32 输入的一条计算路径**
- TF32 不是需要单独保存的 19-bit 模型 dtype；输入和输出仍可按 FP32 存储，矩阵乘法内部使用较低有效精度的 Tensor Core 路径。
- `torch.backends.cuda.matmul.allow_tf32` 等配置会影响部分 FP32 矩阵乘法是否采用 TF32；实际行为还取决于 PyTorch、CUDA、算子和硬件。不能把配置开关直接当成实测吞吐结论。
- 因此，Q5 的核心判断是：FP32 / TF32 主要改变计算路径，BF16 / FP16 同时改变存储宽度和计算路径；真实 Tensor Core 使用情况需要固定 workload 和 profiler 验证。

### Q5小验证：A100 的计算路径候选

比较 FP32、TF32、BF16 和 FP16 的存储表示与计算路径候选；结果只表示理论路径，真实 Tensor Core 使用情况需要 GPU profiler 验证。


```python
def a100_precision_path(requested_dtype, allow_tf32=True):
    """返回 A100 的理论路径候选；不估算 speedup，也不证明真实 Tensor Core 执行。"""
    table = {
        'fp32': {'storage_dtype': 'fp32', 'compute_path': 'tf32' if allow_tf32 else 'fp32', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': allow_tf32},
        'tf32': {'storage_dtype': 'fp32', 'compute_path': 'tf32', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': True},
        'bf16': {'storage_dtype': 'bf16', 'compute_path': 'bf16_tensor_core', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': True},
        'fp16': {'storage_dtype': 'fp16', 'compute_path': 'fp16_tensor_core', 'accumulation_dtype': 'fp32', 'tensor_core_candidate': True},
    }
    if requested_dtype not in table:
        raise ValueError(f'unsupported dtype: {requested_dtype}')
    return {**table[requested_dtype], 'requires_gpu_validation': True}

for dtype in ['fp32', 'tf32', 'bf16', 'fp16']:
    print(dtype, '->', a100_precision_path(dtype))
assert a100_precision_path('fp32', allow_tf32=False)['compute_path'] == 'fp32'
assert a100_precision_path('fp32')['storage_dtype'] == 'fp32'
assert a100_precision_path('bf16')['tensor_core_candidate'] is True
try:
    a100_precision_path('int8')
except ValueError:
    print('✅ dtype 边界校验通过')
else:
    raise AssertionError('本题只比较 A100 的 FP32 / TF32 / BF16 / FP16 路径')

```

## Q6：H100 的 FP8 Tensor Core 为什么通常同时使用 E4M3 和 E5M2？


随着模型规模和吞吐要求增加，FP8 提供了比 16-bit 更低的存储和计算成本；H100 的 Tensor Core 为 FP8 提供了原生矩阵计算路径。这里要把“格式选择”和“硬件支持”分开理解：格式决定数值范围与精度，硬件和框架决定它是否真正进入 FP8 计算路径。

8-bit 格式需要在指数范围和尾数精度之间取舍，因此常见 FP8 格式包括两种侧重点不同的变体：

1. **E4M3 格式**（4 位指数 + 3 位尾数）：
   - **侧重：精度**。
   - **用途**：常用于前向传播和激活值，但实际选择取决于模型、量化策略和框架实现。
2. **E5M2 格式**（5 位指数 + 2 位尾数）：
   - **侧重：动态范围**。
   - **用途**：常用于梯度等需要更大动态范围的张量；实际映射也取决于训练实现和数值校准。

常见的 `HYBRID` recipe 会让前向使用 E4M3、反向使用 E5M2：前向激活更看重精度，反向梯度更看重动态范围。但这不是所有模型的固定规则，具体选择还受张量分布、缩放因子和框架 recipe 影响。FP8 通常需要根据 `amax` 历史计算 scaling factor，才能把高精度张量映射到有限的 FP8 范围。

下面的代码只验证格式策略和缩放需求，不执行 FP8 量化，也不能证明 H100 已经使用 FP8 Tensor Core；真实硬件路径需要 Transformer Engine、CUDA profiler 或固定 workload benchmark。

### Q6小验证：FP8 格式策略与缩放需求

比较 `HYBRID`、`E4M3` 和 `E5M2` 三种策略，记录前向 / 反向格式以及是否需要 scaling factor。


```python
FP8_MAX_FINITE = {'E4M3': 448.0, 'E5M2': 57_344.0}


def fp8_scale_from_amax(amax, fp8_format):
    """根据观测绝对最大值计算映射到 FP8 范围的理论缩放因子。"""
    if amax <= 0:
        raise ValueError('amax must be positive')
    if fp8_format not in FP8_MAX_FINITE:
        raise ValueError(f'unsupported FP8 format: {fp8_format}')
    return FP8_MAX_FINITE[fp8_format] / amax


def fp8_variant_policy(recipe='hybrid', forward_amax=1.0, backward_amax=1.0):
    """返回前向/反向格式及由各自 amax 得到的理论缩放因子。"""
    recipes = {
        'hybrid': {
            'forward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision'},
            'backward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range'},
        },
        'e4m3': {
            'forward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision'},
            'backward': {'format': 'E4M3', 'exp_bits': 4, 'mantissa_bits': 3, 'focus': 'precision'},
        },
        'e5m2': {
            'forward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range'},
            'backward': {'format': 'E5M2', 'exp_bits': 5, 'mantissa_bits': 2, 'focus': 'range'},
        },
    }
    if recipe not in recipes:
        raise ValueError(f'未知 FP8 recipe: {recipe}')
    stages = recipes[recipe]
    return {
        'recipe': recipe,
        'stages': stages,
        'forward_scale': fp8_scale_from_amax(forward_amax, stages['forward']['format']),
        'backward_scale': fp8_scale_from_amax(backward_amax, stages['backward']['format']),
        'scaling_required': True,
        'scaling_source': 'amax history',
        'requires_gpu_validation': True,
    }

FORWARD_AMAX = 12.0   # 可修改：前向激活的绝对最大值。
BACKWARD_AMAX = 800.0 # 可修改：反向梯度的绝对最大值。
for recipe in ['hybrid', 'e4m3', 'e5m2']:
    print(recipe, '->', fp8_variant_policy(recipe, FORWARD_AMAX, BACKWARD_AMAX))
assert fp8_variant_policy('hybrid')['stages']['forward']['format'] == 'E4M3'
assert fp8_variant_policy('hybrid')['stages']['backward']['format'] == 'E5M2'
assert fp8_variant_policy('hybrid')['scaling_required'] is True
assert fp8_variant_policy('hybrid', 2.0, 2.0)['forward_scale'] == 224.0
try:
    fp8_variant_policy('unknown')
except ValueError:
    print('✅ 非法 FP8 recipe 校验通过')
else:
    raise AssertionError('未知 FP8 recipe 应报错')

```

---

## 相关阅读
本节可以继续进入混合精度执行和训练显存预算；需要核对框架行为时查阅 PyTorch AMP 文档。
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)：查看 autocast 与 GradScaler 如何参与混合精度执行。
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)
- [12. TensorCore and Mixed Precision | Tensor Core 与混合精度](./12_TensorCore_and_Mixed_Precision.md)
