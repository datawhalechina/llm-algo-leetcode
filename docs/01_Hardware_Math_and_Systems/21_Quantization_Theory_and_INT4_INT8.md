# 21. Quantization Theory and INT4 INT8 | 量化理论与 INT4/INT8

**难度：** Medium | **环境：** CPU-first | **标签：** `量化压缩`, `INT4/INT8`, `量化理论` | **目标人群：** 系统性能入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

量化通过有限离散值表示权重、激活值或缓存，目标是在质量门槛内降低存储、搬运或计算成本。不同对象和粒度会产生不同的量化误差与元数据代价。

本节建立“表示范围 → 缩放因子（scale）与粒度 → 误差 → 执行条件”的基础关系，为量化与压缩专题中的校准、训练和推理分支提供数值入口。

**关键词：** `INT8`、`INT4`、缩放因子

![量化对象与介入时机](../public/02_PyTorch_Algorithms/21_quantization_object_timing.svg)
## 前置阅读

**导语：** 先复习数据格式、参数量和 GPU 内存层级，再理解低比特量化如何改变存储、带宽和推理吞吐。

- [01. Data Types and Precision | 大模型的数据格式与混合精度](./01_Data_Types_and_Precision.md)
- [02. LLM Params and FLOPs | 大模型参数量与算力推导](./02_LLM_Params_and_FLOPs.md)
- [03. GPU Architecture and Memory | GPU 物理架构与内存层级](./03_GPU_Architecture_and_Memory.md)

## Q1：为什么量化能显著减少显存和带宽压力？


如果把权重从 FP16 压到 INT8，单个参数的存储从 2 Bytes 变成 1 Byte，理论上权重显存约减半；如果压到 INT4，则理论上进一步降到 0.5 Byte，权重体积还会继续下降。

下面用一个 7B 模型例子计算不同位宽下的权重存储量：

| 权重格式 | 每参数字节数 | 7B 模型权重显存 | 相对 FP16 |
| --- | --- | --- | --- |
| FP16 | 2 Bytes | 14 GB | 1x |
| INT8 | 1 Byte | 7 GB | 0.5x |
| INT4 | 0.5 Byte | 3.5 GB | 0.25x |

这对推理很重要，原因不只是“模型更小了”，还因为：
- 显存占用下降后，更大的 batch 或更长上下文更容易装进去
- 读权重时需要搬运的数据更少，HBM 带宽压力也会下降
- 在一些带宽受限的场景里，吞吐会明显改善

但要注意，量化通常不会让所有成本都按比特数线性下降：
- 激活值可能仍然保留更高精度
- 部分层会保留 FP16 / BF16 累加
- 反量化和 scale 处理也有额外开销
- INT8 / INT4 是否真的加速，还取决于硬件是否原生支持低比特矩阵运算；如果硬件没有对应 Tensor Core / MMA 支持，量化有时只会省显存，不一定省时间

所以量化的收益通常是“显存更小 + 带宽更低 + 吞吐更高”，而不是单纯的“位宽缩小了多少，速度就提升多少”。

### Q1小验证：实现模型权重显存计算函数

实现一个函数，计算给定参数量和数据格式的模型权重显存占用。

```python
def calculate_weight_memory(num_params_b, dtype):
    """
    计算模型权重的显存占用。

    Args:
        num_params_b: 参数量（单位：B，即十亿）
        dtype: 数据类型，可选 'fp16', 'bf16', 'int8', 'int4'

    Returns:
        memory_gb: 显存占用（单位：GB）
    """
    bytes_per_param = {'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}[dtype]
    return num_params_b * bytes_per_param


def test_calculate_weight_memory():
    result = calculate_weight_memory(7, 'fp16')
    assert abs(result - 14.0) < 1e-9
    result = calculate_weight_memory(7, 'int8')
    assert abs(result - 7.0) < 1e-9
    result = calculate_weight_memory(7, 'int4')
    assert abs(result - 3.5) < 1e-9
    print('✅ calculate_weight_memory tests passed')

# 运行测试
test_calculate_weight_memory()
```

### Q1扩展验证：对比不同数据格式

使用上面的函数，对比 7B 模型在不同数据格式下的权重显存占用。

```python
# 直接计算 7B 模型在不同格式下的权重显存占用
num_params = 7
dtypes = ['fp16', 'bf16', 'int8', 'int4']

print('7B 模型权重显存占用对比：')
print('-' * 40)
for dtype in dtypes:
    memory = calculate_weight_memory(num_params, dtype)
    print(f'{dtype.upper():<6} {memory:>6.1f} GB')

```

## Q2：对称量化、非对称量化、per-tensor、per-channel 有什么区别？


最常见的量化写法可以写成：

```text
q = round(x / scale) + zero_point
```

其中：
- `scale` 决定数值映射比例
- `zero_point` 决定零点是否偏移

常见组合有四种：
- **对称量化**：`zero_point = 0`，实现简单，常用于权重
- **非对称量化**：保留 `zero_point`，更适合分布偏移明显的数据
- **per-tensor**：整个张量共用一组 scale
- **per-channel**：每个通道单独一组 scale，通常精度更好，但元数据更多

经验上：
- 权重量化常常更适合 per-channel
- 激活量化常常更依赖校准数据
- 极低比特时，误差主要不来自平均值，而来自离群值和分布偏斜

为什么激活更难量化？
- 权重分布通常相对稳定，离群值更少
- 激活会随 token、层和上下文变化，分布波动更大
- 一些激活通道可能出现明显离群值，直接压到低比特时更容易失真
- 这也是为什么很多方案会做权重-激活协同处理，或者引入 SmoothQuant 这类预处理思路

这也是为什么真正落地时，量化不是“把 dtype 改小”这么简单，而是要同时决定 scale、zero point、分组粒度和累加精度。

### Q2小验证：实现最小的 per-tensor 对称量化

本题只实现一个最小分支：整个张量共用一组 scale，且 zero-point 固定为 0。非对称量化和 per-channel 量化先通过概念理解，不要求在本题完整实现。`num_bits=4` 时只模拟 4-bit 的数值范围，返回的 PyTorch Tensor 仍是 `torch.int8`，不代表真实 packed INT4 存储。


```python
import torch


def quantize_per_tensor(x, num_bits=8):
    """对张量做对称 per-tensor 量化。

    `num_bits=4` 只控制逻辑量化范围；返回值仍用 int8 Tensor 保存，未实现 bit packing。
    """
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() / qmax if x.numel() > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
    scale = torch.clamp(scale, min=1e-8)
    # 这里用 int8 承载量化整数，便于观察映射和误差；真实 INT4 需要额外 packing。
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int8)
    return q, scale


def dequantize_per_tensor(q, scale):
    """把量化整数恢复为 float32；不模拟真实 kernel 的累加路径。"""
    return q.to(torch.float32) * scale


def per_channel_scales(x, axis=0, num_bits=8):
    """返回 per-channel 对称量化的 scale，只用于观察粒度差异。"""
    if x.ndim != 2 or axis not in (0, 1):
        raise ValueError('示例只接受二维张量，axis 必须为 0 或 1')
    qmax = 2 ** (num_bits - 1) - 1
    reduce_dims = (1,) if axis == 0 else (0,)
    scale = x.abs().amax(dim=reduce_dims, keepdim=True) / qmax
    return torch.clamp(scale, min=1e-8)


def test_quantize_per_tensor():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q, scale = quantize_per_tensor(x, 8)
    x_hat = dequantize_per_tensor(q, scale)
    assert q.dtype == torch.int8 and x_hat.shape == x.shape
    print('q:', q.tolist(), 'scale:', float(scale))
    print('x_hat:', x_hat.tolist())
    print('✅ quantize_per_tensor tests passed')


test_quantize_per_tensor()

```

### Q2扩展验证：观察量化误差

比较原始张量和反量化张量的误差。


```python
# 直接观察 8-bit 和 4-bit 的量化误差差异
torch.manual_seed(0)
x = torch.randn(1024) * 2

for bits in [8, 4]:
    q, scale = quantize_per_tensor(x, bits)
    x_hat = dequantize_per_tensor(q, scale)
    mse = torch.mean((x - x_hat) ** 2).item()
    max_err = torch.max(torch.abs(x - x_hat)).item()
    print(f'{bits}-bit -> MSE={mse:.6f}, max_err={max_err:.6f}')

matrix = torch.tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
print('per-tensor scale shape:', tuple(quantize_per_tensor(matrix, 8)[1].shape))
print('per-channel scale shape:', tuple(per_channel_scales(matrix, axis=0, num_bits=8).shape))

```


```python
def test_quantization_practice():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q8, s8 = quantize_per_tensor(x, 8)
    x8 = dequantize_per_tensor(q8, s8)
    q4, s4 = quantize_per_tensor(x, 4)
    x4 = dequantize_per_tensor(q4, s4)
    assert q8.dtype == torch.int8 and q4.dtype == torch.int8
    assert x8.shape == x.shape and x4.shape == x.shape
    assert torch.mean((x - x4).abs()) >= torch.mean((x - x8).abs())
    print('✅ 21 Quantization tests passed')

test_quantization_practice()
```

## Q3：量化对象、处理时机和交付形式为什么会影响部署？

量化方案不能只按 INT4、INT8 的位宽来命名，还要同时看它改变了哪类状态、在哪个阶段发生，以及最终由哪个 backend 解释量化产物。可以沿着下面的链条判断：

| 量化对象 | 处理时机 | 典型产物 | 首先验证什么 |
|---|---|---|---|
| 权重 | 部署前 | GPTQ / AWQ / GGUF 文件 | 是否能加载、权重显存和质量 |
| 权重与激活 | 训练中或运行时 | QAT 模型 / FP8 路径 | kernel、数值稳定性和吞吐 |
| KV Cache | 请求执行中 | 低比特 Cache 表示 | 上下文容量、并发、延迟和质量 |

处理时机决定校准或训练成本，交付形式决定 backend 兼容性；因此“显存变小”只是候选收益，是否值得仍要回到固定 workload 做质量和性能验证。可以把判断顺序记成：量化对象决定影响哪类成本，处理时机决定如何获得量化参数，交付形式决定谁来解释量化产物。

这里的 PTQ 表示训练完成后的量化，QAT 表示训练或微调过程中模拟量化误差；GPTQ 和 AWQ 是常见的权重校准路线，GGUF 是文件格式与交付封装，不等同于一种通用量化算法。

### Q3小验证：按处理时机和交付形式整理量化方法

把 PTQ、QAT、GPTQ、AWQ 和 GGUF 放回各自的层级，区分量化发生的时机、误差处理方法和最终交付形式。


```python
def quantization_method_catalog():
    """返回量化路径元数据，不代表具体 backend 的完整配置。"""
    return {
        'PTQ': {'stage': '训练完成后', 'artifact': '量化权重', 'backend': '取决于部署引擎'},
        'QAT': {'stage': '训练或微调中', 'artifact': '量化感知模型', 'backend': '取决于部署引擎'},
        'GPTQ': {'stage': '部署前校准', 'artifact': 'GPTQ 权重', 'backend': '支持 GPTQ 的引擎'},
        'AWQ': {'stage': '部署前校准', 'artifact': 'AWQ 权重', 'backend': '支持 AWQ 的引擎'},
        'GGUF': {'stage': '模型交付', 'artifact': '文件格式', 'backend': 'llama.cpp 等'},
    }


def select_quantization_path(method: str) -> dict:
    """根据方法名返回后续实验应关注的阶段、产物和 backend。"""
    catalog = quantization_method_catalog()
    if method not in catalog:
        raise ValueError(f'未知量化方法：{method}，可选值：{sorted(catalog)}')
    return {'method': method, **catalog[method]}


for method in ['PTQ', 'GPTQ', 'AWQ', 'GGUF']:
    print(select_quantization_path(method))

assert select_quantization_path('GGUF')['backend'] == 'llama.cpp 等'

```

## Q4：量化什么时候要考虑 QAT？


QAT（Quantization-Aware Training）不是第一选择，但它在以下场景里很有价值：

- PTQ 之后精度损失过大，比如困惑度或任务指标下降明显
- 模型本身对低比特特别敏感，尤其是较小模型或生成类任务
- 你有足够的训练数据和算力，能够接受再训练或微调成本

一句话判断：
- 如果目标是“快速落地”，先用 PTQ
- 如果目标是“把低比特精度尽量拉回来”，再考虑 QAT

QAT 的代价是训练流程更复杂、成本更高，但它能把量化误差直接纳入训练过程，是 PTQ 之外的重要补救路线。下面的判断只用于组织实验，不是自动证明 QAT 一定优于 PTQ；最终仍要比较固定验证集上的质量和训练成本。

### Q4小验证：什么时候该考虑 QAT？

把量化误差和训练预算放在一起，判断是否需要从 PTQ 转向 QAT。


```python
def qat_recommendation(ptq_drop, acceptable_drop=0.5, retrain_budget_hours=0, sensitivity='medium'):
    score = 0
    if ptq_drop > acceptable_drop:
        score += 2
    if sensitivity == 'high':
        score += 1
    if retrain_budget_hours >= 10:
        score += 1
    if ptq_drop > acceptable_drop and retrain_budget_hours >= 10:
        recommendation = 'QAT'
    else:
        recommendation = 'PTQ'
    return {
        'recommendation': recommendation,
        'risk_score': score,
        'ptq_drop': ptq_drop,
        'acceptable_drop': acceptable_drop,
    }

cases = [
    (0.2, 0.5, 0, 'low'),
    (0.8, 0.5, 12, 'high'),
    (0.6, 0.5, 8, 'high'),
]
for case in cases:
    print(case, '->', qat_recommendation(*case))
print('QAT is worth considering when PTQ drop is too large and retraining budget exists')

```

## Q5：为什么理论压缩不能直接证明低比特方案可用？


位宽账本只能给出理论存储下界。要把低比特表示交给后续专题验证，还需要同时记录四类条件：

- **“INT4 一定比 INT8 好”**
  不对。比特更低并不自动更优，误差和硬件支持都可能让 INT4 更难用。

- **“量化只是改一下 dtype”**
  不对。真正的量化会涉及校准、分组、反量化、累加精度和 kernel 支持。

- **“量化一定不影响效果”**
  不对。不同层、不同通道、不同模型对低比特的容忍度差别很大。

- **“量化只影响权重”**
  也不完整。推理时真正卡住性能的常常还包括激活、缓存和带宽。

因此，本节只整理量化候选的数值与执行风险；质量、真实显存、执行路径和吞吐必须在量化与压缩专题的固定工作负载中验证。下面的风险函数只是教学用检查表，不是硬件性能或模型质量的预测器。

### Q5小验证：量化配置里最常见的问题

看看哪些配置更容易踩坑，而不是把量化简单理解成“降 dtype”。


```python
def quantization_risk(bitwidth, hardware_support=True, calibration_quality=1.0, activation_sensitive=False):
    """用可解释条件做量化风险的教学筛查，不预测真实质量或吞吐。"""
    score = 0
    if bitwidth <= 4:
        score += 2
    if not hardware_support:
        score += 2
    if calibration_quality < 0.7:
        score += 1
    if activation_sensitive:
        score += 1
    if score >= 4:
        level = 'high'
    elif score >= 2:
        level = 'medium'
    else:
        level = 'low'
    return {
        'risk_level': level,
        'risk_score': score,
        'bitwidth': bitwidth,
        'evidence_level': 'teaching_checklist',
    }

cases = [
    (8, True, 0.9, False),
    (4, True, 0.8, True),
    (4, False, 0.6, True),
]
for case in cases:
    print(case, '->', quantization_risk(*case))
print('quantization fails when bitwidth, calibration, and hardware support are all under pressure')

```

---
## 相关阅读

**导语：** 学完本节后，可以沿权重表示、运行时低精度和部署验证继续学习。

- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [41. FP8 Runtime Low-Precision Paths | FP8 运行时低精度路径](../02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)
---
