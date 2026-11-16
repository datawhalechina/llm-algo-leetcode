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

## Q1：位宽降低后，权重和量化元数据分别占多少空间？

权重从FP16改为INT8时，每个参数由2字节降为1字节；改为INT4时，两个参数可以打包到一个字节。下面先算纯权重载荷，再加上缩放因子等元数据。GB按十进制计算。

| 权重格式 | 每参数字节数 | 7B模型纯权重载荷 | 相对FP16 |
| --- | ---: | ---: | ---: |
| FP16 | 2 | 14 GB | 1 |
| INT8 | 1 | 7 GB | 0.5 |
| INT4 | 0.5 | 3.5 GB | 0.25 |

分组量化还会为每组保存缩放因子（scale），非对称量化通常另存零点（zero-point）。例如1024个权重采用INT4、每128个一组、每组保存2字节缩放因子，合计为512字节权重加16字节元数据，共528字节；FP16则为2048字节。减小分组后，缩放因子变多，存储也会增加。

权重载荷减少，可降低权重读取的带宽需求，并为缓存或激活腾出空间。完整执行成本还包括反量化、元数据读取、布局转换和实际内核调用，应在固定负载下测量。

### Q1机制计算器：分别计算权重载荷与元数据

先核对7B模型的纯权重容量，再用1024个权重的小例子检查分组元数据。两者分别使用GB和字节，避免把打包后的文件大小与单个张量的dtype混为一谈。


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

def quantized_weight_bytes(num_params, bitwidth=4, group_size=128, scale_bytes=2, zero_point_bytes=0):
    """返回打包权重和每组元数据的字节数；忽略文件头与额外对齐。"""
    if type(num_params) is not int or num_params < 0 or type(group_size) is not int or group_size <= 0:
        raise ValueError('参数数目须为非负整数，分组大小须为正整数')
    if bitwidth not in (4, 8) or scale_bytes <= 0 or zero_point_bytes < 0:
        raise ValueError('使用4或8位表示，以及有效的元数据字节数')
    groups = (num_params + group_size - 1) // group_size
    payload = (num_params * bitwidth + 7) // 8
    metadata = groups * (scale_bytes + zero_point_bytes)
    return {'weight_bytes': payload, 'metadata_bytes': metadata, 'total_bytes': payload + metadata}

assert quantized_weight_bytes(1024) == {'weight_bytes': 512, 'metadata_bytes': 16, 'total_bytes': 528}
assert quantized_weight_bytes(1024, group_size=32)['total_bytes'] == 576

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

print('1024个INT4权重，每128个一组：', quantized_weight_bytes(1024))

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

### Q2对照验证：观察对称量化的映射与误差

整个张量共用一个缩放因子，零点固定为0。示例支持4位和8位数值范围；量化整数均用INT8张量保存，实际INT4打包存储由Q1单独计算。缩放因子在FP32中计算，避免全零FP16输入的最小缩放值下溢。


```python
import torch


def quantize_per_tensor(x, num_bits=8):
    """模拟4位或8位对称量化；返回INT8整数张量和FP32缩放因子。"""
    if num_bits not in (4, 8):
        raise ValueError('本例支持4位和8位量化')
    source = x.to(torch.float32)
    if not torch.isfinite(source).all():
        raise ValueError('输入须为有限数值')
    qmax = 2 ** (num_bits - 1) - 1
    absmax = source.abs().max() if source.numel() else source.new_tensor(0.0)
    scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    q = torch.clamp(torch.round(source / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_per_tensor(q, scale):
    """把量化整数恢复为FP32值，用于比较数值误差。"""
    return q.to(torch.float32) * scale


def per_channel_scales(x, axis=0, num_bits=8):
    """按二维张量的行或列计算FP32缩放因子；全零通道使用1。"""
    if x.ndim != 2 or axis not in (0, 1) or x.numel() == 0:
        raise ValueError('使用非空二维张量，axis为0或1')
    if num_bits not in (4, 8):
        raise ValueError('本例支持4位和8位量化')
    source = x.to(torch.float32)
    if not torch.isfinite(source).all():
        raise ValueError('输入须为有限数值')
    qmax = 2 ** (num_bits - 1) - 1
    reduce_dims = (1,) if axis == 0 else (0,)
    absmax = source.abs().amax(dim=reduce_dims, keepdim=True)
    return torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))


def test_quantize_per_tensor():
    """核对量化整数、误差上界与全零输入。"""
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q, scale = quantize_per_tensor(x, 8)
    x_hat = dequantize_per_tensor(q, scale)
    assert q.tolist() == [-127, -64, 0, 64, 127]
    assert torch.max(torch.abs(x_hat - x)) <= scale / 2 + 1e-7
    zero_q, zero_scale = quantize_per_tensor(torch.zeros(3, dtype=torch.float16), 4)
    assert torch.all(zero_q == 0) and zero_scale.item() == 1.0
    print('量化整数：', q.tolist(), '缩放因子：', float(scale))


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

## Q4：PTQ质量下降后，何时考虑QAT？

训练后量化（PTQ）先在固定验证集上测量质量下降。若下降幅度在给定门槛内，可以保留这条候选路径；若超过门槛，则需要调整位宽、分组或校准方法，也可以尝试量化感知训练（QAT）。

QAT会增加训练时间、数据和实现成本。比较预算时，应先对当前模型估算一次试验所需时间，再与可用预算比较。例如同一验证指标允许下降0.5个百分点，PTQ实际下降0.6个百分点；即使只有8小时预算，也不能把不合格PTQ改判为合格。若这次QAT试验预计需10小时，应保留“调整量化方案或增加预算”的结论。QAT试验完成后仍要重新测量质量。

### Q4小验证：分别检查质量门槛和试验预算

质量下降与门槛使用同一指标、同一单位；下面均按百分点。训练时间来自当前试验的给定估算，而不是通用的10小时门槛。


```python
def qat_recommendation(
    ptq_drop, acceptable_drop=0.5, retrain_budget_hours=0,
    sensitivity='medium', estimated_qat_hours=None,
):
    """根据给定质量下降和试验耗时，列出PTQ候选或下一步动作。"""
    if min(ptq_drop, acceptable_drop, retrain_budget_hours) < 0:
        raise ValueError('质量下降、门槛和预算须为非负数')
    if estimated_qat_hours is not None and estimated_qat_hours <= 0:
        raise ValueError('QAT试验耗时须为正数')
    quality_met = ptq_drop <= acceptable_drop
    if quality_met:
        recommendation = 'PTQ_candidate'
    elif estimated_qat_hours is None:
        recommendation = 'estimate_QAT_cost'
    elif retrain_budget_hours >= estimated_qat_hours:
        recommendation = 'QAT_trial'
    else:
        recommendation = 'adjust_quantization_or_budget'
    return {
        'ptq_quality_met': quality_met,
        'recommendation': recommendation,
        'inspect_sensitive_layers': sensitivity == 'high',
    }

cases = [
    dict(ptq_drop=0.2, retrain_budget_hours=0),
    dict(ptq_drop=0.6, retrain_budget_hours=12, estimated_qat_hours=10),
    dict(ptq_drop=0.6, retrain_budget_hours=8, estimated_qat_hours=10),
]
for case in cases:
    print(case, '→', qat_recommendation(**case))
assert qat_recommendation(0.5)['ptq_quality_met']
assert qat_recommendation(0.6, retrain_budget_hours=8, estimated_qat_hours=10) == {
    'ptq_quality_met': False,
    'recommendation': 'adjust_quantization_or_budget',
    'inspect_sensitive_layers': False,
}
assert qat_recommendation(0.6, retrain_budget_hours=10, estimated_qat_hours=10)['recommendation'] == 'QAT_trial'

```

## Q5：从理论压缩进入执行验证，还需要检查什么？

位宽账本说明表示需要多少字节，数值对照说明给定输入的舍入误差。进入实际负载后，还要检查校准数据是否覆盖目标输入、敏感层的误差如何变化，以及后端能否直接使用低比特产物。

低比特内核可用时，继续测量反量化和完整调用的耗时；需要回退到其他实现时，则记录回退后的空间和时间。下面分别列出这些待检查条件，不把质量与硬件支持加权成一个风险分数。

### Q5小验证：把执行条件与待检查项分开

固定量化位宽，改变低比特内核是否可用，观察下一步需要核对的执行路径。


```python
def quantization_risk(bitwidth, hardware_support=True, calibration_quality=1.0, activation_sensitive=False):
    """根据给定条件列出待检查项；calibration_quality是外部给定指标。"""
    if bitwidth not in (4, 8) or not 0 <= calibration_quality <= 1:
        raise ValueError('位宽使用4或8，给定校准指标须在[0, 1]内')
    checks = ['validation_quality', 'complete_call_time', 'peak_memory']
    if bitwidth == 4:
        checks.append('packing_and_metadata')
    if not hardware_support:
        checks.append('fallback_execution_path')
    if activation_sensitive:
        checks.append('activation_error_by_layer')
    return {
        'low_bit_kernel_available': hardware_support,
        'given_calibration_quality': calibration_quality,
        'required_checks': checks,
    }

for kernel_available in (True, False):
    print(quantization_risk(4, hardware_support=kernel_available, activation_sensitive=True))
assert 'fallback_execution_path' in quantization_risk(4, hardware_support=False)['required_checks']
assert 'fallback_execution_path' not in quantization_risk(4, hardware_support=True)['required_checks']

```

---
## 相关阅读

**导语：** 学完本节后，可以沿权重表示、运行时低精度和部署验证继续学习。

- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [41. FP8 Runtime Low-Precision Paths | FP8 运行时低精度路径](../02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)
---
