# 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `显存预算`, `ZeRO` | **目标人群：** 系统性能入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

这一页用可运行代码计算 DDP、ZeRO 和训练状态账本。参数、梯度和优化器状态先单独核算；激活值、通信缓冲区、allocator reserved、kernel workspace 等完整峰值需要在真实 workload 中测量。

这一页是 `Part 01` 的显存账本基础，主要服务 `监督微调路线` 的训练预算，也连接 `显存优化专题` 中的 ZeRO、LoRA 和 QLoRA 讨论。本节是 **CPU-first；GPU 用于扩展验证**：CPU 代码可以验证参数、梯度、optimizer state 和 ZeRO 分摊公式，但不能证明 activation、通信缓冲、workspace、碎片和 reserved memory 的真实峰值。完成后，你应该能把单卡训练状态拆成参数、梯度和 optimizer state，并说明 ZeRO 改变了哪一部分驻留关系；真实峰值仍需在目标 workload 上测量。

**关键词：** `VRAM`, `ZeRO`, `AdamW`

本节主对应显存优化路线的 Task1（账本底座）。公式结果可以作为 Task3 / 73 / 76 的预算假设，并由 75 放入真实结果做预算判断；本节本身不是 Task6 的 profiling 收口。

---

## 前置阅读

**导语：** 先用混合精度和训练状态公式建立预算，再把结果用于 LoRA、QLoRA 和 ZeRO/FSDP 的方案比较。

- [05. Communication Topologies | 通信拓扑与分布式基石](./05_Communication_Topologies.md)
- [03. GPU Architecture and Memory | GPU 物理架构与内存层级](./03_GPU_Architecture_and_Memory.md)
- [Group 1C: Distributed Communication and Memory Sharing | 1C: 多卡通信与显存共享](./1C.md)

## 相关阅读

**导语：** 如果想继续把显存账本接到训练控制、低比特微调和项目选型上，可以沿这条主线继续往下看。

- [12. Gradient Accumulation | 梯度累积](../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
---
## Q1：DDP 显存计算

<details>
<summary>点击展开查看解析</summary>

在 DDP 下，每张卡都保存完整模型、完整梯度和完整优化器状态。对一种常见的混合精度 Adam 近似，模型参数和梯度按 2 bytes/parameter，优化器状态按 12 bytes/parameter，得到 **16Φ**。这不是普适常数：优化器、master weights、参数/梯度 dtype 和实现都会改变结果。

- 模型参数：`2Φ`
- 梯度：`2Φ`
- 优化器状态：`12Φ`

把这三部分相加，就得到 `16Φ` 这一条训练状态估算。它不包含 activation、通信缓冲、临时 workspace、显存碎片或框架的 reserved memory，因此不能直接当作训练峰值或 OOM 上限。

</details>
### Q1小验证：DDP 显存计算

```python
def training_state_breakdown(num_params_b, model_dtype='fp16', optimizer='adam'):
    """Return a theoretical training-state ledger in decimal GB.

    The result covers parameters, gradients and optimizer state only.
    It assumes one byte accounting per parameter and does not model
    activations, communication buffers, workspace or allocator reserve.
    """
    if num_params_b < 0:
        raise ValueError('num_params_b must be non-negative')
    try:
        model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
        optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]
    except KeyError as exc:
        raise ValueError('unsupported dtype or optimizer') from exc
    values = {
        'parameters_gb': num_params_b * model_bytes,
        'gradients_gb': num_params_b * model_bytes,
        'optimizer_state_gb': num_params_b * optimizer_bytes,
    }
    values['training_state_gb'] = sum(values.values())
    return values


def calculate_ddp_memory(num_params_b, model_dtype='fp16', optimizer='adam'):
    return training_state_breakdown(num_params_b, model_dtype, optimizer)['training_state_gb']

ledger = training_state_breakdown(7, 'bf16', 'adam')
print('7B parameters, BF16 + Adam theoretical training-state ledger (decimal GB):')
for name, value in ledger.items():
    print(f'  {name}: {value:.1f}')
```


```python
def test_calculate_ddp_memory():
    try:
        result = calculate_ddp_memory(7, 'fp16', 'adam')
        assert result == 112, f"错误：期望 112 GB，实际 {result} GB"

        result = calculate_ddp_memory(7, 'fp16', 'sgd')
        assert result == 56, f"错误：期望 56 GB，实际 {result} GB"

        print("✅ DDP 显存函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_ddp_memory()
```

## Q2：ZeRO 显存计算

<details>
<summary>点击展开查看解析</summary>

ZeRO 的核心思想是把训练状态分摊到多张 GPU 上：

- **ZeRO-1**：切分优化器状态
- **ZeRO-2**：切分优化器状态和梯度
- **ZeRO-3**：切分参数、梯度和优化器状态

因此，随着 stage 提升，单卡显存会持续下降，但通信和调度复杂度会增加。

</details>
### Q2小验证：ZeRO 显存计算

```python
def calculate_zero_memory(num_params_b, zero_stage, num_gpus, model_dtype='fp16', optimizer='adam'):
    """估算理想均匀切分下的单卡训练状态显存。

    只计算参数、梯度和优化器状态；不包含 activation、通信 buffer、
    workspace、切分粒度和 allocator reserve，因此不能直接保证不 OOM。
    """
    if num_params_b < 0 or num_gpus <= 0:
        raise ValueError('num_params_b must be non-negative and num_gpus must be positive')
    try:
        model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
        optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]
    except KeyError as exc:
        raise ValueError('unsupported dtype or optimizer') from exc
    gradient_bytes = model_bytes

    if zero_stage == 0 or zero_stage == 'ddp':
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes
    elif zero_stage == 1:
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes / num_gpus
    elif zero_stage == 2:
        bytes_per_param = model_bytes + gradient_bytes / num_gpus + optimizer_bytes / num_gpus
    elif zero_stage == 3:
        bytes_per_param = (model_bytes + gradient_bytes + optimizer_bytes) / num_gpus
    else:
        raise ValueError('zero_stage must be 0/ddp, 1, 2 or 3')

    return num_params_b * bytes_per_param
```


```python
def test_calculate_zero_memory():
    try:
        result = calculate_zero_memory(7, 1, 8, 'fp16', 'adam')
        assert abs(result - 38.5) < 1e-9, f"错误：ZeRO-1 应该是 38.5 GB，实际 {result} GB"

        result = calculate_zero_memory(7, 2, 8, 'fp16', 'adam')
        assert abs(result - 26.25) < 1e-9, f"错误：ZeRO-2 应该是 26.25 GB，实际 {result} GB"

        result = calculate_zero_memory(7, 3, 8, 'fp16', 'adam')
        assert abs(result - 14) < 1e-9, f"错误：ZeRO-3 应该是 14 GB，实际 {result} GB"

        print("✅ ZeRO 显存函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_zero_memory()
```

### Q2扩展验证：最大模型规模反推

给定 GPU 显存容量和 ZeRO stage，反推最大可训练参数量。

```python
def max_trainable_params(gpu_memory_gb, num_gpus, zero_stage, overhead_ratio=0.2, model_dtype='fp16', optimizer='adam'):
    """Estimate parameter scale after a lumped safety reserve.

    ``overhead_ratio`` is a teaching knob for activations and communication;
    it is not a measured peak-memory fraction and cannot guarantee no OOM.
    """
    if gpu_memory_gb <= 0 or num_gpus <= 0:
        raise ValueError('gpu_memory_gb and num_gpus must be positive')
    if not 0 <= overhead_ratio < 1:
        raise ValueError('overhead_ratio must be in [0, 1)')
    available_memory = gpu_memory_gb * (1 - overhead_ratio)
    try:
        model_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}[model_dtype]
        gradient_bytes = model_bytes
        optimizer_bytes = {'adam': 12, 'sgd': 4}[optimizer]
    except KeyError as exc:
        raise ValueError('unsupported dtype or optimizer') from exc

    if zero_stage == 0 or zero_stage == 'ddp':
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes
    elif zero_stage == 1:
        bytes_per_param = model_bytes + gradient_bytes + optimizer_bytes / num_gpus
    elif zero_stage == 2:
        bytes_per_param = model_bytes + gradient_bytes / num_gpus + optimizer_bytes / num_gpus
    elif zero_stage == 3:
        bytes_per_param = (model_bytes + gradient_bytes + optimizer_bytes) / num_gpus
    else:
        raise ValueError('zero_stage must be 0/ddp, 1, 2 or 3')

    return available_memory / bytes_per_param
```


```python
def test_max_trainable_params():
    try:
        result = max_trainable_params(80, 8, 'ddp', 0.2)
        assert abs(result - 4) < 1e-9, f"错误：DDP 应该最多训练 4B，实际 {result}B"

        result = max_trainable_params(80, 8, 3, 0.2)
        assert abs(result - 32) < 1e-9, f"错误：ZeRO-3 应该最多训练 32B，实际 {result}B"

        print("✅ 最大模型反推函数测试通过！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_max_trainable_params()
```

## Q3：8×80GB GPU 下的最大模型规模估算

**问题：** 如果你手上只有 8 张标称 80GB 的 GPU，并且用一个 20% 的教学预留系数近似 activation 和通信开销，DDP、ZeRO-1、ZeRO-2、ZeRO-3 的训练状态估算分别能支撑多大的模型？

请把四种策略放在同一个表里比较最大可训练模型规模。

<details>
<summary>点击展开查看解析</summary>

把 DDP、ZeRO-1、ZeRO-2、ZeRO-3 放在同一个表里，比较不同策略的理论参数规模。这个结果只表示“在当前账本和预留假设下可能容纳”，不等于真实训练可运行上限；最终仍要用目标 batch、序列长度、checkpoint、通信和优化器实现做 GPU 实测。

</details>

```python
gpu_memory = 80
num_gpus = 8
overhead_ratio = 0.2
strategies = [
    ('DDP', 'ddp'),
    ('ZeRO-1', 1),
    ('ZeRO-2', 2),
    ('ZeRO-3', 3),
]

print('8 x A100 80GB 的最大可训练模型规模（FP16 + Adam，预留 20% 显存）：')
print('-' * 78)
for name, stage in strategies:
    max_params = max_trainable_params(gpu_memory, num_gpus, stage, overhead_ratio)
    print(f"{name:<8} {max_params:>8.2f}B 参数")
```
