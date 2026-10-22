# 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `显存预算`, `ZeRO` | **目标人群：** 需要建立训练显存账本的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

训练显存由参数、梯度、优化器状态、激活值、通信缓冲区和临时工作区共同组成。不同状态的复制方式与生命周期不同，因此峰值不能只按参数量估算。

本节建立“训练状态 → 单卡账本 → DDP 复制 → ZeRO 分片 → 容量预算”的关系。完成后，你应能估算各类状态的理论占用，并判断多卡分片后仍需实测哪些峰值对象。

**关键词：** `VRAM`, `ZeRO`, `AdamW`

![本节概念关系](../public/01_Hardware_Math_and_Systems/06_vram_zero_map.svg)

---

## 前置阅读

**导语：** 先回顾 dtype 的字节数、参数量和通信拓扑，再用同一套参数与 optimizer 假设比较 DDP、ZeRO-1/2/3 的单卡训练状态。

- [Part 01 · 01 数据类型与精度](./01_Data_Types_and_Precision.md)
- [Part 01 · 02 参数量与 FLOPs](./02_LLM_Params_and_FLOPs.md)
- [Part 01 · 05 通信拓扑](./05_Communication_Topologies.md)

---
## Q1：DDP 下每张卡保存哪些训练状态？


在 DDP 下，每张卡都保存完整模型、完整梯度和完整优化器状态。下面用 BF16 + Adam、7B 参数作为示例，把每参数字节数和理论占用放在同一张账本表中；Adam 的状态字节数取教学估算值，若实现另存 FP32 master weights、使用不同 optimizer 或改变参数 / 梯度 dtype，必须重新计算。

| 训练状态 | 每参数字节数（BF16 + Adam 示例） | 7B 理论占用 | DDP 是否完整驻留 |
| --- | ---: | ---: | --- |
| 参数 | 2 bytes | 14 GB | 是 |
| 梯度 | 2 bytes | 14 GB | 是 |
| Adam 状态 | 12 bytes | 84 GB | 是 |
| 合计 | 16 bytes | 112 GB | 是 |

这张表只覆盖参数、梯度和 optimizer state，不包含 activation、通信缓冲、临时 workspace、显存碎片或框架的 reserved memory。代码验证还应检查：改变 dtype / optimizer 后账本会变化，且三类状态仍被分别记录。


### Q1小验证：拆分 DDP 单卡训练状态账本

```python
def training_state_breakdown(num_params_b, model_dtype='fp16', optimizer='adam'):
    """返回十进制 GB 的训练状态理论账本。

    结果只覆盖参数、梯度和优化器状态，不模拟激活、通信缓冲、workspace
    或 allocator reserve。
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
    """返回 DDP 单卡训练状态账本总量；不包含 activation 和 workspace。"""
    return training_state_breakdown(num_params_b, model_dtype, optimizer)['training_state_gb']

ledger = training_state_breakdown(7, 'bf16', 'adam')
print('7B parameters, BF16 + Adam theoretical training-state ledger (decimal GB):')
for name, value in ledger.items():
    print(f'  {name}: {value:.1f}')
```


```python
def test_calculate_ddp_memory():
    result = calculate_ddp_memory(7, 'fp16', 'adam')
    assert result == 112, f"错误：期望 112 GB，实际 {result} GB"

    result = calculate_ddp_memory(7, 'fp16', 'sgd')
    assert result == 56, f"错误：期望 56 GB，实际 {result} GB"

    bf16_ledger = training_state_breakdown(7, 'bf16', 'adam')
    assert bf16_ledger['parameters_gb'] == 14
    assert bf16_ledger['optimizer_state_gb'] == 84
    assert training_state_breakdown(7, 'fp32', 'adam')['training_state_gb'] == 140
    assert training_state_breakdown(7, 'bf16', 'sgd')['training_state_gb'] == 56
    try:
        training_state_breakdown(7, 'int8', 'adam')
    except ValueError:
        pass
    else:
        raise AssertionError('非法 dtype 应该被拒绝')
    print("✅ DDP 显存函数测试通过！")

test_calculate_ddp_memory()
```

## Q2：ZeRO 如何分摊训练状态？


ZeRO 的核心思想是把训练状态分摊到多张 GPU 上。先看每个 stage 切分哪些状态，再用 `calculate_zero_memory()` 计算理想单卡占用；这里假设状态可以均匀切分，表格不包含 activation、通信 buffer、workspace、切分粒度和 allocator reserve。

| ZeRO 阶段 | 分片状态 | 仍完整驻留的状态 | 理想单卡账本 | 主要代价 |
| --- | --- | --- | --- | --- |
| DDP / ZeRO-0 | 无 | 参数、梯度、优化器状态 | 全量状态 | 显存占用高 |
| ZeRO-1 | 优化器状态 | 参数、梯度 | 参数 + 梯度 + 优化器分片 | 优化器状态通信 |
| ZeRO-2 | 优化器状态、梯度 | 参数 | 参数 + 梯度分片 + 优化器分片 | 梯度同步和重建 |
| ZeRO-3 | 参数、梯度、优化器状态 | 无完整训练状态 | 三类状态都按卡分摊 | 参数重建、通信和调度复杂度更高 |

选择 stage 时，先问清楚要分摊的是 optimizer state、gradient 还是 parameter；这三个对象的分片范围不同，通信与实现复杂度也不同。


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
    result = calculate_zero_memory(7, 1, 8, 'fp16', 'adam')
    assert abs(result - 38.5) < 1e-9, f"错误：ZeRO-1 应该是 38.5 GB，实际 {result} GB"

    result = calculate_zero_memory(7, 2, 8, 'fp16', 'adam')
    assert abs(result - 26.25) < 1e-9, f"错误：ZeRO-2 应该是 26.25 GB，实际 {result} GB"

    result = calculate_zero_memory(7, 3, 8, 'fp16', 'adam')
    assert abs(result - 14) < 1e-9, f"错误：ZeRO-3 应该是 14 GB，实际 {result} GB"

    assert calculate_zero_memory(7, 'ddp', 8, 'fp16', 'adam') > calculate_zero_memory(7, 3, 8, 'fp16', 'adam')
    assert calculate_zero_memory(7, 1, 4, 'fp32', 'adam') > calculate_zero_memory(7, 1, 4, 'bf16', 'adam')
    try:
        calculate_zero_memory(7, 4, 8, 'fp16', 'adam')
    except ValueError:
        pass
    else:
        raise AssertionError('非法 ZeRO stage 应该被拒绝')
    print("✅ ZeRO 显存函数测试通过！")

test_calculate_zero_memory()
```

### 账本工具：根据容量反推最大模型规模

这个辅助函数把显存容量、ZeRO stage 和预留系数转换为最大可训练参数量，供后面的策略决策表调用。

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
    result = max_trainable_params(80, 8, 'ddp', 0.2)
    assert abs(result - 4) < 1e-9, f"错误：DDP 应该最多训练 4B，实际 {result}B"

    result = max_trainable_params(80, 8, 3, 0.2)
    assert abs(result - 32) < 1e-9, f"错误：ZeRO-3 应该最多训练 32B，实际 {result}B"

    assert max_trainable_params(80, 8, 3, 0.0) > max_trainable_params(80, 8, 3, 0.2)
    print("✅ 最大模型反推函数测试通过！")

test_max_trainable_params()
```

## Q3：如何把 ZeRO 账本转成容量候选表，并区分理论峰值？

**问题：** 如果你手上只有 8 张标称 80GB 的 GPU，并且用一个 20% 的教学预留系数近似 activation 和通信开销，如何把 DDP、ZeRO-1、ZeRO-2、ZeRO-3 放到同一张决策表中？

请把四种状态布局放在同一个表里比较最大可训练模型规模，并登记通信代价和实现复杂度的定性风险；可以调用前面的账本函数完成规模反推。这里生成的是容量候选证据，完整并行策略筛选在第 26 节完成。


把 DDP、ZeRO-1、ZeRO-2、ZeRO-3 放在同一个表里，同时比较单卡训练状态、相对节省、可容纳参数规模，以及通信代价和工程复杂度。这里的“低 / 中 / 高”只是第一轮筛选标签：ZeRO stage 越高，通常需要更多状态分片、重建和同步。表格中的显存和模型规模都是理论账本结果，不等于真实训练可运行上限；最终应把同一模型、dtype、optimizer、batch、seq_len 和 checkpoint 配置带入 73–76 的 GPU 实验，检查 peak allocated、reserved、step time 和 OOM。



```python
gpu_memory = 80
num_gpus = 8
overhead_ratio = 0.2
def compare_zero_strategies(gpu_memory_gb, num_gpus, overhead_ratio=0.2, model_dtype='fp16', optimizer='adam'):
    """生成 DDP/ZeRO 策略表，比较容量收益与定性通信代价。

    显存数字仍是理论账本；communication_cost 和 engineering_cost
    是帮助学习者做第一轮筛选的定性标签，不是多卡 benchmark 结果。
    """
    strategies = [('DDP', 'ddp'), ('ZeRO-1', 1), ('ZeRO-2', 2), ('ZeRO-3', 3)]
    tradeoffs = {
        'DDP': ('低', '低'),
        'ZeRO-1': ('中', '中'),
        'ZeRO-2': ('中高', '中高'),
        'ZeRO-3': ('高', '高'),
    }
    baseline = calculate_zero_memory(1, 'ddp', num_gpus, model_dtype, optimizer)
    rows = []
    for name, stage in strategies:
        per_gpu_state_gb = calculate_zero_memory(1, stage, num_gpus, model_dtype, optimizer)
        rows.append({
            'evidence_level': 'theoretical_ledger',
            'strategy': name,
            'per_gpu_state_gb_per_1b': round(per_gpu_state_gb, 3),
            'memory_saving_ratio': round(1 - per_gpu_state_gb / baseline, 3),
            'max_trainable_params_b': round(max_trainable_params(gpu_memory_gb, num_gpus, stage, overhead_ratio, model_dtype, optimizer), 3),
            'communication_cost': tradeoffs[name][0],
            'engineering_cost': tradeoffs[name][1],
        })
    return rows

strategies = compare_zero_strategies(gpu_memory, num_gpus, overhead_ratio)
assert strategies[0]['strategy'] == 'DDP'
assert strategies[-1]['memory_saving_ratio'] > strategies[0]['memory_saving_ratio']

print('8 x A100 80GB 的 ZeRO 理论账本决策表（FP16 + Adam，预留 20% 显存）：')
print('证据级别：theoretical_ledger；不是 GPU 实测峰值。')
print(f"{'策略':<10} {'单卡状态 GB/1B':>16} {'节省比例':>10} {'最大参数 B':>12} {'通信代价':>10} {'工程代价':>10}")
print('-' * 76)
for row in strategies:
    print(f"{row['strategy']:<10} {row['per_gpu_state_gb_per_1b']:>16.3f} {row['memory_saving_ratio']:>10.3f} {row['max_trainable_params_b']:>12.3f} {row['communication_cost']:>10} {row['engineering_cost']:>10}")


```

---
## 相关阅读
本节可以继续接到 ZeRO/FSDP 的运行时管理与 Part 02 的状态分片机制。
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)：查看 ZeRO 不同阶段如何分片 optimizer state、梯度和参数。
- [PyTorch Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html)：了解 FSDP 的参数分片、通信和运行时管理。
- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md)

---
