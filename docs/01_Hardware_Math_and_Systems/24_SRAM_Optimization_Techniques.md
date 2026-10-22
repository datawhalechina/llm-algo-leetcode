# 24. SRAM Optimization Techniques | SRAM 优化技术

**难度：** Hard | **环境：** CPU-first | **标签：** `硬件系统`, `SRAM`, `访存优化` | **目标人群：** 需要理解 GPU 片上访存代价的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

片上共享内存可以让一次数据搬入被多个线程或多个计算步骤复用，但也会引入容量、同步、访问冲突和寄存器占用约束。复用次数增加并不自动等于端到端加速。
本节建立“矩阵分块 → 共享内存复用 → 同步与冲突 → 资源准入”的关系。完成后，你应能估算减少的显存流量，并判断一种分块设计是否满足片上资源限制。

**关键词：** 共享内存、矩阵分块、存储体冲突

![本节概念关系](../public/01_Hardware_Math_and_Systems/24_sram_optimization_chain.svg)



## 前置阅读

**导语：** 先理解 shared memory 如何通过数据复用减少重复搬运，再观察访问布局、同步和寄存器占用如何影响 tile 的实际收益。

- [15. CUDA Execution Model | CUDA 执行模型](./15_CUDA_Execution_Model.md)
- [16. Warp Block SharedMemory Basics | Warp、Block 与 Shared Memory 基础](./16_Warp_Block_SharedMemory_Basics.md)
- [23. TensorCore Deep Dive | Tensor Core 深度剖析](./23_TensorCore_Deep_Dive.md)

## Q1：共享内存复用能减少多少全局读取？

同一数据块被使用多次时，可以每次从全局内存读取，也可以先搬入共享内存，再由线程复用。先假设每次全局读取都需要搬运整个数据块，暂不计缓存命中：数据块大小乘以使用次数，就是未复用方案的读取字节数；共享内存方案只搬入一次，同时增加片上写入、读取和同步。

下面分开记录各类字节数和同步次数。全局读取减少量是这个访问模型的结果，实际 HBM 流量还受缓存影响；字节数与同步次数不能直接相减得到耗时。

### Q1小验证：固定数据块，只改变使用次数

先预测使用一次和使用多次的全局读取量，再检查共享内存写入、读取与同步的增量。


```python
def reuse_traffic_ledger(tile_bytes, uses, sync_points=1):
    """计算整块重复读取模型的字节数，不包含缓存和耗时估计。"""
    for name, value in {'tile_bytes': tile_bytes, 'uses': uses}.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} 必须是正整数')
    if type(sync_points) is not int or sync_points < 0:
        raise ValueError('sync_points 必须是非负整数')
    return {
        'global_read_bytes_without_reuse': tile_bytes * uses,
        'global_read_bytes_with_reuse': tile_bytes,
        'global_read_bytes_saved': tile_bytes * (uses - 1),
        'shared_write_bytes': tile_bytes,
        'shared_read_bytes': tile_bytes * uses,
        'sync_points': sync_points,
    }


TILE_BYTES = 4096  # 可修改：每次完整使用的数据块大小，单位字节。
for uses in [1, 2, 4]:
    print(uses, '次使用 ->', reuse_traffic_ledger(TILE_BYTES, uses))
assert reuse_traffic_ledger(4096, 1)['global_read_bytes_saved'] == 0
assert reuse_traffic_ledger(4096, 4)['global_read_bytes_saved'] == 12288
assert reuse_traffic_ledger(4096, 4, 3)['sync_points'] == 3
try:
    reuse_traffic_ledger(4096, 0)
except ValueError:
    pass
else:
    raise AssertionError('使用次数为零时应报错')

```

## Q2：bank conflict 为什么会把本来并行的访问串行化？


bank conflict 发生在多个线程同时访问不同地址、却落到同一个 bank 时，硬件需要分轮次服务这些请求，因此原本并行的访问可能退化为串行。判断冲突不能只看 bank 数量，还要观察 stride、padding 和对齐方式；同一地址的 broadcast 读取则不能简单按普通冲突处理。

### Q2小验证：访问模式为什么重要

让一个 warp 的线程按不同 stride、padding 或 broadcast 方式访问 shared memory，观察它们落到哪些 bank，以及最严重的冲突度。

```python
def bank_access_report(stride, banks=32, threads=32, element_words=1, padding_words=0, access_mode='strided'):
    """报告一个 warp 的 bank 映射，不把冲突转换成虚构的耗时。

    padding_words 用于模拟改变行步长；broadcast 用于区分同地址读取
    与不同地址落入同一 bank 的情况。真实冲突还会受到数据类型、
    广播规则和具体访问指令影响，因此这里只验证映射机制。
    """
    if stride <= 0 or banks <= 0 or threads <= 0 or element_words <= 0 or padding_words < 0:
        raise ValueError('stride、banks、threads 和 element_words 必须为正数，padding_words 不能为负数')
    if access_mode not in {'strided', 'broadcast'}:
        raise ValueError("access_mode 必须是 'strided' 或 'broadcast'")
    effective_stride = stride + padding_words
    bank_ids = ([0] * threads if access_mode == 'broadcast' else
                [(lane * effective_stride * element_words) % banks for lane in range(threads)])
    counts = {bank: bank_ids.count(bank) for bank in sorted(set(bank_ids))}
    max_conflict = max(counts.values())
    return {
        'bank_ids': bank_ids,
        'unique_banks': len(counts),
        'effective_stride': effective_stride,
        'access_mode': access_mode,
        'max_conflict_degree': max_conflict,
        'serialized_risk': access_mode == 'strided' and max_conflict > 1,
    }

for stride in [1, 2, 16, 32, 33]:
    report = bank_access_report(stride)
    print(stride, '->', {k: report[k] for k in ['unique_banks', 'max_conflict_degree', 'serialized_risk']})
assert bank_access_report(1)['max_conflict_degree'] == 1
assert bank_access_report(32)['max_conflict_degree'] == 32
assert bank_access_report(2)['max_conflict_degree'] == 2
assert bank_access_report(32, padding_words=1)['max_conflict_degree'] == 1
assert bank_access_report(32, access_mode='broadcast')['serialized_risk'] is False
try:
    bank_access_report(0)
except ValueError:
    print('✅ bank 映射参数校验通过')
else:
    raise AssertionError('stride 为 0 时应报错')

```

## Q3：分块、填充和缓冲如何限制驻留块数？

矩阵乘法的一个线程块可以同时保存左矩阵的 $M\times K$ 子块和右矩阵的 $K\times N$ 子块。设每元素占 $b$ 字节，右矩阵每行额外填充 $p$ 个元素，缓冲份数为 $q$，共享内存用量为：

$$
S_{\mathrm{block}}=q\,b\,[MK+K(N+p)].
$$

这里 $M,N,K$ 是分块维度，$p$ 用于改变右矩阵的行步长，$q=2$ 表示为两组数据预留双缓冲。双缓冲只是容量条件，是否形成计算与搬运重叠还取决于执行调度。代码用假设的每 SM 预算，分别计算共享内存、线程数和块数限制，取最小值作为驻留块数上限；该上限尚未计入寄存器、分配粒度等限制。

| 设计项 | 可计算的变化 | 需要同时检查 |
|:---|:---|:---|
| 分块维度 | 左右子块的元素数与共享内存字节数 | 更大的工作集可能减少驻留块数 |
| 行填充 | 增加 $Kp$ 个元素的容量 | 存储体映射是否改善，见前一问 |
| 缓冲份数 | 共享内存用量按份数增长 | 调度是否真正重叠搬运与计算 |
| 每块线程数 | 线程预算允许的驻留块数 | 多个资源上限取最小值 |

### Q3小验证：比较单缓冲、双缓冲与行填充

固定分块维度，分别改变缓冲份数或填充量，检查共享内存字节数和驻留块数上限。


```python
def tile_resource_ledger(tile_m, tile_n, tile_k, element_bytes=4, buffers=1,
                         padding=0, threads_per_block=256,
                         smem_per_sm_bytes=65536, max_threads_per_sm=2048,
                         max_blocks_per_sm=16):
    """按给定预算估算驻留上限；预算是教学输入，不代表特定 GPU。"""
    positive = {
        'tile_m': tile_m, 'tile_n': tile_n, 'tile_k': tile_k,
        'element_bytes': element_bytes, 'buffers': buffers,
        'threads_per_block': threads_per_block,
        'smem_per_sm_bytes': smem_per_sm_bytes,
        'max_threads_per_sm': max_threads_per_sm,
        'max_blocks_per_sm': max_blocks_per_sm,
    }
    if any(type(value) is not int or value <= 0 for value in positive.values()):
        raise ValueError('维度、字节数、份数与资源预算必须是正整数')
    if type(padding) is not int or padding < 0:
        raise ValueError('padding 必须是非负整数')
    shared_bytes = buffers * element_bytes * (
        tile_m * tile_k + tile_k * (tile_n + padding)
    )
    limits = {
        'shared_memory': smem_per_sm_bytes // shared_bytes,
        'threads': max_threads_per_sm // threads_per_block,
        'blocks': max_blocks_per_sm,
    }
    resident_blocks = min(limits.values())
    return {
        'shared_bytes_per_block': shared_bytes,
        'padding_bytes_per_block': buffers * element_bytes * tile_k * padding,
        'resident_block_limits': limits,
        'resident_blocks_upper_bound': resident_blocks,
        'active_thread_fraction_upper_bound':
            resident_blocks * threads_per_block / max_threads_per_sm,
    }


for buffers, padding in [(1, 0), (2, 0), (1, 1)]:
    print(buffers, padding, '->', tile_resource_ledger(
        32, 32, 32, buffers=buffers, padding=padding
    ))
single = tile_resource_ledger(32, 32, 32)
double = tile_resource_ledger(32, 32, 32, buffers=2)
padded = tile_resource_ledger(32, 32, 32, padding=1)
assert single['shared_bytes_per_block'] == 8192
assert double['shared_bytes_per_block'] == 16384
assert single['resident_blocks_upper_bound'] == 8
assert double['resident_blocks_upper_bound'] == 4
assert padded['padding_bytes_per_block'] == 128
assert padded['resident_blocks_upper_bound'] == 7
assert tile_resource_ledger(128, 128, 128)['resident_blocks_upper_bound'] == 0

```

## Q4：寄存器需求增加后，哪些资源限制会先出现？

分块与临时值增加时，每线程寄存器需求可能上升。在给定每线程预算下，先记录需求超出量；再把每线程需求乘以每块线程数，估算一个块需要多少寄存器。每 SM 寄存器总量除以每块需求，给出另一项驻留块数上限；它与上一问的共享内存和线程上限共同约束驻留。

下面假设输入需求就是每线程分配量，暂不计寄存器分配粒度。超出所设预算说明需要检查编译结果，而不等于已经产生同等数量的溢出读写：编译器可能缩短变量生命周期、重算或把值存入本地内存。实际溢出字节数与耗时要查看编译报告及执行指标。

### Q4小验证：固定线程数，只增加寄存器需求

观察每块寄存器总量、预算超出量和驻留块数上限，再与上一问的共享内存上限取最小值。


```python
def register_resource_ledger(registers_per_thread, threads_per_block=256,
                             per_thread_budget=64, registers_per_sm=65536,
                             max_threads_per_sm=2048, max_blocks_per_sm=16):
    """按假设分配量计算寄存器约束，不推算实际溢出或加速比。"""
    values = (registers_per_thread, threads_per_block, per_thread_budget,
              registers_per_sm, max_threads_per_sm, max_blocks_per_sm)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError('寄存器数量、线程数与预算必须是正整数')
    registers_per_block = registers_per_thread * threads_per_block
    return {
        'registers_per_block': registers_per_block,
        'excess_registers_per_thread': max(
            registers_per_thread - per_thread_budget, 0
        ),
        'resident_blocks_upper_bound': min(
            registers_per_sm // registers_per_block,
            max_threads_per_sm // threads_per_block,
            max_blocks_per_sm,
        ),
    }


SHARED_LIMIT = tile_resource_ledger(32, 32, 32)['resident_blocks_upper_bound']
for registers in [32, 64, 80, 96]:
    record = register_resource_ledger(registers)
    record['combined_resident_blocks_upper_bound'] = min(
        SHARED_LIMIT, record['resident_blocks_upper_bound']
    )
    print(registers, '寄存器/线程 ->', record)
assert register_resource_ledger(32)['resident_blocks_upper_bound'] == 8
assert register_resource_ledger(64)['resident_blocks_upper_bound'] == 4
assert register_resource_ledger(80)['excess_registers_per_thread'] == 16
assert register_resource_ledger(80)['resident_blocks_upper_bound'] == 3
assert register_resource_ledger(64)['excess_registers_per_thread'] == 0

```

## 相关阅读

**导语：** 把片上复用继续连接到 Triton block 和 FlashAttention，观察 tile 设计怎样减少高带宽内存访问。

- [CUDA C Programming Guide：Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)：查阅 shared memory、bank conflict 和同步语义。
- [18. Triton Block Model | Triton Block 模型](./18_Triton_Block_Model.md)
- [20. FlashAttention Sim | FlashAttention 模拟](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
---
