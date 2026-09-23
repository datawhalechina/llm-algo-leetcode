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

本节围绕一个问题展开：一次数据搬入能否被多个线程或多个计算步骤复用。你将依次分析 shared memory 的复用收益、访问布局、同步代价和寄存器占用，并学会判断 tile 设计何时值得采用。

**关键词：** `shared memory`, `tile`, `bank conflict`

![本节概念关系](../public/01_Hardware_Math_and_Systems/24_sram_optimization_chain.svg)



## 前置阅读

**导语：** 先理解 shared memory 如何通过数据复用减少重复搬运，再观察访问布局、同步和寄存器占用如何影响 tile 的实际收益。

- [15. CUDA Execution Model | CUDA 执行模型](./15_CUDA_Execution_Model.md)
- [16. Warp Block SharedMemory Basics | Warp、Block 与 Shared Memory 基础](./16_Warp_Block_SharedMemory_Basics.md)
- [23. TensorCore Deep Dive | Tensor Core 深度剖析](./23_TensorCore_Deep_Dive.md)

## Q1：shared memory 的收益为什么必须和同步代价一起看？

<details><summary>点击展开查看解析</summary>

shared memory 只有在数据会被多个线程或多个计算步骤重复使用时，才有机会用片上复用替换 HBM 访问。数据只读一次时，搬入和同步的成本可能超过收益；复用次数增加后，才值得继续检查 bank conflict、occupancy 和 kernel 指标。

下面的代码用教学成本模型比较复用收益与同步代价，不代表真实 kernel 的时间比例。
</details>
### Q1小验证：复用够不够

先算复用带来的收益，能不能覆盖同步和搬运。

```python
def sram_gain(reuse_times, sync_points=0, hbm_cost=10, smem_cost=2, sync_cost=3):
    """用教学成本模型比较复用收益与同步代价。"""
    if reuse_times < 0 or sync_points < 0 or min(hbm_cost, smem_cost, sync_cost) <= 0:
        raise ValueError('复用次数、同步次数不能为负，成本必须为正数')
    reuse_benefit = max(reuse_times - 1, 0) * (hbm_cost - smem_cost)
    penalty = sync_points * sync_cost
    return {'net_gain_proxy': reuse_benefit - penalty, 'worth_it_proxy': reuse_benefit > penalty}

for case in [(1, 0), (2, 1), (5, 1), (5, 3)]:
    print(case, '->', sram_gain(*case))
assert sram_gain(5, 1)['worth_it_proxy']
assert not sram_gain(1, 3)['worth_it_proxy']
try:
    sram_gain(-1)
except ValueError:
    print('✅ SRAM 成本参数校验通过')
else:
    raise AssertionError('负复用次数应报错')

```

## Q2：bank conflict 为什么会把本来并行的访问串行化？

<details><summary>点击展开查看解析</summary>

bank conflict 发生在多个线程同时访问不同地址、却落到同一个 bank 时，硬件需要分轮次服务这些请求，因此原本并行的访问可能退化为串行。判断冲突不能只看 bank 数量，还要观察 stride、padding 和对齐方式；同一地址的 broadcast 读取则不能简单按普通冲突处理。
</details>
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

## Q3：tile、布局和占用率为什么要一起设计？

<details><summary>点击展开查看解析</summary>

tile 决定工作集和并行粒度，layout 决定数据能否连续访问，occupancy 反映是否有足够的资源让这些 tile 并行驻留。只调 tile 而忽略另外两项，复用收益可能被不连续访存或 block 资源不足抵消。下面的表格先整理三者的职责，再用代码比较候选方案。

| 设计维度 | 主要作用 | 风险信号 | 观察或调整方向 |
| --- | --- | --- | --- |
| tile 大小 | 决定数据复用和并行粒度 | 过小复用不足，过大占用资源 | 调整 tile，观察复用与资源占用 |
| memory layout | 决定数据是否连续访问 | bank conflict、访存不连续 | 调整 stride、布局和对齐 |
| occupancy | 决定同时驻留的活跃线程数量 | block 资源不足、并发度下降 | 检查寄存器和 shared memory 使用 |
| 三者组合 | 决定片上优化能否持续发挥作用 | 理论复用高但吞吐不升 | 用 profiler 验证真实 kernel |

表格中的组合关系是设计判断，代码里的 score 只是 proxy；真实 occupancy、shared memory 使用量和吞吐仍需 GPU profiler 验证。
</details>
### Q3小验证：切块和布局是不是绑在一起

把 tile、layout 和占用率放在同一组候选配置中比较。

```python
def layout_tile_score(tile, contiguous=True, occupancy=1.0, reuse=1):
    """整理 tile、布局、occupancy 和复用的教学 proxy。

    design_score_proxy 只帮助比较输入条件，不计算真实 occupancy、
    shared memory 使用量或 kernel 吞吐；这些量需要 GPU profiler 验证。
    """
    if tile <= 0 or reuse < 0 or not 0 <= occupancy <= 1:
        raise ValueError('tile、reuse 和 occupancy 参数不合法')
    if not contiguous:
        return {'design_score_proxy': 0, 'reason': 'layout_fragmented'}
    score = tile * reuse * occupancy
    return {'design_score_proxy': round(score, 2), 'reason': 'good' if occupancy > 0.5 else 'low_occupancy'}

plans = [(64, True, 0.8, 1), (128, True, 0.9, 3), (128, False, 0.9, 3), (256, True, 0.4, 4)]
for plan in plans:
    print(plan, '->', layout_tile_score(*plan))
assert layout_tile_score(128, True, 0.8, 2)['design_score_proxy'] > 0
assert layout_tile_score(128, False, 0.8, 2)['design_score_proxy'] == 0

```

## Q4：寄存器 spill 为什么会成为片上优化的资源上限？

<details><summary>点击展开查看解析</summary>

tile 或临时值变大时，寄存器需求可能超过预算，编译器就会把部分值 spill 到更慢的存储层级。这样会重新增加访存成本，并可能降低 occupancy，抵消前面通过复用获得的收益。

因此，片上优化要同时检查寄存器预算、spill 风险和 block 资源。下面的 net_gain 只是教学 proxy，不是实际 spill 成本。
</details>
### Q4小验证：寄存器压力如何抵消片上复用收益

比较寄存器需求、spill 风险和 occupancy，再判断 tile 扩大后是否仍然值得。

```python
def spill_tradeoff(registers_needed, register_budget=64, reuse_gain=0, occupancy=1.0):
    """用教学 proxy 展示 register spill 如何抵消复用收益。"""
    if min(registers_needed, register_budget) <= 0 or reuse_gain < 0 or not 0 <= occupancy <= 1:
        raise ValueError('寄存器参数、reuse_gain 或 occupancy 不合法')
    overflow = max(registers_needed - register_budget, 0)
    spill_penalty = overflow * 2
    reuse_bonus = reuse_gain * 3
    occupancy_penalty = round(max(1.0 - occupancy, 0) * 4, 2)
    net = reuse_bonus - spill_penalty - occupancy_penalty
    return {'net_gain_proxy': round(net, 2), 'spill': overflow > 0, 'occupancy_penalty_proxy': occupancy_penalty}

plans = [
    ('safe_tiling', 40, 64, 1, 0.9),
    ('reuse_rich', 64, 64, 3, 0.8),
    ('spill_risk', 80, 64, 4, 0.8),
    ('spill_heavy', 96, 64, 5, 0.5),
]
for name, regs, budget, reuse, occ in plans:
    print(name, '->', spill_tradeoff(regs, budget, reuse, occ))
assert spill_tradeoff(80, 64, 4, 0.8)['spill']
assert not spill_tradeoff(40, 64, 1, 0.9)['spill']

```

## 相关阅读

**导语：** 把片上优化继续接到 Triton block、FlashAttention 和推理验证，会更容易看清 SRAM 复用怎样变成真实吞吐收益。

- [CUDA C Programming Guide：Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)：查阅 shared memory、bank conflict 和同步语义。
- [Triton](https://github.com/triton-lang/triton)：观察 block、tile 和片上数据复用如何写成 kernel。
- [18. Triton Block Model | Triton Block 模型](./18_Triton_Block_Model.md)
- [20. FlashAttention Sim | FlashAttention 模拟](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
- [34. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](../02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.md)
---