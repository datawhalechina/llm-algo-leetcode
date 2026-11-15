# 31. GPU Virtualization and MIG | GPU 虚拟化与 MIG

**难度：** Medium | **环境：** CPU-first | **标签：** `硬件系统`, `GPU 虚拟化`, `MIG` | **目标人群：** 硬件约束学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

同一个模型独占设备时延迟稳定，与其他任务共享后，尾延迟却可能升高。为任务划出独立资源可以减少部分争用，但也可能让一个分区空闲、另一个分区容量不足，并降低总体吞吐。

本节先区分共享与分区的资源边界，用理想容量例子检查任务能否放入，再比较给定服务指标的吞吐、尾延迟和波动。隔离既可能减少争用，也可能留下无法借用的空闲资源；完成后可在性能优化与部署专题中核对目标设备的实际表现。

**关键词：** 多租户、资源分区、MIG、尾延迟

![GPU 共享、MIG 切分与服务证据](../public/01_Hardware_Math_and_Systems/31_gpu_isolation_mig.svg)

---
## 前置阅读

先确认各阶段的设备需求，再讨论多个任务如何共享这些资源。

- [34 异构加速器与运行时约束](./34_Heterogeneous_Accelerators_and_Runtime_Constraints.md)：阶段预算、数据交接与执行条件。

## Q1：共享与分区分别改变哪些资源边界？

多租户指多个任务或使用者共用设备资源。一个任务反复访问大数组时，其他任务可能遇到缓存或显存带宽争用；把任务放到不同进程，不等于它们获得独立的计算与内存资源。NVIDIA的多进程服务（MPS）用于组织多个进程并发执行，共享的缓存、带宽和容量仍需考虑；多实例GPU（MIG）则在支持的设备上按受支持配置划分GPU实例，为计算及内存路径提供相应资源边界。两者的隔离范围见[NVIDIA并发机制比较](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html)。

先用理想等分理解容量代价：总内存12 GB分成两片，每片6 GB；任务A需要8 GB、B需要2 GB时，总需求虽只有10 GB，A仍放不进自己的分区。下面的等分计算只用于推导余量，实际MIG配置应从目标设备支持的实例规格中选择。

### Q1 资源计算：等分后每片多少资源，哪些整数单元剩余？

假设内存可以连续等分，计算单元只能按整数分配。先复现12 GB分为两片，再看80 GB扣除2 GB预留、108个计算单元分为7片时的余量；GB使用十进制单位。


```python
import math


def partition_resources(total_memory_gb, total_compute_units, slices, reserved_memory_gb=0.0):
    """计算等分切片后的每片资源与整数计算单元碎片。"""
    if (type(total_memory_gb) not in (int, float)
            or not math.isfinite(total_memory_gb) or total_memory_gb <= 0
            or type(total_compute_units) is not int or total_compute_units <= 0
            or type(slices) is not int or not 0 < slices <= total_compute_units):
        raise ValueError('resource totals and slices must be positive')
    if (type(reserved_memory_gb) not in (int, float)
            or not math.isfinite(reserved_memory_gb)
            or not 0 <= reserved_memory_gb < total_memory_gb):
        raise ValueError('reserved memory must be in [0, total_memory)')
    usable_memory_gb = total_memory_gb - reserved_memory_gb
    compute_units_per_slice = total_compute_units // slices
    return {
        'memory_gb_per_slice': usable_memory_gb / slices,
        'compute_units_per_slice': compute_units_per_slice,
        'unused_compute_units': total_compute_units - compute_units_per_slice * slices,
    }

assert partition_resources(12, 8, 2)['memory_gb_per_slice'] == 6
for slices in [2, 3, 7]:
    print(slices, 'slices ->', partition_resources(80, 108, slices, reserved_memory_gb=2))
assert partition_resources(80, 108, 7, 2)['unused_compute_units'] == 3

```

## Q2：总容量够时，为什么仍有任务放不进分区？

沿用A需8 GB、B需2 GB的例子。共享12 GB时，容量账本余下2 GB；分成两个6 GB区后，A缺2 GB，B却空闲4 GB。固定分区中的空闲空间不自动属于另一任务。给A分配8 GB、B分配4 GB时，两者在这个容量模型中都能放下，剩余2 GB在B区；这一构造安排不表示目标硬件一定提供8／4 GB的实例组合。

资源是否足够与隔离是否有效是不同判断。下面只检查显式任务—分区分配，恰好等于预算时可容纳；实际实例支持、带宽和调度表现继续由目标配置与运行记录核对。

### Q2 容量对照：共享、等分与非等分分别剩多少？

保持A／B的需求不变，只改变给定分区容量，分别读取超额与空闲量。


```python
def partition_fit(task_memory_gb: dict, partition_memory_gb: dict) -> dict:
    """检查一任务对应一分区的理想容量；单位GB，返回每区余量或超额。"""
    if set(task_memory_gb) != set(partition_memory_gb) or not task_memory_gb:
        raise ValueError("每个任务须对应一个给定分区")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (*task_memory_gb.values(), *partition_memory_gb.values())):
        raise ValueError("任务需求与分区容量须为有限非负数")
    rows = {}
    for task, required in task_memory_gb.items():
        remaining = partition_memory_gb[task] - required
        rows[task] = {"fits": remaining >= 0,
                      "free_gb": max(remaining, 0), "shortfall_gb": max(-remaining, 0)}
    return rows


tasks = {"A": 8, "B": 2}
equal = partition_fit(tasks, {"A": 6, "B": 6})
unequal = partition_fit(tasks, {"A": 8, "B": 4})
assert 12 - sum(tasks.values()) == 2
assert equal["A"] == {"fits": False, "free_gb": 0, "shortfall_gb": 2}
assert equal["B"] == {"fits": True, "free_gb": 4, "shortfall_gb": 0}
assert all(row["fits"] for row in unequal.values())
assert unequal["A"]["free_gb"] == 0 and unequal["B"]["free_gb"] == 2
print("等分分区：", equal)
print("给定非等分分区：", unequal)

```

## Q3：给定性能记录能否同时满足吞吐与尾延迟目标？

容量允许任务进入后，仍需比较同一请求集合下的服务表现。假设共享方案每秒120次请求，P50为30 ms、P99为150 ms；隔离方案每秒108次，P50为32 ms、P99为80 ms。P50和P99分别是50%与99%请求延迟不超过的分位值。给定记录中吞吐下降10%，P99下降约46.7%；这些数值用于练习比较，不由上一问的分区大小直接推导。

本例要求吞吐至少90次／秒、P99不超过100 ms、多轮P99的最大值与最小值之差不超过20 ms。服务等级目标（SLO）是对这类指标设定的要求；它来自当前任务，而非通用合格线。下面先计算变化，再检查三个条件能否同时满足。

### Q3 指标对照：先计算变化，再检查共同目标

给定隔离方案A的吞吐／P99为108／80，方案B为70／55。B的尾延迟更低，但仍可能因吞吐不足被排除；边界值与无满足方案的情况也需分别核对。


```python
def isolation_delta(shared, isolated):
    """比较共享与隔离方案的吞吐、P99 和尾延迟放大。"""
    for record in (shared, isolated):
        if any(type(record[key]) not in (int, float)
               or not math.isfinite(record[key]) or record[key] <= 0
               for key in ('throughput_rps', 'p50_ms', 'p99_ms')):
            raise ValueError('指标必须是有限正数')
        if record['p99_ms'] < record['p50_ms']:
            raise ValueError('P99 不能小于 P50')
    return {
        'throughput_change_ratio': isolated['throughput_rps'] / shared['throughput_rps'] - 1,
        'p99_reduction_ratio': 1 - isolated['p99_ms'] / shared['p99_ms'],
        'shared_tail_amplification': shared['p99_ms'] / shared['p50_ms'],
        'isolated_tail_amplification': isolated['p99_ms'] / isolated['p50_ms'],
    }

shared = {'throughput_rps': 120, 'p50_ms': 30, 'p99_ms': 150}
isolated = {'throughput_rps': 108, 'p50_ms': 32, 'p99_ms': 80}
delta = isolation_delta(shared, isolated)
assert math.isclose(delta['p99_reduction_ratio'], 7 / 15)
assert math.isclose(delta['throughput_change_ratio'], -0.1)
print(delta)


def select_isolation_plan(plans, max_p99_ms, min_throughput_rps, max_jitter_ms):
    """先应用延迟、吞吐与抖动硬门槛，再选吞吐最高的方案。"""
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (max_p99_ms, min_throughput_rps, max_jitter_ms)):
        raise ValueError('门槛必须是有限非负数')
    eligible = {}
    rejected = {}
    for name, metrics in plans.items():
        if any(type(metrics[key]) not in (int, float)
               or not math.isfinite(metrics[key]) or metrics[key] < 0
               for key in ('p99_ms', 'throughput_rps', 'jitter_ms')):
            raise ValueError('候选指标必须是有限非负数')
        if metrics['p99_ms'] == 0:
            raise ValueError('P99 必须大于零')
        reasons = []
        if metrics['p99_ms'] > max_p99_ms:
            reasons.append('p99_slo')
        if metrics['throughput_rps'] < min_throughput_rps:
            reasons.append('throughput')
        if metrics['jitter_ms'] > max_jitter_ms:
            reasons.append('jitter')
        (rejected if reasons else eligible)[name] = reasons if reasons else metrics
    selected = max(eligible, key=lambda name: eligible[name]['throughput_rps']) if eligible else None
    return {'selected': selected, 'eligible': list(eligible), 'rejected': rejected}

plans = {
    'shared': {'throughput_rps': 120, 'p99_ms': 150, 'jitter_ms': 35},
    'isolated_a': {'throughput_rps': 108, 'p99_ms': 80, 'jitter_ms': 12},
    'isolated_b': {'throughput_rps': 70, 'p99_ms': 55, 'jitter_ms': 8},
}
decision = select_isolation_plan(plans, max_p99_ms=100, min_throughput_rps=90, max_jitter_ms=20)
assert decision['selected'] == 'isolated_a'
assert select_isolation_plan(
    {'boundary': {'throughput_rps': 90, 'p99_ms': 100, 'jitter_ms': 20}},
    max_p99_ms=100, min_throughput_rps=90, max_jitter_ms=20)['selected'] == 'boundary'
assert select_isolation_plan(
    {'slow': plans['isolated_b']}, max_p99_ms=100,
    min_throughput_rps=90, max_jitter_ms=20)['selected'] is None
assert set(decision['rejected']) == {'shared', 'isolated_b'}
print(decision)

```

## 相关阅读

把容量条件与同负载服务记录接到成本比较，再按设备核对真实隔离能力。

- [33 TCO与成本](./33_TCO_and_Cost_Model.md)：用相同成功标准下的业务产出计算单位成本。
- [性能优化·测量与诊断方法](../topic_discussion/performance_optimization/methods/intro.md)：固定请求集合与重复测量范围。
- [NVIDIA MIG用户指南](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/introduction.html)：核对实例资源边界及目标设备支持的配置。
