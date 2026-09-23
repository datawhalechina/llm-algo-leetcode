# 43. Unified Memory Management | 统一训练内存账本
**难度：** Medium | **环境：** CPU-first | **标签：** `反向传播`, `训练内存`, `显存账本` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/43_Unified_Memory_Management.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

统一训练内存账本要解决的不是“内存很多层”这个事实，而是如何把参数、梯度、优化器状态、activation 和临时工作集放进同一张预算表。如果每一类状态都各自为政，系统很快就会出现容量够但峰值不稳、带宽够但迁移太多的问题。19 节和 42 节分别说明重算、搬运两种策略如何工作；本节把这些策略放回同一张账本，比较它们影响了哪类对象、峰值和迁移代价。

**关键词：** `working set`, `residency`, `offload`, `budget`

---

## 前置阅读

**导语：** 进入本节前，先能从显存账本读出参数、梯度、优化器状态、activation 和临时工作集的占用，再观察这些对象如何共享预算和驻留空间。本节不重新推导 checkpoint 或 offload 的实现，而是比较两类策略放入统一账本后的影响。

- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)

---

### Step 1: 先把工作集拆开

- 至少区分参数、梯度、优化器状态、activation 和临时缓冲。
- 先看谁是常驻，谁是峰值，谁可以迁移。
- 统一预算表后，才能讨论 offload 或统一调度是否值得。

![统一训练内存账本总览](../public/02_PyTorch_Algorithms/43_unified_memory_overview.svg)

### Step 2: 把内存层次写成可比较账本

![统一内存放置的权衡](../public/02_PyTorch_Algorithms/43_placement_tradeoff.svg)

- 为每类对象记录大小、驻留位置、峰值时段和是否可迁移。
- 看峰值是否来自同一时刻的叠加，而不是单块对象本身。
- 将 19、42 的策略结果映射到同一组字段，避免只比较单一显存数字。

### Step 3: 用峰值和迁移代价一起决策

- 只看峰值下降不够，还要看是否引入了过高迁移开销。
- 真正值得保留的，是峰值下降明显且步时开销可接受的方案。
- 这里的结论是账本级比较，不替代 19、42 中对重算或搬运机制的单独验证。

### Step 4: 动手实战

1. 补全 `summarize_memory_ledger`，汇总各类对象预算。
2. 补全 `plan_memory_placement`，按容量判断哪些对象需要迁移。
3. 补全 `evaluate_memory_plan`，输出是否保留统一管理方案。

### 提示

- `TODO 1` 建议按这个顺序做：`total_gb -> resident_gb -> movable_gb -> return dict`。
- `TODO 2` 先记录 `on_device / offloaded / used`，再按容量判断每个对象放在哪里。
- `TODO 3` 先算峰值下降，再结合迁移开销判断 `keep_plan`。


```python
from typing import Dict, List

```


```python
def summarize_memory_ledger(components: List[Dict[str, float]]) -> Dict[str, float]:
    """
    TODO 1: 汇总各类对象预算。
    """
    # 提示：先算 total_gb，再统计 resident_gb 和 movable_gb。
    # total_gb = ???
    # resident_gb = ???
    # movable_gb = ???
    raise NotImplementedError


def plan_memory_placement(components: List[Dict[str, float]], device_budget_gb: float) -> Dict[str, object]:
    """
    TODO 2: 按容量判断哪些对象需要迁移。
    """
    # 提示：先创建 on_device / offloaded / used，再按 resident、movable 和容量判断放置。
    # on_device = ???
    # offloaded = ???
    # used = ???
    raise NotImplementedError


def evaluate_memory_plan(baseline_peak_gb: float, planned_peak_gb: float, migration_overhead_ms: float, max_overhead_ms: float) -> Dict[str, object]:
    """
    TODO 3: 输出是否保留统一管理方案。
    """
    # 提示：先算 peak_reduction_gb，再判断 keep_plan。
    # peak_reduction_gb = ???
    # keep_plan = ???
    raise NotImplementedError

```


```python
def test_unified_memory_template():
    try:
        components = [
            {'name': 'weights', 'size_gb': 8.0, 'resident': True, 'movable': False},
            {'name': 'gradients', 'size_gb': 4.0, 'resident': True, 'movable': False},
            {'name': 'optimizer_state', 'size_gb': 4.0, 'resident': False, 'movable': True},
            {'name': 'activations', 'size_gb': 4.0, 'resident': False, 'movable': True},
        ]
        summary = summarize_memory_ledger(components)
        assert summary == {'total_gb': 20.0, 'resident_gb': 12.0, 'movable_gb': 8.0}
        placement = plan_memory_placement(components, device_budget_gb=15.0)
        assert placement['on_device'] == ['weights', 'gradients']
        assert placement['offloaded'] == ['optimizer_state', 'activations']
        decision = evaluate_memory_plan(22.0, 15.5, migration_overhead_ms=12.0, max_overhead_ms=20.0)
        assert decision['peak_reduction_gb'] == 6.5
        assert decision['keep_plan'] is True
        print('测试通过：统一内存管理模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_unified_memory_template()

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
def summarize_memory_ledger(components: List[Dict[str, float]]) -> Dict[str, float]:
    """
    TODO 1: 汇总各类对象预算。
    """
    # 提示：先算 total_gb，再统计 resident_gb 和 movable_gb。
    # total_gb = ???
    # resident_gb = ???
    # movable_gb = ???
    total_gb = sum(component.get('size_gb', 0.0) for component in components)
    resident_gb = sum(component.get('size_gb', 0.0) for component in components if component.get('resident', False))
    movable_gb = sum(component.get('size_gb', 0.0) for component in components if component.get('movable', False))
    return {'total_gb': total_gb, 'resident_gb': resident_gb, 'movable_gb': movable_gb}


def plan_memory_placement(components: List[Dict[str, float]], device_budget_gb: float) -> Dict[str, object]:
    """
    TODO 2: 按容量判断哪些对象需要迁移。
    """
    # 提示：先创建 on_device / offloaded / used，再按 resident、movable 和容量判断放置。
    # on_device = ???
    # offloaded = ???
    # used = ???
    on_device = []
    offloaded = []
    used = 0.0
    for component in components:
        name = component.get('name', 'component')
        size_gb = component.get('size_gb', 0.0)
        resident = component.get('resident', False)
        movable = component.get('movable', False)
        if resident and used + size_gb <= device_budget_gb:
            on_device.append(name)
            used += size_gb
        elif movable and used + size_gb <= device_budget_gb:
            on_device.append(name)
            used += size_gb
        else:
            offloaded.append(name)
    return {'on_device': on_device, 'offloaded': offloaded, 'device_used_gb': used}


def evaluate_memory_plan(baseline_peak_gb: float, planned_peak_gb: float, migration_overhead_ms: float, max_overhead_ms: float) -> Dict[str, object]:
    """
    TODO 3: 输出是否保留统一管理方案。
    """
    # 提示：先算 peak_reduction_gb，再判断 keep_plan。
    # peak_reduction_gb = ???
    # keep_plan = ???
    peak_reduction_gb = baseline_peak_gb - planned_peak_gb
    return {'peak_reduction_gb': peak_reduction_gb, 'keep_plan': peak_reduction_gb > 0 and migration_overhead_ms <= max_overhead_ms}

```

### 解析

**1. TODO 1：汇总各类对象预算**
- 先算 `total_gb`，再分别统计常驻对象的 `resident_gb` 和可迁移对象的 `movable_gb`。
- 这一步的目标是先把“总量、常驻量、可迁移量”写进同一张账本，避免只看单一对象做局部决策。

**2. TODO 2：按容量判断哪些对象需要迁移**
- 先建立 `on_device / offloaded / used`，再根据 `resident`、`movable` 和设备预算决定每个对象放在哪里。
- 这里先做最小放置策略，不追求最优搜索，而是先固定“预算不足时谁留下、谁迁走”的基本规则。

**3. TODO 3：输出是否保留统一管理方案**
- 先计算 `peak_reduction_gb`，再结合迁移开销判断 `keep_plan` 是否成立。
- 统一内存管理不是只看峰值下降，还必须确认迁移开销没有把收益吞掉。

**4. 这页的定位**
- 统一内存管理先做账本，再做放置，最后才谈迁移策略。
- 峰值下降和迁移开销必须一起看，否则很容易得到看似省显存、实际拖慢训练的方案。

## 相关阅读

完成统一预算表后，可以继续观察自动调优如何搜索配置，以及性能分析如何验证峰值下降和步时代价。

- [PyTorch CUDA 内存管理文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [ZeRO-Infinity 原论文：Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [算子优化 Task5：成本模型与 Profiling](../topic_discussion/operator_optimization/05_cost_model_and_profiling.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
