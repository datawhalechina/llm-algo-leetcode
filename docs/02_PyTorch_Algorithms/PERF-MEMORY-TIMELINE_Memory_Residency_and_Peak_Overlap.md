# PERF-MEMORY-TIMELINE. Memory Residency and Peak Overlap | 显存驻留与峰值重叠

**难度：** Medium | **环境：** [CPU基础练习，GPU可选](#runtime-environment) | **标签：** `性能分析`、`显存时间线`、`对象驻留` | **目标人群：** 具备张量与训练基础的读者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

删掉一个大张量的变量，显存为什么可能没有下降？切片视图可能仍指向同一块存储；即使存储已经可复用，分配器也可能继续保留内存，供下一次申请使用。对象消失、存储可复用和内存归还设备驱动，是不同的时刻。

本节追踪对象从创建到释放的过程，按独立存储计算同一时刻占用多少字节，再找出哪些存储在峰值时同时存在。随后比较提前释放与重新计算的取舍，检查减少重叠是否真的降低设备峰值，以及增加了多少时间。

**关键词：** 对象引用、共享存储、驻留区间、活动字节、分配器预留

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 使用[BASE-CPU](../guide.md#base-cpu)，运行[Step 4的题目与测试](#mechanism-practice)。
- **框架核对：** 同样使用[BASE-CPU](../guide.md#base-cpu)，完成练习后运行[框架一致性核对](#framework-check)。
- **GPU实测：** 环境和工作负载见[Step 5](#gpu-experiment)。

## 前置阅读

先区分当前张量占用与分配器预留，再把时间线中的对象对应到训练状态。

- [Part00·18 显存分析](../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)：理解已分配与预留内存。
- [43. 训练内存账本](./43_Unified_Training_Memory_Ledger.md)：区分参数、梯度、优化器状态和激活。

## Step 1：区分对象引用、存储与分配器

张量对象描述形状、步长和存储位置，底层存储实际持有数据。视图只改变访问方式，可能与原对象共享同一块存储；这些对象也称为别名。驻留区间表示一份存储仍被持有的时间，从而把“对象多大”变成“哪些对象同时占用容量”。

最后一个对象引用消失后，CUDA流上尚未完成的使用还可能推迟复用；可复用的块也可能留在分配器缓存中。观察窗口开始前已经存在的存储要作为初始对象登记，窗口结束时仍存活的参数则可以是正常状态。

| 对象或事件 | 记录什么 | 影响什么数量 |
| --- | --- | --- |
| 对象引用 | 指向的存储与创建、释放时刻 | 一个存储可能有多个持有者 |
| 独立存储 | 容量、类别与当前引用 | 共享时只计一次字节 |
| 同刻事件顺序 | 释放与创建的实际先后 | 不同顺序可能产生不同瞬时峰值 |
| 分配器与异步工作 | 活跃分配、等待完成与缓存块 | 逻辑引用变化不一定立即改变运行时统计 |

![从引用变化到同一时刻的独立存储量](../public/02_PyTorch_Algorithms/perf_memory_timeline_overview.svg)

## Step 2：共享视图为何不重复计数字节

先找出仍被引用的独立存储，再累加容量。例如12个FP32元素占48字节，从中切出的4元素视图虽然只访问16字节，仍持有原来的48字节存储；删掉原对象而保留视图，存储仍在。克隆则创建新存储，应另外计数。

设时刻 $t$ 存活的对象集合为 $O(t)$，对象 $o$ 指向存储 $g(o)$，独立存储集合为 $S(t)$，每份存储容量为 $b_s$ 字节，活动存储总量为 $M(t)$。同一存储只分配一次类别；教学计算在最后一个引用释放后移除它，实际设备复用时点另由运行时确认。

$$
S(t)=\lbrace g(o):o\in O(t)\rbrace,\qquad
M(t)=\sum_{s\in S(t)}b_s.
$$

| 变化 | 引用怎样变化 | 独立存储量怎样变化 |
| --- | --- | --- |
| 创建独立张量 | 增加对象与存储 | 增加相应字节 |
| 创建共享视图 | 增加一个存储持有者 | 不重复增加 |
| 释放一个别名 | 减少一个持有者 | 其他引用存在时保持不变 |
| 释放最后引用 | 移除最后逻辑持有者 | 教学计数减少，实际复用还要等相关工作完成 |

## Step 3：在同一个状态中寻找峰值对象

峰值是同一时刻独立存储字节的最高值，不是各类别最大值的相加。设 $\mathcal T$ 是按实际事件顺序排列的状态序号集合，每次事件后的活动字节为 $M_k$。峰值 $M_{\mathrm{peak}}$ 与首次达到峰值的序号 $k^{\ast}$ 如下；若最高值持续一段时间，这段时间就是峰值驻留区间。

图中四类独立存储在同一时间重叠，合计100MiB。峰值时间、类别和存活对象都应从同一个状态读取；时间戳相同也要保留真实先后。提前释放一个跨越峰值的对象后，仍需扫描整个窗口，检查其他时刻是否出现新的最高值。

$$
M_{\mathrm{peak}}=\max_{k\in\mathcal T}M_k,\qquad
k^{\ast}=\min\lbrace k\in\mathcal T:M_k=M_{\mathrm{peak}}\rbrace.
$$

| 观察 | 可以确认什么 | 接着查看什么 |
| --- | --- | --- |
| 重叠减少后逻辑峰值下降 | 当前对象模型中的最高占用变小 | 设备实测峰值、输出与时间 |
| 删除引用但字节不变 | 仍可能有别名持有存储 | 其他引用与未完成的异步使用 |
| 活跃占用下降而预留不变 | 分配器可能仍缓存已释放块 | 实际分配记录与设备可用内存 |
| 各类别峰值不在同一时刻 | 类别最大值不能一起出现 | 整体峰值时的同一组对象 |

![独立存储的重叠区间与活动字节曲线](../public/02_PyTorch_Algorithms/perf_memory_peak_overlap.svg)

<a id="mechanism-practice"></a>

## Step 4：实现引用变化、存储去重与峰值归因

输入事件包含毫秒时间、对象标识、动作、存储字节和类别；可选 `storage_id` 表示共享存储，未提供时默认每个对象独占存储。窗口开始前已有的对象用起点创建记录表示；这里的`allocate/free`表示引用创建与释放；共享存储的字节与类别必须一致，不能用局部视图大小冒充整块存储。相同时间戳按输入先后处理，采集时丢失顺序则无法恢复真实瞬时峰值。

骨架负责字段校验、排序与快照装配，你将补全非法状态判断、独立存储汇总及峰值选择。测试覆盖同刻事件顺序、最后一个引用释放、重复峰值、未释放状态和非法输入。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：检查引用状态转移<br>`normalize_memory_events` | 识别重复创建和未知释放 | 时间有限、字节为正整数；同刻按输入先后；释放元数据与创建一致 | 非法字段、重复引用、未知释放、元数据冲突与同刻顺序 |
| TODO 2：按独立存储计算驻留量<br>`build_residency_timeline` | 按存储去重后计算活动字节 | 多个别名只计一次；类别与对象集合属于同一事件后状态 | 两个别名、释放一个和最后一个引用、存储重新分配 |
| TODO 3：选择同一峰值状态<br>`peak_overlap_report` | 找出最大字节和首次达到峰值的状态 | 不相加各类别独立最大值；空时间线无峰值 | 180 字节峰值、相同峰值取首次、类别求和与空图拒绝 |

```python
from collections import defaultdict
from math import isfinite
from typing import Dict, List, Sequence
```


```python
def normalize_memory_events(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """按时间和同刻原始顺序校验对象引用与底层存储的生命周期。"""
    normalized = []
    for order, event in enumerate(events):
        action = event.get("action")
        object_id = event.get("object_id")
        category = event.get("category")
        storage_id = event.get("storage_id", object_id)
        size = event.get("bytes")
        try:
            time_ms = float(event.get("time_ms", -1.0))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid event time") from error
        if (action not in {"allocate", "free"} or not isfinite(time_ms) or time_ms < 0
                or not all(isinstance(value, str) and value.strip()
                           for value in (object_id, category, storage_id))
                or not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            raise ValueError("invalid memory event")
        normalized.append({
            "time_ms": time_ms, "event_order": order, "object_id": object_id,
            "storage_id": storage_id, "action": action, "bytes": size, "category": category,
        })
    normalized.sort(key=lambda event: (event["time_ms"], event["event_order"]))
    live = {}
    for event in normalized:
        object_id = event["object_id"]
        # ==========================================
        # TODO 1：检查引用状态转移
        # ==========================================
        # duplicate_allocation = ???  # 对已存活对象重复创建引用。
        # unknown_free = ???  # 释放未存活的对象引用。
        raise NotImplementedError("TODO 1：请检查引用状态转移")
        if duplicate_allocation or unknown_free:
            raise ValueError("invalid object lifecycle transition")
        if event["action"] == "allocate":
            for item in live.values():
                if item["storage_id"] == event["storage_id"]:
                    if (item["bytes"], item["category"]) != (event["bytes"], event["category"]):
                        raise ValueError("shared storage metadata must agree")
            live[object_id] = event
        else:
            allocation = live[object_id]
            if any(event[key] != allocation[key] for key in ("storage_id", "bytes", "category")):
                raise ValueError("free metadata does not match allocation")
            del live[object_id]
    return normalized


def build_residency_timeline(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """在每次引用变化后按独立存储计量，保留对象和存储集合。"""
    events = normalize_memory_events(events)
    live = {}
    timeline = []
    for event in events:
        if event["action"] == "allocate":
            live[event["object_id"]] = event
        else:
            del live[event["object_id"]]
        # ==========================================
        # TODO 2：按独立存储计算驻留量
        # ==========================================
        # unique_storages = ???  # 按 storage_id 去重；别名共享同一底层存储。
        # active_bytes = ???  # 独立存储总字节，不能把每个别名重复相加。
        raise NotImplementedError("TODO 2：请按独立存储计算驻留量")
        category_bytes = defaultdict(int)
        for item in unique_storages.values():
            category_bytes[item["category"]] += item["bytes"]
        timeline.append({
            "time_ms": event["time_ms"], "event_order": event["event_order"],
            "active_bytes": active_bytes, "live_object_ids": sorted(live),
            "live_storage_ids": sorted(unique_storages),
            "category_bytes": dict(sorted(category_bytes.items())),
        })
    return timeline


def peak_overlap_report(timeline: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """报告第一次达到最大存储字节时的同一份事件后状态。"""
    if not timeline:
        raise ValueError("timeline must not be empty")
    # ==========================================
    # TODO 3：选择同一峰值状态
    # ==========================================
    # peak_bytes = ???  # 全时间线上的最大活动存储字节数。
    # peak = ???  # 首次达到峰值的完整事件后状态。
    raise NotImplementedError("TODO 3：请选择同一峰值状态")
    return {
        "peak_time_ms": peak["time_ms"], "peak_event_order": peak["event_order"],
        "peak_bytes": peak_bytes, "live_object_ids": list(peak["live_object_ids"]),
        "live_storage_ids": list(peak["live_storage_ids"]),
        "category_bytes": dict(peak["category_bytes"]),
    }
```

## 测试

```python
def _memory_events():
    """构造权重跨窗口驻留、输入与激活共同形成峰值的事件。"""
    return [
        {"time_ms": 0.0, "object_id": "weights", "action": "allocate", "bytes": 100, "category": "model"},
        {"time_ms": 1.0, "object_id": "input", "action": "allocate", "bytes": 20, "category": "batch"},
        {"time_ms": 2.0, "object_id": "activation", "action": "allocate", "bytes": 60, "category": "activation"},
        {"time_ms": 3.0, "object_id": "input", "action": "free", "bytes": 20, "category": "batch"},
        {"time_ms": 4.0, "object_id": "activation", "action": "free", "bytes": 60, "category": "activation"},
    ]


def _assert_memory_error(function, events):
    """要求非法生命周期明确触发 ValueError。"""
    try:
        function(events)
    except ValueError:
        return
    raise AssertionError("非法事件必须被拒绝")


def test_normalize_memory_events():
    """不同时间按时间排序，窗口结束仍可保留活跃对象。"""
    events = normalize_memory_events(list(reversed(_memory_events())))
    assert events[0]["object_id"] == "weights"
    assert events[-1]["action"] == "free"
    assert all(event["storage_id"] == event["object_id"] for event in events)


def test_invalid_memory_events():
    """覆盖非法字段、非有限时间、重复引用、未知释放与元数据冲突。"""
    allocation = _memory_events()[0]
    for change in [
        {"action": "resize"}, {"object_id": ""}, {"category": ""},
        {"bytes": 0}, {"bytes": 1.5}, {"bytes": True},
        {"time_ms": -1}, {"time_ms": float("nan")}, {"time_ms": float("inf")},
    ]:
        _assert_memory_error(normalize_memory_events, [{**allocation, **change}])
    _assert_memory_error(normalize_memory_events, [allocation, dict(allocation)])
    _assert_memory_error(normalize_memory_events, [{**allocation, "action": "free"}])
    _assert_memory_error(normalize_memory_events, [
        allocation, {**allocation, "time_ms": 1, "action": "free", "bytes": 99},
    ])
    _assert_memory_error(normalize_memory_events, [
        {**allocation, "storage_id": "shared"},
        {**allocation, "object_id": "view", "storage_id": "shared", "bytes": 50},
    ])


def test_residency_and_peak():
    """校验逐次字节、峰值组成与同一状态中的类别求和。"""
    timeline = build_residency_timeline(_memory_events())
    assert [point["active_bytes"] for point in timeline] == [100, 120, 180, 160, 100]
    peak = peak_overlap_report(timeline)
    assert peak["peak_bytes"] == 180 and peak["peak_time_ms"] == 2.0
    assert peak["live_object_ids"] == ["activation", "input", "weights"]
    assert sum(peak["category_bytes"].values()) == peak["peak_bytes"]
    assert timeline[0]["category_bytes"] == {"model": 100}


def test_same_time_event_order():
    """同刻先释放再分配与先分配再释放，具有不同的真实瞬时峰值。"""
    first = {"time_ms": 0, "object_id": "z", "action": "allocate", "bytes": 100, "category": "workspace"}
    free = {**first, "time_ms": 1, "action": "free"}
    second = {**first, "time_ms": 1, "object_id": "a"}
    reuse = build_residency_timeline([first, free, second])
    overlap = build_residency_timeline([first, second, free])
    assert [point["active_bytes"] for point in reuse] == [100, 0, 100]
    assert [point["active_bytes"] for point in overlap] == [100, 200, 100]
    assert peak_overlap_report(reuse)["peak_bytes"] == 100
    assert peak_overlap_report(overlap)["peak_bytes"] == 200


def test_shared_storage_release():
    """两个对象共享存储只计一次，最后一个引用释放后才减去字节。"""
    owner = {"time_ms": 0, "object_id": "owner", "storage_id": "s", "action": "allocate", "bytes": 100, "category": "model"}
    view = {**owner, "time_ms": 1, "object_id": "view"}
    timeline = build_residency_timeline([
        owner, view, {**owner, "time_ms": 2, "action": "free"},
        {**view, "time_ms": 3, "action": "free"},
    ])
    assert [point["active_bytes"] for point in timeline] == [100, 100, 100, 0]
    assert timeline[1]["live_object_ids"] == ["owner", "view"]
    assert timeline[1]["live_storage_ids"] == ["s"]
    assert timeline[2]["category_bytes"] == {"model": 100}
    assert timeline[-1]["live_storage_ids"] == []


def test_reallocation_after_release():
    """引用和存储全部释放后，可以重新使用标识且改变分配大小。"""
    first = _memory_events()[0]
    timeline = build_residency_timeline([
        first, {**first, "time_ms": 1, "action": "free"},
        {**first, "time_ms": 2, "bytes": 200},
    ])
    assert [point["active_bytes"] for point in timeline] == [100, 0, 200]


def test_peak_ties_and_category_maxima():
    """相同峰值选首次状态，各类别独立最大值不得拼成一个峰值。"""
    events = _memory_events() + [
        {"time_ms": 5, "object_id": "workspace", "action": "allocate", "bytes": 80, "category": "workspace"},
    ]
    timeline = build_residency_timeline(events)
    peak = peak_overlap_report(timeline)
    assert timeline[-1]["active_bytes"] == peak["peak_bytes"] == 180
    assert peak["peak_time_ms"] == 2.0
    assert "workspace" not in peak["category_bytes"]
    assert peak["category_bytes"] == {"activation": 60, "batch": 20, "model": 100}


def test_empty_memory_timeline():
    """空事件流可规范化和重建，但没有可归因的峰值状态。"""
    assert normalize_memory_events([]) == []
    assert build_residency_timeline([]) == []
    _assert_memory_error(peak_overlap_report, [])


def run_memory_timeline_tests():
    """核对事件次序、共享存储、峰值一致性和非法边界。"""
    test_normalize_memory_events()
    test_invalid_memory_events()
    test_residency_and_peak()
    test_same_time_event_order()
    test_shared_storage_release()
    test_reallocation_after_release()
    test_peak_ties_and_category_maxima()
    test_empty_memory_timeline()
    print("✅ 事件顺序、共享存储与峰值归因测试通过")


run_memory_timeline_tests()
```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难，再向下查看参考代码与解析。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
def normalize_memory_events(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """按时间和同刻原始顺序校验对象引用与底层存储的生命周期。"""
    normalized = []
    for order, event in enumerate(events):
        action = event.get("action")
        object_id = event.get("object_id")
        category = event.get("category")
        storage_id = event.get("storage_id", object_id)
        size = event.get("bytes")
        try:
            time_ms = float(event.get("time_ms", -1.0))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid event time") from error
        if (action not in {"allocate", "free"} or not isfinite(time_ms) or time_ms < 0
                or not all(isinstance(value, str) and value.strip()
                           for value in (object_id, category, storage_id))
                or not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            raise ValueError("invalid memory event")
        normalized.append({
            "time_ms": time_ms, "event_order": order, "object_id": object_id,
            "storage_id": storage_id, "action": action, "bytes": size, "category": category,
        })
    normalized.sort(key=lambda event: (event["time_ms"], event["event_order"]))
    live = {}
    for event in normalized:
        object_id = event["object_id"]
        # TODO 1：检查引用状态转移
        duplicate_allocation = event["action"] == "allocate" and object_id in live
        unknown_free = event["action"] == "free" and object_id not in live
        if duplicate_allocation or unknown_free:
            raise ValueError("invalid object lifecycle transition")
        if event["action"] == "allocate":
            for item in live.values():
                if item["storage_id"] == event["storage_id"]:
                    if (item["bytes"], item["category"]) != (event["bytes"], event["category"]):
                        raise ValueError("shared storage metadata must agree")
            live[object_id] = event
        else:
            allocation = live[object_id]
            if any(event[key] != allocation[key] for key in ("storage_id", "bytes", "category")):
                raise ValueError("free metadata does not match allocation")
            del live[object_id]
    return normalized


def build_residency_timeline(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """在每次引用变化后按独立存储计量，保留对象和存储集合。"""
    events = normalize_memory_events(events)
    live = {}
    timeline = []
    for event in events:
        if event["action"] == "allocate":
            live[event["object_id"]] = event
        else:
            del live[event["object_id"]]
        # TODO 2：按独立存储计算驻留量
        unique_storages = {item["storage_id"]: item for item in live.values()}
        active_bytes = sum(item["bytes"] for item in unique_storages.values())
        category_bytes = defaultdict(int)
        for item in unique_storages.values():
            category_bytes[item["category"]] += item["bytes"]
        timeline.append({
            "time_ms": event["time_ms"], "event_order": event["event_order"],
            "active_bytes": active_bytes, "live_object_ids": sorted(live),
            "live_storage_ids": sorted(unique_storages),
            "category_bytes": dict(sorted(category_bytes.items())),
        })
    return timeline


def peak_overlap_report(timeline: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """报告第一次达到最大存储字节时的同一份事件后状态。"""
    if not timeline:
        raise ValueError("timeline must not be empty")
    # TODO 3：选择同一峰值状态
    peak_bytes = max(point["active_bytes"] for point in timeline)
    peak = next(point for point in timeline if point["active_bytes"] == peak_bytes)
    return {
        "peak_time_ms": peak["time_ms"], "peak_event_order": peak["event_order"],
        "peak_bytes": peak_bytes, "live_object_ids": list(peak["live_object_ids"]),
        "live_storage_ids": list(peak["live_storage_ids"]),
        "category_bytes": dict(peak["category_bytes"]),
    }
```

### 解析

#### TODO 1：检查创建与释放顺序

- 同一个存活对象不能重复创建；没有创建或已经释放的对象不能再次释放。时间相同的事件按输入顺序处理，不能把释放统一移到创建前。
- 非法事件测试检查这些条件，存储元数据还要由骨架核对；同刻顺序测试检查重排是否制造假峰值。

#### TODO 2：以存储而不是对象名计数

- 按存储标识去重，再分别累加总字节与类别字节。释放一个别名时，只要另一引用仍在，整块存储还应计入。
- 测试先释放原对象、再释放最后视图，并在释放后重新分配同一标识，检查计数是否随实际持有关系变化。

#### TODO 3：读取第一次峰值的完整状态

- 找到最大活动字节后，取第一次达到它的状态，同时读取类别和存活对象，避免拼接不同时刻的数据。
- 并列峰值测试同时让各类别最大值错开出现；空输入没有可报告的峰值，应明确停止。

<a id="framework-check"></a>

## 框架一致性核对（可选）

使用真实 PyTorch CPU 张量、切片视图与克隆，读取公开存储接口，并将显式引用变化记录为 Step 4 事件流。共同输入为 12 个 FP32 元素及其中 4 元素的偏移视图；只核对引用表对应的存储计量，不测量 Python 对象回收、进程内存或 CUDA 分配器的实际释放。

| 核对对象 | Step 4 教学实现 | 成熟框架 CPU 路径 | 共同输入与通过条件 |
| --- | --- | --- | --- |
| 偏移视图共享存储 | `storage_id` 相同的引用只计一次 | 切片与 `Tensor.untyped_storage().data_ptr()` | 视图的数据起始地址偏移，但底层存储标识与原张量相同 |
| 视图元素量与存储容量 | 事件 `bytes` 使用整块存储容量 | `numel()`、`element_size()`、`untyped_storage().nbytes()` | 视图元素占 16 Byte，底层存储仍为 48 Byte |
| 移除原引用与创建克隆 | `build_residency_timeline` 保留别名，克隆使用新存储 | 引用表移除原张量，`clone()` 创建独立存储 | 视图仍能读取；事件后字节依次为 48、48、48、64、16、0 |
| 峰值对象与容量 | `peak_overlap_report` 从同一状态读取对象与字节 | 引用表同时持有视图与克隆的存储观测 | 峰值为 64 Byte，共存对象为视图与克隆 |

```python
from importlib.metadata import version

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    framework_check = {
        "status": "skipped",
        "reason": "缺少 PyTorch；按 docs/guide.md#base-cpu 准备 BASE-CPU 后运行本单元。",
    }
else:
    def check_shared_storage_framework_conformance():
        """用真实 CPU 张量的存储标识和容量核对教学驻留计量。"""
        live, events, observed_bytes = {}, [], []
        bytes_per_value = torch.empty((), dtype=torch.float32, device="cpu").element_size()

        def snapshot():
            # 观测显式引用表仍持有的存储；不是进程内存或分配器释放测量。
            storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                        for tensor in live.values()}
            observed_bytes.append(sum(storages.values()))

        def add(name, tensor):
            live[name] = tensor
            storage = tensor.untyped_storage()
            events.append({"time_ms": len(events), "action": "allocate", "object_id": name,
                           "storage_id": str(storage.data_ptr()), "bytes": storage.nbytes(),
                           "category": "activation"})
            snapshot()

        def remove(name):
            tensor = live.pop(name)
            storage = tensor.untyped_storage()
            events.append({"time_ms": len(events), "action": "free", "object_id": name,
                           "storage_id": str(storage.data_ptr()), "bytes": storage.nbytes(),
                           "category": "activation"})
            snapshot()

        add("owner", torch.arange(12, dtype=torch.float32, device="cpu"))
        add("view", live["owner"][2:6])
        owner_storage = live["owner"].untyped_storage().data_ptr()
        view_storage = live["view"].untyped_storage().data_ptr()
        view_size = live["view"].numel() * live["view"].element_size()
        view_storage_size = live["view"].untyped_storage().nbytes()
        offset_view = live["view"].data_ptr() != live["owner"].data_ptr()
        remove("owner")
        view_survives = torch.equal(live["view"], torch.arange(2, 6, dtype=torch.float32, device="cpu"))
        add("clone", live["view"].clone())
        clone_is_independent = live["clone"].untyped_storage().data_ptr() != view_storage
        remove("view")
        remove("clone")

        timeline = build_residency_timeline(events)
        peak = peak_overlap_report(timeline)
        expected = [12, 12, 12, 16, 4, 0]
        expected_bytes = [count * bytes_per_value for count in expected]
        checks = {
            "offset_view_shares_owner_storage": owner_storage == view_storage and offset_view,
            "view_elements_smaller_than_backing_storage": view_size == 4 * bytes_per_value
                                                         and view_storage_size == 12 * bytes_per_value,
            "view_survives_owner_reference_removal": view_survives,
            "clone_uses_independent_storage": clone_is_independent,
            "reference_table_bytes_match_expected": observed_bytes == expected_bytes,
            "teaching_timeline_matches_tensor_storage": [point["active_bytes"] for point in timeline]
                                                        == observed_bytes,
            "peak_state_matches_coexisting_storages": peak["peak_bytes"] == 16 * bytes_per_value
                                                       and peak["live_object_ids"] == ["clone", "view"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AssertionError(f"框架一致性核对失败：{failed}")
        return {
            "status": "passed", "check_kind": "framework_conformance",
            "versions": {"torch": version("torch")}, "checks": checks,
            "storage_bytes_by_event": observed_bytes, "peak_bytes": peak["peak_bytes"],
        }

    framework_check = check_shared_storage_framework_conformance()

framework_check
```

<a id="gpu-experiment"></a>

## Step 5：GPU 机制实验——比较驻留重叠与重算代价

真实设备上的分配器缓存、临时张量和计算时间会改变容量取舍。本实验让激活与工作区重叠，或在工作区创建前释放激活，比较峰值和重算开销。

| 学习问题 | CPU 已确认 | GPU 要确认与后续使用 |
| --- | --- | --- |
| 缩短驻留重叠能否减少真实分配峰值 | 事件顺序、共享存储去重与峰值选择；真实 CPU 张量存储关系 | 比较构造张量的活跃／预留峰值、最后一轮校验和与时间；真实别名释放与完整分配轨迹需另测 |

<a id="gpu-experiment-setup"></a>

### 5.1 环境与固定工作负载

本实验使用 `PERF-CUDA-MECHANISM`，首次运行按[性能机制GPU环境](../topic_discussion/performance_optimization/gpu_quickstart.md#perf-cuda-mechanism-quickstart)创建和预检环境。固定输入与当前登记见[实验资产表](../topic_discussion/performance_optimization/measurement_assets.md)。

| 配置层 | 必须说明的内容 | 修改规则 |
| --- | --- | --- |
| 运行环境 | 单张 CUDA GPU；`PERF-CUDA-MECHANISM`；参考显存 8 GiB | Profile 预检未通过时停止 |
| 模型 / 数据来源 | 随机构造的 FP32 常驻张量，两条路径共享本次输入，无外部数据下载 | 当前协议未固定随机种子，复测不保证输入逐元素相同 |
| 工作负载 | 8192 × 2048 张量，每个主要张量约 64 MiB | 改变形状或数据类型后使用新标签 |
| 配对变量 | 保留激活并创建平方输出 → 释放激活后克隆常驻张量、原位重算 | 输入与计算目标相同，检查活动峰值与时间的取舍 |
| 测量方法 | 预热 5 次、每条路径测量 20 次；CUDA 事件时间，分配／保留峰值，最后一轮校验和 | 同步后重置峰值；分组采样不是逐轮交替配对，校验和不等于逐元素 `allclose` |
| 结果资产 | `benchmarks/results/perf_memory_timeline/<workload_id>_<run_label>.json` | 新运行不覆盖旧结果，5.3 可独立读取 |

```python
RUN_MODE = "dry_run"  # 默认值 dry_run；改为 real_gpu 后执行真实显存峰值对照。
RUN_LABEL = "default"  # 默认值 default；正式复测使用唯一标签并保留历史结果。
ENVIRONMENT_PROFILE = "PERF-CUDA-MECHANISM"  # 运行前必须通过的环境配置 ID。
SEMANTIC_ID = "PERF-MEMORY-TIMELINE"  # 机制页稳定标识。
WORKLOAD_ID = "perf-memory-peak-overlap"  # 生命周期与峰值重叠实验的稳定标识。
WORKLOAD_VERSION = "v2"  # 默认 v2 测量协议；重叠与重算路径分别保留峰值和时间样本。
ROWS = 8192  # 默认 8192 行，正整数；增大提高存储与重算开销，改变后用新标签。
HIDDEN_SIZE = 2048  # 默认 2048 列，正整数；改变后不与旧形状合并解释。
WARMUP_STEPS = 5  # 默认预热 5 次，正整数；不计入正式样本。
REPEAT_STEPS = 20  # 默认每条路径测量 20 次，正整数；影响统计规模。
RESULT_RELATIVE_DIR = "benchmarks/results/perf_memory_timeline"  # 结果 JSON 的仓库相对目录。
```

### 5.2 执行并保存 JSON

预检通过后执行两条驻留路径，分别保存每次测量的分配峰值、保留峰值、延迟与校验和。已有同标签结果时停止，不覆盖历史数据；本单元不从 CUDA 分配器快照自动重建对象生命周期。

```python
from pathlib import Path
import json
import statistics
import sys
import subprocess

if RUN_MODE not in {"dry_run", "real_gpu"}:
    raise ValueError("RUN_MODE 只能为 dry_run 或 real_gpu")
for name, value in {"ROWS": ROWS, "HIDDEN_SIZE": HIDDEN_SIZE,
                    "WARMUP_STEPS": WARMUP_STEPS, "REPEAT_STEPS": REPEAT_STEPS}.items():
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")

if RUN_MODE == "real_gpu":
    import torch
    from tools.project_runtime import resolve_project_root, save_json_record_once

    if not RUN_LABEL or RUN_LABEL == "default" or Path(RUN_LABEL).name != RUN_LABEL:
        raise ValueError("正式运行需要不含目录分隔符的唯一 RUN_LABEL")
    project_root = resolve_project_root()
    subprocess.run([sys.executable, str(project_root / "tools/environment_preflight.py"),
                    "--profile", ENVIRONMENT_PROFILE], check=True)
    result_dir = project_root / RESULT_RELATIVE_DIR
    result_path = result_dir / f"{WORKLOAD_ID}_{RUN_LABEL}.json"
    if result_path.exists():
        raise FileExistsError("同标签结果已存在：读取已有结果，或更换 RUN_LABEL 后重测")
    result_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PERF-MEMORY-TIMELINE 真实实验需要 CUDA GPU")

    device = torch.device("cuda")
    persistent = torch.randn(ROWS, HIDDEN_SIZE, device=device)

    def run_path(recompute):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        activation = persistent * 2.0
        if recompute:
            del activation
            workspace = persistent.clone()
            workspace.mul_(2.0).square_()
        else:
            workspace = activation.square()
        checksum = workspace.sum()
        end.record()
        end.synchronize()
        sample = {
            "latency_ms": float(start.elapsed_time(end)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "checksum": float(checksum.item()),
        }
        del workspace
        return sample

    for _ in range(WARMUP_STEPS):
        run_path(False)
        run_path(True)
    overlap_samples = [run_path(False) for _ in range(REPEAT_STEPS)]
    recompute_samples = [run_path(True) for _ in range(REPEAT_STEPS)]
    quality_passed = abs(overlap_samples[-1]["checksum"] - recompute_samples[-1]["checksum"]) <= 1e-3 * abs(overlap_samples[-1]["checksum"])

    def summarize(samples):
        return {
            "latency_median_ms": statistics.median(item["latency_ms"] for item in samples),
            "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in samples),
            "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in samples),
        }

    baseline = summarize(overlap_samples)
    candidate = summarize(recompute_samples)
    comparison = {
        "allocated_reduction_ratio": 1.0 - candidate["peak_allocated_bytes"] / baseline["peak_allocated_bytes"],
        "reserved_reduction_ratio": 1.0 - candidate["peak_reserved_bytes"] / baseline["peak_reserved_bytes"],
        "latency_ratio": candidate["latency_median_ms"] / baseline["latency_median_ms"],
    }
    record = {
        "schema_version": "performance-memory-timeline-mechanism/v2",
        "profile_id": ENVIRONMENT_PROFILE,
        "semantic_id": SEMANTIC_ID,
        "run_label": RUN_LABEL,
        "framework": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda},
        "hardware": {"device": torch.cuda.get_device_name(0), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "workload": {"id": WORKLOAD_ID, "version": WORKLOAD_VERSION, "rows": ROWS, "hidden_size": HIDDEN_SIZE, "warmup_steps": WARMUP_STEPS, "repeat_steps": REPEAT_STEPS},
        "baseline": {"name": "overlap_activation_workspace", **baseline},
        "candidate": {"name": "release_then_recompute", **candidate},
        "comparison": comparison,
        "quality": {"checksum_close": bool(quality_passed)},
        "samples": {"baseline": overlap_samples, "candidate": recompute_samples},
        "failure": None,
        "evidence_level": "local_mechanism_benchmark",
        "decision": "accept_for_peak_reduction" if quality_passed and candidate["peak_allocated_bytes"] < baseline["peak_allocated_bytes"] else "tune",
    }
    save_json_record_once(result_path, record)
else:
    print("⏭️ RUN_MODE=dry_run；未执行 GPU 显存时间线实验。")
```

### 5.3 读取结果

按已有`RUN_LABEL`读取记录，核对输入与协议版本，再从原始样本重算峰值和延迟统计。结果检查与输出数值检查分别展示，原始记录保持不变。


```python
from pathlib import Path
import json
from tools.project_runtime import resolve_project_root
from tools.performance_result_review import read_performance_result

result_path = resolve_project_root() / RESULT_RELATIVE_DIR / f"{WORKLOAD_ID}_{RUN_LABEL}.json"
if not result_path.is_file():
    if RUN_MODE == "real_gpu":
        raise FileNotFoundError(f"未找到结果：{result_path}")
    memory_record = {"status": "not_run", "result_path": str(result_path)}
    print(memory_record)
else:
    memory_record, result_review = read_performance_result(
        result_path, semantic_id=SEMANTIC_ID, run_label=RUN_LABEL,
        workload_id=WORKLOAD_ID, current_version=WORKLOAD_VERSION,
    )
    print(json.dumps(result_review, ensure_ascii=False, indent=2))
    print(json.dumps({key: memory_record.get(key) for key in (
        "schema_version", "run_label", "framework", "hardware", "workload",
        "failure", "evidence_level", "decision", "quality",
    )}, ensure_ascii=False, indent=2))
    print("路径 | 延迟中位数 ms | 分配峰值 bytes | 保留峰值 bytes")
    for name in ("baseline", "candidate"):
        item = memory_record.get(name, {})
        print(name, item.get("latency_median_ms"), item.get("peak_allocated_bytes"),
              item.get("peak_reserved_bytes"))
    print("派生对比：", memory_record.get("comparison"))
    print("路径 | 轮次 | 延迟 ms | 分配峰值 bytes | 保留峰值 bytes | 校验和")
    for name, samples in memory_record.get("samples", {}).items():
        for index, sample in enumerate(samples, 1):
            print(name, index, sample.get("latency_ms"), sample.get("peak_allocated_bytes"),
                  sample.get("peak_reserved_bytes"), sample.get("checksum"))
```

### 5.4 解释证据与决策

当前登记为 [rtx5070ti_memory_v1](../../benchmarks/results/perf_memory_timeline/perf-memory-peak-overlap_rtx5070ti_memory_v1.json)，工作负载版本为 v2。原始结果无失败，最后一轮校验和接近，释放激活后重算降低了活动分配峰值，同时明显增加计算时间。

这组结果显示了活动分配峰值下降与计算时间增加之间的取舍。采用前还需补逐元素输出误差、共享别名测试和完整分配时间线。两条路径的分配器保留峰值相同；若要继续分析碎片，应查看分配块与空闲块的记录。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 数值检查 | `quality.checksum_close` 与 `failure` | 最后一轮校验和检查通过，无失败 | 累计值检查通过；采用前再逐元素比较两条路径的输出 |
| 活动分配峰值 | `peak_allocated_bytes` | 约192 → 128MiB，下降约33.3% | 限定于本次构造张量；需要真实训练收益时使用目标模型工作负载 |
| 时间代价 | 延迟中位数与 `latency_ratio` | 0.80 → 1.49 ms，变为基线约 1.86 倍 | 原始决策仅针对峰值下降；是否采用还取决于吞吐预算 |
| 分配器保留 | `peak_reserved_bytes` | 两条路径均为194MiB | 保留量未下降，设备归还与碎片情况需另查分配记录 |
## 相关阅读

将引用与峰值关系放回训练时，再检查保存、重算和搬运怎样改变反向所需状态。

- [19. 激活检查点](./19_Activation_Checkpointing.md)：追踪被丢弃与重新计算的中间结果。
- [76. 激活重算、卸载与预算比较](./76_Activation_Checkpoint_Offload_Benchmark.md)：比较实际训练峰值与额外时间。
