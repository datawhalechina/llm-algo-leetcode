# 46. Communication Profiling with NCCL | NCCL 通信剖析
**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `NCCL`, `性能剖析` | **目标人群：** 并行通信学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

ZeRO、Pipeline Parallelism 和 Tensor Parallelism 都会让多个 rank 协同完成一次训练或推理。随后需要回答的便是：通信究竟花在了哪里，哪些时间真正暴露在关键路径上。

本节从通信时间线出发，学习如何区分计算与 collective、计算有效 overlap，并用暴露通信时间定位应优先处理的热点。你将把零散的事件记录收敛为一条可解释的通信瓶颈分析链路。

**关键词：** `nccl`, `all-reduce`, `communication profiling`, `overlap`

---

## 前置阅读

**导语：** 先了解常见 collective 如何在多卡间传输数据；阅读本节时，再把这些 collective 放回 compute—communication 时间线中观察。

- [P1: 05. Communication Topologies | 通信拓扑与分布式基石](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)
- [P1: 20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)
- [29. Tensor Parallelism Sim | Tensor 并行模拟（可选回顾）](./29_Tensor_Parallelism_Sim.md)

---
### Step 1：通信为什么会进入关键路径

多卡训练或推理中，单独观察 GPU 利用率不能判断系统是否被通信限制：某些 rank 可能在等待 collective 完成，另一些 rank 仍在计算。把计算与 collective 放到同一时间线上，才能判断通信是否正在延长一个训练 step 或请求的完成时间。

通信剖析需要建立三类证据：

- **通信对象**：all-reduce、broadcast、reduce-scatter，还是其他 collective；
- **时间与传输量**：每类 collective 花了多久、搬了多少 bytes，以及不同 rank 是否存在差异；
- **关键路径位置**：通信与 forward / backward 等 compute 区间在时间线上如何相对出现。

在进入时间线前，先用一个统一近似解释通信成本：

```text
single_collective_time ≈ startup_latency + payload_bytes / effective_bandwidth
total_communication_time ≈ sync_count × single_collective_time
```

`startup_latency` 表示启动一次 collective 的固定成本，`effective_bandwidth` 已包含拓扑、竞争和实现效率的影响。小 payload 更容易被启动延迟主导；大 payload 更容易被带宽主导；单次通信不慢但同步频率过高，同样会累积成关键路径。

| Collective | 常见 payload | 语义 | 首先检查 |
|---|---|---|---|
| AllReduce | 梯度或部分输出 | 所有 rank 得到聚合结果 | payload、同步次数、拓扑 |
| AllGather | 参数/激活分片 | 收集各 rank 分片 | shard 大小、峰值与等待 |
| ReduceScatter | 梯度或部分和 | 聚合后保留本 rank 分片 | 分片均衡、后续依赖 |
| All-to-All | token / expert payload | 各 rank 按目的地交换数据 | 负载偏斜、跨节点流量 |

![NCCL 通信分析总览](../public/02_PyTorch_Algorithms/46_nccl_profiling_overview.svg)

### Step 2：通信时间线需要记录什么

真实 NCCL、PyTorch Profiler 或 Nsight Systems 的输出形式不同，但都能整理为统一的时间线事件。以下四类字段共同描述发生了什么、持续多久、传输了多少，以及哪些时间没有被计算隐藏。

| 证据 | 典型字段 | 用途 |
|------|----------|------|
| 计算区间 | `name`、`start`、`end` | 标出 forward、backward 等可覆盖通信的时间段 |
| 通信区间 | `collective`、`start`、`end`、`bytes` | 判断是哪种 collective 在何时搬运了多少数据 |
| 执行上下文 | `rank`、`stream`、同步点 | 辅助解释某个 rank 为什么等待或无法重叠 |
| 聚合指标 | `count`、`time`、`exposed_time` | 比较各 collective 的频率、总成本和关键路径成本 |

### Step 3：时间重叠、暴露通信时间与热点归因

一段通信若完全位于计算区间左侧或右侧，就不会重叠；其余情况存在时间交集。真正用于性能判断的不是“是否重叠”这个布尔值，而是这段交集究竟覆盖了多少通信时间。

从时间线判断通信代价时，关注四个概念：

- **通信总时长**：所有 collective 持续时间的总和；
- **有效重叠时间**：通信实际落在计算覆盖范围内的时长；
- **重叠比例**：有效重叠时间在通信总时长中的占比，用于判断可隐藏的程度；
- **暴露通信时间**：未被计算覆盖的通信时间，用于决定优先检查哪个 collective。

有效重叠时间来自通信区间与计算区间的交集，暴露通信时间则是通信中未被覆盖的剩余部分。若多个计算区间相互重叠，不能重复累计同一段通信；因此要先合并覆盖范围，再判断可隐藏时间。

![通信事件如何读时间线](../public/02_PyTorch_Algorithms/46_collective_trace.svg)

### Step 4：CPU 机制实现——从时间线到热点

题目区提供通信成本估算、事件记录和时间线导出的骨架。你需要完成四个机制判断：估算单类 collective 的理论成本、计算有效 overlap、按 collective 汇总暴露时间、从汇总结果找出优先热点。

| TODO | 函数 / 机制 | 输入与关键变量 | 输出与不变量 | 测试关注点 |
| --- | --- | --- | --- | --- |
| 1 | `estimate_collective_time`：通信成本 | `payload_bytes`、`startup_latency`、`bandwidth`、`sync_count` | 返回启动与传输成本随同步次数累积的估算值 | 小 payload、零同步、非法带宽 |
| 2 | `_overlap_duration`：有效 overlap | 通信 `start / end`、`compute_events`、交集区间 | 返回合并后的覆盖时长；同一时段不能被重复计入 | 多段 compute 覆盖同一通信区间 |
| 3 | `summarize`：collective 归因 | `events`、`by_op`、`exposed_time` | 每个 op 聚合 `count / time / bytes / exposed_time` | 同一 op 的多次事件能否正确合并 |
| 4 | `find_priority_hotspot`：热点选择 | `summary['by_op']`、各 op 暴露时间 | 返回暴露时间最大的 op 及其证据 | 热点是否由暴露时间而非总时间决定 |


```python
from dataclasses import dataclass
from typing import Any, Dict, List

```


```python
@dataclass
class CommEvent:
    op: str
    start: float
    end: float
    bytes: int
    overlap_with_compute: bool = False

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


class NCCLProfilerSim:
    """极简版 NCCL 通信 profiling 模拟器。"""

    def __init__(self):
        self.events: List[CommEvent] = []
        self.compute_events: List[Dict[str, float]] = []

    @staticmethod
    def estimate_collective_time(payload_bytes: int, startup_latency: float, bandwidth_bytes_per_s: float, sync_count: int = 1) -> float:
        """估算重复 collective 的总时间；时间单位与 `startup_latency` 一致。"""
        if payload_bytes < 0 or startup_latency < 0 or bandwidth_bytes_per_s <= 0 or sync_count < 0:
            raise ValueError("payload/latency/sync_count 必须非负，bandwidth 必须为正")
        # ==========================================
        # TODO 1: 计算启动成本与 payload / bandwidth 的传输成本，并乘同步次数。
        # 提示：先计算一次 collective 的时间，再由 sync_count 累积。
        # ==========================================
        # single_collective_time = ???
        raise NotImplementedError

    def add_compute(self, name: str, start: float, end: float) -> None:
        """记录一个 compute 区间；`name` 只用于时间线解释。"""
        self.compute_events.append({"name": name, "start": start, "end": end})

    def add_comm(self, op: str, start: float, end: float, bytes: int) -> None:
        """记录一个 collective；`bytes` 是该次 collective 的传输量。"""
        event = CommEvent(op=op, start=start, end=end, bytes=bytes)
        event.overlap_with_compute = self._has_overlap(event.start, event.end)
        self.events.append(event)

    def _has_overlap(self, start: float, end: float) -> bool:
        return any(not (end <= c["start"] or start >= c["end"]) for c in self.compute_events)

    def _overlap_duration(self, start: float, end: float) -> float:
        """返回通信区间被 compute 覆盖的有效时长，避免多段 compute 重复计数。"""
        intersections = sorted(
            (max(start, c["start"]), min(end, c["end"]))
            for c in self.compute_events
            if max(start, c["start"]) < min(end, c["end"])
        )
        if not intersections:
            return 0.0
        # ==========================================
        # TODO 2: 合并重叠的交集区间，得到有效 overlap 时长。
        # 提示：依次比较当前区间的 start 与已合并最后区间的 end；
        # 相交就扩展 end，否则开始一个新区间。
        # ==========================================
        # merged = ???
        raise NotImplementedError

    def summarize(self) -> Dict[str, Any]:
        total_comm_time = sum(e.duration for e in self.events)
        overlap_time = sum(self._overlap_duration(e.start, e.end) for e in self.events)
        by_op: Dict[str, Dict[str, float]] = {}
        for e in self.events:
            exposed_time = e.duration - self._overlap_duration(e.start, e.end)
            # ==========================================
            # TODO 3: 按 collective 汇总 count、time、bytes 和 exposed_time。
            # 提示：先用 setdefault 取得统计桶，再在同一个桶上累加四个字段；
            # exposed_time 代表该 op 未被 compute 隐藏的时间。
            # ==========================================
            # item = ???
            raise NotImplementedError
        return {
            "num_comm_events": len(self.events),
            "total_comm_time": total_comm_time,
            "overlap_time": overlap_time,
            "overlap_ratio": min(overlap_time, total_comm_time) / max(total_comm_time, 1e-8),
            "exposed_comm_time": max(total_comm_time - overlap_time, 0.0),
            "by_op": by_op,
        }

    def find_priority_hotspot(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """从按 op 汇总的证据中找出最优先处理的 exposed communication hotspot。"""
        # ==========================================
        # TODO 4: 选择 exposed_time 最大的 collective，并返回 op、exposed_time 与 bytes。
        # 提示：`summary["by_op"]` 的键是 collective 名称，值是统计桶。
        # ==========================================
        # op, item = ???
        raise NotImplementedError

    def timeline(self) -> List[Dict[str, Any]]:
        records = []
        for e in sorted(self.events, key=lambda x: (x.start, x.end, x.op)):
            records.append({"op": e.op, "start": e.start, "end": e.end, "duration": e.duration, "bytes": e.bytes, "overlap": e.overlap_with_compute})
        return records

```


```python
# 测试你的实现：成本模型、时间线、归因和边界条件分别验证。
def test_collective_cost_model():
    try:
        estimated = NCCLProfilerSim.estimate_collective_time(100, 0.1, 1000.0, sync_count=3)
        assert round(estimated, 6) == 0.6
        assert NCCLProfilerSim.estimate_collective_time(100, 0.1, 1000.0, sync_count=0) == 0.0
        try:
            NCCLProfilerSim.estimate_collective_time(100, 0.1, 0.0)
            raise AssertionError('bandwidth 为零时应拒绝估算')
        except ValueError:
            pass
    except NotImplementedError as e:
        raise NotImplementedError('请先完成 TODO 1：通信成本模型。') from e

def _build_profiler_fixture():
    profiler = NCCLProfilerSim()
    profiler.add_compute('forward', 0.0, 2.0)
    profiler.add_comm('all_reduce', 1.0, 2.5, 128 * 1024)
    profiler.add_comm('broadcast', 2.6, 3.0, 64 * 1024)
    profiler.add_compute('backward', 3.0, 5.0)
    profiler.add_comm('reduce_scatter', 3.5, 4.3, 96 * 1024)
    profiler.add_comm('all_reduce', 4.5, 5.5, 32 * 1024)
    return profiler

def test_nccl_event_timeline():
    try:
        profiler = _build_profiler_fixture()
        timeline = profiler.timeline()
        assert len(timeline) == 4
        assert timeline[0]['op'] == 'all_reduce'
        assert timeline[0]['overlap'] is True
        assert timeline[1]['overlap'] is False
    except NotImplementedError as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e

def test_nccl_summary_contract():
    try:
        summary = _build_profiler_fixture().summarize()
        assert summary['num_comm_events'] == 4
        assert round(summary['total_comm_time'], 6) == 3.7
        assert round(summary['overlap_time'], 6) == 2.3
        assert round(summary['exposed_comm_time'], 6) == 1.4
        assert summary['by_op']['all_reduce']['count'] == 2
        assert summary['by_op']['all_reduce']['bytes'] == 160 * 1024
        hotspot = _build_profiler_fixture().find_priority_hotspot(summary)
        assert hotspot['op'] == 'all_reduce'
        assert round(hotspot['exposed_time'], 6) == 1.0
    except NotImplementedError as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


def test_overlap_duration_does_not_double_count_compute():
    try:
        profiler = NCCLProfilerSim()
        profiler.add_compute('forward', 0.0, 3.0)
        profiler.add_compute('backward', 2.0, 4.0)
        assert profiler._overlap_duration(1.0, 5.0) == 3.0
        assert profiler._overlap_duration(4.0, 5.0) == 0.0
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e

test_collective_cost_model()
test_nccl_event_timeline()
test_nccl_summary_contract()
test_overlap_duration_does_not_double_count_compute()
print('✅ NCCLProfilerSim 测试通过')

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
# TODO：下面是题目区的参考实现。

@dataclass
class CommEvent:
    op: str
    start: float
    end: float
    bytes: int
    overlap_with_compute: bool = False

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


class NCCLProfilerSim:
    """极简版 NCCL 通信 profiling 模拟器。"""

    def __init__(self):
        self.events: List[CommEvent] = []
        self.compute_events: List[Dict[str, float]] = []

    @staticmethod
    def estimate_collective_time(payload_bytes: int, startup_latency: float, bandwidth_bytes_per_s: float, sync_count: int = 1) -> float:
        """估算重复 collective 的总时间；时间单位与 `startup_latency` 一致。"""
        if payload_bytes < 0 or startup_latency < 0 or bandwidth_bytes_per_s <= 0 or sync_count < 0:
            raise ValueError("payload/latency/sync_count 必须非负，bandwidth 必须为正")
        # ==========================================
        # TODO 1: 计算启动成本与 payload / bandwidth 的传输成本，并乘同步次数。
        # 提示：先计算一次 collective 的时间，再由 sync_count 累积。
        # ==========================================
        # single_collective_time = ???
        single_collective_time = startup_latency + payload_bytes / bandwidth_bytes_per_s
        return sync_count * single_collective_time

    def add_compute(self, name: str, start: float, end: float) -> None:
        """记录一个 compute 区间；`name` 只用于时间线解释。"""
        self.compute_events.append({"name": name, "start": start, "end": end})

    def add_comm(self, op: str, start: float, end: float, bytes: int) -> None:
        """记录一个 collective；`bytes` 是该次 collective 的传输量。"""
        event = CommEvent(op=op, start=start, end=end, bytes=bytes)
        event.overlap_with_compute = self._has_overlap(event.start, event.end)
        self.events.append(event)

    def _has_overlap(self, start: float, end: float) -> bool:
        return any(not (end <= c["start"] or start >= c["end"]) for c in self.compute_events)

    def _overlap_duration(self, start: float, end: float) -> float:
        """返回通信区间被 compute 覆盖的有效时长，避免多段 compute 重复计数。"""
        intersections = sorted(
            (max(start, c["start"]), min(end, c["end"]))
            for c in self.compute_events
            if max(start, c["start"]) < min(end, c["end"])
        )
        if not intersections:
            return 0.0
        # ==========================================
        # TODO 2: 合并重叠的交集区间，得到有效 overlap 时长。
        # 提示：依次比较当前区间的 start 与已合并最后区间的 end；
        # 相交就扩展 end，否则开始一个新区间。
        # ==========================================
        # merged = ???
        merged = [list(intersections[0])]
        for seg_start, seg_end in intersections[1:]:
            if seg_start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], seg_end)
            else:
                merged.append([seg_start, seg_end])
        return sum(seg_end - seg_start for seg_start, seg_end in merged)

    def summarize(self) -> Dict[str, Any]:
        total_comm_time = sum(e.duration for e in self.events)
        overlap_time = sum(self._overlap_duration(e.start, e.end) for e in self.events)
        by_op: Dict[str, Dict[str, float]] = {}
        for e in self.events:
            exposed_time = e.duration - self._overlap_duration(e.start, e.end)
            # ==========================================
            # TODO 3: 按 collective 汇总 count、time、bytes 和 exposed_time。
            # 提示：先用 setdefault 取得统计桶，再在同一个桶上累加四个字段；
            # exposed_time 代表该 op 未被 compute 隐藏的时间。
            # ==========================================
            # item = ???
            item = by_op.setdefault(e.op, {"count": 0, "time": 0.0, "bytes": 0, "exposed_time": 0.0})
            item["count"] += 1
            item["time"] += e.duration
            item["bytes"] += e.bytes
            item["exposed_time"] += exposed_time
        return {
            "num_comm_events": len(self.events),
            "total_comm_time": total_comm_time,
            "overlap_time": overlap_time,
            "overlap_ratio": min(overlap_time, total_comm_time) / max(total_comm_time, 1e-8),
            "exposed_comm_time": max(total_comm_time - overlap_time, 0.0),
            "by_op": by_op,
        }

    def find_priority_hotspot(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """从按 op 汇总的证据中找出最优先处理的 exposed communication hotspot。"""
        # ==========================================
        # TODO 4: 选择 exposed_time 最大的 collective，并返回 op、exposed_time 与 bytes。
        # 提示：`summary["by_op"]` 的键是 collective 名称，值是统计桶。
        # ==========================================
        # op, item = ???
        op, item = max(summary["by_op"].items(), key=lambda pair: pair[1]["exposed_time"])
        return {"op": op, "exposed_time": item["exposed_time"], "bytes": item["bytes"]}

    def timeline(self) -> List[Dict[str, Any]]:
        records = []
        for e in sorted(self.events, key=lambda x: (x.start, x.end, x.op)):
            records.append({"op": e.op, "start": e.start, "end": e.end, "duration": e.duration, "bytes": e.bytes, "overlap": e.overlap_with_compute})
        return records

```

### 解析

本题先用 latency–bandwidth 模型估算通信成本，再把观测事件整理为时间线，依次计算有效重叠、按 collective 归因，并选择应优先检查的通信热点。答案代码保留题目区的数据流，只补全四个机制判断。

**TODO 1：估算 collective 成本**

- 一次通信由固定启动延迟和 `payload / effective_bandwidth` 两部分组成。
- `sync_count` 让学习者看到：单次通信不慢并不代表高频同步的总成本低。

**TODO 2：合并有效重叠区间**

- 先裁剪通信区间与每段计算区间的交集，再合并彼此相交的片段。
- 这样多段 compute 覆盖同一段通信时，不会把可隐藏时间重复计算。

**TODO 3：按 collective 汇总暴露时间**

- 对同一种 collective 累加出现次数、总时长、传输量和未被计算覆盖的时间。
- 总时长较大不必然最影响吞吐；暴露时间才是比较关键路径代价的直接证据。

**TODO 4：选择优先热点**

- 从各 collective 的汇总结果中选择暴露时间最大的对象。
- 时间线仍用于解释该对象为何暴露，例如缺少可重叠计算或某些 rank 到达较晚。

### Step 5：可选 GPU 复测——NCCL collective 规模基准

先用 CPU 实验理解“什么时间会暴露在关键路径上”，再在真实多卡环境中比较四类 collective 随 payload 增长的时间与有效带宽。

| 学习问题 | CPU 已确认 | GPU 要确认与后续使用 |
| --- | --- | --- |
| collective 类型与 payload 如何改变通信成本 | 时间区间、有效 overlap、按 op 聚合与优先热点 | 比较 All-Reduce、All-Gather、Reduce-Scatter、All-to-All 的规模曲线，再到 48、79 对照真实 workload trace |

#### 5.1 环境与固定 workload

默认关闭；以下配置固定 world size、dtype、payload 梯度、warmup 与重复次数。

| 类别 | 固定并记录的字段 | 用途 |
| --- | --- | --- |
| 运行环境 | GPU 数、PyTorch / CUDA、NCCL、world size | 确认 collective 可启动 |
| workload | backend、每 rank payload、warmup、repeats | 固定通信规模与运行方式 |
| 对照矩阵 | 四类 collective × 多档 payload | 区分启动延迟区与带宽主导区 |
| 输出 | `TRACE_PATH`、默认关闭开关 | 保存 JSON 与 failure |

```python
# 默认关闭；确认至少有两张 CUDA GPU 后再改为 True。
from pathlib import Path

RUN_NCCL_BENCHMARK = False
WORLD_SIZE = 2  # 真实 NCCL collective 最少需要两个 rank。
BACKEND = 'nccl'  # GPU collective 使用 NCCL。
DTYPE = 'float32'
PAYLOAD_ELEMENTS = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
WARMUP = 5
REPEATS = 20
RESULTS_DIR = Path('benchmarks/results/46_communication_profiling')
TRACE_PATH = RESULTS_DIR / 'gpu_nccl_collective_matrix.json'
TORCHRUN = [
    'torchrun', '--standalone', '--nproc_per_node', str(WORLD_SIZE),
    'tools/run_parallel_mechanism_benchmark.py', '--mode', 'collective_matrix',
    '--payload-elements', ','.join(map(str, PAYLOAD_ELEMENTS)), '--dtype', DTYPE,
    '--warmup', str(WARMUP), '--repeats', str(REPEATS), '--output', str(TRACE_PATH),
]
print({'enabled': RUN_NCCL_BENCHMARK, 'world_size': WORLD_SIZE, 'payload_elements': PAYLOAD_ELEMENTS, 'result': str(TRACE_PATH)})
```

#### 5.2 执行 collective 并保存结果

运行后仅 rank 0 写入 JSON。每一行对应一种 collective 与一档 payload，并记录最慢 rank 时间、各 rank 时间和有效 payload 带宽。

```python
if RUN_NCCL_BENCHMARK:
    import subprocess
    import torch
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f'需要至少 {WORLD_SIZE} 张 CUDA GPU，当前只有 {torch.cuda.device_count()} 张。')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(TORCHRUN, check=True)
else:
    print('跳过 NCCL benchmark：将 RUN_NCCL_BENCHMARK 改为 True 后执行 5.2。')
```

#### 5.3 读取结果、解释指标与形成决策

先检查运行状态，再按 operation 和 payload 排序结果。小 payload 主要暴露启动成本，大 payload 更能反映链路带宽；该曲线用于解释 48 的热点，不替代 79 的端到端扩展效率。

```python
import json

if not TRACE_PATH.exists():
    print(f'尚无 GPU 结果：先运行 5.2，预期路径为 {TRACE_PATH}')
else:
    record = json.loads(TRACE_PATH.read_text(encoding='utf-8'))
    metrics = record.get('metrics', {})
    print({
        'decision': record.get('decision'),
        'evidence_level': record.get('evidence_level'),
        'failure': record.get('failure'),
        'rows': metrics.get('rows', []),
    })
```

| 读取项 | 判断与下一步 |
| --- | --- |
| 运行状态 | `failure` 非空时记录 NCCL 或多进程初始化问题，不比较耗时 |
| 可比口径 | `world_size / backend / payload / environment` 必须与对照一致 |
| 核心指标 | 按 operation / payload 读取最慢 rank 时间、rank 离散度与有效带宽 |
| 结论 | 用 `evidence_level / decision` 决定是否到 48、79 采集真实 workload trace |
## 相关阅读

想把本节的时间线判断用于真实工作负载，可继续查看 profiler 工具、通信热点归因与多卡项目证据。

- [NCCL 官方仓库](https://github.com/NVIDIA/nccl)
- [PyTorch Profiler 文档](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [48. Communication Hotspots and Mitigation | 通信热点与缓解策略](./48_Communication_Hotspots_and_Mitigation.md)
- [47. MoE Expert Parallel | MoE 专家并行](./47_MoE_Expert_Parallel.md)
- [79. Distributed Parallel Benchmark | 分布式并行 Benchmark](./79_Distributed_Parallel_Benchmark.md)
