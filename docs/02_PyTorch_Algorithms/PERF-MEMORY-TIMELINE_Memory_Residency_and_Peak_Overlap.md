# PERF-MEMORY-TIMELINE. Memory Residency and Peak Overlap | 显存驻留与峰值重叠

**难度：** Medium　|　**环境：** CPU 可完成生命周期练习；单卡 GPU 用于验证真实分配峰值　|　**标签：** `性能分析` `显存时间线` `对象驻留`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

显存峰值由某一时刻仍然存活的对象共同形成。只看总分配量或 `reserved` 无法判断哪些对象真正跨越峰值；本节从分配与释放事件重建驻留区间、峰值对象集合和可验证的生命周期假设。
## 前置阅读

- [Part 00 · 18 显存分析](../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)
- [性能分析 · 04 显存时间线与驻留状态](../topic_discussion/performance_analysis/04_memory_timeline_and_residency.md)
### Step 1：分配事件必须保留对象身份

时间、对象标识、字节数、类别和动作共同定义一次生命周期变化。对象重复分配、释放未知对象或非正字节数会破坏后续驻留判断，必须在形成时间线前拒绝。

| 字段 | 作用 | 非法情况 |
| --- | --- | --- |
| `time_ms` | 排列生命周期事件 | 时间为负或无法转换为数值 |
| `object_id` | 连接同一对象的分配与释放 | 标识为空、重复分配或未知释放 |
| `bytes` | 更新活动容量 | 分配字节数不为正 |
| `category` | 解释峰值组成 | 缺少对象类别 |
| `action` | 区分进入与离开驻留集 | 不是 `allocate` 或 `free` |

![从分配事件到显存峰值归因](../public/02_PyTorch_Algorithms/perf_memory_timeline_overview.svg)
### Step 2：驻留区间决定活动字节

对象从分配时刻进入驻留集，到释放时刻离开。每个事件之后重新计算活动字节和类别构成，才能确定峰值发生时哪些状态仍然存活。

| 对象 | 生命周期 | 对峰值的影响 |
| --- | --- | --- |
| 长生命周期状态 | 跨越大部分阶段 | 构成稳定容量底座 |
| 激活或缓存 | 随阶段、长度或并发增长 | 决定工作负载相关容量 |
| 临时工作区 | 区间短但容量可能很大 | 与其他对象重叠时抬高峰值 |
| 分配器保留 | 不等于当前活动对象 | 需要与 `allocated` 和快照共同解释 |
### Step 3：峰值归因要检查重叠对象

缩短一个对象的生命周期，只有在它跨越峰值时刻时才可能降低峰值。判别实验应保持形状与计算目标不变，只改变驻留关系，再检查峰值时刻和对象集合是否随之变化。

| 观察 | 可以支持的判断 | 仍不能证明 |
| --- | --- | --- |
| 峰值随对象释放时刻前移而下降 | 该对象参与了原峰值重叠 | 策略在所有工作负载下都收益 |
| `allocated` 下降而 `reserved` 不变 | 活动对象已释放，分配器仍保留块 | 一定存在不可用碎片 |
| 峰值不变但运行时间上升 | 生命周期改动未命中峰值且引入额外工作 | 其他容量策略无效 |

![对象生命周期形成峰值重叠](../public/02_PyTorch_Algorithms/perf_memory_peak_overlap.svg)
### Step 4：实现驻留时间线与峰值报告

题目给定对象分配与释放事件。你需要先验证事件契约，再重建每个事件后的活动对象集合，最后报告峰值时刻、活动字节和类别构成。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：规范化生命周期事件<br>`normalize_memory_events` | 校验并排序生命周期事件 | 对象先分配后释放；分配字节为正；同一时刻先分配后释放 | 非法动作、重复分配与未知释放 |
| TODO 2：重建对象驻留<br>`build_residency_timeline` | 更新活动对象与容量 | 每个事件后保存活动字节、对象和类别构成 | 活动容量增减与未释放对象 |
| TODO 3：定位峰值重叠<br>`peak_overlap_report` | 找到峰值及其对象集合 | 空时间线报错；峰值字段来自同一时间点 | 峰值时刻、字节数与类别求和 |

```python
from collections import defaultdict
from typing import Dict, List, Sequence

```


```python
def normalize_memory_events(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Validate allocation/free events and return them in deterministic order."""
    action_order = {"allocate": 0, "free": 1}
    normalized = []
    for event in events:
        action = str(event.get("action", ""))
        object_id = str(event.get("object_id", ""))
        category = str(event.get("category", ""))
        time_ms = float(event.get("time_ms", -1.0))
        size = int(event.get("bytes", 0))
        if action not in action_order or not object_id or not category or time_ms < 0 or size <= 0:
            raise ValueError("invalid memory event")
        normalized.append({"time_ms": time_ms, "object_id": object_id, "action": action, "bytes": size, "category": category})
    normalized.sort(key=lambda event: (event["time_ms"], action_order[event["action"]], event["object_id"]))
    live = set()
    for event in normalized:
        object_id = event["object_id"]
        # ==========================================
        # TODO 1：规范化生命周期事件
        # ==========================================
        # duplicate_allocation = ???  # allocate 指向已存活对象。
        # unknown_free = ???  # free 指向未存活对象。
        raise NotImplementedError("TODO 1：请规范化显存事件")
        if duplicate_allocation or unknown_free:
            raise ValueError("invalid object lifecycle transition")
        if event["action"] == "allocate":
            live.add(object_id)
        else:
            live.remove(object_id)
    return normalized


def build_residency_timeline(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return active bytes, live objects, and category bytes after each event."""
    live = {}
    timeline = []
    for event in events:
        if event["action"] == "allocate":
            live[event["object_id"]] = event
        else:
            live.pop(event["object_id"])
        category_bytes = defaultdict(int)
        for item in live.values():
            category_bytes[item["category"]] += item["bytes"]
        # ==========================================
        # TODO 2：重建对象驻留
        # ==========================================
        # active_bytes = ???  # 当前全部存活对象的总字节数。
        # snapshot = ???  # 当前时刻、活动字节、对象 ID 和类别构成。
        # timeline = ???  # 追加当前驻留快照后的时间线。
        raise NotImplementedError("TODO 2：请重建驻留时间线")
    return timeline


def peak_overlap_report(timeline: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Return the peak point and the objects that overlap there."""
    if not timeline:
        raise ValueError("timeline must not be empty")
    # ==========================================
    # TODO 3：定位峰值重叠
    # ==========================================
    # peak = ???  # 峰值时刻对应的完整时间线记录。
    raise NotImplementedError("TODO 3：请生成峰值重叠报告")
    return {"peak_time_ms": peak["time_ms"], "peak_bytes": peak["active_bytes"],
            "live_object_ids": peak["live_object_ids"], "category_bytes": peak["category_bytes"]}

```

### 测试

```python
def _memory_events():
    """Return one deterministic lifecycle with a three-object peak."""
    return [
        {"time_ms": 0.0, "object_id": "weights", "action": "allocate", "bytes": 100, "category": "model"},
        {"time_ms": 1.0, "object_id": "input", "action": "allocate", "bytes": 20, "category": "batch"},
        {"time_ms": 2.0, "object_id": "activation", "action": "allocate", "bytes": 60, "category": "activation"},
        {"time_ms": 3.0, "object_id": "input", "action": "free", "bytes": 20, "category": "batch"},
        {"time_ms": 4.0, "object_id": "activation", "action": "free", "bytes": 60, "category": "activation"},
    ]


def test_normalize_memory_events():
    """Validate deterministic ordering and lifecycle rejection."""
    events = normalize_memory_events(list(reversed(_memory_events())))
    assert events[0]["object_id"] == "weights"
    bad = _memory_events() + [{"time_ms": 5.0, "object_id": "ghost", "action": "free", "bytes": 1, "category": "other"}]
    try:
        normalize_memory_events(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("释放未知对象必须报错")


def test_residency_and_peak():
    """Validate active-byte updates and peak object attribution."""
    timeline = build_residency_timeline(normalize_memory_events(_memory_events()))
    assert [point["active_bytes"] for point in timeline] == [100, 120, 180, 160, 100]
    peak = peak_overlap_report(timeline)
    assert peak["peak_bytes"] == 180
    assert peak["peak_time_ms"] == 2.0
    assert peak["live_object_ids"] == ["activation", "input", "weights"]
    assert sum(peak["category_bytes"].values()) == peak["peak_bytes"]


def run_memory_timeline_tests():
    """Run event, residency, and peak-overlap tests."""
    test_normalize_memory_events()
    test_residency_and_peak()
    print("✅ 显存事件、驻留时间线与峰值重叠测试通过")


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
from collections import defaultdict
from typing import Dict, List, Sequence


# TODO 1：规范化生命周期事件
def normalize_memory_events(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Validate allocation/free events and return them in deterministic order."""
    action_order = {"allocate": 0, "free": 1}
    normalized = []
    for event in events:
        action = str(event.get("action", ""))
        object_id = str(event.get("object_id", ""))
        category = str(event.get("category", ""))
        time_ms = float(event.get("time_ms", -1.0))
        size = int(event.get("bytes", 0))
        if action not in action_order or not object_id or not category or time_ms < 0 or size <= 0:
            raise ValueError("invalid memory event")
        normalized.append({"time_ms": time_ms, "object_id": object_id, "action": action, "bytes": size, "category": category})
    normalized.sort(key=lambda event: (event["time_ms"], action_order[event["action"]], event["object_id"]))
    live = set()
    for event in normalized:
        object_id = event["object_id"]
        if event["action"] == "allocate":
            if object_id in live:
                raise ValueError("object allocated twice")
            live.add(object_id)
        else:
            if object_id not in live:
                raise ValueError("free references unknown object")
            live.remove(object_id)
    return normalized


# TODO 2：重建对象驻留
def build_residency_timeline(events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return active bytes, live objects, and category bytes after each event."""
    live = {}
    timeline = []
    for event in events:
        if event["action"] == "allocate":
            live[event["object_id"]] = event
        else:
            live.pop(event["object_id"])
        category_bytes = defaultdict(int)
        for item in live.values():
            category_bytes[item["category"]] += item["bytes"]
        timeline.append({
            "time_ms": event["time_ms"],
            "active_bytes": sum(item["bytes"] for item in live.values()),
            "live_object_ids": sorted(live),
            "category_bytes": dict(sorted(category_bytes.items())),
        })
    return timeline


# TODO 3：定位峰值重叠
def peak_overlap_report(timeline: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Return the peak point and the objects that overlap there."""
    if not timeline:
        raise ValueError("timeline must not be empty")
    peak = max(timeline, key=lambda point: (point["active_bytes"], -point["time_ms"]))
    return {
        "peak_time_ms": peak["time_ms"],
        "peak_bytes": peak["active_bytes"],
        "live_object_ids": peak["live_object_ids"],
        "category_bytes": peak["category_bytes"],
    }
```

### 解析

#### TODO 1：规范化生命周期事件

- 同一对象只能先分配后释放，分配字节必须为正；同一时刻的处理顺序不能制造负容量。
- 实现统一字段、排序并检查生命周期；非法动作、重复分配和未知释放测试分别验证契约。

#### TODO 2：重建对象驻留

- 每条时间线记录的活动字节、对象集合和类别构成必须来自同一时刻。
- 实现按事件更新活动集合后再汇总容量；容量增减和未释放对象测试确认状态没有漂移。

#### TODO 3：定位峰值重叠

- 峰值不是各类别最大值之和，而是单个时刻同时驻留对象的总量。
- 实现从同一条峰值记录读取时间、字节和对象；峰值字段与类别求和测试验证归因一致。
### Step 5：GPU 机制实验——生命周期与真实峰值

CPU 练习重建对象驻留；GPU 实验保持输入与输出目标不变，对比“激活与工作区重叠”和“释放激活后重算”两条路径的峰值与时间。

| 学习问题 | CPU 已确认 | GPU 要确认与后续使用 |
| --- | --- | --- |
| 缩短跨峰值对象的生命周期是否降低真实峰值 | 驻留区间、峰值时刻与对象集合 | 输出是否一致、`max_memory_allocated` 是否下降、重算时间代价是多少 |

#### 5.1 环境与固定工作负载

本实验使用 `PERF-CUDA-MECHANISM`。首次运行按[专题 Quickstart](../topic_discussion/performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart)创建环境，实验登记见[性能分析实验资产表](../topic_discussion/performance_analysis/experiment_assets.md)。

| 配置层 | 本节固定口径 | 修改规则 |
| --- | --- | --- |
| 运行环境 | 单张 CUDA GPU；`PERF-CUDA-MECHANISM`；参考显存 8 GiB | Profile 预检未返回 `status=ok` 时停止实验 |
| 工作负载 | 8192 × 2048 FP32 张量；重叠路径与重算路径 | 改变形状或 dtype 后使用新运行标签 |
| 正确性 | 两条路径输出 `allclose` | 正确性失败时不解释峰值收益 |
| 测量 | 5 次预热、20 次正式重复；峰值 allocated / reserved 与延迟中位数 | 每次运行前同步并重置峰值统计 |
| 结果资产 | `benchmarks/results/perf_memory_timeline/<workload_id>_<run_label>.json` | 5.3 独立读取该路径 |

```python
RUN_MODE = "dry_run"  # 默认值 dry_run；改为 real_gpu 后执行真实显存峰值对照。
RUN_LABEL = "default"  # 默认值 default；正式复测使用唯一标签并保留历史结果。
ENVIRONMENT_PROFILE = "PERF-CUDA-MECHANISM"  # 运行前必须通过的环境配置 ID。
SEMANTIC_ID = "PERF-MEMORY-TIMELINE"  # 机制页稳定标识。
WORKLOAD_ID = "perf-memory-peak-overlap"  # 生命周期与峰值重叠实验的稳定标识。
WORKLOAD_VERSION = "v2"  # v2 统一结果身份并增加峰值与延迟派生指标。
ROWS = 8192  # 输入张量行数；显存不足时优先下调。
HIDDEN_SIZE = 2048  # 输入张量列数。
WARMUP_STEPS = 5  # 不进入正式结果的预热次数。
REPEAT_STEPS = 20  # 每条路径的正式重复次数。
RESULT_RELATIVE_DIR = "benchmarks/results/perf_memory_timeline"  # 结果 JSON 的仓库相对目录。
```

#### 5.2 执行实验并保存 JSON

```python
from pathlib import Path
import json
import statistics
import sys
import torch

if RUN_MODE == "real_gpu":
    if not torch.cuda.is_available():
        raise RuntimeError("PERF-MEMORY-TIMELINE 真实实验需要 CUDA GPU")
    from tools.project_runtime import resolve_project_root

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

    result_dir = resolve_project_root() / RESULT_RELATIVE_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{WORKLOAD_ID}_{RUN_LABEL}.json"
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
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    print("⏭️ RUN_MODE=dry_run；未执行 GPU 显存时间线实验。")
```

#### 5.3 读取结果

```python
from pathlib import Path
import json
from tools.project_runtime import resolve_project_root

if RUN_MODE == "real_gpu":
    result_path = resolve_project_root() / RESULT_RELATIVE_DIR / f"{WORKLOAD_ID}_{RUN_LABEL}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"未找到结果：{result_path}")
    memory_record = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps(memory_record, ensure_ascii=False, indent=2))
else:
    memory_record = {"status": "not_run", "decision": "enable_real_gpu_after_profile_preflight"}
    print(memory_record)
```

#### 5.4 解释证据与填写结果表

RTX 5070 Ti 实测中，释放激活后重算通过校验和检查，并降低活动分配峰值；分配器保留峰值没有变化，且重算路径明显变慢。该结果呈现了活动分配峰值与重算时间的取舍。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 校验和检查 | checksum 与 `failure` | 校验和检查通过；无失败 | 两条路径的输出总和接近；本次未记录逐元素误差 |
| 活动峰值 | `peak_allocated_bytes` | 201.33 → 134.22 MB，下降约 33.3% | 缩短激活生命周期确实减少峰值时刻的活动对象 |
| 时间代价 | 延迟中位数与 `latency_ratio` | 0.80 → 1.49 ms，约 1.86× | 只有容量是硬约束时才考虑该重算候选 |
| 分配器保留 | `peak_reserved_bytes` | 两条路径均约 203.42 MB | allocator 保留量未下降，不能解释为碎片改善或已归还系统显存 |
## 相关阅读

- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [Part 02 · 43 训练内存账本](./43_Unified_Training_Memory_Ledger.md)
- [反向传播与训练机制 · 05 激活检查点与卸载](../topic_discussion/backpropagation_training_mechanism/05_checkpointing_and_offload.md)
