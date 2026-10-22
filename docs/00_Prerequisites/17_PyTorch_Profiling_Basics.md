# 17. PyTorch Profiling Basics | PyTorch 性能分析基础

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `profiling`, `性能分析` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

确认运行结果有效后，如果一次运行变慢，需要同时观察算子汇总、阶段时间线和重复测量分布。单次总耗时无法说明时间消耗在哪个阶段，也无法区分稳定热点与偶然波动。

完成本节后，你应能建立带单位的时间证据，区分汇总表与追踪记录的用途，并把局部热点放回完整训练步骤。先用热点选择下一项检查，再用对照实验检验原因。

**关键词：** `profiler`, `trace`, `latency`

![Profiling 时间证据图](../public/00_Prerequisites/17_profiling_evidence_map.svg)

## 前置阅读
**导语：** 先从 0E 组页了解诊断顺序，并用 19 节确认运行有效，再把一次运行拆成可比较的时间证据。
- [0E 组页](./0E.md)
- [Part 00 · 19 调试与异常定位](./19_Debugging_and_Anomaly_Localization.md)

## Q1：如何建立一次可比较的性能记录？

假设你已经有一个通过正确性检查的固定模型和输入。一次运行等待时间变长，要看 latency；单位时间处理的样本或 token 变少，要看 throughput。第一次运行还可能包含初始化成本，所以先预热，再重复测量，最后把时间结果和 profiler 热点放进同一条记录。

为了让两次结果可比较，至少固定模型、输入形状、dtype、batch 和线程设置；先丢弃预热轮次，再保留全部重复样本，最后报告中位数和 P95。中位数描述典型运行，P95 暴露慢样本；单次运行和只报告平均值都不足以描述分布。

| 记录项 | 作用 |
|:---|:---|
| latency samples | 保留每轮时间，避免汇总值掩盖抖动 |
| median / P95 | 分别描述典型延迟和慢样本延迟 |
| throughput | 判断单位时间处理多少样本或 token |
| profiler hotspot | 找到时间集中的算子或阶段，选择下一项检查 |
| 环境记录 | 保存模型、输入、dtype 和版本，保证结果可复查 |


```python
import math
import platform
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

# 修改一个输入后重新运行本节，比较相同范围的时间记录。
SEED, WARMUP, REPEATS, BATCH_SIZE = 7, 2, 20, 32
torch.manual_seed(SEED)
model = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 10)).eval()
inputs = torch.randn(BATCH_SIZE, 100)

with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(inputs)
    latency_samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        _ = model(inputs)
        latency_samples.append((time.perf_counter() - start) * 1000)


def nearest_rank(values, fraction):
    """按最近秩计算分位数；中位数另用 statistics.median。"""
    if not values or not 0 <= fraction <= 1:
        raise ValueError('values 不能为空，fraction 必须位于 [0, 1]')
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('时间样本必须是非负有限值')
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


latency_median_ms = statistics.median(latency_samples)
latency_p95_ms = nearest_rank(latency_samples, 0.95)
throughput_samples_s = inputs.shape[0] / (latency_median_ms / 1000)
trace_dir = TemporaryDirectory()
trace_path = Path(trace_dir.name) / '17_profile_trace.json'
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with torch.no_grad():
        output = model(inputs)

hotspot_table = prof.key_averages().table(sort_by='cpu_time_total', row_limit=5)
workload = {
    'model': 'Linear(100, 200) -> ReLU -> Linear(200, 10)',
    'scope': 'forward_only', 'model_mode': 'eval', 'grad_enabled': False,
    'input_shape': tuple(inputs.shape), 'dtype': str(inputs.dtype),
    'device': str(inputs.device), 'seed': SEED,
    'warmup': WARMUP, 'repeats': REPEATS,
    'measurement_method': 'cpu_perf_counter',
    'intraop_threads': torch.get_num_threads(),
    'interop_threads': torch.get_num_interop_threads(),
    'torch_version': torch.__version__, 'python_version': platform.python_version(),
}
summary = {
    'evidence_type': 'time', 'status': 'measured', 'activity_scope': 'cpu',
    'workload': workload,
    'latency_samples_ms': latency_samples,
    'latency_median_ms': latency_median_ms,
    'latency_p95_ms': latency_p95_ms,
    'throughput_samples_s': throughput_samples_s,
    'throughput_definition': 'batch_size / median_forward_seconds',
    'hotspot_table': hotspot_table,
}
print(summary)

```

## Q1 验证：用预热、重复样本和延迟分布建立基线

这里验证 summary 保留原始样本，并同时包含中位数、P95、吞吐和热点报表。


```python
assert summary['evidence_type'] == 'time' and summary['activity_scope'] == 'cpu'
assert summary['workload']['repeats'] == len(summary['latency_samples_ms']) == 20
assert summary['latency_median_ms'] > 0 and summary['throughput_samples_s'] > 0
assert summary['latency_p95_ms'] >= summary['latency_median_ms']
assert 'aten::linear' in summary['hotspot_table'] or 'aten::addmm' in summary['hotspot_table']
assert summary['workload']['scope'] == 'forward_only'
assert summary['workload']['grad_enabled'] is False
assert summary['workload']['intraop_threads'] > 0
assert nearest_rank([1, 2, 3, 100], 0.95) == 100
assert statistics.median([1, 2, 3, 100]) == 2.5
print('✅ CPU 前向基线、测量范围与热点记录通过')

```

## Q2：为什么主机时间不能直接代替设备时间？

主机负责准备输入、发起算子并在必要时等待，设备负责执行 kernel；数据搬运和同步还可能穿插在两者之间。CPU time 只能说明主机侧调用、调度和等待，不能代替 CUDA kernel 时间。当前 CPU-first 练习先验证主机活动记录；进入 GPU 实验后，必须在相同 workload 下同时采集 CUDA 活动、搬运与同步事件。

| 观察项 | 代表什么 | 常见解释 |
|:---|:---|:---|
| 主机时间 | CPU 侧调用、调度和等待 | 可能存在数据准备、启动或同步开销 |
| 设备时间 | GPU kernel 执行区间 | 反映设备侧计算活动 |
| 搬运与同步 | 数据传输、事件依赖和显式等待 | 解释主机与设备时间线之间的空隙 |


```python
# CPU-first 页面只采集主机活动；本结果不能外推 CUDA kernel 时间。
with profile(activities=[ProfilerActivity.CPU]) as cpu_prof:
    with torch.no_grad():
        _ = model(inputs)
cpu_time_table = cpu_prof.key_averages().table(sort_by='cpu_time_total', row_limit=5)
print(cpu_time_table)

```

## Q2 验证：检查 CPU 侧的调用时间基线

该验证检查 CPU 侧 profiling 接口与阶段计时结果。CUDA kernel、同步和数据搬运时间应在 Part 02 的 GPU 实验中使用 CUDA 活动记录采集。


```python
assert 'aten::linear' in cpu_time_table or 'aten::addmm' in cpu_time_table
assert len(cpu_time_table) > 0
print('✅ 主机活动记录通过；设备、搬运与同步证据需在 GPU 实验中采集')

```

## Q3：什么时候需要从汇总表升级到 trace？

汇总表适合回答“哪个算子或阶段累计时间最高”；如果还要知道阶段的先后、等待关系或多次迭代的变化，就需要 trace。trace 是带时间戳的事件记录，可以先检查事件数量和名称，再交给 Chrome Trace 或 TensorBoard Profiler 查看完整时间线。


```python
import json

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        with torch.no_grad():
            _ = model(inputs)
        prof.step()
prof.export_chrome_trace(str(trace_path))
trace_record = json.loads(trace_path.read_text(encoding='utf-8'))
trace_events = trace_record.get('traceEvents', [])
named_events = [event.get('name') for event in trace_events if event.get('name')]
print('trace path:', trace_path)
print('event count:', len(trace_events))
print('sample event names:', named_events[:8])

```

## Q3 验证：检查可回看的 trace 文件

验证代码先读取 trace 的事件列表，再清理临时文件。需要人工查看完整时间线时，可以把 `trace_path` 改为希望保留的输出目录。


```python
assert trace_path.exists() and trace_path.stat().st_size > 0
assert len(trace_events) > 0 and len(named_events) > 0
print('✅ trace 导出与事件解析通过:', trace_path)
trace_dir.cleanup()
assert not trace_path.exists()
print('✅ 临时 trace 已清理')

```

## Q4：如何把局部热点放回完整训练步骤？

如果一个训练步骤总耗时变长，只测某个 Linear 算子还不能说明问题：慢点可能在 forward、loss、backward，也可能在 optimizer step。把这些阶段放进同一个最小训练闭环，并用标签标记，才能把局部热点放回完整训练过程。本例单独采集一个完整训练步骤的阶段记录，不沿用 Q1 的前向延迟和吞吐。各标签互不嵌套，阶段占比的分母是这些标签的时间之和；它不包含未标记的 Python 开销，也不是重复测量的端到端时间。


```python
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
targets = torch.randint(0, 10, (32,))
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with torch.profiler.record_function('zero_grad'):
        optimizer.zero_grad()
    with torch.profiler.record_function('forward'):
        outputs = model(inputs)
    with torch.profiler.record_function('loss'):
        loss = criterion(outputs, targets)
    with torch.profiler.record_function('backward'):
        loss.backward()
    with torch.profiler.record_function('optimizer_step'):
        optimizer.step()
training_table = prof.key_averages().table(sort_by='cpu_time_total', row_limit=20)
stage_labels = ['zero_grad', 'forward', 'loss', 'backward', 'optimizer_step']
stage_events = [
    {'name': event.key, 'total_ms': event.cpu_time_total / 1000.0, 'calls': event.count}
    for event in prof.key_averages()
    if event.key in stage_labels
]
stage_total_ms = sum(event['total_ms'] for event in stage_events)
stage_hotspot = max(stage_events, key=lambda event: event['total_ms'])
stage_hotspot = {
    **stage_hotspot,
    'share_pct': round(stage_hotspot['total_ms'] / stage_total_ms * 100, 1),
}
time_evidence = {
    'evidence_type': 'time',
    'status': 'measured',
    'activity_scope': 'cpu',
    'workload': {
        **workload, 'scope': 'training_step', 'model_mode': 'train',
        'grad_enabled': True, 'optimizer': 'SGD', 'warmup': 0, 'repeats': 1,
        'measurement_method': 'cpu_profiler_non_nested_stages',
    },
    'stage_events': stage_events,
    'labeled_stage_sum_ms': stage_total_ms,
    'share_denominator': 'sum_of_non_nested_stage_times',
    'stage_labels': stage_labels,
    'hotspot': stage_hotspot,
    'interpretation_limit': 'hotspot_is_candidate_not_root_cause',
}
print(training_table)
print('time evidence:', time_evidence)

```

## Q4 验证：检查训练阶段标签和热点结果

这里复用前一个单元的结果，确认五个训练阶段都被标记、损失有限，且训练阶段记录没有混入前向基线的延迟样本。


```python
assert {event['name'] for event in stage_events} == set(stage_labels)
assert torch.isfinite(loss).item()
assert time_evidence['status'] == 'measured'
assert len(trace_events) > 0
assert time_evidence['workload']['scope'] != summary['workload']['scope']
assert 'latency_samples_ms' not in time_evidence
assert time_evidence['labeled_stage_sum_ms'] == sum(event['total_ms'] for event in stage_events)
assert time_evidence['hotspot']['name'] in stage_labels
assert 0 < time_evidence['hotspot']['share_pct'] <= 100
assert time_evidence['interpretation_limit'] == 'hotspot_is_candidate_not_root_cause'
print('✅ 训练阶段标签、时间证据与解释边界通过')

```

## 相关阅读
**导语：** 完成本节后，可以继续学习显存账本、硬件瓶颈模型和真实 profiling 项目。
- [Part 00 · 20 性能证据与实验设计](./20_Performance_Evidence_and_Experiment_Design.md)
- [Part 01 · 13 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [PyTorch Profiler 官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
