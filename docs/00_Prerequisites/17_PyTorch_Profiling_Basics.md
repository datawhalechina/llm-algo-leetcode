# 17. PyTorch Profiling Basics | PyTorch 性能分析基础

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `profiling`, `性能分析` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


当一次运行变慢时，先用 profiler 观察一次运行由哪些算子和阶段组成，再比较单次 `latency_ms`、带单位的 `throughput` 和训练阶段耗时。本节从一个 CPU 可运行的小模型开始，学习如何读汇总表、导出 trace，并把局部 hotspot 放回完整训练步骤。这样可以把“感觉变慢”转换成下一步可复查的时间证据。

先用 CPU 建立稳定的测量方法，再把相同的记录字段迁移到 GPU 实验中。

**关键词：** `profiler`, `trace`, `latency`

![Profiling 时间证据图](../public/00_Prerequisites/17_profiling_evidence_map.svg)

## 前置阅读
**导语：** 先从 0E 组页了解性能问题的观察入口，再用本页的最小模型把一次运行拆成可比较的时间证据。
- [0E 组页](./0E.md)

## Q1：如何建立一次可比较的性能记录？

假设你已经有一个固定的模型和输入。一次运行等待时间变长，要看 latency；单位时间处理的样本或 token 变少，要看 throughput。第一次运行还可能包含初始化成本，所以先预热，再重复测量，最后把时间结果和 profiler 热点放进同一条记录。

为了让两次结果可比较，至少固定模型、输入形状、dtype、batch 和线程设置；先丢弃预热轮次，再重复采样，最后报告平均值或中位数。单次运行适合发现线索，不足以证明稳定收益。

| 记录项 | 作用 |
|:---|:---|
| latency | 判断一次运行等待多久，报告平均值或中位数 |
| throughput | 判断单位时间处理多少样本或 token |
| profiler hotspot | 找到时间集中的算子或阶段 |
| 环境记录 | 保存模型、输入、dtype 和版本，保证结果可复查 |


```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


import time

model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10)
)
inputs = torch.randn(32, 100)
for _ in range(2):
    _ = model(inputs)
latency_samples = []
for _ in range(5):
    start = time.perf_counter()
    _ = model(inputs)
    latency_samples.append((time.perf_counter() - start) * 1000)
latency_ms = sum(latency_samples) / len(latency_samples)
throughput_samples_s = inputs.shape[0] / (latency_ms / 1000)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = model(inputs)

hotspot_table = prof.key_averages().table(sort_by='cpu_time_total', row_limit=5)
summary = {
    'latency_ms': round(latency_ms, 3),
    'throughput_samples_s': round(throughput_samples_s, 1),
    'hotspot_table': hotspot_table,
}
print(summary)

```

### Q1 代码：用预热、重复采样和 CPU profiler 建立基线

这里验证前面的 summary 同时包含 latency、throughput 和热点报表；它是可比较的 CPU 基线，不代表稳定的 GPU benchmark。


```python
assert len(latency_samples) == 5
assert summary['latency_ms'] > 0 and summary['throughput_samples_s'] > 0
assert 'aten::linear' in summary['hotspot_table'] or 'aten::addmm' in summary['hotspot_table']
print('✅ CPU 性能基线与热点记录通过')

```

## Q2：CPU 时间和 CUDA 时间分别说明什么？

一次 GPU 运算通常经历“CPU 发起调用 → 数据到达 GPU → GPU 执行 kernel → CPU 等待或继续工作”。CPU time 记录主机侧调用和等待，CUDA time 记录 GPU kernel 执行；两者差异较大时，再检查同步、数据搬运或调度，而不是只看 CPU 排名。

| 观察项 | 代表什么 | 常见解释 |
|:---|:---|:---|
| CPU time | 主机侧调用、调度和等待 | 可能存在数据准备或同步开销 |
| CUDA time | GPU kernel 执行时间 | 反映设备侧计算耗时 |
| CPU / CUDA 差异 | 主机与设备执行不同步 | 检查同步、搬运或调度 |


```python
# 这里先记录 CPU profiler 的算子时间；GPU kernel 时间在对应 GPU 实验中记录。
with profile(activities=[ProfilerActivity.CPU]) as prof:
    _ = model(inputs)
print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=5))

```

### Q2 代码：验证 CPU 侧的调用时间基线

这里确认 CPU 侧的 profiling 接口可以独立工作；真正的 CUDA 时间、同步和数据搬运需要在 Part02 的 GPU 实验中采集。


```python
assert 'aten::linear' in hotspot_table or 'aten::addmm' in hotspot_table
print('✅ CPU profiling 基线通过；CUDA 实测在 Part02 GPU 实验中完成')

```

## Q3：什么时候需要从汇总表升级到 trace？

汇总表适合回答“哪个算子或阶段总时间最高”；如果还要知道阶段的先后、等待关系或多次迭代的变化，就需要 trace。trace 是可回看的执行记录，TensorBoard handler 只是查看它的一种工具出口。


```python
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        _ = model(inputs)
        prof.step()
prof.export_chrome_trace('trace.json')
print('trace 已导出到 trace.json')

```

### Q3 代码：检查可回看的 trace 文件

这里不展开图形界面，只确认 trace 文件已经生成；TensorBoard 是后续查看这类 trace 的另一种入口。


```python
from pathlib import Path
trace_path = Path('trace.json')
assert trace_path.exists() and trace_path.stat().st_size > 0
print('✅ trace 导出通过:', trace_path)

```

## Q4：如何把局部热点放回完整训练步骤？

如果一个训练步骤总耗时变长，只测某个 Linear 算子还不能说明问题：慢点可能在 forward、loss、backward，也可能在 optimizer step。把这些阶段放进同一个最小训练闭环，并用标签标记，才能把局部热点放回完整训练过程。


```python
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
print(training_table)

```

### Q4 代码：检查训练阶段标签和热点结果

这里复用前一个单元的结果，确认五个训练阶段都被标记，并且 loss 可以正常计算。


```python
assert all(name in training_table for name in ['zero_grad', 'forward', 'loss', 'backward', 'optimizer_step'])
assert torch.isfinite(loss).item()
print('✅ 训练阶段标签、热点结果和 loss 检查通过')

```

## 相关阅读
**导语：** 完成本节后，可以继续学习显存账本、硬件瓶颈模型和真实 profiling 项目。
- [18. Memory Profiling and Optimization | 显存分析与优化](./18_Memory_Profiling_and_Optimization.md)
- [20. Profiling and Memory Ledger | 性能剖析与显存账本](./20_Profiling_and_Memory_Ledger.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [PyTorch Profiler 官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [74. Profiling Driven End-to-End Optimization | Profiling 驱动的端到端优化](../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)
