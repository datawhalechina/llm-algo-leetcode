# 17. PyTorch Profiling Basics | PyTorch 性能分析基础

**难度：** Medium | **环境：** GPU optional | **标签：** `PyTorch`, `profiling`, `性能分析` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会用 `torch.profiler` 找热点；会看 CPU / CUDA 时间分布；会把 profiling 结果变成下一步排查动作。

**关键词：** `profiler`, `trace`, `latency`

## 前置阅读
**导语：** 先看 0E 组页，把注意力和性能分析的边界对齐，再进入这一页会更顺。
- [16. Attention Mechanism Intro | Attention 机制导论](./16_Attention_Mechanism_Intro.md)
- [0E 组页](./0E.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读
**导语：** 本页先把 profiling 的最小判断讲清楚；如果想继续看显存占用和内存账本，再顺着看下面这一页。
- [18. Memory Profiling and Optimization | 显存分析与优化](./18_Memory_Profiling_and_Optimization.md)

## Q1：性能问题先要回答哪几个判断？

任何性能问题都先问四件事：慢的是计算、内存还是通信；是单步慢还是累计慢；是 CPU 慢还是 GPU 真慢；测量前有没有把同步和噪声控制住。


```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10)
)
inputs = torch.randn(32, 100)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = model(inputs)

print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=5))

```

## Q1验证：最慢算子是否可以直接看到？

这里先跑一个最小 CPU profiler，确认报表能出来，且最耗时的算子能被排序出来。


```python
model = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 10))
inputs = torch.randn(32, 100)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    _ = model(inputs)
table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
assert 'aten::linear' in table or 'aten::addmm' in table
print('✅ profiler 基础通过')

```

## Q2：什么时候必须区分 CPU 时间和 CUDA 时间？

如果有 GPU，就要把 CPU 时间和 CUDA 时间分开看。很多看起来慢的问题，真正慢的可能不是算子本身，而是同步、搬运或调度。


```python
if torch.cuda.is_available():
    model_cuda = model.cuda()
    inputs_cuda = inputs.cuda()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _ = model_cuda(inputs_cuda)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
else:
    print('当前环境没有 GPU，跳过 CUDA profiling')

```

## Q2验证：CPU / CUDA 报表是否能区分？

这里确认：有 GPU 时能看到 CUDA 报表，没有 GPU 时至少 CPU 报表还能工作。


```python
if torch.cuda.is_available():
    model_cuda = model.cuda()
    inputs_cuda = inputs.cuda()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _ = model_cuda(inputs_cuda)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=5)
    assert 'CUDA' in table or 'cuda_time_total' in table
    print('✅ CPU / CUDA 区分通过')
else:
    print('✅ 当前环境无 GPU，CPU profiling 可用')

```

## Q3：什么时候必须导出 trace 或接入 TensorBoard？

当你需要复盘一整段执行路径时，trace 比单次表格更直观；当你要和训练过程对齐时，TensorBoard 更方便持续查看。


```python
from torch.profiler import tensorboard_trace_handler

with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("./log/profiler")) as prof:
    for _ in range(3):
        _ = model(inputs)
        prof.step()
print('trace handler 已执行')

```

## Q3验证：trace handler 是否能工作？

这里不用展开图形界面，只确认 trace handler 的最小接口能被触发。


```python
with profile(activities=[ProfilerActivity.CPU]) as prof:
    _ = model(inputs)
prof.export_chrome_trace('trace.json')
print('✅ trace 导出通过')

```

## Q4：什么时候必须把 profiling 接到训练骨架里？

如果你想知道慢在 forward、backward 还是 optimizer step，就不能只测单个算子，而要把 profiling 接进最小训练闭环。


```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
targets = torch.randint(0, 10, (32,))
with profile(activities=[ProfilerActivity.CPU]) as prof:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))

```

## Q4验证：最小训练步里的热点是否可见？

这里直接把 forward、loss、backward 和 step 包进 profiler，确认训练闭环的热点能被看见。


```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
targets = torch.randint(0, 10, (32,))
with profile(activities=[ProfilerActivity.CPU]) as prof:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
assert 'aten::linear' in table or 'aten::addmm' in table
print('✅ 最小训练 profiling 通过')

```
