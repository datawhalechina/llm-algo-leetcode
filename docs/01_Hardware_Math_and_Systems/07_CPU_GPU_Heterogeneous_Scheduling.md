# 07. CPU GPU Heterogeneous Scheduling | CPU 与 GPU 异构调度

**难度：** Medium | **环境：** GPU optional | **标签：** `CUDA`, `Scheduling`, `Host-Device` | **目标人群：** 异构调度入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把 CPU / GPU 协同、通信延迟和调度重叠讲清楚，重点是知道什么时候该把任务留在 CPU，什么时候该把任务交给 GPU。

**关键词：** `host`, `device`, `PCIe`
## 前置阅读
**导语：** 这一页先把 CPU / GPU 协同、通信延迟和调度重叠讲清楚，再决定什么时候该把任务留在 CPU，什么时候该把任务交给 GPU。
- [Group 1C: Distributed Communication and Memory Sharing | 1C: 多卡通信与显存共享](./1C.md)
- [15. CUDA Execution Model | CUDA 执行模型](./15_CUDA_Execution_Model.md)
- [17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](./17_CUDA_Stream_and_Asynchrony.md)
## 相关阅读
**导语：** 如果想继续把异构调度和 launch / graph 的关系补完整，可以接着看这些页。
- [Part 03: Triton Kernel Development | 第三部分：Triton 算子开发](../03_Triton_Kernels/intro.md)
- [05. Triton 性能调优与基准测试 (Autotune & Profiling)](../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md)
- [18. CUDA Graph and JIT Compile | CUDA Graph 与 JIT 编译](../04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile.md)
## Q1：Host 和 Device 分别扮演什么角色？

<details><summary>点击展开查看解析</summary>

Host 通常负责控制流、数据准备和调度，Device 负责大规模并行计算。

两者之间的核心瓶颈，往往不是“谁更强”，而是数据在两边之间搬运的成本和频率。

所以异构调度的第一件事，是先知道哪个环节适合留在 CPU，哪个环节应该下放到 GPU。
</details>
### Q1小验证：谁负责什么

先把控制流和并行计算分开。

```python
def role_of(where):
    return {'host': 'control', 'device': 'parallel compute'}.get(where, 'unknown')

for where in ['host', 'device']:
    print(where, '->', role_of(where))
```

## Q2：CUDA Streams 是如何隐藏通信延迟的？

<details><summary>点击展开查看解析</summary>

CUDA Streams 的作用，是让数据搬运和计算有机会重叠。

如果把拷贝和计算放在合适的队列里，GPU 就不必一边等数据一边空转，而是可以在一部分数据搬运时去做另一部分计算。

这不是让通信消失，而是把通信藏进了别的工作里。
</details>
### Q2小验证：什么时候 overlap 有意义

搬运和计算都存在时，才有 overlap 的空间。

```python
def can_overlap(copy_ms, compute_ms):
    return copy_ms > 0 and compute_ms > 0

print(can_overlap(8, 20))
```

## Q3：CPU Offload 在训练和推理中分别怎么用？

<details><summary>点击展开查看解析</summary>

CPU Offload 的目标，是把一部分暂时不需要常驻 GPU 的状态挪到 CPU，缓解显存压力。

在训练中，常见对象可能是优化器状态、部分参数或激活；在推理中，则可能是权重分层驻留或临时状态管理。

它的核心代价是更频繁的数据搬运，所以通常要和调度策略一起看，而不能只看“显存省了多少”。
</details>
### Q3小验证：卸载为什么有代价

省显存通常会换来更多搬运。

```python
def offload_tradeoff(saved_vram, transfer_cost):
    return saved_vram - transfer_cost

print(offload_tradeoff(10, 3))
```

## Q4：怎么判断一套异构调度方案是不是“看起来很并行，但实际没省多少时间”？

<details><summary>点击展开查看解析</summary>

判断标准不是“有没有用了很多并行词汇”，而是实际是否把等待时间压下去了。

如果 CPU 和 GPU 之间仍然频繁同步、通信和计算没有重叠、或者某个阶段仍然在卡住整条流水线，那么表面并行并不会带来实际收益。

所以异构调度最终还是要回到吞吐、延迟和重叠效率上看。
</details>
### Q4小验证：并行看起来很多，为什么还是慢

先看有没有真正重叠，而不是只看任务数量。

```python
def fake_parallelism(overlap=False, sync_points=0):
    return overlap and sync_points < 2

print(fake_parallelism(overlap=False, sync_points=3))
```

## ⚠️ 常见误区

- 并行任务多不等于更快。
- 异构调度的关键是重叠和边界划分。
- Offload 省显存，但会带来搬运成本。
- 先看吞吐和延迟，再看并行标签。