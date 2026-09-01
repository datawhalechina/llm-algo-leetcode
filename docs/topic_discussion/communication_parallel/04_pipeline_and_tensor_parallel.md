# 04. Pipeline and Tensor Parallel | Pipeline 与 Tensor Parallel

## 页面目标

这一页回答的是：当数据并行和状态切分仍不够时，如何在层级切分与算子切分之间选择，以及 pipeline bubble、同步频率和通信粒度如何影响结果。

本页的输出是切分候选：模型切在哪里、通信发生多频繁、流水线空泡和小 batch 风险是否可接受。

这一页回答的是：为什么系统会从“复制 batch / 切状态”继续走向“切层”或“切算子”，以及 Pipeline 和 Tensor Parallel 各自在解决什么约束。

## 问题起点

当模型本体已经大到不能只靠数据并行和状态切分解决时，系统就会继续问：

- 能不能把不同层放到不同卡上？
- 能不能把同一层内部的矩阵乘分到多卡上？

这就是 Pipeline 和 Tensor Parallel 的起点。

## 你要先确认什么

- 当前问题是层级太深、单层太大，还是单卡算子已经放不下。
- 你更能接受 pipeline bubble，还是更频繁的张量同步。
- micro-batch 调度是否已经会影响整体效率。

## 核心矛盾

Pipeline 的核心矛盾是：层被切开后，计算可以分摊，但不同阶段会产生 bubble 和排队；Tensor Parallel 的核心矛盾是：单层内部被切开后，算子能扩开，但同步会变得更频繁。

## 演化路径

1. 先判断切层还是切算子。
2. 若切层，就进入 micro-batch、bubble 和 stage balance。
3. 若切算子，就进入张量切分、AllGather / ReduceScatter 等通信代价。
4. 最后用 profiling 和 benchmark 比较哪种代价更能接受。

## 关键取舍

- Pipeline 更怕阶段不平衡和气泡。
- Tensor Parallel 更怕高频同步和小 batch 下通信吞噬收益。
- 两者都不是“切开就更快”，而是“切开后换一种成本”。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- Pipeline parallel 资料：理解 micro-batch 和 bubble。
- Tensor parallel 资料：理解张量切分和通信同步的对应关系。

## 对应 Part 02

- `28` Pipeline Parallelism MicroBatch
- `29` Tensor Parallelism Sim
- `46` Communication Profiling with NCCL

## 典型阅读入口

- [05 Expert Parallel and Communication Hotspots](./05_expert_parallel_and_communication_hotspots.md)
- [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md)

## 本节要点

Pipeline 主要改变层级和时序，Tensor Parallel 主要改变单层算子；二者都必须结合 batch、拓扑和通信频率评估，不能只按“能否切开”判断。

## 进入下一页

如果切分对象进一步变成动态路由的专家，进入 [05 Expert Parallel 与通信热点](./05_expert_parallel_and_communication_hotspots.md)；否则直接带着候选方案进入 [06 Benchmark 与并行决策](./06_benchmark_and_parallel_decision.md)。

Pipeline 和 Tensor Parallel 都是在继续扩模型本体，但一个主要付出 bubble，另一个主要付出同步。
