# 01. Why Parallel and Communication | 为什么需要并行与通信

## 页面目标

这一页回答两个问题：

- 系统为什么会从单卡自然走向多卡并行？
- 为什么一旦走向并行，通信就不再是附属问题，而会变成新的主约束？

本页的输出是并行问题定义：单卡边界是什么、需要摊开的对象是什么，以及预期会引入哪类通信和同步代价。

## 问题起点

并行通常不是从“我想学一种新技术”开始，而是从已有主线里出现的两个约束开始：

- 单卡显存装不下模型、optimizer state、activation 或 batch；
- 单卡算得下，但吞吐和训练时长无法接受。

这两个约束会把系统推向多卡。但只要一跨卡，原本在单卡内部看不见的同步、带宽、等待和调度，就会立刻成为新的问题。

## 你要先确认什么

- 当前问题先是“装不下”，还是“跑不快”。
- 你更缺的是显存，还是整体吞吐。
- 加卡后最可能冒出来的是同步等待、pipeline bubble，还是张量切分通信。

## 核心矛盾

并行的核心矛盾是：通过切分模型、状态或计算把单卡压力摊开，但摊开的代价一定会在通信、同步、调度或空转里回来。通信并行专题的主线，就是在“扩规模”和“扩开销”之间找平衡。

## 演化路径

1. 先识别单卡约束来自显存还是时间。
2. 再决定要切的是状态、层、算子，还是专家。
3. 一旦跨卡，就必须考虑通信原语、拓扑和等待时间。
4. 最后回到 benchmark，判断并行是否真的换来了规模或吞吐收益。

## 关键取舍

- 不加卡，问题留在单卡预算里。
- 一加卡，问题就会迁移到同步和调度。
- 并行不是默认更快，而是默认“更复杂”。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- AllReduce / 数据并行基础资料：理解为什么同步会成为多卡的第一层代价。
- 并行系统资料：理解为什么切分层级会决定通信形态。

## 对应 Part 02

- `27` ZeRO Optimizer Sim
- `28` Pipeline Parallelism MicroBatch
- `29` Tensor Parallelism Sim
- `46` Communication Profiling with NCCL
- `79` Distributed Parallel Benchmark

## 典型阅读入口

- [02 Data Parallel and Synchronization](./02_data_parallel_and_synchronization.md)
- [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md)

## 本节要点

系统走向并行，通常不是因为“并行更高级”，而是因为单卡约束先把你推到了多卡边界。

## 进入下一页

先进入 [02 数据并行与梯度同步](./02_data_parallel_and_synchronization.md)，从最直观的多卡扩展方式开始建立通信基线。
