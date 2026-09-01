# 03. State Sharding and ZeRO | 状态切分与 ZeRO

## 页面目标

这一页回答的是：为什么系统会从朴素数据并行走向 ZeRO，以及 ZeRO 真正切开的是什么。

本页的输出是状态切分决策：显存节省来自哪类状态、额外通信发生在哪里，以及这笔通信成本是否值得接受。

## 问题起点

很多时候，加卡后显存还是紧，因为参数、梯度和 optimizer state 仍然在每张卡上重复驻留。ZeRO 的出现，就是因为“复制计算”解决不了“状态冗余”。

## 你要先确认什么

- 当前显存瓶颈是参数、梯度，还是 optimizer state。
- 数据并行本身是否已经足够，只是状态复制太浪费。
- 你能接受多少额外通信来换状态切分收益。

## 核心矛盾

ZeRO 的核心矛盾是：它通过切分状态换显存，但切分后的访问、收集和同步会增加新的通信和管理成本。

## 演化路径

1. 先从数据并行识别状态重复问题。
2. 再把参数、梯度和 optimizer state 拆开看。
3. 用 ZeRO 把这些状态按阶段切分。
4. 最后回到 benchmark，判断显存收益是否值得对应时间代价。

## 关键取舍

- 显存越紧，ZeRO 的吸引力越大。
- 通信越贵，ZeRO 的副作用越明显。
- ZeRO 解决的是“状态驻留”，不是所有并行问题。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- ZeRO 经典论文：理解状态切分为何会显著改变显存账本。

## 对应 Part 02

- `06` VRAM Calculation and ZeRO
- `27` ZeRO Optimizer Sim
- `73 / 79` 训练与分布式 benchmark

## 典型阅读入口

- [02 Data Parallel and Synchronization](./02_data_parallel_and_synchronization.md)
- [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md)

## 本节要点

ZeRO 不是新的训练语义，而是把重复驻留的状态拆开，以通信成本换显存空间。

## 进入下一页

如果状态切分仍无法容纳模型，进入 [04 Pipeline 与 Tensor Parallel](./04_pipeline_and_tensor_parallel.md)；如果显存已满足预算，则回到 benchmark 比较吞吐和扩展效率。
