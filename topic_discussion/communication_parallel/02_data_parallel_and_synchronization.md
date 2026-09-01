# 02. Data Parallel and Synchronization | 数据并行与梯度同步

## 页面目标

这一页回答的是：数据并行在解决什么问题，为什么梯度同步会成为最先出现的通信代价。

本页的输出是同步基线：计算复制带来了多少收益，AllReduce 和等待占据了多少代价，以及何时需要从 DDP 转向状态切分。

## 问题起点

数据并行是最直观的多卡扩展方式：每张卡各算一份 batch，再把梯度同步回来。但它最容易被误解成“多几张卡就线性加速”。真正的边界在于：

- 每张卡虽然都在算，但最后必须同步；
- batch 变大后优化行为会变化；
- 同步等待和通信带宽会逐步吞掉收益。

## 你要先确认什么

- 当前扩卡目标是吞吐，还是只是让 batch 能放下。
- AllReduce / 同步等待是否已经成为主要耗时。
- 有效 batch 扩大后，训练节奏是否仍可接受。

## 核心矛盾

数据并行的核心矛盾是：计算很容易横向复制，但参数更新必须重新汇总。于是，扩卡把算力放大了，也把同步代价放大了。

## 演化路径

1. 先从最朴素的数据并行开始。
2. 然后观察 AllReduce、同步等待和扩卡效率。
3. 如果状态太大，再继续转向 ZeRO 或更细的状态切分。

## 关键取舍

- 数据并行最简单，但同步最直接。
- batch 扩大可能带来吞吐收益，也可能带来优化行为变化。
- 同步问题不解决，后续加更多卡只会继续放大等待。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- AllReduce / NCCL 基础资料：理解同步原语和通信语义。
- 数据并行扩展资料：理解 scaling efficiency 为什么不会天然线性。

## 对应 Part 02

- `20` NCCL and AllReduce Basics
- `27` ZeRO Optimizer Sim
- `46` Communication Profiling with NCCL

## 典型阅读入口

- [03 State Sharding and ZeRO](./03_state_sharding_and_zero.md)
- [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md)

## 本节要点

数据并行的第一收益是复制计算，第一代价是同步更新。

## 进入下一页

如果主要问题是参数、梯度或优化器状态重复驻留，进入 [03 状态切分与 ZeRO](./03_state_sharding_and_zero.md)；如果同步本身已是主瓶颈，先保留通信证据进入 benchmark 验证。
