# 06. Benchmark and Parallel Decision | Benchmark 与并行决策

## 页面目标

这一页负责把前面的并行判断收束到 benchmark：这套并行方案到底值不值得继续保留。

本页的输出是并行交付决策：明确显存、吞吐、延迟、通信占比、扩展效率和复杂度之间的取舍，并选择 `keep / tune / reject`。

## 问题起点

并行专题里最常见的误判是：显存降了、卡数多了，所以一定更好。工程上真正要问的是：

- step time、throughput 和扩展效率有没有真正改善；
- 通信等待、bubble、load imbalance 有没有把收益吃掉；
- 这套方案换来的规模收益，是否值得它引入的复杂度。

## 你要先确认什么

- workload 是否固定。
- baseline 和 candidate 是否只改一个关键变量。
- 你的目标更偏显存、吞吐，还是训练时长。

## 为什么 benchmark 是最后一页

没有验证页，通信并行专题就会停在“方法分类”。有了这一页，才会回到真正的工程判断：这套切分方案是否真的带来了可复用收益。

## 判定原则

- `keep`：收益不明显，或者通信 / 调度代价太高。
- `tune`：方向对，但切分粒度、micro-batch、overlap 或路由还要继续调。
- `switch`：收益稳定，并且和显存、吞吐或训练时长目标匹配。

## 报告应该怎么写

一个合格的并行方案报告至少要同时说明：

- 你切的是状态、层、算子还是 expert；
- 显存、step time、throughput、等待时间分别怎么变；
- 通信热点是否解释了收益变化；
- 最终是继续保留、继续调优，还是换方案。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献与工程入口

- `46` Communication Profiling with NCCL
- `79` Distributed Parallel Benchmark
- `66` Inference Performance Comparison

## 典型阅读入口

- [03 State Sharding and ZeRO](./03_state_sharding_and_zero.md)
- [04 Pipeline and Tensor Parallel](./04_pipeline_and_tensor_parallel.md)
- [05 Expert Parallel and Communication Hotspots](./05_expert_parallel_and_communication_hotspots.md)

## 项目结论

并行路线最终不是靠“方法名更复杂”成立，而是靠 benchmark 和通信热点解释成立。

## 回到项目

将结论回填到 `79 分布式并行 benchmark -> 80 MoE 专家并行 benchmark -> 81 分布式推理项目`。单卡模拟只能解释机制；真实多卡结论必须保留 world size、拓扑、通信库和 workload。
