# 05. Expert Parallel and Communication Hotspots | Expert Parallel 与通信热点

## 页面目标

这一页回答的是：为什么 MoE / Expert Parallel 会把通信问题再推高一层，以及 profiling 在并行专题里为什么必须存在。

本页的输出是动态并行归因：路由不均、专家间通信、拓扑拥塞和 overlap 分别贡献了什么代价。

## 问题起点

当系统走到 MoE 或 expert parallel 时，并行已经不再只是“切层”或“切算子”，而开始涉及：

- token 到 expert 的动态路由；
- expert 分布不均；
- 通信热点和等待时间进一步复杂化。

这时如果不做 profiling，很容易只知道“变慢了”，却不知道慢在哪里。

## 你要先确认什么

- 问题是来自 expert routing，还是来自更底层的 NCCL / 同步等待。
- 热点是持续性的，还是只在某些 batch / 路由模式下出现。
- 你要解决的是路由不均、通信拥塞，还是 overlap 不足。

## 核心矛盾

Expert Parallel 的核心矛盾是：稀疏计算可以减少无效算力，但动态路由会把通信和负载不均重新放大。于是，系统节省的不是所有成本，只是把成本换到了更难观测的位置。

## 演化路径

1. 先识别是否已经进入 expert parallel / 动态路由问题。
2. 再用 profiling 把热点定位到同步、等待、路由还是 load imbalance。
3. 最后回到 benchmark，看这种复杂并行是否真的划算。

## 关键取舍

- MoE / expert parallel 能换来更大模型规模，但通信复杂度会显著上升。
- profiling 在这里不是辅助工具，而是主诊断入口。
- 如果热点解释不清，继续加卡或继续加 expert 通常只会把问题放大。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- MoE / expert parallel 资料：理解稀疏路由为什么天然带来通信与负载不均。
- NCCL / profiling 工程资料：理解等待时间和 overlap 如何被量化。

## 对应 Part 02

- `46` Communication Profiling with NCCL
- `47` MoE Expert Parallel
- `80` MoE Expert Parallel Benchmark

## 典型阅读入口

- [04 Pipeline and Tensor Parallel](./04_pipeline_and_tensor_parallel.md)
- [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md)

## 本节要点

Expert Parallel 往往不是“通信更多”这么简单，而是“通信更动态、更不均、更依赖 profiling 才看得清”。

## 进入下一页

把专家负载、通信热点和单卡/多卡基线带入 [06 Benchmark 与并行决策](./06_benchmark_and_parallel_decision.md)，确认 MoE 并行是否真的换回了可接受收益。
