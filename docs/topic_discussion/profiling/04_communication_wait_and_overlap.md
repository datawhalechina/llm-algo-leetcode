# 04 通信等待与 Overlap

## 页面目标

本节负责把 profiling 的视角扩到多卡：谁在等、哪里没 overlap、为什么理论并行收益没有兑现。

本节的输出是通信归因：等待发生在哪个集体通信或阶段、是否存在可重叠空间，以及问题属于切分策略、拓扑还是 workload。不要把 GPU 利用率低直接等同于算力不足。

## 问题起点

多卡变慢时，常见误判是：

- 以为算子本身慢
- 以为 GPU 利用率低就是单卡问题

但真实情况往往是：

- 同步等待高
- overlap 没生效
- communication hotspot 把收益吃掉了

## 关键观察点

- communication wait
- all-reduce / all-gather hotspots
- pipeline bubble
- overlap 是否存在

## 通信归因的最小条件

先建立单卡或单进程 baseline，再观察 collective 的持续时间、调用频率、等待空洞和计算重叠。`all-reduce`、`all-gather` 或 `all-to-all` 的时间变长，只能说明通信路径发生变化；要判断它是否是扩展效率下降的主因，还要对齐计算时间、消息大小、卡间拓扑和同步点。

因此，“GPU 利用率低”不是充分证据，“通信占比高”也不自动等于通信可以优化。需要用同一 workload 对比单卡、多卡或不同切分方案。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- [Part 02 · 46 NCCL 通信性能分析](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.md)
- [Part 02 · 79 分布式并行基准测试](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md)
- [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)

## 本节要点

多卡 profiling 的关键不是“看更多 trace”，而是解释为什么通信和等待把理想收益吃掉了。

## 进入下一页

把通信归因和单卡基线一起带入 [05 Benchmark 设计与回归验证](./05_benchmark_design_and_regression_validation.md)，确认多卡策略是否真的改善了端到端结果。
