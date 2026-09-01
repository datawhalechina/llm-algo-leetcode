# 01 为什么 Profiling 值得单独成章

## 页面目标

本节先解释 profiling 的目标：不是“多看图”，而是把一个性能猜测变成证据链。

本节的输出是一份最小问题定义：现象、影响指标、可能边界和需要采集的证据。完成后再进入时间拆分，而不是直接猜具体 kernel。

## 问题起点

训练和推理里的“慢”，往往有很多表象：

- step time 变长
- TTFT 变高
- 多卡扩展不稳
- 显存降了但吞吐也掉了

如果没有 profiling，很多结论都停留在“怀疑某个模块慢”，而不是“证明确实慢在这里”。

## 核心矛盾

profiling 想得到更可靠的判断，但代价是：

- 采集会更复杂
- 图和表会更多
- 更容易被局部热点误导

所以这条专题的关键，不是把工具列全，而是教人怎样建立一条可靠的诊断链。

## 最小证据链

先写出“现象—指标—假设”，再选择工具。例如“训练变慢”需要先区分 step time 上升、吞吐下降，还是输入管线等待；只有指标确定后，trace 才能验证对应阶段。

| 记录 | 示例 | 作用 |
|:---|:---|:---|
| 现象 | step time 从 200 ms 升到 260 ms | 说明问题范围 |
| 假设 | CPU 到 GPU 的输入搬运增加 | 提供可证伪方向 |
| 证据 | trace 中出现更长的 memcpy 与空闲区间 | 支持或否定假设 |
| 对照 | 固定数据和 batch 后重新测量 | 检查归因是否稳定 |

没有这条链时，profiler 输出只是观察材料，不是瓶颈结论。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- [Part 00 · 0E 调试基础](../../00_Prerequisites/0E.md)、[Part 00 · 17 Profiling 基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)、[Part 00 · 20 Profiling 与显存账本](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb)：Profiling 的入门和前置桥。
- [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)：显存优化路线的最终收口；[Part 02 · 79 分布式并行基准测试](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb) 和 [Part 02 · 46 NCCL 通信性能分析](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb)：分布式和通信场景的延伸。

## 本节要点

profiling 的价值在于：它把“感觉慢”变成“证据证明慢在哪里”。

## 进入下一页

先进入 [02 时间拆分与 Trace 阅读](./02_time_breakdown_and_trace_reading.md)，把抽象的性能问题转成可观察的时间区间。
