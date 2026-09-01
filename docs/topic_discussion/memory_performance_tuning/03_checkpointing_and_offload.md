# 03. Checkpointing and Offload | Checkpointing 与 Offload

## 页面目标

本节回答的是：为什么 checkpointing 和 offload 会成为训练显存优化主线，以及它们分别在拿什么换空间。

## 问题起点

当确认 activation 是训练峰值主因后，针对这类瓶颈主要有两条路线：

- 少存一些，回头再算；
- 先存着，但搬到别的存储层。

这正是 checkpointing 和 offload 的本质区别。它们都在处理 activation 驻留，却通过完全不同的代价模型完成；它们不是训练显存优化的全部方案。

## 你要先确认什么

- activation 是否已经是训练峰值主因。
- 当前系统更能接受重算，还是更能接受搬运。
- 带宽、PCIe / NVLink 路径是否足以支撑 offload。

## 核心矛盾

`checkpointing` 的矛盾是“少留状态，但后向时要多做一次前向片段重算”；`offload` 的矛盾是“状态仍然存在，但需要跨存储层搬运并等待返回”。两者都不是免费收益，差别只在于你把代价付给计算还是传输。

## 演化路径

1. 先识别 activation 是否值得优化。
2. 用 checkpointing 把一部分状态从“存储”改成“重算”。
3. 用 offload 把状态从 GPU 驻留改成外部存储层驻留。
4. 再用 profiling 看重算和搬运是否把时间赔过头。

## 从机制到项目证据

本节的关键不是背诵 checkpoint 和 offload 的定义，而是为每个候选方案写清楚三件事：减少了哪类 GPU 驻留、把代价转移到重算还是搬运、需要用哪一个指标证明代价可接受。对应项目链是 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) 先建立 baseline，再用 [76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) 比较候选，由 [75 Memory Budget Compression](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) 按预算门槛做选择，最后由 [74 Profiling Driven Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) 检查重算、搬运和 kernel 代价是否真的解释了结果。

## 关键取舍

- `checkpointing` 更适合计算相对便宜、重算可接受的片段。
- `offload` 更适合显存太紧但外部带宽还可承受的环境。
- 两者可以组合，但组合后更需要 benchmark，不然很容易只看到显存下降，看不到训练时间恶化。

## 先看对象，再看策略

| 如果主要问题是 | 优先考虑 | 不要误判为 |
|---|---|---|
| 单个 micro-batch 的激活峰值 | Gradient Accumulation 或减小 micro-batch | 它不会减少参数和优化器状态 |
| 反向所需的中间激活过多 | Checkpointing | 它不会自动减少总参数显存 |
| GPU 激活驻留超过预算，但主机带宽可承受 | Offload | 模拟中的传输时间不是实际 PCIe/NVLink 测量 |

`42 Activation Offload` 只做可读的预算模拟：它根据激活块大小、预计复用距离和假设带宽计算“理论上搬多少、估算多久”。它不创建真实 CUDA tensor，也不实现 pinned memory、异步拷贝或传输重叠。因此它只能帮助学习者建立代价模型，不能替代 `76` 的真实 GPU benchmark。

本节的出口不是选出一个永远最优的策略，而是写出一条可验证的假设：例如“checkpoint 预计减少激活驻留，但允许不超过某个吞吐损失”；随后由 `73 / 76 / 75` 用同一 workload 验证，再由 `74` 检查假设是否与 profiler 证据一致。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- activation checkpointing 经典论文：理解时间换空间的基本模式。
- activation offload / memory hierarchy 相关资料：理解搬运路径为什么常成为隐藏成本。

## 对应 Part 02

- [19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md)
- [42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.md)
- [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)、[76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)、[74 Profiling Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)

## 典型阅读入口

- [02 Training Memory Pressure](./02_training_memory_pressure.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 本节要点

checkpointing 和 offload 都是在省 activation，但一个主要赔计算，一个主要赔传输。
