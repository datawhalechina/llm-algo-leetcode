# 性能分析判断手册 | Profiling Casebook

先固定输入、阶段、环境与基线，保存目标指标、质量要求、逐次样本和失败原因，再按下表寻找第一个检查对象。缺少条件时先补测，有时间线或驻留记录后再选择判别对照。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 训练步或请求延迟上升 | 阶段时间、P50/P99、TTFT/TPOT 与工作负载是否完整 | 带阶段标记的性能剖析轨迹 | [正文 03：时间线、I/O 与关键路径](./03_time_breakdown_and_trace_reading.md) | [正文 05：瓶颈归因](./05_bottleneck_attribution.md) |
| GPU 出现空闲段 | CPU 准备、H2D、同步和 GPU 计算的先后关系 | 同一时间轴上的主机与设备事件 | [Part 00 · 17 性能分析基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb) | [性能优化 · 05 计算、通信与 I/O](../performance_optimization/05_compute_communication_and_io.md) |
| 峰值显存抬高或发生 OOM | `allocated`、`reserved`、峰值阶段与对象生命周期 | 显存时间线或显存快照（snapshot），标出同时驻留对象 | [正文 04：显存时间线与对象驻留](./04_memory_timeline_and_residency.md) | [性能优化 · 03 训练显存账本](../performance_optimization/03_training_memory_ledger.md) |
| 多卡没有线性扩展 | 每卡阶段时间、集合通信与通信暴露时间 | 每个进程的阶段时间与集合通信轨迹 | [Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb) | [通信与并行专题](../communication_parallelism/intro.md) |
| 候选看起来更快 | 基线与候选是否只改变一个变量，质量是否通过要求 | 配对重复样本、区间、失败率与质量记录 | [正文 06：基准设计与回归验证](./06_benchmark_design_and_regression_validation.md) | [性能优化 · 06 单候选准入](../performance_optimization/06_cost_model_and_production_decision.md) |
| 已完成归因，需要决定下一步 | 证据是否支持保留、继续调参或回退 | 假设、证据位置、适用工作负载与回归结果 | [正文 07：从归因到行动](./07_diagnosis_and_action_decision.md) | [性能优化专题入口](../performance_optimization/intro.md) |
