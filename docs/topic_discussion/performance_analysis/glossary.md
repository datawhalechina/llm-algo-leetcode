# 性能分析专题术语表

本表解释性能分析中容易混淆的测量对象与判断。符号、单位和计算条件见[公式索引](./formula_reference.md)，中英文写法见[术语规范](../../terminology.md)；代码标识符、工具名称、命令参数和结果 JSON 字段保留原始英文。

| 中文主称 | 英文 / 缩写 | 本专题含义 | 容易混淆之处 | 学习入口 |
|:---|:---|:---|:---|:---|
| 指标 | metric | 描述时间、吞吐、容量、失败或扩展结果的带单位观测量 | 单个指标变化不能直接证明根因 | [01 测量问题](./01_why_profiling_matters.md) |
| 工作负载 | workload | 模型、输入、形状、批次、阶段和执行条件的固定组合 | 不能只用数据集或模型名代替 | [02 可重复基线](./02_reproducible_baseline.md) |
| 基线 / 候选 | baseline / candidate | 在相同工作负载和环境下配对比较的参照与待评估方案 | 两次未对齐条件的独立运行 | [06 回归验证](./06_benchmark_design_and_regression_validation.md) |
| 预热 | warmup | 正式测量前用于稳定初始化、缓存和编译状态的运行 | 不能把预热结果混入正式样本 | [02 可重复基线](./02_reproducible_baseline.md) |
| 执行轨迹 | trace | 带时间戳的事件、资源与依赖记录 | 热点汇总不包含完整先后和等待关系 | [03 时间线与关键路径](./03_time_breakdown_and_trace_reading.md) |
| 性能剖析 | profiling | 将时间或资源归到阶段、算子、计算内核、状态或调用栈的测量过程 | 性能剖析结果不是瓶颈结论 | [04 对象驻留](./04_memory_timeline_and_residency.md) |
| 关键路径 | critical path | 在事件持续时间与依赖完整的模型中，决定完成时间的最长依赖链 | 缺少资源顺序或等待时只得到下界；单个累计耗时最大的事件不一定在关键路径上 | [Part 02 · PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.md) |
| 暴露等待 | exposed wait | 未被其他有效工作隐藏、实际延长关键路径的等待时间 | 原始通信或搬运持续时间可能被重叠隐藏 | [03 时间线与关键路径](./03_time_breakdown_and_trace_reading.md) |
| 对象驻留 | object residency | 参数、激活、缓存或缓冲区在设备上存在的时间区间 | 对象字节数之和不等于真实峰值 | [04 对象驻留](./04_memory_timeline_and_residency.md) |
| 活跃分配 / 分配器保留 | allocated / reserved | 已分配给活跃对象的空间，以及分配器保留以便复用的空间 | 两项独立最大值之差不是同一时刻的空闲保留量；差值也不直接等于碎片 | [04 显存观测](./04_memory_timeline_and_residency.md) |
| 算术强度 | arithmetic intensity | 每搬运一个字节完成的浮点运算量 | 理论 FLOPs 不能代替实际吞吐 | [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.md) |
| 机器平衡点 | machine balance | 匹配计算路径的理论计算吞吐与存储层级带宽上界之比，单位 FLOP/byte | 只给出理想 Roofline 分界，不直接说明端到端瓶颈 | [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.md) |
| 经验参考 | empirical reference | 在指定形状、精度、预热和计时范围下测得的参考吞吐或带宽 | 其他形状的观测可略高于参考；相对参考比例不是硬件利用率 | [05 资源参考](./05_bottleneck_attribution.md) |
| 瓶颈归因 | bottleneck attribution | 用时间线、资源指标和判别实验确定限制目标指标的主约束 | 热点排序或低利用率只形成候选解释 | [05 瓶颈归因](./05_bottleneck_attribution.md) |
| 守护条件 | guardrail | 候选必须同时满足的正确性、质量、失败率或资源边界 | 目标指标改善不能覆盖守护条件失败 | [06 回归验证](./06_benchmark_design_and_regression_validation.md) |
| 性能回归 | performance regression | 候选相对基线出现超过噪声和预设阈值的退化 | 单次变慢或均值变化不足以判定回归 | [Part 02 · PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md) |
| 配对比值 | paired ratio | 同一对候选耗时除以基线耗时；延迟类指标小于 1 表示该对更快 | 配对比值中位数不等于两组耗时中位数之比 | [06 配对统计](./06_benchmark_design_and_regression_validation.md) |
| 置信区间 | confidence interval | 在指定采样假设和置信水平下，用于推断目标统计量的区间 | 不是原始样本的取值范围，也不替代有效样本数和相关性检查 | [06 区间与门槛](./06_benchmark_design_and_regression_validation.md) |
| 回归通过 / 收益达标 | regression pass / benefit validation | 分别表示未超过允许退化范围，以及达到预先要求的改善 | 允许小幅变慢的回归检查通过不等于加速，二者门槛分别记录 | [公式索引 · 配对判断](./formula_reference.md#配对回归、收益与失败分母) |
| 证据等级 | evidence level | 区分机制计算、单卡实测、系统追踪和项目决策的证据强度 | 工具越底层不表示结论自动越强 | [07 诊断交接](./07_diagnosis_and_action_decision.md) |

## 使用规则

- 首次出现采用“中文主称（English）”，后续使用中文主称或表中通用缩写。
- `CPU`、`GPU`、`CUDA`、`I/O`、`TTFT`、`TPOT`、`P50`、`P95`、`P99` 可直接使用。
- 工具名、API、命令、配置键和 JSON 字段保留原始英文；普通叙述使用“工作负载、基线、候选、执行轨迹、性能剖析、计算内核”。
- 指标说明必须带单位、聚合方法、测量范围和工作负载，不能只写“更快”“显存更低”或“利用率更高”。
- 术语修订逐处判断，不对 Notebook 代码、工具输出或结果 JSON 做无上下文批量替换。
