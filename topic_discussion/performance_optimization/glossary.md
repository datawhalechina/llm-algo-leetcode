# 性能优化专题术语表

本表统一解释测量、诊断、候选和组合决策中的术语。定义服务于当前问题，首次参与机制解释时仍需在正文说明；公式、单位和假设见[公式索引](formula_reference.md)，中英文写法见[术语规范](../../docs/terminology.md)。

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 指标 | metric | 描述时间、吞吐、容量、失败或扩展结果的带单位观测量 | 单个指标变化不能直接证明根因 | [01 测量问题](methods/01_why_profiling_matters.md) |
| 工作负载 | workload | 模型、输入、形状、批次、阶段和执行条件的固定组合 | 不能只用数据集或模型名代替 | [02 可重复基线](methods/02_reproducible_baseline.md) |
| 预热 | warmup | 正式测量前用于稳定初始化、缓存和编译状态的运行 | 不能把预热结果混入正式样本 | [02 可重复基线](methods/02_reproducible_baseline.md) |
| 执行轨迹 | trace | 带时间戳的事件、资源与依赖记录 | 热点汇总不包含完整先后和等待关系 | [03 时间线与关键路径](methods/03_time_breakdown_and_trace_reading.md) |
| 性能剖析 | profiling | 将时间或资源归到阶段、算子、计算内核、状态或调用栈的测量过程 | 性能剖析结果不是瓶颈结论 | [04 对象驻留](methods/04_memory_timeline_and_residency.md) |
| 关键路径 | critical path | 在事件持续时间与依赖完整的模型中，决定完成时间的最长依赖链 | 缺少资源顺序或等待时只得到下界；单个累计耗时最大的事件不一定在关键路径上 | [Part 02 · PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) |
| 暴露等待 | exposed wait | 未被其他有效工作隐藏、实际延长关键路径的等待时间 | 原始通信或搬运持续时间可能被重叠隐藏 | [03 时间线与关键路径](methods/03_time_breakdown_and_trace_reading.md) |
| 对象驻留 | object residency | 参数、激活、缓存或缓冲区在设备上存在的时间区间 | 对象字节数之和不等于真实峰值 | [04 对象驻留](methods/04_memory_timeline_and_residency.md) |
| 活跃分配 / 分配器保留 | allocated / reserved | 已分配给活跃对象的空间，以及分配器保留以便复用的空间 | 两项独立最大值之差不是同一时刻的空闲保留量；差值也不直接等于碎片 | [04 显存观测](methods/04_memory_timeline_and_residency.md) |
| 算术强度 | arithmetic intensity | 每搬运一个字节完成的浮点运算量 | 理论 FLOPs 不能代替实际吞吐 | [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) |
| 机器平衡点 | machine balance | 匹配计算路径的理论计算吞吐与存储层级带宽上界之比，单位 FLOP/byte | 只给出理想 Roofline 分界，不直接说明端到端瓶颈 | [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) |
| 经验参考 | empirical reference | 在指定形状、精度、预热和计时范围下测得的参考吞吐或带宽 | 其他形状的观测可略高于参考；相对参考比例不是硬件利用率 | [05 资源参考](methods/05_bottleneck_attribution.md) |
| 瓶颈归因 | bottleneck attribution | 用时间线、资源指标和判别实验确定限制目标指标的主约束 | 热点排序或低利用率只形成候选解释 | [05 瓶颈归因](methods/05_bottleneck_attribution.md) |
| 性能回归 | performance regression | 候选相对基线出现超过噪声和预设阈值的退化 | 单次变慢或均值变化不足以判定回归 | [Part 02 · PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) |
| 配对比值 | paired ratio | 同一对候选耗时除以基线耗时；延迟类指标小于 1 表示该对更快 | 配对比值中位数不等于两组耗时中位数之比 | [06 配对统计](methods/06_benchmark_design_and_regression_validation.md) |
| 置信区间 | confidence interval | 在指定采样假设和置信水平下，用于推断目标统计量的区间 | 不是原始样本的取值范围，也不替代有效样本数和相关性检查 | [06 区间与门槛](methods/06_benchmark_design_and_regression_validation.md) |
| 回归通过 / 收益达标 | regression pass / benefit validation | 分别表示未超过允许退化范围，以及达到预先要求的改善 | 允许小幅变慢的回归检查通过不等于加速，二者门槛分别记录 | [公式索引 · 配对判断](formula_reference.md#配对回归、收益与失败分母) |
| 证据等级 | evidence level | 区分机制计算、单卡实测、系统追踪和项目决策的证据强度 | 工具越底层不表示结论自动越强 | [07 诊断交接](methods/07_diagnosis_and_action_decision.md) |
| 目标指标 | target metric | 本轮直接希望改善的指标，明确单位、统计方法、测量区间与工作负载 | 同时观察到的所有指标 | [01 明确优化目标与比较条件](./01_performance_problem_and_benchmark.md) |
| 样本拼接 | packing | 将较短样本拼入同一序列，减少填充，同时保持标签与注意力边界符合训练要求 | 仅按长度分桶；无条件跨样本共享上下文 | [03 训练输入、状态与资源账本](./03_training_memory_ledger.md) |
| 守护条件 | guardrail | 候选必须同时满足的正确性、质量、失败率、容量或成本要求 | 目标指标本身 | [01 明确优化目标与比较条件](./01_performance_problem_and_benchmark.md) |
| 比较基线 | baseline | 与候选保持输入和测量条件可比、仅有预期改动差异的参照方案 | 任意一次历史运行；要求主动变化的精度或配置也相同 | [01 明确优化目标与比较条件](./01_performance_problem_and_benchmark.md) |
| 候选机制 | candidate mechanism | 针对已知约束提出、可通过配对实验被证伪的改动 | 尚未说明作用对象的优化建议 | [02 资源约束与候选代价](./02_observability_and_profiling.md) |
| 主约束 | primary constraint | 当前最直接限制目标指标的资源上界、容量边界或关键路径依赖 | 累计耗时最大的算子 | [02 资源约束与候选代价](./02_observability_and_profiling.md) |
| 候选假设 | candidate hypothesis | 指出要改变的对象、预计减少量、新增开销，以及哪些对照结果会使预期不成立 | 只有名称和开关的优化建议 | [PERF-CANDIDATE](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb) |
| 候选代价 | candidate cost | 候选新增的计算、搬运、同步、内存、质量或运维负担 | 只记录候选收益 | [PERF-CANDIDATE](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb) |
| 代价转移 | cost transfer | 候选减少一种开销，却把时间或容量压力转移到另一阶段或资源 | 主约束已经消失 | [PERF-CANDIDATE](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.ipynb) |
| 暴露时间 | exposed time | 未被有效工作隐藏、实际延长关键路径的计算、通信、搬运或等待时间 | 某事件的原始持续时间 | [PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) |
| 执行路径证据 | execution-path evidence | 证明候选实际命中目标加载器、计算内核、数据类型、缓存或调度路径的记录 | 配置中声明启用了候选 | [06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| 单候选准入 | single-candidate gate | 在组合策略前，独立检查一个候选的正确性、质量、性能、容量和成本 | 看到局部速度提升即接受 | [06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| 策略交互 | strategy interaction | 多个候选同时启用后出现的收益叠加、抵消、资源争用或新依赖 | 单项结果的简单相加 | [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) |
| 边际收益 | marginal gain | 在已经启用其他候选后，再加入一个候选带来的实测增量；采用比较通常以同轮满足共同质量与资源要求的最佳单项为参照，注明使用耗时降幅、吞吐增量还是其他指标 | 相对原基线的全部收益；两个单项收益的算术和；与超出资源预算的最快单项直接比较采用价值 | [07 策略组合与瓶颈转移](./07_end_to_end_performance_engineering.md) |
| 活动字节 | active bytes | 本专题局部矩阵中，一个候选的参数字节与输入缓冲区字节之和 | 包含所有激活、工作区与其他候选的运行时峰值 | [实验资产表](./experiment_assets.md#机制实验) |
| 矩阵分配峰值 | matrix peak allocated | 多组候选模型和缓冲区共同驻留、同次矩阵测量中的已分配显存峰值 | 单个候选独立运行的显存峰值；分配器保留量 | [实验资产表](./experiment_assets.md#机制实验) |
| 微批量与有效批量 | microbatch / effective batch | 微批量是一次前向与反向处理的样本量；有效批量还取决于累积次数与数据并行规模 | 只增加累积次数就能减少单次激活或梯度存储 | [03 训练输入、状态与资源账本](./03_training_memory_ledger.md) |
| 固定负载收益 | fixed-workload gain | 在相同输入、批量或并发与质量要求下，由受控改动带来的性能变化 | 增大并发后新增的吞吐 | [04 推理状态容量](./04_inference_memory_and_kv_cache.md) |
| 扩容收益 | capacity-enabled gain | 状态占用减少后，在独立容量实验中提高批量或并发得到的吞吐或承载提升 | 同一负载下的直接加速；基线 OOM 时的无限加速比 | [04 推理状态容量](./04_inference_memory_and_kv_cache.md) |
| 瓶颈转移 | bottleneck shift | 原主约束减弱后，另一资源、阶段或依赖成为新的主要限制 | 优化失败或性能回归 | [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) |
| 容量门槛 | capacity gate | 显存、内存、缓存、队列或并发上限必须满足的硬约束 | 平均利用率目标 | [06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| 回归阈值 | regression threshold | 结合噪声分布和业务容忍度预先设定的退化边界 | 单次样本差值 | [PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) |
| 最小收益与允许退化 | minimum gain / allowed regression | 最小收益规定主目标至少改善多少，允许退化规定其他指标最多变差多少；容量优化可允许一定耗时增加，两种门槛在测量前分别约定 | 任意变慢都拒绝；守护指标未退化就算达到主目标 | [06 单候选执行与准入](./06_cost_model_and_production_decision.md) |
| 回退目标 | rollback target | 组合失败后明确恢复到基线、某个稳定单项候选或上一已接受组合 | 只有失败条件而没有恢复对象 | [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb) |
| 回退条件 | rollback condition | 候选触发质量、正确性、容量、成本或稳定性失败后恢复到既定目标的条件 | 事后人工判断 | [07 策略组合与瓶颈转移](./07_end_to_end_performance_engineering.md) |
| 条件化结论 | conditional conclusion | 只对已验证的模型、输入、硬件、后端和测量方法成立的结论 | 对所有环境都成立的普遍结论 | [实验资产表](./experiment_assets.md) |

## 使用规则

- 首次出现采用“中文主称（English）”，后续使用中文主称或表中通用缩写。
- `CPU`、`GPU`、`CUDA`、`I/O`、`TTFT`、`TPOT`、`P50`、`P95`、`P99` 可直接使用。
- 工具名、API、命令、配置键和 JSON 字段保留原始英文；普通叙述使用“工作负载、基线、候选、执行轨迹、性能剖析、计算内核”。
- 指标说明必须带单位、聚合方法、测量范围和工作负载，不能只写“更快”“显存更低”或“利用率更高”。
- 术语修订逐处判断，不对 Notebook 代码、工具输出或结果 JSON 做无上下文批量替换。
