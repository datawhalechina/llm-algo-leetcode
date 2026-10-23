# 性能优化问题链：从现象到候选决策

跟踪同一组输入的基线与改动：先确定慢或容量不足出现在哪个阶段，再选择能减少相应工作量的策略，最后比较单项和组合。每次新增改动都记录收益与开销，判断最初的限制是否减轻、其他阶段是否成为新的限制。

## 建立比较基线并提出候选

从[性能分析诊断](../performance_analysis/07_diagnosis_and_action_decision.md)取得受限阶段、已排除原因与待比较的改动，再按[比较条件](./01_performance_problem_and_benchmark.md)固定输入、环境、测量方法和质量要求，只改变要检验的配置。需要扩大批量或并发时另建容量实验，不能把工作量变化当作同负载下的直接加速。

![固定条件后比较基线与候选](../../docs/public/topic_discussion/performance_optimization/performance_comparison_contract.svg)

用[资源约束与候选代价](./02_observability_and_profiling.md)把测量转为可检验的假设。例如显存峰值来自保存激活时，重算可能减少驻留，却增加前向计算；预期减少量、新增开销和不成立的条件都应写入比较。

![从归因证据到候选假设](../../docs/public/topic_discussion/performance_optimization/attribution_to_candidate_hypothesis.svg)

## 根据主要约束选择分支

| 已确认的主要约束 | 需要继续解释的关系 | 正文入口 | 返回主线时带回的证据 |
|:---|:---|:---|:---|
| 输入等待或训练状态峰值 | 输入供给、状态生命周期与单步时间怎样相互影响 | [03 训练输入、状态与资源账本](./03_training_memory_ledger.md) | 取批等待、暴露 H2D、峰值对象、单步时间、吞吐与质量 |
| 请求状态容量 | 长度、并发、KV 分配与复用怎样影响服务指标 | [04 推理状态容量与 KV Cache](./04_inference_memory_and_kv_cache.md) | 每词元状态、峰值、命中/驱逐、TTFT、TPOT、P99 与质量 |
| 计算、搬运、通信或排队等待 | 哪个事件位于关键路径，依赖是否允许重叠 | [05 计算、通信与 I/O](./05_compute_communication_and_io.md) | 前驱关系、总时间、暴露时间、资源争用和失败状态 |

### 训练输入与状态

训练分支先检查输入批次何时就绪，再观察参数、激活、梯度、优化器状态、工作区和通信缓冲区在什么阶段同时驻留。

![训练输入供给与状态峰值时间线](../../docs/public/topic_discussion/performance_optimization/training_state_peak_overlap.svg)

模型更新范围和训练目标共同决定状态。全参数训练与 LoRA 的可训练参数不同；偏好优化和强化学习又可能引入参考、奖励或价值模型及轨迹生成。LoRA 也可用于这些训练，因此按实际配置核对模型数量、激活和更新状态。

![不同训练形态的资源路径](../../docs/public/topic_discussion/performance_optimization/post_training_resource_paths.svg)

### 推理状态容量

请求长度和活跃数量决定逻辑状态需求，模型结构与 KV 表示决定每词元字节；两者共同影响实际块分配、可用并发与调度。先在固定请求负载下比较，再扫描新增容量允许的并发，分别记录延迟、吞吐和质量。

![KV 容量、并发与服务延迟关系](../../docs/public/topic_discussion/performance_optimization/kv_capacity_concurrency_latency.svg)

### 跨路径：计算、通信与 I/O

按输入、模型产物、运行时状态和输出回写检查传输与后续计算的依赖。预取或双缓冲只有减少了原来暴露的等待，才可能缩短总时间；主机内存、链路争用与输出完整性同时检查。训练更新、请求调度和集合通信的具体机制分别见[训练机制](../backpropagation_training_mechanism/intro.md)、[后训练优化](../post_training_optimization/intro.md)、[推理优化](../inference_optimization/intro.md)和[通信与并行](../communication_parallelism/intro.md)。

![输入、产物、状态和输出的 I/O 检查路径](../../docs/public/topic_discussion/performance_optimization/io_critical_path.svg)

## 从单候选准入到组合复测

[06 单候选执行与准入](./06_cost_model_and_production_decision.md)先检查候选的实际执行路径，再检查正确性、质量、性能、容量与成本。正确性、质量和资源安全通过的候选可进入组合对照；尚未达到收益目标的 `tune` 候选需保留未通过项，检查组合能否补足收益。正确性失败、质量超限或执行不受支持时先修复或停用，不能用另一项的性能收益抵消失败。最终采用由组合实测和项目门槛决定。

![单候选执行与准入判断](../../docs/public/topic_discussion/performance_optimization/production_admission_gate.svg)

[策略组合与瓶颈转移](./07_end_to_end_performance_engineering.md)在同次实验中比较基线、单项和组合，测量组合相对最佳单项的边际收益。只有先后应用会改变产物或依赖时，才增加顺序对照；随后用执行轨迹或判别实验检查新的限制，并保留可恢复的单项或基线。

![局部候选到组合决策](../../docs/public/topic_discussion/performance_optimization/strategy_combination_decision.svg)
