# 性能优化判断手册

使用本表前，固定模型与数据版本、运行环境、输入分布、批量或并发、预热与重复规则，以及正确性、质量和 SLO 门槛。保留本次配置、逐次样本、阶段指标、失败状态和实际执行路径。

按现象找到首个检查对象，再收集表中的最小记录。原因不清时补测，已确认原因时选择改动；完成单项对照后再决定是否需要组合。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 训练单步时间变长 | 输入、前向、反向、参数更新和同步阶段 | 同一工作负载的阶段时间、执行轨迹、取批等待和失败记录 | [03 训练输入、状态与资源账本](./03_training_memory_ledger.md) | 归因不清时进入[性能分析](../performance_analysis/intro.md)；候选完成后进入[06 单候选准入](./06_cost_model_and_production_decision.md) |
| 峰值显存升高 | 峰值时仍驻留的参数、激活、梯度、优化器状态、KV、工作区和通信缓冲区 | 对象大小、创建/释放时刻、已分配/保留显存和 OOM 阶段 | [03 训练资源账本](./03_training_memory_ledger.md)或[04 推理状态容量](./04_inference_memory_and_kv_cache.md) | 使用 [PERF-MEMORY-TIMELINE](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.md)重建峰值 |
| 有效批量不变但峰值或单步时间改变 | 微批量、梯度累积、保存激活和更新次数 | 微批量、累积步数、有效词元数、峰值及完成一次更新的总时间 | [03 训练输入、状态与资源账本](./03_training_memory_ledger.md) | 转入[训练机制](../backpropagation_training_mechanism/intro.md)核对更新语义，再回到[06 单候选准入](./06_cost_model_and_production_decision.md) |
| DPO / GRPO 训练慢但 GPU 未持续繁忙 | 多模型驻留、响应长度、轨迹生成、奖励/验证和更新比例 | 采样/验证/更新时间、有效样本率、峰值和质量 | [03 训练资源账本](./03_training_memory_ledger.md)与[05 关键路径](./05_compute_communication_and_io.md) | 算法语义转入[后训练优化](../post_training_optimization/intro.md)，系统候选进入[06 单候选准入](./06_cost_model_and_production_decision.md) |
| 理论账本与 GPU 实测差距很大 | 分配器保留、工作区、碎片、通信缓冲区和生命周期重叠 | 理论字节、已分配/保留峰值、对象时间线 | [03 训练资源账本](./03_training_memory_ledger.md)或[04 推理状态容量](./04_inference_memory_and_kv_cache.md) | 使用[性能分析](../performance_analysis/intro.md)取得运行时内存证据 |
| 模型加载、转换或首请求很慢 | 产物读取、反序列化、格式转换、首次 H2D、编译和预热 | 冷启动分段时间、产物版本与格式、I/O 字节和稳态对照 | [05 计算、通信与 I/O](./05_compute_communication_and_io.md) | 在目标加载路径分段测量；[PERF-IO](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.md)提供局部测量方法，真实首请求仍需目标后端复测 |
| GPU 利用率低 | CPU 等待、输入 I/O、同步、通信和计算空洞 | CPU/GPU 时间线、取批等待、暴露通信和内核间隙 | [05 计算、通信与 I/O](./05_compute_communication_and_io.md) | 使用 [PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.md)确认前驱关系 |
| 已确认单个算子的计算或访存限制 | 输入形状、数据类型、实际内核、启动次数和读写量 | 正确性、局部计时、模型阶段占比及资源参考 | [02 资源约束与候选代价](./02_observability_and_profiling.md) | 转入[算子优化](../operator_optimization/intro.md)比较实现，再回到目标模型工作负载复测 |
| 性能剖析中的耗时明显变差 | 采集范围、同步点、采集开销和独立复测 | 无采集与有采集的配对结果、采集配置和原始样本 | [性能分析](../performance_analysis/intro.md) | 使用 [PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md)判断差异是否超过波动 |
| 吞吐提高但延迟变差 | 批量、队列、并发、TTFT、TPOT 和 P99 | 固定请求分布下的吞吐、阶段延迟、队列和峰值 | [04 推理状态容量](./04_inference_memory_and_kv_cache.md) | 调度机制转入[推理优化](../inference_optimization/intro.md)，候选进入[06 单候选准入](./06_cost_model_and_production_decision.md) |
| KV 交接或网络等待拉长 | 传输字节、序列化、网络、资源池和消费依赖 | 状态字节、传输总时间、暴露时间、TTFT/TPOT/P99 | [05 计算、通信与 I/O](./05_compute_communication_and_io.md) | 多资源池机制转入[推理优化](../inference_optimization/intro.md)，多卡路径转入[通信与并行](../communication_parallelism/intro.md) |
| 轨迹生成快但有效更新吞吐低 | 无效响应、验证器、环境失败、权重刷新和策略新鲜度 | 原始轨迹吞吐、有效样本率、验证/刷新等待和质量 | [05 计算、通信与 I/O](./05_compute_communication_and_io.md) | 转入[后训练优化](../post_training_optimization/intro.md)核对轨迹生命周期，再进入[06 单候选准入](./06_cost_model_and_production_decision.md) |
| 流式输出、日志或检查点拖慢尾部 | 回压、磁盘/网络写入、CPU 编码和同步点 | 输出队列、写入时间、P99、完整性和失败记录 | [05 计算、通信与 I/O](./05_compute_communication_and_io.md) | 使用 [PERF-IO](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.md)验证暴露等待 |
| 量化后显存下降但质量或 P99 退化 | 数据类型、量化粒度、产物、执行后端、计算内核和独立质量集 | 产物版本、目标内核、容量、P99 和质量对照 | [04 推理状态容量](./04_inference_memory_and_kv_cache.md)与[06 单候选准入](./06_cost_model_and_production_decision.md) | 量化机制转入[量化与压缩](../quantization_compression/intro.md)，推理路径转入[推理优化](../inference_optimization/intro.md) |
| 扩大并发后吞吐提高 | 原始请求清单、各方案的并发扫描与服务时限 | 固定并发对照、可用并发上限、峰值、尾延迟、失败与拒绝 | [04 推理状态容量](./04_inference_memory_and_kv_cache.md) | 将同负载收益与扩容收益分开登记；基线 OOM 的配置不计算加速比 |
| 多卡扩展不理想 | 集合通信、拓扑、负载、计算/通信占比和重叠 | 单卡/多卡配对时间、暴露通信、扩展效率和失败恢复 | [05 计算、通信与 I/O](./05_compute_communication_and_io.md) | 转入[通信与并行](../communication_parallelism/intro.md)验证真实多卡机制 |
| 局部策略有效但组合后退化 | 共享资源、执行顺序、重复收益和新关键路径 | 同次实验的基线、单项与组合样本、执行轨迹和回退目标；顺序有影响时另测反向应用 | [07 策略组合与瓶颈转移](./07_end_to_end_performance_engineering.md) | 用 [PERF-COMPOSITION](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.md)检查边际收益，再采集依赖或资源证据判断原因 |
