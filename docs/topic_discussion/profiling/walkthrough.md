# 性能分析（Profiling）深入阅读

性能分析处理的不是某个工具的操作步骤，而是一条从现象到行动的证据链：

`问题定义 → 基线测量 → 假设 → 时间 / 内存 / 通信证据 → 对照实验 → 回归验证`

它是一条横向支撑路线。训练显存问题回到 `73 → 76 → 75 → 74`，推理问题回到 `66 → 68 → 69 → 70`，多卡问题回到 `79 → 80 → 81`；Profiling 负责帮助这些项目定位瓶颈和检查证据是否充分。

## 第一段：先定义问题，再选择工具

“系统变慢”不是足够精确的问题描述。先确认下降的是 step time、吞吐、TTFT、TPOT、端到端延迟、显存余量还是多卡扩展效率，再固定输入形状、并发、warmup、迭代次数和软件环境。

对应 [01 为什么需要性能分析](./01_why_profiling_matters.md)。该页负责把现象转换成可测指标和待验证假设。

## 第二段：建立可重复的基线

先用轻量计时和显存 API 取得 baseline，再决定是否需要 trace。`time.perf_counter()` 适合端到端耗时，`torch.cuda.Event` 适合 GPU 区间计时，CUDA 测量需要正确同步；`nvidia-smi` 适合观察进程级显存和利用率，不等同于 kernel 时间。

对应 [02 时间拆分与 Trace 阅读](./02_time_breakdown_and_trace_reading.md)。先有基线，后面的热点和时间线才有比较对象。

## 第三段：从时间线提出可证伪的假设

`torch.profiler` 可以帮助区分 CPU / CUDA 时间、调用次数、shape 和显存活动；Chrome Trace 可以观察阶段顺序、同步和搬运。看到一个热点只说明它值得检查，不足以证明它是端到端瓶颈。

如果峰值显存或驻留状态发生变化，转入 memory timeline / snapshot；如果多卡出现空洞或扩展效率下降，继续检查 collective、同步和 overlap。

对应 [03 显存时间线与驻留状态](./03_memory_timeline_and_residency.md) 和 [04 通信等待与 Overlap](./04_communication_wait_and_overlap.md)。

## 第四段：把假设和优化动作对齐

不同证据指向不同路线：算子热点可能进入算子或编译优化，activation 或 KV Cache 峰值进入显存优化，请求排队和 decode 阶段进入推理优化，collective 等待进入通信与并行。Profiling 负责归因和分流，不替代这些专题对策略的定义。

对照实验只能修改目标变量。例如要验证 checkpoint，应保持模型、dtype、batch、序列长度、训练步数和数据不变；要验证推理 backend，则保持 prompt、输出长度、并发和采样参数一致。

## 第五段：用回归验证收口

优化后至少复核三类结果：

- 目标指标是否改善；
- 质量、稳定性、显存或其他阶段是否退化；
- 在相同条件和重复运行下，结果是否仍然成立。

对应 [05 基准测试设计与回归验证](./05_benchmark_design_and_regression_validation.md) 和 [06 从诊断到行动决策](./06_diagnosis_and_action_decision.md)。最终输出应包含 trace 或表格、baseline / candidate、环境和行动理由，而不是只有一张截图。

## 工具升级路径

工具按问题粒度升级：

`轻量计时 → torch.profiler → Nsight Systems → Nsight Compute / NCCL`

升级条件是当前证据无法回答下一个问题，而不是工具越高级越好。CPU 可以完成计时接口、阶段标记和报告逻辑练习；真实 GPU 才能验证 CUDA kernel、GPU 显存、Tensor Core 和 stream overlap；多卡证据还需要相应的分布式环境。
