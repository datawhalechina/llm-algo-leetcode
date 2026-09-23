# 08. Distributed Inference and Parallelism | 分布式推理与并行扩展

## 页面目标

当单实例无法同时满足容量、延迟或吞吐目标时，学习如何判断是否需要扩展，以及如何比较并行方式、通信代价和部署收益。本节把 Part 02 的通信机制、并行选型和 79–81 项目串成一条证据链，最后把结果交给 06 做统一决策。

## 为什么需要分布式推理

分布式不是“GPU 数量越多越好”。先建立单实例 baseline，再确认瓶颈来自模型容量、请求容量、计算吞吐还是通信等待；只有增加并行度能够改善目标指标，并且通信与负载不均的代价可控时，扩展方案才值得保留。

| 触发信号 | 可能原因 | 先补什么证据 |
|:---|:---|:---|
| 单卡装不下模型或 KV 状态 | 模型、Cache 或 batch 超出容量 | 峰值显存、Cache 使用量、模型状态账本 |
| 单卡吞吐不足 | 计算、访存或请求容量受限 | kernel/阶段时间、吞吐、队列等待 |
| 增加 GPU 后收益很小 | 通信暴露、同步或负载不均 | 通信时间、暴露时间、空转和扩展效率 |
| MoE workload 扩展异常 | token dispatch、All-to-All 或专家不均衡 | expert load、通信字节数、热点专家 |

## 通信观测与热点缓解

先回答“通信发生在哪里”，再回答“能否把它移出关键路径”。通信时间、等待时间和计算重叠必须分开记录；只看端到端吞吐，无法判断收益来自并行计算还是 workload 变化。

| 学习阶段 | 需要回答的问题 | 机制入口 | 形成的判断 |
|:---|:---|:---|:---|
| 通信观测 | 通信时间、等待和计算重叠分别是多少？ | [46 NCCL 通信 Profiling](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.md) | 区分带宽受限、同步受限和负载不均 |
| 热点缓解 | 哪些通信暴露在关键路径，应该如何减少或隐藏？ | [48 通信热点与缓解](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md) | 形成 overlap、批次、拓扑或调度假设 |
| 并行选型 | 当前模型和互联条件适合哪种切分？ | [49 并行策略选型](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md) | 明确 DP、TP、PP、EP 的适用条件与代价 |
| MoE 分支 | token dispatch 和 All-to-All 是否成为主导成本？ | [47 MoE 专家并行](../../02_PyTorch_Algorithms/47_MoE_Expert_Parallel.md) | 只在 MoE workload 下进入专家并行比较 |

## 并行策略与通信代价

| 策略 | 切分对象 | 主要通信 | 适合先问的问题 |
|:---|:---|:---|:---|
| DP | 请求或数据副本 | 梯度或状态同步；推理侧主要是请求分发 | 单副本是否已经无法满足容量？ |
| TP | 层内权重或激活 | All-Reduce、All-Gather 或 Reduce-Scatter | 互联带宽能否承受高频同步？ |
| PP | 模型层级 | stage 间激活传递 | pipeline 气泡和负载是否可接受？ |
| EP | MoE 专家 | token dispatch / All-to-All | 专家负载和通信是否均衡？ |

选择顺序应是：先确认能否装下，再确认通信链路，再估计同步频率与 payload，最后比较扩展后的吞吐、P95/P99、峰值显存和成本。CPU 模拟可以验证切分和决策逻辑，但不能替代真实多卡通信结果。

## 项目验证与证据链

分布式项目至少保留单实例 baseline、GPU 数量、并行策略、通信字节数、通信时间、暴露通信时间、空转或负载不均、端到端吞吐、P95/P99、峰值显存和质量。79–81 的分工是递进的：先测并行扩展，再验证 MoE 专家并行，最后收束为分布式推理决策。

| 项目 | 主要问题 | 最小证据 |
|:---|:---|:---|
| [79 分布式并行基准](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md) | GPU 数量和并行策略是否带来可解释的扩展收益？ | 单实例对照、通信、扩展效率、P95/P99 |
| [80 MoE 专家并行](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md) | token dispatch 和专家负载是否成为瓶颈？ | expert load、All-to-All、空转、吞吐 |
| [81 分布式推理验证](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md) | 方案是否值得迁移到真实推理部署？ | workload、通信、质量、成本和决策 |

项目报告应保留配置与环境、请求级结果、服务级指标和 trace/profile。汇总表只能保存结论，不能替代原始结果；`communication_ms`、`exposed_comm_ms`、`idle_gap_ms` 和 `overlap_ratio` 应与总 elapsed 分开记录。

## 与 Task4 和 Task6 的边界

Task4 的 `handoff` 关注单次 Serving 路径上的局部状态交接、请求尾延迟和池利用率；本节 Task6 关注跨 GPU / 跨实例通信、集合通信、并行扩展和部署成本。两者都可以记录通信时间，但优化对象和决策尺度不同。

完成本节后，再回到 [06 端到端基准与决策](./06_benchmark_and_decision.md)，用统一 workload 判断方案应当 `accept / tune / reject`。

## 判断框架

| 现象 | 优先判断 | 下一步 |
|:---|:---|:---|
| 单卡装不下 | 容量或模型切分问题 | 先看显存账本，再评估 TP/PP |
| 多卡吞吐没有线性提升 | 通信暴露或负载不均 | 查看通信时间、空转和 overlap |
| MoE 扩展明显变慢 | All-to-All 或专家热点 | 进入 EP 与负载均衡分析 |
| 吞吐提升但 P99 变差 | 通信或排队进入关键路径 | 回到 Serving 和统一 benchmark |

相关阅读：论文 [Megatron-LM](https://arxiv.org/abs/2104.04473)、[DistServe](https://arxiv.org/abs/2401.09670)；开源实现 [NCCL](https://github.com/NVIDIA/nccl)、[vLLM distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)。
