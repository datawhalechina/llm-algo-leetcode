# 性能分析（Performance Analysis）

> 专题类型：横切支撑　主服务目标：瓶颈定位与证据归因

> 导读：这个专题不从工具按钮出发，而是训练你把“哪里慢、为什么慢、改完是否真的更好”串成一条可复现的证据链，再决定回到推理、显存、通信或算子层继续优化。

## 专题定位与 Infra 层定位

本专题串起 profiling 主线：先定义问题，再用时间热点、memory timeline、通信等待和 benchmark 验证形成证据链，最后收成 `inspect / optimize / validate / revert` 的行动建议。Profiling 贯穿 Infra-L1–Infra-L5，是证据方法而不是独立软件层：Infra-L1 看硬件利用率与带宽，Infra-L2 看 kernel、算子库、通信库和编译结果，Infra-L3 看框架与运行时，Infra-L4 看服务请求、KV Cache 和吞吐，Infra-L5 看资源调度与回归治理。

它的作用是把计算、内存、通信三条能力轴放到同一条时间线上，再决定应该回到哪一层优化。若问题已经明确变成显存预算或推理选型，应转到对应专题；若已定位到单个算子、访存或 kernel 融合，则转到算子优化；若问题是图变换、IR、lowering 或 backend 选择，则转到编译与图优化。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的性能与项目路线进入，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 74、79 等项目节。73、76、75 属于显存优化路线的训练侧项目，Profiling 在其中提供证据方法；74 是显存路线的最终收口项目，同时复用本专题的方法。

## 前置阅读

建议先掌握 [Part 00 · 0E 调试基础](../../00_Prerequisites/0E.md) 与 [Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)，再进入 Part 02 的训练、推理或并行项目。若只想定位单个性能问题，可直接从下面的 Task 对应正文开始。

## 工具分层与环境边界

本专题不要求一开始安装完整的 GPU 工具链。工具应随着问题粒度逐级增加：先用轻量测量确认现象，再用框架级 profiler 找方向，只有在需要解释系统重叠或 kernel 细节时，才进入 Nsight 和分布式工具。

| 层级 | 工具 | 主要回答的问题 | 环境要求 |
|:---|:---|:---|:---|
| Level 0：轻量测量 | `time.perf_counter()`、`torch.cuda.Event`、`torch.cuda.synchronize()`、`nvidia-smi` | 总耗时、GPU 计时、峰值显存和进程状态 | CPU 可做部分验证；GPU 计时需要 CUDA |
| Level 1：框架级 | `torch.profiler`、Chrome Trace、TensorBoard | 时间热点、CPU/GPU 时间线、算子排序和训练阶段 | CPU 可运行基础示例；CUDA trace 需要 GPU |
| Level 2：系统级 | Nsight Systems | CPU-GPU overlap、stream、同步点、数据搬运和服务阶段 | NVIDIA GPU、Nsight Systems |
| Level 3：kernel / 分布式级 | Nsight Compute、NCCL trace / debug log | occupancy、访存吞吐、Tensor Core、通信等待和 overlap | GPU；多卡分析还需要分布式环境 |

`73`、`76`、`75` 是显存优化路线的训练侧项目，分别负责基线、策略比较和预算决策；`74` 使用 Level 0-2 的证据对显存优化方案做最终端到端收口。`79` 和 `46` 延伸到 Level 3 的通信与并行问题。Colab / ModelScope 学习者完成 Level 0-1 即可，Level 2-3 标记为 GPU 服务器扩展，不作为主线前置。

## 主学习线

本专题不单列 Task0：Task0 在各主路线中负责建立训练、推理或显存对象的共同语言，Profiling 从 Task1 的测量与调试桥接开始。`Task1-6` 指向 `Part 00 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | profiling 与调试前置桥 | [Part 00 · 0E 调试基础](../../00_Prerequisites/0E.md) → [Part 02 · 17 自动求导基础](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb) | [01 为什么需要性能分析](./01_why_profiling_matters.md) |
| Task2 | 时间热点与 trace 阅读 | [Part 00 · 20 Profiling 与显存账本](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb) | [02 时间分解与 Trace 阅读](./02_time_breakdown_and_trace_reading.md) |
| Task3 | memory profiling 与异常定位 | [Part 00 · 18 显存分析与优化](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb) → [Part 00 · 19 调试与异常定位](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb) → [Part 00 · 20 Profiling 与显存账本](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb) | [03 显存时间线与驻留状态](./03_memory_timeline_and_residency.md) |
| Task4 | 训练 / 推理证据采集与通信等待 | [Part 00 · 19 调试与异常定位](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb) → [Part 00 · 20 Profiling 与显存账本](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb) → [Part 02 · 22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [Part 02 · 46 NCCL 通信性能分析](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb) | [04 通信等待与重叠](./04_communication_wait_and_overlap.md) |
| Task5 | benchmark 设计与回归验证 | [Part 02 · 46 通信性能分析与 NCCL](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb) → [Part 02 · 79 分布式并行基准测试](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb) → [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) | [05 基准测试设计与回归验证](./05_benchmark_design_and_regression_validation.md) |
| Task6 | 回归验证与行动建议 | [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) | [06 诊断与行动决策](./06_diagnosis_and_action_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“现在到底慢在哪里”“看到一个热点后下一步该做什么”时，再回来看对应的专题正文。想看汇总版就进 [性能分析正文](./casebook.md)，想按连续故事线走一遍就进 [性能分析深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[显存优化](../memory_performance_tuning/intro.md) 负责预算与代价取舍，[推理优化](../inference_optimization/intro.md) 负责请求链路判断，[通信与并行](../communication_parallel/intro.md) 负责多卡等待和切分代价。

## 专题结论

Profiling 不能直接产生“优化成功”的结论，它提供的是从现象到决策所需的证据。完整结论至少要经过：

`固定 workload → 建立 baseline → 提出可证伪假设 → 采集匹配证据 → 对照 candidate → 检查质量与回归 → 形成行动`

在显存优化项目中，`73` 建立训练 baseline，`76` 比较显存策略，`75` 做预算敏感性，`74` 用真实 trace 解释端到端代价；Profiling 不替代这四个项目的策略选择。通信与并行问题再进入 `79–81`，推理请求问题进入 `66–70`。

因此，单个热点、单次 trace 或单项显存下降只能支持“需要继续检查”的判断。只有在模型、硬件、软件版本、workload、重复次数和质量门槛都明确时，才可以把结果升级为可复现的 `validate` 或 `revert` 决策。

## 环境与验证

基础 trace 阅读和部分模拟实验可先用 CPU；真实 GPU profiling、显存时间线和多卡通信需要对应 GPU 或分布式环境。建议固定 workload、warmup、迭代次数和随机种子，并将结果保存为 JSON；跨机器比较时同时记录 PyTorch、CUDA、驱动、GPU 型号和并行配置。
