# 性能优化（Performance Optimization）

> 专题类型：主学习路线　副标题：从性能分析、显存管理、通信代价到端到端决策

## 页面导语

性能问题很少只由一个算子或一个参数造成。一次训练或推理运行，可能同时受到计算、显存驻留、数据搬运、通信等待、请求排队和部署成本的影响。本专题把这些因素放到同一条证据链中：先定义可测量的问题，再定位瓶颈对象，最后用对照实验判断优化是否值得保留。

本专题的主线不是记忆工具名称，而是学习如何回答三个问题：哪里慢或放不下？代价从哪里转移到了哪里？实验结果是否足以支持工程决策？原有的[显存优化专题](../memory_performance_tuning/intro.md)和[性能分析专题](../profiling/intro.md)仍然保留，分别作为资源机制和 profiling 方法的深入入口。

## 如何开始

建议从 Task0 建立性能问题的共同语言，再按实际问题进入 Task1–Task6。没有 GPU 时可以先完成指标、账本、状态和决策逻辑；真实峰值显存、GPU 利用率、通信等待和 backend 收益需要在对应硬件上复测。

## 主学习路线与验证出口

| Task / 主题 | 本阶段要回答的问题 | 主要学习内容 | 主要验证出口 | 正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 性能工程全景与 Benchmark 方法论 | 怎样把“变慢、爆显存或吞吐不足”改写成可比较的问题？ | 性能对象、指标语义、workload、baseline、证据等级和结果记录 | 一份固定条件下可复查的 benchmark 计划 | [01 性能问题与 Benchmark 基础](./01_performance_problem_and_benchmark.md) |
| Task1 · 可观测性与 Profiling 工具链 | 怎样知道时间、显存和等待到底消耗在哪里？ | 轻量计时、框架 profiler、trace、显存时间线和证据归因 | 时间热点、驻留对象和待验证瓶颈假设 | [02 可观测性与 Profiling](./02_observability_and_profiling.md) |
| Task2 · 显存账本与训练侧优化 | 训练显存由哪些对象构成，哪种策略值得采用？ | 参数、梯度、优化器状态、Activation、Checkpoint、Offload 和预算 | 显存账本、策略对照、峰值和 step time | [03 显存账本与训练优化](./03_training_memory_ledger.md) |
| Task3 · 推理侧显存优化与 KV Cache 管理 | KV Cache 如何增长、复用和受到容量约束？ | KV Cache 生命周期、分页、前缀复用、量化、驱逐和并发容量；迁移入口：[显存优化 Task4](../memory_performance_tuning/intro.md) | cache 命中、峰值显存、TTFT / TPOT、并发和质量 | [04 推理显存与 KV Cache](./04_inference_memory_and_kv_cache.md) |
| Task4 · 计算、通信与 I/O 瓶颈分析 | 性能损失来自计算、通信、数据搬运还是 I/O？ | Roofline 直觉、kernel 时间、CPU-GPU 搬运、collective、数据管道和 overlap | 阶段时间分解、通信等待、带宽证据和瓶颈归因 | [05 计算、通信与 I/O 瓶颈](./05_compute_communication_and_io.md) |
| Task5 · 成本模型与生产级性能工程 | 一个优化方案的真实收益和工程成本如何估算？ | 成本模型、容量、吞吐、延迟、质量、稳定性、backend 和回归；迁移入口：[显存优化 Task5](../memory_performance_tuning/intro.md) | 候选方案比较、预算敏感性和 `accept / tune / reject` | [06 成本模型与生产决策](./06_cost_model_and_production_decision.md) |
| Task6 · 端到端性能工程实战 | 如何把多个局部优化组合成可交付结论？ | 问题定义、profiling、策略组合、真实 benchmark、回归和项目报告 | 带环境、workload、证据和限制条件的项目结论 | [07 端到端性能工程实战](./07_end_to_end_performance_engineering.md) |

## 学习入口与相关项目

主线项目按“机制 → 测量 → 对照 → 决策”组织：

- 通用推理性能基线：[Part 02 · 66 推理性能对比](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)
- 训练性能基线：[Part 02 · 73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- Profiling 驱动优化：[Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)
- 显存策略对照：[Part 02 · 75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md)、[Part 02 · 76 激活检查点与卸载](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)
- 显存路线迁移：显存优化 [Task4：KV Cache 与显存容量](../memory_performance_tuning/intro.md) → 性能优化 Task3；显存优化 [Task5：量化与显存容量扩展](../memory_performance_tuning/intro.md) → 性能优化 Task5
- 多卡与通信证据：[Part 02 · 79–81 分布式项目](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md)

项目节提供真实实验出口，专题正文负责解释方法和判断逻辑；二者不要求一一对应，一个 Task 可以连接多个正文小节和项目。

## Part 02 迁移关系

原有训练内存与调优页面按职责迁移到新的专题和相邻路线，不再继续作为 2.5 组的主线：

| 原页面 | 新承接位置 | 迁移后的职责 |
|:---|:---|:---|
| Part 02 · 43 统一训练内存 | Task2 · [显存账本与训练侧优化](./03_training_memory_ledger.md) | 统一记录参数、梯度、优化器状态、activation、临时工作集与迁移代价 |
| Part 02 · 44 自动调优框架 | [算子优化 Task5](../operator_optimization/05_cost_model_and_profiling.md) | 以 shape、dtype、资源约束和 profiling 证据筛选候选配置 |
| Part 02 · 45 显存裁剪规划 | Task2 + [75 显存预算压缩项目](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) | 将峰值来源、预算留边和裁剪顺序转化为可验证的策略决策 |

迁移完成前保留原 Notebook，避免历史页面和交叉引用失效；新的学习入口以本表中的专题正文和项目页为准。

## 跨专题入口

- 请求阶段、KV Cache、Serving 和 backend：进入[推理优化](../inference_optimization/intro.md)。
- 显存对象、Checkpoint、Offload 和资源策略：进入[显存优化专题](../memory_performance_tuning/intro.md)。
- 时间线、trace 和瓶颈归因：进入[性能分析专题](../profiling/intro.md)。
- Kernel、访存、融合和 CUDA/Triton：进入[算子优化](../operator_optimization/intro.md)。
- 量化表示与部署格式：进入[量化与压缩](../quantization/intro.md)。
- 多卡切分、通信和扩展效率：进入[通信与并行](../communication_parallel/intro.md)。

## 环境与证据

CPU 适合验证公式、账本、状态转移、指标聚合和决策逻辑；GPU 或真实 backend 才能确认峰值显存、吞吐、延迟、通信等待和部署收益。每个结论都应记录模型、硬件、dtype、batch、sequence length、并发、warmup、重复次数、baseline 和 candidate。

## 阅读入口

- 想按问题连续学习：查看[性能优化问题链](./walkthrough.md)。
- 想根据现象选择测量和策略：查看[性能优化判断手册](./casebook.md)。
- 想学习完整专题正文：按 Task0–6 阅读 01–07。
