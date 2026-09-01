# 通信与并行（Communication and Parallelism）

> 专题类型：横切支撑　主服务目标：多卡切分与通信取舍

## 专题定位与 Infra 层定位

本专题串起多卡并行主线：先看通信原语和拓扑，再看 DDP、FSDP、ZeRO、Pipeline、Tensor Parallel 和专家并行分别切了什么，最后把通信热点和 benchmark 收回并行选型结论。它横跨五层：Infra-L1 决定互联拓扑和带宽，Infra-L2 提供 NCCL 等通信原语，Infra-L3 决定切分与运行时，Infra-L4 延伸到分布式 Serving，Infra-L5 才负责集群资源调度、多机实验和服务治理。

这里重点关注切分层级与通信代价；并行结论必须同时解释计算、显存、通信和扩展效率的变化。若问题先表现为单机推理速度，应转到推理优化；若问题只剩单个通信 kernel 或算子实现，应转到算子优化；若涉及图变换、IR 或 backend 选择，再转到编译与图优化。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的分布式与并行路线进入，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 79、80、81 等项目节。专题正文可以作为并行策略的决策索引，不要求学习者一开始就拥有多卡机器。

## 前置阅读

建议先掌握 [Part 01 · 05 Communication Topologies](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb) 与 [Part 01 · 20 NCCL and AllReduce Basics](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb) 提供的 GPU、通信与系统基础，再补读并行相关内容。进入真实 benchmark 前，应理解 world size、rank、集体通信、数据/张量/流水线/专家并行，以及显存、通信和计算之间的基本权衡。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 通信拓扑与 AllReduce 前置 | [Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb) → [Part 01 · 20 NCCL 与 AllReduce 基础](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb) | [01 为什么需要并行与通信](./01_why_parallel_and_communication.md) |
| Task2 | DDP 到 FSDP / ZeRO 的状态分摊 | [Part 01 · 06 显存计算与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) → [Part 02 · 27 ZeRO 优化器模拟](../../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.ipynb) | [02 数据并行与同步](./02_data_parallel_and_synchronization.md) |
| Task3 | Pipeline Parallel 的时序与气泡 | [Part 02 · 28 流水线并行与微批次](../../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.ipynb) | [04 流水线并行与张量并行](./04_pipeline_and_tensor_parallel.md) |
| Task4 | Tensor Parallel 的切分与代价 | [Part 02 · 29 张量并行模拟](../../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.ipynb) | [04 流水线并行与张量并行](./04_pipeline_and_tensor_parallel.md) |
| Task5 | 通信 profiling、热点定位与策略选型 | [Part 02 · 46 通信性能分析与 NCCL](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb) → [Part 02 · 47 MoE 专家并行](../../02_PyTorch_Algorithms/47_MoE_Expert_Parallel.ipynb) → [Part 02 · 48 通信热点与缓解](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.ipynb) → [Part 02 · 49 并行策略选型](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.ipynb) → [Part 02 · 79 分布式并行基准测试](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb) | [05 专家并行与通信热点](./05_expert_parallel_and_communication_hotspots.md) |
| Task6 | 并行项目收口 | [Part 02 · 80 MoE 专家并行基准测试](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb) → [Part 02 · 81 分布式推理项目](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb) | [06 基准测试与并行决策](./06_benchmark_and_parallel_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“为什么多卡不一定更快”“不同切分到底换来了什么”时，再回来看对应的专题正文。想看汇总版就进 [通信与并行正文](./casebook.md)，想按连续故事线走一遍就进 [通信与并行深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[性能分析](../profiling/intro.md) 负责证据链与等待热点，[显存优化](../memory_performance_tuning/intro.md) 负责显存分摊 trade-off，[监督微调与训练工程](../fine_tuning_training/intro.md) 负责训练工程闭环，[算子优化](../operator_optimization/intro.md) 负责具体 kernel 与算子实现，[编译与图优化](../compiler_graph_optimization/intro.md) 负责执行模型与 backend 约束。

## 项目结论

推荐的实践闭环是 `79 分布式并行 benchmark -> 80 MoE 专家并行 benchmark -> 81 分布式推理项目`。最终结论应明确并行策略、GPU 数量、通信占比、吞吐/延迟、显存和扩展效率；单卡模拟只能帮助理解机制，不能替代真实多卡结论。

## 环境与验证

并行原理和通信模拟可先用 CPU 或单 GPU；真实 DDP、NCCL、多卡训练和分布式推理需要匹配的多 GPU、驱动、CUDA、通信库和网络拓扑。建议固定 world size、batch、输入长度和 warmup，并分别保存单卡基线、多卡结果、日志与环境信息。
