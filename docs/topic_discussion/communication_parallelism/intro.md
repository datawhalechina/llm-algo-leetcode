# 通信与并行（Communication and Parallelism）

> 专题类型：横向支撑　学习目标：理解多卡如何分工、如何通信，以及如何用证据选择并行方案。

单卡无法同时满足模型容量、训练吞吐或服务规模时，系统需要把数据、训练状态或模型计算分给多张 GPU。**分布式**描述这种分工方式；**通信**描述分工后各张 GPU 如何交换梯度、参数、激活或 token 路由结果，重新组成一次正确的训练或推理。

两者不是并列的两套技术。分布式切分先决定谁保存、谁计算、谁负责更新；多卡通信再决定交换什么、何时交换以及经过什么链路；最终共同影响容量、吞吐、等待、长尾和扩展效率。

切分能摊开单卡压力，也会引入集合通信、同步点、链路等待和调度成本。本专题帮助学习者判断“能否放下、能否更快、代价是否可接受”。完整问题链见[深入阅读](./walkthrough.md)，故障定位见[案例集](./casebook.md)，术语差异见[专题术语表](./glossary.md)。

## 学习路线

Task0–1 建立多卡执行、拓扑与集合通信的共同语言；Task2 处理训练状态分片，Task3 处理模型内部的层、张量和序列切分，Task4 比较 token 到专家与请求到副本两类动态路由。Task5 用时间线解释“为什么多卡没有更快”，Task6 再把候选策略和真实多卡项目收口为决策。

路线图把多卡问题、Task0–6 的推进关系和每阶段的验证出口放在一起。

![通信与并行学习路线：从多卡分工到系统决策](../../public/topic_discussion/communication_parallelism/communication_parallel_roadmap.svg)

| Task | 要回答的问题 | 主学习线 | 专题正文 | 交付物 |
|:---|:---|:---|:---|:---|
| Task0 · 多卡约束与执行模型 | 单卡为什么放不下或跑不快？多卡分工会新增哪些同步与等待？ | [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) → [05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md) | [01 为什么需要并行与通信](./01_why_parallel_and_communication.md) | 单卡约束、切分候选与多卡代价清单 |
| Task1 · 拓扑、Collective 与同步 | All-Reduce、All-Gather、Reduce-Scatter、All-to-All 分别交换什么？ | [Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md) → [20 NCCL 与 AllReduce](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md) | [02 数据并行与同步](./02_data_parallel_and_synchronization.md) | collective、payload、同步频率与拓扑代价对照 |
| Task2 · 训练状态分片与 ZeRO | 参数、梯度、优化器状态如何从复制变为分片？ | 共享前置：[Part 00 · 11 优化器与损失](../../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md) → [12 训练生命周期](../../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)；核心：[Part 01 · 06 显存计算与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md) → [Part 02 · 27 ZeRO 优化器模拟](../../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md) | [03 状态分片与 ZeRO](./03_state_sharding_and_zero.md) | 状态账本与通信交换说明 |
| Task3 · 模型内部与上下文切分 | PP、TP、CP 分别沿层、张量和序列切开什么？如何组合进 Device Mesh？ | [Part 01 · 26 并行策略决策](../../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md) → [Part 02 · 28 Pipeline 并行](../../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md) / [29 Tensor 并行](../../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md) / [PAR-CONTEXT 上下文并行](../../02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.md) | [04 模型与上下文切分](./04_pipeline_and_tensor_parallel.md) | stage、Head / feature、sequence 布局与组合约束表 |
| Task4 · 动态路由与负载 | token 到专家和请求到副本的路由，分别如何形成容量、局部性与长尾问题？ | 专家分支：[Part 01 · 22 MoE 参数与计算](../../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.md) → [Part 02 · 46 MoE 专家并行](../../02_PyTorch_Algorithms/46_MoE_Expert_Parallel.md)；Serving 分支：[PAR-REPLICA 副本路由](../../02_PyTorch_Algorithms/PAR-REPLICA_Serving_Replica_Routing.md) | [05 动态路由与负载](./05_expert_parallel_and_communication_hotspots.md) | expert / replica 路由、容量、局部性和最慢执行单元证据 |
| Task5 · 通信调度、Profiling 与 Overlap | payload 与同步频率如何形成通信时间？怎样减少真正暴露在关键路径上的部分？ | [Part 01 · 17 CUDA Stream 与异步](../../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md) → [27 通信调度优化](../../01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.md) → [29 CUDA Stream 高级调度](../../01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.md) → [Part 02 · 47 NCCL 通信剖析](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.md) → [48 通信热点](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md) | [06 通信 Profiling 与 Overlap](./06_communication_profiling_and_overlap.md) | collective、payload、同步频率、overlap 与热点归因 |
| Task6 · 候选准入与多卡决策 | 怎样把容量、通信、吞吐、长尾和成本收口为可交付结论？ | 证据前置：[Part 00 · 20 性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.md) → [Part 02 · 49 并行策略选择](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md)；项目：训练状态并行 [79](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md)、[模型内部并行](../../02_PyTorch_Algorithms/87_Model_Internal_Parallel_Benchmark.md)、MoE 并行 [80](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md)、分布式推理 [81](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md) | [07 多卡 Benchmark 与并行决策](./07_benchmark_and_parallel_decision.md) | 四类项目分别形成 baseline / candidate / failure / decision 报告 |

知识地图把同一个多卡问题拆成“先有什么约束、如何切分、需要交换什么、最终观察什么”。它不是额外的阅读顺序；遇到显存、同步或扩展效率问题时，可用它回查对应机制。

![通信与并行知识地图：从分工对象到多卡证据](../../public/topic_discussion/communication_parallelism/communication_parallel_knowledge_map.svg)

## 扩展与跨专题入口

无法解释参数、梯度和优化器状态时，先补 Task2 标出的训练生命周期；无法解释带宽、延迟和拓扑时，回到 Task0–1；准备形成项目结论时，再补 Task6 的实验设计与证据口径。

Task5 关注梯度、参数、激活、序列状态或 token 路由的通信—计算重叠；请求排队、Continuous Batching 与 Prefill / Decode 路由进入[推理优化 · Serving 调度](../inference_optimization/07_serving_scheduling_and_pd.md)。AFD 的主归属同样是推理优化，通信专题只提供 activation、routing metadata、connector 与 overlap 的分析语言。副本路由同时依赖队列、缓存局部性和通信证据；瓶颈归因可继续进入[性能分析](../performance_optimization/methods/intro.md)，设备与 backend 选择可进入[部署与异构系统](../deployment_heterogeneous/intro.md)。

## 实验与项目出口

四类项目按训练状态、模型内部、MoE 和分布式推理分流，共享 `distributed-benchmark/v1` 外层契约，但不要求按编号连续执行。80 的路由和容量证据可被架构专题引用，81 的 Serving 结果可被推理专题引用；PP、TP 与 CP 的综合证据进入 `PAR-MODEL-INTERNAL`，不再写入 79。

机制页使用 `torchrun + torch.distributed + NCCL` 验证真实 collective、rank 布局和 payload；训练项目再使用 Accelerate、FSDP 或 DeepSpeed，分布式推理项目使用独立的 vLLM / SGLang 环境。首次创建双卡环境见[实验 Quickstart](./quickstart.md)，页面、workload、结果路径和证据状态见[实验资产表](./experiment_assets.md)，精确依赖与硬件快照见[GPU 环境资产表](../../gpu_environment_assets.md)。CPU 机制、环境预检、真实 NCCL、真实模型训练和 Serving benchmark 分属不同证据等级。

| 稳定项目 ID | 项目职责 | 当前承载 | 项目必须补充的证据 |
|:---|:---|:---|:---|
| `PAR-TRAINING-STATE` | DDP、FSDP、ZeRO 的训练状态与显存扩展 | 79 | 每 rank 状态、峰值显存、吞吐和通信时间 |
| `PAR-MODEL-INTERNAL` | TP、PP、CP、Device Mesh 与混合并行 | [87 模型内部并行基准](../../02_PyTorch_Algorithms/87_Model_Internal_Parallel_Benchmark.md) | 布局正确性、局部容量、collective 和扩展效率 |
| `PAR-MOE-EXPERT` | EP、dispatch/combine、capacity、overflow 与负载偏斜 | 80 | 有效 token、All-to-All、最慢 rank 和溢出率 |
| `PAR-DISTRIBUTED-INFERENCE` | Serving 副本、模型并行与 PD 分离 | 81 | TTFT、TPOT、吞吐、尾延迟、缓存与通信代价 |
