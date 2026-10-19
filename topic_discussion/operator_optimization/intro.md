# 算子优化（Operator Optimization）

> 专题类型：主学习路线　学习目标：在语义不变的前提下改变计算图、数据路径或计算内核，并证明局部收益能够传递到真实工作负载。

性能分析回答“时间和资源去了哪里”，算子优化回答“已经定位的局部瓶颈怎样改变”。路线从算子语义和硬件成本出发，先建立分块、边界和访存契约，再进入经典计算内核、图级改写与融合、Attention 关键路径、性能剖析与 CUDA 下探；没有明确热点时，不应直接以重写计算内核作为起点。

## 学习路线

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 算子语义与硬件成本 | 一个算子的时间主要消耗在计算、访存还是启动与调度？ | 核心：[Part 01 · 01 数据类型](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb) → [Part 01 · 02 参数与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb) → [Part 01 · 03 GPU 架构](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) | 语义契约 → FLOPs/字节 → 资源上限 → 瓶颈假设 | [01 算子性能心智模型](./01_why_operator_optimization_matters.md) |
| Task1 · 计算内核语义与正确性 | 参考语义怎样映射为正确的分块计算内核？ | 核心：[Part 01 · 08 CUDA / Triton 模型](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb) → [Part 01 · 18 Triton Block](../../01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb) → [Part 03 · 01 向量加法](../../03_Triton_Kernels/01_Triton_Vector_Addition.ipynb) | 参考实现 → 程序 ID → 掩码与边界 → 数值对照 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) |
| Task2 · 经典计算内核与数据路径 | GEMM、Softmax 和归一化怎样通过分块、归约与片上复用减少数据移动？ | 核心：[Part 03 · 04 GEMM](../../03_Triton_Kernels/04_Triton_GEMM_Tutorial.ipynb) → [Part 03 · 06 Softmax](../../03_Triton_Kernels/06_Triton_Fused_Softmax.ipynb) → [Part 03 · 06.5 设计模式](../../03_Triton_Kernels/06_5_Triton_Design_Patterns.md) | 全局读写 → 分块 → 片上复用 → 归约 → 写回 | [02 计算内核语义与内存](./02_kernel_semantics_and_memory.md) |
| Task3 · 图级改写与融合判断 | 哪些图节点可以消除、重排或融合，何时会因布局、动态形状或资源压力失效？ | 核心：[Part 01 · 09 编译与图优化](../../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb) → [Part 01 · 19 算子融合](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb) → [Part 03 · 03 融合 RMSNorm](../../03_Triton_Kernels/03_Triton_Fused_RMSNorm.ipynb)<br>动手练习：[计算图 IR 与图优化 Notebook](../ai_compiler/01_graph_ir_and_optimization.ipynb) | 依赖图 → 语义等价 → 中间读写 → 后端约束 → 融合候选 | [03 图级优化、融合与计算内核组合](./03_fusion_and_kernel_composition.md) |
| Task4 · Attention 与模型关键算子 | RoPE、Attention、分页状态和分块算法怎样改变读写路径？ | 核心：[Part 01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb) → [Part 02 · 20 FlashAttention 模拟](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) → [Part 03 · 07 RoPE](../../03_Triton_Kernels/07_Triton_Fused_RoPE.ipynb) → [Part 03 · 08 FlashAttention](../../03_Triton_Kernels/08_Triton_Flash_Attention.ipynb) | 模型语义 → 分块状态 → 数据路径 → 计算内核 → 数值与容量验证 | [04 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) |
| Task5 · 性能剖析、自动调优与 CUDA 下探 | 指标怎样区分计算、访存、占用率和启动瓶颈，并决定下一个改动？ | 核心：[Part 00 · 17 性能分析](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb) → [Part 01 · 13 瓶颈分析](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb) → [Part 03 · 05 自动调优](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb) | 固定输入 → 微基准 → 执行轨迹 → 配置搜索 → 原因归因 | [05 成本模型与性能剖析](./05_cost_model_and_profiling.md) |
| Task6 · 实现路径与端到端决策 | 何时选择 PyTorch、`torch.compile`、Triton 或 CUDA，局部收益怎样进入模型工作负载？ | 项目：[Part 03 · 13 LLaMA3 Block](../../03_Triton_Kernels/13_Triton_Llama3_Block_Project.ipynb)；扩展：[Part 04 · 21 CUDA / Triton / PyTorch](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.ipynb) | 语义基线 → 实现候选 → 微基准 → 模型复测 → 维护成本 → 采用决策 | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) |

![算子优化学习路线](../../docs/public/topic_discussion/operator_optimization/operator_optimization_task_route.svg)

![算子优化知识地图](../../docs/public/topic_discussion/operator_optimization/operator_optimization_knowledge_map.svg)

## 扩展与跨专题入口

| 遇到的问题 | 优先入口 | 算子优化的职责 |
|:---|:---|:---|
| 不知道整体慢在哪里 | [性能分析](../performance_analysis/intro.md)、[性能优化](../performance_optimization/intro.md) | 接收已经定位到局部计算或访存路径的瓶颈 |
| 不确定 Attention、MLP、MoE 或记忆结构的语义 | [模型架构演进](../llm_architecture_evolution/intro.md) | 在结构语义稳定后再改变图、访存或计算内核 |
| 请求、KV 或调度导致变慢 | [推理优化](../inference_optimization/intro.md) | 只优化确认位于关键路径的 Attention、RoPE、量化或模型组件 |
| 多卡通信与扩展效率异常 | [通信与并行](../communication_parallelism/intro.md) | 处理已经确认暴露的本地计算或通信计算内核 |
| 低比特表示或质量异常 | [量化与压缩](../quantization_compression/intro.md) | 实现低比特计算路径，不替代量化策略与质量决策 |

按需回补硬件存储层次、Tensor Core、Warp、共享内存和编译流程时进入 Part 01；需要比较 CUDA 实现与系统运行时时进入 Part 04。连续学习使用[算子优化问题链](./walkthrough.md)，按现象定位使用[算子优化判断手册](./casebook.md)，概念辨析见[专题术语表](./glossary.md)，运行 Part 03 前先查看[算子优化 Quickstart](./quickstart.md)。

## 实验与项目出口

CPU 参考实现只验证语义、边界与误差；真实计算内核时间、自动调优、Tensor Core、占用率和后端差异必须在匹配 GPU 上验证。每轮对照固定形状、布局、数据类型、误差阈值、预热、重复次数、软件版本和基线。

环境复用、验证顺序、现有结果与待补证据统一登记在[实验资产表](./experiment_assets.md)。

| 证据阶段 | 要回答的问题 | 输出 |
|:---|:---|:---|
| 正确性 | 候选与参考实现在边界输入上是否语义一致？ | 输入分布、形状、数据类型、误差和失败样本 |
| 微基准 | 相同输入下是否减少时间、字节或峰值资源？ | 延迟、带宽、FLOPs、显存和调优配置 |
| 路径归因 | 收益或退化来自哪个执行路径？ | 计算内核调用、占用率、访存、启动与执行轨迹 |
| 工作负载复测 | 局部收益能否传递到模型或服务？ | 配对基线、质量、端到端指标、环境和失败记录 |
