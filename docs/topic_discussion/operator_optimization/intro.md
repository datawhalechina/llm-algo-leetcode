# 算子优化（Operator Optimization）

> 专题类型：主学习路线　副标题：从计算图、Triton Kernel 到 CUDA 下探与端到端证据

## 页面导语

算子优化的目标不是孤立地让某个 kernel 更复杂，而是在输入输出语义不变的前提下，减少不必要的数据移动、kernel launch 或硬件空转，并证明收益能传递到真实 workload。它承接性能优化已经定位出的局部瓶颈：性能优化回答“慢在哪里”，本专题回答“图、kernel 或执行路径怎样改，为什么值得改”。

路线从硬件成本与图变换进入 Triton 实现，再经过经典算子与 Attention，最后按需要下探到 CUDA。Part01 提供硬件、执行与编译前置；Part03 是 Triton 主实践；Part04 提供 CUDA、Tensor Core、warp、shared memory 与 runtime 的深层实现。Part00 与 Part02 只承担张量/调试基础和模型组件场景，不构成主路线。

![算子优化学习路线](../../public/topic_discussion/operator_optimization/operator_optimization_task_route.svg)

## 如何开始

- **希望理解“为什么慢”：** Task0 → Task1，先用计算/访存和图结构判断候选。
- **希望动手写 Triton：** Task0 → Task2 → Task3，再进入 Task5 调优。
- **希望优化 Attention 或推理路径：** 回补 Task0/1 后进入 Task4；不要直接把服务级缓存或调度问题写成 kernel 问题。
- **希望下探 CUDA：** 完成 Task2/3/5 后进入 Task6，在有明确实现瓶颈时比较 PyTorch、`torch.compile`、Triton 与 CUDA。

CPU 可验证语义、边界、reference 与成本估算；真实 kernel 时间、autotune、Tensor Core、occupancy 和 backend 差异需要匹配 GPU。每次对照均固定 shape、layout、dtype、精度阈值、warmup、重复次数、软件版本与 baseline。

## Task0–6 主学习路线

| Task / 主题 | 要回答的问题 | 主实践与验证出口 | 专题正文入口 |
|:---|:---|:---|:---|
| Task0 · 算子语义与硬件成本 | 一个算子的时间主要花在计算、访存还是 launch/调度？ | [Part01 · 01 dtype](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md) → [02 参数与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md) → [03 GPU 架构](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)；输出 FLOPs、bytes 与 roofline 假设 | [01 算子性能心智模型](./01_why_operator_optimization_matters.md) |
| Task1 · 图级优化与融合判断 | 哪些图节点能消除、重排或融合，何时会因寄存器压力或动态形状失效？ | [Part01 · 09 编译与图优化](../../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.md) → [19 算子融合](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md) → [Part04 · 18 CUDA Graph / JIT](../../04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile.md)；练习：[计算图 IR 与图优化 Notebook](../ai_compiler/01_graph_ir_and_optimization.md)；输出 graph rewrite、lowering 假设与融合候选 | [03 图级优化、融合与 Kernel 组合](./03_fusion_and_kernel_composition.md) |
| Task2 · Triton 编程模型与正确性 | 如何将 reference 语义写成正确的 block-level kernel？ | [Part01 · 08 CUDA / Triton 模型](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md) → [18 Triton Block](../../01_Hardware_Math_and_Systems/18_Triton_Block_Model.md) → [Part03 · 01 Vector Add](../../03_Triton_Kernels/01_Triton_Vector_Addition.md) → [02 Fused SwiGLU](../../03_Triton_Kernels/02_Triton_Fused_SwiGLU.md) → [03 RMSNorm](../../03_Triton_Kernels/03_Triton_Fused_RMSNorm.md) | [02 Kernel 语义与内存](./02_kernel_semantics_and_memory.md) |
| Task3 · 经典 Kernel 与数据路径 | GEMM、Softmax、Norm 如何通过 tile、归约、片上复用与向量化减少访存？ | [Part03 · 04 GEMM](../../03_Triton_Kernels/04_Triton_GEMM_Tutorial.md) → [06 Softmax](../../03_Triton_Kernels/06_Triton_Fused_Softmax.md) → [06.5 设计模式](../../03_Triton_Kernels/06_5_Triton_Design_Patterns.md)；输出 reference / Triton 对照 | [03 融合与 Kernel 组合](./03_fusion_and_kernel_composition.md) |
| Task4 · Attention 与模型关键算子 | RoPE、Attention、分页状态和分块算法怎样改变读写路径？ | [Part01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md) → [Part02 · 20 FlashAttention 模拟](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md) → [Part03 · 07 RoPE](../../03_Triton_Kernels/07_Triton_Fused_RoPE.md) → [08 FlashAttention](../../03_Triton_Kernels/08_Triton_Flash_Attention.md) → [09 PagedAttention](../../03_Triton_Kernels/09_Triton_PagedAttention.md) | [04 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) |
| Task5 · Profiling、autotune 与 CUDA 下探 | 如何用指标区分计算、访存、occupancy 与 launch 瓶颈，并选择下一个改动？ | [Part00 · 17 Profiling](../../00_Prerequisites/17_PyTorch_Profiling_Basics.md) → [Part01 · 13 瓶颈分析](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) → [23 Tensor Core](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.md) → [24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md) → [Part03 · 05 Autotune](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md) → [12 内存模型与调试](../../03_Triton_Kernels/12_Triton_Memory_Model_and_Debug.md) | [05 成本模型与 Profiling](./05_cost_model_and_profiling.md) |
| Task6 · 综合项目与实现路径决策 | 何时选 PyTorch、`torch.compile`、Triton 或 CUDA，如何证明局部收益进入模型 workload？ | [Part03 · 10 Quantization](../../03_Triton_Kernels/10_Triton_Quantization.md) → [11 Multi-LoRA](../../03_Triton_Kernels/11_Triton_Multi_LoRA.md) → [13 LLaMA3 Block 项目](../../03_Triton_Kernels/13_Triton_Llama3_Block_Project.md) → [Part04 · 15 CUDA Kernel](../../04_CUDA_and_System_Optimization/15_CUDA_Custom_Kernel_Intro.md) → [16 Shared Memory](../../04_CUDA_and_System_Optimization/16_CUDA_Shared_Memory_Optimization.md) → [21 CUDA / Triton / PyTorch](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.md) | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) |

## Part00–04 对应关系

| 来源 Part | 在算子优化中的定位 | 主要小节 / 资产 | 对应 Task |
|:---|:---|:---|:---|
| Part00 · 前置知识 | 最小张量、reference、调试与 profiling 习惯 | 05 Tensor、17 Profiling、19 调试、20 性能账本 | 支撑 Task0 / 2 / 5，不单独形成 kernel 路线 |
| Part01 · 硬件、数学与系统 | 性能模型、存储层次、执行模型、编译与融合的机制前置 | 01–03、08–09、12–19、23–24、30–32 | Task0–5 的原理底座 |
| Part02 · PyTorch 算法实战 | 模型组件和真实 workload 场景 | 04 Attention、05 Block、20 FlashAttention 模拟、40–41 低比特、66–67/73–74 结果证据 | 主要支撑 Task4 / 6；不替代 kernel 实现 |
| Part03 · Triton 算子开发 | 主实践线：从正确 kernel 到项目 | 01–13、设计模式、内存调试、FAQ | Task2–6 的主要实现资产 |
| Part04 · CUDA 与系统优化 | CUDA 下探与实现路径比较 | 15–16、02.1 Bank Conflict、03 MMA、04 Warp、17–18、21 | Task1 / 5 / 6 的深层实现与选型 |

![算子优化知识地图](../../public/topic_discussion/operator_optimization/operator_optimization_knowledge_map.svg)

## 正文与项目证据

专题正文负责从语义、图变换、数据路径、硬件执行到证据的判断链；不逐页复述 Part03 的实现代码。Task1 在正文 01、03–06 中依次处理 graph rewrite、lowering、legalization、schedule 与 backend 约束，不再把它们维护成另一条并列路线。

| 证据层 | 必须回答的问题 | 典型记录 |
|:---|:---|:---|
| 正确性 | 输出是否与 reference 在阈值内一致？ | 输入分布、shape、dtype、误差与边界样本 |
| 微基准 | kernel 是否在相同输入下更快或更省？ | latency、bandwidth、FLOPs、peak memory、autotune 配置 |
| profiler | 收益或退化来自什么执行路径？ | kernel 调用、occupancy、访存、launch、trace |
| workload | 模型或服务是否真正受益？ | baseline/candidate、质量、端到端指标、环境与失败记录 |

## 按需回补与跨专题入口

| 当你遇到的问题 | 应进入的专题 | 算子优化在其中的角色 |
|:---|:---|:---|
| 不知道整体慢在哪里 | [性能优化](../performance_optimization/intro.md)、[性能分析](../performance_analysis/intro.md) | 接收已定位的局部 compute / memory 瓶颈 |
| 请求、KV 或服务调度导致变慢 | [推理优化](../inference_optimization/intro.md) | 优化明确暴露的 Attention、RoPE、量化或模型组件 kernel |
| 多卡通信、all-to-all 或扩展效率问题 | [通信与并行](../communication_parallelism/intro.md) | 仅处理确认位于关键路径的本地计算/通信 kernel |
| 低比特表示、artifact 与质量问题 | [量化与压缩](../quantization_compression/intro.md) | 实现或分析 quantization kernel，不替代量化策略与部署决策 |

## 阅读入口

- 想按问题连续推进：查看[算子优化问题链](./walkthrough.md)。
- 想根据现象查找实现动作：查看[算子优化判断手册](./casebook.md)。
- 需要图变换、IR 或 backend 细节：从[03 图级优化、融合与 Kernel 组合](./03_fusion_and_kernel_composition.md)进入，并按链接回查 Part01 · 09 与 Part04。
