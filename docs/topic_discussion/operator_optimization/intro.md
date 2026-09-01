# 算子优化（Operator Optimization）

> 专题类型：主学习路线　主服务目标：Kernel 性能与端到端收益

## 页面导语

本专题关注一个具体问题：同一个数学算子，为什么换一种 tile、数据布局、访存路径或 kernel 实现，性能和显存表现会不同？路线以 Part03 的 Triton 为主线，使用 Part01 建立硬件与执行模型，再用 Part04 的 CUDA 和系统机制理解更底层的优化边界。

## 如何开始

推荐先具备 Part02 的 Tensor、Attention 和 Block 基础，再按 Task0–6 推进。没有 GPU 时可以完成算子语义、CPU 正确性和成本模型；GPU 只用于验证 kernel 时间、编译成本、autotune、显存和端到端收益。

- Part01：解释硬件、并行层次、访存和性能约束；
- Part03：实现和调试 Triton kernel，是本路线的主要实践入口；
- Part04：补充 CUDA、异步执行、共享内存、Tensor Core 和系统级优化；
- 原有[编译与图优化](../compiler_graph_optimization/intro.md)：负责 graph rewrite、IR、lowering 和编译器决策，不与本专题合并。

如果只是想定位瓶颈，先看[性能分析](../profiling/intro.md)；如果问题首先表现为请求延迟或 KV Cache，再转到[推理优化](../inference_optimization/intro.md)。

## 主学习线与分级

`Task0-6` 是路线节点；表中的 `Part 01 · xx` 表示共享前置，未标注的入口主要来自 Part03 和 Part04。专题正文只提供判断框架，不替代 Notebook 中的实现和 benchmark。

| Task | 学习内容 | 主学习线 / 项目入口 | 学习顺序 | 专题正文 |
|:---|:---|:---|:---|:---|
| Task0 | 算子语义、GPU 执行与性能边界 | [Part 01 · 08 编程模型](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md) → [Part 01 · 15 CUDA 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.md) | 数学语义 → 并行执行 → 性能指标 | [01 为什么需要算子优化](./01_why_operator_optimization_matters.md) |
| Task1 | Triton Kernel 基础 | [Part 01 · 18 Triton Block 模型](../../01_Hardware_Math_and_Systems/18_Triton_Block_Model.md) → [Part 03 · 01 Triton 向量加法](../../03_Triton_Kernels/01_Triton_Vector_Addition.md) → [Part 03 · 04 Triton GEMM](../../03_Triton_Kernels/04_Triton_GEMM_Tutorial.md) | tile → load/store → kernel → 正确性 | [02 Kernel 语义与内存访问](./02_kernel_semantics_and_memory.md) |
| Task2 | 访存、布局与片上存储 | [Part 01 · 16 Warp / Block / Shared Memory](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.md) → [Part 01 · 24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md) → [Part 03 · 12 Triton 内存模型](../../03_Triton_Kernels/12_Triton_Memory_Model_and_Debug.md) | global memory → SRAM → layout → occupancy | [02 Kernel 语义与内存访问](./02_kernel_semantics_and_memory.md) |
| Task3 | 算子融合与模型组件 Kernel | [Part 01 · 19 算子融合基础](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md) → [Part 03 · 03 融合 RMSNorm](../../03_Triton_Kernels/03_Triton_Fused_RMSNorm.md) → [Part 03 · 06 融合 Softmax](../../03_Triton_Kernels/06_Triton_Fused_Softmax.md)；扩展 [Part 03 · 08 FlashAttention](../../03_Triton_Kernels/08_Triton_Flash_Attention.md) | 依赖 → 中间张量 → fusion → 端到端 | [03 融合与 Kernel 组合](./03_fusion_and_kernel_composition.md) |
| Task4 | Tensor Core、CUDA 与异步执行 | [Part 01 · 23 Tensor Core 深入](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.md) → [Part 04 · 21 CUDA / Triton / PyTorch 对照](../../04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch.md) → [Part 04 · 17 CUDA Stream 与数据传输](../../04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer.md) | instruction → kernel → stream → overlap | [04 CUDA 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) |
| Task5 | Autotune、Profiling 与 Kernel 选择 | [Part 03 · 05 Triton 自动调优](../../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md) → [Part 03 · 14 Triton 最佳实践](../../03_Triton_Kernels/14_Triton_Best_Practices_and_FAQ.md)；扩展 [Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) | 候选参数 → 搜索 → trace → 归因 | [05 成本模型与性能分析](./05_cost_model_and_profiling.md) |
| Task6 | Block 与端到端项目验证 | [Part 03 · 13 Triton Llama3 Block 项目](../../03_Triton_Kernels/13_Triton_Llama3_Block_Project.md) → [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) | 算子 → Block → workload → 决策 | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) |

### 核心与扩展分级

核心路径先验证算子语义和数值正确性，再比较固定 shape 下的 kernel；扩展路径进入真实 GPU、动态 shape、Tensor Core、CUDA、autotune 和端到端 workload。节点变少、单个 kernel 变快或编译成功，都不能单独证明系统收益。

| Task | 核心路径 | 扩展路径 | 学习顺序 | 环境级别 |
|:---|:---|:---|:---|:---|
| Task0 | Part 01 · 08、15 与指标定义 | 不同 GPU 执行模型对照 | 语义 → 执行 | Practice-P0 |
| Task1 | Triton 01、04 与 Part 01 · 18 | CUDA kernel、Tensor Core | tile → kernel | Practice-P0/P1 |
| Task2 | Part 01 · 16、24 与 Triton 12 | layout、occupancy、异步搬运 | memory → layout | Practice-P0/P1 |
| Task3 | Part 01 · 19、Triton 03/06 | FlashAttention、复杂 fusion | 依赖 → fusion | Practice-P1/P2 |
| Task4 | Part 01 · 23 与 CUDA 对照 | shared memory、stream、CUDA Graph | kernel → execution | Practice-P1/P2 |
| Task5 | Triton autotune 与 profiling | 多 shape、多 GPU、多 backend 搜索 | 搜索 → 归因 | Practice-P1/P2 |
| Task6 | Triton Block 项目 | 74 trace 与端到端比较 | 局部 → 系统 | Practice-P1/P2 |

### Part 01 共享前置

这些小节在本专题中只承担共享基础角色；它们可以服务于推理、显存和编译路线，不能把其中的理论或模拟直接写成某个 GPU 的实测结论。

| P1 前置 | 本路线使用它回答什么 | 不能直接推出什么 |
|:---|:---|:---|
| [Part 01 · 08 编程模型](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md)、[Part 01 · 15 CUDA 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.md) | Triton、CUDA、PyTorch 和 GPU 执行边界 | 某个实现一定更快 |
| [Part 01 · 16 Warp / Shared Memory](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.md)、[Part 01 · 23 Tensor Core](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.md) | 并行层次、同步和矩阵指令 | 一个 tile 配置适合所有 GPU |
| [Part 01 · 19 算子融合](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)、[Part 01 · 24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md) | 中间张量、数据驻留和读写代价 | fusion 一定降低端到端延迟 |
| [Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) | 建立 kernel 和系统热点证据 | 单次 trace 能代表所有 workload |

## 路线交叉边界

大模型架构负责 Attention、MoE、MLA 等结构本身；本专题只在结构落到 kernel、fusion、layout 或执行计划时介入。编译与图优化负责图变换、IR、lowering 和 backend 选择；本专题负责生成或优化具体 kernel。推理和显存优化负责服务指标与资源预算，Task6 再把局部 kernel 结果接回端到端验证。

## 学习方式与项目产出

先完成 CPU-first 的参考实现、数值对齐和边界测试，再进行 GPU microbenchmark；确认 kernel 有收益后，再用固定 workload 检查端到端延迟、吞吐、显存和编译成本。每次实验至少记录输入 shape、dtype、基线实现、候选实现、GPU、后端版本、warmup 和迭代次数。

## 环境与验证

CPU 可以验证数学语义、边界处理、输出对齐和部分成本模型；真实 Triton/CUDA kernel、autotune、Tensor Core、CUDA Graph 和性能结论需要 GPU。没有匹配 workload 或真实 trace 时，只能记录为优化假设，不能写成通用结论。
