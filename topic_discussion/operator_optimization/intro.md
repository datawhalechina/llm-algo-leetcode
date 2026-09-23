# 算子优化（Operator Optimization）

> 专题类型：主学习路线　副标题：从硬件直觉、图级优化到 Triton / CUDA Kernel

## 页面导语

算子优化不是把某个 kernel 写得更复杂，而是理解一个算子为什么慢，并在语义、数值、访存、执行和端到端收益之间建立证据链。本专题从 GPU 性能心智模型开始，经过图级优化、Triton、经典算子和 Attention，最后回到 profiling、实现路径和综合项目决策。

主线职责是：性能优化负责定位系统瓶颈，算子优化负责在 kernel 层面消除可解释的瓶颈；两者在 profiling 和端到端验证处交汇，但问题视角不同。

![算子优化：从数学语义到端到端收益](../../docs/public/topic_discussion/operator_optimization/operator_optimization_overview.svg)

## 如何开始

推荐按 Task0–6 顺序学习。Task0 建立硬件与性能直觉，Task1 把图级变换接到算子融合，Task2–4 逐步进入 Triton、经典算子和 Attention，Task5 训练 profiling 与调优闭环，Task6 完成 PyTorch → Triton → CUDA 的实现路径比较。

没有 GPU 时，可以先完成公式、数据流、CPU 正确性和成本模型；GPU 阶段再验证真实 kernel 时间、显存、autotune、occupancy 和端到端收益。CPU 结果不能直接写成 GPU 加速结论。

## Task0–6 主学习路线

| Task | 目标与核心问题 | 主要内容 | 交付物 | 正文支撑 |
|:---|:---|:---|:---|:---|
| Task0 | 硬件与性能心智模型：时间花在计算、访存还是调度？ | block / warp / thread、SM 与 occupancy；HBM、L2、shared memory、寄存器；Roofline；中间张量和 memory traffic | 为一个简单算子估算计算量与访存量，判断计算密集或访存密集 | [01 算子性能心智模型](./01_why_operator_optimization_matters.md) |
| Task1 | 图级优化与算子融合：哪些节点应该融合，融合何时反而变慢？ | 常量折叠、死代码消除、公共子表达式消除；中间张量落地；融合边界；`torch.compile` 与 Inductor Pass | 为小计算图设计融合方案，并用 `torch.compile` 验证 | [02 图级优化与融合](./graph_compiler/02_graph_structure_and_fusion_decisions.md)、[03 融合与 Kernel 组合](./03_fusion_and_kernel_composition.md) |
| Task2 | Triton 编程模型与基础 Kernel：如何写出第一个正确 kernel？ | block-level 编程、`tl.load / tl.store`、program id、mask、block size；Fused RMSNorm / SwiGLU；Safe Softmax；Triton 与 CUDA 的边界 | 实现 Fused RMSNorm，与 PyTorch 对比正确性和性能 | [02 Kernel 语义与内存](./02_kernel_semantics_and_memory.md) |
| Task3 | 经典算子实现：GEMM 与 Softmax 如何减少访存？ | Tiled GEMM、shared memory、寄存器复用；Online Safe Softmax；向量化加载和合并访存；Triton 与 cuBLAS / cuDNN 对照 | 实现 Tiled GEMM 和 Safe Softmax，并用 profiler 分析 occupancy 与 memory throughput | [03 融合与 Kernel 组合](./03_fusion_and_kernel_composition.md) |
| Task4 | Attention 算子攻坚：FlashAttention 与 PagedAttention 如何改变数据流？ | 分块、在线 Softmax、FlashAttention-2、KV Cache 分页、RoPE Triton 实现 | 实现或复现 FlashAttention-2 前向路径，对比显存与延迟 | [04 执行与硬件约束](./04_cuda_execution_and_hardware_constraints.md) |
| Task5 | 算子性能分析与调优：慢在哪里，应该如何改？ | Nsight Compute、occupancy、memory throughput、compute utilization；瓶颈分类；访存模式、bank conflict、autotune、迭代复测 | 对一个 kernel 完成 profile → 优化 → 复测，保留每次迭代证据 | [05 成本模型与 Profiling](./05_cost_model_and_profiling.md) |
| Task6 | 综合项目与底层下探：Triton 什么时候不够，需要 CUDA？ | Quantization Kernel、Multi-LoRA、Attention 变体；PyTorch / `torch.compile` / Triton / CUDA 对照；warp 原语、shared memory、PTX 路径 | 完成一个真实算子从 baseline 到优化的报告，给出实现路径决策 | [06 基准测试与项目验证](./06_benchmark_and_project_validation.md) |

## Part 01–04 的职责

| 来源 | 在本路线中的职责 | 主要回答的问题 |
|:---|:---|:---|
| Part 01 | 硬件、并行和编译机制基础 | 数据经过哪些存储层、线程层次和指令路径？ |
| Part 02 | Transformer 组件桥接 | 算子位于 Block 哪一段，哪些组件适合融合或分块？ |
| Part 03 | Triton 主实践线 | 如何把语义写成 kernel，进行调试、融合和 autotune？ |
| Part 04 | CUDA 与系统深化 | 如何控制更底层的执行、异步搬运、共享内存和 Tensor Core？ |

编译与图优化不再作为并列专题，而是作为 Task1 的支撑模块：[图级优化与编译支撑](./graph_compiler/intro.md)。它负责解释 graph rewrite、IR、lowering、legalization、schedule 和 backend 约束；本专题继续负责具体 kernel、访存和性能证据。

## 正文建设

专题正文不逐字复述 Task，而是把多个 Notebook 中的机制串成判断链：

| 正文方向 | 主要回答的问题 | 当前入口 |
|:---|:---|:---|
| 硬件与性能模型 | 为什么算子会慢，计算量和访存量如何估算？ | `01_why_operator_optimization_matters.md` |
| 图级优化与融合 | 图变换怎样减少中间张量和 kernel launch？ | `graph_compiler/02_graph_structure_and_fusion_decisions.md` |
| Kernel 语义与访存 | 输出契约、layout、mask 和数据路径如何保持正确？ | `02_kernel_semantics_and_memory.md` |
| 融合与经典算子 | Fusion、GEMM、Softmax 的收益和边界是什么？ | `03_fusion_and_kernel_composition.md` |
| 执行与 Attention | Tensor Core、warp、shared memory、FlashAttention 如何协同？ | `04_cuda_execution_and_hardware_constraints.md` |
| Profiling 与调优 | 如何把 profiler 输出转成下一次改动？ | `05_cost_model_and_profiling.md` |
| 项目验证 | 如何证明局部 kernel 收益传递到了端到端？ | `06_benchmark_and_project_validation.md` |

后续需要补充一页专门的“经典算子实现”正文，以及一页专门的“Attention / FlashAttention 机制”正文；当前先复用已有融合、执行和项目验证页承载入口。

## 验证闭环

```text
算子语义
  → CPU / reference correctness
  → Triton / CUDA kernel
  → microbenchmark
  → profiler 归因
  → 端到端 workload
  → accept / tune / reject
```

每次实验至少记录输入 shape、dtype、基线实现、候选实现、GPU、PyTorch / Triton / CUDA 版本、warmup、迭代次数、正确性阈值和证据等级。单个 kernel 变快、编译成功或节点数减少，都不能单独证明系统收益。

## 跨专题入口

- [性能优化](../performance_optimization/intro.md)：定位 Compute、Memory、Communication 和 I/O 瓶颈。
- [推理优化](../inference_optimization/intro.md)：提供 Prefill、Decode、KV Cache、Serving 和 backend 场景。
- [显存优化](../memory_performance_tuning/intro.md)：解释激活、权重、缓存和搬运的容量约束。
- [量化与压缩](../quantization/intro.md)：提供 INT4、INT8、FP8 和低比特部署场景。
- [通信与并行](../communication_parallel/intro.md)：提供多卡切分、通信原语和扩展效率证据。

## 环境与证据边界

CPU 可以验证数学语义、边界处理、输出对齐和部分成本模型；真实 Triton / CUDA kernel、autotune、Tensor Core、CUDA Graph 和性能结论需要匹配的 GPU 与 workload。更换 GPU、PyTorch、CUDA、Triton 或编译选项后，应重新完成 smoke test，并把环境记录写入实验报告。

图像资产与维护规则见[算子优化图册](./07_visual_assets.md)。
