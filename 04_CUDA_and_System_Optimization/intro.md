# 第四部分：CUDA C++ 与系统优化

## 🎯 本部分概览

本部分聚焦 CUDA C++、系统性能优化、分布式训练工程和架构选型。

Part 3 教的是“如何用 Triton 快速写出高性能算子”，Part 4 教的是“Triton 底层是怎么工作的，以及什么时候需要手写 CUDA”。

### 章节结构

- 模块一（01-04）：CUDA 编程基础，手写 Kernel，理解硬件
- 模块二（05-08）：系统级性能优化，从单卡到多卡的调度
- 模块三（09-12）：分布式训练工程，多机多卡训练
- 模块四（13-16）：架构视野与总结，技术选型与成本优化

### 未来扩展

- `02.1`：Bank Conflict Deep Dive
- `07.1`：Double Buffering Deep Dive
- `09.1`：Ring-AllReduce Deep Dive

### 环境边界

- **整体定位：GPU-required**
- **完整体验**：需要 NVIDIA GPU，推荐 Linux + CUDA
- **代码审计结果**：本章直接面向 CUDA kernel、通信、系统优化和架构选型
- **阅读说明**：可以先阅读文本，但完整验收需要 GPU 会话

### 前置页面

- [3.1 基础篇](../03_Triton_Kernels/intro.md)
- [3.2 过渡篇](../03_Triton_Kernels/intro.md)
- [3.3 进阶 A：Attention 优化](../03_Triton_Kernels/intro.md)
- [3.4 进阶 B：推理优化](../03_Triton_Kernels/intro.md)
- [3.5 项目篇](../03_Triton_Kernels/intro.md)

### Part 3 前导路径

如果你还没有完成 Triton 主线，建议先完成 Part 3 的 `01-14`，尤其是 `07-14` 的 GPU 实战页，再回来继续 Part 4。

### 后续页面

- [4.1 CUDA 编程基础](./4_1.md)
- [4.2 系统级性能优化](./4_2.md)
- [4.3 分布式训练工程](./4_3.md)
- [4.4 架构视野](./4_4.md)
