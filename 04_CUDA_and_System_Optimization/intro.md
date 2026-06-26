# 第四部分：CUDA C++ 与系统优化

## 🎯 本部分概览

本部分聚焦 CUDA C++、系统性能优化、分布式训练工程和架构选型。它是从 Triton 走向更底层 CUDA 与系统边界的独立章节。

这条主线可以概括为 `CUDA Kernel -> System Optimization -> Distributed Engineering -> Architecture Trade-off`。

### 环境边界

- **整体定位：GPU-required**
- **完整体验**：需要 NVIDIA GPU，推荐 Linux + CUDA
- **代码审计结果**：本章直接面向 CUDA kernel、通信、系统优化和架构选型
- **阅读说明**：可以先阅读文本，但完整验收需要 GPU 会话

### 学习组划分

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **4.1: CUDA 编程基础** | 15-16 | Custom Kernel / Shared Memory | Hard |
| **4.2: 系统级性能优化** | 17-18 | Streams / Graph / JIT | Hard |
| **4.3: 分布式训练工程** | 19-20 | 通信原语 / ZeRO & Offload | Very Hard |
| **4.4: 架构视野** | 21-22 | 技术选型 / TCO | Very Hard |

### 前置页面

- [3.1 Triton 基础](../03_Triton_Kernels/intro.md)
- [3.2 Triton 进阶](../03_Triton_Kernels/intro.md)
- [3.3 Triton 项目](../03_Triton_Kernels/intro.md)

### Part 3 前导路径

如果你还没有完成 Triton 主线，建议先完成 Part 3 的 `01-14` 再回来继续 Part 4。

### 后续页面

- [4.1 CUDA 编程基础](./4_1.md)
- [4.2 系统级性能优化](./4_2.md)
- [4.3 分布式训练工程](./4_3.md)
- [4.4 架构视野](./4_4.md)
