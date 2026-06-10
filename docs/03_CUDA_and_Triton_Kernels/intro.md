# Chapter 3: CUDA C++ 与 Triton 算子开发

## 🎯 本章概览

本章聚焦大模型算子的高性能实现，重点是把 Chapter 2 的算法实现落到 Triton / CUDA / 分布式层。

### 零基础过渡 5 Task

| Task | 对应入口 | 一句话目标 |
|:---|:---|:---|
| Task 1 | [3.1 Triton 基础](./3_1.md) | 认识 Triton 的编程模型和基础 kernel 写法。 |
| Task 2 | [3.1 Triton 基础](./3_1.md) | 从向量加法、RMSNorm、SwiGLU 进入融合算子思维。 |
| Task 3 | [3.2 Triton 进阶](./3_2.md) | 把 softmax、RoPE、FlashAttention 和 KV cache 串起来。 |
| Task 4 | [3.3 Triton 项目](./3_3.md) | 学会调试、profiling 和项目化落地。 |
| Task 5 | [3.4 CUDA 内核与显存优化](./3_4.md) | 进一步进入 CUDA、通信与系统扩展。 |

### 学习组划分

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **3.1: Triton 基础** | 01-05 | Triton 入门与融合 | Medium |
| **3.2: Triton 进阶** | 06-11 | 复杂算子实现 | Hard |
| **3.3: Triton 项目** | 12-13 | 调试与综合项目 | Hard |
| **3.4: CUDA 内核与显存优化** | 15, 18-19 | Streams / Custom Kernel / Shared Memory | Very Hard |
| **3.5: CUDA 系统扩展** | 16-17, 20 | 通信、ZeRO 与技术选型 | Very Hard |

### 环境边界（代码审计版）

- **整体定位：GPU-required**
- **完整体验**：需要 NVIDIA GPU，且推荐 Linux + CUDA + Triton
- **代码审计结果**：Chapter 3 的 Triton / CUDA notebook 直接面向 GPU 内核、显存和通信行为，不能把 CPU 作为完整替代
- **例外说明**：少数页面可能支持 CPU fallback 或仅用于阅读，但不构成 Chapter 3 的标准运行路径

### 前置页面

- [2.1 基础算子](../02_PyTorch_Algorithms/2_1.md)
- [2.5 反向传播与显存优化](../02_PyTorch_Algorithms/2_5.md)

### 后续页面

- [3.1 Triton 基础](./3_1.md)
- [3.4 CUDA 内核与显存优化](./3_4.md)
- [3.5 CUDA 系统扩展](./3_5.md)

### 环境说明

详细的 GPU / CUDA / Triton 环境分层与平台建议，请见 [使用指南](../guide.md)。
