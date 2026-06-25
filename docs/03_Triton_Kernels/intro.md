# 第三部分：Triton 算子开发

## 🎯 本部分概览

本部分聚焦大模型算子的高性能实现，重点是把第二部分的算法实现落到 Triton 层。

这条主线可以概括为 `PyTorch -> Triton`：先在 PyTorch 层理解算法与行为，再用 Triton 把算子高效落到 GPU。

如果你是在 Colab 里打开本部分，优先选择免费的 `T4 GPU`，或者任意可用的 GPU runtime。然后先运行 notebook 开头的环境准备单元；该单元会在 `triton` 缺失时自动安装依赖，避免直接在正文里 `import triton` 时报错。

### 零基础过渡 5 Task

| Task | 对应入口 | 一句话目标 |
|:---|:---|:---|
| Task 1 | [3.1 基础篇](../03_Triton_Kernels/3_1.md) | 认识 Triton 的编程模型和基础 kernel 写法。 |
| Task 2 | [3.2 过渡篇](../03_Triton_Kernels/3_2.md) | 通过 Softmax 和设计模式完成从基础算子到复杂算子的过渡。 |
| Task 3 | [3.3 进阶A：Attention优化](../03_Triton_Kernels/3_3.md) | 把 RoPE、FlashAttention 和 PagedAttention 串成一条 Attention 主线。 |
| Task 4 | [3.4 进阶B：推理优化](../03_Triton_Kernels/3_4.md) | 进入 Quantization 和 Multi-LoRA 的推理优化线。 |
| Task 5 | [06.5 Triton 设计模式与过渡总结](../03_Triton_Kernels/06_5_Triton_Design_Patterns.md) | 把 Triton 的常用模式收束成可复用骨架。 |

### 学习组划分

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **3.1: 基础篇** | 01-05 | Triton 入门与基础融合 | Medium |
| **3.2: 过渡篇** | 06, 06.5 | Safe Softmax 与设计模式桥接 | Medium |
| **3.3: 进阶A：Attention优化** | 07-09 | RoPE / FlashAttention / PagedAttention | Hard |
| **3.4: 进阶B：推理优化** | 10-11 | Quantization 与 Multi-LoRA | Hard |
| **3.5: 项目篇** | 12-14 | 调试、内存模型与综合项目 | Hard |

### 环境边界（代码审计版）

- **整体定位：Triton-required**
- **完整体验**：需要 NVIDIA GPU，且推荐 Linux + CUDA + Triton
- **代码审计结果**：第三部分的 Triton notebook 直接面向 GPU 内核与融合算子行为，不能把 CPU 作为完整替代
- **例外说明**：少数页面可能支持 CPU fallback 或仅用于阅读，但不构成第三部分的标准运行路径

### 前置页面

- [2.1 基础算子](../02_PyTorch_Algorithms/2_1.md)
- [2.5 反向传播与显存优化](../02_PyTorch_Algorithms/2_5.md)

### Part 1 前导路径

如果你希望先把 Part 3 的认知桥搭稳，建议回看 Part 1 的这条路径：

- **基础认知层**：`1B / 1D`
- **Triton 前置层**：`18 / 19`
- **分布式与系统边界**：`20`

如果你对 GPU 访存、block / warp、shared memory、算子融合还不熟，先按 `1B -> 1D -> 18 -> 19` 的顺序回看，再进入 3.1 / 3.2 / 3.3 / 3.4 / 3.5 会更顺。

### 后续页面

- [3.1 基础篇](../03_Triton_Kernels/3_1.md)
- [3.2 过渡篇](../03_Triton_Kernels/3_2.md)
- [06.5 Triton 设计模式与过渡总结](../03_Triton_Kernels/06_5_Triton_Design_Patterns.md)

### 环境说明

详细的 GPU / CUDA / Triton 环境分层与平台建议，请见 [使用指南](../guide.md)。
