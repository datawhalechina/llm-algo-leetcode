# 第三部分：Triton 算子开发

## 🎯 本部分概览

本部分聚焦大模型算子的高性能实现，重点是把第二部分的算法实现落到 Triton 层。

这条主线可以概括为 `PyTorch -> Triton`：先在 PyTorch 层理解算法与行为，再用 Triton 把算子高效落到 GPU。

从叙事上看，Part 3 其实分成两条并列主线：

- **主线一：Attention 优化（07-09）**，回答“Attention 怎么算得更快、更省显存？”
- **主线二：推理优化（10-11）**，回答“推理服务怎么支持更多用户、更省带宽？”

前者偏单次算子与 KV Cache 路径，后者偏权重压缩与多租户路由。它们共同构成完整的推理优化视图，而不是一条严格线性递进的章节链。

如果你是在 Colab 里打开本部分，优先选择免费的 `T4 GPU`，或者任意可用的 GPU runtime。然后先运行 notebook 开头的环境准备单元；该单元会在 `triton` 缺失时自动安装依赖，避免直接在正文里 `import triton` 时报错。

这不是从零开始的入门章，建议先完成 Part 1 的 `1B`、`1D`（其中的 `18 / 19` 也属于这条前导链）和 Part 2 的基础算子、模型组装相关内容，再进入 Part 3。

### Part 3 学习路径

- 先从 `3.1 基础篇` 进入，完成 Triton 的编程模型和基础 kernel 写法。
- 再到 `3.2 过渡篇`，用 Softmax 和设计模式把基础算子过渡到复杂算子。
- 接着顺着两条主线推进：
  - `3.3 进阶A：Attention优化`
  - `3.4 进阶B：推理优化`
- `06.5 Triton 设计模式与过渡总结` 负责把前面的常用模式收束成可复用骨架。
- 最后进入 `3.5 项目篇`，做调试、内存模型和综合项目。

### 学习组划分

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **3.1: 基础篇** | 01-05 | Triton 入门与基础融合 | Medium |
| **3.2: 过渡篇** | 06, 06.5 | Safe Softmax 与设计模式桥接 | Medium |
| **3.3: 进阶A：Attention优化** | 07-09 | RoPE / FlashAttention / PagedAttention | Hard |
| **3.4: 进阶B：推理优化** | 10-11 | Quantization 与 Multi-LoRA | Hard |
| **3.5: 项目篇** | 12-14 | 调试、内存模型与综合项目 | Hard |

### 环境边界（代码审计版）

- **整体定位：GPU-required**
- **完整体验**：需要 NVIDIA GPU，且推荐 Linux + CUDA + Triton
- **代码审计结果**：第三部分的 Triton notebook 直接面向 GPU 内核与融合算子行为，不能把 CPU 作为完整替代
- **例外说明**：少数页面可能支持 CPU fallback 或仅用于阅读，但不构成第三部分的标准运行路径

### 前置页面

- [2.1 基础算子](../docs/02_PyTorch_Algorithms/2_1.md)
- [2.5 反向传播与显存优化](../docs/02_PyTorch_Algorithms/2_5.md)

### Part 1 前导路径

如果你希望先把 Part 3 的认知桥搭稳，建议回看 Part 1 的这条路径：

- **基础认知层**：`1B / 1D`
- **Triton 前置层**：`18 / 19`（隶属 Part 1 的 `1D：异构调度与算子编程`）
- **分布式与系统边界**：`20`

如果你对 GPU 访存、block / warp、shared memory、算子融合还不熟，先按 `1B -> 1D -> 18 -> 19` 的顺序回看，再进入 3.1 / 3.2 / 3.3 / 3.4 / 3.5 会更顺。

### 后续页面

- [3.1 基础篇](../docs/03_Triton_Kernels/3_1.md)
- [3.2 过渡篇](../docs/03_Triton_Kernels/3_2.md)
- [06.5 Triton 设计模式与过渡总结](../docs/03_Triton_Kernels/06_5_Triton_Design_Patterns.md)

### 环境说明

详细的 GPU / CUDA / Triton 环境分层与平台建议，请见 [使用指南](../docs/guide.md)。
