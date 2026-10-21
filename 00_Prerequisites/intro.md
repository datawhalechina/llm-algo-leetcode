# 第零部分：前置知识与环境准备

## Part 概览

第零部分覆盖 Python、NumPy、PyTorch、数据入口、训练循环、调试和性能意识，是 Part 2-4 的共同前置。完成这些页面后，学习者应能阅读代码与 shape、跟踪训练状态，并根据时间、容量和正确性证据排查问题。当前 Part 0 采用 notebook-first 统一管理，逻辑编号为 `01 ~ 20`，并按 `0A ~ 0E` 五组组织。


## Part 资产总览

本章分为 5 个学习组，按“Python / NumPy 与工程入口 → Tensor / Autograd → 模型与训练生命周期 → Transformer Block 与模型组件 → 调试与性能”的顺序推进，逐步连接函数契约、张量变换、训练状态、数据入口和模型组件。

| 学习组 | 核心职责 | 逻辑编号 | 组内目标 |
|:---|:---|:---|:---|
| [0A](./0A.md) | Python、NumPy 与工程入口 | `01 ~ 04` | 4 节 |
| [0B](./0B.md) | PyTorch 张量与自动求导 | `05 ~ 08` | 4 节 |
| [0C](./0C.md) | PyTorch 模型与训练生命周期 | `09 ~ 12` | 4 节 |
| [0D](./0D.md) | Transformer Block 与模型组件 | `13 ~ 16` | 4 节 |
| [0E](./0E.md) | 运行诊断与实验基础 | `17 ~ 20` | 4 节 |

## 学习路径

Part 0 建议按组推进，逐步补齐表达、数据入口、PyTorch、训练直觉、调试和性能意识这些基础能力。若你的目标是尽快进入 Part 2，优先把 `0B / 0C / 0D` 串起来，再回看 `0A` 和 `0E` 的补充内容。

### 推荐顺序

- 快速入门：先看 [0A](./0A.md) → [0B](./0B.md)
- 面向 Part 2：先看 [0B](./0B.md) → [0C](./0C.md) → [0D](./0D.md)，再补 [0A](./0A.md) 和 [0E](./0E.md)。
- 系统学习：按 [0A](./0A.md) → [0B](./0B.md) → [0C](./0C.md) → [0D](./0D.md) → [0E](./0E.md) 顺序推进

### 面向后续部分

- [0A](./0A.md)、[0B](./0B.md)：服务 Part 2 / Part 3 的函数、数据入口与张量前置，重点帮你读懂 shape、layout、索引、配置 / I/O、计算图和梯度边界。
- [0C](./0C.md)、[0D](./0D.md)：服务 Part 2 的训练与模型前置，重点帮你接上 `nn.Module`、checkpoint、optimizer、训练闭环、residual stream 和基础组件机制。
- [0E](./0E.md)：服务 Part 2-4 的运行诊断前置，先检查结果有效性，再采集时间与容量证据并设计最小对照实验。

### Part 1 对照

Part 0 和 Part 1 是两层互补的前置：Part 0 先把代码写法、数据入口和训练入口补齐，Part 1 再把数量级、硬件和系统判断讲深。对于 Part 2 的学习来说，Part 1 不是必须先读完，但它能把“为什么这样写代码”背后的资源与系统原因补全。

- [P1: 1A](../01_Hardware_Math_and_Systems/1A.md)：`0A` / `0B` 的数量级对照，补数据类型、精度、参数量和 FLOPs 的判断。
- [P1: 1B](../01_Hardware_Math_and_Systems/1B.md)：`0B` / `0D` / `0E` 的性能对照，补 GPU 架构、访存和 profiling 直觉。
- [P1: 1C](../01_Hardware_Math_and_Systems/1C.md)：`0C` / `0D` / `0E` 的训练对照，补多卡通信、VRAM 计算和 checkpoint 代价。
- [P1: 1D](../01_Hardware_Math_and_Systems/1D.md)：`0B` / `0D` / `0E` 的执行对照，补 CUDA / Triton、调度和算子编程视角。
- [P1: 1E](../01_Hardware_Math_and_Systems/1E.md)：`0B` / `0E` 的编译对照，补图变换、算子融合与动态形状。
- [P1: 1F](../01_Hardware_Math_and_Systems/1F.md)：`0C` / `0E` 的设备适配对照，补运行条件、多任务隔离与成本计算。

## 环境说明

- **默认环境：** 使用 [`BASE-CPU`](../docs/guide.md#base-cpu) 运行题目与测试。
- **扩展实验：** 页面标注为 `GPU optional` 或 `GPU required` 时，再按对应专题 Quickstart 准备环境。
