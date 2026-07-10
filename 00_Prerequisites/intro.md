# 第零部分：前置知识与环境准备

## Part 概览

第零部分用于补齐进入后续学习所需的基础能力，内容覆盖 Python、NumPy、PyTorch、训练循环、调试和性能意识，是 Part 2-5 的共同前置。当前 Part 0 采用 notebook-first 统一管理，逻辑编号为 `01 ~ 20`，并按 `0A ~ 0E` 五组收口。


## Part 资产总览

本章分为 5 个学习组，先补语言和表达，再补 PyTorch、训练直觉、调试与性能意识。

| 学习组 | 核心职责 | 逻辑编号 | 组内目标 |
|:---|:---|:---|:---|
| [0A](./0A.md) | 基础语言与数据表示 | `01 ~ 04` | 4 节 |
| [0B](./0B.md) | PyTorch 张量与自动求导 | `05 ~ 08` | 4 节 |
| [0C](./0C.md) | PyTorch 模型构建 | `09 ~ 12` | 4 节 |
| [0D](./0D.md) | 训练与模型直觉 | `13 ~ 16` | 4 节 |
| [0E](./0E.md) | 调试与性能 | `17 ~ 20` | 4 节 |

## 学习路径

Part 0 建议按组推进，逐步补齐表达、PyTorch、训练直觉、调试和性能意识这些基础能力。

### 推荐顺序

- 快速入门：先看 [0A](./0A.md) → [0B](./0B.md)
- 系统学习：按 [0A](./0A.md) → [0B](./0B.md) → [0C](./0C.md) → [0D](./0D.md) → [0E](./0E.md) 顺序推进

### 面向后续部分

- [0A](./0A.md)、[0B](./0B.md)：服务 Part 2 / Part 3 的表达与张量前置。
- [0C](./0C.md)、[0D](./0D.md)：服务 Part 2 的模型与训练前置。
- [0E](./0E.md)：服务 Part 2-5 的调试与性能前置。

### Part 1 对照

Part 0 和 Part 1 是两层互补的前置：Part 0 先把代码写法和训练入口补齐，Part 1 再把数量级、硬件和系统判断讲深。

- [1A](../01_Hardware_Math_and_Systems/1A.md)：`0A` / `0B` 的数量级对照，补数据类型、精度、参数量和 FLOPs 的判断。
- [1B](../01_Hardware_Math_and_Systems/1B.md)：`0B` / `0D` / `0E` 的性能对照，补 GPU 架构、访存和 profiling 直觉。
- [1C](../01_Hardware_Math_and_Systems/1C.md)：`0C` / `0D` / `0E` 的训练对照，补多卡通信、VRAM 计算和 checkpoint 代价。
- [1D](../01_Hardware_Math_and_Systems/1D.md)：`0B` / `0D` / `0E` 的执行对照，补 CUDA / Triton、调度和算子编程视角。
- [1E](../01_Hardware_Math_and_Systems/1E.md)：`0C` / `0E` 的选型对照，补编译优化、硬件判断和 TCO 视角。

## 环境说明

- 默认按 `CPU-first` 设计
- 少数页面会标注 `GPU optional` 或 `GPU required`
