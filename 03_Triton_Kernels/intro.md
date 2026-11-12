# Part 03: Triton Kernel Development | 第三部分：Triton 算子开发

## Part Overview | Part 概览

本部分承接 Part 02 的 PyTorch 算子与模型机制，聚焦 Triton 程序映射、kernel 实现、正确性和微基准；Part 04 再展开 CUDA C++、Runtime 与系统执行机制。这是学习内容的层次衔接，不表示每个工程优化都必须按 PyTorch、Triton、CUDA 依次重写。

## Part Asset Overview | Part 资产总览

本章按五组组织：先学习 Triton 程序与数据布局，再建立常用 kernel 和调优能力；之后按目标选择 Attention 或推理 kernel 分支，最后进入调试、框架集成与 Block 验证。各组的具体页面和组内顺序由对应组页维护。

| 学习组 | 职责作用 | 与其他组的关系 |
|:---|:---|:---|
| [3.1](./3_1.md) | 建立 Triton 程序、Block 与数据布局的映射 | 3.2 及后续 kernel 分支的共同基础 |
| [3.2](./3_2.md) | 用融合、归约、矩阵计算、任务分配和调优建立常用 kernel 能力 | 3.3、3.4 的实现基础；持久化是 GEMM 分支，自动调优是可复用的方法，不是读完所有算子后的终点 |
| [3.3](./3_3.md) | 实现 Attention、变长序列与 KV 寻址相关 kernel | 面向 Attention 路径的专题分支 |
| [3.4](./3_4.md) | 实现推理侧权重、适配器与 KV 状态计算路径 | 面向推理场景的专题分支，与 3.3 按目标选学 |
| [3.5](./3_5.md) | 调试 kernel，核对编译产物与实际执行，再按需接入 PyTorch 或模型组件 | 编译与集成验证出口；各页独立运行，不要求串行完成 |

## Learning Path | 学习路径

先用3.1建立索引、掩码与步长基础，再从3.2选择当前目标需要的计算结构。融合、行归约、矩阵乘法不是后续每条路线都必须完成的前置；持久化、反向与调优也按问题进入。3.3、3.4分别展开Attention与推理侧内核，3.5处理调试、编译和集成。

### Recommended Order | 推荐顺序

- 起步：完成[3.1](./3_1.md)，再从[3.2](./3_2.md)选择逐元素融合、行归约或矩阵分块案例；想系统练习常用算子时，可按组页的基本案例顺序展开。
- Attention方向：已有行归约与矩阵分块基础后，进入[3.3](./3_3.md)；需要反向、持久化或调优时再回到3.2相应分支，不把它们当作进入Attention的必修页。
- 推理kernel方向：按[3.4](./3_4.md)选择权重、适配器或KV路径，再回补所需的矩阵累加、归约与量化表示。
- 调试与集成：实现基础kernel后，即可按问题进入[3.5](./3_5.md)的调试、编译或PyTorch集成；组件练习已提供共享Attention实现，可以直接核对布局和调用，再按需回看3.3的算法。
- 完整学习：先完成基础与基本算子，再选择条件扩展、Attention和推理内核，最后做组件复测；这是推荐阅读顺序，不是文件执行依赖。

### Next Steps | 后续衔接

- 进入 Part 04 前，按需要回看 [3.2](./3_2.md) 的 kernel 与调优基础，以及 [3.5](./3_5.md) 中与组件集成相关的内容；Part 04 的主目标是 CUDA/Runtime/系统机制，不是 Triton 课程的强制下一步。

## Environment Notes | 环境说明

- **真实 kernel 执行：** 需要 NVIDIA GPU、CUDA 可用的 PyTorch 与 Triton；创建和预检见[算子优化实验 Quickstart](../topic_discussion/operator_optimization/quickstart.md)。
- **CPU 的用途：** 可阅读机制、运行不触发 Triton 的辅助逻辑；不能据此判定 GPU kernel 正确或更快。GPU 不可用时，目标 kernel 测试必须明确标记未运行。
