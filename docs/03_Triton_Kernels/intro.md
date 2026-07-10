# Part 03: Triton Kernel Development | 第三部分：Triton 算子开发

## Part Overview | Part 概览

本部分位于 Part 2 之后、Part 4 之前，重点是把第二部分的算法实现进一步落到 Triton 层，形成从 `PyTorch -> Triton` 的算子实现过渡：先理解算子行为，再把 kernel、融合、调优和 profiling 真正落到 GPU 上。它承担的是“从框架级实现走向高性能 kernel”的中间层作用，也是后续 CUDA 系统优化的重要前导。

Part 3 更像一张 Triton 实战地图：基础 kernel、设计模式、Attention 优化、推理优化和项目收口可以从不同入口进入，最后都汇到可运行、可调优的 kernel 实战。

## Part Asset Overview | Part 资产总览

本章内容按 5 个主题组组织，后续页面也沿该结构继续扩展。

> 导航说明：先看总览，再进入具体组页。
> 组页负责组内阅读顺序与资产收口，不需要一次性读完全部页面。
> Part 3 既是 Triton 算子开发目录，也是 Part 2 之后、Part 4 之前的实现衔接层。

| 学习组 | 职责作用 | 当前内容映射 | 每组多少节 |
|:---|:---|:---|:---|
| [3.1](./3_1.md) | 建立 Triton 编程模型和基础 kernel 直觉 | [01](./01_Triton_Vector_Addition.md)、[02](./02_Triton_Fused_SwiGLU.md)、[03](./03_Triton_Fused_RMSNorm.md)、[04](./04_Triton_GEMM_Tutorial.md)、[05](./05_Triton_Autotune_and_Profiling.md) | 5 |
| [3.2](./3_2.md) | 过渡到 Softmax 和设计模式 | [06](./06_Triton_Fused_Softmax.md)、[06.5](./06_5_Triton_Design_Patterns.md) | 2 |
| [3.3](./3_3.md) | 推进 Attention 路径上的算子优化 | [07](./07_Triton_Fused_RoPE.md)、[08](./08_Triton_Flash_Attention.md)、[09](./09_Triton_PagedAttention.md) | 3 |
| [3.4](./3_4.md) | 补齐推理侧的压缩和多 LoRA | [10](./10_Triton_Quantization.md)、[11](./11_Triton_Multi_LoRA.md) | 2 |
| [3.5](./3_5.md) | 收口调试、内存模型与综合项目 | [12](./12_Triton_Memory_Model_and_Debug.md)、[13](./13_Triton_Llama3_Block_Project.md)、[14](./14_Triton_Best_Practices_and_FAQ.md) | 3 |

## Learning Path | 学习路径

Part 3 可以按多条入口理解：基础入门入口先把 Triton 编程模型和基础 kernel 立住；Attention 优先、推理优先和项目优先入口则可以从不同工程目标切入，最后都回到项目篇。

### Recommended Order | 推荐顺序

- 基础入门入口：先看 [3.1](./3_1.md) -> [3.2](./3_2.md)
- Attention 优先入口：先看 [3.1](./3_1.md) -> [3.2](./3_2.md) -> [3.3](./3_3.md) -> [3.5](./3_5.md)
- 推理优先入口：先看 [3.1](./3_1.md) -> [3.2](./3_2.md) -> [3.4](./3_4.md) -> [3.5](./3_5.md)
- 项目优先入口：先看 [3.5](./3_5.md)，再回补 [3.1](./3_1.md)、[3.2](./3_2.md)、[3.3](./3_3.md)、[3.4](./3_4.md)
- 系统学习：按 [3.1](./3_1.md) -> [3.2](./3_2.md) -> [3.3](./3_3.md) -> [3.4](./3_4.md) -> [3.5](./3_5.md) 顺序推进

### Next Steps | 后续衔接

- 基础认知层：先看 [3.1](./3_1.md)、[3.2](./3_2.md)，把 Triton 编程模型、Softmax 和设计模式先立住，再按需要进入 [3.3](./3_3.md)。
- Attention 与推理层：先看 [3.3](./3_3.md)、[3.4](./3_4.md)，把 Attention 优化、量化和多 LoRA 的实现链路理顺，主要衔接后续项目篇。
- 项目收口：最后看 [3.5](./3_5.md)，把调试、内存模型和综合项目串起来，并为 Part 4 的系统优化提供实现背景。

## Environment Notes | 环境说明

- 整体定位：GPU-required
- 这里只写 Part 级统一前提，不点到具体节号
- 完整体验需要 NVIDIA GPU，推荐 Linux + CUDA + Triton
- 少数 notebook 可能支持 CPU fallback 或仅用于阅读，但不构成第三部分的标准运行路径
