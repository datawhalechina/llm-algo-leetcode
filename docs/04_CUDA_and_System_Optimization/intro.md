# Part 04: CUDA C++ and System Optimization | 第四部分：CUDA C++ 与系统优化

## Part Overview | Part 概览

本部分位于 Part 3 之后，聚焦 CUDA C++、系统性能优化、分布式训练工程和架构选型。它承担的是把 Triton 级算子进一步下沉到 CUDA 内核与系统层的作用，也是后续硬件规划和成本视角的收口层。

内容上，本部分沿着“从 kernel 到 system、从实现到选型”的路径展开：先理解 CUDA 编程基础和硬件执行方式，再过渡到系统级性能优化、分布式训练工程，最后落到架构视野与成本判断。

## Part Asset Overview | Part 资产总览

本部分分为 4 个学习组，先建立 CUDA 编程与硬件直觉，再过渡到系统优化、分布式训练和架构视野。

| 学习组 | 职责作用 | 当前内容映射 | 每组多少节 |
|:---|:---|:---|:---|
| [4.1](./4_1.md) | 建立 CUDA 编程与硬件执行直觉 | [15](./15_CUDA_Custom_Kernel_Intro.md)、[16](./16_CUDA_Shared_Memory_Optimization.md)、[02.1](./02_1_Bank_Conflict_Deep_Dive.md)、[03](./03_Tensor_Core_MMA_Programming.md)、[04](./04_Warp_Level_Primitives.md) | 5 |
| [4.2](./4_2.md) | 理解系统级性能优化手段 | [05](./17_PyTorch_CUDA_Streams_and_Transfer.md)、[06](./18_CUDA_Graph_and_JIT_Compile.md)、[07](./07_Async_Data_Prefetch_and_Double_Buffering.md)、[07.1](./07_1_Double_Buffering_Deep_Dive.md)、[08](./08_Memory_Pool_and_VRAM_Management.md) | 5 |
| [4.3](./4_3.md) | 形成分布式训练工程链路 | [09](./19_Distributed_Communication_Primitives.md)、[09.1](./09_1_Ring_AllReduce_Deep_Dive.md)、[10](./20_DeepSpeed_Zero_Config.md)、[11](./11_Communication_Computation_Overlap_Advanced_Scheduling.md)、[12](./12_Heterogeneous_Training_CPU_Offload_NVMe_Offload.md) | 5 |
| [4.4](./4_4.md) | 建立架构视野与成本判断 | [13](./21_CUDA_vs_Triton_vs_PyTorch.md)、[14](./22_TCO_and_Hardware_Selection.md)、[15](./15_Inference_Service_Architecture_Design.md)、[16](./16_Future_Hardware_Trends.md) | 4 |

## Learning Path | 学习路径

Part 4 可以按多条入口理解：Kernel 优先入口先把 CUDA 编程与执行模型立住；系统优先、分布式优先和架构优先入口则可以从不同工程目标切入，最后都回到架构与成本收口。

### Recommended Order | 推荐顺序

- Kernel 优先入口：先看 [4.1](./4_1.md) -> [4.2](./4_2.md) -> [4.3](./4_3.md) -> [4.4](./4_4.md)
- 系统优先入口：先看 [4.2](./4_2.md) -> [4.3](./4_3.md) -> [4.4](./4_4.md)
- 分布式优先入口：先看 [4.3](./4_3.md) -> [4.4](./4_4.md)
- 架构优先入口：先看 [4.4](./4_4.md)，再回补 [4.1](./4_1.md) -> [4.3](./4_3.md)

### Next Steps | 后续衔接

- kernel 与硬件层：先看 [4.1](./4_1.md)、[4.2](./4_2.md)，把 CUDA、shared memory、调度和异步执行的底座立起来。
- 分布式与工程层：先看 [4.3](./4_3.md)，把通信原语、ZeRO / Offload 和多机多卡工程链路串起来。
- 架构与选型层：先看 [4.4](./4_4.md)，把 CUDA、Triton、PyTorch 的技术边界和成本判断串起来。

## Environment Notes | 环境说明

- 整体定位：GPU-required
- 完整体验需要 NVIDIA GPU，推荐 Linux + CUDA
- 少数页面可能支持阅读或局部执行，但不构成第四部分的标准运行路径
