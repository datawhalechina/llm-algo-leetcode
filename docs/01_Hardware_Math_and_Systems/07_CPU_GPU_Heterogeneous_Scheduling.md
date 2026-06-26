# 07. CPU GPU Heterogeneous Scheduling | CPU 与 GPU 异构调度 (CPU & GPU Heterogeneous Scheduling)

**难度：** Hard | **标签：** `系统架构`, `异构计算`, `Offload` | **目标人群：** 核心 Infra 与算子开发

在深度学习系统中，GPU 并非孤立运行的，它通常需要接受 CPU 的指令和数据投喂。当模型的参数和中间状态庞大到连 GPU 集群的显存（如多张 80GB A100）都难以完全装下时，如何利用廉价但海量的 CPU 内存（Host RAM）参与训练和推理，成为了系统优化的关键。

本节将从 CPU 与 GPU 的基础交互开始，逐步深入到流水线重叠与更激进的 Offload（卸载）技术。

如果先记一组数量级直觉，可以先抓住下面这个事实：

| 项目 | 典型量级 | 直觉 |
| --- | --- | --- |
| PCIe 4.0 x16 | 64 GB/s（双向） | CPU-GPU 之间的主瓶颈 |
| A100 HBM | 1.5 - 2.0 TB/s | GPU 侧本地显存带宽远高于 PCIe |
| 7B FP16 参数 | 14 GB | 单次搬运已经是“整块大对象” |
| 8 卡 DP 梯度同步 | 约 28 GB / iteration | 很容易把 PCIe/IB 压满 |

## 本节如何和实战篇配合

这一节不单独配 Notebook，而是和后面的实战页一起学：

- 先看本文，建立 Host / Device、PCIe、CUDA Streams 和 Offload 的直觉
- 再去实战篇看数据传输、流式调度和显存卸载的实现
- 如果后面要做训练加速或推理加速，这一页负责告诉你**为什么 CPU 和 GPU 之间会成为瓶颈**，实战篇负责告诉你**怎么把延迟隐藏起来**

这节的目标不是让你立刻会写复杂的异构调度器，而是让你能判断：问题是在传输、在调度，还是在显存容量。

> **相关阅读**:  
> 请前往实战篇进行相关代码练习：  
> [`../04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer.ipynb`](../04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer.md)  

---

## Q1：在异构计算中，Host 和 Device 分别扮演什么角色？它们之间的核心物理瓶颈是什么？

<details>
<summary>点击展开查看解析</summary>

在典型的异构计算（如基于 NVIDIA GPU 的服务器）体系中，存在两个核心的实体：
1. **Host (宿主机)**：指代 CPU 及其直接挂载的系统内存（RAM）。它负责运行操作系统、网络通信、数据预处理以及向 GPU 下发计算指令，扮演“指挥官”的角色。
2. **Device (设备)**：指代 GPU 及其自带的显存（HBM）。它拥有极高的并发计算能力，负责执行密集的张量矩阵运算，扮演“执行者”的角色。

**核心物理瓶颈：PCIe 总线带宽**
Host 与 Device 之间的数据搬运通常通过 PCIe (Peripheral Component Interconnect Express) 总线完成。
- **带宽悬殊**：以 PCIe 4.0 x16 为例，其双向理论带宽仅约为 64 GB/s（单向 32 GB/s）。相比之下，A100 内部的 HBM 带宽高达 1.5 - 2.0 TB/s。
- **系统影响**：这意味着，如果频繁地在 CPU 内存和 GPU 显存之间拷贝数据（如反复执行 `tensor.to('cuda')` 和 `tensor.to('cpu')`），极慢的 PCIe 传输将使得 GPU 的计算核心长时间处于饥饿等待状态，导致严重的性能下降。

**一个粗略的时间对比**：

- 以 7B 模型的 FP16 参数为例，权重大小约 `14 GB`
- 若通过 PCIe 4.0 以约 `32 GB/s` 的单向吞吐搬运，光传输就要约 `0.44 s`
- 对于单次前向推理来说，真正的计算时间通常远小于这类整块传输的等待时间

所以，很多“看起来只是拷贝一次数据”的代码，实际上会让 GPU 长时间卡在等 PCIe 上。
</details>

---

## Q2：既然跨设备传输 (PCIe) 如此缓慢，底层框架是如何利用 CUDA Streams 隐藏通信延迟的？

<details>
<summary>点击展开查看解析</summary>

为了解决 PCIe 带宽瓶颈，大模型训练框架（如 Megatron-LM、DeepSpeed）广泛采用了 **计算与通信重叠 (Overlap Computation and Communication)** 技术。

**核心机制：CUDA Streams 与异步执行**
在 CUDA 编程模型中，Stream（流）是一个按照顺序执行的指令队列。不同的 Stream 之间可以并行执行。通过 DMA (Direct Memory Access) 控制器，GPU 可以在不占用其计算核心（SM）的情况下，从 CPU 内存中异步拉取数据。

一个更好记的重叠公式是：

```text
总时间 ≈ max(传输时间, 计算时间)
```

只有当传输被计算“盖住”时，Overlap 才真正起作用；否则总时间仍然会被较慢的一边主导。

**重叠流水线的实现**：
假设我们需要处理一大批数据，可以将其切分为多个微块 (Micro-batches)：
- **阶段 1**：Stream 1 将数据块 A 从 Host 拷贝到 Device。
- **阶段 2**：Stream 1 开始在 GPU 上计算块 A。**同时**，Stream 2 开始将数据块 B 从 Host 拷贝到 Device。
- **阶段 3**：当块 A 计算完毕时，块 B 的数据刚好到达显存，Stream 2 可以直接接续开始计算块 B。

通过这种精密的异步调度，缓慢的 PCIe 传输时间会被尽可能“隐藏”在 GPU 密集的矩阵乘法计算时间之中，从而提升整体硬件利用率。

**什么时候无法完美重叠？**

- 传输时间明显大于计算时间，小模型或超长 KV Cache 搬运时尤其常见
- 没有正确设置异步拷贝参数，例如未使用 `non_blocking=True`
- Stream 设计过于复杂，反而引入了额外调度开销
- 数据预处理、CPU 端准备时间也被忽略，导致“GPU 端看起来重叠了”，实际上整体 pipeline 仍然停顿
</details>

---

## Q3：在资源极度受限的情况下，什么是 CPU Offload (卸载) 技术？它在训练和推理中分别如何应用？

<details>
<summary>点击展开查看解析</summary>

当我们在 06 节提到的 ZeRO-3 切分策略依然无法将模型塞进 GPU 显存时，系统就会采用 **Offload (卸载) 技术**：将部分显存压力转移到廉价且容量庞大的 CPU 内存（甚至 NVMe 固态硬盘）上。这本质上是用 PCIe 带宽换取显存容量。

1. **训练期的 ZeRO-Offload**：
   - 优化器状态（如 Adam 的动量和方差）通常占据了最多的显存。ZeRO-Offload 策略会将这部分状态全部转移到 CPU 内存中保存。
   - **执行流程**：GPU 完成反向传播算出梯度后，将梯度通过 PCIe 传给 CPU。CPU 利用自身的计算能力执行参数更新（Optimizer Step），然后将更新后的权重再通过 PCIe 传回 GPU，准备下一轮前向传播。
   - **量级直觉**：以 7B 模型为例，单卡梯度大约就是 `14 GB`；如果每次 iteration 都要做这类 GPU→CPU→GPU 的传输，光 PCIe 搬运就可能达到 `0.44 s` 量级，CPU 更新和同步开销还会继续叠加。它本质上是“用时间换显存”。

2. **推理期的 KV Cache Offload (如 vLLM 中的实现)**：
   - 在处理超长上下文或面临极高的并发请求时，GPU 显存可能无法容纳所有用户的 KV Cache。
   - **执行流程**：推理引擎会将当前暂时不活跃请求的 KV Cache “踢出”显存，保存到 CPU 内存中（换出，Evict）。当该请求再次被调度执行时，再从 CPU 内存拉回显存（换入，Swap-in）。这一机制完美复刻了操作系统中的**虚拟内存分页与缺页中断机制**，极大提升了单机的并发承载上限。
   - **开销分析**：换出和换入都会再次占用 PCIe 带宽。如果请求频繁切换，CPU RAM 就不再是“额外容量”，PCIe 会迅速变成新的瓶颈。vLLM 这类系统通常会结合预取 (Prefetch) 和缓存策略来降低抖动。
</details>

---

## Q4：如何判断一套异构调度方案是不是“看起来很并行，但实际上没省多少时间”？

<details>
<summary>点击展开查看解析</summary>

可以用一个很实用的判断框架来检查：

| 维度 | 无优化 | CUDA Streams Overlap | CPU Offload |
| --- | --- | --- | --- |
| 核心策略 | 串行传输 + 计算 | 传输与计算重叠 | 显存压力转移到 CPU |
| PCIe 利用率 | 低 | 高 | 高，但延迟更大 |
| 显存节省 | 0 | 0 | 显著 |
| 速度影响 | 基准 | 可能接近无影响 | 通常变慢 |
| 适用场景 | 显存充足 | 大多数训练/推理 | 显存极度受限 |

## ⚠️ 常见误区

- `tensor.to('cuda')` 不是总能自动异步；要正确使用异步拷贝条件，才能获得 overlap。
- Stream 不是越多越好，太多会增加调度复杂度。
- Offload 不是“额外扩容”，而是把显存瓶颈换成带宽和延迟瓶颈。
- CPU Offload 往往会降低吞吐，只是在显存不足时的妥协方案。
- KV Cache Offload 不能无限扩展并发，最终仍然会受 CPU 内存带宽和 PCIe 带宽限制。
</details>
