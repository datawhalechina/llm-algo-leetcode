# 讨论题 08：编程模型演进 (Programming Models: CUDA & Triton)

**难度：** Hard | **标签：** `算子开发`, `编程模型`, `CUDA`, `Triton` | **目标人群：** 核心 Infra 与算子开发

尽管 PyTorch 提供了非常丰富的算子库，但在大模型训练和高性能推理中，算法工程师经常遇到现有算子组合无法满足性能需求的场景。掌握底层自定义算子的开发能力，是理解框架边界、深入硬件本质的重要路径。

本节我们将从自定义算子的动机出发，深入探讨 NVIDIA 原生的 CUDA C++ 编程模型，以及由 OpenAI 引领的下一代算子开发语言 Triton。

## 本节如何和实战篇配合

这一节不单独配 Notebook，而是和 Chapter 3 的 CUDA / Triton 实战页一起学：

- 先看本文，建立 thread / warp / block / grid、shared memory、Triton block 映射的直觉
- 再看 Chapter 3 的实战页，把这些概念落到具体 kernel 代码里
- 如果后面要自己写算子，这一页负责让你知道**编程模型为什么这样设计**，实战篇负责让你知道**如何真正实现和优化**

这节的目标不是让你立刻记住所有 CUDA API，而是让你能区分“线程级写法”和“块级写法”，并知道 Triton 为什么能降低门槛。

> **相关阅读**:  
> 请前往实战篇进行相关代码练习：  
> [`../04_CUDA_and_System_Optimization/15_CUDA_Custom_Kernel_Intro.ipynb`](../04_CUDA_and_System_Optimization/15_CUDA_Custom_Kernel_Intro.ipynb)  
> [`../04_CUDA_and_System_Optimization/16_CUDA_Shared_Memory_Optimization.ipynb`](../04_CUDA_and_System_Optimization/16_CUDA_Shared_Memory_Optimization.ipynb)  

---

## Q1：为什么我们需要用 CUDA 或 Triton 编写自定义算子？PyTorch 原生操作的瓶颈在哪里？

<details>
<summary>点击展开查看解析</summary>

在 PyTorch 等 Eager 模式的深度学习框架中，复杂算法通常是由多个基础操作（如加法、乘法、Softmax 等）拼接而成的。这种方式虽然易于开发，但在大模型场景下会暴露出两个严重的性能瓶颈：

1. **算子下发开销 (Kernel Launch Overhead)**：
   每次调用 PyTorch 的基础操作，CPU 都需要向 GPU 发送一次执行指令 (Kernel Launch)。对于小算子（如逐元素加法），GPU 的计算时间可能只有几微秒，而 CPU 的指令下发和调度时间却需要几十微秒。这会导致 GPU 大量时间处于空闲等待状态。

2. **显存墙瓶颈 (Memory Bound)**：
   每执行一个独立算子，GPU 都需要从低速的全局显存 (HBM) 中读取数据，计算完毕后再写回 HBM。如果是多个串联的操作（例如 RMSNorm 中的平方、求均值、除法），中间结果的反复读写会极大地浪费 HBM 带宽。

**解决方案：算子融合 (Operator Fusion)**
通过编写自定义的 CUDA 或 Triton 算子，我们可以将多个连续的计算步骤融合到一个完整的 Kernel 中执行。数据只需从 HBM 读取一次，放入高速的 SRAM 中进行所有阶段的计算，最后只将最终结果写回 HBM。这种优化通常能带来数倍的性能提升，FlashAttention 就是这一理念的典型例子。

**一个更直观的数量级例子**：

| 场景 | Kernel 数量 | 启动开销 | 计算时间 | 启动开销占比 |
| --- | --- | --- | --- | --- |
| 原生 PyTorch 的 3 个小算子 | 3 | 约 30 μs | 约 3 μs | 很高 |
| 融合后的单个 Kernel | 1 | 约 10 μs | 约 10 μs | 明显更低 |

如果一个 Kernel 的有效计算时间本来就很短，启动开销就会成为主要浪费。

**算术强度视角**：

```text
Arithmetic Intensity = FLOPs / Bytes
```

- 逐元素操作（例如 `y = x * 2`）的算术强度很低，通常更容易 Memory Bound
- 大矩阵乘法的算术强度很高，通常更容易 Compute Bound
- 把多个逐元素操作融合起来，通常能把算术强度提高 3-5 倍，从而把很多“小而碎”的操作从 Memory Bound 往上推
</details>

---

## Q2：请简述 CUDA 的线程层级结构与 GPU 硬件执行单元的对应关系。

<details>
<summary>点击展开查看解析</summary>

在使用 CUDA C++ 编写算子时，理解软件线程结构与底层 GPU 硬件多处理器的映射机制，是写出高性能代码的基础。

**1. 软件层级 (CUDA 编程视角)**：
- **Thread (线程)**：最基本的执行单元，通常负责计算张量中的一个标量元素。
- **Warp (线程束)**：32 个连续的线程组成一个 Warp。这是 GPU 硬件调度和执行的最小基本单位，遵循单指令多线程 (SIMT, Single Instruction Multiple Threads) 原则。
- **Block (线程块)**：由多个 Warp 组成（如 128 或 256 个线程）。**通常会由同一个 SM 负责调度和执行一个 Block。**
- **Grid (网格)**：由大量 Block 组成，代表了整个 Kernel 启动的总工作量。

**2. 硬件层级 (GPU 物理架构)**：
- **SP (Streaming Processor / CUDA Core)**：负责执行具体浮点或整数运算的物理单元，对应执行一个 Thread。
- **SM (Streaming Multiprocessor)**：流式多处理器。包含大量的 SP、高速共享内存 (Shared Memory / SRAM)、寄存器堆和指令调度器。**通常会将一个 Block 分配给一个 SM 运行。**
- **GPU 芯片**：包含几十到一百多个 SM（例如 H100 包含 132 个 SM），负责统筹执行整个 Grid 的工作负载。

**一个可落地的配置例子（A100 上的简单算子）**：

- 每个 SM 最多支持 2048 个线程
- 每个 Block 最多 1024 个线程
- 常见配置可以写成 `<<<grid=80, block=256>>>`
- `256` 个线程相当于 `8` 个 warp
- 理想情况下，如果寄存器和共享内存资源允许，一个 SM 可以同时驻留多个 Block 来提高吞吐
</details>

---

## Q3：在 CUDA 编程中，跨线程的数据共享和同步机制是如何限制和优化的？

<details>
<summary>点击展开查看解析</summary>

理解了 Q2 的软硬件映射后，我们就能清晰地把握 CUDA 编程中关于数据交互的物理边界。

1. **块内的高速交互 (Intra-Block)**：
   - 因为同一个 Block 的线程都在同一个 SM 上运行，它们可以访问 SM 内部容量极小（数十到数百 KB）但速度极快的**共享内存 (Shared Memory)**。
   - 开发者可以将需要重复访问的数据预取到共享内存中。在块内部，可以通过 `__syncthreads()` 指令实现轻量级、低延迟的线程同步机制。

**一个便于记忆的量化表**：

| 层级 | A100 量级 | 直觉 |
| --- | --- | --- |
| Shared Memory | 每 SM 约 164 KB | 容量很小，但足够做块内复用 |
| 访问延迟 | 约 20-30 cycles | 很快 |
| HBM 访问延迟 | 约 300-500 cycles | 慢很多 |

Shared Memory 通常比 HBM 快一个数量级以上，但它的容量也小得多。

2. **跨块的隔离墙 (Inter-Block)**：
   - 跨 Block 的线程可能被分配到不同的 SM，甚至分属于不同的时间片执行，其执行顺序通常不可预测。
   - **核心原则**：不要在 Kernel 执行期间依赖 Block A 去等待 Block B 的计算结果，否则很容易引发全局死锁 (Deadlock)。跨 Block 的数据同步通常要通过结束当前 Kernel 的执行，将结果写回全局内存 (HBM)，再启动下一个 Kernel 来实现。
</details>

---

## Q4：为什么 OpenAI 提出的 Triton 语言能明显降低算子开发的门槛？它与 CUDA 的范式有何不同？

<details>
<summary>点击展开查看解析</summary>

开发一个高性能的 CUDA C++ Kernel 需要手动管理极为繁琐的底层细节：分配共享内存、处理 Warp 级别的同步、规避存储体冲突 (Bank Conflict)、手动实施显存的向量化读取等。这具有极高的门槛。

OpenAI 提出的 Triton 通过编译器的抽象，显著改变了算子的开发范式。

**1. 编程粒度的变化：从 Thread 到 Block**
- **CUDA 范式 (SIMT)**：开发者通常需要以单一线程的视角编写代码（例如：“我这个 Thread 负责计算张量里的 `[i, j]` 标量元素”）。
- **Triton 范式 (Block-based)**：将编程粒度提升到了张量块。开发者可以像编写普通的 PyTorch 代码一样，直接操作一个大小为 `[BLOCK_SIZE]` 的张量块（Tile）。例如，执行 `tl.load()` 会自动将整个块的数据并行拉入 SRAM。

**2. 编译器的底层自动化**
当你写出基于 Block 的 Python 逻辑后，Triton 编译器会自动在底层完成复杂的工程处理：
- 自动将 Block 拆分并映射给底层 SM 的线程。
- 自动分析数据的生命周期，完成 Shared Memory 的分配。
- 自动插入底层的计算和内存同步屏障 (Barrier)。
- 自动生成全局内存的向量化读写指令，以尽量提高 HBM 带宽利用率。

**一个非常小的代码对比**：

```python
# Triton 风格（Block 级）
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(output_ptr + offs, x + y)
```

```cpp
// CUDA 风格（Thread 级）
__global__ void add_kernel(float* x, float* y, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = x[idx] + y[idx];
}
```

**总结**：Triton 让算法工程师能够以较低的 Python 心智负担，实现接近甚至媲美专家级 CUDA C++ 代码的硬件性能。它是目前开发自定义算子（如 FlashAttention、Fused RMSNorm）较高效的生产力工具。

</details>

---

## ⚠️ 常见误区

- `Triton` 不是天然比 `CUDA` 慢，写对了通常可以接近甚至持平
- `Block` 不是越大越好，过大可能导致寄存器溢出或 SM 资源不足
- `Warp` 是一个逻辑调度单位，固定为 32 线程
- `__syncthreads()` 只能在同一个 Block 内使用，不能跨 Block
- 学 Triton 仍然要理解 GPU 架构，不然很难写出高性能 kernel
