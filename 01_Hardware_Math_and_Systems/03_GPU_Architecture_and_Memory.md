# 03. GPU 物理架构、内存层级与核心硬件单元
> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial.ipynb`](../03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial.ipynb)
> [`../03_CUDA_and_Triton_Kernels/08_Triton_Flash_Attention.ipynb`](../03_CUDA_and_Triton_Kernels/08_Triton_Flash_Attention.ipynb)

在算法工程师的面试中，仅仅懂 PyTorch 是不够的。大语言模型 (LLM) 是典型的 **Memory Bound (访存受限)** 与 **Compute Bound (算力受限)** 交织的产物。
如果你不能将软件算法映射到 GPU 的物理硬件上，就无法写出高性能的 Triton/CUDA 算子。

本节我们将深入 GPU 的核心物理架构，涵盖计算单元 (Tensor Core)、内存结构 (SRAM vs HBM)、以及它们在现代大模型算法（如 FlashAttention）中的实际应用。

---

## 1. 架构演进：从 V100 到 H100 (Architecture Evolution)

NVIDIA 的 GPU 架构代际演进，本质上是为了适应深度学习（尤其是 Transformer）对**混合精度矩阵计算**和**极高显存带宽**的无底洞需求。

*   **Volta 架构 (V100 - 2017)**: 
    *   **革命性引入**：首次引入了专为深度学习矩阵乘加 (MMA) 设计的 **Tensor Core (张量核心)**，支持 FP16 混合精度计算。
*   **Ampere 架构 (A100 - 2020)**:
    *   **大模型标配**：支持了 **TF32 (Tensor Float 32)** 和更广泛的 FP16/BF16。
    *   **架构升级**：大幅提升了 HBM2 (High Bandwidth Memory) 的带宽（达到 1.5 - 2 TB/s）和 SRAM (L2 Cache) 的容量（达到 40MB）。引入了 MIG (多实例 GPU) 和非对称稀疏化 (Sparse Tensor Core)。
*   **Hopper 架构 (H100 - 2022)**:
    *   **专为 LLM 而生**：引入了原生的 **FP8 数据格式**和 **Transformer Engine**。
    *   **内存与调度**：加入了 Thread Block Cluster 和 TMA (Tensor Memory Accelerator)，允许在不经过寄存器的情况下直接进行 HBM 到 SRAM 的异步数据搬运，彻底解放了带宽。

---

## 2. 核心计算单元：Tensor Core 详解

**普通 CUDA Core vs Tensor Core**
*   **CUDA Core (FP32/INT32)**: 每次时钟周期只能执行一个标量的 FMA (Fused Multiply-Add，乘加) 操作：`d = a * b + c`。
*   **Tensor Core (FP16/BF16/FP8)**: 专为矩阵乘法设计。在单个时钟周期内，它可以执行一个完整的 $4 \t\times 4$ 矩阵的 MMA (Matrix Multiply-Accumulate) 操作：`D = A * B + C`。

**为什么它这么快？**
Tensor Core 利用了半精度 (FP16) 或更低精度 (FP8) 来加速乘法，同时使用单精度 (FP32) 的累加器来保证加法精度。由于 Transformer 的自注意力和 MLP 几乎全是密集的矩阵乘法 (GEMM)，Tensor Core 的算力通常是普通 CUDA Core 的 **10倍以上**（例如 A100 的 FP16 Tensor Core 算力可达 312 TFLOPs）。

---

## 3. 内存层级：打破 Memory Bound (SRAM vs HBM)

GPU 的内存结构像一个金字塔，越靠近计算单元的速度越快，但容量越小：

1.  **Registers (寄存器)**：
    *   速度最快（<1 个周期），容量极小（每个线程几十个 32-bit 寄存器）。
    *   如果变量太多发生 **Register Spilling (寄存器溢出)**，数据会被赶到极其缓慢的 Local Memory (物理上位于 HBM)。
2.  **Shared Memory (SRAM / 片上共享内存)**：
    *   速度极快（~19 TB/s），每个 SM (流多处理器) 只有几百 KB。
    *   **极度关键**：它是同一个 Block 内所有线程协作、交换数据的唯一高速通道。**Triton 的核心魔法就是帮你自动化管理了 SRAM 的分配和调度。**
3.  **L2 Cache**: 
    *   所有 SM 共享，几十 MB，用于缓冲 HBM 的读写。
4.  **HBM (全局显存 / Global Memory)**:
    *   容量大 (40GB ~ 80GB)，但速度相对极慢 (1.5 TB/s ~ 3 TB/s)。
    *   如果算子的每一次计算都需要去 HBM 走一遭（如 PyTorch 原生的多次小操作），就会触发严重的 **Memory Bound (访存受限)**。

---

## 4. 实际案例：FlashAttention 如何榨干 SRAM

在标准的自注意力机制中，$S = QK^T$ 产生了一个尺寸为 $N \t\times N$ 的巨大矩阵。
*   **PyTorch 原生**：计算出 $S$，把它**写回 HBM**；读取 $S$ 计算 Softmax，再**写回 HBM**；读取 Softmax 结果和 $V$，计算出最终结果。这种反复读写 $O(N^2)$ 大小数据的行为，直接导致了显存溢出 (OOM) 和速度极慢。

*   **FlashAttention 的底层逻辑 (Tiling + SRAM)**：
    1.  **切块 (Tiling)**：将巨大的 $Q, K, V$ 切成小块 (Blocks)，使得这些小块**刚好能塞进容量只有几百 KB 的 SRAM 中**。
    2.  **在 SRAM 内完成一切 (Fusion)**：把 $Q_{block}$ 和 $K_{block}$ 加载到 SRAM，利用 Tensor Core 算出 $S_{block}$。
    3.  **在线归约 (Online Softmax)**：在 SRAM 内部直接更新局部最大值和指数和，避免写回 $S$。
    4.  最后再乘以 $V_{block}$，把最终结果写回 HBM。
    **结论**：把 $O(N^2)$ 的 HBM 读写完全消除，变成了只有 $O(N)$ 的读写。**FlashAttention 不是减少了计算量，而是通过 SRAM 彻底消灭了 Memory Bound！**

---

## 5. 多卡拓扑：PCIe vs NVLink

当单卡装不下模型时，我们需要分布式训练。GPU 之间的物理连接方式决定了通信带宽 (Communication Bound)：

*   **PCIe (外围组件互连)**：
    *   传统的插槽，带宽有限 (PCIe Gen4 双向 64 GB/s)。
    *   **拓扑痛点**：跨 GPU 通信通常需要经过 PCIe Switch 甚至 CPU，延迟高、带宽低。
*   **NVLink (NVIDIA 私有互连)**：
    *   专为 GPU-to-GPU 设计的高速通道。
    *   **A100 的 NVLink 3.0**：每条链路 50 GB/s，单卡 12 条，总双向带宽高达 **600 GB/s**。这比 PCIe 快了近 10 倍。
    *   **NVSwitch**：允许同一台物理机内的 8 张 GPU 实现全互连 (All-to-All) 的无阻塞通信，这是跑满 `All-Reduce` 和 `All-Gather` 极限带宽的硬件基础。
