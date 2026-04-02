# 讨论题 05：通信拓扑与分布式基石 (Communication & Distributed Topology)

**难度：** Hard | **标签：** `系统架构`, `分布式训练`, `Megatron` | **目标人群：** 核心 Infra 与算子开发

> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../02_PyTorch_Algorithms/20_Tensor_Parallelism_Sim.ipynb`](../02_PyTorch_Algorithms/20_Tensor_Parallelism_Sim.ipynb)
> [`../03_CUDA_and_Triton_Kernels/16_Distributed_Communication_Primitives.ipynb`](../03_CUDA_and_Triton_Kernels/16_Distributed_Communication_Primitives.ipynb)

> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../02_PyTorch_Algorithms/20_Tensor_Parallelism_Sim.ipynb`](../02_PyTorch_Algorithms/20_Tensor_Parallelism_Sim.ipynb)
> [`../03_CUDA_and_Triton_Kernels/16_Distributed_Communication_Primitives.ipynb`](../03_CUDA_and_Triton_Kernels/16_Distributed_Communication_Primitives.ipynb)

在大模型的训练中，单卡的算力是微不足道的。所有的千亿参数模型都是在几千甚至上万张 GPU 组成的集群上跑出来的。当算法工程师跨过单机代码的门槛，面临的最大挑战就是**通信瓶颈 (Communication Bottleneck)**。

---

## Q1：机内通信与机外通信的带宽差距有多大？为什么这是分布式策略设计的核心？

<details>
<summary>点击展开查看解析 (Solutions)</summary>

在现代 AI 服务器集群（如基于 NVIDIA DGX A100/H100）中，GPU 之间交换数据有两条主要的高速公路：

1. **机内通信 (Intra-node Communication)**：
   - *技术*：**NVLink** (GPU 之间的点对点直连) 和 **NVSwitch** (全互联交换芯片)。
   - *速度*：单向约 300 GB/s，双向约 **600 GB/s (A100)** / **900 GB/s (H100)**。
   - *特点*：速度非常恐怖，仿佛 8 张卡在共用一块大规模的显存。
2. **机外通信 (Inter-node Communication)**：
   - *技术*：**InfiniBand (IB)** 或者是带 RDMA 的以太网 (RoCE)。
   - *速度*：目前顶级的网卡（如 ConnectX-6/7）带宽通常是 200 Gbps 或 400 Gbps，换算成字节大约是 **25 GB/s ~ 50 GB/s**。
   - *特点*：跨服务器通信，受限于网线、交换机，甚至机架之间的物理距离。

**核心结论：两者带宽相差了 10 倍到 20 倍以上！** 

**这就是为什么我们要设计非常复杂的 3D 并行策略（DP + TP + PP）：**
任何需要极高频次、极大数据量同步的操作（如张量并行 Tensor Parallelism，前向和反向每层都要做一次 All-Reduce 通信），**必须被死死地按在单台机器（8 张卡）内部！** 如果跨机做 TP，那 90% 的时间 GPU 都在闲置等待那根可怜的 50 GB/s 的网线传数据。
</details>

---

## Q2：请简述大模型最常用的三种集合通信原语：All-Reduce, All-Gather, Reduce-Scatter 的区别。

<details>
<summary>点击展开查看解析 (Solutions)</summary>

这些是 MPI（Message Passing Interface）以及 NVIDIA 的 NCCL（NVIDIA Collective Communication Library）中最核心的操作。理解它们，就能徒推导导 ZeRO 优化器和各种并行的通信量。

假设我们有 4 张卡（GPU 0~3），每张卡上有一段数据（如张量并行的切片，或各自算出的梯度）：

1. **All-Reduce（全规约）**：
   - *输入*：每张卡上有大小相等的一个张量 $T_i$。
   - *操作*：将所有卡上的 $T_i$ 按照某种数学运算（最常用的是求和 `Sum`）合并成一个完整的结果 $T_{sum} = T_0 + T_1 + T_2 + T_3$。
   - *输出*：合并后的 $T_{sum}$ 被完整地广播回**每一张卡**上。
   - *应用场景*：数据并行 (DP) 中用来同步梯度；张量并行 (TP) 的 RowParallelLinear 中同步前向/反向输出。
2. **All-Gather（全收集）**：
   - *输入*：每张卡上有一个被切分成小块的张量片段（大小为 $S$）。
   - *操作*：所有卡交换自己手上的片段，最终拼凑出一个完整的大规模张量（大小为 $4S$）。
   - *输出*：每张卡都获得了一份**完整拼凑好的大规模张量**。
   - *应用场景*：ZeRO-3 (前向传播时收集被切片的权重)；序列并行 (Sequence Parallelism) 收集切分的序列。
3. **Reduce-Scatter（规约并散布）**：
   - *输入*：每张卡上有一个大规模的张量 $T_i$。
   - *操作*：这是 **Reduce + Scatter** 的合体。首先像 All-Reduce 一样把所有卡的 $T_i$ 求和得到一个大规模张量 $T_{sum}$，然后把它均匀切成 4 份。
   - *输出*：GPU 0 只拿到切好的第 0 份，GPU 1 拿到第 1 份...（每张卡上的结果只有原来的 $1/4$ 大小）。
   - *应用场景*：ZeRO-2 (反向传播后计算切片的梯度)；Megatron Sequence Parallelism 中替代 All-Reduce 节省一半带宽。

*核心数学奥秘*：在底层 NCCL 的 Ring 算法实现中，一次 **All-Reduce** 实际上就是先做一次 **Reduce-Scatter**，再做一次 **All-Gather**。
</details>

---

## Q3：为什么张量并行（TP）通常被限制在单机内？而流水线并行（PP）可以跨机？

<details>
<summary>点击展开查看解析 (Solutions)</summary>

这是大模型分布式训练（如 Megatron-LM）架构设计中最基础的黄金法则。

**1. 张量并行 (Tensor Parallelism, TP) 的极度饥渴：**
- *机制*：TP 是把一个大规模的矩阵乘法切成好几块，让不同的卡算。例如算 $Y = XA$。
- *通信频率*：**极高**。在 Transformer 的每一层的自注意力块和 MLP 块，只要碰到矩阵乘法结束，所有的卡就必须立刻停下来，做一个非常庞大的 All-Reduce 通信，把各自算了一半的结果加起来，才能进入下一层。
- *结论*：如果跨机做 TP，非常缓慢的网卡速度会让整个集群卡死在 All-Reduce 上。因此，TP 度 (TP Size) 通常最大为 8（即单台带有 NVLink 的服务器）。

**2. 流水线并行 (Pipeline Parallelism, PP) 的点对点传递：**
- *机制*：PP 是把一个深达 80 层的 Transformer 切成好几段。比如机器 A 负责 1~20 层，算完后把中间激活值 (Activations) 直接扔给机器 B（负责 21~40 层）。
- *通信频率*：**极低**。只有在微批次 (Micro-batch) 跨越机器边界时，才会发生一次点对点 (Point-to-Point, Send/Recv) 的通信。传输的数据量仅仅是边界处的激活值大小，相比于全连接层的权重和梯度，小了几个数量级。
- *结论*：PP 对网络带宽的要求最低。因此，当我们用成百上千台服务器训练万亿参数模型时，最高维度的切分永远是 流水线并行 (跨机) 和 数据并行 (跨机)，而最底层的切分是 张量并行 (单机)。
</details>
