# 讨论题 04：注意力机制变体与显存优化 (Attention Variants & Memory Optimization)

**难度：** Medium | **标签：** `模型架构`, `Attention`, `KV Cache` | **目标人群：** 通用基础 (算法/Infra)

在自回归生成过程中，大型语言模型面临严重的访存瓶颈 (Memory Bound)，主要原因之一在于每次生成新 Token 时都需要频繁读取之前所有 Token 的 KV Cache。为了减少显存占用并提升推理吞吐量，业界从**模型架构改进**（如 MQA/GQA/MLA）和**底层系统内存管理**（如 PagedAttention）两个维度提出了多种解决方案。

## 本节如何和实战篇配合

这一节不单独配 Notebook，而是和 Chapter 2 / 3 的实战页一起学：

- 先看本文，理解 KV Cache、MHA/MQA/GQA/MLA 和 PagedAttention 的直觉
- 再去 Chapter 2 / 3 的实战页，看这些概念如何影响实际推理和内存管理
- 如果后面要优化推理吞吐，这一页负责告诉你**为什么 KV Cache 会成为瓶颈**，实战篇负责告诉你**怎么验证优化是否有效**

这节的目标不是让你立刻实现一个完整推理引擎，而是让你能判断：问题到底是 Attention 结构、缓存组织，还是内存碎片。

> **相关阅读**:  
> 请前往实战篇进行相关代码练习：  
> [`../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb`](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)  
> [`../02_PyTorch_Algorithms/17_vLLM_PagedAttention.ipynb`](../02_PyTorch_Algorithms/17_vLLM_PagedAttention.ipynb)  
> [`../03_Triton_Kernels/09_Triton_PagedAttention.ipynb`](../03_Triton_Kernels/09_Triton_PagedAttention.ipynb)  

---

## Q1：自回归生成中，标准多头注意力 (MHA) 的 KV Cache 显存占用是如何计算的？为什么它是推理的主要瓶颈？

<details>
<summary>点击展开查看解析</summary>

在标准的多头注意力机制 (Multi-Head Attention, MHA) 中，假设有 H 个 Query 头，那么同样也有 H 个 Key 头和 Value 头。

**KV Cache 计算公式**：
对于每一层，每个 Token 需要缓存的 KV 显存大小为：
Size = 2 × (K 和 V) × H（头数）× d_head（单头维度）× Bytes_per_param

**一个数量级例子（以 LLaMA 2 7B 的典型配置为例）**：

```text
H = 32
d_head = 128
Bytes_per_param = 2（FP16）

每层、每个 token 的 KV Cache：
2 × 32 × 128 × 2 = 16,384 bytes = 16 KB

32 层后，每个 token 的 KV Cache：
16 KB × 32 = 512 KB

当序列长度为 2048 时，单序列 KV Cache：
512 KB × 2048 ≈ 1 GB

如果 batch size = 32：
约 32 GB 仅用于 KV Cache
```

**为什么它是主要瓶颈？**
随着生成长度 (Sequence Length) 和并发请求数 (Batch Size) 的增加，KV Cache 的体积会呈线性甚至超线性增长。由于自回归生成每步只产生一个 Token，GPU 往往需要频繁读取之前积累的庞大 KV Cache，将其从显存 (HBM) 送入计算所需的片上缓存 (如 SRAM) 参与计算。这种极低的计算/访存比 (Arithmetic Intensity) 导致 GPU 的计算核心大量时间处于空闲等待状态，形成严重的访存受限 (Memory Bound)。
</details>

---

## Q2：MQA (Multi-Query Attention) 和 GQA (Grouped-Query Attention) 是如何通过架构改进缓解 KV Cache 压力的？

<details>
<summary>点击展开查看解析</summary>

为了从根本上削减需要搬运的数据量，研究人员改变了 Attention 的投影结构：

1. **MQA (Multi-Query Attention)**:
   - **机制**：无论有多少个 Query 头，所有 Query 头都**共享仅仅 1 个 Key 头和 1 个 Value 头**。
   - **收益**：KV Cache 中与 Key/Value 相关的头数从 `H` 降到 `1`，因此缓存大小近似缩小为 MHA 的 `1/H`。这会明显降低访存需求，并提升推理速度。
   - **代价**：模型表达能力有所下降，可能影响复杂任务的生成质量，且训练不够稳定。

2. **GQA (Grouped-Query Attention)**:
   - **机制**：一种折中方案。将 Query 头进行分组（例如 32 个 Query 头分成 8 组），每组内的 Query 头共享 1 对 Key/Value 头。
   - **收益**：KV Cache 中与 Key/Value 相关的头数从 `H` 降到 `G`，因此缓存大小近似缩小为 MHA 的 `G/H`。例如 `H = 32, G = 8` 时，KV Cache 约为 MHA 的 `1/4`，也就是压缩了 `75%`。在保持较强表达能力的同时，推理成本明显下降，是目前工业界的主流方案之一。
</details>

---

## Q3：PagedAttention 是如何从系统层面（内存管理）解决 KV Cache 显存碎片的？

<details>
<summary>点击展开查看解析</summary>

除了修改模型架构，系统层面的优化同样重要。早期的推理引擎在显存中为每个请求预先分配一块连续的内存空间用于存放 KV Cache。

**连续分配的痛点**：
生成文本的长度是不可预知的。如果预分配过大，会导致严重的**内部碎片 (Internal Fragmentation)**；如果请求动态变化，会导致显存中出现大量无法被利用的**外部碎片 (External Fragmentation)**。在不少传统实现中，有效显存利用率会明显偏低。

连续预分配的问题不在于“算力不够”，而在于“显存切得不够灵活”。请求一旦长度分布不一致，显存就容易被不同长度的预留区间切碎。

**PagedAttention (vLLM 的核心技术) 的解决方案**：
借鉴操作系统中的虚拟内存分页机制。
1. **分页管理**：将 KV Cache 划分为固定大小的内存块（Block，例如每个 Block 存放 16 个 Token 的数据）。
2. **非连续存储**：不同 Token 的 Block 在物理显存中不需要连续存储，而是通过一个块表 (Block Table) 进行映射。
3. **按需分配**：只有当系统真正生成新 Token 且当前 Block 写满时，才会动态分配下一个物理 Block。
**收益**：显著缓解了显存外部碎片问题，提升了显存利用率，从而允许服务器在相同显存下承载更多的并发请求。

一个简化判断是：连续预分配更容易“显存被切碎”，PagedAttention 更接近“按页管理、按需扩展”，因此通常更适合长短不一、并发较高的推理场景。
</details>

---

## Q4：为什么 DeepSeek 提出的 MLA (Multi-Head Latent Attention) 能实现更高比例的 KV Cache 压缩？

<details>
<summary>点击展开查看解析</summary>

MLA (Multi-Head Latent Attention) 是 DeepSeek-V2/V3 模型中首创的核心架构，旨在尽量保留接近 MHA 的表达能力，同时实现比 MQA 更小的 KV Cache。

**机制与原理**：
1. **低秩压缩 (Low-Rank Compression)**：MLA 并不直接缓存庞大的 K 和 V 矩阵。相反，它将过去的 KV 信息压缩成一个低维度的隐状态向量 (Latent Vector, c_t) 进行存储。
2. **动态恢复**：在注意力计算时，模型读取极小的隐状态向量 c_t，通过投影矩阵实时将其恢复成需要的 Key 和 Value 参与点积运算。
3. **RoPE 解耦**：为了兼容旋转位置编码 (RoPE)，MLA 将位置信息与内容信息解耦，单独缓存少量的 RoPE 相关的 Key 向量。

**收益**：
MLA 通过计算换显存。虽然在推理时增加了少量的矩阵乘法计算量，但由于大模型推理是 Memory Bound 的，这种权衡极具性价比。它在保持接近 MHA 的表达能力的同时，将 KV Cache 的体积压缩到更低的水平。对于具体能压缩多少，仍取决于 latent 维度、RoPE 处理方式和模型配置。
</details>
