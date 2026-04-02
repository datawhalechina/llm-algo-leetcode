# 讨论题 4：Attention 变体与显存优化 (MQA / GQA / MLA)

> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb`](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)
> [`../02_PyTorch_Algorithms/15_vLLM_PagedAttention.ipynb`](../02_PyTorch_Algorithms/15_vLLM_PagedAttention.ipynb)
> [`../03_CUDA_and_Triton_Kernels/10_Triton_KV_Cache_and_PagedAttention.ipynb`](../03_CUDA_and_Triton_Kernels/10_Triton_KV_Cache_and_PagedAttention.ipynb)

**难度：** Medium | **标签：** `模型架构`, `Attention`, `KV Cache` | **目标人群：**通用基础 (算法/Infra)

在上一节我们得知，大模型的自回归生成是极度的 Memory-bound，因为每次生成都要读取之前所有 Token 的 KV Cache。
为了强行减小这个庞大的内存读取量，算法界对 Attention 的架构动了刀子。

---

## Q1：请简述 MQA, GQA 和 MLA 的本质区别，以及它们是如何优化 KV Cache 占用的？

<details>
<summary>点击展开查看解析</summary>

假设隐藏层维度为 $d$。每个 Token 的 KV Cache 占用 $2 \t\times d \t\times 2$ (Bytes, 假设 BF16)。

1. **MQA (Multi-Query Attention)**：
   - *做法*：无论有多少个 Query 头，强行让它们**共享仅仅 1 个 Key 头和 1 个 Value 头**。
   - *收益*：KV Cache 大小直接缩小为原来的 $\frac{1}{\text{num\_heads}}$！推理速度高效飙升。
   - *代价*：模型表达能力严重下降，训练容易不稳定。

2. **GQA (Grouped-Query Attention, 工业标准如 LLaMA 2/3)**：
   - *做法*：折中方案。将 Query 头分组（例如 32 个 Query 头分成 8 组），每 4 个 Query 头共享 1 个 Key/Value 头。
   - *收益*：性能接近 MQA（KV Cache 缩小为 1/8），但模型能力几乎和原始的多头注意力（MHA）一样强。我们在 `02` 目录的代码实战中实现过它的张量 `repeat` 逻辑。

3. **MLA (Multi-Head Latent Attention, DeepSeek-V2/V3 首创)**：
   - *做法*：非常硬核的矩阵低秩分解。它不存庞大的 K 和 V，而是只存一个被严重压缩过的隐藏状态（Latent Vector），在计算 Attention 时再把它投影恢复成 K 和 V，或者利用 RoPE 解耦技术融合。
   - *收益*：在保持 MHA 的强大陆力的同时，把 KV Cache 的容量压缩到了甚至比 MQA 还小的极致。
</details>
