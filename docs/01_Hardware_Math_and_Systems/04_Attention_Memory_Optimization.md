# 04. Attention Memory Optimization | 注意力机制变体与显存优化

**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `Attention`, `KV Cache` | **目标人群：** 系统性能入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

自回归生成里的 attention 往往不是单纯算得慢，而是读得太多。每生成一个新 token，都要反复访问之前积累下来的 KV cache；上下文一长，瓶颈就会很快从矩阵乘法转到显存访问和缓存组织上。也正因为如此，注意力优化既有模型侧的结构改动，也有系统侧的内存管理改动。

这是一节**机制前置节**：它区分 Attention 结构、KV Cache 表示和缓存组织三类问题，主要服务 `推理优化路线`，也为 `显存优化路线` 提供 KV Cache 视角。本节是 **CPU-first；GPU 用于扩展验证**：CPU 代码可以验证张量形状、缓存增长和理论字节数；真实 GPU 或 backend 才能验证实际峰值显存、带宽、并发和服务吞吐。显存路线在这里重点观察 KV Cache 如何进入显存账本，不把本节的理论结果当成具体设备结论。

**关键词：** `MHA`, `MQA`, `GQA`

---
## 前置阅读
**导语：** 这一页先把单卡硬件、通信和显存推导接上，再进入注意力变体和推理内存优化，方便把 KV Cache 的问题放回整体系统里看。

- [Group 1B: Single-GPU Hardware and Memory Optimization | 1B: 单卡硬件与访存优化](./1B.md)
- [Group 1C: Distributed Communication and Memory Sharing | 1C: 多卡通信与显存共享](./1C.md)
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)
## 相关阅读
**导语：** 如果要继续把注意力优化接到具体实现和推理系统里，可以沿这条主线继续往下看。
- [04. Attention MHA GQA | 多头注意力](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
- [09. Triton PagedAttention | KV Cache 间接寻址](../03_Triton_Kernels/09_Triton_PagedAttention.md)
- [08. Triton Flash Attention | 真正的 Flash Attention 前向算子](../03_Triton_Kernels/08_Triton_Flash_Attention.md)

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
KV Cache 的容量通常随层数、序列长度、并发序列数和 KV 头数近似线性增长；prefill 阶段的 Attention 计算量还会随序列长度呈二次增长。decode 阶段每步只产生一个 Token，却要反复读取已有 KV Cache，因此可能受到显存带宽和缓存组织影响，是否成为 Memory Bound 仍取决于模型、dtype、并发和实现。
</details>
### Q1小验证：KV Cache 的线性增长直觉

把上下文长度翻倍，再看缓存大小是不是也近似翻倍。

```python
def kv_cache_bytes(seq_len, layers, heads, head_dim, dtype_bytes=2, batch_size=1):
    """估算 decode 阶段 KV Cache 的理论字节数。

    不包含 block metadata、padding、量化 scale、allocator reserve 或
    runtime workspace；这里的 heads 应该是 KV heads，而不是 query heads。
    """
    if min(seq_len, layers, heads, head_dim, dtype_bytes, batch_size) <= 0:
        raise ValueError('shape values 和 dtype_bytes 必须为正数')
    return 2 * batch_size * seq_len * layers * heads * head_dim * dtype_bytes

for seq_len in [1024, 2048, 4096]:
    size_gb = kv_cache_bytes(seq_len, 32, 32, 128) / 1e9
    print(f'seq_len={seq_len:4d} -> KV cache ≈ {size_gb:5.2f} GB')
```

## Q2：MQA (Multi-Query Attention) 和 GQA (Grouped-Query Attention) 是如何通过架构改进缓解 KV Cache 压力的？

<details>
<summary>点击展开查看解析</summary>

为了减少需要搬运的数据量，研究人员改变了 Attention 的投影结构：

1. **MQA (Multi-Query Attention)**:
   - **机制**：无论有多少个 Query 头，所有 Query 头都**共享仅仅 1 个 Key 头和 1 个 Value 头**。
   - **收益**：KV Cache 中与 Key/Value 相关的头数从 `H` 降到 `1`，因此缓存大小近似缩小为 MHA 的 `1/H`。这会明显降低访存需求，并提升推理速度。
   - **代价**：共享 K/V 头会改变模型表达容量，可能影响部分任务质量；影响程度取决于模型结构、训练方式和具体 checkpoint。

2. **GQA (Grouped-Query Attention)**:
   - **机制**：一种折中方案。将 Query 头进行分组（例如 32 个 Query 头分成 8 组），每组内的 Query 头共享 1 对 Key/Value 头。
   - **收益**：KV Cache 中与 Key/Value 相关的头数从 `H` 降到 `G`，因此在其他条件相同时，缓存大小近似缩小为 MHA 的 `G/H`。例如 `H = 32, G = 8` 时，KV Cache 约为 MHA 的 `1/4`；实际速度和质量变化仍取决于模型与 backend。
</details>
### Q2小验证：头数变化为什么会直接影响缓存

固定上下文长度，只改变 KV 头数，看看缓存怎么缩。

```python
def kv_cache_gb(seq_len, layers, kv_heads, head_dim, dtype_bytes=2, batch_size=1):
    return kv_cache_bytes(seq_len, layers, kv_heads, head_dim, dtype_bytes, batch_size) / 1e9

seq_len = 4096
layers = 32
head_dim = 128
for name, kv_heads in [('MHA', 32), ('GQA', 8), ('MQA', 1)]:
    print(f'{name:>3s}: kv_heads={kv_heads:2d}, KV cache ≈ {kv_cache_gb(seq_len, layers, kv_heads, head_dim):5.2f} GB')
```

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
**收益**：可以减少连续预分配带来的浪费，并改善可用显存的组织；实际并发提升还取决于 block size、调度、kernel 和其他 workspace。

一个简化判断是：连续预分配更容易“显存被切碎”，PagedAttention 更接近“按页管理、按需扩展”，因此通常更适合长短不一、并发较高的推理场景。
</details>
### Q3小验证：分页后的显存利用率

```python
def contiguous_reserved_tokens(lengths, max_len):
    return len(lengths) * max_len


def paged_reserved_tokens(lengths, block_size):
    return sum(((length + block_size - 1) // block_size) * block_size for length in lengths)

lengths = [64, 96, 128, 320, 512]
max_len = max(lengths)
block_size = 128
contiguous = contiguous_reserved_tokens(lengths, max_len)
paged = paged_reserved_tokens(lengths, block_size)
utilization = sum(lengths) / paged
waste = 1 - utilization

print(f'Contiguous reservation: {contiguous} tokens')
print(f'Paged reservation: {paged} tokens')
print(f'Paged utilization: {utilization:.1%}')
print(f'Paged waste: {waste:.1%}')

```

## Q4：为什么 DeepSeek 提出的 MLA (Multi-Head Latent Attention) 能实现更高比例的 KV Cache 压缩？

<details>
<summary>点击展开查看解析</summary>

MLA (Multi-Head Latent Attention) 是以 DeepSeek-V2/V3 为代表的一类 latent attention 设计，目标是在保留有效注意力信息的同时压缩 KV Cache 表示。具体缓存格式、RoPE 处理和收益取决于模型实现。

**机制与原理**：
1. **低秩压缩 (Low-Rank Compression)**：MLA 并不直接缓存庞大的 K 和 V 矩阵。相反，它将过去的 KV 信息压缩成一个低维度的隐状态向量 (Latent Vector, c_t) 进行存储。
2. **动态恢复**：在注意力计算时，模型读取极小的隐状态向量 c_t，通过投影矩阵实时将其恢复成需要的 Key 和 Value 参与点积运算。
3. **RoPE 解耦**：为了兼容旋转位置编码 (RoPE)，MLA 将位置信息与内容信息解耦，单独缓存少量的 RoPE 相关的 Key 向量。

**收益**：
MLA 通过额外计算和表示变换换取更小的缓存；是否值得取决于 latent 维度、RoPE 处理方式、模型质量和 backend。它可能降低 KV Cache 占用，但不能据此直接推出固定的速度或质量收益。
</details>

```python
def mla_gain(seq_len, kv_heads, compression_ratio=0.5):
    """用一个比例模型展示 KV 表示压缩的数量级直觉。

    这不是 MLA 的完整 cache 公式，也不代表实际质量或吞吐收益。
    """
    if seq_len <= 0 or kv_heads <= 0 or not 0 < compression_ratio <= 1:
        raise ValueError('seq_len、kv_heads 必须为正数，compression_ratio 必须在 (0, 1]')
    # MLA 不是简单压缩，而是把 KV cache 里的冗余表示换成更紧凑的路径。
    base = 2 * seq_len * kv_heads
    compressed = base * compression_ratio
    return {'base_units': base, 'compressed_units': round(compressed, 2), 'saving_ratio': round(1 - compression_ratio, 2)}

for case in [(1024, 32, 0.5), (4096, 32, 0.25), (4096, 16, 0.25)]:
    print(case, '->', mla_gain(*case))
print('MLA helps when the compressed representation still preserves useful attention structure')

```

## ⚠️ 常见误区

- `KV cache` 不是只和 token 数有关，它还和层数、batch size、KV 头数一起增长。
- `MQA / GQA` 不是单纯改名字，而是在实打实地压低缓存体积。
- `PagedAttention` 解决的是缓存管理和碎片化，不等于表示压缩。
- `MLA` 解决的是表示体积，不等于把调度和分配问题也一并解决。