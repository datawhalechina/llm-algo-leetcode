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

生成式推理每生成一个 token，都要访问前面 token 的 Key / Value。序列变长时，KV Cache 会同时带来容量和带宽压力；而 MQA、GQA 通过减少 KV 头数改变缓存规模，但不会自动解决缓存分配、复用和服务调度问题。
本节沿着“注意力结构 → KV Cache 成本 → 运行时组织”展开：先用公式和小张量理解 MHA 的缓存增长，再比较 MQA / GQA 如何改变 KV 头数，最后区分结构压缩与缓存管理对系统成本的不同影响。完成后，你应能根据层数、序列长度、并发和 KV 头数估算缓存规模，并解释不同优化作用在哪一类成本上。
学完本节后，你可以从 KV Cache 的组成、增长方式和共享方式出发，判断一个优化是在减少缓存表示，还是在改善缓存组织与复用。

**关键词：** `MHA`, `MQA`, `GQA`

![本节概念关系](../public/01_Hardware_Math_and_Systems/04_attention_kv_cache_map.svg)


---

## 前置阅读
**导语：** 先回顾 GPU 内存层级和 MHA/GQA 的基本结构，再理解 KV Cache 如何随层数、序列长度和并发增长，以及架构变体与运行时组织分别改变了什么。

- [Part 01 · 01 数据类型与精度](./01_Data_Types_and_Precision.md)
- [Part 01 · 03 GPU 架构与显存](./03_GPU_Architecture_and_Memory.md)
- [Part 02 · 04 Attention（MHA / GQA）](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
  
---
## Q1：自回归生成中，标准多头注意力（MHA）的 KV Cache 如何形成显存与带宽压力？

<details>
<summary>点击展开查看解析</summary>

标准多头注意力（MHA）为每一层、每个历史 token 保存 K 和 V。理论字节数可以写成：

```text
KV Cache bytes = 2 × batch × layers × seq_len × kv_heads × head_dim × dtype_bytes
```

其中 `2` 对应 K 和 V。以 `batch=1`、`layers=32`、`seq_len=2048`、`kv_heads=32`、`head_dim=128`、FP16 为例，单序列 KV Cache 约为 1.07 GB；并发序列增加到 32 时，理论容量需求也会近似扩大 32 倍。

缓存会经历 Prefill、Decode 和回收三个阶段：Prefill 建立初始 K/V，Decode 每生成一个 token 就追加并读取已有缓存，请求结束后释放或回收空间。序列长度和并发增长会带来容量压力，Decode 的反复读取还可能带来带宽压力；是否成为 Memory Bound 仍取决于模型、dtype、并发和实现。本节小验证先记录容量随序列长度变化的结果，真实延迟、带宽影响和回收策略交给推理项目验证。

| 变量 | 含义 | 对理论 KV Cache 的影响 |
| --- | --- | --- |
| `layers` | Transformer 层数 | 近似线性增长 |
| `seq_len` | 已缓存的上下文长度 | 近似线性增长 |
| `batch` / 并发 | 同时驻留的请求数 | 近似线性增长 |
| `kv_heads` | K/V 头数 | 近似线性增长，GQA / MQA 会改变它 |
| `head_dim` | 单个头的维度 | 近似线性增长 |
| `dtype_bytes` | 每个元素的字节数 | 近似线性增长，FP16 / BF16 通常为 2 |
</details>
### Q1小验证：观察 KV Cache 随序列长度的变化

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
base = kv_cache_bytes(2048, 32, 32, 128)
doubled = kv_cache_bytes(4096, 32, 32, 128)
assert doubled == 2 * base
print('seq_len doubled -> theoretical KV Cache ratio:', doubled / base)

```

## Q2：MQA (Multi-Query Attention) 和 GQA (Grouped-Query Attention) 是如何通过架构改进缓解 KV Cache 压力的？

<details>
<summary>点击展开查看解析</summary>

为了减少需要搬运的数据量，研究人员改变了 Attention 的投影结构：

1. **MQA (Multi-Query Attention)**:
   - **机制**：无论有多少个 Query 头，所有 Query 头都**共享仅仅 1 个 Key 头和 1 个 Value 头**。
   - **收益**：KV Cache 中与 Key/Value 相关的头数从 `H` 降到 `1`，因此缓存大小近似缩小为 MHA 的 `1/H`，同时降低潜在的 KV 读取量；端到端速度是否提升仍取决于 kernel、backend 和 workload。
   - **代价**：共享 K/V 头会改变模型表达容量，可能影响部分任务质量；影响程度取决于模型结构、训练方式和具体 checkpoint。

2. **GQA (Grouped-Query Attention)**:
   - **机制**：一种折中方案。将 Query 头进行分组（例如 32 个 Query 头分成 8 组），每组内的 Query 头共享 1 对 Key/Value 头。
   - **收益**：KV Cache 中与 Key/Value 相关的头数从 `H` 降到 `G`，因此在其他条件相同时，缓存大小近似缩小为 MHA 的 `G/H`。例如 `H = 32, G = 8` 时，KV Cache 约为 MHA 的 `1/4`，潜在读取量也随之下降。

下面的表格把三种结构放在同一口径下比较。表中的比例是理论容量比例；实际速度、质量和 backend 收益仍需在固定 workload 上验证。

| 机制 | Query 头数 | KV 头数 | KV Cache 理论比例 | 主要变化 | 主要代价 |
| --- | ---: | ---: | ---: | --- | --- |
| MHA | H | H | 1 | 每个 Query 头对应独立 K/V 头 | KV Cache 最大 |
| GQA | H | G | G/H | 多个 Query 头共享一组 K/V 头 | 需要匹配的模型结构或 checkpoint |
| MQA | H | 1 | 1/H | 所有 Query 头共享一组 K/V 头 | 表达能力和任务质量可能受影响 |
</details>
### Q2小验证：头数变化为什么会直接影响缓存

固定上下文长度，只改变 KV 头数，看看缓存怎么缩。

```python
def kv_cache_gb(seq_len, layers, kv_heads, head_dim, dtype_bytes=2, batch_size=1):
    """返回 KV Cache 的十进制 GB 数量，用于比较不同 KV 头数。"""
    return kv_cache_bytes(seq_len, layers, kv_heads, head_dim, dtype_bytes, batch_size) / 1e9

seq_len = 4096
layers = 32
head_dim = 128
mha_gb = kv_cache_gb(seq_len, layers, 32, head_dim)
for name, kv_heads in [('MHA', 32), ('GQA', 8), ('MQA', 1)]:
    current_gb = kv_cache_gb(seq_len, layers, kv_heads, head_dim)
    print(f'{name:>3s}: kv_heads={kv_heads:2d}, KV cache ≈ {current_gb:5.2f} GB, 压缩比例={current_gb / mha_gb:.3f}')
ratios = {name: kv_cache_gb(seq_len, layers, heads, head_dim) / mha_gb for name, heads in [('MHA', 32), ('GQA', 8), ('MQA', 1)]}
assert ratios == {'MHA': 1.0, 'GQA': 0.25, 'MQA': 1 / 32}
print('head-sharing ratios are theoretical cache ratios; speed and quality require a real workload')
```

## Q3：为什么 MLA 能通过 latent 表示压缩 KV Cache？

<details>
<summary>点击展开查看解析</summary>

MLA（Multi-Head Latent Attention）是以 DeepSeek-V2/V3 为代表的一类 latent attention 设计，目标是在保留有效注意力信息的同时压缩 KV Cache 表示。具体缓存格式、RoPE 处理和收益取决于模型实现。

**机制与原理**：
1. **低秩压缩**：模型不直接缓存完整的 K 和 V，而是保存更紧凑的 latent 表示。
2. **动态恢复**：计算 Attention 时，通过投影路径恢复参与点积所需的表示，减少长期驻留的数据量。
3. **RoPE 解耦**：位置信息与内容信息采用不同路径处理，具体缓存项取决于实现。

MLA 是用额外的表示变换和恢复计算换取更小的缓存，不应直接推导出固定的速度或质量收益。这里的比例模型不使用具体 DeepSeek checkpoint 的 latent 维度，结果只能说明“压缩—恢复计算”的关系，不能预测某个真实模型的显存。是否值得，要结合 latent 维度、RoPE 处理、backend 和固定 workload 验证。
</details>
### Q3小验证：压缩比例与额外计算的权衡

改变 latent 表示的压缩比例，比较缓存单位数和重构代理量；结果只表达机制数量级，不等同于真实模型的显存、延迟或质量收益。

```python
def mla_gain(seq_len, kv_heads, compression_ratio=0.5):
    """用比例模型展示 MLA latent 表示压缩带来的数量级变化。

    这不是 MLA 的完整 cache 公式，也不代表实际质量、吞吐或 kernel FLOPs。
    reconstruction_units_proxy 只是额外恢复路径的教学代理。
    """
    if seq_len <= 0 or kv_heads <= 0 or not 0 < compression_ratio <= 1:
        raise ValueError('seq_len、kv_heads 必须为正数，compression_ratio 必须在 (0, 1]')
    base = 2 * seq_len * kv_heads
    compressed = base * compression_ratio
    reconstruction_units = compressed * 2
    return {
        'base_units': base,
        'compressed_units': round(compressed, 2),
        'saving_ratio': round(1 - compression_ratio, 2),
        'reconstruction_units_proxy': round(reconstruction_units, 2),
        'evidence_level': 'teaching_proxy',
    }

for case in [(1024, 32, 0.5), (4096, 32, 0.25), (4096, 16, 0.25)]:
    print(case, '->', mla_gain(*case))
assert mla_gain(4096, 32, 0.25)['saving_ratio'] == 0.75
print('reconstruction_units_proxy 只是额外恢复路径的教学代理，不是 MLA kernel FLOPs')

```

## Q4：PagedAttention 如何用分页管理减少 KV Cache 的空间浪费？

<details>
<summary>点击展开查看解析</summary>

除了修改模型架构，系统层面的缓存管理同样重要。早期推理实现常为每个请求预留连续空间，但生成长度不可预知，长短请求混合时容易造成预留浪费和空间组织困难。

**机制与原理**：
1. **固定大小的 block**：把 KV Cache 按 token 切成固定大小的块。
2. **非连续存储**：物理显存中的 block 不要求连续，通过 block table 找到逻辑 token 对应的物理位置。
3. **按需增长**：请求需要更多 token 时再申请 block，减少按最大长度预留造成的浪费。

**收益**：
PagedAttention 改变的是缓存的分配和寻址方式，不是 KV 表示本身；实际并发收益还取决于 block size、调度、kernel、prefix sharing 和 workspace。

需要把两类系统机制分开：PagedAttention 主要解决物理 block 的分配、寻址和回收；Prefix Cache / RadixAttention 主要解决不同请求之间的共享前缀复用。前者回答“缓存放在哪里”，后者回答“已有前缀能否重复使用”。

| 机制 | 主要解决的问题 | 观察指标 |
| --- | --- | --- |
| PagedAttention | 请求内部的 block 分配、寻址与回收 | 空间利用率、碎片、可容纳请求数 |
| Prefix Cache / RadixAttention | 请求之间的共享前缀复用 | 命中率、复用 token 数、prefill 节省 |
</details>
### Q4小验证：分页后的显存利用率

比较连续预留与按 block 向上取整后的槽位；结果只观察空间利用率，不等同于 vLLM 的真实 block allocator 或服务吞吐。

```python
def contiguous_reserved_tokens(lengths, max_len):
    """模拟每个请求预留 max_len 个连续 token 槽位。"""
    if not lengths or max_len <= 0:
        raise ValueError('lengths 不能为空，max_len 必须为正数')
    if any(length <= 0 or length > max_len for length in lengths):
        raise ValueError('每个请求长度必须在 (0, max_len] 内')
    return len(lengths) * max_len


def paged_reserved_tokens(lengths, block_size):
    """模拟按固定 block_size 向上取整后的分页 token 槽位。"""
    if not lengths or block_size <= 0:
        raise ValueError('lengths 不能为空，block_size 必须为正数')
    if any(length <= 0 for length in lengths):
        raise ValueError('请求长度必须为正数')
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
assert paged <= contiguous
assert 0 < utilization <= 1

```

---

## 相关阅读
本节可以从注意力原理继续接到真实模型实现、FlashAttention、PagedAttention 和推理性能分析。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：回到 MHA 的原始结构和缩放点积注意力。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：理解 IO-aware 分块如何降低 Attention 的显存访问。
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)：了解 KV Cache 分页管理如何进入真实推理服务。
- [22. vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)：观察 block 分配与分页寻址的机制模拟。
- [24. SGLang RadixAttention](../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)：观察 Radix Tree 如何组织共享前缀。
- [34. Prefix Cache Matching and Reuse](../02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.md)：连接共享前缀匹配与复用；分块预填充见 38。
- [37. KV Cache Scheduling](../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.md)：继续学习多请求缓存调度。
- [Part 01 · 11 KV Cache 与显存增长](./11_KV_Cache_and_Memory_Growth.md)
- [Part 01 · 14 FlashAttention 显存模型](./14_FlashAttention_Memory_Model.md)
- [09. Triton PagedAttention | KV Cache 间接寻址](../03_Triton_Kernels/09_Triton_PagedAttention.md)
- [08. Triton Flash Attention | 真正的 Flash Attention 前向算子](../03_Triton_Kernels/08_Triton_Flash_Attention.md)

---