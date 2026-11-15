# 11. KV Cache and Memory Growth | KV Cache 与显存增长

**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `KV Cache`, `显存增长` | **目标人群：** 需要理解长上下文推理显存成本的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

注意力机制会为历史 Token 保存键（K）和值（V）；上下文越长、并发请求越多，键值缓存（KV Cache）占用越大。容量增长会直接限制可服务的请求数。
本节建立“层数与 KV 头数 → 序列长度 → 并发 → 缓存容量”的账本。完成后，你应能估算单请求与并发场景的 KV 状态量，并判断问题首先来自表示、物理组织还是复用策略。


**关键词：** `KV Cache`、序列长度、容量增长

![本节概念关系](../public/01_Hardware_Math_and_Systems/11_kv_cache_growth_map.svg)


---
## 前置阅读

**导语：** 先掌握数据类型、Attention 状态和显存账本，再计算序列长度与并发增加后 KV Cache 如何变化。
- [Part 01 · 01 数据类型与精度](./01_Data_Types_and_Precision.md)
- [Part 01 · 04 Attention 与显存优化](./04_Attention_Memory_Optimization.md)
---
## Q1：为什么 KV cache 会随着上下文长度增长？


KV cache 本质上是在解码阶段保存历史 token 的 Key 和 Value。每生成一个新 token，模型都要把这一 token 在每一层、每一个 KV 头上的 K/V 追加进缓存里，所以它的增长不是“偶尔增加一点”，而是随着上下文长度持续线性增长。

更具体地说，标准自回归推理里，缓存大小通常可以写成：

$$
\mathrm{KVBytes} = 2 \times L \times B \times H_{kv} \times D \times S \times b
$$

其中：

- $L$ 是层数
- $B$ 是 batch size
- $H_{kv}$ 是 KV 头数
- $D$ 是 head dim
- $S$ 是上下文长度
- $b$ 是每元素字节数，对应代码中的 `dtype_bytes`；FP16／BF16为2字节，FP32为4字节
- 前面的 $2$ 表示同时存 K 和 V

公式计算各层使用相同维度、K和V使用相同类型时，缓存张量本身的理论字节数。实际分配还需计入元数据、布局填充和内存管理开销。

这个公式告诉我们一个很关键的事实：**KV cache 是“历史状态成本”**。上下文越长，历史 token 越多，缓存就越大；batch 越大，缓存也会同步放大；层数越多，这个成本还会在每一层重复一次。

因此，长上下文和高并发会持续抬高 KV Cache 容量需求；它是否先于计算、带宽或其他运行时状态成为限制，还要结合模型结构、并发和后端实测。这里先把“每个历史 token 都要保留 K/V 状态”的增长规律算清楚。

### Q1小验证：KV cache 增长直觉

先固定2层、1个序列、3个KV头、头维4和5个历史位置：K、V共含 $2\times2\times1\times3\times4\times5=240$ 个元素。每元素2字节时为480 B，4字节时为960 B；再把上下文长度翻倍，核对缓存字节数随之翻倍。

```python
def kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim, batch_size=1, dtype_bytes=2):
    """估算 K 和 V 的理论存储量；不包含 allocator、workspace 和碎片。"""
    values = (seq_len, num_layers, num_kv_heads, head_dim, batch_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、层数、头数、维度、batch 和 dtype 字节数必须为正数')
    return 2 * seq_len * num_layers * num_kv_heads * head_dim * batch_size * dtype_bytes

# 固定小例子同时核对元素数量与每元素字节数。
assert kv_cache_bytes(5, 2, 3, 4, dtype_bytes=2) == 480
assert kv_cache_bytes(5, 2, 3, 4, dtype_bytes=4) == 960
assert kv_cache_bytes(10, 2, 3, 4, dtype_bytes=2) == 960

examples = [(1024, 32, 32, 128), (2048, 32, 32, 128), (4096, 32, 32, 128)]
for seq_len, layers, kv_heads, head_dim in examples:
    size_gb = kv_cache_bytes(seq_len, layers, kv_heads, head_dim) / 1e9
    print(f"seq_len={seq_len:4d} -> KV cache ≈ {size_gb:5.2f} GB")
```

### 数量级速览

固定缓存表示后，先比较长度、并发、层数与KV头数各自增加时的容量变化；分页分配造成的空槽在Q3单独计算。

| 变化项 | 对 KV cache 的影响 | 直觉 |
| --- | --- | --- |
| `seq_len` 翻倍 | 翻倍 | 历史 token 变多 |
| `batch_size` 翻倍 | 翻倍 | 不同序列各自保存历史状态 |
| `num_layers` 翻倍 | 翻倍 | 每层单独保存 K/V |
| `num_kv_heads` 翻倍 | 翻倍 | 标准 K/V 表示中需要保存的头数增加 |

## Q2：为什么 MHA、MQA 和 GQA 会影响 KV cache 压力？


它们影响的是 **KV cache 的头数维度**。

- **MHA (Multi-Head Attention)**：每个 query head 都有自己对应的一组 K/V，缓存压力最大。
- **MQA (Multi-Query Attention)**：多个 query head 共享同一组 K/V，KV cache 立刻变小。
- **GQA (Grouped-Query Attention)**：介于 MHA 和 MQA 之间，把 query heads 分组共享 K/V，在显存和表达能力之间做折中。

从缓存角度看，真正决定显存大小的不是 query heads 有多少，而是 **要存多少组 K/V**。所以只要 KV 头数下降，缓存就会按比例下降。

这也是为什么很多长上下文模型会采用 MQA 或 GQA：它们不只是“改了 attention 的形式”，而是在直接压低推理时的 KV cache 成本。

### Q2小验证：头数与显存直觉

固定上下文长度和层数，只改变 KV 头数，观察显存变化。

```python
def kv_cache_gb(seq_len, num_layers, num_kv_heads, head_dim, batch_size=1, dtype_bytes=2):
    """把 KV Cache 理论字节数换算为十进制 GB。"""
    return kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim, batch_size, dtype_bytes) / 1e9

seq_len = 4096
num_layers = 32
head_dim = 128
for name, kv_heads in [("MHA", 32), ("GQA", 8), ("MQA", 1)]:
    print(f"{name:>3s}: kv_heads={kv_heads:2d}, KV cache ≈ {kv_cache_gb(seq_len, num_layers, kv_heads, head_dim):5.2f} GB")
```

## Q3：KV Cache 变大之后，应该优化缓存的组织还是表示？


先判断压力来自哪里：单 token 的缓存账本过大，优先考虑改变缓存表示；缓存容量可以接受，但多请求下预留和碎片造成浪费，则优先考虑改变缓存组织。表中的表示规模和分页数量仍是机制模型；要判断真实 backend 的显存、延迟和吞吐收益，需要固定 workload 测量。下面用一张决策表把三类问题分开。

| 观察到的问题 | 优先改变的对象 | 代表机制 | 主要观察量 |
| --- | --- | --- | --- |
| 单 token KV 成本过大 | 缓存表示 | MQA / GQA / MLA | 每 token、每层的缓存字节数 |
| 多请求预留或碎片明显 | 缓存组织 | PagedAttention | 页数、尾页浪费、复用情况 |
| 前缀重复导致重复计算 | 缓存复用 | Prefix Cache（延伸） | 命中率、复用 token 数 |


### Q3小验证：先算每个位置的字节数，再按页分配

用一组缩小的尺寸计算：2层、1个序列、2个KV头、头维4，每元素2字节。MHA 每个位置每层保存 $2\times2\times4=16$ 个元素；采用共享潜在向量4维、位置键2维的表示时，改为保存6个元素。这里的位置键是 RoPE 分支保存的位置表示，与潜在向量一起计入缓存。

5个历史位置的 MHA 状态为320 B，压缩表示为120 B；若每页4个位置，需要2页、共8个槽位，压缩表示的页容量为192 B。其中3个槽位尚未使用。将长度改为8后，页容量仍是192 B，尾页空闲变为0。前缀长度单独记录；判断共享是否节省物理页，还需给出其他请求及它们的共享关系。


```python
def paged_attention_pages(seq_len, page_size):
    """计算分页所需页数；只观察分配粒度，不模拟真实 allocator。"""
    if seq_len <= 0 or page_size <= 0:
        raise ValueError('seq_len 和 page_size 必须为正数')
    return (seq_len + page_size - 1) // page_size

def paged_cache_waste_tokens(seq_len, page_size):
    """计算最后一页未被使用的 token 槽位。"""
    pages = paged_attention_pages(seq_len, page_size)
    return pages * page_size - seq_len

def mla_cache_bytes(seq_len, num_layers, latent_dim, batch_size=1, dtype_bytes=2, rope_dim=0):
    """计算共享潜在向量及位置键的缓存字节数；rope_dim=0 表示仅计内容分支。"""
    values = (seq_len, num_layers, latent_dim, batch_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、层数、latent_dim、batch 和 dtype 字节数必须为正数')
    if type(rope_dim) is not int or rope_dim < 0:
        raise ValueError('rope_dim 须为非负整数')
    return seq_len * num_layers * (latent_dim + rope_dim) * batch_size * dtype_bytes

def compare_cache_strategies(seq_len, page_size, base_kv_heads, head_dim, latent_dim, prefix_len=0, num_layers=32, batch_size=1, dtype_bytes=2, rope_dim=0):
    """比较逻辑状态与按页分配的字节数；各序列等长，前缀仅计可复用长度。

    压缩缓存保存共享潜在向量和位置键；页容量含尾页空槽，
    元数据、工作区和跨请求共享节省另行计算。
    """
    if prefix_len < 0 or prefix_len > seq_len:
        raise ValueError('prefix_len 必须位于 [0, seq_len]')
    pages = paged_attention_pages(seq_len, page_size)
    base_bytes = kv_cache_bytes(seq_len, num_layers, base_kv_heads, head_dim, batch_size, dtype_bytes)
    compact_bytes = mla_cache_bytes(seq_len, num_layers, latent_dim, batch_size, dtype_bytes, rope_dim)
    paged_bytes = mla_cache_bytes(pages * page_size, num_layers, latent_dim, batch_size, dtype_bytes, rope_dim)
    return {
        'management': {
            'pages': pages,
            'waste_tokens': paged_cache_waste_tokens(seq_len, page_size),
        },
        'representation': {
            'base_kv_bytes': base_bytes,
            'compact_kv_bytes': compact_bytes,
            'paged_compact_kv_bytes': paged_bytes,
            'representation_saving_ratio': 1 - compact_bytes / base_bytes,
        },
        'reuse': {'prefix_reuse_tokens': prefix_len},
    }


case = dict(page_size=4, base_kv_heads=2, head_dim=4, latent_dim=4,
            rope_dim=2, num_layers=2, batch_size=1, dtype_bytes=2)
result = compare_cache_strategies(seq_len=5, prefix_len=2, **case)
assert result['management'] == {'pages': 2, 'waste_tokens': 3}
assert result['representation'] == {
    'base_kv_bytes': 320, 'compact_kv_bytes': 120,
    'paged_compact_kv_bytes': 192, 'representation_saving_ratio': 0.625,
}
assert result['reuse']['prefix_reuse_tokens'] == 2
filled = compare_cache_strategies(seq_len=8, **case)
assert filled['management']['waste_tokens'] == 0
assert filled['representation']['compact_kv_bytes'] == filled['representation']['paged_compact_kv_bytes'] == 192
print('5个位置 ->', result)
print('8个位置 ->', filled)

```

---
## 相关阅读

本节可以从 KV Cache 公式回到 Attention 的头数设计，再继续进入分页管理、前缀复用和请求调度。

- [22. vLLM 与 PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [24. Prefix Cache 匹配与复用](../02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse.md)
- [34. SGLang 与 RadixAttention](../02_PyTorch_Algorithms/34_SGLang_RadixAttention.md)
---
