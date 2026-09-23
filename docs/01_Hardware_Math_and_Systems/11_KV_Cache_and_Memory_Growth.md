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

本节用缓存公式解释长上下文和并发带来的显存增长，再区分缓存表示规模与缓存组织方式。先计算单条请求的 KV Cache 状态账本，再把结果用于分析分页、复用和调度中的容量约束。

在理解 Attention 如何产生和保存 Key、Value 后，本节把这些状态放进具体 workload，重点判断 KV Cache 会增长到多大、显存预算是否足够，以及下一步应改变表示还是改变组织方式。


**关键词：** `KV cache`, `sequence length`, `memory growth`

![本节概念关系](../public/01_Hardware_Math_and_Systems/11_kv_cache_growth_map.svg)


---
## 前置阅读

**导语：** 先掌握数据类型、Attention 状态和显存账本，再计算序列长度与并发增加后 KV Cache 如何变化。
- [Part 01 · 01 数据类型与精度](./01_Data_Types_and_Precision.md)
- [Part 01 · 04 Attention 与显存优化](./04_Attention_Memory_Optimization.md)
- [Part 01 · 06 显存计算与 ZeRO](./06_VRAM_Calculation_and_ZeRO.md)
---
## Q1：为什么 KV cache 会随着上下文长度增长？

<details>
<summary>点击展开查看解析</summary>

KV cache 本质上是在解码阶段保存历史 token 的 Key 和 Value。每生成一个新 token，模型都要把这一 token 在每一层、每一个 KV 头上的 K/V 追加进缓存里，所以它的增长不是“偶尔增加一点”，而是随着上下文长度持续线性增长。

更具体地说，标准自回归推理里，缓存大小通常可以写成：

$$
\text{KV Cache Bytes} \approx 2 \times L \times B \times H_{kv} \times D \times S
$$

其中：
- $L$ 是层数
- $B$ 是 batch size
- $H_{kv}$ 是 KV 头数
- $D$ 是 head dim
- $S$ 是上下文长度
- 前面的 $2$ 表示同时存 K 和 V

这个公式告诉我们一个很关键的事实：**KV cache 是“历史状态成本”**。上下文越长，历史 token 越多，缓存就越大；batch 越大，缓存也会同步放大；层数越多，这个成本还会在每一层重复一次。

所以长上下文推理里，最先爆的往往不是算力，而是显存。只要把“每个 token 都要被长期保留”这件事想清楚，KV cache 的增长规律就很自然了。
</details>
### Q1小验证：KV cache 增长直觉

把上下文长度翻倍，再看缓存大小是不是也几乎翻倍。

```python
def kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim, batch_size=1, dtype_bytes=2):
    """估算 K 和 V 的理论存储量；不包含 allocator、workspace 和碎片。"""
    values = (seq_len, num_layers, num_kv_heads, head_dim, batch_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、层数、头数、维度、batch 和 dtype 字节数必须为正数')
    return 2 * seq_len * num_layers * num_kv_heads * head_dim * batch_size * dtype_bytes

examples = [(1024, 32, 32, 128), (2048, 32, 32, 128), (4096, 32, 32, 128)]
for seq_len, layers, kv_heads, head_dim in examples:
    size_gb = kv_cache_bytes(seq_len, layers, kv_heads, head_dim) / 1e9
    print(f"seq_len={seq_len:4d} -> KV cache ≈ {size_gb:5.2f} GB")
```

### 数量级速览

| 变化项 | 对 KV cache 的影响 | 直觉 |
| --- | --- | --- |
| `seq_len` 翻倍 | 近似翻倍 | 历史 token 变多，缓存线性增长 |
| `batch_size` 翻倍 | 近似翻倍 | 多路请求共享同一层结构，但缓存要按样本复制 |
| `num_layers` 翻倍 | 近似翻倍 | 每层都要单独保存一份 K/V |
| `num_kv_heads` 增加 | 近似线性增加 | KV 头越多，缓存越大 |

这一张表的目的，是先把“增长方向”记牢，再去看后面的 MHA / MQA / GQA 和分页管理。
## Q2：为什么 MHA、MQA 和 GQA 会影响 KV cache 压力？

<details>
<summary>点击展开查看解析</summary>

它们影响的是 **KV cache 的头数维度**。

- **MHA (Multi-Head Attention)**：每个 query head 都有自己对应的一组 K/V，缓存压力最大。
- **MQA (Multi-Query Attention)**：多个 query head 共享同一组 K/V，KV cache 立刻变小。
- **GQA (Grouped-Query Attention)**：介于 MHA 和 MQA 之间，把 query heads 分组共享 K/V，在显存和表达能力之间做折中。

从缓存角度看，真正决定显存大小的不是 query heads 有多少，而是 **要存多少组 K/V**。所以只要 KV 头数下降，缓存就会按比例下降。

这也是为什么很多长上下文模型会采用 MQA 或 GQA：它们不只是“改了 attention 的形式”，而是在直接压低推理时的 KV cache 成本。
</details>
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

<details>
<summary>点击展开查看解析</summary>

先判断压力来自哪里：单 token 的缓存账本过大，优先考虑改变缓存表示；缓存容量可以接受，但多请求下预留和碎片造成浪费，则优先考虑改变缓存组织。表中的表示规模和分页数量仍是机制模型；要判断真实 backend 的显存、延迟和吞吐收益，需要固定 workload 测量。下面用一张决策表把三类问题分开。

| 观察到的问题 | 优先改变的对象 | 代表机制 | 主要观察量 |
| --- | --- | --- | --- |
| 单 token KV 成本过大 | 缓存表示 | MQA / GQA / MLA | 每 token、每层的缓存字节数 |
| 多请求预留或碎片明显 | 缓存组织 | PagedAttention | 页数、尾页浪费、复用情况 |
| 前缀重复导致重复计算 | 缓存复用 | Prefix Cache（延伸） | 命中率、复用 token 数 |

</details>
### Q3小验证：缓存管理与表示压缩的对照

把“缓存管理”和“表示压缩”分开看，再判断它们各自对显存的影响。

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

def mla_cache_bytes(seq_len, num_layers, latent_dim, batch_size=1, dtype_bytes=2):
    """估算 latent 表示的理论存储量；不是完整 MLA kernel 账本。"""
    values = (seq_len, num_layers, latent_dim, batch_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、层数、latent_dim、batch 和 dtype 字节数必须为正数')
    return seq_len * num_layers * latent_dim * batch_size * dtype_bytes

def compare_cache_strategies(seq_len, page_size, base_kv_heads, head_dim, latent_dim, prefix_len=0, num_layers=32, batch_size=1, dtype_bytes=2):
    """并列展示缓存组织与表示压缩的可观察量。

    PagedAttention 关注页数和尾页浪费；MLA proxy 关注每 token 的表示大小；
    prefix_len 只记录可复用的前缀 token 数。三者不是同一个指标，也不等于真实 backend 的端到端收益。
    """
    if prefix_len < 0 or prefix_len > seq_len:
        raise ValueError('prefix_len 必须位于 [0, seq_len]')
    base_bytes = kv_cache_bytes(seq_len, num_layers, base_kv_heads, head_dim, batch_size, dtype_bytes)
    compressed_bytes = mla_cache_bytes(seq_len, num_layers, latent_dim, batch_size, dtype_bytes)
    return {
        'management': {
            'pages': paged_attention_pages(seq_len, page_size),
            'waste_tokens': paged_cache_waste_tokens(seq_len, page_size),
        },
        'representation': {
            'base_kv_bytes': base_bytes,
            'latent_bytes_proxy': compressed_bytes,
            'toy_representation_saving_ratio': round(1 - compressed_bytes / base_bytes, 4),
        },
        'reuse': {
            'prefix_reuse_tokens': prefix_len,
        },
    }

result = compare_cache_strategies(4096, 128, 32, 128, 64, prefix_len=1024)
print('management ->', result['management'])
print('representation ->', result['representation'])
assert result['management']['pages'] == 32
assert result['management']['waste_tokens'] == 0
assert result['reuse']['prefix_reuse_tokens'] == 1024
assert 0 < result['representation']['toy_representation_saving_ratio'] < 1
try:
    paged_attention_pages(4096, 0)
except ValueError:
    print('✅ 分页参数校验通过')
else:
    raise AssertionError('page_size 为 0 时应报错')
```

---
## 相关阅读

本节可以从 KV Cache 公式回到 Attention 的头数设计，再继续进入分页管理、前缀复用和请求调度。

- [04. Attention（MHA / GQA）](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
- [KV Cache 与 PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [22. vLLM 与 PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [24. SGLang 与 RadixAttention](../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- [34. Prefix Cache 匹配与复用](../02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.md)
---