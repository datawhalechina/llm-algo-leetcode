# 16. Attention Mechanism Intro | Attention 机制导论

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `Attention`, `mask` | **目标人群：** 准备进入 Transformer 结构与优化内容的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/16_Attention_Mechanism_Intro.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

Attention 先用 Q/K 形成匹配分数，再经过可见性约束和 Softmax 得到读取权重，最后聚合 V。多头布局改变张量视图，序列长度和位置机制则限定标准 Attention 能表达和承载什么。

完成本节后，你应能追踪 Attention 的最小闭环，组合 causal 与 padding mask，并判断 self/cross Attention 和多头布局的 shape contract。

**关键词：** `Q`, `K`, `V`, `mask`, `shape contract`

![Attention 的表示、可见性与聚合数据流](../public/00_Prerequisites/16_attention_dataflow_map.svg)

## 前置阅读
**导语：** 13 节给出 Attention 分支在 Block 中的位置，15 节说明分支读取 residual stream 前的统计控制；本页集中处理 token 间的信息聚合。
- [13. Transformer Block Dataflow and Residual Stream | Transformer Block 数据流与残差通路](./13_Transformer_Block_Dataflow_and_Residual_Stream.md)
- [15. Normalization Techniques | 归一化技术](./15_Normalization_Techniques.md)
- [0D 组页](./0D.md)

## Q1：Q/K/V 怎样组成完整 Attention 闭环？

Q 表示当前 token 的查询，K 提供匹配索引，V 提供被聚合的内容。实现时要连续追踪 `QKᵀ / √D_h → mask → softmax → weights × V`，而不是把四步拆成彼此无关的公式。


```python
import math
import torch
import torch.nn.functional as F


def build_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def masked_softmax(scores, mask=None, dim=-1):
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    shifted = scores - scores.max(dim=dim, keepdim=True).values
    probs = torch.softmax(shifted, dim=dim)
    if mask is not None:
        mask_f = mask.to(dtype=probs.dtype)
        probs = probs * mask_f
        denom = probs.sum(dim=dim, keepdim=True)
        probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.zeros_like(probs))
    return probs


def attention_weights(q, k, mask=None):
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    return masked_softmax(scores, mask=mask, dim=-1)


def scaled_dot_product_attention(q, k, v, mask=None):
    weights = attention_weights(q, k, mask=mask)
    output = torch.matmul(weights, v)
    return output, weights


q = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
k = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
mask = build_causal_mask(2).unsqueeze(0)
weights = attention_weights(q, k, mask=mask)
output, _ = scaled_dot_product_attention(q, k, v, mask=mask)
print('weights：')
print(weights)
print('output：')
print(output)

```

## Q1 验证：匹配权重和 V 聚合是否同时正确？

因果约束下，第一个 query 只能读取第一个 value；第二个 query 可以读取两个位置。权重与输出都应保持预期 shape，并与 PyTorch 参考实现一致。


```python
q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
mask = build_causal_mask(2).unsqueeze(0)
output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
expected_weights = torch.tensor([[[1.0, 0.0], [0.19557033, 0.80442965]]])

assert weights.shape == (1, 2, 2)
assert output.shape == (1, 2, 2)
assert torch.allclose(weights, expected_weights, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(output, torch.matmul(expected_weights, v), atol=1e-5, rtol=1e-5)
print('✅ Q/K 匹配与 V 聚合闭环通过')

```

## Q2：可见性 mask 与稳定 Softmax 怎样共同工作？

自回归 Attention 用 causal mask 遮蔽未来 token，padding mask 排除补齐的无效 key；两类约束可组合成 `[batch, heads, query_length, key_length]` 的可见性 mask。Softmax 前减去行最大值可避免指数溢出，全遮蔽行则应返回有限的零权重。


```python
def build_attention_mask(key_padding_mask, query_length, causal):
    """构造可广播到多头分数的 [B, 1, S_q, S_k] 可见性 mask。"""
    batch_size, key_length = key_padding_mask.shape
    visible = key_padding_mask[:, None, None, :].expand(batch_size, 1, query_length, key_length)
    if causal:
        if query_length != key_length:
            raise ValueError('causal self-attention requires query_length == key_length')
        visible = visible & build_causal_mask(query_length)[None, None, :, :]
    return visible


key_padding_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
self_mask = build_attention_mask(key_padding_mask, query_length=4, causal=True)
cross_mask = build_attention_mask(key_padding_mask, query_length=2, causal=False)

scores = torch.tensor([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]])
score_mask = torch.tensor([[True, True, False], [False, False, False]])
probs = masked_softmax(scores, mask=score_mask)
print('self / cross mask shape：', self_mask.shape, cross_mask.shape)
print('masked probabilities：', probs)

```

## Q2 验证：mask 语义、广播 shape 和概率归一化是否正确？

因果约束应形成下三角可见性，padding 位置始终不可读；存在可见项的行权重和为 1，全遮蔽行保持有限并返回全零。


```python
expected_causal = torch.tensor([
    [True, False, False, False],
    [True, True, False, False],
    [True, True, True, False],
    [True, True, True, True],
])
assert torch.equal(build_causal_mask(4), expected_causal)
assert self_mask.shape == (2, 1, 4, 4)
assert cross_mask.shape == (2, 1, 2, 4)
assert not self_mask[0, 0, 0, 1]
assert not self_mask[0, 0, 3, 3]
assert cross_mask[0, 0, 0].tolist() == [True, True, True, False]
assert torch.allclose(probs[0].sum(), torch.tensor(1.0))
assert torch.equal(probs[1], torch.zeros(3))
assert torch.isfinite(probs).all()
print('✅ causal/padding mask 与稳定 Softmax 通过')

```

## Q3：自注意力与交叉注意力的 shape contract 有什么不同？

两者都要求 batch、head 和 Q/K 特征维可以完成矩阵乘法，K 与 V 还必须共享来源序列长度。自注意力通常满足 `S_q = S_k = S_v`；交叉注意力允许查询序列长度 `S_q` 与来源序列长度 `S_k` 不同，因此不能把“所有序列长度相同”写成通用契约。


```python
def attention_contract_ok(q_shape, k_shape, v_shape, mode):
    """检查 [B, H, S, D] 形式的自注意力或交叉注意力契约。"""
    if mode not in {'self', 'cross'}:
        return False
    if any(len(shape) != 4 or any(type(size) is not int or size <= 0 for size in shape)
           for shape in (q_shape, k_shape, v_shape)):
        return False
    same_batch = q_shape[0] == k_shape[0] == v_shape[0]
    same_head = q_shape[1] == k_shape[1] == v_shape[1]
    score_feature_match = q_shape[3] == k_shape[3]
    source_length_match = k_shape[2] == v_shape[2]
    self_length_match = q_shape[2] == k_shape[2] if mode == 'self' else True
    return same_batch and same_head and score_feature_match and source_length_match and self_length_match


print('self contract：', attention_contract_ok((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='self'))
print('cross contract：', attention_contract_ok((2, 4, 3, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='cross'))
```

## Q3 验证：Self / cross attention 的合法与非法 shape 能否区分？

Self-attention 要求查询与来源序列长度一致；cross-attention 允许二者不同，但 K/V 的来源长度必须一致。


```python
assert attention_contract_ok((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='self')
assert attention_contract_ok((2, 4, 3, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='cross')
assert not attention_contract_ok((2, 4, 3, 16), (2, 4, 8, 16), (2, 4, 7, 32), mode='cross')
assert not attention_contract_ok((2, 4, 8), (2, 4, 8, 16), (2, 4, 8, 32), mode='self')
assert not attention_contract_ok((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 32), mode='unknown')
print('✅ self / cross attention shape contract 通过')

```

## Q4：多头 Attention 怎样在 head 维和 hidden 维之间转换？

多头 Attention 把 `[B, S, D]` 拆成 `[B, H, S, D_h]`，让不同 head 独立计算，再把结果合回 hidden 维。拆分与合并必须保持元素顺序和总维度不变；这一布局契约也是后续阅读 GQA、FlashAttention 和 fused kernel 的基础。


```python
def split_heads(x, num_heads):
    """把 [B, S, D] 转成 [B, H, S, D_h]。"""
    batch_size, seq_len, hidden_size = x.shape
    if hidden_size % num_heads != 0:
        raise ValueError('hidden_size must be divisible by num_heads')
    head_dim = hidden_size // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x):
    """把 [B, H, S, D_h] 合回 [B, S, D]。"""
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
heads = split_heads(x, num_heads=2)
restored = merge_heads(heads)
print('head shape:', tuple(heads.shape))
print('round trip:', torch.equal(restored, x))

```

## Q4 验证：手写多头数据流是否与 PyTorch 参考实现一致？

除布局往返外，还应比较每个 head 的手写 Attention 与 `scaled_dot_product_attention`，确认分数缩放、softmax 和 V 聚合没有发生语义偏差。


```python
x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 10
q = split_heads(x, num_heads=2)
k = split_heads(x + 0.1, num_heads=2)
v = split_heads(x - 0.2, num_heads=2)
manual_output, _ = scaled_dot_product_attention(q, k, v)
reference_output = F.scaled_dot_product_attention(q, k, v)

assert q.shape == (1, 2, 3, 4)
assert torch.equal(merge_heads(split_heads(x, 2)), x)
assert torch.allclose(manual_output, reference_output, atol=1e-6, rtol=1e-6)
print('✅ 多头布局与 PyTorch SDPA 对照通过')

```

## Q5：标准 Attention 还没有解决哪两个问题？

每个 query 都与全部 key 形成分数，因此 score 矩阵随序列长度按平方增长。另一方面，若不加入位置编码或位置偏置，自注意力只根据内容匹配；同步打乱 token 顺序只会同步打乱输出，模型无法仅凭 Attention 区分原始次序。


```python
def attention_score_elements(batch_size, num_heads, seq_len):
    """返回标准自注意力分数张量 [B, H, S, S] 的元素数。"""
    return batch_size * num_heads * seq_len * seq_len


short_scores = attention_score_elements(batch_size=2, num_heads=4, seq_len=8)
long_scores = attention_score_elements(batch_size=2, num_heads=4, seq_len=16)

tokens = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
permutation = torch.tensor([2, 0, 1])
output, _ = scaled_dot_product_attention(tokens, tokens, tokens)
permuted = tokens[:, permutation]
permuted_output, _ = scaled_dot_product_attention(permuted, permuted, permuted)
print('score elements：', short_scores, '->', long_scores)

```

## Q5 验证：长度翻倍和顺序置换是否符合机制预期？

序列长度翻倍时分数元素数应变为四倍；没有位置机制时，置换输入后输出只发生同样的置换。


```python
assert long_scores == 4 * short_scores
assert torch.allclose(permuted_output, output[:, permutation], atol=1e-6)
print('✅ Attention 二次增长与位置缺失特性通过')

```

## 相关阅读

掌握最小 Attention 语义后，再分别进入位置、注意力表示、显存与推理状态问题。

- [Part 01 · 04 Attention Memory Optimization | Attention 显存优化](../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)
- [Part 02 · 03 RoPE 旋转位置编码](../02_PyTorch_Algorithms/03_RoPE_Tutorial.md)
- [Part 02 · 04 MHA / GQA](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
- [Part 02 · 22 vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [大模型架构演进 · 注意力表示与 KV Cache](../topic_discussion/llm_architecture_evolution/04_attention_evolution.md)
