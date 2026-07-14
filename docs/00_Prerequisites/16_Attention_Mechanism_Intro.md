# 16. Attention Mechanism Intro | Attention 机制导论

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `注意力`, `mask` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/16_Attention_Mechanism_Intro.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：理解 Q / K / V 的维度职责；理解 causal mask 的作用；能写出最小的 scaled dot-product attention。前面已经看过训练、激活和归一化，现在来到把 token 关系串起来的核心模块。对不熟 NLP / Transformer / LLM 的学习者来说，可以先把它理解成“序列模型里让 token 彼此建立联系的核心模块”：Embedding 先把 token id 变成向量，Attention 再决定每个 token 应该关注哪些位置。

这一页先看最小闭环：语义、mask、score、value 聚合。

如果你把前面的 block 看成“Norm + Attention + MLP + Residual”的组合，这一页就是第一次把其中最核心的 token 交互部分单独拆出来。

**关键词：** `Q`, `K`, `V`

## 前置阅读
**导语：** 先看 0D 组页，把归一化和训练稳定性的边界对齐，再进入这一页会更顺。
- [15. Normalization Techniques | 归一化技术](./15_Normalization_Techniques.md)
- [0D 组页](./0D.md)
- [04. Attention Memory Optimization | Attention 显存优化](../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)

## 相关阅读
**导语：** 本页先把 Q / K / V 和 causal mask 的最小判断讲清楚；如果想继续看注意力实现和显存模型，再顺着看下面这一页。
- [17. PyTorch Profiling Basics | PyTorch 性能分析基础](./17_PyTorch_Profiling_Basics.md)
- [14. FlashAttention Memory Model | FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)

## Q1：Q / K / V 分别承担什么职责？

Attention 的核心不是公式，而是三组向量的分工：Q 提出查询，K 提供可匹配的信息，V 提供被聚合的内容。先把这三个职责分开，后面看代码才不会混。对于没有大模型基础的人，可以先把它理解成“token 之间如何互相问答”：Q 负责问，K 负责被比对，V 负责被取回。


```python
import math
import torch


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

## Q1验证：最小 attention 输出是否可见？

这里直接把权重和输出打印出来，确认 Q/K/V 的最小流程已经跑通。


```python
q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
mask = build_causal_mask(2).unsqueeze(0)
weights = attention_weights(q, k, mask=mask)
expected = torch.tensor([[[1.0, 0.0], [0.19557033, 0.80442965]]])
assert torch.allclose(weights, expected, atol=1e-5, rtol=1e-5)
print('✅ attention_weights 通过')

```

## Q2：什么时候必须先看 causal mask 的语义？

只要是自回归场景，就必须遮住未来 token。mask 不是附加项，而是保证语义正确的必要条件。这里可以先把它理解成“只允许看见历史，不允许偷看未来”。


```python
mask = build_causal_mask(4)
print('causal mask：')
print(mask.int())
print('最后一行：', mask[-1].int().tolist())
print('第一行：', mask[0].int().tolist())

```

## Q2验证：causal mask 是否真的是下三角？

这里直接确认上三角被遮住，未来位置不会参与当前 token 的聚合。


```python
mask = build_causal_mask(4)
expected = torch.tensor([
    [True, False, False, False],
    [True, True, False, False],
    [True, True, True, False],
    [True, True, True, True],
])
assert torch.equal(mask, expected)
print('✅ causal mask 通过')

```

## Q3：什么时候必须先看 softmax 的数值稳定性？

Attention 里 score 常常会先减去最大值再做 softmax，目的不是“多写一步”，而是避免数值爆掉。对于序列很长的模型，这一步会直接影响训练是否稳定。


```python
scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mask = torch.tensor([[True, True, False], [True, True, True]])
probs = masked_softmax(scores, mask=mask)
print('masked_softmax：')
print(probs)
print('row sum：', probs.sum(dim=-1))

```

## Q3验证：softmax 和 mask 的稳定性是否正确？

这里直接确认：被 mask 的位置会变成 0，未被 mask 的位置行和为 1。


```python
scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mask = torch.tensor([[True, True, False], [True, True, True]])
probs = masked_softmax(scores, mask=mask)
assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
assert torch.allclose(probs[0, 2], torch.tensor(0.0))
print('✅ masked_softmax 通过')

```

## Q4：什么时候必须把 QK^T、mask 和 V 串成一个最小闭环？

只要你想读懂 Attention 的实现，就必须先把 score、mask、softmax 和 value 聚合串起来看，而不是分开背公式。这里的最小闭环，其实就是 Transformer block 里最核心的那一小段。


```python
q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
mask = build_causal_mask(2).unsqueeze(0)
output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
print('weights:')
print(weights)
print('output:')
print(output)

```

## Q4验证：Attention 的最小闭环是否可见？

这里直接检查权重和输出 shape，确认最小 attention 流程已经完整闭合。


```python
q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
mask = build_causal_mask(2).unsqueeze(0)
output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
assert output.shape == (1, 2, 2)
assert weights.shape == (1, 2, 2)
print('✅ Attention 闭环通过')

```

## Q5：Q / K / V 的 shape contract 怎么判断？

先看 batch、head、seq、hidden 四个维度是否拆对，再谈 attention 是否能对齐。对没有架构基础的人，可以先把它记成“query、key、value 必须在同一个批次和头数下对齐”。


```python
def attention_contract_ok(q_shape, k_shape, v_shape):
    same_batch = q_shape[0] == k_shape[0] == v_shape[0]
    same_head = q_shape[1] == k_shape[1] == v_shape[1]
    seq_match = q_shape[2] == k_shape[2] and v_shape[2] >= q_shape[2]
    return same_batch and same_head and seq_match


print('ok:', attention_contract_ok((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16)))
print('bad:', attention_contract_ok((2, 4, 8, 16), (2, 4, 7, 16), (2, 4, 8, 16)))
# 输出示例: ok -> True, bad -> False

```

## Q6：mask 放在 softmax 前还是后，为什么？

mask 必须先作用在 logits 上，再做 softmax；如果先 softmax，再 mask，概率归一化就会被破坏。这里可以先记成一句最小规则：先遮住不该看的位置，再把剩下的位置归一化。


```python
def mask_before_softmax(mask_first):
    if mask_first:
        return 'stable_logits_then_softmax'
    return 'probability_broken'


print('case1:', mask_before_softmax(True))
print('case2:', mask_before_softmax(False))
# 输出示例: stable_logits_then_softmax / probability_broken

```

## Q7：attention 里最常见的错误是逻辑错还是布局错？

先分清是 mask / softmax / 聚合逻辑错，还是 head / seq / hidden 的布局错；这两类错的修法完全不同。对于刚接触 Transformer 的人，逻辑错更像“算错了该看谁”，布局错更像“维度排错了”。


```python
def classify_attention_bug(logic_ok, layout_ok):
    if not logic_ok and not layout_ok:
        return {'bug': 'both', 'next': 'fix_logic_then_layout'}
    if not logic_ok:
        return {'bug': 'logic', 'next': 'check_mask_softmax_value_path'}
    if not layout_ok:
        return {'bug': 'layout', 'next': 'check_shape_contract_and_stride'}
    return {'bug': 'ok', 'next': 'profile_and_compare'}


print(classify_attention_bug(False, True))
print(classify_attention_bug(True, False))
# 输出示例: logic / layout / both

```

## Q8：什么时候该停在最小 attention，什么时候该继续看优化实现？

当最小 attention 已经能解释语义，但性能或显存成了主问题时，再去看 fused softmax、FlashAttention 或更底层实现。这里的分界很简单：先保证你看得懂语义，再考虑让它更快更省。


```python
def next_attention_step(correctness_ok, perf_ok, memory_ok):
    if not correctness_ok:
        return 'fix_minimal_attention_first'
    if not perf_ok or not memory_ok:
        return 'study_fused_or_flashattention'
    return 'stay_with_minimal_model'


print(next_attention_step(True, False, True))
print(next_attention_step(True, True, True))
# 输出示例: study_fused_or_flashattention / stay_with_minimal_model

```
