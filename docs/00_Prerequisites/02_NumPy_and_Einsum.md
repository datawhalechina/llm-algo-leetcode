# 02. NumPy and Einsum | NumPy 与 Einsum

**难度：** Easy | **环境：** CPU-first | **标签：** `NumPy`, `广播`, `einsum` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/02_NumPy_and_Einsum.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：先看 shape，再看公式；能把简单循环改写成数组运算；能用 `einsum` 表达常见的维度关系。这里先把 NumPy 当成“维度思维的练习场”：学会用数组、索引、广播和 `einsum` 讲清楚 shape 关系，后面进 Tensor、mask 和 attention 时会顺很多。

**关键词：** `ndarray`, `shape`, `einsum`

## 前置阅读
**导语：** 先看 0A 组页，把 Part 0 里 Python 基础和数组思维的边界对齐，再进入这一页会更顺。
- [01. Python Essentials for LLM | Python 基础与数据表示](./01_Python_Essentials_for_LLM.md)
- [0A 组页](./0A.md)
- [P1: 01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读
**导语：** 本页先把 ndarray、广播和 einsum 的最小判断讲清楚；如果想继续把对象封装和配置 / I/O 的写法补完整，可以顺着看下面这些页。
- [03. Python OOP and Utility Patterns | Python 面向对象与工具模式](./03_Python_OOP_and_Utility_Patterns.md)

## Q1：ndarray、索引和广播分别解决什么问题？

这一页先把三个最常见的 NumPy 判断立住：数据是不是同质的、能不能按维度切片、能不能用广播省掉手写循环。后面你看到 `mask`、`shape`、`batch` 时，先回到这三个判断。对 Part 2 来说，这一题就是在提前练“维度和位置怎么对上”；等你进入 0B 里的 Tensor、layout 和 mask 时，这个判断会直接复用。


```python
import numpy as np


def build_causal_mask(seq_len):
    """返回 causal mask，shape = [seq_len, seq_len]。"""
    upper = np.triu(np.ones((seq_len, seq_len), dtype=np.int32), k=1)
    return upper == 0


def matmul_einsum(a, b):
    """使用 einsum 实现二维矩阵乘法。"""
    return np.einsum('ij,jk->ik', a, b)


def batch_attention_scores(q, k):
    """计算 batch attention scores，shape = [b, q, k]。"""
    return np.einsum('bqd,bkd->bqk', q, k)


def rms_normalize(x, eps=1e-6):
    """沿最后一维做 RMSNorm 风格归一化。"""
    denom = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return x / denom

```

## Q1验证：因果掩码和矩阵乘法

这里先验证两个最常见的场景：一个是 mask 形状是否正确，一个是 `einsum` 能不能替代普通矩阵乘法。这个验证的目的，不是背公式，而是先确认“维度对了，结果才可能对”。


```python
def test_build_causal_mask():
    mask = build_causal_mask(4)
    expected = np.array([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert np.array_equal(mask, expected)
    print('✅ build_causal_mask 通过')


def test_matmul_einsum():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = matmul_einsum(a, b)
    expected = np.array([[19, 22], [43, 50]])
    assert np.array_equal(result, expected)
    print('✅ matmul_einsum 通过')


test_build_causal_mask()
test_matmul_einsum()

```

## Q2：什么时候该用 `einsum` 直接表达维度关系？

当你已经能看清 `B / T / D / H` 这些维度时，`einsum` 的作用不是炫技，而是把维度关系直接写在式子里，避免先写一堆 reshape 再猜结果对不对。对 Part 2 来说，它更像是“把 shape contract 直接写出来”的工具。


```python
def test_batch_attention_scores():
    q = np.array([[[1, 0], [0, 1]]])
    k = np.array([[[1, 2], [3, 4]]])
    result = batch_attention_scores(q, k)
    expected = np.array([[[1, 3], [2, 4]]])
    assert np.array_equal(result, expected)
    print('✅ batch_attention_scores 通过')


def test_rms_normalize():
    x = np.array([[3.0, 4.0]])
    y = rms_normalize(x, eps=0.0)
    expected = np.array([[0.84852814, 1.13137085]])
    assert np.allclose(y, expected)
    print('✅ rms_normalize 通过')


test_batch_attention_scores()
test_rms_normalize()

q = np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
k = np.array([[[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]]])
print('Attention scores shape:', batch_attention_scores(q, k).shape)
print(batch_attention_scores(q, k))
print('Causal mask:')
print(build_causal_mask(5).astype(int))

```

## Q3：什么时候必须先看 shape，再看公式？

后面进 PyTorch、Attention 和更复杂的张量操作时，先看 shape 再看公式通常更稳。只要你能先确认 batch、seq、feature 和 head 这些维度，后面的 reshape、transpose 和广播就不容易写错。这里可以把它当成 0B 的前置：先看维度，再看实现；进到 Tensor、embedding 和 attention 时，这个顺序最重要。


```python
def split_heads(x, num_heads):
    b, t, d = x.shape
    assert d % num_heads == 0
    head_dim = d // num_heads
    return x.reshape(b, t, num_heads, head_dim).transpose(0, 2, 1, 3)


def merge_heads(x):
    b, h, t, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * head_dim)


def add_feature_bias(x, bias):
    return x + bias

```

## Q3验证：shape contract 和 head 拆分是否正确？

这里直接看两件事：`(B, T, D)` 的数据加上 feature bias 后是否还能对齐，以及 split / merge heads 后 shape 是否能回到原样。只要这一步能对上，后面的 Tensor 和 attention 语法就不会太陌生，也更容易接到 0B 里的 layout 和 mask。


```python
x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
bias = np.array([1, 10, 100, 1000])
y = add_feature_bias(x, bias)

print('原始 shape：', x.shape)
print('加 bias 后的第一行：', y[0, 0])
assert np.array_equal(y[0, 0], x[0, 0] + bias)

heads = split_heads(x, 2)
print('split 后 shape：', heads.shape)
merged = merge_heads(heads)
print('merge 后 shape：', merged.shape)
assert np.array_equal(merged, x)
print('✅ shape contract 通过')

```
