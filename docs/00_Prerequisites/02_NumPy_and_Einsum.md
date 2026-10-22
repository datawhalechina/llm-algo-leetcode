# 02. NumPy and Einsum | NumPy 与 Einsum

**难度：** Easy | **环境：** CPU-first | **标签：** `NumPy`, `广播`, `einsum` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/02_NumPy_and_Einsum.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

多维数组的结果不仅取决于数值，还取决于维度、索引、广播和收缩关系。形状看似兼容时，也可能出现元素顺序改变或错误广播。

完成本节后，你应能追踪数组变形前后的元素与维度，判断广播是否合法，并用显式维度关系核对矩阵运算。

**关键词：** `ndarray`, `broadcast`, `einsum`, `shape contract`

![NumPy shape 与维度运算](../public/00_Prerequisites/02_numpy_shape_einsum_map.svg)

## 前置阅读
**导语：** 先具备 Python 序列、字典和函数契约，再进入多维数组表达。
- [01. Python Essentials for LLM | Python 基础与数据表示](./01_Python_Essentials_for_LLM.md)
- [0A 组页](./0A.md)

## Q1：索引和广播怎样改变数组的内容与 shape？

索引选择位置，广播把较小数组按兼容维度扩展到计算中。添加 feature bias 时，bias 的最后一维必须与特征维一致；选择样本时，布尔 mask 的长度必须与被索引维一致。

```python
import numpy as np


def add_feature_bias(values, bias):
    """为 [..., feature] 数组添加一维 feature bias。"""
    values = np.asarray(values)
    bias = np.asarray(bias)
    if bias.ndim != 1 or values.shape[-1] != bias.shape[0]:
        raise ValueError('bias length must match the last feature dimension')
    return values + bias


values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
bias = np.array([1, 10, 100, 1000])
adjusted = add_feature_bias(values, bias)
selected = adjusted[np.array([True, False])]
print('adjusted shape:', adjusted.shape)
print('selected shape:', selected.shape)
```

## Q1 验证：广播结果和非法 feature 维是否可检查？

```python
values = np.arange(24).reshape(2, 3, 4)
bias = np.array([1, 10, 100, 1000])
adjusted = add_feature_bias(values, bias)
assert adjusted.shape == values.shape
assert np.array_equal(adjusted[0, 0], values[0, 0] + bias)

try:
    add_feature_bias(values, np.ones(3))
except ValueError as error:
    assert 'feature' in str(error)
else:
    raise AssertionError('不匹配的 feature bias 应触发 ValueError')
print('✅ 索引与广播通过')
```

## Q2：reshape、transpose 与 head 拆分必须保持什么？

拆分 head 会改变维度解释和排列顺序，但不能丢失或重复元素。hidden size 必须能被 head 数整除；合并时要先把维度转回 `[B, T, H, D_h]`，再恢复 `[B, T, D]`。

```python
def split_heads(x, num_heads):
    """将 [B, T, D] 变为 [B, H, T, D_h]。"""
    if x.ndim != 3 or num_heads <= 0 or x.shape[-1] % num_heads != 0:
        raise ValueError('hidden size must be divisible by num_heads')
    batch, tokens, hidden = x.shape
    head_dim = hidden // num_heads
    return x.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)


def merge_heads(x):
    """将 [B, H, T, D_h] 恢复为 [B, T, H * D_h]。"""
    if x.ndim != 4:
        raise ValueError('head tensor must be rank 4')
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, tokens, heads * head_dim)


x = np.arange(2 * 3 * 8).reshape(2, 3, 8)
heads = split_heads(x, num_heads=2)
print('split:', heads.shape, 'merged:', merge_heads(heads).shape)
```

## Q2 验证：拆分和合并是否保持元素不变？

```python
x = np.arange(2 * 3 * 8).reshape(2, 3, 8)
heads = split_heads(x, num_heads=2)
assert heads.shape == (2, 2, 3, 4)
assert np.array_equal(merge_heads(heads), x)

try:
    split_heads(np.zeros((2, 3, 7)), num_heads=2)
except ValueError as error:
    assert 'divisible' in str(error)
else:
    raise AssertionError('不能整除的 hidden size 应触发 ValueError')
print('✅ shape contract 通过')
```

## Q3：什么时候用 `einsum` 表达维度关系？

当矩阵乘法包含 batch、query、key 等命名维度时，`einsum` 可以把收缩维直接写进表达式。使用前仍要确认参与收缩的维度大小一致，不能让简洁写法掩盖 shape 错误。

```python
def matrix_multiply(a, b):
    """计算 [M, K] × [K, N] -> [M, N]。"""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('expected compatible [M, K] and [K, N] arrays')
    return np.einsum('mk,kn->mn', a, b)


def batched_pair_scores(query, key):
    """计算 [B, Q, D] 与 [B, K, D] 的两两分数 [B, Q, K]。"""
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError('query and key must be rank-3 arrays')
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError('batch and feature dimensions must match')
    return np.einsum('bqd,bkd->bqk', query, key)


query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
key = np.array([[[1.0, 2.0], [3.0, 4.0]]])
print(batched_pair_scores(query, key))
```

## Q3 验证：Einsum 是否与普通矩阵乘法一致？

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
assert np.array_equal(matrix_multiply(a, b), a @ b)

query = np.array([[[1, 0], [0, 1]]])
key = np.array([[[1, 2], [3, 4]]])
expected = np.array([[[1, 3], [2, 4]]])
assert np.array_equal(batched_pair_scores(query, key), expected)
print('✅ Einsum 维度关系通过')
```

## 相关阅读

掌握数组维度后，继续学习对象封装，或进入 PyTorch 张量的布局与设备语义。

- [03. Python OOP and Utility Patterns | Python 面向对象与工具模式](./03_Python_OOP_and_Utility_Patterns.md)
- [05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](./05_PyTorch_Tensor_Fundamentals.md)
