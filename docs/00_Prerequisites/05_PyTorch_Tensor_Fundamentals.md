# 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `张量`, `shape` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/05_PyTorch_Tensor_Fundamentals.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会看 Tensor 的 shape、dtype 和 device；会做基本的 shape 变换；会把索引和 mask 用在最小张量操作里。

**关键词：** `tensor`, `shape`, `dtype`

## 前置阅读
**导语：** 先看 0B 组页，把张量思维和 NumPy 的边界对齐，再进入这一页会更顺。
- [04. Python Config and I/O Patterns | Python 配置与 I/O 模式](./04_Python_Config_and_IO_Patterns.md)
- [0B 组页](./0B.md)
- [01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读
**导语：** 本页先把 Tensor、shape 和 dtype 的最小判断讲清楚；如果想继续看张量布局和索引，再顺着看下面这一页。
- [06. PyTorch Tensor Layout and Indexing | PyTorch 张量布局与索引](./06_PyTorch_Tensor_Layout_and_Indexing.md)
- [03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)

## Q1：Tensor、shape 和 dtype 分别解决什么问题？

进入 PyTorch 后，先别急着看模型，先确认 Tensor 是不是你要的数值容器。最先要看的三件事是 shape、dtype 和 device，它们决定数据能不能继续往下流。


```python
import torch
import numpy as np


def describe_tensor(x):
    return {
        'shape': tuple(x.shape),
        'dtype': str(x.dtype).replace('torch.', ''),
        'device': str(x.device),
        'requires_grad': x.requires_grad,
    }


t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
n = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print('Tensor 描述：', describe_tensor(t))
print('NumPy 形状：', n.shape, 'dtype:', n.dtype)
print('Tensor->NumPy：', t.numpy())

```

## Q1验证：Tensor 和 NumPy 的最小对照是否一致？

这里先确认两件事：Tensor 的 shape / dtype 是否正确，转换成 NumPy 后数值是否还一致。


```python
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
n = t.numpy()
assert tuple(t.shape) == (2, 2)
assert str(t.dtype) == 'torch.float32'
assert np.array_equal(n, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
print('✅ Tensor 基本属性通过')

```

## Q2：什么时候必须先做 shape 变换？

一旦进入多头、批量、序列这些结构，shape 变换就不是装饰动作，而是主流程。先能把 `view / reshape / permute / transpose` 的作用分清，后面看模型代码才不会乱。


```python
x = torch.arange(24).reshape(2, 3, 4)
print('原始 shape：', x.shape)
print('flatten 后：', x.view(-1).shape)
print('reshape 成 (3, 8)：', x.reshape(3, 8).shape)
print('permute 后：', x.permute(0, 2, 1).shape)
print('transpose 后：', x.transpose(1, 2).shape)

```

## Q2验证：shape contract 是否保持？

这里直接检查几个最常见的变换：能不能展平、能不能转回、维度顺序有没有真的按预期交换。


```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.permute(0, 2, 1)
z = y.transpose(1, 2)
assert x.view(-1).shape == (24,)
assert y.shape == (2, 4, 3)
assert z.shape == (2, 3, 4)
print('✅ shape 变换通过')

```

## Q3：什么时候必须先处理索引和 mask？

只要你要取局部片段、过滤无效位置、构造 causal mask，索引和 mask 就是主逻辑，不是辅助代码。后面 attention、padding 和 loss masking 都会复用它。


```python
x = torch.arange(1, 13).reshape(3, 4)
mask = x % 2 == 0
print('原始张量：')
print(x)
print('布尔 mask：')
print(mask)
print('masked 后：')
print(x.masked_fill(~mask, -1))

def build_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

print('causal mask：')
print(build_causal_mask(5).int())

```

## Q3验证：mask 和 causal 关系是否正确？

这里直接检查两件事：偶数元素是否被保留，以及 causal mask 是否是下三角。


```python
x = torch.arange(1, 13).reshape(3, 4)
mask = x % 2 == 0
masked = x.masked_fill(~mask, -1)
assert masked.tolist() == [[-1, 2, -1, 4], [-1, 6, -1, 8], [-1, 10, -1, 12]]
expected = [
    [True, False, False, False],
    [True, True, False, False],
    [True, True, True, False],
    [True, True, True, True],
]
assert build_causal_mask(4).tolist() == expected
print('✅ mask 通过')

```
