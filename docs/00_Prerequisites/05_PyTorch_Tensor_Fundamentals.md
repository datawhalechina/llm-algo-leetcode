# 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `张量`, `shape` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/05_PyTorch_Tensor_Fundamentals.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会看 Tensor 的 shape、dtype 和 device；会做基本的 shape 变换；会把索引和 mask 用在最小张量操作里。你可以先把它看成从 NumPy 数组走向 PyTorch Tensor 的最短桥：先把数据放进来，再确认它能不能继续往模型里流。后面 Part 2 里只要开始碰输入构造、形状整理和 mask，这一页的语法就会直接用上，同时也会碰到 `torch.tensor`、`from_numpy`、`as_tensor` 和 dtype/device 转换。

如果你主要来自 NumPy，可以先把 Tensor 看成‘带 dtype、device 和 autograd 的数组容器’；这页最重要的不是记 API 名字，而是把 `ndarray -> Tensor` 的最短翻译链看顺。

**关键词：** `tensor`, `shape`, `dtype`

## 前置阅读
**导语：** 先看 0B 组页，把张量思维和 NumPy 的边界对齐，再进入这一页会更顺。
- [04. Python Config and Data Entry | Python 配置与数据入口](./04_Python_Config_and_Data_Entry.md)
- [0B 组页](./0B.md)
- [P1: 01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读
**导语：** 本页先把 Tensor、shape 和 dtype 的最小判断讲清楚，再去看后面的 shape/mask 练习会更顺。
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)

## Q1：Tensor、shape 和 dtype 分别解决什么问题？

进入 PyTorch 后，先别急着看模型，先确认 Tensor 是不是你要的数值容器。对于主要用过 NumPy 的学习者，可以先把 Tensor 看成带 dtype、device 和 autograd 的数组容器；你最先要看的三件事是 shape、dtype 和 device，它们决定数据能不能继续往下流。

这里先把最常见的属性接口认熟：`x.shape` 看维度，`x.dtype` 看数值类型，`x.device` 看数据放在哪；顺手把最常见的创建和转换语法也认一下：`torch.tensor`、`torch.from_numpy`、`torch.as_tensor`、`to(dtype=...)`。


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


# `torch.tensor` 会创建新 Tensor；`from_numpy` / `as_tensor` 会尽量复用 NumPy 的底层内存。
t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
n = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
t_from_np = torch.from_numpy(n)
t_as = torch.as_tensor(n)
# 先看四个最常用属性，再看不同构造方式的差异。
print('Tensor 描述：', describe_tensor(t))
print('NumPy 形状：', n.shape, 'dtype:', n.dtype)
print('from_numpy：', describe_tensor(t_from_np))
print('as_tensor：', describe_tensor(t_as))
print('Tensor->NumPy：', t.numpy())
print('dtype 转换为 int64：', t.to(torch.int64).dtype)

# 这里改 NumPy，观察共享内存的张量会一起变。
n[0, 0] = 99
print('修改 NumPy 后，from_numpy 的首元素：', t_from_np[0, 0].item())
print('修改 NumPy 后，as_tensor 的首元素：', t_as[0, 0].item())
print('torch.tensor 创建的 t 不受影响：', t[0, 0].item())

```

## Q1验证：Tensor 和 NumPy 的最小对照是否一致？

这里先确认两件事：Tensor 的 shape / dtype 是否正确，转换成 NumPy 后数值是否还一致。你可以把它理解成先确认“容器对不对”，再确认“里面的值对不对”。这一组最主要的语法目标，是先会读懂 `tensor -> numpy -> tensor` 这条最短转换链。


```python
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
n = t.numpy()
assert tuple(t.shape) == (2, 2)
assert str(t.dtype) == 'torch.float32'
assert np.array_equal(n, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
print('✅ Tensor 基本属性通过')

```

## Q2：什么时候必须先做 shape 变换？

一旦进入多头、批量、序列这些结构，shape 变换就不是装饰动作，而是主流程。先能把 `view / reshape / permute / transpose` 的作用分清，后面看模型代码才不会乱。这里先记住：`view` 更偏零拷贝，`reshape` 更稳，`permute / transpose` 更像换维度顺序；如果你只是想改 dtype，不要把它和 shape 变换混在一起。


```python
x = torch.arange(24).reshape(2, 3, 4)
# 先看最常见的几个变换：展平、重排、换维度顺序。
print('原始 shape：', x.shape)
print('flatten 后：', x.view(-1).shape)
print('reshape 成 (3, 8)：', x.reshape(3, 8).shape)
print('permute 后：', x.permute(0, 2, 1).shape)
print('transpose 后：', x.transpose(1, 2).shape)
print('permute 和 transpose 结果一致吗：', torch.equal(x.permute(0, 2, 1), x.transpose(1, 2)))
# 这里重点感受不同 API 的语义，而不是死记输出。

```

## Q2验证：shape contract 是否保持？

这里直接检查几个最常见的变换：能不能展平、能不能转回、维度顺序有没有真的按预期交换。你要把 `view / reshape / permute / transpose` 的输出 shape 和它们的语义一起记住；同时记住，改 dtype 不是改 shape。


```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.permute(0, 2, 1)
z = y.transpose(1, 2)
# `view` 负责看形状，`permute` / `transpose` 负责换轴，`reshape` 负责更稳地整理形状。
assert x.view(-1).shape == (24,)
assert y.shape == (2, 4, 3)
assert z.shape == (2, 3, 4)
print('✅ shape 变换通过')

```

## Q3：什么时候必须先确认 dtype 够不够用？

进入后续算子之前，先确认 Tensor 的 dtype 是否对得上。最常见的是两类：浮点张量负责数值计算，整型张量负责索引或 id。这里先记住 `float()`、`long()` 和 `to(dtype=...)` 这几个最小转换接口；更复杂的 mask 和 layout 语法放到 06 里再看。


```python
idx = torch.tensor([0, 1, 2], dtype=torch.long)
vals = idx.float()
# 整型张量更适合做索引；浮点张量更适合做数值计算。
print('idx dtype：', idx.dtype)
print('vals dtype：', vals.dtype)
print('to int64：', vals.to(torch.int64).dtype)

# `to(dtype=...)` 是最常见的类型转换入口。
float_x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print('float_x -> long：', float_x.to(torch.int64))

```

## Q3验证：dtype 和整数索引是否正确？

这里直接检查两件事：整数索引是否还是整型，浮点张量转成整型后 dtype 是否变化正确。你要记住的是，`part 2` 里很多索引类报错，本质上都是 dtype 不对，而不是数值不对。


```python
idx = torch.tensor([0, 1, 2], dtype=torch.long)
vals = idx.float()
assert idx.dtype == torch.int64
assert vals.to(torch.int64).dtype == torch.int64
assert vals.dtype == torch.float32
print('✅ dtype 和索引通过')

```

### 本节小结

- 先把 `Tensor / shape / dtype / device` 认清，再看后面的训练接口。
- `shape` 负责结构，`dtype` 负责数值语义，`device` 负责运行位置。
- `clone / detach / from_numpy` 是最容易和前置边界混在一起的点。
