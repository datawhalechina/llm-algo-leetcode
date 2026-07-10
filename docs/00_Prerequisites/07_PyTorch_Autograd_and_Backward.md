# 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `自动求导`, `反向传播` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会判断 Tensor 什么时候需要梯度；会用 `backward()` 观察梯度流动；会区分 `no_grad`、`detach` 和梯度累积。

**关键词：** `requires_grad`, `backward`, `detach`

## 前置阅读
**导语：** 先看 0B 组页，把张量布局和梯度边界对齐，再进入这一页会更顺。
- [06. PyTorch Tensor Layout and Indexing | PyTorch 张量布局与索引](./06_PyTorch_Tensor_Layout_and_Indexing.md)
- [0B 组页](./0B.md)
- [01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读
**导语：** 本页先把 Autograd 和 backward 的最小判断讲清楚；如果想继续看梯度清理和无梯度模式，再顺着看下面这一页。
- [08. PyTorch Grad Hygiene and No-Grad | PyTorch 梯度习惯与无梯度模式](./08_PyTorch_Grad_Hygiene_and_No_Grad.md)

## Q1：Autograd 解决什么问题？

PyTorch 的 autograd 负责自动记录前向计算，并在 `backward()` 时把梯度一路回传到需要梯度的叶子节点。先把这个机制看成“自动记账”，后面就不会把梯度问题误解成普通数值问题。


```python
import torch


x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
print('y:', y.item())
y.backward()
print('x.grad:', x.grad.item())

```

## Q1验证：最小反向传播是否正确？

这里直接检查解析结果：`y = x^2 + 3x` 在 `x=2` 时，梯度应该是 `2x + 3 = 7`。


```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
y.backward()
assert x.grad.item() == 7.0
print('✅ backward 通过，x.grad =', x.grad.item())

```

## Q2：什么时候必须关心 `requires_grad` 和 `grad_fn`？

只要你在调试训练图，先看两个地方：这个 Tensor 会不会被追踪，以及它是不是由某个操作生成的中间结果。前者看 `requires_grad`，后者看 `grad_fn`。


```python
a = torch.tensor(1.0)
b = torch.tensor(1.0, requires_grad=True)
c = b * 2 + 1
print('a.requires_grad =', a.requires_grad)
print('b.requires_grad =', b.requires_grad)
print('c.grad_fn =', type(c.grad_fn).__name__)

leaf = b
print('leaf.is_leaf =', leaf.is_leaf)

```

## Q2验证：叶子节点和历史记录是否清楚？

这里直接确认：默认张量不追踪梯度，显式打开后会追踪，运算结果会带上 `grad_fn`。


```python
a = torch.tensor(1.0)
b = torch.tensor(1.0, requires_grad=True)
c = b * 2 + 1
assert a.requires_grad is False
assert b.requires_grad is True
assert c.grad_fn is not None
print('✅ requires_grad 和 grad_fn 通过')

```

## Q3：什么时候必须处理梯度累积、`no_grad` 和 `detach`？

训练时如果不清零，梯度会累加；推理时如果还追踪梯度，只会白白占内存；如果某一段不想让梯度继续往后走，就要用 `detach()`。


```python
w = torch.tensor(2.0, requires_grad=True)
loss1 = w * 2
loss1.backward()
print('第一次 grad:', w.grad.item())
loss2 = w * 3
loss2.backward()
print('累积后 grad:', w.grad.item())

w.grad.zero_()
with torch.no_grad():
    tmp = w * 4
print('no_grad 下 tmp.requires_grad =', tmp.requires_grad)
detached = (w * 5).detach()
print('detach 后 detached.requires_grad =', detached.requires_grad)

```

## Q3验证：梯度累积和梯度边界是否符合预期？

这里直接检查三件事：梯度会不会累加，`no_grad` 是否关闭追踪，`detach` 是否切断计算图。


```python
w = torch.tensor(2.0, requires_grad=True)
loss1 = w * 2
loss1.backward()
loss2 = w * 3
loss2.backward()
assert w.grad.item() == 5.0

w.grad.zero_()
with torch.no_grad():
    tmp = w * 4
assert tmp.requires_grad is False

detached = (w * 5).detach()
assert detached.requires_grad is False
print('✅ 梯度边界通过')

```

## Q4：什么时候需要自定义梯度计算？

如果默认 autograd 不能表达你的前向和反向，或者你想把一个操作写成可控的小模块，就会用到 `autograd.Function`。这里先把最小接口看懂，不展开复杂实现。


```python
from torch.autograd import Function


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x


x = torch.tensor(3.0, requires_grad=True)
y = Square.apply(x)
y.backward()
print('自定义梯度：', x.grad.item())

```

## Q4验证：自定义反向是否和解析结果一致？

这里直接检查 `x=3` 时的梯度是否为 `6`，先确认最小接口写对，再谈更复杂的算子。


```python
x = torch.tensor(3.0, requires_grad=True)
y = Square.apply(x)
y.backward()
assert x.grad.item() == 6.0
print('✅ 自定义 autograd 通过')

```
