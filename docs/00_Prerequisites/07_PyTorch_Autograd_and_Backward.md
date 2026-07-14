# 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `自动求导`, `反向传播` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会判断 Tensor 什么时候需要梯度；会用 `backward()` 观察梯度流动；会区分 `no_grad`、`detach` 和梯度累积。你可以把这一页看成‘张量已经会排版了，现在开始看它怎么被计算图串起来’。后面 Part 2 里只要开始写训练循环、看梯度边界或做自定义算子，这一页的判断方式就会直接用上。

如果你还没有建立计算图直觉，可以先把它看成前向时临时搭起的一张依赖图：张量和算子是节点，数据依赖是边，`backward()` 负责沿着这张图把梯度往回传。

**关键词：** `requires_grad`, `backward`, `detach`

## 前置阅读
**导语：** 先看 0B 组页，把张量布局和梯度边界对齐，再进入这一页会更顺。
- [06. PyTorch Tensor Layout and Indexing | PyTorch 张量布局与索引](./06_PyTorch_Tensor_Layout_and_Indexing.md)
- [0B 组页](./0B.md)

## 相关阅读
**导语：** 本页先把 Autograd 和 backward 的最小判断讲清楚；如果想继续看梯度清理和无梯度模式，再顺着看下面这一页。
- [08. PyTorch Grad Hygiene and No-Grad | PyTorch 梯度习惯与无梯度模式](./08_PyTorch_Grad_Hygiene_and_No_Grad.md)

## Q1：Autograd 机制和计算图分别解决什么问题？

PyTorch 的 autograd 负责自动记录前向计算，并在 `backward()` 时把梯度一路回传到需要梯度的叶子节点。前向计算可以先理解成临时搭出一张计算图：每次运算都会记下输入和输出的依赖关系，`backward()` 再沿着这张图把梯度回传。先把这个机制看成“自动记账”，后面就不会把梯度问题误解成普通数值问题。这里重点熟悉 `requires_grad=True`、`backward()` 和 `.grad` 这三个最常见的接口。

```python
import torch


x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
print('y.item()：', y.item())
# `backward()` 会沿着前向图把梯度写回叶子节点的 `.grad`。
y.backward()
print('x.grad：', x.grad.item())
print('x.is_leaf：', x.is_leaf)

```

## Q1验证：最小反向传播是否正确？

这里直接检查解析结果：`y = x^2 + 3x` 在 `x=2` 时，梯度应该是 `2x + 3 = 7`。这类最小算例的目的，是先把反传的方向和数值对上。


```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
y.backward()
assert x.grad.item() == 7.0
print('✅ backward 通过，x.grad =', x.grad.item())

```

## Q2：什么时候必须关心 `requires_grad` 和 `grad_fn`？

只要你在调试训练图，先看两个地方：这个 Tensor 会不会被追踪，以及它是不是由某个操作生成的中间结果。前者看 `requires_grad`，后者看 `grad_fn`。这两个属性一个描述“会不会记账”，一个描述“这笔账是怎么生成的”。


```python
a = torch.tensor(1.0)
b = torch.tensor(1.0, requires_grad=True)
c = b * 2 + 1
# `requires_grad` 只管要不要追踪，`grad_fn` 只管它是不是前面运算生成的结果。
print('a.requires_grad =', a.requires_grad, '| a.grad_fn =', a.grad_fn)
print('b.requires_grad =', b.requires_grad, '| b.grad_fn =', b.grad_fn)
print('c.grad_fn =', type(c.grad_fn).__name__)
print('a.is_leaf =', a.is_leaf, '| b.is_leaf =', b.is_leaf, '| c.is_leaf =', c.is_leaf)

leaf = b
print('leaf.is_leaf =', leaf.is_leaf)

```

## Q2验证：叶子节点和历史记录是否清楚？

这里直接确认：默认张量不追踪梯度，显式打开后会追踪，运算结果会带上 `grad_fn`。你要记住的是，叶子节点通常自己没有 `grad_fn`，但会在 `.grad` 里接到回传结果。


```python
a = torch.tensor(1.0)
b = torch.tensor(1.0, requires_grad=True)
c = b * 2 + 1
assert a.requires_grad is False
assert b.requires_grad is True
assert c.grad_fn is not None
assert a.grad_fn is None
print('✅ requires_grad 和 grad_fn 通过')

```

## Q3：什么时候必须关心 leaf、`retain_grad()` 和 `is_leaf`？

如果你想看中间张量的梯度，就要先分清叶子节点和非叶子节点。默认情况下，真正会接到 `.grad` 的通常是叶子节点；中间结果如果也要保留梯度，就得显式调用 `retain_grad()`。这里先把 `is_leaf`、`retain_grad()` 和 `.grad` 这几个接口连起来看，边界管理留到 08 再讲。


```python
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
# `b` 是中间结果，默认不会把 `.grad` 留下来；需要时要显式 `retain_grad()`。
b.retain_grad()
loss = b * b
loss.backward()
print('a.is_leaf =', a.is_leaf, '| a.grad =', a.grad.item())
print('b.is_leaf =', b.is_leaf, '| b.grad =', b.grad.item())

# 更复杂的调试场景里，还会配合 `register_hook()` 看梯度流过哪里。

```

## Q3验证：叶子节点和中间结果的梯度是否都清楚？

这里直接检查两件事：叶子节点是否能接到 `.grad`，中间结果在显式 `retain_grad()` 之后是否也能保留梯度。你要记住的是，Autograd 会自动记录图，但中间结果默认不会替你把梯度存起来。


```python
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
b.retain_grad()
loss = b * b
loss.backward()
assert a.is_leaf is True
assert b.is_leaf is False
assert a.grad.item() == 36.0
assert b.grad.item() == 12.0
print('✅ leaf 和 retain_grad 通过')

```

## Q4：什么时候需要自定义梯度计算？

如果默认 autograd 不能表达你的前向和反向，或者你想把一个操作写成可控的小模块，就会用到 `autograd.Function`。这里先把最小接口看懂，不展开复杂实现。重点是 `forward(ctx, ...)` 负责把需要的中间量存进 `ctx`，`backward(ctx, grad_output)` 再把梯度按链式法则算回来。


```python
from torch.autograd import Function


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        # `ctx` 是前向和反向之间传递信息的容器。
        # forward 里先把输入存下来，backward 时还要用它恢复梯度链路。
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # 自定义反向只要按链式法则把局部梯度乘回去即可。
        return grad_output * 2 * x


x = torch.tensor(3.0, requires_grad=True)
# `apply` 是自定义 Function 的标准调用入口。
y = Square.apply(x)
y.backward()
print('自定义梯度：', x.grad.item())

```

## Q4验证：自定义反向是否和解析结果一致？

这里直接检查 `x=3` 时的梯度是否为 `6`，先确认最小接口写对，再谈更复杂的算子。这个验证的作用，是确认 `forward` 存的信息和 `backward` 用的信息能够闭环。


```python
x = torch.tensor(3.0, requires_grad=True)
y = Square.apply(x)
y.backward()
assert x.grad.item() == 6.0
print('✅ 自定义 autograd 通过')

```

### 本节小结

- `backward()` 依赖计算图把梯度往回传。
- `requires_grad / grad_fn / is_leaf / retain_grad()` 是最重要的观察点。
- 自定义 `autograd.Function` 时，关键是让 `forward` 和 `backward` 用同一套中间信息闭环。
