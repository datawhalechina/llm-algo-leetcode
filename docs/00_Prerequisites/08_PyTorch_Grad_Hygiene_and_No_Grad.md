# 08. PyTorch Grad Hygiene and No Grad | PyTorch 梯度习惯与无梯度模式

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `梯度控制`, `无梯度模式` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会区分训练、验证和推理的梯度边界；会把 `zero_grad()` 放到正确位置；会分清 `no_grad()` 和 `detach()`。这一页主要承接上一页的计算图机制，把‘梯度会怎么流’进一步收束成‘梯度什么时候该停’。后面 Part 2 里只要开始写训练循环、做评估或排查显存，这一页的语法就会直接用上。
**关键词：** `zero_grad`, `no_grad`, `detach`

## 前置阅读
**导语：** 先看 0B 组页，把 Autograd 和梯度边界对齐，再进入这一页会更顺。
- [07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](./07_PyTorch_Autograd_and_Backward.md)
- [0B 组页](./0B.md)

## 相关阅读
**导语：** 本页先把训练、验证和推理中的梯度边界讲清楚；如果想继续看模型封装和参数管理，再顺着看下面这一页。
- [09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](./09_PyTorch_nn_Module_Basics.md)

## Q1：训练、验证和推理的梯度边界分别是什么？

如果一段代码只是在算指标或做推理，它通常不该继续追踪梯度。先把这三个阶段分清，后面 `no_grad()` 的使用就不会乱。这里先记住：训练阶段会建图，验证 / 推理阶段通常不需要建图；更纯的推理场景还可以用 `inference_mode()`。


```python
import torch


x = torch.tensor([2.0], requires_grad=True)
y = x * 3
# 训练时这里会建立计算图，`backward()` 后梯度会写回叶子节点。
y.backward()
print('训练时 grad:', x.grad.item())

# `no_grad` 包住的整段代码都不会继续追踪梯度，常用于验证 / 推理。
with torch.no_grad():
    z = x * 4
print('推理时 requires_grad:', z.requires_grad)

# `inference_mode()` 是更偏纯推理的上下文，也不会建图。
with torch.inference_mode():
    z2 = x * 5
print('inference_mode 下 requires_grad:', z2.requires_grad)

```

## Q1验证：边界切换是否符合预期？

这里直接确认：训练时会产生梯度，`no_grad()` 里不会继续追踪梯度。你要记住的是，梯度边界不是数值问题，而是计算图是否还在继续生长。


```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3
y.backward()
assert x.grad.item() == 3.0

with torch.no_grad():
    z = x * 4
assert z.requires_grad is False
print('✅ 边界切换通过')

```

## Q2：什么时候必须先清零梯度？

只要你准备开始下一轮参数更新，就要先清零梯度。不然上一轮残留会继续累积，最后你看到的不是当前 batch 的结果，而是累计后的混合结果。这里重点记住 `zero_()`、`backward()` 和下一轮更新的顺序；真实训练循环里通常会写成 `optimizer.zero_grad(set_to_none=True)`。


```python
w = torch.tensor(1.0, requires_grad=True)
loss1 = w * 2
# 如果不清零，梯度就会叠加到上一次的结果上。
loss1.backward()
print('第一次 grad:', w.grad.item())
loss2 = w * 3
loss2.backward()
print('累积后的 grad:', w.grad.item())

# `zero_()` 是原地清零，训练循环里通常在下一轮 backward 前调用。
w.grad.zero_()
loss3 = w * 4
loss3.backward()
print('清零后 grad:', w.grad.item())

```

## Q2验证：梯度累积和清零是否可控？

这里确认三件事：梯度会累加，`zero_()` 能清掉旧值，下一次 backward 会重新开始。你要把它看成训练循环里最基本的“先清再算”语法习惯。


```python
w = torch.tensor(1.0, requires_grad=True)
loss1 = w * 2
loss1.backward()
loss2 = w * 3
loss2.backward()
assert w.grad.item() == 5.0
w.grad.zero_()
loss3 = w * 4
loss3.backward()
assert w.grad.item() == 4.0
print('✅ 梯度累积和清零通过')

```

## Q3：什么时候该用 `detach()`，什么时候该用 `no_grad()`？

`no_grad()` 是用来包住一段不需要追踪的代码；`detach()` 是把某个中间结果从计算图里切出来。前者更像“这段都不要追踪”，后者更像“这个值到这里为止”。这里重点看上下文管理器和张量方法这两种不同写法。


```python
a = torch.tensor(2.0, requires_grad=True)
b = a * 2
# `detach()` 会保留数值，但切断后续梯度传播。
c = b.detach()

# `no_grad()` 是一整段不建图的上下文，里面的结果默认不追踪梯度。
with torch.no_grad():
    d = a * 4

print('b.requires_grad =', b.requires_grad)
print('c.requires_grad =', c.requires_grad)
print('d.requires_grad =', d.requires_grad)

```

## Q3验证：`detach()` 和 `no_grad()` 是否切断梯度？

这里直接检查：中间结果 `detach()` 后不再追踪，`no_grad()` 里计算出来的张量也不会追踪。你要记住的是，这两者都能切断梯度，但切断的粒度不一样。


```python
a = torch.tensor(2.0, requires_grad=True)
b = a * 2
c = b.detach()
with torch.no_grad():
    d = a * 4
assert c.requires_grad is False
assert d.requires_grad is False
print('✅ detach 和 no_grad 通过')

```

### 本节小结

- `train / eval / no_grad / detach` 负责把梯度边界管住。
- `zero_grad` 是训练循环里最基础的清零习惯。
- 推理、验证和训练最好共享同一套接口，但不要共享错误的梯度状态。
