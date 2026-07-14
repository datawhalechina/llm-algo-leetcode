# 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `训练闭环`, `优化器` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/11_PyTorch_Optimizers_and_Loss.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：理解 `loss` 和 `optimizer` 的分工；掌握 `zero_grad()`、`backward()`、`step()` 的顺序；能跑通一个最小训练步。前面已经看过模型怎么存，这一页开始看模型怎么更新。阅读顺序可以按这条线走：先分清 loss 和 optimizer 的角色，再看训练步顺序，然后学会先排查训练流程，最后把最小闭环跑通。这里先把它当成 Part 2 里最常回看的“训练步入口”来看：你看到一段训练代码，先找 loss、梯度清理和参数更新分别在哪。训练模式切换通常也会出现在这一步里，`model.train()` / `model.eval()` 影响的是模型行为，不是 loss 的定义。

**关键词：** `loss`, `optimizer`, `step`

## 前置阅读
**导语：** 先看 0C 组页，把状态管理和训练闭环的边界对齐，再进入这一页会更顺。
- [10. PyTorch State_dict and Persistence | PyTorch 状态管理与持久化](./10_PyTorch_State_dict_and_Persistence.md)
- [0C 组页](./0C.md)
- [02. LLM Params and FLOPs | 大模型参数量与 FLOPs](../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md)

## 相关阅读
**导语：** 本页先把 loss、optimizer 和最小训练步讲清楚；如果想继续看数据接口和 batch 契约，再顺着看下面这一页。
- [12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](./12_PyTorch_Minimal_Training_Interface.md)

## Q1：loss 和 optimizer 分别解决什么问题？

`loss` 把任务目标变成一个可以优化的标量，`optimizer` 负责根据梯度更新参数。训练能不能学起来，先看这两个角色是不是分清了。这里先记住：`loss` 负责“算差多少”，`optimizer` 负责“怎么改参数”。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(pred, target):
    # 回归任务里，loss 通常就是把误差压成一个标量。
    return torch.mean((pred - target) ** 2)


def cross_entropy_loss(logits, target):
    # 分类任务里，logits 是未归一化分数，target 是类别 id。
    return F.cross_entropy(logits, target)


pred = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
target = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
print('MSE 示例：', mse_loss(pred, target).item())

logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
labels = torch.tensor([0, 1])
print('CrossEntropy 示例：', cross_entropy_loss(logits, labels).item())

```

## Q1验证：基本 loss 的数值是否正确？

这里直接检查 MSE 和交叉熵的调用是否返回预期数值。


```python
pred = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
target = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
loss = mse_loss(pred, target)
assert torch.allclose(loss, torch.tensor(1.5))

logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
labels = torch.tensor([0, 1])
ce = cross_entropy_loss(logits, labels)
assert torch.allclose(ce, F.cross_entropy(logits, labels))
print('✅ loss 基本验证通过')

```

## Q2：什么时候必须先清梯度，再 backward，再 step？

训练循环里最容易乱的就是顺序：先清梯度，再前向，再反向，再更新。顺序错了，看到的就不是当前 batch 的结果。这里先记住训练步的标准三连：`zero_grad()` -> `backward()` -> `step()`；如果模型里有 dropout 或 norm 之类的行为切换，也要先明确是 `train()` 还是 `eval()`。


```python
def train_one_step(model, x, target, optimizer):
    # `train()` 会把模型切到训练行为，dropout / norm 之类模块会据此工作。
    model.train()
    # 先清掉上一轮残留梯度，再做前向和反向。
    optimizer.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, target)
    loss.backward()
    # `step()` 才是真正更新参数的地方。
    optimizer.step()
    return loss.item()


model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

x = torch.tensor([[1.0], [2.0], [3.0]])
target = 2 * x + 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print('训练前损失：', mse_loss(model(x), target).item())

```

## Q2验证：一步训练后 loss 是否下降？

这里直接看一次更新前后 loss 是否变小，确认顺序和更新真的生效。


```python
model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

x = torch.tensor([[1.0], [2.0], [3.0]])
target = 2 * x + 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before = mse_loss(model(x), target).item()
loss = train_one_step(model, x, target, optimizer)
after = mse_loss(model(x), target).item()
assert abs(loss - before) < 1e-9
assert after < before
print('✅ 一步训练通过，before=', before, 'after=', after)

```

## Q3：什么时候必须先检查 zero_grad 和训练流程？

如果 loss 在变小，但模型表现没变，先检查训练流程是否正确，而不是马上换模型。常见错误往往是梯度没清、更新没发生或者顺序错了。这里重点是把“问题在流程还是在模型”先分开；Q3 主要是排错，别和上面的训练步顺序混在一起。


```python
w = torch.tensor(1.0, requires_grad=True)
loss1 = w * 2
loss1.backward()
print('第一次 grad:', w.grad.item())
loss2 = w * 3
loss2.backward()
print('累积后 grad:', w.grad.item())
w.grad.zero_()
loss3 = w * 4
loss3.backward()
print('清零后 grad:', w.grad.item())

```

## Q3验证：梯度清零和累积是否可控？

这里直接确认：梯度会累加，`zero_()` 可以清掉历史值。


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
print('✅ zero_grad / 累积通过')

```

## Q4：什么时候必须把训练步串成一个最小闭环？

只要你要验证一套训练逻辑，最好先把单步更新跑通，再去谈更复杂的 batch、scheduler 或混合精度。这里的最小闭环就是：前向算 loss，反向拿梯度，optimizer 更新参数。验证时还可以顺手把 `model.eval()` 和 `model.train()` 的行为差别看一眼；Q4 负责把前面的概念收成一条完整训练链。


```python
model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

print('初始 training mode:', model.training)
model.eval()
print('eval mode:', model.training)
model.train()
print('train mode:', model.training)

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target = 2 * x + 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
print('初始损失：', mse_loss(model(x), target).item())
for step in range(3):
    loss = train_one_step(model, x, target, optimizer)
    print(f'step {step + 1}: loss={loss:.4f}')
print('训练后预测：', model(x).detach().squeeze().tolist())

```

## Q4验证：最小训练闭环是否真的能更新参数？

这里直接看训练前后预测值有没有变，确认 optimizer 的更新链路已经打通。


```python
model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target = 2 * x + 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
before = model(x).detach().clone()
for _ in range(3):
    train_one_step(model, x, target, optimizer)
after = model(x).detach().clone()
assert not torch.allclose(before, after)
print('✅ 最小训练闭环通过')

```
