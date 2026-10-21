# 08. PyTorch Grad Hygiene and No Grad | PyTorch 梯度边界与状态控制

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `梯度控制`, `参数冻结` | **目标人群：** 已会基本 backward、准备学习训练循环和微调的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

训练代码需要同时管理模块行为、计算图记录、历史梯度和参数可训练性。这四类状态彼此独立，混为一谈会造成意外建图、梯度残留或错误冻结。

完成本节后，你应能区分四类控制边界，并判断训练、验证、局部切断和参数冻结时哪些状态应该改变。

**关键词：** `zero_grad`, `no_grad`, `detach`, `requires_grad`

![梯度控制概念关系](../public/00_Prerequisites/08_grad_hygiene_map.svg)

## 前置阅读
**导语：** 先理解 backward、VJP 与 saved tensors，再学习训练、验证和参数冻结中的梯度开关。
- [07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](./07_PyTorch_Autograd_and_Backward.md)
- [0B 组页](./0B.md)

## Q1：模块模式与梯度模式分别控制什么？

`model.train()` / `eval()` 改变 Dropout、BatchNorm 等模块行为，但不会自动打开或关闭梯度；`no_grad()` 与 `inference_mode()` 才控制当前上下文是否记录计算图。验证和推理通常要同时设置正确的模块模式和梯度模式。

```python
import torch
import torch.nn as nn


dropout = nn.Dropout(p=1.0)
x = torch.ones(1, 2, requires_grad=True)

dropout.train()
train_output = dropout(x)
dropout.eval()
eval_output = dropout(x) * 2
with torch.no_grad():
    no_grad_output = dropout(x) * 2
with torch.inference_mode():
    inference_output = dropout(x) * 2

print({
    'train_output': train_output.tolist(),
    'eval_output': eval_output.tolist(),
    'eval_requires_grad': eval_output.requires_grad,
    'no_grad_requires_grad': no_grad_output.requires_grad,
    'inference_tensor': torch.is_inference(inference_output),
})
```

## Q1 验证：`eval()` 是否仍会建图？

```python
dropout = nn.Dropout(p=1.0)
x = torch.ones(1, 2, requires_grad=True)

dropout.train()
train_output = dropout(x)
dropout.eval()
eval_output = dropout(x) * 2
with torch.no_grad():
    no_grad_output = dropout(x) * 2
with torch.inference_mode():
    inference_output = dropout(x) * 2

assert torch.equal(train_output, torch.zeros_like(x))
assert torch.equal(eval_output, x * 2)
assert eval_output.requires_grad is True
assert no_grad_output.requires_grad is False and not torch.is_inference(no_grad_output)
assert inference_output.requires_grad is False and torch.is_inference(inference_output)
print('✅ 模块模式与梯度模式通过')
```

## Q2：梯度为什么会累积，`set_to_none=True` 改变了什么？

PyTorch 默认把新梯度加到已有 `.grad`。一次新的参数更新前通常要清理旧梯度。原地清零保留零 Tensor；`optimizer.zero_grad(set_to_none=True)` 将 `.grad` 设为 `None`，下一次 backward 再创建梯度。两者都能开始新的累计窗口，但状态表示不同。

```python
parameter = nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([parameter], lr=0.1)

(parameter * 2).backward()
(parameter * 3).backward()
print('accumulated grad:', parameter.grad.item())

optimizer.zero_grad(set_to_none=True)
print('after set_to_none:', parameter.grad)
(parameter * 4).backward()
print('new grad:', parameter.grad.item())
```

## Q2 验证：累计窗口能否被两种方式重新开始？

```python
parameter = nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([parameter], lr=0.1)
(parameter * 2).backward()
(parameter * 3).backward()
assert parameter.grad.item() == 5.0

parameter.grad.zero_()
assert parameter.grad.item() == 0.0
(parameter * 4).backward()
assert parameter.grad.item() == 4.0

optimizer.zero_grad(set_to_none=True)
assert parameter.grad is None
(parameter * 5).backward()
assert parameter.grad.item() == 5.0
print('✅ 梯度累积与清理通过')
```

## Q3：`detach()`、`no_grad()` 与 `inference_mode()` 的作用范围有何不同？

`detach()` 只切断指定 Tensor 与既有计算图的连接；`no_grad()` 让上下文中的新运算不被记录；`inference_mode()` 还关闭更多 Autograd 维护状态，更适合不会再次进入训练图的纯推理。离开上下文后，默认梯度记录会恢复。

```python
source = torch.tensor(2.0, requires_grad=True)
tracked = source * 2
detached = tracked.detach()

with torch.no_grad():
    no_grad_result = source * 3
with torch.inference_mode():
    inference_result = source * 4
restored = source * 5

print({
    'tracked': tracked.requires_grad,
    'detached': detached.requires_grad,
    'no_grad': no_grad_result.requires_grad,
    'no_grad_is_inference': torch.is_inference(no_grad_result),
    'inference': inference_result.requires_grad,
    'inference_is_inference': torch.is_inference(inference_result),
    'restored': restored.requires_grad,
})
```

## Q3 验证：局部切断与上下文切断是否互不混淆？

```python
source = torch.tensor(2.0, requires_grad=True)
tracked = source * 2
detached = tracked.detach()
with torch.no_grad():
    no_grad_result = source * 3
with torch.inference_mode():
    inference_result = source * 4
restored = source * 5

assert tracked.requires_grad is True
assert detached.requires_grad is False and detached.item() == tracked.item()
assert no_grad_result.requires_grad is False and not torch.is_inference(no_grad_result)
assert inference_result.requires_grad is False and torch.is_inference(inference_result)
assert restored.requires_grad is True
restored.backward()
assert source.grad.item() == 5.0
print('✅ detach 与梯度上下文通过')
```

## Q4：冻结参数为什么不等于把整个 forward 放进 `no_grad()`？

冻结层的参数不需要 `.grad`，但它的输出仍可能进入后续可训练层。将整个 forward 放入 `no_grad()` 会切断所有后续训练图；正确做法是把冻结参数设为 `requires_grad=False`，并只把可训练参数交给 optimizer。

```python
class TwoStageModel(nn.Module):
    """冻结第一阶段参数，同时保留第二阶段的可训练计算图。"""

    def __init__(self):
        super().__init__()
        self.frozen_stage = nn.Linear(2, 2)
        self.trainable_stage = nn.Linear(2, 1)

    def forward(self, x):
        return self.trainable_stage(torch.relu(self.frozen_stage(x)))


model = TwoStageModel()
for parameter in model.frozen_stage.parameters():
    parameter.requires_grad_(False)

trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.SGD(trainable_parameters, lr=0.1)
loss = model(torch.ones(2, 2)).sum()
loss.backward()

print('frozen grads:', [parameter.grad for parameter in model.frozen_stage.parameters()])
print('trainable grads present:', [parameter.grad is not None for parameter in model.trainable_stage.parameters()])
```

## Q4 验证：冻结层无梯度时，后续可训练层是否仍能更新？

```python
model = TwoStageModel()
for parameter in model.frozen_stage.parameters():
    parameter.requires_grad_(False)

trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.SGD(trainable_parameters, lr=0.1)
before = [parameter.detach().clone() for parameter in model.trainable_stage.parameters()]
model(torch.ones(2, 2)).sum().backward()

assert all(parameter.grad is None for parameter in model.frozen_stage.parameters())
assert all(parameter.grad is not None for parameter in model.trainable_stage.parameters())
optimizer.step()
after = list(model.trainable_stage.parameters())
assert any(not torch.allclose(old, new) for old, new in zip(before, after))
print('✅ 参数冻结与可训练路径通过')
```

### 本节小结

- `train/eval` 控制模块行为，`no_grad/inference_mode` 控制是否记录新计算图。
- 梯度默认累积；清零为 0 与设为 `None` 都能开始新的累计窗口，但状态不同。
- `detach()` 切断单个 Tensor，参数冻结只阻止对应参数接收梯度。
- 冻结部分参数时仍要为后续可训练层保留计算图。

## 相关阅读
**导语：** 接下来把梯度状态放入模型对象、优化器和完整训练循环。
- [09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](./09_PyTorch_nn_Module_Basics.md)
- [11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](./11_PyTorch_Optimizers_and_Loss.md)
- [反向传播与训练机制](../topic_discussion/backpropagation_training_mechanism/intro.md)
