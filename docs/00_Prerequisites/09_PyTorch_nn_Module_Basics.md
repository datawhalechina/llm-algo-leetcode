# 09. PyTorch nn Module Basics | PyTorch nn.Module 基础

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `模型封装`, `状态管理` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/09_PyTorch_nn_Module_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会用 `nn.Module` 封装最小模型；会区分参数、缓冲区和普通属性；会读懂 `state_dict()` 和 `parameters()`。

**关键词：** `nn.Module`, `Parameter`, `state_dict`

## 前置阅读
**导语：** 先看 0C 组页，把梯度边界和模型封装的边界对齐，再进入这一页会更顺。
- [08. PyTorch Grad Hygiene and No-Grad | PyTorch 梯度习惯与无梯度模式](./08_PyTorch_Grad_Hygiene_and_No_Grad.md)
- [0C 组页](./0C.md)
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)

## 相关阅读
**导语：** 本页先把 `nn.Module`、参数注册和状态保存的最小判断讲清楚；如果想继续看 state_dict 的持久化细节，再顺着看下面这一页。
- [10. PyTorch State_dict and Persistence | PyTorch 状态管理与持久化](./10_PyTorch_State_dict_and_Persistence.md)

## Q1：`nn.Module` 解决什么问题？

模型不是一堆散落函数，而是一个有边界的对象。`nn.Module` 的作用就是把参数注册、前向逻辑、子模块组合和状态保存统一起来。


```python
import torch
import torch.nn as nn


class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = SimpleLinear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = SimpleLinear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


model = TwoLayerMLP(4, 8, 2)
print('参数总量：', count_parameters(model))
print('named_parameters：', [name for name, _ in model.named_parameters()])

```

## Q1验证：参数注册和前向输出是否正确？

这里直接检查两件事：自定义层能不能跑，MLP 的输出 shape 和参数总量是否符合预期。


```python
def test_simple_linear():
    layer = SimpleLinear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0]]))
        layer.bias.copy_(torch.tensor([0.5]))
    x = torch.tensor([[3.0, 4.0]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[2.5]]))
    print('✅ SimpleLinear 通过')


def test_two_layer_mlp():
    model = TwoLayerMLP(4, 8, 2)
    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 2)
    expected_params = (4 * 8 + 8) + (8 * 2 + 2)
    assert count_parameters(model) == expected_params
    print('✅ TwoLayerMLP 通过')


test_simple_linear()
test_two_layer_mlp()

```

## Q2：什么时候必须区分 `Parameter`、`buffer` 和普通属性？

只要一个值需要被保存和恢复，但不应该被优化器更新，它就更像 `buffer`。普通属性则只是辅助计算或配置，不该混进参数管理里。


```python
class ToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer('scale', torch.tensor([2.0]))
        self.name = 'toy'

    def forward(self, x):
        return x * self.weight * self.scale


m = ToyModule()
print('parameters：', [n for n, _ in m.named_parameters()])
print('buffers：', [n for n, _ in m.named_buffers()])
print('state_dict keys：', list(m.state_dict().keys()))

```

## Q2验证：state_dict 是否同时包含参数和 buffer？

这里确认 `state_dict()` 会包含参数和 buffer，但不会把普通属性当成可保存状态。


```python
m = ToyModule()
keys = list(m.state_dict().keys())
assert 'weight' in keys
assert 'scale' in keys
assert 'name' not in keys
print('✅ state_dict 结构通过')

```

## Q3：什么时候必须看 `forward()` 和子模块组合？

只要模型不止一层，就要看 `forward()` 怎么把子模块串起来。这里真正重要的不是“写法”，而是“前向逻辑和模块边界是否清楚”。


```python
x = torch.randn(2, 4)
model = TwoLayerMLP(4, 8, 2)
y = model(x)
print('输出 shape：', y.shape)
print('模块层级：', model)

```

## Q3验证：子模块组合是否正常？

这里直接检查前向输出 shape 和模块结构，确认组合后的网络还能正常执行。


```python
model = TwoLayerMLP(4, 8, 2)
x = torch.randn(5, 4)
y = model(x)
assert y.shape == (5, 2)
print('✅ forward 和子模块组合通过')

```

## Q4：什么时候需要处理 `state_dict` 的保存和恢复？

只要你希望模型状态能稳定落盘、加载和复现，就要把 `state_dict()` 和 `load_state_dict()` 当成标准接口。


```python
model = TwoLayerMLP(3, 6, 2)
before = {k: v.clone() for k, v in model.state_dict().items()}
buffer = model.state_dict()
model.load_state_dict(buffer)
after = model.state_dict()

for key in before:
    assert torch.allclose(before[key], after[key])
print('✅ state_dict roundtrip 通过')

```

## Q4验证：保存和恢复是否不丢状态？

这里只检查一件事：读出来再写回去，参数和 buffer 的值还在不在。


```python
model = TwoLayerMLP(3, 6, 2)
before = {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(model.state_dict())
after = model.state_dict()
for key in before:
    assert torch.allclose(before[key], after[key])
print('✅ state_dict 保持一致')

```
