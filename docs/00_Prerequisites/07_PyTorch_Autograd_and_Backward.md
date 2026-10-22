# 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `自动求导`, `反向传播` | **目标人群：** 准备学习训练循环、反向状态或显存机制的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

前向传播把张量与算子连接成计算依赖，反向传播沿这些依赖把上游梯度传回参数。非标量输出、分支、残差和共享参数会改变梯度的传递与汇合方式。

完成本节后，你应能解释向量—雅可比积、反向保存状态和梯度累积，并判断一条梯度路径为何存在、汇合或中断。

**关键词：** `requires_grad`, `backward()`, `VJP`, `saved_tensors`

![本节概念关系](../public/00_Prerequisites/07_autograd_backward_map.svg)

## 前置阅读
**导语：** 先理解张量的形状、布局与索引，再把注意力转向计算依赖和梯度流。
- [06. PyTorch Tensor Layout and Indexing | PyTorch 张量布局与索引](./06_PyTorch_Tensor_Layout_and_Indexing.md)
- [0B 组页](./0B.md)

## Q1：自动求导如何沿计算图回传标量梯度？

带 `requires_grad=True` 的叶子张量参与运算后，结果会记录 `grad_fn`。标量结果调用 `backward()` 时，PyTorch 从上游梯度 1 开始应用链式法则，并把结果累积到叶子张量的 `.grad`。

```python
import torch


x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
print('y:', y.item(), 'grad_fn:', type(y.grad_fn).__name__)
y.backward()
print('x.grad:', x.grad.item())
```

## Q1 验证：解析梯度与累积行为是否正确？

```python
x = torch.tensor(2.0, requires_grad=True)
(x * x + 3 * x).backward()
assert x.grad.item() == 7.0
(x * x + 3 * x).backward()
assert x.grad.item() == 14.0
x.grad.zero_()
assert x.grad.item() == 0.0
print('✅ 标量反向传播与梯度累积通过')
```

## Q2：叶子张量、`grad_fn` 与 `retain_grad()` 分别说明什么？

叶子张量是梯度最终写入的位置；运算生成的非叶子张量通过 `grad_fn` 记录来源，但默认不保留 `.grad`。调试中间梯度时可以调用 `retain_grad()`，这不会改变求导公式，只增加一份可观察状态。

```python
leaf = torch.tensor(2.0, requires_grad=True)
intermediate = leaf * 3
intermediate.retain_grad()
loss = intermediate.square()
loss.backward()

print({
    'leaf_is_leaf': leaf.is_leaf,
    'intermediate_is_leaf': intermediate.is_leaf,
    'intermediate_grad_fn': type(intermediate.grad_fn).__name__,
    'leaf_grad': leaf.grad.item(),
    'intermediate_grad': intermediate.grad.item(),
})
```

## Q2 验证：叶子和中间结果的梯度是否符合链式法则？

```python
leaf = torch.tensor(2.0, requires_grad=True)
intermediate = leaf * 3
intermediate.retain_grad()
intermediate.square().backward()
assert leaf.is_leaf is True and intermediate.is_leaf is False
assert leaf.grad.item() == 36.0
assert intermediate.grad.item() == 12.0
print('✅ 叶子张量、grad_fn 与 retain_grad 通过')
```

## Q3：非标量输出为什么需要显式上游梯度？

向量输出没有唯一的“对所有元素求导”结果。`backward(gradient)` 实际计算向量—雅可比积：给定上游向量 `v`，返回 `vᵀJ`。不同上游梯度代表后续计算对各输出位置赋予了不同权重。

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.square()
upstream = torch.tensor([1.0, 0.5])
y.backward(upstream)

print('output:', y.tolist())
print('upstream:', upstream.tolist())
print('vjp:', x.grad.tolist())
```

## Q3 验证：VJP 是否等于上游梯度乘局部导数？

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
upstream = torch.tensor([1.0, 0.5])
x.square().backward(upstream)
expected = upstream * 2 * x.detach()
assert torch.allclose(x.grad, expected)

try:
    torch.tensor([2.0, 3.0], requires_grad=True).square().backward()
except RuntimeError as error:
    assert 'scalar' in str(error)
else:
    raise AssertionError('非标量输出缺少上游梯度时应失败')
print('✅ 向量—雅可比积通过')
```

## Q4：反向传播为什么需要保存前向张量？

某些局部导数依赖前向输入或输出，因此自动求导会保存反向传播所需张量，并在反向阶段读取。`saved_tensors_hooks` 可以观察这条保存—读取链；它说明的是反向依赖，不等同于中间张量的 `.grad`。保存对象的数量和生命周期会进一步影响训练内存。

```python
events = []


def pack_hook(tensor):
    """记录前向传播保存的张量形状，并原样返回保存对象。"""
    events.append(('save', tuple(tensor.shape)))
    return tensor


def unpack_hook(tensor):
    """记录反向传播读取的张量形状，并恢复保存对象。"""
    events.append(('load', tuple(tensor.shape)))
    return tensor


x = torch.tensor([2.0, 3.0], requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    loss = x.square().sum()
    loss.backward()

print('saved tensor events:', events)
```

## Q4 验证：保存与读取是否分别出现在前向传播和反向传播阶段？

```python
events = []
x = torch.tensor([2.0, 3.0], requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    x.square().sum().backward()

event_names = [name for name, _ in events]
assert 'save' in event_names and 'load' in event_names
assert event_names.index('save') < event_names.index('load')
assert torch.allclose(x.grad, torch.tensor([4.0, 6.0]))
print('✅ 反向保存张量的生命周期通过')
```

## Q5：自定义 `Function` 如何闭合前向与反向契约？

自定义 `Function` 的 `forward()` 计算输出，并用 `ctx.save_for_backward()` 保存局部导数所需状态；`backward()` 读取这些状态，将上游梯度乘以局部导数后返回输入梯度。返回梯度的数量必须与需要求导的输入对应。

```python
from torch.autograd import Function


class Square(Function):
    """演示前向保存状态与反向消费上游梯度的最小 Function。"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x


x = torch.tensor(3.0, requires_grad=True)
Square.apply(x).backward()
print('custom gradient:', x.grad.item())
```

## Q5 验证：自定义梯度是否通过解析值与 gradcheck？

```python
x = torch.tensor(3.0, requires_grad=True)
Square.apply(x).backward()
assert x.grad.item() == 6.0

gradcheck_input = torch.tensor([1.5, -2.0], dtype=torch.double, requires_grad=True)
assert torch.autograd.gradcheck(Square.apply, (gradcheck_input,))
print('✅ 自定义 autograd 通过')
```

## Q6：分支、残差与共享参数的梯度如何汇合？

一条张量被多个分支使用时，反向传播会把各分支返回的梯度相加。残差连接 `main + x` 为 `x` 提供一条局部导数为 1 的恒等路径；共享参数同时参与主分支和正则分支时，两条路径的贡献都会累积到参数的 `.grad`。这里要区分三类量：上游梯度是某个节点收到的反向输入，中间张量的梯度需要 `retain_grad()` 才会保留，叶子参数的 `.grad` 则是所有可达分支贡献的累积结果。

```python
x = torch.tensor(2.0, requires_grad=True)
shared_weight = torch.tensor(3.0, requires_grad=True)

main = shared_weight * x
merged = main + x  # 残差恒等路径
merged.retain_grad()
main_loss = merged.square()
regularization_loss = shared_weight.square()
loss = main_loss + regularization_loss
loss.backward()

print({
    'upstream_at_merged': merged.grad.item(),
    'x_grad_after_residual_merge': x.grad.item(),
    'shared_weight_grad_after_branch_merge': shared_weight.grad.item(),
})
```

## Q6 验证：恒等路径和共享参数的分支贡献是否正确相加？

```python
upstream_to_merged = 2 * merged.detach()
expected_x_grad = upstream_to_merged * (shared_weight.detach() + 1)
weight_grad_from_main = upstream_to_merged * x.detach()
weight_grad_from_regularization = 2 * shared_weight.detach()

assert torch.allclose(merged.grad, upstream_to_merged)
assert torch.allclose(x.grad, expected_x_grad)
assert torch.allclose(
    shared_weight.grad,
    weight_grad_from_main + weight_grad_from_regularization,
)
print('✅ 残差恒等路径与共享参数的梯度汇合通过')
```

### 本节小结

- 标量 `backward()` 从上游梯度 1 开始，非标量输出需要显式上游梯度。
- `retain_grad()` 保存中间梯度；反向保存张量保存局部反向公式需要的前向状态，二者职责不同。
- 自定义 `Function` 必须让前向保存的信息、反向传播的输入输出和梯度数量形成闭环。
- 分支会在反向阶段汇合梯度；残差恒等路径和共享参数都会改变最终累积到 `.grad` 的贡献。

## 相关阅读
**导语：** 继续学习训练循环中的梯度边界，并把反向保存张量连接到训练状态和显存账本。
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [08. PyTorch Grad Hygiene and No-Grad | PyTorch 梯度习惯与无梯度模式](./08_PyTorch_Grad_Hygiene_and_No_Grad.md)
- [反向传播与训练机制](../topic_discussion/backpropagation_training_mechanism/intro.md)
