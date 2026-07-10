# 15. Normalization Techniques | 归一化技术

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `归一化`, `稳定性` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/15_Normalization_Techniques.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：理解 BatchNorm 和 LayerNorm 的最小区别；知道训练态和推理态统计量为什么不同；会看最小归一化实现和输出分布。

**关键词：** `BatchNorm`, `LayerNorm`, `running stats`

## 前置阅读
**导语：** 先看 0D 组页，把激活函数和训练稳定性的边界对齐，再进入这一页会更顺。
- [14. Activation Functions | 激活函数](./14_Activation_Functions.md)
- [0D 组页](./0D.md)
- [12. TensorCore and Mixed Precision | TensorCore 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)

## 相关阅读
**导语：** 本页先把 BatchNorm、LayerNorm 和训练态 / 推理态的最小判断讲清楚；如果想继续看 Attention 如何接上这些稳定性直觉，再顺着看下面这一页。
- [16. Attention Mechanism Intro | Attention 机制导论](./16_Attention_Mechanism_Intro.md)

## Q1：BatchNorm 和 LayerNorm 分别解决什么问题？

归一化的核心作用，是让训练中的数值分布更稳定。BatchNorm 更依赖 batch 统计，LayerNorm 更偏向样本内部统计。


```python
import torch
import torch.nn.functional as F


def batch_norm_train(x, gamma, beta, eps=1e-5):
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    normalized = (x - batch_mean) / torch.sqrt(batch_var + eps)
    return normalized * gamma + beta, batch_mean, batch_var


def batch_norm_eval(x, gamma, beta, running_mean, running_var, eps=1e-5):
    normalized = (x - running_mean) / torch.sqrt(running_var + eps)
    return normalized * gamma + beta


def layer_norm_last_dim(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta


x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
gamma = torch.ones(3)
beta = torch.zeros(3)
print('BatchNorm 输出：')
y, mean, var = batch_norm_train(x, gamma, beta)
print(y)
print('LayerNorm 输出：')
print(layer_norm_last_dim(x, gamma, beta))

```

## Q1验证：两种归一化的输出是否可见？

这里直接打印 BatchNorm 和 LayerNorm 的输出，先看它们对同一组输入会做什么。


```python
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
gamma = torch.tensor([1.0, 1.5, 2.0])
beta = torch.tensor([0.0, 1.0, -1.0])
y, mean, var = batch_norm_train(x, gamma, beta)
expected_mean = torch.tensor([1.5, 3.0, 4.5])
expected_var = torch.tensor([0.25, 1.0, 2.25])
expected = (x - expected_mean) / torch.sqrt(expected_var + 1e-5) * gamma + beta
assert torch.allclose(mean, expected_mean)
assert torch.allclose(var, expected_var)
assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
print('✅ BatchNorm 基本验证通过')

```

## Q2：什么时候必须区分训练态和推理态？

训练态和推理态可能用不同统计量。BatchNorm 在训练时看 batch，在推理时看 running stats；LayerNorm 通常不依赖 batch 统计。


```python
def update_running_stats(running_mean, running_var, batch_mean, batch_var, momentum=0.1):
    new_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
    new_var = (1.0 - momentum) * running_var + momentum * batch_var
    return new_mean, new_var


running_mean = torch.tensor([0.0, 0.0, 0.0])
running_var = torch.tensor([1.0, 1.0, 1.0])
batch_mean = torch.tensor([2.0, 4.0, 6.0])
batch_var = torch.tensor([3.0, 5.0, 7.0])
new_mean, new_var = update_running_stats(running_mean, running_var, batch_mean, batch_var, momentum=0.5)
print('new_mean:', new_mean.tolist())
print('new_var:', new_var.tolist())

```

## Q2验证：running stats 是否按预期更新？

这里直接检查 running_mean 和 running_var 是否按 momentum 更新。


```python
running_mean = torch.tensor([0.0, 0.0, 0.0])
running_var = torch.tensor([1.0, 1.0, 1.0])
batch_mean = torch.tensor([2.0, 4.0, 6.0])
batch_var = torch.tensor([3.0, 5.0, 7.0])
new_mean, new_var = update_running_stats(running_mean, running_var, batch_mean, batch_var, momentum=0.5)
assert torch.allclose(new_mean, torch.tensor([1.0, 2.0, 3.0]))
assert torch.allclose(new_var, torch.tensor([2.0, 3.0, 4.0]))
print('✅ running stats 通过')

```

## Q3：什么时候必须把归一化和数值分布一起看？

如果训练震荡严重，先别急着换架构，先看归一化和学习率是否合理。归一化本质上是在控制每层特征的统计范围。


```python
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
gamma = torch.tensor([1.0, 2.0, 3.0])
beta = torch.tensor([0.5, 0.0, -0.5])
y = layer_norm_last_dim(x, gamma, beta)
print('LayerNorm 输出：')
print(y)
print('输入均值：', x.mean(dim=-1).tolist())
print('输出均值：', y.mean(dim=-1).tolist())

```

## Q3验证：LayerNorm 是否按最后一维归一化？

这里直接看每一行的输出均值是否被拉回到稳定范围。


```python
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
gamma = torch.tensor([1.0, 2.0, 3.0])
beta = torch.tensor([0.5, 0.0, -0.5])
y = layer_norm_last_dim(x, gamma, beta)
expected = F.layer_norm(x, x.shape[-1:], gamma, beta, eps=1e-5)
assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
print('✅ LayerNorm 通过')

```

## Q4：什么时候必须同时看归一化、激活和残差？

后面你看到 Transformer block、Pre-Norm、Post-Norm 时，归一化通常不是孤立出现的，而是和激活、残差一起控制稳定性。


```python
x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
gamma = torch.ones(3)
beta = torch.zeros(3)
bn_y, bn_mean, bn_var = batch_norm_train(x, gamma, beta)
ln_y = layer_norm_last_dim(x, gamma, beta)
print('BatchNorm mean:', bn_mean.tolist())
print('BatchNorm var:', bn_var.tolist())
print('BatchNorm 输出：')
print(bn_y)
print('LayerNorm 输出：')
print(ln_y)

```

## Q4验证：不同归一化的输出是否可以直接对照？

这里直接把 BatchNorm 和 LayerNorm 的输出放在一起，确认它们解决的问题确实不同。


```python
x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
gamma = torch.ones(3)
beta = torch.zeros(3)
bn_y, _, _ = batch_norm_train(x, gamma, beta)
ln_y = layer_norm_last_dim(x, gamma, beta)
assert bn_y.shape == x.shape
assert ln_y.shape == x.shape
print('✅ 归一化对照通过')

```

## Q5：什么时候先调归一化，再调激活或残差？

如果问题主要来自层间统计漂移、训练震荡或不同层数值范围差异，先调归一化；如果统计本身已经稳了，再去看激活和残差。


```python
def fix_norm_first(variance_shift, layer_drift, activation_saturation):
    if variance_shift or layer_drift:
        return {'first_fix': 'normalization', 'next': 'inspect_running_stats'}
    if activation_saturation:
        return {'first_fix': 'activation', 'next': 'inspect_non_linearity'}
    return {'first_fix': 'residual_or_lr', 'next': 'profile_more'}


print(fix_norm_first(True, False, False))
print(fix_norm_first(False, False, True))
# 输出示例: normalization / activation / residual_or_lr

```

## Q6：什么时候归一化会改变训练动态而不是只改输出值？

只要归一化影响了梯度尺度、running stats 或不同 token / feature 的相对关系，它改变的就不只是输出值，而是训练动态本身。


```python
def norm_changes_dynamics(affects_gradient, updates_running_stats, changes_relative_scale):
    score = sum([affects_gradient, updates_running_stats, changes_relative_scale])
    if score >= 2:
        return {'dynamic_change': True, 'next': 're-tune_lr_or_warmup'}
    return {'dynamic_change': False, 'next': 'continue'}


print(norm_changes_dynamics(True, True, False))
print(norm_changes_dynamics(False, False, True))
# 输出示例: dynamic_change=True 时通常要联动学习率或 warmup

```
