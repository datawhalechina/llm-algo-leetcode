# 14. Activation Functions | 激活函数

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `非线性`, `激活` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/14_Activation_Functions.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：理解 ReLU、GELU 和 SiLU 的常见写法；知道激活函数在非线性建模中的作用；形成对激活分布和数值范围的基本判断。训练骨架已经跑起来之后，接下来要看的是模型为什么不只是线性堆叠。对还不熟 NLP / Transformer 的学习者来说，可以先把激活函数理解成“让线性层多一层非线性”的工具：它会在 MLP、门控结构和 Transformer block 里反复出现。

如果你把 LLaMA 风格 block 看成“Norm -> Attention -> Norm -> MLP”的重复结构，那么激活函数最常待的地方就是 MLP 或门控结构里。

**关键词：** `relu`, `gelu`, `silu`

## 前置阅读
**导语：** 先看 0D 组页，把训练骨架和激活分布的边界对齐，再进入这一页会更顺。
- [13. Simple Neural Network Training | 简单神经网络训练循环](./13_Simple_Neural_Network_Training.md)
- [0D 组页](./0D.md)
- [12. TensorCore and Mixed Precision | TensorCore 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)

## 相关阅读
**导语：** 本页先把常见激活函数和数值分布的最小判断讲清楚；如果想继续看归一化怎么配合激活使用，再顺着看下面这一页。
- [15. Normalization Techniques | 归一化技术](./15_Normalization_Techniques.md)

## Q1：激活函数在模型里解决什么问题？

激活函数的作用不是“多写一个函数名”，而是把线性堆叠变成可学习表达。后面看到 MLP、门控和 Transformer block 时，先把这个最小作用想清楚。对于没有大模型基础的人，可以先把它理解成“给模型加非线性开关”，让一层层线性变换真正组合出复杂模式。


```python
import math
import torch
import torch.nn.functional as F


def relu(x):
    return torch.clamp(x, min=0)


def gelu_exact(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x):
    return x * torch.sigmoid(x)


def activation_summary(x):
    return {
        'relu_mean': float(relu(x).mean().item()),
        'gelu_mean': float(gelu_exact(x).mean().item()),
        'silu_mean': float(silu(x).mean().item()),
    }


x = torch.linspace(-3, 3, steps=7)
print('输入：', x.tolist())
print('ReLU：', relu(x).tolist())
print('GELU：', [round(v, 4) for v in gelu_exact(x).tolist()])
print('SiLU：', [round(v, 4) for v in silu(x).tolist()])

```

## Q1验证：不同激活的输出形态是否可见？

这里直接把同一组输入喂给不同激活，看看它们对负值、零值和正值的处理是否符合预期。


```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
assert torch.allclose(relu(x), F.relu(x))
assert torch.allclose(gelu_exact(x), F.gelu(x), atol=1e-6, rtol=1e-6)
assert torch.allclose(silu(x), F.silu(x))
print('✅ 基本激活函数通过')

```

## Q2：什么时候必须看激活函数的数值分布？

如果激活把输出截断得太狠，梯度和特征分布就容易受影响。看激活函数时，不只看公式，还要看它会不会把输入分布改得太极端。这里先把它当成“分布整形器”来看，而不是单纯的算子。


```python
x = torch.tensor([-3.0, -1.0, 0.0, 2.0])
summary = activation_summary(x)
print('激活摘要：', summary)
print('ReLU 平均值：', summary['relu_mean'])
print('GELU 平均值：', summary['gelu_mean'])
print('SiLU 平均值：', summary['silu_mean'])

```

## Q2验证：激活摘要是否能反映分布差异？

这里直接检查不同激活的均值是否不同，先建立“激活会改变分布”的直觉。


```python
x = torch.tensor([-3.0, -1.0, 0.0, 2.0])
summary = activation_summary(x)
assert set(summary) == {'relu_mean', 'gelu_mean', 'silu_mean'}
assert summary['relu_mean'] == float(relu(x).mean().item())
print('✅ activation_summary 通过')

```

## Q3：什么时候必须关注不同激活的使用场景？

ReLU、GELU、SiLU 不是随便替换的。ReLU 简单，GELU 平滑，SiLU 常见于门控结构。先知道它们在什么地方常出现，后面看 block 才不容易迷路。对 `Transformer -> LLM` 这条线来说，GELU / SiLU 经常和 MLP、gated MLP 一起出现。


```python
def describe_activation(name):
    mapping = {
        'relu': 'simple and sparse',
        'gelu': 'smooth, common in Transformer',
        'silu': 'often used in gated blocks',
    }
    return mapping[name]


print('relu ->', describe_activation('relu'))
print('gelu ->', describe_activation('gelu'))
print('silu ->', describe_activation('silu'))

```

## Q3验证：常见激活的场景标签是否清楚？

这里不比公式，只确认每个名字对应的常见阅读方式是否能稳定说出来。


```python
assert describe_activation('relu') == 'simple and sparse'
assert 'Transformer' in describe_activation('gelu')
assert 'gated' in describe_activation('silu')
print('✅ 激活场景标签通过')

```

## Q4：什么时候必须把激活和统计一起看？

只要你开始排查训练稳定性，就不能只看公式，要把激活和数值统计一起看。激活函数本质上也是在控制分布。这里可以把它和前面的“训练骨架”连起来：激活不只是中间层的数学操作，也会影响整条训练链路的稳定性。


```python
x = torch.linspace(-3, 3, steps=7)
print('输入:', x.tolist())
print('ReLU:', relu(x).tolist())
print('GELU:', [round(v, 4) for v in gelu_exact(x).tolist()])
print('SiLU:', [round(v, 4) for v in silu(x).tolist()])
print('摘要:', activation_summary(x))

```

## Q4验证：统计输出是否和输入分布一起可见？

这里直接把同一组输入的三种激活和摘要一起打印出来，确认“看函数”和“看分布”是同步进行的。


```python
x = torch.linspace(-3, 3, steps=7)
summary = activation_summary(x)
assert summary['relu_mean'] >= summary['silu_mean'] - 1e-6
print('✅ 激活与统计通过')

```

## Q5：什么时候要优先换激活，而不是先改归一化？

如果问题主要出在特征被截断、门控不顺或者负值信息丢太多，先看激活；如果问题是整体统计不稳，再优先看归一化。对于还不熟架构的人，可以简单记成：激活先管“有没有非线性”，归一化再管“数值稳不稳”。


```python
def activation_or_norm(issue_type):
    mapping = {
        'truncation': 'change_activation',
        'gating': 'change_activation',
        'unstable_stats': 'change_normalization',
        'gradient_flow': 'inspect_both',
    }
    return mapping.get(issue_type, 'profile_more')


for issue in ['truncation', 'unstable_stats', 'gradient_flow']:
    print(issue + ':', activation_or_norm(issue))
# 输出示例: truncation -> change_activation, unstable_stats -> change_normalization

```

## Q6：什么时候激活函数其实是在帮你做数值稳定性控制？

当你发现激活分布过宽、极端值太多或者梯度容易爆时，就不能只把激活当成非线性，而要把它看成稳定训练的一部分。激活不只是“把值变一下”，也在影响后面的 norm、attention 和 MLP 是否容易训练。


```python
def activation_stability(max_abs, has_heavy_tail, gradients_explode):
    if max_abs > 10 or has_heavy_tail:
        return {'signal': 'stability_risk', 'next': 'inspect_activation_and_norm'}
    if gradients_explode:
        return {'signal': 'gradient_risk', 'next': 'inspect_activation_scaling'}
    return {'signal': 'ok', 'next': 'continue'}


print(activation_stability(12, True, False))
print(activation_stability(4, False, True))
# 输出示例: stability_risk / gradient_risk

```
