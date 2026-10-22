# 15. Normalization Techniques | 归一化技术

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `归一化`, `统计轴` | **目标人群：** 准备理解 Transformer 统计路径的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/15_Normalization_Techniques.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

归一化需要同时回答三个问题：统计量沿哪些维度计算，公式是否执行中心化，以及 Norm 位于 residual 分支的什么位置。BatchNorm、LayerNorm 与 RMSNorm 在这些方面并不等价。

完成本节后，你应能判断统计量来自哪个维度，区分训练态和推理态的状态依赖，比较 LayerNorm 与 RMSNorm，并解释 Pre-Norm 与 Post-Norm 的数据流差异。

**关键词：** `BatchNorm`, `LayerNorm`, `RMSNorm`, `Pre-Norm`, `running stats`

![归一化的统计来源、公式与残差位置](../public/00_Prerequisites/15_normalization_paths_map.svg)

## 前置阅读
**导语：** 13 节给出 Norm 在 Block 中的位置，14 节展示非线性如何改变分支数值；本页集中处理 Norm 的统计和放置规则。
- [13. Transformer Block Dataflow and Residual Stream | Transformer Block 数据流与残差通路](./13_Transformer_Block_Dataflow_and_Residual_Stream.md)
- [14. Activation Functions | 激活函数](./14_Activation_Functions.md)
- [0D 组页](./0D.md)

## Q1：BatchNorm 和 LayerNorm 使用的统计轴有什么不同？

BatchNorm 使用同一特征在 batch 中的统计量，LayerNorm 使用单个样本最后一维的统计量。统计轴不同，会改变它们对 batch 大小、序列位置和训练/推理模式的依赖；Transformer 通常需要不依赖其他样本的逐 token 归一化。


```python
import torch
import torch.nn as nn
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

## Q1 验证：两种归一化的输出是否可见？

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

BatchNorm 在训练时使用当前 batch 的统计量，并更新 running stats；推理时改用累计统计量。LayerNorm 沿样本内部维度计算，因此通常不依赖 batch 统计。


```python
batch_norm = nn.BatchNorm1d(3, affine=False, momentum=0.5, eps=1e-5)
train_batch = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
batch_norm.train()
train_output = batch_norm(train_batch)
running_mean_after_train = batch_norm.running_mean.clone()
running_var_after_train = batch_norm.running_var.clone()

eval_batch = torch.tensor([[10.0, 20.0, 30.0]])
batch_norm.eval()
eval_output = batch_norm(eval_batch)
print('running mean:', running_mean_after_train.tolist())
print('eval output:', eval_output.tolist())

```

## Q2 验证：训练态与推理态是否使用了不同统计来源？

训练输出应使用当前 batch 统计量；running variance 按 PyTorch 的无偏方差口径更新，推理输出再使用累计统计量。


```python
expected_running_mean = 0.5 * train_batch.mean(dim=0)
expected_running_var = 0.5 * torch.ones(3) + 0.5 * train_batch.var(dim=0, unbiased=True)
expected_eval = (eval_batch - expected_running_mean) / torch.sqrt(expected_running_var + batch_norm.eps)
assert torch.allclose(train_output.mean(dim=0), torch.zeros(3), atol=1e-6)
assert torch.allclose(running_mean_after_train, expected_running_mean)
assert torch.allclose(running_var_after_train, expected_running_var)
assert torch.allclose(eval_output, expected_eval, atol=1e-6, rtol=1e-6)
print('✅ BatchNorm 训练态 / 推理态统计通过')

```

## Q3：`eps` 为什么是归一化公式的一部分？

当一组特征完全相同或方差极小时，直接除以标准差会产生除零或放大误差。`eps` 给分母设置数值下限，使常量输入也能得到有限输出。


```python
constant = torch.ones(2, 4)
centered = constant - constant.mean(dim=-1, keepdim=True)
variance = centered.pow(2).mean(dim=-1, keepdim=True)
without_eps = centered / torch.sqrt(variance)
with_eps = centered / torch.sqrt(variance + 1e-6)
print('without eps：', without_eps)
print('with eps：', with_eps)

```

## Q3 验证：`eps` 是否阻止了非有限输出？

常量输入的方差为 0；不加 `eps` 会出现非有限值，加上 `eps` 后输出应为有限的 0。


```python
assert variance.eq(0).all()
assert not torch.isfinite(without_eps).all()
assert torch.isfinite(with_eps).all()
assert torch.equal(with_eps, torch.zeros_like(with_eps))
print('✅ eps 数值稳定性通过')

```

## Q4：LayerNorm 与 RMSNorm 分别保留了哪些操作？

LayerNorm 先减去最后一维的均值，再按标准差缩放；RMSNorm 不做均值中心化，只按均方根缩放。两者都保持输入 shape，并通常沿 hidden 维使用一组缩放参数。RMSNorm 的输出不保证均值为 0，因此比较时应分别检查“零均值”和“单位均方根”，而不是把两者当作相同公式。`eps` 用来避免输入尺度接近 0 时除零。

```python
def rms_norm_last_dim(x, weight, eps=1e-6):
    """沿最后一维做 RMSNorm，不执行均值中心化。"""
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(mean_square + eps)
    return normalized * weight


x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]])
weight = torch.ones(3)
layer_output = layer_norm_last_dim(x, weight, torch.zeros(3), eps=1e-6)
rms_output = rms_norm_last_dim(x, weight, eps=1e-6)

print('LayerNorm row mean:', layer_output.mean(dim=-1).tolist())
print('RMSNorm row mean:', rms_output.mean(dim=-1).tolist())
print('RMSNorm row RMS:', rms_output.pow(2).mean(dim=-1).sqrt().tolist())
```

## Q4 验证：两种归一化的不变量是否符合各自定义？

这里分别验证 LayerNorm 的零均值和 RMSNorm 的单位均方根，并与 `torch.nn.functional.rms_norm` 对照实现结果。

```python
x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]])
weight = torch.ones(3)
layer_output = layer_norm_last_dim(x, weight, torch.zeros(3), eps=1e-6)
rms_output = rms_norm_last_dim(x, weight, eps=1e-6)
expected_rms = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=1e-6)

assert torch.allclose(layer_output.mean(dim=-1), torch.zeros(2), atol=1e-6)
assert torch.allclose(rms_output, expected_rms, atol=1e-6, rtol=1e-6)
assert torch.allclose(rms_output.pow(2).mean(dim=-1), torch.ones(2), atol=1e-5)
assert not torch.allclose(rms_output[0].mean(), torch.tensor(0.0), atol=1e-6)
print('✅ LayerNorm / RMSNorm 对照通过')
```

## Q5：Pre-Norm 与 Post-Norm 把统计控制放在什么位置？

Pre-Norm 先归一化分支输入，再把分支输出加回原始 residual；Post-Norm 先合并 residual 与分支，再归一化结果。两者使用相同归一化公式，但 identity path 保留方式不同。


```python
def pre_norm_block(x, branch_scale=0.5, eps=1e-6):
    """先归一化分支输入，再保留原始 residual。"""
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return x + branch_scale * normalized


def post_norm_block(x, branch_scale=0.5, eps=1e-6):
    """先合并 residual 与分支，再归一化结果。"""
    combined = x + branch_scale * x
    return F.layer_norm(combined, (x.shape[-1],), eps=eps)


x = torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 8.0]])
print('Pre-Norm block：', pre_norm_block(x))
print('Post-Norm block：', post_norm_block(x))

```

## Q5 验证：两种放置方式是否保留不同的 residual 统计？

Post-Norm 输出沿 hidden 维重新中心化；Pre-Norm 保留原始 residual 的行均值。两者 shape 相同，但统计结果不应被误认为相同。


```python
pre_output = pre_norm_block(x)
post_output = post_norm_block(x)
assert pre_output.shape == post_output.shape == x.shape
assert torch.allclose(post_output.mean(dim=-1), torch.zeros(2), atol=1e-6)
assert torch.allclose(pre_output.mean(dim=-1), x.mean(dim=-1), atol=1e-6)
assert not torch.allclose(pre_output, post_output)
print('✅ Pre-Norm / Post-Norm 放置差异通过')

```

## 相关阅读

建立统计轴、公式和 residual 位置的直觉后，可以继续观察 RMSNorm 与 Attention 怎样进入真实 Transformer Block。

- [16. Attention Mechanism Intro | Attention 机制导论](./16_Attention_Mechanism_Intro.md)
- [Part 02 · 01 RMSNorm](../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.md)
- [Part 02 · 05 LLaMA3 Block](../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)
- [大模型架构演进 · 归一化演进](../topic_discussion/llm_architecture_evolution/03_norm_evolution.md)
