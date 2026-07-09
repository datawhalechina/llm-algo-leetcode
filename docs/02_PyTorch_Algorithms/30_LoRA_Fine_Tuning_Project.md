# 30. LoRA Fine-Tuning Project | LoRA 微调项目

**难度：** Hard | **环境：** CPU-first

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/30_LoRA_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**标签：** `项目实战`, `LoRA`, `Finetuning` | **目标人群：** 模型微调与工程部署

**关键词：** `LoRA`, `training`, `project`, `profiling`

## 前置阅读

**导语：** 先把训练、对齐和显存优化主线看完，再做 LoRA 项目更能体现跨模块联动。
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](./27_ZeRO_Optimizer_Sim.md)

## 相关阅读

**导语：** 项目页之后，建议继续看推理性能和训练性能分析。
- [31. Inference Performance Comparison | 推理性能对比实验](./31_Inference_Performance_Comparison.md)
- [32. Training Performance Analysis | 训练性能分析](./32_Training_Performance_Analysis.md)


## 项目目标

这个项目的目标不是再讲一遍 LoRA 原理，而是把它放进一个可交付的微调流程里：先冻结骨干模型，再插入低秩适配器，最后通过训练、显存和速度三条线验证它到底省了什么、换来了什么。

- 先把 LoRA 接到一个最小可训练模型上。
- 再比较全参数微调和 LoRA 微调在训练参数量、显存占用和训练速度上的差别。
- 最后把这些结果和 `Part 1` 的 profiling 方法串起来，形成一个最小闭环。

## 实验对象

这页建议使用一个足够小、但结构完整的语言模型作为骨架，例如一个两层 MLP / GRU 风格的 toy causal LM，或者一个简化的 Transformer block。重点不在模型多大，而在于能不能稳定复现下面三个动作：

1. 冻结 base model 的大部分参数。
2. 只为少数线性层注入 LoRA adapter。
3. 在同一批样本上跑出可比较的 loss 曲线。

## 实现步骤

1. **搭建基线**：先得到一个可以正常前向和反向的小模型，记录全参数微调时的训练参数量。
2. **插入 LoRA**：把 LoRA 加到注意力投影或 MLP 线性层上，明确哪些权重冻结、哪些权重可训练。
3. **跑训练闭环**：在一小批样本上训练若干步，确认 loss 可以下降。
4. **记录性能指标**：至少记录 `trainable params`、`peak memory`、`step time` 和 `loss curve`。
5. **做对照复盘**：把 LoRA 结果和全参数微调做对照，说明它省了什么、代价是什么。


```python
import math

```


```python
def lora_trainable_params(in_dim, out_dim, rank):
    """Estimate trainable LoRA parameters for a single linear layer."""
    return rank * (in_dim + out_dim)


def full_linear_params(in_dim, out_dim):
    return in_dim * out_dim


def lora_param_ratio(in_dim, out_dim, rank):
    return lora_trainable_params(in_dim, out_dim, rank) / full_linear_params(in_dim, out_dim)


for hidden_size, rank in [(4096, 8), (4096, 16), (8192, 16)]:
    trainable = lora_trainable_params(hidden_size, hidden_size, rank)
    total = full_linear_params(hidden_size, hidden_size)
    ratio = lora_param_ratio(hidden_size, hidden_size, rank)
    print(f"hidden={hidden_size}, rank={rank} -> trainable={trainable:,}, full={total:,}, ratio={ratio:.4%}")

```

## 需要记录的指标

这个项目至少要留下四类结果，后续才方便和 `31 / 32` 串起来：

- **参数侧**：可训练参数占比、LoRA rank、插入层数。
- **资源侧**：峰值显存、激活占用、是否触发明显的 offload 或 checkpointing 变化。
- **时间侧**：单步耗时、吞吐、是否出现明显的同步开销。
- **效果侧**：loss 是否下降、训练是否稳定、对比 baseline 是否有可见收益。

如果后面要把这个项目和 profiling 线合起来，最关键的是把 `参数侧` 和 `资源侧` 写清楚，因为它们最容易解释 LoRA 到底“轻”在哪里。

## 复盘问题

- LoRA 省下来的主要是哪些参数和哪一类显存？
- 如果把 rank 提高，训练参数和收益会怎么变？
- LoRA 什么时候适合和 gradient accumulation、checkpointing 一起看？
- 这个项目的结论，如何接到 `Part 1` 的 profiling 方法和 `Part 2` 的训练分析？

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
def lora_trainable_params(in_dim, out_dim, rank):
    """Estimate trainable LoRA parameters for a single linear layer."""
    return rank * (in_dim + out_dim)


def full_linear_params(in_dim, out_dim):
    return in_dim * out_dim


def lora_param_ratio(in_dim, out_dim, rank):
    return lora_trainable_params(in_dim, out_dim, rank) / full_linear_params(in_dim, out_dim)

```

### 测试


```python
def test_lora_project_template():
    trainable = lora_trainable_params(8, 8, 2)
    total = full_linear_params(8, 8)
    ratio = lora_param_ratio(8, 8, 2)

    assert trainable == 32
    assert total == 64
    assert abs(ratio - 0.5) < 1e-12
    print("✅ LoRA 项目模板代码通过基础校验。")


test_lora_project_template()

```
