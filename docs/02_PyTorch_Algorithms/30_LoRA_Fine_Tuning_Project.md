# 30. LoRA Fine Tuning Project | LoRA 微调项目

**难度：** Hard | **环境：** CPU-first

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/30_LoRA_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**标签：** `项目实战`, `LoRA`, `Finetuning` | **目标人群：** 模型微调与工程部署

**关键词：** `LoRA`, `training`, `project`, `profiling`

这个项目的目标不是再讲一遍 LoRA 原理，而是把它放进一个可交付的微调流程里：先冻结骨干模型，再插入低秩适配器，最后通过训练、显存和速度三条线验证它到底省了什么、换来了什么。它承接 `2.3` 的训练闭环、`2.5` 的显存优化，也方便和 `Part 1` 的 profiling 方法串成一个最小闭环。

## 前置阅读

**导语：** 先把 LoRA、本体训练和显存优化主线看完，再做项目会更容易体现跨模块联动。
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [11. LR Schedulers WSD Cosine | WSD 余弦学习率调度器](./11_LR_Schedulers_WSD_Cosine.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [13. Simple Neural Network Training | 简单神经网络训练](./13_End_to_End_Fine_Tuning_Experiment.md)
- [P0: 20. Profiling and Memory Ledger | 性能剖析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)

## 相关阅读

**导语：** 项目页之后，建议继续看推理性能和训练性能分析。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
- [31. Inference Performance Comparison | 推理性能对比实验](./31_Inference_Performance_Comparison.md)
- [32. Training Performance Analysis | 训练性能分析](./32_Training_Performance_Analysis.md)

### Step 1: 定义问题与固定 baseline
先把 LoRA 接到一个最小可训练模型上，并明确 baseline 是什么；模型、输入、batch 和评测指标先保持一致。

- 先搭一个可以正常前向和反向的小模型。
- 冻结 base model 的大部分参数，只保留 LoRA 旁路可训练。
- 说明你要比较的是全参微调还是 LoRA 微调。

### Step 2: 跑通 baseline

先在同一批样本上跑出全参数微调的 baseline，确认基础链路稳定。

- 记录训练参数量、loss 曲线、step time 和 peak memory。
- 确认 baseline 能稳定收敛，不要一开始就上复杂优化。

### Step 3: 插入 LoRA 并对比

把 LoRA 加到 attention projection 或 MLP linear layer 上，再看训练参数量、显存和速度会怎么变。

- 冻结底座，只训练低秩旁路。
- 比较全参微调和 LoRA 微调在训练参数量、显存占用和训练速度上的差别。
- 观察 loss 是否能继续下降，以及训练是否稳定。

### Step 4: 复测与复盘

把 LoRA 结果和 baseline 做对照，说明它省了什么、代价是什么，并输出最终结论。

- 输出一张 “baseline vs LoRA” 的对比表。
- 记录本次改动对显存、速度和效果的影响。
- 如果还有后续优化空间，再考虑 rank、插层位置或 accumulation 的取舍。


```python
# ==========================================
# TODO: 完成 LoRA 参数统计的三个函数
# 1. 计算单层 LoRA 的可训练参数量
#    trainable_params = ???
# 2. 计算完整线性层的参数量
#    total_params = ???
# 3. 计算 LoRA 参数占比
#    ratio = ???
# 提示: 先写 LoRA 旁路，再写全参基线，最后做比例对比
# ==========================================
def lora_trainable_params(in_dim, out_dim, rank):
    # TODO 1: trainable_params = ???
    pass

def full_linear_params(in_dim, out_dim):
    # TODO 2: total_params = ???
    pass

def lora_param_ratio(in_dim, out_dim, rank):
    # TODO 3: ratio = ???
    pass

raise NotImplementedError("请先完成 TODO 代码！")

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 计算单层 LoRA 的可训练参数量
def lora_trainable_params(in_dim, out_dim, rank):
    """Estimate trainable LoRA parameters for a single linear layer."""
    return rank * (in_dim + out_dim)


# TODO 2: 计算完整线性层的参数量
def full_linear_params(in_dim, out_dim):
    return in_dim * out_dim


# TODO 3: 计算 LoRA 参数占比
def lora_param_ratio(in_dim, out_dim, rank):
    return lora_trainable_params(in_dim, out_dim, rank) / full_linear_params(in_dim, out_dim)

for hidden_size, rank in [(4096, 8), (4096, 16), (8192, 16)]:
    trainable = lora_trainable_params(hidden_size, hidden_size, rank)
    total = full_linear_params(hidden_size, hidden_size)
    ratio = lora_param_ratio(hidden_size, hidden_size, rank)
    print(f"hidden={hidden_size}, rank={rank} -> trainable={trainable:,}, full={total:,}, ratio={ratio:.4%}")

```

### 解析

**1. TODO 1 (单层 LoRA 的可训练参数量)**
- LoRA 旁路通常由两个低秩矩阵组成，参数量与 rank 成正比。
- 对一个线性层来说，可训练参数可以近似写成 `rank * (in_dim + out_dim)`。
- 这一步的关键是只统计 LoRA adapter，不把冻结的底座参数算进去。

**2. TODO 2 (完整线性层的参数量)**
- 全参数线性层的参数量就是 `in_dim * out_dim`。
- 它对应的是不加 LoRA 时的基线，用来和 LoRA 方案做对照。

**3. TODO 3 (LoRA 参数占比)**
- 参数占比就是 `trainable / total`。
- 这个比例越小，说明在保持适配能力的同时，训练侧的参数开销越低。

**4. 进阶思考**
- 如果 rank 提高，参数量和收益会如何一起变化？
- 如果把 LoRA 插到更多层上，参数占比和训练收益会怎样变？
- 这页的三个指标，如何和 `31 / 32` 里的性能分析串起来？

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
