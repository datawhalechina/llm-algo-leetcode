# 02. LoRA / PEFT Design | LoRA 与参数高效微调

## 页面目标

这一页回答的是：在不改动全部参数的前提下，LoRA 为什么能把微调接进来，以及应该把 adapter 挂在哪些层上。

## 你要先确认什么

- target modules 是否选对。
- `r / alpha / dropout` 是否和目标任务匹配。
- 可训练参数比例是否足够。
- merge 之后推理路径是否一致。

## 演化路径

LoRA 不是“加一个小模块”这么简单，它本质上是在冻结 base model 的前提下，给关键线性层加一个低秩旁路。

1. 先冻结主模型，降低训练成本。
2. 再挑选最关键的线性层挂 adapter。
3. 用低秩矩阵表达参数增量。
4. 通过 `r / alpha / dropout` 控制容量和稳定性。
5. 训练结束后决定是否 merge 回主模型。

这条线的关键不是“有 LoRA”，而是“LoRA 挂在哪、挂多大、怎么合并”。

## 方案比较入口

当基础 LoRA 已经能够稳定训练，再进入 [31 LoRA Variants Theory](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.md)，先把不同变体的 `rank / alpha / dropout / target_modules` 统一成可比较的规格。需要真实 benchmark 时，再进入 [63 LoRA Variants Benchmark](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.md)，同时比较可训练参数量、训练稳定性、质量和资源成本。

方案比较不追求脱离场景的“最优 LoRA”，而是根据优先级分流：显存紧张时先看参数量和峰值显存，质量优先时看评测和生成样例，交付优先时还要看实现复杂度、merge 路径和复现成本。

## 常见误区

- target modules 选太少，导致可学习容量不足。
- target modules 选太多，训练代价接近全参微调。
- 只盯 rank，不看数据规模和任务难度。
- merge 后没有核对推理路径，导致结果和训练阶段不一致。

## 经典阅读入口

- [10 LoRA Tutorial](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- [26 QLoRA and 4bit Quantization](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [31 LoRA Variants Theory](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.md)
- [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- [63 LoRA Variants Benchmark](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.md)

## 前置关系

- 先看 `01`，确认监督口径。
- 再看 `02`，理解 LoRA 是怎么接进这个监督闭环的。

## 本节要点

LoRA 的核心价值是把微调成本压下来，但它不是默认正确的。
挂载位置、秩和合并策略都需要和任务一起判断。
