# 03 归一化演化（Norm Evolution）

## 页面目标

这一页回答的是：为什么 norm 的位置和形式会改变深层 Transformer 的训练稳定性，以及 RMSNorm、pre-norm 等选择如何进入 block。

本页的输出是稳定性接口：明确 norm 解决的是哪类数值与优化问题，而不是把它当作孤立的替换层。

这一页解释归一化为什么会从 LayerNorm 一路演化到 RMSNorm、Pre-Norm 和更深层的稳定化设计。

## 问题起点

归一化解决的不是“让数值好看”这种表面问题，而是：

- 让深层网络更稳定
- 让 residual 路径更可控
- 让训练不那么容易发散

在大模型里，norm 的位置和类型直接影响：

- 梯度流动
- 收敛速度
- 训练稳定性
- block 级残差行为

## 演化过程

### BatchNorm 时代

在 Transformer 之前，BatchNorm 是深度网络里最典型的稳定化工具之一。

它依赖 batch 统计量来约束激活分布，适合很多视觉模型，但放到序列模型里就会遇到明显限制：

- 序列长度可变，统计行为不够统一
- 推理时 batch 往往更小，统计稳定性会下降
- 对 token 级别建模来说，batch 维度并不是最自然的归一化轴

所以，BatchNorm 更像是“深度网络先把激活稳住”的起点，但它不是 Transformer 的最优落点。

### LayerNorm 时代

LayerNorm 提供了更适合序列建模的归一化思路，它按样本、按 token 做规范化，而不是依赖 batch 统计。

它的价值在于把每层输出拉回一个相对可控的尺度，但在 Transformer 里，norm 的位置很快就暴露成关键变量：

- norm 不只是“数值归一”
- 它会直接影响 residual 的信息保真度
- 深层网络是否稳定，往往取决于 block 输入输出分布是否可控

这也是为什么 Transformer 语境里，LayerNorm 会迅速成为基准方案。

### Pre-Norm / Post-Norm 讨论

Transformer 训练中，norm 放在 attention / MLP 前后，会直接影响梯度传播和深层训练稳定性。

这不是一个“实现风格”问题，而是优化路径问题：

- post-norm 更接近早期 Transformer 的直觉实现，但深层训练更容易不稳
- pre-norm 把规范化放到子层前面，通常更利于梯度穿过很多层
- 当层数、学习率和 warmup 一起变化时，这个差异会被明显放大

所以 pre-norm 的流行，本质上是训练可扩展性逼出来的结果。

### RMSNorm 时代

RMSNorm 去掉了均值中心化，只保留尺度归一化，常见于现代 LLM。

它的优势通常体现在：

- 计算更轻
- 在大模型里通常足够稳定
- 和 residual / block 设计搭配更自然

RMSNorm 能在现代 LLM 里普及，不只是因为它“更简单”，而是因为大模型越来越关注：

- 训练时能否稳定收敛
- 推理时能否减少不必要的计算
- block 内部是否能保持一致的数值行为

它代表的是从“尽可能标准化”转向“只保留必要约束”的设计倾向。

### 更深层的稳定化设计

- NormFormer 增加额外归一化来改善训练
- DeepNorm 试图让极深 Transformer 更稳定
- 一些模型还会结合 residual scaling 或局部 trick

这一阶段说明 norm 已经不只是一个单点算子，而是深层训练系统的一部分：

- 当层数变深，norm 的位置会和 residual 共同决定梯度路径
- 额外的 norm / scaling 往往是在补偿深层优化中的数值放大或衰减
- 更激进的模型会把 norm、残差缩放和 block 设计一起调整

换句话说，norm 的演化史就是 Transformer 训练稳定性的演化史。

### DyT / Normalization-Free Frontiers

更前沿的一条路线，开始尝试把“归一化”从显式层变成更轻量的逐点变换。

DyT 就是这类思路的代表之一：它用类似 `tanh` 的动态门控式变换替代传统 norm，希望在不依赖标准归一化层的前提下，仍然获得稳定训练效果。

它的意义不在于“把 norm 完全删掉”，而在于：

- 挑战 norm 必须显式存在的传统假设
- 把稳定性的一部分职责交给更轻量的点函数
- 为更简单的 Transformer 结构提供新的实验方向

所以，从 BatchNorm 到 LayerNorm，再到 RMSNorm，最后到 DyT，实际上是从“依赖统计归一”走向“依赖更轻的动态约束”。

## 代表模型

- `LLaMA`：RMSNorm 是其标准 block 组件之一
- `Gemma`：结构上也强调 norm 的稳定性和工程可用性
- `DeepSeek`：在现代大模型中继续沿用高效 norm 设计

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) | 归一化时代的起点，理解深度网络如何先把激活稳住。 |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | 归一化的基础入口，理解后续所有变体的起点。 |
| [Transformers without tears: Improving the normalization of self-attention](https://www.amazon.science/publications/transformers-without-tears-improving-the-normalization-of-self-attention) | 帮助理解 Transformer 里 norm 放置方式为什么会影响稳定性。 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 解释 RMSNorm 为什么成为现代 LLM 的常用选择。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [NormFormer: Improved Transformer Pretraining with Extra Normalization](https://arxiv.org/abs/2110.09456) | 展示额外归一化如何影响训练和性能。 |
| [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) | 代表更深层网络的稳定化思路，适合看 norm 与深度的关系。 |
| [Transformers without Normalization](https://arxiv.org/abs/2503.10622) | DyT 这一类 normalization-free Transformer 的代表入口。 |

## 与 Part 02 的对应关系

- [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.md) 直接讲 RMSNorm
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md) 里可以看到 norm 如何进入 block 组装
- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md) 和 [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.md) 可以用于核对真实模型中的 norm 变体

## 可视化提示

建议画一张“norm 演进时间线”：

- LayerNorm
- Pre-Norm / Post-Norm
- RMSNorm
- NormFormer / DeepNorm

再补一张“block 内 norm 位置图”，标出：

- attention 前的 norm
- MLP 前的 norm
- residual 与 norm 的相对关系

## 进入下一页

进入 [04 Attention Evolution](./04_attention_evolution.md)，继续观察稳定的 hidden state 如何建立 token 间的上下文关系。

## 阅读建议

如果你要继续扩展，建议接着看：

- `04_attention_evolution.md`
- `05_rope_position_encoding.md`
- `06_block_residual_path.md`
