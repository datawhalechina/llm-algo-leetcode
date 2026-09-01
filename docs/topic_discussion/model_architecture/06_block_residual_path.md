# 06 Block 与 Residual 主干（Block / Residual Path）

## 页面目标

这一页解释现代 Transformer block 是怎么把 norm、attention、MLP 和 residual 组织成一条可训练、可扩展的信息流。

本页的输出是 block 主干：能沿 hidden state 路径说明每个组件的位置、残差如何传递，以及哪些接口允许后续替换 attention 或 MLP。

## 问题起点

单独看 `Norm`、`Attention`、`MLP` 都不够，因为真正决定模型行为的是它们如何被组装在一起。

block 组装关注的是：

- 各组件的先后顺序
- residual 如何让信息跨层传播
- pre-norm / post-norm 为什么会影响训练稳定性
- dense block 如何为 MoE 或结构技巧留接口

## 演化过程

### 早期 Transformer block

最早的 Transformer block 由 attention、FFN 和 residual 堆起来，核心目标是让序列建模可训练。

这个阶段的关键是先证明一件事：只要把 self-attention 和 FFN 按合适顺序堆叠起来，模型就能学到足够强的上下文表示。

但与此同时，block 里的几个问题也很快暴露出来：

- residual 怎么加才不破坏主干信号
- norm 放前还是放后会影响梯度流
- attention 和 FFN 谁先谁后会影响训练行为

### Pre-Norm 成为主流

现代大模型更常见的是 pre-norm block：

- 先 norm，再做 attention / MLP
- 再通过 residual 把主干信息送回去

这样更利于深层训练稳定性。

pre-norm 的流行本质上是在解决深层优化中的“信号衰减”问题：

- 先把输入拉到稳定尺度，再送入子层
- 子层输出通过 residual 回到主干，减少信息丢失
- 深层堆叠时，每层都更像是在稳定底座上做增量修改

这让 block 从“简单堆叠”变成了“稳定迭代”。

### Dense block 的标准化

随着 LLaMA 类模型普及，很多现代 block 的组织方式逐渐收敛：

- RMSNorm
- self-attention
- residual
- RMSNorm
- MLP / SwiGLU
- residual

这套排列之所以成为事实上的标准，是因为它同时兼顾了：

- 数值上稳定
- 工程上统一
- 和推理缓存、长上下文、并行训练都能对接

所以现代 decoder-only 的 block，已经不只是局部模块，而是整个训练与推理链路里的标准单元。

### 结构扩展

在这个基础上，还可以继续扩展：

- MoE 替换 MLP
- attention 的 head 结构变化
- 长上下文位置编码调整
- 真实实现里的局部 trick

这说明 block 不是固定模板，而是一个可持续生长的接口：

- 当 attention 需要降成本，block 可以替换 head 结构
- 当 MLP 需要更强表达，block 可以切换到 SwiGLU 或 MoE
- 当上下文变长，block 可以接入更强的位置编码或缓存机制

换句话说，block 的演化史就是现代 LLM 主干如何被不断加壳和重构的过程。

## 代表模型

- `LLaMA`：现代 dense block 的参考样本
- `Gemma`：在 block 组织上强调工程可用性
- `DeepSeek`：在 block 之上继续叠加更激进的注意力或专家结构

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer block 的原始结构入口。 |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | 理解 pre-norm / post-norm 对 block 稳定性的影响。 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 解释现代 block 中 norm 选择的趋势。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 现代 block 组装的高质量样本。 |
| [NormFormer: Improved Transformer Pretraining with Extra Normalization](https://arxiv.org/abs/2110.09456) | 代表在 block 内继续做归一化增强的思路。 |
| [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) | 代表更深 block 的稳定化设计。 |

## 与 Part 02 的对应关系

- [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.md)、[Part 02 · 02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.md)、[Part 02 · 03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.md) 和 [Part 02 · 04 Attention（MHA/GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md) 的组件都在这里重新组装
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md) 是最直接的 block 级案例
- [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md) 和 [Part 02 · 07 MoE 负载均衡损失](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md) 展示把 MLP 扩展为 MoE 的方式
- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md) 和 [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.md) 用于核对真实模型中的 block 变体

## 可视化提示

建议画一张 block 总图，至少标出：

- input hidden state
- first norm
- attention
- residual add
- second norm
- MLP / SwiGLU
- second residual add

最好同时标出：

- pre-norm 的位置
- MoE 替换 MLP 的位置
- 真实实现中的局部变体

## 进入下一页

进入 [07 MLP / FFN Evolution](./07_mlp_ffn_evolution.md)，继续观察 block 中负责表示扩展的 MLP 如何从 dense FFN 演化到门控结构。

## 阅读建议

如果你已经看过：

- `03_norm_evolution.md`
- `04_attention_evolution.md`
- `07_mlp_ffn_evolution.md`
- `05_rope_position_encoding.md`

那么这一页就是把它们重新合成一张 block 图的地方。
