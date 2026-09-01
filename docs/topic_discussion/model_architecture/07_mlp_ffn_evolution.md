# 07 MLP / FFN 演化（MLP / FFN Evolution）

## 页面目标

这一页解释 Transformer 里的 FFN / MLP 为什么从最朴素的两层线性结构，逐步演化到 GELU、SwiGLU 和更复杂的门控设计。

本页的输出是容量扩展接口：理解 gate、up、down 和激活如何改变 block 表达能力，以及为什么 MLP 是 MoE 替换的主要位置。

## 问题起点

Attention 负责 token 之间的信息交互，但每个 token 自己内部的非线性变换主要依赖 MLP / FFN。

所以 MLP / FFN 解决的是：

- token 表示如何在通道维度上重组
- 非线性表达能力如何增强
- 计算成本和表达能力如何平衡

## 演化过程

### 经典 FFN

最基础的 Transformer FFN 是两层线性变换中间加激活函数。

它的核心作用很简单：

- 提供 token 内部的非线性变换
- 在通道维度上重组信息
- 补足 attention 主要负责的 token 间交互

最早的 FFN 虽然朴素，但已经证明了一个事实：只靠 attention 不够，模型还需要一个强一点的逐 token 变换器。

### GELU / ReLU 变体

不同激活函数影响收敛速度和表达能力，是 FFN 的第一层演化。

激活函数的变化看似局部，实际会直接影响：

- 非线性曲线的平滑度
- 梯度传播的稳定性
- 大规模训练时的收敛表现

从 ReLU 到 GELU，再到后来的门控变体，本质上是在寻找更适合大模型优化的非线性形状。

### GLU / SwiGLU

门控 MLP 引入 gate 路径，让模型可以更灵活地控制信息通过。

现代 LLM 中，SwiGLU 很常见，因为它在效果和成本之间通常更均衡。

这一阶段的关键变化是 FFN 不再只是“变换”，而是开始“选择”：

- gate 分支决定哪些信息更值得通过
- up / down 投影负责重排通道空间
- 模型因此可以在相近的参数预算下获得更强表达能力

这也是为什么很多现代 block 会把 MLP 设计成门控结构，而不是纯粹的两层线性网络。

### 更复杂的 MLP 结构

有些模型会在 FFN 上继续做局部修改，比如：

- 宽度和比例调整
- 权重共享
- 与 MoE 结合

一旦 FFN 和 MoE 结合，MLP 就从“每个 token 都走同一条路”变成“不同 token 走不同子路”：

- 普通 token 可以保持 dense 计算
- 关键 token 可以被路由到更合适的专家
- 计算预算和表达能力第一次可以按 token 动态分配

这意味着 FFN 的演化已经超出了单层激活函数的范围，开始进入路由和分工问题。

## 代表模型

- `LLaMA`：SwiGLU 是标准结构的一部分
- `Gemma`：同样强调高效且稳定的 MLP 设计
- `DeepSeek`：会在更大结构里继续组织 FFN / MoE 的关系

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 原始 Transformer FFN 的入口。 |
| [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) | 解释 GELU 为什么成为常见激活函数。 |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 解释门控 FFN 为什么会进入现代 Transformer。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [SwiGLU](https://arxiv.org/abs/2002.05202) | 现代 LLM 中最常见的 MLP 变体之一。 |
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 高质量展示 SwiGLU 如何进入现代 block。 |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | 代表把 FFN 扩展成 MoE 的重要方向。 |

## 与 Part 02 的对应关系

- [Part 02 · 02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.md) 直接讲 SwiGLU
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md) 里可以看到 MLP / FFN 如何进入 block
- [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md) 和 [Part 02 · 07 MoE 负载均衡损失](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md) 展示 FFN 如何进一步升级为 MoE

## 可视化提示

建议画一张 FFN 演进图：

- FFN
- FFN + GELU
- GLU / SwiGLU
- MoE-FFN

并标出：

- gate / up / down 路径
- hidden size 扩张比例
- 与 attention 分支的职责边界

## 进入下一页

进入 [09 MoE / Sparsity Evolution](./09_moe_sparsity_evolution.md)，观察 dense MLP 如何进一步变成 router 与 experts 组成的稀疏路径。

## 阅读建议

如果你要继续扩展，建议接着看：

- `06_block_residual_path.md`
- `08_representative_models.md`
- `09_moe_sparsity_evolution.md`
