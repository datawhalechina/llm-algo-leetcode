# 02. Autograd 与 Attention Backward | Autograd and Attention Backward

## 页面目标

本页将 PyTorch 的 autograd 机制与 attention 的反向链路放在一起，目标是对齐“机制接口”和“算子级梯度路径”。

本页的输出是算子级反传证据：能解释 `grad_fn`、`saved_tensors` 与 attention backward 的关系，并区分公式不变和执行路径改变。

## 核心问题

### 1. `grad_fn` 是什么

它表示当前张量是怎么被计算出来的，以及后续 backward 应该从哪条路径回传。

### 2. `saved_tensors` 为什么存在

因为 backward 往往需要前向中间状态。框架会把必要张量保留下来，以便反向时重用。

### 3. 自定义 `autograd.Function` 在做什么

它把 forward 和 backward 显式拆开，让你能手写梯度路径，验证自己是否真的理解了反传。

### 4. attention backward 的反向顺序怎么记

最实用的记法是：

`dV -> dP -> dS -> dQ / dK`

先求最容易的分支，再穿过 softmax 回到打分矩阵，最后回到 query 和 key。

## 机制分解

attention backward 里最容易混淆的，不是公式本身，而是中间状态的依赖顺序：

- `V` 的梯度最直接，因为它只受 attention 权重影响
- `P` 是 softmax 之后的概率矩阵，反向时要先穿过它
- `S` 是打分矩阵，通常还要经过缩放和 mask
- `Q / K` 依赖于打分矩阵对输入投影的链路

所以这条链路的重点是：

- 先找最容易求的梯度
- 再穿过 softmax 的耦合关系
- 最后回到 query 和 key 的投影路径

### 现代实现差异：FlashAttention backward / fused backward / recompute

如果只停在手推 `dV -> dP -> dS -> dQ / dK`，你理解的是“数学链路”；真正到现代训练系统里，还要再看三件事：

#### 1. FlashAttention backward 在改什么

FlashAttention 不是只优化前向，它同样会重写 backward 的执行路径。

它的关键不是改梯度公式，而是改这些中间状态怎么被访问：

- 尽量不把完整 attention score / probability 矩阵落回高带宽显存
- 把 forward / backward 都改写成更适合 tile 化和在线归约的形式
- 用更小的中间状态，换更低的 IO 成本

因此，FlashAttention backward 的核心直觉是：

- 梯度公式没变
- 保存点和访问路径变了
- 真正省下来的通常是 `saved_tensors` 的驻留成本和 IO 开销

#### 2. fused backward 在改什么

现代实现里，很多 backward 不再按“一个逻辑步骤一个 kernel”拆开执行，而是尽量做 `fused backward`。

它的目标通常是：

- 减少中间张量写回和再次读取
- 让一段连续的梯度计算在更少的 kernel 边界里完成
- 降低 launch 开销和全局显存往返

所以 fused backward 解决的不是“公式太慢”，而是“实现太碎”。

最典型的收益是：

- 本来需要单独 materialize 的中间量，不再显式写回
- 多个小 kernel 之间的同步和搬运成本下降
- backward 更接近“以算换存，以融合换 IO”

#### 3. recompute 在 attention backward 里为什么常见

重算并不只属于 checkpointing 页，它在 attention backward 里本身就很常见。

原因很直接：

- 保存完整中间 attention 状态太贵
- 某些中间量可以在 backward 时按 tile 或按块重新算出来
- 与其长期保留，不如临时重算

所以在现代 attention backward 里，常见的不是“全存”或“全不存”，而是更细的权衡：

- 哪些状态值得保
- 哪些状态按块重算更划算
- 哪些状态通过 fused kernel 顺手带过去

## 如何把公式和实现对上

先用小张量检查 `dQ / dK / dV` 的形状、数值和 causal mask，再观察实现是否 materialize 完整的 score 或 probability。CPU 小例子能够验证公式和 mask；它不能推出 FlashAttention、fused backward 或其他 kernel 在某张 GPU 上的吞吐收益。需要比较真实保存、重算和 IO 代价时，应进入 GPU benchmark 或 profiler。

这也是为什么 `saved_tensors`、`FlashAttention` 和 `recompute` 应该放在同一页里看，而不是完全拆开。

## 典型误区

- `grad_fn` 不是梯度本身，它只是记录这个张量的生成路径。
- `saved_tensors` 不是白送的，保存越多，显存压力越大。
- attention backward 的代价不只在公式，还在中间状态和 softmax 的稳定性处理。
- `FlashAttention backward` 不是“新公式”，而是“新执行路径”。
- `fused backward` 不是单一算法名，它更像是一类减少中间写回的实现策略。
- `recompute` 不只出现在通用 checkpointing，也会直接出现在 attention backward 的局部实现里。

## 对应来源

- [Part 02 · 17 自动求导基础](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | attention 结构和 causal mask 的共同起点。 |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | 看 attention 的 backward 代价如何被重新组织成更省 IO 的执行路径。 |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | 看更现代的 work partitioning 如何继续影响 forward / backward 的 kernel 组织。 |
| [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767) | 把 attention backward 放回 autodiff 的统一语境里看。 |

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [torch.autograd](https://docs.pytorch.org/docs/stable/autograd) | 直接看 PyTorch autograd 的官方定义、接口和行为边界。 |
| [torch.nn.attention](https://docs.pytorch.org/docs/stable/nn.attention.html) | 看 PyTorch 当前对 attention backend 的抽象和选择方式。 |

## 阅读建议

- 先把 `dV -> dP -> dS -> dQ / dK` 这条链背顺。
- 再回头看 `grad_fn` 和 `saved_tensors`。
- 如果你关心工程层面，重点看 `FlashAttention backward / fused backward / recompute` 这三件事是怎么一起出现的。

## 进入下一页

进入 [03 损失对齐与显存账本](./03_loss_alignment_memory_ledger.md)，先保证监督口径正确，再分析 backward 的显存组成。
