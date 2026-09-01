# 09 MoE 与稀疏化演化（MoE / Sparsity Evolution）

## 页面目标

这一页解释为什么 dense FFN 会走向 sparse routing，MoE 为什么要把“容量”和“路由”拆开设计，以及这种稀疏化如何影响训练、通信和推理。

本页的输出是稀疏结构边界：明确 router、Top-K、expert、负载均衡和通信代价分别属于模型、训练还是系统问题。

## 问题起点

当模型规模继续变大时，最先撞墙的往往不是“模型不会做事”，而是“每个 token 都走同一条 dense 路径太贵”。

MoE 想解决的核心矛盾是：

- 不想让每个 token 都付出完整 dense FFN 的计算成本
- 但又不想把模型直接压缩成更小的 dense network
- 希望容量可以继续变大，而每个 token 的实际计算量仍然可控

这就是稀疏化的出发点：让参数容量增长得更快，但让激活计算增长得更慢。

## 演化过程

### Dense FFN 的容量上限

在标准 Transformer 里，FFN / MLP 是每个 token 都要走的 dense 路径。

这条路线简单、稳定、好训练，但它也有一个自然上限：

- 每个 token 都在重复访问同一组参数
- 计算量和 token 数、层数、宽度一起线性增长
- 继续堆宽度和深度，边际收益会越来越差

所以 MoE 不是对 dense FFN 的“替代品”这么简单，而是对容量分配方式的重写。

### Router + Experts 的稀疏化思路

MoE 把原来统一的 dense FFN 拆成两部分：

- `router` 决定 token 应该去哪些 expert
- `experts` 负责真正的变换计算

这带来一个关键变化：

- 每个 token 只激活少数几个专家
- 模型总参数可以很大
- 但单 token 的激活成本不会按总参数线性增长

从表达能力上看，这是“参数稀疏激活”；从系统角度看，这是“按 token 动态分配算力”。

### Top-K 路由与负载均衡

真正落地时，router 不能只是简单选最优 expert，否则很容易出现某些专家被挤爆、另一些专家长期闲置。

因此 MoE 通常要处理两件事：

- `Top-K` 路由：每个 token 选少数几个专家
- `load balancing`：尽量让 token 在专家之间分布更均匀

这一步是 MoE 从“概念上很美”走到“工程上可训”的关键。
如果没有负载均衡，稀疏化会很快退化成瓶颈专家和空闲专家并存。

### 训练到推理的系统化落地

MoE 不是只改模型结构，它还会把训练和推理都拖进系统问题里：

- 训练时需要更复杂的 token dispatch 和 gather
- 多专家会引入跨设备通信开销
- 推理时路由行为会影响 latency、吞吐和稳定性

所以 MoE 的演化，最后一定会走到 expert parallel、通信优化和部署约束上。

## 代表模型

- `Switch Transformers`：经典的稀疏路由入口，强调简单的 sparse expert 设计
- `GShard`：展示大规模 MoE 如何和分布式训练结合
- `Mixtral`：把稀疏专家带到更广泛的现代 LLM 语境里
- `DeepSeek-V2 / V3`：展示 MoE 如何和现代 attention、长上下文和工程效率一起演化

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | MoE 早期经典入口，理解 sparse gating 的起点。 |
| [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) | 理解 MoE 如何和分布式训练、切分策略结合。 |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | 理解 Top-1 / Top-K 路由和简单高效稀疏化思路。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 现代 MoE 在开源 LLM 里的代表样本。 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 展示 MoE 如何和高效 attention 及长上下文一起设计。 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 更前沿的 MoE / 稀疏化 / 系统化结构演化样本。 |

## 与 Part 02 的对应关系

- [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md) 直接讲 router 如何分配 token
- [Part 02 · 07 MoE 负载均衡损失](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md) 直接讲为什么 MoE 需要负载均衡损失
- `06_block_residual_path` 说明 MoE 在 block 里替换 dense MLP 的位置
- `08_representative_models` 里可以看到哪些模型已经把 MoE 当成结构选项
- [Part 02 · 79 分布式并行 benchmark](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md) 和 [Part 02 · 80 MoE 专家并行 benchmark](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md) 会把 MoE 的工程化和 benchmark 进一步展开

## 可视化提示

建议画两张图：

- 一张 dense FFN 到 MoE 的对比图，标出 `router -> experts -> combine`
- 一张 expert dispatch 图，标出 token 如何被路由、如何在专家间分配、如何回收输出

如果要再往前走一步，可以把 `Top-K`、负载均衡和 expert parallel 分别标在不同层级上，这样能看出 MoE 为什么同时是模型问题和系统问题。

> 图册占位：MoE / Sparsity 演进图尚未生成，当前先使用本页的 dense → router → experts 文字流程。

## 进入下一页

进入 [08 Representative Models / Cross Module Comparison](./08_representative_models.md)，把前面各组件放回真实模型的组合中。

## 阅读建议

如果你已经看过：

- `07_mlp_ffn_evolution.md`
- `06_block_residual_path.md`
- `04_attention_evolution.md`

这一页就是把 dense block 的一部分改写成 sparse expert 路径的地方。
