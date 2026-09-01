# 02. Training Memory Pressure | 训练侧显存压力

## 页面目标

本节回答的是：训练为什么会 OOM，训练显存通常先被谁吃掉，以及 batch、activation、optimizer state 应该怎么分开看。

## 问题起点

训练显存问题最常见的表象是“一上大 batch 就炸”，但真正把系统拖垮的往往不是 batch 这个单变量，而是：

- activation 在前向和反向之间持续保留；
- gradient accumulation 把有效 batch 放大；
- optimizer state 和梯度常驻；
- 某些层的中间状态在后向阶段集中抬高峰值。

因此，训练显存不能只问“batch 能多大”，而要问“哪个资源对象正在把峰值顶上去”。

## 你要先确认什么

- OOM 出现在 step 一开始，还是中后段逐渐堆高。
- activation 是不是峰值主因。
- 是否已经在用 accumulation，却仍然没有把账本拆清楚。

## 核心矛盾

训练侧的核心矛盾是：模型希望保留足够多的中间状态做反传，但系统又必须把这些状态压进有限显存预算。越大的 effective batch、越长的序列、越深的模型，越会把这个矛盾推到前台。

需要区分两个 batch：`micro-batch` 决定一次前向/反向需要保留多少 activation，`effective batch` 还包含梯度累积的步数。梯度累积通常可以降低单次 activation 峰值，但不会自动减少参数、梯度或 optimizer state；如果原本已经是 `batch_size=1`，继续增加累积步数主要改变优化步节奏，不会继续降低单步 activation 峰值。

## 它如何承接 Task0

[Part00 07 Autograd and Backward](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md) 解释计算图和梯度流，[18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md) 进一步说明 loss、logits 和中间激活在反向传播中的生命周期；需要研究 Attention 反向时再看 [17 Attention Backward](../../02_PyTorch_Algorithms/17_Autograd_Basics.md)。本节把这些机制转换成显存问题：哪些张量必须保留、哪些张量可以重算、哪些状态只是 optimizer 或 batch 组织带来的常驻成本。

## 演化路径

1. 先从 batch / sequence length 的粗调开始。
2. 再分清 parameters、gradients、optimizer state、activations 谁是主因。
3. 如果确认 activation 是主因，再进入 checkpointing 和 offload；如果主因是参数、梯度或 optimizer state，则转向 sharding、ZeRO、量化或其他状态压缩路线。
4. 把候选收益放回 `73 → 76 → 75 → 74`：先建立 baseline，再比较策略、检查预算敏感性，最后用 profiling 解释时间代价。

Task1 到 Task2 的边界在这里：Task1 负责说明对象、规模和硬件代价；Task2 才讨论 accumulation、checkpoint 和 offload 如何改变 activation 的生命周期或驻留位置。不要用 Task1 的理论账本直接替代 Task2 的真实训练测量。

## 关键取舍

- 直接缩 batch 最简单，但会改变吞吐和训练节奏。
- `gradient accumulation` 看似省显存，本质是在时间和 step 组织上换空间。
- activation 优化通常能立刻见效，但很少没有时间代价。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- large batch / gradient accumulation 相关资料：理解 effective batch 如何改变显存与优化步节奏。
- activation memory / training system 论文：理解训练峰值为何多在中间状态上。

## 证据边界

CPU 实验可以检查张量生命周期、梯度对齐和策略账本；GPU 实验才能确认 activation 峰值、重算代价、搬运代价和 OOM 边界。本节的估算或小规模运行不替代 73、76 的固定 workload 结果。

## 对应 Part 02

- [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- [Part00 07 Autograd and Backward](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)、[18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md)、[19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md)
- [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)

## 典型阅读入口

- [03 Checkpointing and Offload](./03_checkpointing_and_offload.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## Task2 的策略账本

| 机制 | 主要减少的对象 | 没有减少的对象 | 代价 | 本阶段证据 |
|---|---|---|---|---|
| Gradient Accumulation | 单个 micro-batch 的激活峰值 | 参数、梯度、优化器状态 | 更多微步，吞吐和 step 节奏变化 | 12 的有效 batch 练习 |
| Checkpointing | 需要长期保存的中间激活 | 参数、梯度、优化器状态 | 反向阶段重算 | 19 的正确性与 toy 峰值对比 |
| Offload | GPU 上驻留的部分激活 | 激活总量 | CPU-GPU 搬运、带宽和同步等待 | 42 的预算模拟，真实结论交给 76 |

### Task2 结束时应能回答

1. 当前 OOM 的主因是参数、梯度、优化器状态还是激活？
2. 选择的策略具体减少了哪一类 GPU 驻留？
3. 代价转移到了微步数量、重算，还是 CPU-GPU 搬运？
4. 如何用 peak memory、reserved memory、step time、吞吐和质量指标验证？

Task2 的 Notebook 只负责建立机制和小规模证据，不直接给出真实大模型收益百分比。需要形成项目结论时，进入 `73 → 76 → 75` 的固定 workload、预算和决策流程。

## 本节要点

训练显存问题首先是资源对象问题，其次才是 batch 参数问题。
