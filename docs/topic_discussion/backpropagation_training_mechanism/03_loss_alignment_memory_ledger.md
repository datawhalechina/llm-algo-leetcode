# 03. Loss Backward、标签对齐与显存账本 | Loss Backward, Label Alignment and Memory Ledger

## 页面目标

本页集中处理训练中最容易混淆的两件事：监督边界如何定义，以及 backward 为什么会增加显存压力。

本页的输出是可信账本：确认 labels、mask、shift、ignore_index 的监督边界，并拆开 activation、参数、梯度、优化器状态和系统缓冲。

## 核心问题

### 1. 为什么 `mask / shift / ignore_index` 很重要

训练里不是所有 token 都应该参与监督。prompt、padding、response、EOS 需要不同口径，否则 loss 会对错位置。

### 2. next-token loss 怎么对齐

自回归训练里，当前位置的 logits 预测下一个 token，所以必须做 shift。

### 3. 为什么激活会占住显存

只要某个中间量在 backward 还要用，它就不能随便丢。激活、参数、梯度、优化器状态和系统缓冲一起构成训练显存账本。

## 机制分解

label alignment 里最关键的不是“有没有算 loss”，而是“loss 在哪里算”：

- prompt 通常不应该被当成监督目标
- response 区间才是主要学习对象
- padding 必须被排除，否则梯度会被无意义 token 污染
- causal LM 里还要做 shift，确保当前位置预测下一个 token

显存账本里最容易漏掉的是一个事实：训练显存不只有 activation。

- 参数
- 梯度
- optimizer state
- 临时 workspace
- 通信缓冲区

都会参与训练预算。

## 账本与测量的对应关系

理论账本回答“哪些对象可能占用显存”，CUDA 统计回答“本次 workload 的峰值如何表现”。两者不应直接画等号：allocator reserved、算子 workspace 和系统进程可能使实测值高于账本；反之，某些 activation 在阶段结束后已经释放，也不会一直叠加。CPU 可以检查账本公式和监督边界，GPU 才能验证峰值与 OOM 边界。

## 典型误区

- `ignore_index` 不是“随便填个值”，它是监督边界的一部分。
- shift 不只是 shape 对齐，它决定预测目标到底是谁。
- 看到显存满了，不代表全部都是 activation。

## 对应来源

- [Part 02 · 09 SFT 训练循环](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)
- [Part 02 · 18 激活与损失反向传播](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md)
- [Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md)

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | 反向信号如何塑造表示，是理解 loss 监督边界的基础语境。 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | decoder self-attention 和 causal mask 是理解自回归监督口径的起点。 |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | 先理解显存账本为什么会逼出 checkpointing。 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 看参数、梯度和 optimizer state 如何一起进入显存预算。 |

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | 直接看 `ignore_index`、reduction 和 target 口径。 |

## 阅读建议

- 先确认监督区间，再谈模型是否学得好。
- 如果你已经知道 next-token prediction，就重点看 mask / shift / ignore_index。

## 进入下一页

如果主要压力来自 activation，进入 [04 Checkpointing 与 Offload](./04_checkpointing_and_offload.md)；如果主要问题是更新频率或有效 batch，再进入 [05 梯度累积、训练闭环与 Profiling](./05_accumulation_decision_profiling.md)。
