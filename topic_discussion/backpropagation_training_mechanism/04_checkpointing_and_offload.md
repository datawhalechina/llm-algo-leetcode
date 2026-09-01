# 04. Checkpointing 与 Offload | Checkpointing and Offload

## 页面目标

本页比较两种常见的显存策略：用重算换空间，以及用搬运换空间。

本页的输出是策略边界：明确 checkpointing 消除的是哪类保存，offload 改变的是状态驻留位置，以及算力、带宽和 wall time 各自承担什么代价。

## 核心问题

### 1. checkpointing 做了什么

它不保存所有激活，而是只保存少量检查点。反向传播需要时，再从检查点重新计算中间段。

### 2. offload 做了什么

它把部分状态从 GPU 搬到别的存储层，比如 CPU 或更慢的内存层。

### 3. 它们有什么区别

checkpointing 是重算，offload 是搬运。两者都在省 GPU 显存，但代价来源不同。

## 机制分解

checkpointing / offload 不是同一个维度的方案：

- checkpointing 改变的是“前向状态要不要保留”
- offload 改变的是“状态保留在哪里”
- 两者都能省 GPU 显存，但一个吃算力，一个吃带宽

因此需要先区分它们的作用边界：

- 如果显存不够但算力还富余，checkpointing 往往更直接
- 如果显存特别紧但重算已经太贵，offload 才更有价值
- 如果带宽太弱，offload 可能把瓶颈从显存换成传输

> 图册占位：Checkpointing 取舍图尚未生成，当前以本页的计算、显存和时间代价说明为准。

> 图册占位：Offload 取舍图尚未生成，当前以本页的搬运、显存和吞吐代价说明为准。

## 选择前先确认压力来源

如果峰值主要来自 activation，checkpointing 才有直接作用；如果状态可以安全地放到 CPU 且 PCIe / NVLink 传输仍可接受，offload 才值得尝试；如果峰值来自参数、梯度或 optimizer state，应转向 dtype、LoRA、ZeRO 等方案。小 workload 可能看不出明显节省，不能用一次 CPU 运行推断 GPU 收益。实际策略比较由 [Part 02 · 76 激活检查点与 Offload 对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)负责。

## 典型误区

- checkpointing 省的是激活显存，不是参数显存和优化器状态显存。
- offload 不是 checkpointing 的另一种说法。
- 是否开启这两类方案，不能只看显存，还要看 wall time 和带宽代价。

## 对应来源

- [Part 02 · 19 激活检查点与 Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)
- [Part 02 · 42 激活 Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb)

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | checkpointing 的经典起点。 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | offload / partition / memory hierarchy 的系统起点。 |
| [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840) | 看 CPU offload 如何被纳入训练系统。 |
| [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) | 看 GPU / CPU / NVMe 分层如何继续扩展 offload 路线。 |

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint) | 看哪些张量可以通过重算从账本里拿掉。 |

## 阅读建议

- 先把 checkpointing 和 offload 区分开。
- 这页的重点是代价模型，不是 API 语法。

## 进入下一页

把策略带来的显存和时间变化带入 [05 梯度累积、训练闭环与 Profiling](./05_accumulation_decision_profiling.md)，统一 effective batch、step time 和验证口径。
