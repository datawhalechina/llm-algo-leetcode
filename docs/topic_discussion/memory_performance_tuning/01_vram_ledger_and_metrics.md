# 01. VRAM Ledger and Metrics | 显存账本与指标口径

## 页面目标

这一页回答两个问题：

- 显存优化为什么要先从资源账本开始，而不是直接缩 batch 或量化？
- `peak memory`、`reserved memory`、`throughput` 和时间代价应该怎么一起看？

## 问题起点

显存问题最容易被误判成“参数设太大了”。但真实系统里，峰值显存通常是多个资源对象叠加出来的：

- 参数与 optimizer state 常驻；
- activation 在训练前后向之间占据中间峰值；
- KV cache 会随上下文和 batch 继续增长；
- 框架和后端还会留下 buffer、workspace 和 fragment。

如果没有一份清晰账本，后续所有“显存优化”都会变成试错。

## 你要先确认什么

- 当前讨论的是训练显存、推理显存，还是两者混合。
- 峰值是来自参数、activation、optimizer state，还是 KV cache。
- 你要优化的是“装得下”，还是“装下以后还能跑得值”。

## 账本骨架

```text
total memory budget
  │
  ├─ parameters
  ├─ optimizer state
  ├─ gradients
  ├─ activations
  ├─ KV cache
  └─ framework / backend buffers
```

## 为什么账本先于技巧

显存优化的动作很多，但每个动作只改其中一部分对象：

- `checkpointing` 改 activation；
- `offload` 改 activation / optimizer state 的驻留位置；
- `ZeRO / sharding` 改参数、梯度和 optimizer state 的分摊方式；
- `KV cache paging / quantization` 改推理缓存的增长和表示；
- 量化可能改权重，也可能改 activation 或 cache。

所以如果账本没拆开，就不知道该动哪一个对象，也不知道副作用落在哪。

这也是 Task1 的收口：`Part01/01` 先确定 dtype 的字节数，`02` 提供参数规模，`03` 解释硬件和数据移动代价，`06` 再把参数、梯度和 optimizer state 放入单卡或分片账本。这个顺序建立的是预算假设，不是实际峰值报告；真实峰值要在 73、76 中测量。

## 指标口径

| 指标 | 含义 | 主要关联 |
|:---|:---|:---|
| `peak memory` | 运行中最高显存占用 | 是否装得下、batch 和上下文上限 |
| `reserved memory` | 框架已保留但未必正在使用的显存 | allocator 行为、碎片和 buffer |
| `memory delta` | 优化前后峰值差异 | 哪个手段真的省了资源 |
| `throughput / latency` | 时间代价 | 省显存是否把时间赔掉 |
| `fragmentation` | 空间碎片 | paging、buffer 组织和 allocator 压力 |

## 诊断框架

把显存问题先压成 4 个判断：

1. 是训练显存还是推理显存先顶到预算？
2. 是自然增长，还是中间状态、碎片和组织方式导致的浪费？
3. 当前动作是在省驻留对象，还是在搬运、重算或压缩？
4. 节省的资源是否值得对应的时间代价？

![VRAM ledger](/topic_discussion/memory_performance_tuning/vram_ledger.svg)

## 与 Task1-6 的关系

- `Task1-6` 负责学习顺序，告诉读者先读哪些 notebook。
- `01-06` 负责知识组织，把训练、推理和验证三条显存线收成同一套判断框架。
- 因此，这一页是诊断起点，不是目录索引。

## 文献锚点

- 单卡显存估算与 VRAM ledger 资料：帮助建立资源对象视角。
- PyTorch profiler / memory profiling 文档：帮助理解实测峰值和保留显存的区别。
- ZeRO / activation checkpointing 相关论文：帮助理解为什么账本要先拆成对象。

## 对应 Part 02

- `12` Gradient Accumulation
- `19` Activation Checkpointing and Activation Offload
- `22` vLLM PagedAttention
- `25 / 40 / 41 / 67` 量化与部署
- `73 / 74` 性能分析与端到端优化

这里要区分两类显存：`12 / 19 / 42` 主要讨论训练过程中的 activation、梯度节奏、重算和搬运；`22 / 24 / 34 / 37` 主要讨论推理过程中的 KV Cache、分页、复用和调度。账本相同，但对象生命周期和指标不同。

## 典型阅读入口

- [02 Training Memory Pressure](./02_training_memory_pressure.md)
- [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 本节要点

显存优化的第一步不是动技巧，而是先把峰值拆成对象、把对象放回同一张账本里。
