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

## 与 Task0-6 的关系

- `Task0` 负责理解 autograd 和 backward 生命周期；`Task1` 才把这些状态放回 GPU 显存层级中。
- `intro` 负责学习顺序，告诉读者先读哪些 Notebook；本页负责解释账本为什么这样组织。
- `01-06` 负责知识组织，把训练、推理和验证三条显存线收成同一套判断框架。
- 因此，这一页是诊断起点，不是目录索引。

## 本页输出与下一步

读完本页，至少应能写出一份简化显存账本：参数、梯度、optimizer state、activation、KV cache 和框架 buffer 分别占什么位置，峰值出现在哪个阶段，以及下一步应该测什么。接着按 Task1 主线回看 [02 LLM Params and FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb)、[03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) 和 [06 VRAM Calculation and ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)；需要连接 attention 访存时，再补充 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)。

在 Part01 主线中，这个账本由 `01 dtype → 02 参数规模 → 03 硬件代价 → 06 状态分摊` 逐步建立。这里的账本是估算和诊断语言；真实峰值、吞吐和 OOM 边界仍由 73、76 和 74 验证。

## 文献锚点

- 单卡显存估算与 VRAM ledger 资料：帮助建立资源对象视角。
- PyTorch profiler / memory profiling 文档：帮助理解实测峰值和保留显存的区别。
- ZeRO / activation checkpointing 相关论文：帮助理解为什么账本要先拆成对象。

## 对应 Part 02

- [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
- [19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)
- [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
- [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)、[40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)、[41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)、[67 Quantized Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)
- [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)、[74 Profiling Driven Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)

## 典型阅读入口

- [02 Training Memory Pressure](./02_training_memory_pressure.md)
- [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 本节要点

显存优化的第一步不是动技巧，而是先把峰值拆成对象、把对象放回同一张账本里。
