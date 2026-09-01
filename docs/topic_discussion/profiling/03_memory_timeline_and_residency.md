# 03 Memory Timeline 与 Residency

## 页面目标

本节负责解释 memory timeline 在 profiling 里扮演什么角色，以及它和显存优化中的预算决策有什么区别。

本节的输出是内存行为证据：峰值发生在哪个阶段、对象驻留多久、是否与时间热点重合。这里先定位原因，不直接承诺采用 checkpoint、offload 或量化。

## 问题起点

很多性能问题表面上是“慢”，但背后其实和内存行为有关：

- allocation 频繁震荡
- 某类对象驻留过久
- activation / cache 把阶段切换拖慢

## profiling 视角

在 profiling 里看 memory timeline，是为了回答：

- 哪个阶段内存突然抬高？
- 这次抬高是否和时间热点同步？
- residency pattern 是否说明系统在等内存行为？

这和显存优化里的“怎么压预算”不同。

## 读图顺序

先确定峰值出现的阶段，再确认对象是短暂 workspace、activation、KV Cache 还是 allocator 保留；最后比较它是否与时间热点重合。峰值大小回答“是否可能装不下”，驻留时长和分配频率则帮助解释“为什么会拖慢或产生碎片”。

| 观察 | 能支持的判断 | 还需要什么 |
|:---|:---|:---|
| allocated 峰值上升 | 活跃 tensor 的容量增加 | 对象来源和生命周期 |
| reserved 明显高于 allocated | allocator 保留或碎片可能存在 | snapshot / 分配记录 |
| activation 在 backward 前持续驻留 | 可能存在训练状态压力 | 计算图和 checkpoint 对照 |
| cache 随请求增长 | 可能是上下文或并发压力 | 请求分布与 cache policy |

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- [Part 00 · 18 显存分析与优化](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)
- [Part 00 · 19 激活检查点与 Offload](../../00_Prerequisites/19_Activation_Checkpointing_and_Activation_Offload.md)
- [Part 00 · 20 Profiling 与显存账本](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)

## 本节要点

profiling 里的 memory timeline 首先是证据，不是优化动作本身。

## 进入下一页

如果运行跨越多卡，继续进入 [04 通信等待与 Overlap](./04_communication_wait_and_overlap.md)；如果仍是单卡问题，则带着时间和内存证据进入 [05 Benchmark 设计与回归验证](./05_benchmark_design_and_regression_validation.md)。
