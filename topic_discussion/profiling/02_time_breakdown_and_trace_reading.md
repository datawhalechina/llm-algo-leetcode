# 02 时间拆分与 Trace 阅读

## 页面目标

本节负责把“为什么慢”先拆成可观察的时间问题：operator、kernel、launch、等待和阶段切换。

本节的输出是时间瓶颈假设：问题主要属于计算、调度、启动开销还是等待。只有当时间线显示出内存驻留或分配行为时，才进入下一节。

## 问题起点

“慢”不是一个统一现象。常见情况包括：

- 单个 operator 真慢
- kernel 数量太碎
- Python / runtime 开销高
- 某些阶段有明显等待

如果不先做时间拆分，后面的优化动作常常会打偏。

## 关键观察点

- operator breakdown
- kernel timeline
- launch overhead
- step 内阶段切换
- CPU / GPU overlap

## 时间线如何解释

`operator` 是框架层调用，`kernel` 是设备上的执行单元，`launch` 是提交开销，空白区间可能表示同步、数据准备或资源等待。它们出现在不同层级，不能把 operator 名称直接当成 kernel 瓶颈。

阅读 trace 时先按阶段切片，再比较同一阶段的总时长、调用次数和空闲区间；需要进一步归因时，再结合 shape、输入管线和 CUDA Event。单个 kernel 的耗时不能替代完整 step 或请求的端到端耗时。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- [Part 00 · 17 Profiling 基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)
- [Part 00 · 20 Profiling 与显存账本](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb)
- [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)

## 本节要点

时间拆分是 profiling 的第一入口，因为大多数问题都先表现为“某一段时间不合理”。

## 进入下一页

若时间热点伴随显存抬升或分配震荡，进入 [03 Memory Timeline 与 Residency](./03_memory_timeline_and_residency.md)；否则保留时间假设，直接准备 benchmark 对照。
