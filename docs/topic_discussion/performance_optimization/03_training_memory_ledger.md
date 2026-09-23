# 03 显存账本与训练侧优化

## 本节定位

解释训练显存由哪些对象构成，并比较 Checkpoint、Offload、量化、梯度累积和分片策略的代价。可回看 [Part 02 · 43 统一训练内存账本](../../02_PyTorch_Algorithms/43_Unified_Memory_Management.md)，把专题中的账本字段与可运行实现对应起来。

## 内容安排

1. 显存对象：参数、梯度、优化器状态、Activation、workspace 和缓存。
2. 生命周期：对象何时创建、驻留、释放或转移。
3. 策略机制：Checkpoint 的重算、Offload 的搬运、累积与分片的容量变化。
4. 预算决策：显存收益、step time、吞吐、通信和质量门槛。

## 主要产出

一份训练显存账本，以及 baseline、candidate 和组合策略的对照结论。Checkpoint 与 Offload 的机制实现分别见 [19](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.md) 和 [42](../../02_PyTorch_Algorithms/42_Activation_Offload.md)。

## 从 Part 02 · 45 迁移的显存裁剪决策

原 [Part 02 · 45 通用预留](../../02_PyTorch_Algorithms/45_Reserved_45.md) 中的显存裁剪内容已迁移到本页，作为训练显存账本之后的决策环节：先识别峰值来源，再按收益、时间代价、实现风险和质量影响排序候选动作，最后判断是否接受、继续调优或停止扩展。

迁移后的报告应保留峰值来源、目标预算、候选裁剪顺序、预计峰值变化、实测 step time / throughput、质量门槛和失败记录。具体项目证据由 [Part 02 · 75 显存预算压缩项目](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) 承接；原 Notebook 仅保留迁移入口。
