# 03. 算子融合与 Kernel 组合

## 页面目标

本页解释 fusion 为什么可能减少端到端成本，以及为什么融合并不自动等于加速。判断重点是中间张量的生命周期、数据复用和执行资源之间的交换。

![Fusion：减少写回，也重新分配执行资源](../../docs/public/topic_discussion/operator_optimization/operator_fusion_lifecycle.svg)

## 融合改变了什么

| 变化 | 可能收益 | 可能代价 |
|:---|:---|:---|
| 减少 kernel launch | 调度开销下降 | 单个 kernel 变复杂 |
| 减少中间张量写回 | HBM 流量下降 | register / shared memory 压力上升 |
| 增加片上复用 | 数据搬运减少 | tile 和 layout 约束变强 |
| 合并多个阶段 | 端到端路径变短 | 并行度、可调优空间可能下降 |

## 组合决策

先画出算子依赖和中间张量，再判断哪些阶段共享输入、哪些阶段必须保留同步点。Norm、激活、Softmax、QKV 或 Attention 的融合，应分别说明语义契约、复用关系和边界处理。节点减少只能说明图结构变化，不能直接说明性能收益。

| 检查项 | 需要回答的问题 |
|:---|:---|
| 正确性 | 融合前后的输出和误差是否对齐？ |
| 访存 | 哪些中间结果不再写回全局显存？ |
| 资源 | register、shared memory 和 occupancy 是否恶化？ |
| workload | 哪些 shape、dtype、batch 或序列长度真正受益？ |

## 学习顺序

先阅读 Part 01 的算子融合基础，再用 Part 03 的 RMSNorm、Softmax 和 FlashAttention 实现观察不同组合。复杂图变换、IR 和 lowering 进入本专题的[图级优化与编译支撑模块](./graph_compiler/intro.md)。

## 本页出口

你应能为一次 fusion 提出可验证假设：减少了哪类搬运，增加了哪类资源压力，以及需要用哪些 kernel 和端到端指标验证。
