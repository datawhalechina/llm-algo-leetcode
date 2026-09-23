# 02 图结构与 Fusion 决策

## 页面目标

这一页回答的是：图结构、依赖和 fusion 边界如何决定潜在收益，以及为什么减少节点数量不一定降低真实执行成本。

本页的输出是图级候选：明确可融合边界、layout 和读写代价，再把候选交给 lowering 验证。

这一页负责把 graph optimization 的第一层拆清楚：哪些依赖、节点和中间张量真的值得改写或 fuse。

## 问题起点

图优化最容易被简化成“能 fuse 就 fuse”。  
但真正的问题是：

- 哪些中间结果真的贵
- 哪些依赖会让某种融合失去意义
- 哪些图级重排只是把成本换了个地方

## 关键观察点

- cost vector
- dependency structure
- intermediate tensors
- layout-sensitive fusion

## 进入下一页

进入 [03 Lowering、Legalization 与 Scheduling](./03_lowering_legalization_and_scheduling.md)，检查图级候选是否能被合法地下沉并保持收益。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- `09 AI Compilers and Graph Optimization`
- `19 Operator Fusion Introduction`
- `33 TCO and Cost Model`

## 本节要点

图级判断的重点不是“变少”，而是“变少之后执行成本是否真的下降”。
