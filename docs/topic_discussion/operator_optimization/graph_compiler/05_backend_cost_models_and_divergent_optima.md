# 05 Backend 成本模型与最优解分化

## 页面目标

这一页负责解释为什么同一张图在不同 backend 上会产生不同最优解，以及成本模型为什么必须进入判断。

本页的输出是后端比较假设：明确差异来自算子支持、访存、launch、编译成本还是 runtime 调度，而不是简单归因于“某个 backend 更快”。

## 问题起点

如果只看语义，很多方案都“正确”。  
真正让方案分化的是：

- 目标硬件不同
- 带宽、寄存器、缓存和 launch 成本不同
- 编译器和 runtime 的假设不同

## 关键观察点

- backend cost model
- hardware-specific assumptions
- TCO and deployment constraints
- divergent optima

## 进入下一页

进入 [06 Benchmark 与项目验证](./06_benchmark_and_project_validation.md)，用同一 workload 验证成本模型是否预测了真实结果。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- `33 TCO and Cost Model`
- `09 AI Compilers and Graph Optimization`
- `32 TVM MLIR Deep Practice`

## 本节要点

backend 差异不是噪声，而是“为什么最优解会分化”的主因。
