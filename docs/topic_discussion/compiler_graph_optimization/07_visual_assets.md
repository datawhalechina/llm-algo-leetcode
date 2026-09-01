# 07 Visual Assets

## 页面目标

这页负责收口编译与图优化的图册资产。第一阶段先固定图的职责，后续再补正式 SVG。

## 图册顺序

1. `graph_to_backend_chain`
- 从 graph 到 lowering、execution、benchmark 的总图

> 图示占位：Graph to Backend Chain 尚未生成。

2. `fusion_cost_map`
- 哪些中间张量和依赖让 fusion 真正有意义

> 图示占位：Fusion Cost Map 尚未生成。

3. `lowering_schedule_ladder`
- legalize -> lower -> schedule -> codegen 的链路图

> 图示占位：Lowering and Schedule Ladder 尚未生成。

4. `backend_constraint_map`
- CUDA / Triton / layout / launch 如何限制图级选择

> 图示占位：Backend Constraint Map 尚未生成。

5. `divergent_optima_board`
- 同图不同 backend 的最优解分化图

> 图示占位：Divergent Optima Board 尚未生成。

6. `compiler_benchmark_decision`
- keep / tune / switch backend / reject 的决策图

> 图示占位：Compiler Benchmark Decision 尚未生成。

## 当前状态

第一批和第二批图已补齐，当前图册已覆盖 `01-06` 的主要入口。
