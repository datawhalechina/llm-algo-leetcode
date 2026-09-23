# 编译与图优化正文

这页只做图优化问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 使用顺序

先判断图结构和 fusion 候选，再检查 lowering、schedule、layout 与 backend 约束，最后用 benchmark 验证成本模型。不要把“图能编译”或“节点变少”直接当成性能结论。

## 判断表

先分清问题在图结构、fusion、lowering、schedule、layout 还是 backend 成本模型，再判断 benchmark 差异是不是来自执行级约束。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 图没问题，但性能明显不对 | `graph vs execution mismatch` | [01](./01_why_compiler_and_graph_optimization_matters.md) | 先分清图级问题和执行级问题 |
| 看起来能 fusion，但收益不稳定 | `fusion boundary mismatch` | [02](./02_graph_structure_and_fusion_decisions.md) | 看依赖、layout、读写代价 |
| lowering 以后结果变差 | `lowering / schedule mismatch` | [03](./03_lowering_legalization_and_scheduling.md) | 看 legalize、schedule、kernel 组织 |
| 不同 backend 结果差很多 | `backend cost mismatch` | [04](./04_execution_model_and_backend_constraints.md), [05](./05_backend_cost_models_and_divergent_optima.md) | 看 layout、执行模型、成本模型 |
| benchmark 结论站不住 | `validation gap` | [06](./06_benchmark_and_project_validation.md) | 回到 workload 和 backend 约束一起验证 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| 图结构 | 依赖和 fusion 候选是否成立 | 图对就一定快 |
| lowering | 图到 kernel 的转换是不是合理 | lowering 只是翻译 |
| schedule | 执行顺序和 tile 组织是不是合适 | 只要 legal 就够了 |
| backend 成本模型 | 不同后端为什么会给出不同最优解 | backend 只是实现细节 |
| benchmark | 差异是不是在同一 workload 下成立 | 只看单次结果，不看约束一致性 |

## 本节要点

这页的职责不是再讲一遍编译术语，而是把图优化里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。

## 最小执行审计模板

记录 `图结构 -> fusion 候选 -> lowering/schedule -> backend 约束 -> workload 指标 -> 决策`。至少同时保留端到端延迟、吞吐、显存、编译成本和运行环境。
