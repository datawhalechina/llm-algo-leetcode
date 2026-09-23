# 编译与图优化深入阅读

假设你手里有一张“逻辑上完全正确”的计算图，但它在不同 backend 上跑出来的性能差很多。接下来你会开始怀疑：问题是图结构本身，还是 lowering、schedule、layout、kernel 组织把结果改坏了。

这条线最重要的是按暴露顺序判断：先看图级问题，再看执行级问题，最后看 benchmark 结论是不是站得住。

对应专题正文：[01 为什么编译与图优化值得单独成章](./01_why_compiler_and_graph_optimization_matters.md)。先明确图到执行的完整边界。

## 第一段：先分清图对和跑得好不是一回事

故事通常从“图没错，但性能不对”开始。第一步要先分清：这是图结构、fusion 和依赖本身的问题，还是执行模型的问题。

这一步对应 [02 图结构与 Fusion 决策](./02_graph_structure_and_fusion_decisions.md)。

## 第二段：一旦进入 lowering，问题就不再只是语义等价

lowering、legalization 和 schedule 看起来像“把图翻译下去”，但真正改变结果的，往往是 tile、layout、kernel 组织和执行顺序。

这一步对应 [03 Lowering、Legalization 与 Scheduling](./03_lowering_legalization_and_scheduling.md)。

## 第三段：backend 差异不是噪声，而是主问题

同一张图在不同 backend 上差很多，通常不是偶然误差，而是成本模型、执行模型和实现边界不同。也就是说，backend 差异本身就是判断对象。

这一步对应 [04 执行模型与 Backend 约束](./04_execution_model_and_backend_constraints.md) 和 [05 Backend 成本模型与最优解分化](./05_backend_cost_models_and_divergent_optima.md)。

## 第四段：最后必须回到 benchmark

真正的收口不在“这个图优化听起来更先进”，而在 benchmark 是否证明这条 lowering / schedule / backend 路线更适合当前 workload。把这条故事走完以后，一个更像真实结论的说法通常不是“我们做了 fusion”，而是：图级判断成立，执行级代价也站得住，最终 backend 选择在同一 workload 下更优。

这一步对应 [06 Benchmark 与项目验证](./06_benchmark_and_project_validation.md)，并回到 `66 / 67 / 74` 项目页验证。
