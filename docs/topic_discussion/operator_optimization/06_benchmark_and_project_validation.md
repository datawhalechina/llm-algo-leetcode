# Benchmark 与项目验证

项目验证链路是：参考实现 → 候选 kernel → 数值对齐 → 固定 workload → microbenchmark → 端到端 benchmark → 决策。结论至少区分 `accept`、`tune` 和 `reject`，并保留 GPU、dtype、shape、warmup、迭代次数和 backend 版本。

没有匹配 workload 或真实 trace 时，只能记录为优化假设。
