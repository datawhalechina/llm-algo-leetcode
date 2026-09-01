# 成本模型与 Profiling

Autotune 是在候选配置中搜索，不是保证全局最优。Profiling 用于验证瓶颈假设，应把 kernel 时间、显存、编译成本和端到端指标放在同一 workload 下解释。

单次 trace 只能支持当前环境和 workload 的判断，不能直接推广到所有 GPU。
