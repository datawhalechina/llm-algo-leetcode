# 算子优化判断手册

> 这份手册按性能现象选择排查方向，不替代 Task0–6 的顺序学习。先确认语义和 workload，再决定进入访存、融合、硬件执行还是 autotune。

统一判断链是：

```text
现象 → 语义 / 访存 / 执行对象 → 候选机制 → 证据层级 → accept / tune / reject
```

![算子优化知识地图：语义、数据路径、硬件执行与验证证据](../../public/topic_discussion/operator_optimization/operator_optimization_knowledge_map.svg)

## 按现象选择方向

| 现象 | 优先检查 | 可能机制 | 验证出口 |
|:---|:---|:---|:---|
| 输出 shape 对，但数值误差大 | dtype、mask、边界和累加路径 | 精度控制、边界 tile、参考实现对齐 | [02 Kernel 语义与访存](./02_kernel_semantics_and_memory.md) |
| kernel 能运行但明显变慢 | layout、合并访存、tile 和工作集 | memory coalescing、tile、shared memory | [02 Kernel 语义与访存](./02_kernel_semantics_and_memory.md) |
| fusion 后 kernel 变慢 | register、shared memory、occupancy | 减少写回与资源压力的权衡 | [03 算子融合](./03_fusion_and_kernel_composition.md) |
| 某张 GPU 快，另一张 GPU 慢 | dtype、shape、编译目标和硬件资源 | Tensor Core、warp、block、stream | [04 CUDA 执行](./04_cuda_execution_and_hardware_constraints.md) |
| autotune 结果不稳定 | shape 分布、warmup、测量口径 | 配置搜索、分桶、缓存和编译成本 | [05 成本模型与 Profiling](./05_cost_model_and_profiling.md) |
| kernel 变快但模型不变快 | launch、同步、其他算子或输入管线 | 端到端瓶颈不在当前 kernel | [06 Benchmark 与验证](./06_benchmark_and_project_validation.md) |

## 证据层级

| 证据 | 可以确认什么 | 不能确认什么 |
|:---|:---|:---|
| CPU correctness | 语义、shape、边界和误差 | GPU 带宽、真实 kernel 时间 |
| GPU microbenchmark | 固定 shape 下的 kernel 相对成本 | 模型或服务端到端收益 |
| profiling trace | 当前 workload 的热点和等待 | 所有 shape 都适用 |
| workload benchmark | 固定模型与输入下的系统变化 | 其他 GPU 或其他请求分布 |

## 常见决策

- `accept`：语义对齐，目标 workload 上有可重复收益，且编译和维护成本可接受。
- `tune`：方向合理，但 shape、tile、dtype、调度或证据还不稳定。
- `reject`：语义不对齐、收益不足，或额外资源与维护成本超过预算。

## 相关入口

- 想顺序学习：回到[算子优化专题入口](./intro.md)。
- 想沿一个 kernel 项目推进：阅读[算子优化问题链](./walkthrough.md)。
- 想定位系统热点：转到[性能分析](../profiling/intro.md)。
- 想理解图变换与 lowering：转到[图级优化与编译支撑](./graph_compiler/intro.md)。
