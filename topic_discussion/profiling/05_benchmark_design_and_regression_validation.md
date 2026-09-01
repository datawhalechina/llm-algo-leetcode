# 05 Benchmark 设计与回归验证

## 页面目标

本节负责解释：就算定位出了热点，也不能直接宣布优化成立，必须回到 benchmark 和回归验证。

本节的输出是可比较的 before / after 结果：固定 workload、环境、warmup 和统计口径，并同时记录收益、代价与波动。没有这一步，profiling 只能提供线索，不能提供项目结论。

## 问题起点

profiling 最容易出现的假结论有两类：

- 单次 trace 看起来好了，但 workload 不一致
- 某个指标变好了，但整体系统没变好

## 关键问题

- before / after 是否可比
- 请求分布或 batch 口径是否一致
- 波动是否可接受
- 回归是否可重复

## 最小对照表

| 字段 | baseline / candidate 必须保持一致吗 | 说明 |
|:---|:---:|:---|
| 模型、输入和 workload | 是 | 否则无法判断变化来自哪里 |
| 硬件、软件和精度 | 是 | 跨环境结果应单独分组 |
| 优化开关或目标策略 | 否 | 这是待比较的变量 |
| warmup、迭代次数和重复次数 | 是 | 避免冷启动和噪声误导 |
| 质量评测和失败处理 | 是 | 速度提升不能掩盖质量或 OOM 回归 |

报告至少同时保留均值、分位数或波动范围，以及失败状态；只保存最好的一次运行会高估收益。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)
- [Part 02 · 79 分布式并行基准测试](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb)
- [Part 02 · 66 推理性能比较](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)

## 本节要点

profiling 的结论必须回到 benchmark 才算完成，否则它只是一次局部观察。

## 进入下一页

将 benchmark 结果交给 [06 从诊断到行动决策](./06_diagnosis_and_action_decision.md)，判断是保留优化、继续采证、扩大实验还是回退。
