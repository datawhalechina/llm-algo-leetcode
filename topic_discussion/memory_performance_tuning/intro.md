# 旧显存路线迁移说明 | Legacy Memory Optimization Route

> 状态：纯迁移入口。旧正文已经停止维护并移除，本页只保留历史主题到当前路线的映射。

原“显存优化”专题把训练 activation、checkpoint / offload、推理 KV Cache、量化容量和分布式状态放在同一条路线中。它们仍然重要，但现在已按机制与工程目标迁入更明确的专题。

| 原主题 | 当前学习入口 | 适合解决的问题 |
|:---|:---|:---|
| 训练显存账本、checkpoint、offload | [反向传播与训练机制](../backpropagation_training_mechanism/intro.md) → [性能优化 Task2](../performance_optimization/intro.md) | activation、optimizer state、峰值与训练策略代价 |
| KV Cache 容量、分页与复用 | [推理优化](../inference_optimization/intro.md) → 性能优化 Task3 | 上下文、并发、TTFT / TPOT 与 cache 容量 |
| 低比特表示与容量 | [量化与压缩](../quantization_compression/intro.md) | 权重、激活、KV Cache 与部署 artifact 的精度 / 容量取舍 |
| 多卡状态分摊与通信代价 | [通信与并行](../communication_parallelism/intro.md) | ZeRO、并行切分、collective 与扩展效率 |
| 测量、trace 与回归 | [性能分析](../performance_optimization/methods/intro.md) → [性能优化](../performance_optimization/intro.md) Task5–6 | baseline、证据等级、候选对照与行动结论 |

旧 `01–07`、`walkthrough` 与 `casebook` 的有效内容已并入上表对应专题。新学习者和历史链接维护者都应从当前入口继续，不再向本目录增加正文、图册或案例页。
