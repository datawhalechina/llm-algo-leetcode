# 06. Benchmark and Decision | 端到端基准与决策

## 页面目标

把前面识别出的候选机制放回同一 workload 和服务目标，比较它们是否真的值得保留，并形成可以复查的项目决策。

## 核心机制

`01–05、07、08` 负责解释瓶颈和候选动作，`06` 负责把它们放回同一套实验口径。公平比较要固定模型、backend、dtype、prompt tokens、generated tokens、batch、concurrency 和 cache policy，并确保 baseline 与 candidate 只改变一个主要变量。下图先展示从 workload 到决策的完整流程，表格再说明每个阶段需要产出什么。

![Benchmark 决策流程](../../public/topic_discussion/inference_optimization/benchmark_decision_zh.svg)

| 阶段 | 主要任务 | 关键输出 |
|:---|:---|:---|
| 固定 workload | 统一输入、请求分布和服务目标 | 可复现的实验条件 |
| baseline / candidate | 只改变一个主要变量 | 成对运行结果 |
| 指标与质量 | 采集 TTFT、TPOT、E2E、吞吐、P99、显存和质量 | 可比较的报告字段 |
| 策略决策 | 对照约束和证据等级 | `accept / tune / reject` 与下一步动作 |

一次运行得到的数字不能直接代表稳定结论。至少要区分预热、正式测量和汇总：预热用于排除首次装载影响，正式测量保留每次请求或每个 batch 的结果，汇总时同时报告中心趋势和尾部。不同 workload、不同硬件或不同 backend 的结果应分开记录，不要把它们混成一行平均值。

| 测量阶段 | 建议做法 | 主要检查 |
|:---|:---|:---|
| 预热 | 固定若干次 warmup，不纳入正式结果 | 模型加载、编译和 Cache 初始化是否完成 |
| 正式测量 | 固定 repeats，保留原始 JSON 或 CSV | 每次 TTFT、TPOT、E2E、显存和错误 |
| 汇总 | 报告 mean 或 median，并报告 P50/P95/P99 | 是否存在尾部抖动或异常值 |
| 对照解释 | 只改变一个主要变量，注明 evidence level | 收益是否可归因、能否复查 |

Benchmark 的结论还要绑定服务目标。在线交互、离线批处理和容量优先的系统，对同一组结果的判断可能不同；因此报告中应先写清约束，再解释指标，而不是把最高吞吐默认为最佳方案。

| 服务目标 | 主要约束 | 优先指标 | 常见决策偏好 |
|:---|:---|:---|:---|
| 交互式在线服务 | 首 token 快、尾延迟稳定 | TTFT、P99、超时率 | 即使吞吐略低，也可能 accept |
| 离线批处理 | 单位时间产出高、成本低 | throughput、tokens/s、成本/token | 可接受更高 TTFT |
| 长上下文服务 | Cache 可容纳、质量稳定 | peak memory、Cache 使用量、TTFT | 优先容量和复用收益 |
| 突发并发服务 | 负载上升时保持可用 | admission、P99、拒绝率、恢复时间 | 优先接纳控制和扩缩容 |

成本不能只写 GPU 小时。至少把实例数量、运行时长、有效输出 token、失败或重试请求和质量门槛放到同一份报告中，才能比较量化、Cache、调度或 backend 变化带来的真实收益。

| 成本字段 | 计算方式或记录方式 |
|:---|:---|
| 资源成本 | GPU 数量 × 运行时间 × 单卡小时成本 |
| 单位产出成本 | 资源成本 ÷ 有效输出 tokens |
| 失败代价 | 超时、拒绝、重试和质量不达标请求的比例 |
| 容量收益 | 同一 SLA 下可接纳的并发或 tokens/s |

## 判断框架

本节承接 `01–05、07、08` 的指标和机制判断。先明确服务目标：在线交互优先关注 TTFT / P99，离线批处理可能优先 throughput / cost；再检查报告是否记录下表字段。`accept` 表示当前约束下值得采用，`tune` 表示方向有效但证据或配置不足，`reject` 表示收益不足、代价过高或质量不达标。

运行开关、结果文件和 JSON schema 见 [66–70 推理项目验证清单](../../verification/inference_projects.md)；CPU 可先验证指标聚合和决策逻辑，真实服务指标仍需固定 workload 的 GPU backend。

| 类别 | 最小字段 |
|:---|:---|
| 条件 | model、backend、dtype、prompt tokens、generated tokens、batch、concurrency、cache policy |
| 性能 | TTFT、TPOT、E2E latency、throughput、P99、peak memory |
| 策略约束 | quality、acceptance rate、cache hit rate 或公平性 |
| 结论 | accept、tune、reject、下一步动作 |
