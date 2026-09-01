# 06. Benchmark and Trade-off Decision | Benchmark 与取舍决策

## 页面目标

本节负责把前面的显存判断收束到收益验证：显存到底省了多少、时间赔了多少、最终值不值得留下来。

运行 73–76 项目的具体命令、结果文件和预算敏感性检查，参见[73–76 显存优化项目验证清单](../../verification/memory_projects.md)。

## 问题起点

如果没有验证页，显存专题很容易停在“某个技巧让峰值变小了”。但工程上真正有意义的问题是：

- 这次优化是否只是把显存从一个对象搬到另一个对象；
- 它是否把吞吐和延迟赔得过多；
- 它是否真的扩大了 batch、上下文或部署范围。

## 你要先确认什么

- workload 是否固定。
- baseline 和 candidate 是否只改一个变量。
- 峰值显存下降是否伴随吞吐、延迟或稳定性变化。

## 本节在路线中的位置

前面的页面负责回答“显存由谁占用、可以改什么、代价在哪里”；本节负责把答案变成可复查的项目决策。训练侧按 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) → [76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) → [75 Memory Budget Compression](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) 收集证据，最后由 [74 Profiling Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) 检查瓶颈解释是否成立。

## 为什么 benchmark 是最后一页

显存优化通常横跨训练、推理和部署三条线。只有在同一 workload、同一指标口径下比较，下面这些结论才有意义：

- checkpointing 是否值得；
- offload 是否只是把问题换成传输等待；
- KV cache quantization 是否真的让上下文或并发收益上来；
- 量化或 paging 是否把时间成本赔过头。

## 73–76 的实验步骤

四个项目不是连续改参数，而是逐层建立证据：

1. `73` 固定模型、batch、序列长度、dtype、warm-up 和迭代次数，先建立训练 step time、吞吐和峰值显存 baseline。
2. `76` 在同一 workload 下逐个比较 checkpoint、offload 和 hybrid；同时记录显存节省、吞吐损失、质量和 OOM 状态。
3. `75` 把显存上限、最低吞吐和质量下限写成预算，筛选可行方案，并改变阈值做敏感性分析；如果最佳候选只在一个窄阈值下成立，应标记为预算敏感。
4. `74` 用真实 profiling trace 检查显存变化是否真的对应 activation、搬运、重算或 kernel 瓶颈，而不是只看一个峰值数字；没有 trace 时只能报告证据缺口。
5. 每一步都保存配置、环境、候选结果和 `accept / tune / reject` 决策；只有同一 workload 下的对照才可以进入最终表格。

### 当前证据快照

当前训练侧报告显示，checkpoint 在高压力序列长度下约节省 332 MB 显存，但同时牺牲吞吐；因此 `76` 的结论应写成“在当前预算约束下可行”，而不是“普遍更优”。这只是当前报告的观察，不是跨硬件、跨 workload 的稳定规律。由于当前显存收益低于项目设定的 512 MB 显著收益阈值，`75` 的当前决策仍是 `tune`，不是 `accept`。`74` 尚缺真实 profiler trace，只能作为待补证据的收口页；对应报告见 `benchmarks/results/76_real_gpu_memory.json`。

## 判定原则

- `accept`：显存收益、吞吐/延迟和质量或稳定性约束同时成立。
- `tune`：方向对，但需要继续调 batch、调度、搬运或压缩参数，或者证据还不够稳定。
- `reject`：当前预算和 workload 下收益不足、副作用过大或质量不达标。

## 报告应该怎么写

一个合格的显存优化报告至少要同时说明：

- 你动的是哪一个资源对象；
- 峰值显存变化了多少；
- 吞吐、延迟和稳定性怎么变；
- 这次变化有没有扩大 batch、上下文或部署空间；
- 最终是继续保留、继续调优，还是换方案。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献与工程入口

- [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- [76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)
- [75 Memory Budget Compression Project](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md)
- [74 Profiling Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)
- [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)

## 典型阅读入口

- [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md)
- [03 Checkpointing and Offload](./03_checkpointing_and_offload.md)
- [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md)

## 项目结论

显存优化只有在 `73 -> 76 -> 75` 的训练侧证据成立，并经过 `74` 的真实 profiling 驱动端到端验证后，才算真正完成收口；没有 trace 时，结论必须保留为 `tune / profiling evidence missing`。
