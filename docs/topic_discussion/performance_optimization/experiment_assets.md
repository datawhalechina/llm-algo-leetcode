# 性能优化实验资产表

本表登记单卡机制对照、训练项目结果和交付输入。环境操作见[实验 Quickstart](./quickstart.md)，结果字段说明见[公共结果规范](../../part02_result_assets.md)。每项结论以所链接 JSON 的输入、运行版本和测量范围为准。

## 环境与实验入口

已有性能分析单卡环境可直接复用；读取项目结果时无需重新执行 GPU 实验。下表区分实际测量与读取结果生成决策两类操作。

| 环境 | 实验入口 | 运行条件 |
|:---|:---|:---|
| [BASE-CPU](../../guide.md#base-cpu) | 各机制页 Step 4 的题目与测试 | 按页面说明安装基础依赖 |
| [PERF-CUDA-MECHANISM](../performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart) | PERF-INPUT、PERF-IO、PERF-CANDIDATE、PERF-COMPOSITION | 单卡至少 8 GiB 显存，环境预检通过 |
| 训练项目原环境 | [Part 02 · 73 训练基线](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)、[Part 02 · 76 显存策略](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)、[Part 02 · 74 性能分析与交接](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) | 按各页 5.1 与固定模型输入准备；保留结果中的 PyTorch 版本 |
| [PERF-CUDA-MECHANISM](../performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart) | [Part 02 · 75 AdamW 执行路径](../../02_PyTorch_Algorithms/75_Optimizer_Execution_Path_Benchmark.md) | 单 GPU；PyTorch CUDA；无需下载模型或数据 |
| [PERF-DELIVERY](./quickstart.md#perf-delivery-quickstart) | [Part 05 · 性能优化交付](../../05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/index.md) | CPU；已有 Part 02 · 74 交接包及其引用文件 |
| 其他候选原环境 | [推理优化](../inference_optimization/quickstart.md)、[量化与压缩](../quantization_compression/quickstart.md)、[通信与并行](../communication_parallelism/quickstart.md) | 使用目标后端与对应设备条件，测量结果需保留原始样本和失败记录 |

## 机制实验

当前四项记录均来自 RTX 5070 Ti Laptop GPU 的局部实验。输入流水线与 I/O 使用构造数据；候选和组合使用随机初始化的 SwiGLU MLP，隐藏维度 2048、中间维度 5504，每次处理 8 批输入。它们用于比较输入供给、浮点精度和搬运重叠，不能直接推出完整模型训练质量、服务性能或低比特量化收益。

下表的“延迟下降”按“1 − 候选中位数 / 对照中位数”计算，“加速比”按“对照中位数 / 候选中位数”计算。候选和组合的活动字节指参数与输入缓冲区；同一矩阵中多组模型共同驻留，其分配峰值不能当作每个候选独立运行的峰值。

关键路径、显存驻留、资源参考和配对回归的正式记录统一见[性能分析实验资产表](../performance_analysis/experiment_assets.md)。使用这些结果提出候选时，应核对输入、设备和运行版本；其收益不能与本表实验直接相加。候选矩阵中的复制与计算分阶段计时也不能单独证明优化后的端到端瓶颈已经转移，仍需执行轨迹或判别对照。

| 实验 / 页面 | 固定输入与受控变量 | 当前结果 | 结论与限制 |
|:---|:---|:---|:---|
| [PERF-INPUT · 输入流水线](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.md) | 批量 8、长度 128、特征维度 512；预热 10 步、测量 80 步；比较普通内存、锁页异步复制、双进程预取 | [input_v1](../../../benchmarks/results/perf_input_pipeline/perf-input_rtx5070ti_input_v1.json) | 锁页单进程步骤中位数最低：0.379 ms；双进程预取 P95 最低：0.488 ms，吞吐最高：20302.65 样本/秒。原始决策为接受预取候选，但不同指标各有最优配置；正确性仅以校验和核对，未保留逐步样本 |
| [PERF-IO · I/O 关键路径](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.md) | 64 MiB 本地产物分阶段测量；8 批复制与计算比较串行和双缓冲；预热 5 次、测量 20 次，本地热缓存 | [io_v2](../../../benchmarks/results/perf_io_critical_path/perf-io_rtx5070ti_io_v2.json) | 复制与计算中位数 7.066 → 4.965 ms，加速比 1.423×；保存与加载、输出回写不包含在该加速比中。产物往返检查和流水线校验和检查通过；未验证远端存储或冷缓存 |
| [PERF-CANDIDATE · 约束到候选](../../02_PyTorch_Algorithms/PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md) | 每批 32 / 256 Token；FP32 串行对比 BF16 串行、FP32 双缓冲；预热 12 轮、轮换顺序测量 24 轮 | [candidate_rotating_v2](../../../benchmarks/results/perf_candidate/perf-candidate_rtx5070ti_candidate_rotating_v2.json) | BF16 两档加速比 2.651× / 2.876×，参数与输入缓冲区字节减半；FP32 重叠在 32 Token 下为 `tune`，256 Token 达到 1.03× 加速与 P95 门槛。输出余弦相似度通过 0.999 门槛；这不是训练或任务质量评测 |
| [PERF-COMPOSITION · 策略组合](../../02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.md) | 同一轮比较 FP32 串行、BF16 串行、FP32 重叠、BF16 重叠；尺寸与预热、重复设置同候选实验；使用本次矩阵内的单项对照 | [composition_rotating_v2](../../../benchmarks/results/perf_composition/perf-composition_rtx5070ti_composition_rotating_v2.json) | 相对最佳单项 BF16 串行，组合中位延迟下降 5.57% / 12.16%；两档均为 24/24 轮更快，P95 比值为 0.727 / 0.893。局部规则选择保留组合，回退目标为 BF16 串行；输出相似度由单独前向核对，未证明完整重叠执行的逐元素一致性或项目可采用 |

## 项目实验

下列结果使用 Qwen2.5-0.5B-Instruct，保留各文件原有的 PyTorch 2.8.0+cu128 快照，与上方 PyTorch 2.11.0+cu128 的局部 MLP 测量分开解释。AdamW 完整训练与无动量 SGD 激活隔离实验分别判断；局部显存节省不能提升为完整训练采用结论。

| 项目入口 | 直接输入与比较对象 | 正式结果与交接 | 当前登记结论 |
|:---|:---|:---|:---|
| [Part 02 · 73 训练性能基线](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) | 批量 1、长度 768、AdamW；FP32 与 BF16 混合精度；3 次重复 | [训练基线](../../../benchmarks/results/73_real_gpu_training_rtx5070ti_training_baseline_v1.json) | `tune`：时间改善，显存收益未达到门槛；为后续候选提供训练对照 |
| [Part 02 · 76 重算、卸载与预算敏感性](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) | 同标签 G0 为 AdamW 固定长度对照；G1 改用无动量 SGD，逐档比较激活显存；`close` 对 G0 实测候选重算预算情景 | [G0](../../../benchmarks/results/76_real_gpu_memory_rtx5070ti_activation_memory_v1.json)、[G1](../../../benchmarks/results/76_real_gpu_memory_boundary_rtx5070ti_activation_memory_v1.json)、[输入清单](../../../benchmarks/results/76_input_manifest_rtx5070ti_activation_memory_v1.json)、[并入前的 75 历史预算记录](../../../benchmarks/results/75_memory_budget_decision_rtx5070ti_memory_budget_v1.json) | G0 为 `tune`。G1 在已测长度 1280 下，checkpoint 节省约 2820 MiB、吞吐比为 0.831，达到激活隔离实验门槛；更长序列因安全边界停止，证据仍标记为 `blocked`。74 v7 仅将其登记为激活显存范围内的 `accept`，不外推为完整训练采用。历史 75 记录保留，预算情景与策略实验在 76 收口 |
| [Part 02 · 75 AdamW 执行路径基准](../../02_PyTorch_Algorithms/75_Optimizer_Execution_Path_Benchmark.md) | RTX 5070 Ti Laptop GPU；PyTorch 2.11.0+cu128；固定 AdamW 与学习率；比较 single-tensor、foreach、fused；两种等元素数布局及完整 MLP 训练步；3 次重复 | [原始测量](../../../benchmarks/results/75_optimizer_path/optimizer_path_rtx5070ti_adamw_path_v1.json)、[决策登记](../../../benchmarks/results/75_optimizer_path/optimizer_path_decision_rtx5070ti_adamw_path_v1.json) | `accept`：本 workload 下选 `fused`，完整训练步加速 1.691×，分配显存减少 7.938 MiB，最大参数误差 1.79e-7。只适用于该合成 MLP、GPU 与 PyTorch 组合；优化器单步计时仅作诊断 |
| [Part 02 · 74 性能分析与交接](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) | Profiler 摘要、激活专项及同工作负载训练证据；分别判断完整训练与激活隔离范围 | [报告 v7](../../../benchmarks/results/74_profiling_optimization_rtx5070ti_profiling_dual_scope_v7.json)、[交接包 v7](../../../benchmarks/results/74_profile/performance_handoff_rtx5070ti_profiling_dual_scope_v7.json) | 顶层 `tune`；专项为 `accept_for_activation_memory_scope`，且明确 `promotable_to_full_training=false`，不能登记为完整训练采用 |
| [Part 05 · 性能优化交付](../../05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/index.md) | 读取 Part 02 · 74 交接包及目标工作负载的单项、组合、重新分析和回退记录 | 未登记正式交付结果 | 现有局部机制收益与限定范围结论不足以填入最终 `accept`；需按项目门槛检查直接输入 |

## 结果版本与复现

每项实验以本页链接的完整标签定位当前登记，不能仅凭文件名末尾的“v2”选择结果。重新运行会形成新的硬件与运行时快照；即使模型或形状未变，也不把新运行结果拼入旧对照。复现顺序为环境检查、固定输入、基线与单项对照、组合对照、范围核对及项目判断；候选和组合各自保留本次矩阵内的对照。

| 操作 | 输入与处理 | 输出与停止条件 |
|:---|:---|:---|
| 阅读当前结果 | 打开表中 JSON，核对工作负载、版本、测量范围及原始决策 | 无需运行 GPU；缺失值保持缺失，不用历史值补齐 |
| 复现机制实验 | 按机制页 5.1 设置环境与输入，使用新的运行标签；保留预热、重复和轮换顺序 | 生成新 JSON；预检失败或正确性、资源检查失败时停止后续采用判断 |
| 复现项目实验 | 按项目页阶段执行，核对输入清单与来源文件 | 同一范围内比较；保存失败与安全停止状态，不把专项通过改成完整训练通过 |
| 检查结果完整性 | 核对逐次样本、样本数、分位数算法、分阶段范围、原始决策与回退目标；当前候选、组合和 I/O 流水线的 P95 使用排序后索引 floor(0.95 × (n − 1)) 的样本值 | PERF-INPUT 仅有汇总，不能回算逐次分布；矩阵峰值与活动字节分开解释 |
| 处理本地产物 | PERF-IO 会生成运行目录中的 `artifact.pt` 和 `output.pt` | 当前计时与结论可由 JSON 阅读；重新测量时生成新产物，不能用缺失产物宣称内容复验完成 |

<details>
<summary>历史结果：比较协议与判断条件的变化</summary>

历史记录保留当时输入和决策，不与当前矩阵拼接成收益。

| 历史记录 | 与当前记录的差异 | 阅读限制 |
|:---|:---|:---|
| [IO v1](../../../benchmarks/results/perf_io_critical_path/perf-io_rtx5070ti_io_v1.json) | 只有保存、加载、复制与输出回写的分阶段汇总；v2 增加完整复制与计算流水线计时 | 分阶段时间不能直接作为流水线加速比 |
| [候选初版](../../../benchmarks/results/perf_candidate/perf-candidate_rtx5070ti_candidate_bf16_overlap_v1.json) | 初版按候选分组测量；当前版本增加轮换顺序、P95 门槛，区分活动字节与矩阵峰值 | 旧余弦字段个别值略大于 1，属于数值计算问题，保留原值但不能解释为相似度更高 |
| [组合初版](../../../benchmarks/results/perf_composition/perf-composition_rtx5070ti_composition_bf16_overlap_v1.json) | 初版 256 Token 组合未优于最佳单项；当前协议增加轮换顺序与尾延迟检查 | 条件改变后的新结果不能改写初版 `keep_single` |
| [组合过渡版](../../../benchmarks/results/perf_composition/perf-composition_rtx5070ti_composition_bf16_overlap_v2.json) | 标签含 v2，但工作负载仍为 v1、预热 5 轮、测量 20 轮；当前正式记录预热 12 轮、测量 24 轮 | 不能只根据标签后缀将两份结果视为同一协议 |

</details>
