# 性能分析实验资产表

本表登记诊断实验的固定输入、正式结果和复现条件。环境创建见[Quickstart](./quickstart.md)，单位与判断条件见[公式索引](./formula_reference.md)，公共结果规范见[结果资产说明](../../docs/part02_result_assets.md)。实际版本与硬件由结果 JSON 记录。

## 环境与实验入口

先按实验目标选择环境。CPU 用于核对计算与判断逻辑，单卡用于采集设备事件和峰值；系统追踪、多卡和服务指标使用对应工具或执行环境。

| 环境或工具层 | 直接入口 | 可以形成的记录 |
|:---|:---|:---|
| CPU 基础练习 | [Part 00 · 0E](../../00_Prerequisites/0E.md)、Part 01 资源与执行基础、机制页 Step 4 | 指标定义、事件依赖、驻留账本、资源参考与回归规则 |
| `PERF-CUDA-MECHANISM` | [创建与检查](./quickstart.md#perf-cuda-mechanism-quickstart) | 单卡执行轨迹、对象驻留、经验资源参考、局部时间和配对分布；四个本专题实验共用此 Profile |
| Nsight Systems / Compute | 目标机器上的外部采集 | 系统时间线、计算内核资源指标和暴露等待，保留采集命令与原始报告 |
| 多卡通信环境 | [Part 02 · 47](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.ipynb)与[通信实验资产](../communication_parallelism/experiment_assets.md) | 集合通信、最慢进程和扩展效率 |
| 服务环境 | [推理实验资产](../inference_optimization/experiment_assets.md) | 请求队列、缓存、TTFT、TPOT 和吞吐 |

## 机制实验

下列四项本专题实验共用 `PERF-CUDA-MECHANISM`。时间与依赖使用 TRACE，峰值使用 MEMORY-TIMELINE，资源参考使用 ROOFLINE；已有候选或版本对照时使用 REGRESSION。可直接阅读登记结果，重新采集时再完成环境和输入预检。结论保留各自测量范围，局部结果与项目采用分开判断。

| 实验 ID / 页面 | 固定工作负载 | 主要结果 | 当前结论 | 结果记录 |
|:---|:---|:---|:---|:---|
| [Part 02 · PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) | 2048 方阵；逐步同步对比末尾同步；10 次交替顺序配对 | 主机标注、执行轨迹与配对耗时 | RTX 5070 Ti：中位数 25.89 → 25.32 ms，延迟下降约 2.20%（1.0225×）；10 对中 5 对更快，仅作方向性证据 | [v3 摘要](../../benchmarks/results/perf_trace/perf-trace-critical-path_rtx5070ti_trace_v3.json) · [v3 轨迹](../../benchmarks/results/perf_trace/perf-trace-critical-path_rtx5070ti_trace_v3.trace.json) |
| [Part 02 · PERF-MEMORY-TIMELINE](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb) | 8192 × 2048 FP32；激活与工作区重叠 → 释放激活后重算 | 校验和、allocated / reserved 峰值与延迟样本 | RTX 5070 Ti：校验和检查通过，未记录逐元素误差；allocated 峰值下降约 33.3%，reserved 不变，延迟变为基线的 1.86 倍 | [v1 结果](../../benchmarks/results/perf_memory_timeline/perf-memory-peak-overlap_rtx5070ti_memory_v1.json) |
| [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) | 256 MiB FP32 设备复制、4096 参考 GEMM 与 2048 候选 GEMM | 经验带宽、经验计算吞吐、算术强度与候选效率 | RTX 5070 Ti：394.76 GB/s、20.04 TFLOP/s；模型指向计算侧约束。候选约为参考吞吐的 100.4%，该经验参考不是严格硬上界 | [v1 结果](../../benchmarks/results/perf_roofline/perf-empirical-roofline_rtx5070ti_roofline_v1.json) |
| [Part 02 · PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) | 1 / 32 / 256 Token、隐藏维度 2048；Eager / Compile 配对；允许 5% 延迟回归 | 首次调用、配对比例、自助区间、失败与逐负载决策 | RTX 5070 Ti：三档输出一致；依次为 `reject` / `tune` / `accept`，总决策 `reject`。256 Token 中位配对比例约 1.017，`accept` 仅表示未超过回归门槛，不表示加速 | [v3 结果](../../benchmarks/results/perf_regression/perf-paired-regression_rtx5070ti_regression_v3.json) |

输入与 I/O 的专项对照由性能优化专题登记；需要继续追查等待来源时使用下列交叉入口，不在此重复登记数值。

| 问题 | 交叉机制入口 | 固定与保留什么 |
|:---|:---|:---|
| 训练输入等待 | [Part 02 · PERF-INPUT](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.ipynb) | 固定批次、输入等待与主机—设备搬运时间线；复用单卡机制环境 |
| 产物、状态与输出 I/O | [Part 02 · PERF-IO](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb) | 分开采集阶段服务时间和复制＋计算流水线，保留各自测量范围 |

## 项目实验

诊断记录作为性能优化的输入，正式项目配置、结果和交接状态由[性能优化实验资产表](../performance_optimization/experiment_assets.md)登记。

| 问题 | 项目入口 | 从性能分析带入什么 |
|:---|:---|:---|
| 训练基线 | [Part 02 · 73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) | 模型、批量、长度、正确性、步骤范围与重复测量条件 |
| 候选复测与交接 | [Part 02 · 74 性能剖析驱动优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) | 基线、执行轨迹、驻留、当前归因及回归门槛；具体候选在项目内验证 |

## 结果版本与复现

机制表链接的是当前登记版本。读取与复现按下表区分；输入、后端、硬件或测量方法改变时，旧样本按原条件保留。

| 操作 | 使用什么 | 是否采集 GPU 数据 |
|:---|:---|:---|
| 阅读已有结论 | 打开机制表的 JSON；TRACE 使用同一行的相对轨迹链接，摘要内绝对路径仅对应采集机器 | 否 |
| 复现已登记对照 | 按对应机制页固定输入、受控变量和测量方法，完成预检后使用新的 `RUN_LABEL` | 是；新结果与旧记录分开保存 |
| 更换输入或执行条件 | 明确变化字段，重新采集匹配基线与对照，使用新标签 | 是；不同条件分别解释 |

公共身份、环境和失败字段遵循页首的结果规范；本专题额外核对下列采集条件。未采集的内容明确记录为缺失，不用零或其他测量层的数据代替。

| 分析对象 | 保留的记录 | 核对条件 |
|:---|:---|:---|
| 耗时与事件依赖 | 计时起止、同步位置、预热、逐次样本；采集时保留轨迹与工具配置 | 区分事件持续时间、重叠区间与端到端完成时间；采集耗时与独立计时分开 |
| 显存峰值与对象驻留 | 已分配与保留峰值、输出检查；解释峰值时补充对象创建、释放与同时刻记录 | 峰值数值不代替完整驻留时间线；两个独立最大值不能还原同时刻差值 |
| 资源参考 | 计算量、字节量、参考与目标形状、测量样本和精度 | 经验带宽和计算吞吐属于当前测量条件，不当作严格硬上界 |
| 配对回归 | 配对身份、运行顺序、原始样本、区间方法和水平、阈值与失败分母 | 回归通过和收益达标分别判断；相关样本按数据结构选择重采样方法 |

<details>
<summary>早期记录如何使用</summary>

以下摘要用于解释采集或工作负载的变化，不替代机制表中的当前记录。

| 实验 | 早期记录 | 保留原因 |
|:---|:---|:---|
| TRACE | [v1](../../benchmarks/results/perf_trace/perf-trace-critical-path_rtx5070ti_trace_v1.json)、[v2](../../benchmarks/results/perf_trace/perf-trace-critical-path_rtx5070ti_trace_v2.json) | 解释早期采集与标注差异，不作为稳定同步收益的结论 |
| REGRESSION | [v2 单负载](../../benchmarks/results/perf_regression/perf-paired-regression_rtx5070ti_regression_v2.json) | 与后续三档负载矩阵分别解释 |

</details>
