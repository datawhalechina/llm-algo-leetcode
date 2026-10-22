# 性能分析（Performance Analysis）

> 专题类型：基础支撑　能力出口：把性能现象转化为可复查的测量、归因与回归证据

训练变慢、首词元延迟升高、显存峰值抬升或多卡扩展效率下降时，首先要确定变化发生在哪个阶段、由哪些对象造成，以及测量结果是否超过正常波动。本专题沿着“问题与指标 → 可重复基线 → 时间线 / 驻留状态 → 瓶颈归因 → 回归判断”建立取证能力。时间线回答时间在哪里暴露，驻留状态回答峰值时哪些对象同时存在；二者从同一基线出发，在瓶颈归因处汇合。

性能分析回答“时间和资源去了哪里”；具体策略的机制由训练、推理、量化、通信与算子专题讲解，候选选择、组合复测和工程决策进入[性能优化](../performance_optimization/intro.md)。

## 学习路线

按 Task 顺序学习时使用下表；需要连续完成一次排查时进入[性能分析问题链](./walkthrough.md)，已有现象或结果时使用[性能分析判断手册](./casebook.md)。术语差异查询[术语表](./glossary.md)，跨页核对单位、分母和公式条件时使用[公式索引](./formula_reference.md)。

![性能分析学习路线](../../docs/public/topic_discussion/performance_analysis/profiling_task_route.svg)

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 运行有效性与目标指标 | 当前性能问题对应什么有效运行条件和目标指标？ | 核心：[Part 00 · 19 异常定位](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb) → [Part 00 · 20 性能证据与实验设计](../../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb) | 排除无效运行 → 确定阶段 → 选择目标指标 → 固定质量门槛 | [专题正文 · 01 测量问题与证据链](./01_why_profiling_matters.md) |
| Task1 · 可重复基线 | 环境和工作负载怎样固定，才能区分真实性能变化与测量波动？ | 核心：[Part 00 · 17 性能分析基础](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb) | 记录运行身份 → 固定工作负载 → 预热与重复 → 保存原始样本与失败原因 | [专题正文 · 02 可重复基线](./02_reproducible_baseline.md) |
| Task2 · 时间线、输入与关键路径 | 总耗时由计算、启动、输入搬运、I/O、同步还是等待主导？ | 核心：[Part 01 · 07 CPU/GPU 异构调度](../../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb) → [Part 01 · 15 CUDA 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb) → [Part 01 · 17 CUDA 流](../../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb) → [Part 02 · PERF-TRACE 执行轨迹依赖与关键路径](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb)；扩展：[Part 02 · PERF-INPUT 训练输入流水线性能](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.ipynb) → [Part 02 · PERF-IO 产物、状态与输出 I/O 关键路径](../../02_PyTorch_Algorithms/PERF-IO_Artifact_State_and_Output_IO_Critical_Path.ipynb) | 继承同一基线 → 划分阶段 → 重建事件依赖 → 识别暴露等待 → 确认关键路径 | [专题正文 · 03 时间线、I/O 与关键路径](./03_time_breakdown_and_trace_reading.md) |
| Task3 · 对象驻留与内存剖析 | 哪些状态和缓冲区在峰值时同时驻留，它们何时创建、释放或复用？ | 核心：[Part 00 · 18 显存分析](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb) → [Part 02 · PERF-MEMORY-TIMELINE 显存驻留与峰值重叠](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb)；扩展：[Part 02 · 43 训练内存账本](../../02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger.ipynb) | 继承同一基线 → 记录分配事件 → 重建驻留区间 → 定位峰值对象 → 核对回收时刻 | [专题正文 · 04 显存时间线与驻留状态](./04_memory_timeline_and_residency.md) |
| Task4 · 资源上界与瓶颈归因 | 时间线和驻留证据指向哪个主约束：计算、带宽、容量、通信、I/O 还是排队？ | 核心：[Part 01 · 02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb) → [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) → [Part 01 · 13 性能分析与瓶颈定位](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb) → [Part 02 · PERF-ROOFLINE 算术强度与资源上界](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb)；扩展：[Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb) → [Part 01 · 20 NCCL 与 AllReduce](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb) → [Part 02 · 47 NCCL 通信性能分析](../../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.ipynb) | 汇总时间与驻留证据 → 对照经验上界 → 提出主约束 → 用判别实验证伪 | [专题正文 · 05 瓶颈归因](./05_bottleneck_attribution.md) |
| Task5 · 回归、噪声与阈值 | 当前配对结果支持回归通过、明显退化，还是继续补测？ | 核心：[Part 02 · PERF-REGRESSION 配对基准与回归阈值](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) | 对齐运行身份 → 配对采样 → 估计区间 → 检查守护条件 → 输出回归判断 | [专题正文 · 06 基准测试与回归验证](./06_benchmark_design_and_regression_validation.md) |
| Task6 · 诊断证据与行动交接 | 怎样把时间线、对象驻留、瓶颈归因和回归结果整理为可复查诊断，并交给下一轮优化？ | 核心：复用 Task0–5 的结果；项目：[性能优化](../performance_optimization/intro.md) | 汇总证据 → 记录反例与未知项 → 指定候选类别和复测条件 → 补测或交接优化 | [专题正文 · 07 从归因到行动](./07_diagnosis_and_action_decision.md) |

下图按观察对象、证据形式、资源约束和验证输出组织本专题知识。单次热点或单项指标变化只能形成候选解释，不能直接证明瓶颈归因成立。

![性能分析知识地图](../../docs/public/topic_discussion/performance_analysis/profiling_evidence_chain.svg)

## 扩展与跨专题入口

按需要解释的对象回查下表。已定位主约束、需要比较候选或组合策略时，进入[性能优化](../performance_optimization/intro.md)。

| 需要解释的对象 | 回补入口 | 性能分析使用的证据 |
|:---|:---|:---|
| 训练计算图、激活与优化器状态 | [反向传播与训练机制](../backpropagation_training_mechanism/intro.md) | 前向传播 / 反向传播 / 更新的时间线、状态驻留与峰值 |
| 请求阶段、KV Cache 与调度 | [推理优化](../inference_optimization/intro.md) | TTFT、TPOT、吞吐、并发、队列和状态容量 |
| 低精度、并行与计算内核 | [量化与压缩](../quantization_compression/intro.md)、[通信与并行](../communication_parallelism/intro.md)、[算子优化](../operator_optimization/intro.md) | 量化执行、并行等待、算子时间与机制前后对照 |

## 实验与项目出口

性能证据从轻量计时升级到执行轨迹、对象驻留、资源上界和配对回归。工具选择与升级条件见[正文 03 时间线、I/O 与关键路径](./03_time_breakdown_and_trace_reading.md)，结果路径、实际软件版本、硬件快照和验证状态登记在[实验资产表](./experiment_assets.md)及结果 JSON。首次运行按[实验 Quickstart](./quickstart.md)创建环境；跨专题 Profile 与隔离关系查询[环境资产索引](../../docs/gpu_environment_assets.md)。
