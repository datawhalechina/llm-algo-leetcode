# 量化与压缩（Quantization and Compression）

> 专题类型：横向支撑　主服务目标：数值表示、质量、资源与执行路径取舍

## 页面导语

本专题面向需要判断“什么值得量化、量化误差从哪里来、低精度是否真正改善目标 workload”的学习者。学习重点是把量化对象、数值表示、校准统计、误差控制、产物契约、执行路径和最终证据连成一条可复查的机制链。

量化不是单一的“降低 bit 数”。权重、激活、训练状态和 KV Cache 位于不同生命周期，适用粒度、误差风险和执行条件也不同。同一套量化机制可以服务低资源训练、推理部署和性能优化，但三类 workload 必须分别验证。

当前 Task0–6 以量化为主线；剪枝、蒸馏和结构稀疏沿用同一套“压缩对象—质量代价—执行证据”框架，作为后续压缩扩展。本专题统一使用中文主称，首次出现时附英文：量化产物（artifact）、加载器（loader）、执行后端（backend）、核函数（Kernel）、工作负载（workload）和回退路径（Fallback）；配置字段与结果 JSON 保留原始英文键名。

想沿着一个问题连续学习时，先读[量化问题链：从数值表示到跨 Workload 决策](./walkthrough.md)；遇到质量、容量、加载或吞吐问题时，使用[量化问题排查手册](./casebook.md)定位原因。

## 如何开始

按知识顺序学习时，Task0–2 建立对象、表示、粒度、校准和误差的公共机制；Task3 进入低比特训练，Task4 进入推理运行时，Task5 检查产物与执行路径，Task6 用目标 workload 完成质量与性能决策。遇到 dtype、Tensor Core、KV Cache 或训练机制缺口时，再按需回补 Part 00 / Part 01。

路线图把公共机制、训练与推理分流、执行验证和证据出口放在一起。

![量化与压缩机制路线：公共机制连接训练、推理与性能](../../public/topic_discussion/quantization_compression/quantization_strategy_map.svg)

## 主学习路线与验证出口

下表按“要回答的问题 → 学习入口 → 验证出口”组织。先理解量化机制，再选择训练或推理分支；性能结论必须建立在目标 workload 的复测结果上。

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 对象、表示与证据语言 | 哪些状态可以量化，位宽、范围和 dtype 分别描述什么，局部误差与 workload 证据如何区分？ | 基础：[Part 00 · 05 Tensor](../../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md) → [Part 01 · 01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md) → [Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) | 对象与生命周期 → 数值表示 → 证据等级 | [01 量化对象、表示与误差](./01_quantization_object_and_error.md) |
| Task1 · Scale、Zero Point 与粒度 | 浮点范围怎样映射到整数网格；per-tensor、per-channel、per-group 的误差和元数据代价是什么？ | 核心：[Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) | 映射公式 → 粒度 → 反量化与执行条件 | [01 量化对象、表示与误差](./01_quantization_object_and_error.md) → [04 权重量化](./04_weight_only_compression.md) |
| Task2 · 校准、误差控制与 PTQ | 校准集怎样确定量化参数，权重、激活和层级敏感度怎样进入 PTQ 与混合位宽决策？ | 核心：[Part 02 · 52 校准与误差](../../02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.md) → [Part 02 · 40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) → [Part 02 · 53 SmoothQuant](../../02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant.md)；扩展：[QUANT-MIXED-BIT](../../02_PyTorch_Algorithms/QUANT-MIXED-BIT_Mixed_Bit_Allocation.md) | 校准/评测隔离 → 权重误差 → 激活异常值 → 混合位宽 | [02 PTQ 与 QAT](./02_ptq_and_qat_timing.md) → [04 权重量化](./04_weight_only_compression.md) |
| Task3 · 低比特训练与 QAT | 训练资源受限或 PTQ 质量不足时，QLoRA、QAT 与高精度回退分别解决什么问题？ | 核心：[Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md) → [QUANT-QAT](../../02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.md)；项目：[Part 02 · 65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md) | 低比特底座 → 可训练状态 → 量化误差适配 → 训练证据 | [03 低比特训练适配](./03_low_bit_training_adaptation.md) |
| Task4 · 推理运行时与状态量化 | FP8 计算、激活量化和 KV Cache 量化分别改变哪段执行路径、容量与延迟？ | 计算路径：[Part 01 · 03 GPU 架构](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) → [Part 01 · 12 Tensor Core](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md) → [Part 02 · 41 FP8](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)；状态路径：[Part 01 · 11 KV Cache](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) → [Part 02 · 54 KV Cache 量化](../../02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.md) | 激活/计算格式 → FP8 执行 → Cache 追加状态 → 长上下文证据 | [05 运行期量化路径](./05_fp8_and_kv_cache_quantization.md) |
| Task5 · Artifact、Loader、Kernel 与 Backend | 量化产物包含哪些元数据，加载后是否命中目标 kernel，何时会反量化或回退浮点？ | 机制：[40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) → [41 FP8](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)；项目：[82 量化产物评估](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.md) | artifact → loader → backend → kernel / fallback | [06 执行与 Benchmark 决策](./06_deployment_and_benchmark_decision.md) |
| Task6 · 跨 Workload 质量与性能决策 | 如何分别验证训练稳定性、推理质量、容量、延迟和吞吐，并把结果归因到量化机制或执行路径？ | 训练出口：[26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)、[65 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md)；推理出口：[66 baseline](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) → [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) → [83 Serving Profile](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.md) | 固定条件 → baseline/candidate → 质量与资源 → 归因与决策 | [06 执行与 Benchmark 决策](./06_deployment_and_benchmark_decision.md) |

![量化知识地图：对象、误差、执行路径与证据](../../public/topic_discussion/quantization_compression/quantization_knowledge_map.svg)

## 压缩扩展入口

Task0–6 先把量化机制和执行证据闭环。需要比较量化之外的压缩方法时，再进入[07 剪枝、蒸馏与结构稀疏](./07_pruning_distillation_and_structured_sparsity.md)：该页区分数值表示、连接/结构与教师—学生目标，并说明训练状态、Kernel 和部署证据如何变化。它是扩展正文，不是额外的 Task，也不表示剪枝、蒸馏已经具备与量化同等完整的 Notebook 实践。

## 按需回补：硬件、训练与状态基础

Task 表已经包含 Tensor、dtype、量化理论、训练状态和 KV Cache 等直接前置。只有需要继续解释结构压缩、底层执行或设备兼容性时，再回补下列内容。

| 补充主题 | Notebook 入口 | 建议时机 |
|:---|:---|:---|
| Tensor Core 深入 | [Part 01 · 23 Tensor Core 深入](../../01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.md) | Task4–5；需要解释累加精度和 kernel 条件时 |
| Attention 结构与 KV 表示 | [Part 01 · 04 Attention 显存优化](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md) | Task4；需要区分结构压缩、表示量化与缓存管理时 |
| 异构设备与 Runtime 兼容性 | [Part 01 · 34 异构加速器与 Runtime 约束](../../01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints.md) | Task5；需要检查 dtype、artifact、算子和 runtime 支持时 |

## 跨专题入口

需要沿训练质量与 adapter 继续学习时进入[后训练优化](../post_training_optimization/intro.md)；需要解释 TTFT、TPOT 和 Serving 行为时进入[推理优化](../inference_optimization/intro.md)；需要归因容量、带宽和关键路径收益时进入[性能优化](../performance_optimization/intro.md)与[性能分析](../performance_analysis/intro.md)；需要判断设备、runtime 和 fallback 时进入[部署与异构系统](../deployment_heterogeneous/intro.md)。

## 项目出口与稳定 ID

量化项目按三个独立角色组织，不要求按编号连续执行。

| 稳定项目 ID | 项目职责 | 当前承载 |
|:---|:---|:---|
| `QUANT-ARTIFACT-GATE` | 验证产物、manifest、loader 与质量门槛 | 82 |
| `QUANT-INFERENCE-DEPLOY` | 验证加载、backend 匹配与部署结果 | 67 |
| `QUANT-SERVING-PROFILE` | 验证服务配置、延迟、吞吐与容量 | 83 |

后续项目消费 `artifact_manifest`、`floating_baseline` 或 `deployment_record` 等语义资产，不要求结果必须由固定编号页面生成。

`PT-QLORA-SELECTION`（65）是后训练主归属的训练交叉入口，`INF-FLOAT-BASELINE`（66）是推理主归属的浮点参照。它们可以提供兼容证据，但不属于量化项目的强制执行顺序。

## 环境与验证

量化实验涉及数值机制、训练适配、产物生成和 Serving 执行，不能用一个库替代全部环节。

| 实验层级 | 默认技术栈 | 用途 |
|:---|:---|:---|
| CPU 机制 | PyTorch Tensor；`base.txt + torch-cpu.txt` | 映射、粒度、校准、误差和状态账本 |
| GPU 低精度与 PTQ | torchao、LLM Compressor、compressed-tensors；`torch-cu128.txt + quantization.txt` | weight-only、FP8、GPTQ、AWQ、SmoothQuant 与 artifact |
| 低比特训练 | Transformers、bitsandbytes、PEFT；`qlora.txt` | NF4 底座、LoRA 状态和 adapter 交付 |
| Serving backend | Transformers 固定输入；vLLM 为主、SGLang 对照 | 加载量化产物，验证 kernel、回退和服务指标 |

量化机制与环境验证复用稳定功能基线 `Qwen/Qwen2.5-0.5B-Instruct`，便于生成和复用 GPTQ、AWQ、GGUF 等量化候选；`Qwen/Qwen3-0.6B` 只作为新架构与执行后端兼容性候选。性能实验使用资产表登记的量化性能工作负载，同一实验组必须固定模型、tokenizer、输入和目标量化格式。

Transformers 负责生成固定输入参考和验证模型语义；vLLM 是量化 Serving 的主基线，SGLang 用于同口径 backend 对照。GGUF 候选交给 llama.cpp 的本地/端侧路径，TensorRT-LLM 只在 NVIDIA Engine、插件和版本矩阵明确后进入部署候选。专题页不分别维护这些工具的版本号，具体版本、固定方式和证据边界统一见 [模型、执行栈与环境资产表](../../gpu_environment_assets.md)。

| 契约 | 必须固定的字段 |
|:---|:---|
| 模型与 tokenizer | `model_id`、revision、原始 dtype、chat template |
| 校准与评测 | dataset / workload、revision、split、token 长度；两类输入相互隔离 |
| 配置与产物 | 对象、格式、bit、scale、granularity、group size、`artifact_id`、manifest、hash |
| 执行与输出 | hardware、backend、kernel/fallback、quality、memory、latency/throughput、failure、`evidence_level`、decision |

证据等级依次区分 `cpu_simulation`、`environment_preflight`、`local_mechanism_benchmark`、`artifact_generation`、`backend_benchmark` 与 `project_decision`；较低等级不能替代较高等级结论。环境组合与安装入口统一登记在 [模型、执行栈与环境资产表](../../gpu_environment_assets.md)。GGUF 与端侧执行进入[部署与异构系统](../deployment_heterogeneous/intro.md)，不在本专题重复维护环境说明。
