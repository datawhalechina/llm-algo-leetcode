# 量化与低比特适配（Quantization and Low-Bit Adaptation）

> 专题类型：横向支撑　主服务目标：精度、显存与部署取舍

## 页面导语

本专题面向需要判断“什么值得量化、什么时候量化、量化后能否真正部署”的学习者。学习重点是把量化对象、误差来源、介入时机、执行路径和最终证据连成一条可复查的决策链。

量化不是单一的“降低 bit 数”。权重、激活和 KV Cache 位于不同运行阶段，处理时机、误差来源和 backend 路径也不同。剪枝、蒸馏和一般稀疏化与量化共享压缩决策，但暂作为后续扩展，不混入当前核心 Task。

想沿着问题链连续阅读时，先看[量化与低比特适配深入阅读](./walkthrough.md)；想按现象快速分流时，使用[量化与低比特适配正文](./casebook.md)。

![量化路线：对象、时机、方法与部署证据](../../public/topic_discussion/quantization/quantization_strategy_map.svg)

## 如何开始

按知识顺序从 Task0 开始。路线图说明“下一步进入哪条路径”，知识地图说明“为什么对象、时机、方法和证据会连接起来”；两张图都由 walkthrough 承载，不再设置独立图册页。

量化的项目主线是 `66 浮点 baseline → 67 量化推理与部署`；如果 PTQ 后质量不足，再转入 `65 QLoRA 选型`，生成新的训练适配产物后回到 66 / 67 验证。

## 主学习路线与验证出口

`Task0–6` 指向 Part 01 / Part 02 的学习入口；专题正文负责解释概念、连接机制和整理判断条件，不替代 Notebook 的实现。

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 主要正文入口 |
|:---|:---|:---|:---|
| Task0 · 全局视野 | 压缩什么、何时介入、优化什么目标、需要什么证据？ | [Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [Part 01 · 12 Tensor Core 与混合精度](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md) | [01 量化对象与误差](./01_quantization_object_and_error.md) |
| Task1 · W8A16 基础 | 权重表示、反量化和显存收益如何判断？ | [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) | [02 PTQ 与 QAT 的时机](./02_ptq_and_qat_timing.md) |
| Task2 · PTQ 与 GPTQ/AWQ | 校准数据如何影响误差，artifact 如何产生？ | [Part 02 · 40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) | [04 权重量化](./04_weight_only_compression.md) |
| Task3 · QLoRA 适配 | 质量不足时如何用低比特底座完成训练适配？ | [Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md) → [Part 02 · 65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md)；输出 adapter / merged model manifest | [03 低比特训练适配](./03_low_bit_training_adaptation.md) |
| Task4 · 激活量化、FP8 与 KV Cache | 执行路径和缓存预算分别如何受低精度影响？ | [Part 01 · 03 GPU 架构](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) → [Part 01 · 12 Tensor Core](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md) → [Part 02 · 41 FP8 / KV Cache](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md) | [05 FP8 与 KV Cache 量化](./05_fp8_and_kv_cache_quantization.md) |
| Task5 · artifact 与 backend | 文件能否加载，实际是否走目标执行路径？ | 先读取可选的 [65 QLoRA manifest](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md)，再用 [66 浮点 baseline](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) → [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) | [06 部署与基准测试决策](./06_deployment_and_benchmark_decision.md) |
| Task6 · benchmark 与决策 | 固定 workload 下，候选是否值得保留？ | [66 浮点 baseline](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) 作为 G0 → [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) 完成 G1/G2；质量不足时回到 [65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md) | [06 部署与基准测试决策](./06_deployment_and_benchmark_decision.md) |

这条路线先用 Task0 建立共同口径，再把 weight-only、训练适配、运行时低精度和部署证据分开处理。`adapter` 能加载只能证明训练产物可用，不能替代量化 artifact 在目标 backend 上完成 workload 的证据。

### 65–67 项目闭环

三个项目不是三次重复 benchmark，而是三个连续的证据节点：

| 项目 | 负责什么 | 输出给下一步的内容 |
|:---|:---|:---|
| 65 · QLoRA 选型 | 判断低资源训练适配是否值得继续，并生成 adapter / merged model manifest | 模型 revision、adapter 配置、质量结果和 artifact 路径 |
| 66 · 浮点推理 baseline | 在不引入量化 artifact 的前提下，固定模型、backend 和 workload | G0 浮点结果、统一指标和 baseline contract |
| 67 · 量化推理与部署 | 加载 GPTQ/AWQ/GGUF 等真实 artifact，与 66 的 G0 对照 | kernel/backend 证据、性能质量结果和 accept / tune / reject |

因此，65 的 adapter 不能直接当作 67 的量化权重；若训练适配后仍需部署，必须先确认 artifact 转换，再回到 66/67 重做同一 workload 的对照。

## Task0：量化全景与证据口径

Task0 不要求学习者马上选择 INT4、INT8 或 FP8，而是先建立后续路线共用的判断框架：

| 维度 | 需要回答的问题 | 后续连接 |
|:---|:---|:---|
| 压缩对象 | 是权重、激活，还是 KV Cache？ | 25、41 |
| 介入时机 | 是已有模型的 PTQ，还是需要 QAT / QLoRA？ | 26、40、65 |
| 优化目标 | 主要降低显存、带宽、延迟，还是提高并发？ | 66、67 |
| 执行路径 | 文件格式、loader、backend 和 kernel 是否匹配？ | 40、41、67 |
| 证据等级 | 是公式模拟、GPU 探针、load smoke，还是固定 benchmark？ | 66、67 |

Task0 的交付物是一张“对象—时机—目标—证据”判断表：给定一个量化需求，能够说明先看哪一节、需要什么实验、什么结果才足以支持 `accept / tune / reject`。路线图负责导航，知识地图负责解释关系，[01 量化对象与误差](./01_quantization_object_and_error.md)负责建立公式和误差的共同语言。

![量化知识地图：对象、误差、执行路径与证据](../../public/topic_discussion/quantization/quantization_knowledge_map.svg)

## 按需回补

| 补充主题 | 学习入口 | 建议时机 |
|:---|:---|:---|
| 量化对象与误差 | [Part 01 · 21](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) | Task0 |
| Tensor Core 与 dtype | [Part 01 · 12](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md) | Task0 或 Task4 |
| W8A16 | [Part 02 · 25](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) | Task1 |
| GPTQ / AWQ | [Part 02 · 40](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) | Task2 |
| FP8 / KV Cache | [Part 02 · 41](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md) | Task4 |

## 跨专题入口

- [推理优化](../inference_optimization/intro.md)：请求阶段、Serving 和 backend 行为；
- [显存优化](../memory_performance_tuning/intro.md)：参数、激活、优化器状态和 KV Cache 账本；
- [后训练优化](../post_training_optimization/sft_foundation/intro.md)：QLoRA、LoRA 和训练适配；
- [性能分析](../profiling/intro.md)：kernel、显存、通信和端到端证据归因。

## 项目收口

最终报告区分训练适配产物和推理量化产物，并记录模型 revision、校准数据、量化配置、dtype、硬件、backend、kernel、workload、质量、显存、TTFT / TPOT、吞吐、重复次数和证据等级。用 `accept / tune / reject` 表达决策。

剪枝、蒸馏和稀疏化先作为扩展方向：它们分别改变结构、训练目标或执行路径，待当前量化主线的 artifact—backend—benchmark 闭环稳定后，再决定是否新增正文和 Task。
