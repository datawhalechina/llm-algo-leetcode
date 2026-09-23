# 量化与压缩深入阅读

量化选型从资源约束开始，而不是从算法名称开始。一个已经可以运行的模型，可能面临权重驻留过大、激活峰值过高、KV Cache 挤压并发，或者低精度执行路径没有带来预期速度。不同问题对应不同的压缩对象、介入时机和验证方法。

整条阅读路径使用同一条因果链：

`约束定位 → 压缩对象 → 误差来源 → 介入时机 → 方法 / 格式 → backend 执行 → workload 验证 → 部署决策`

正文页负责解释单个节点，下面的段落负责说明这些节点如何连接。

## 图册与知识地图

本专题的图册由 walkthrough 承载，不再单独设置“视觉资产”阅读页。入口路线图帮助选择学习分支，知识地图帮助理解对象、误差、执行路径和证据之间的关系；局部机制图继续放在对应正文段落附近。

![量化路线：对象、时机、方法与部署证据](../../docs/public/topic_discussion/quantization/quantization_strategy_map.svg)

![量化知识地图：对象、误差、执行路径与证据](../../docs/public/topic_discussion/quantization/quantization_knowledge_map.svg)

路线图回答“下一步学什么”，知识地图回答“为什么这些概念相连”。它们不替代 01–06 正文的局部机制说明。

## 第一段：先定位约束和压缩对象

先判断问题来自权重、激活还是 KV Cache。权重量化主要改变模型常驻容量和权重读取带宽；激活量化影响中间状态、算子执行和数值稳定性；KV Cache 量化影响长上下文请求的 cache 容量、读写带宽和并发上限。

如果对象没有确定，直接比较 INT4、INT8 或 FP8 没有意义。还要明确目标是降低显存、降低带宽、提高并发，还是匹配某种硬件 kernel。

对应 [01 量化对象与误差直觉](./01_quantization_object_and_error.md)。该页建立 scale、zero-point、量化粒度和误差来源的共同口径。

![量化对象：权重、激活与 KV Cache](../../docs/public/topic_discussion/quantization/quantization_objects.svg)

## 第二段：理解误差如何进入系统

量化把连续的浮点值映射到有限的离散等级。scale、zero-point、分组粒度、异常值处理和校准数据共同决定映射误差。误差可能停留在权重层，也可能通过矩阵乘、激活或 KV Cache 读写累积到输出质量。

因此，bit width 只能说明表示范围和存储下界，不能单独说明任务质量或运行速度。需要把量化误差与目标评测集、输入分布和服务 workload 对齐。

这一部分连接 [01 量化对象与误差直觉](./01_quantization_object_and_error.md) 与 [02 PTQ 与 QAT 的介入时机](./02_ptq_and_qat_timing.md)。

## 第三段：决定量化介入时机

PTQ 适合已有模型的快速校准和部署试探；当 PTQ 的质量损失超过边界，且仍有训练预算时，才考虑 QAT 或量化微调。QLoRA 属于训练显存受限下的低比特适配路径，重点是减少可训练状态和维持任务质量，并不等于完成了推理侧权重量化。

选择时需要同时计算校准成本、训练成本、验证成本和部署收益。不能因为 QAT 理论上能恢复质量，就跳过 PTQ 基线；也不能因为 QLoRA 使用低比特基座，就把它的结果直接当作 GPTQ、AWQ 或 GGUF 的推理结果。

对应 [02 PTQ 与 QAT 的介入时机](./02_ptq_and_qat_timing.md) 和 [03 低比特训练适配](./03_low_bit_training_adaptation.md)。

![PTQ 与 QAT 的介入时机](../../docs/public/topic_discussion/quantization/ptq_qat_timing.svg)

## 第四段：区分方法、格式和执行路径

进入权重量化时，需要把三个层次分开：GPTQ / AWQ 是误差控制方法，INT4 / INT8 / NF4 / FP8 是数值表示，GGUF 是文件格式与部署封装；Transformers、bitsandbytes、vLLM、llama.cpp 和 TensorRT-LLM 则决定如何加载和执行。

同一种表示在不同 backend 上可能走不同的反量化和矩阵乘路径。模型文件变小，只能说明存储表示发生变化；是否减少运行时显存、是否使用低比特 kernel、是否提升吞吐，必须由目标 backend 验证。

对应 [04 权重量化与后训练压缩](./04_weight_only_compression.md) 和 [05 FP8 与 KV Cache 量化](./05_fp8_and_kv_cache_quantization.md)。

![权重量化与运行时低精度路径](../../docs/public/topic_discussion/quantization/weight_only_compression.svg)

![FP8 与 KV Cache 量化的路径分流](../../docs/public/topic_discussion/quantization/fp8_kv_cache.svg)

## 第五段：将候选放回真实 workload

部署比较应固定模型、硬件、backend、采样参数、输入长度、输出长度和并发，只替换量化候选。至少记录：

- artifact 是否能被目标 backend 加载；
- 使用的格式、量化方法和执行路径；
- 加载峰值、运行时显存和 allocator 保留量；
- prefill / decode、TTFT、TPOT、端到端延迟和吞吐；
- 固定评测集或任务指标上的质量变化；
- 失败原因、重复次数和适用边界。

CPU 实验可以验证公式、误差、格式字段和决策逻辑；真实 GPU 或 serving backend 才能验证 kernel、显存、延迟、吞吐、并发和兼容性。两类结果需要分开标注。

对应 [06 部署与 Benchmark 决策](./06_deployment_and_benchmark_decision.md)。

![量化知识地图：从对象到部署决策](../../docs/public/topic_discussion/quantization/quantization_knowledge_map.svg)

## 第六段：形成有条件的部署决策

最终判断不是“哪种量化最先进”，而是候选是否在指定约束下满足质量和资源门槛：

- `reject`：无法加载、质量不达标，或资源代价超过预算；
- `tune`：方向可行，但校准数据、粒度、backend、workload 或重复证据不足；
- `accept`：在固定模型、硬件、backend 和 workload 下，质量与资源指标均达标，并且收益具有重复性。

部署主线为 `66 浮点推理 baseline → 67 量化推理与部署`。如果 PTQ 后质量不足，才插入 `65 QLoRA 选择`，再生成新的模型或 artifact，回到 66 / 67 复验；65 不是所有量化部署的前置条件。40 和 41 负责机制补充，不能替代 67 的部署证据。

## 扩展边界：剪枝、蒸馏与稀疏化

剪枝、蒸馏和一般稀疏化共享“压缩—质量—部署证据”这条决策链，但改变的对象不同：剪枝改变结构或连接，蒸馏改变训练目标，稀疏化还要求 backend 具备匹配的执行路径。因此先作为扩展分支保留，待量化主线的 artifact、backend 和 benchmark 闭环稳定后，再决定是否新增专题正文和 Task。
