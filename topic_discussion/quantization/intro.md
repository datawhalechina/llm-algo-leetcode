# 量化与压缩（Quantization and Compression）

> 专题类型：横切支撑　主服务目标：精度、显存与部署取舍

## 专题定位与 Infra 层定位

本专题串起量化主线：先看低比特表示到底在压什么，再看 PTQ、QAT、GPTQ、AWQ、GGUF、FP8、KV Cache Quant 分别在什么时机介入，最后把收益收回推理、显存和部署约束中的最终选型。这里要区分方法与格式：GPTQ / AWQ 主要是后训练权重量化方法，GGUF 主要是量化权重的文件格式与部署封装。量化是跨层方法轴，主要落在 Infra-L2–Infra-L4：Infra-L2 关心低比特算子、kernel 和硬件支持，Infra-L3 关心训练、校准和量化配置，Infra-L4 关心量化模型在推理引擎中的加载、KV Cache 和服务吞吐，Infra-L1 提供显存容量与带宽边界。

模型权重、激活和 KV Cache 是被压缩的负载，不单独构成 Infra 层；Infra-L5 只在模型注册、版本发布、评测回归和资源治理等部署流程中介入。若问题已经明确是请求速度或显存预算，应转到推理优化或显存优化。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的量化与部署路线进入，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 65、66、67 等项目节。量化专题是方法选择支撑线，可以按当前约束切入，不必从 PTQ 到部署完整顺读。

## 前置阅读

建议先掌握 [Part 01 · 21 量化理论与 INT4/INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) 的量化理论与 INT4/INT8 基础，再根据目标补读 [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)、[Part 02 · 40 GPTQ/AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) 和 [Part 02 · 41 FP8 与 KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)。如果重点是训练显存，应同时回看监督微调与显存专题；如果重点是服务吞吐，应先明确推理后端和 workload。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 量化基础与硬件直觉 | [Part 01 · 01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb) → [Part 01 · 12 Tensor Core 与混合精度](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb) → [Part 01 · 21 量化理论与 INT4/INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) | [01 量化对象与误差](./01_quantization_object_and_error.md) |
| Task2 | PTQ / QAT 的介入时机 | [Part 01 · 21 量化理论与 INT4/INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) → [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) → [Part 02 · 26 QLoRA 与 4bit 量化](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) | [02 PTQ 与 QAT 的时机](./02_ptq_and_qat_timing.md) |
| Task3 | 低比特训练与适配 | [Part 02 · 26 QLoRA 与 4bit 量化](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) | [03 低比特训练适配](./03_low_bit_training_adaptation.md) |
| Task4 | GPTQ / AWQ 的后训练压缩 | [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) → [Part 02 · 40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) | [04 权重压缩](./04_weight_only_compression.md) |
| Task5 | FP8 与 KV Cache 量化 | [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) → [Part 01 · 12 Tensor Core 与混合精度](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb) → [Part 02 · 41 FP8 与 KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) | [05 FP8 与 KV Cache 量化](./05_fp8_and_kv_cache_quantization.md) |
| Task6 | 量化部署、benchmark 与项目收口 | [Part 02 · 65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb) → [Part 02 · 66 推理性能比较](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) → [Part 02 · 67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) | [06 部署与基准测试决策](./06_deployment_and_benchmark_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“到底该压权重、激活还是 KV cache”“PTQ 和 QAT 哪条路更适合当前约束”时，再回来看对应的专题正文。想看汇总版就进 [量化与压缩正文](./casebook.md)，想按连续故事线走一遍就进 [量化与压缩深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[推理优化](../inference_optimization/intro.md) 负责服务速度与部署链路，[显存优化](../memory_performance_tuning/intro.md) 负责 VRAM 压缩 trade-off，[监督微调与训练工程](../fine_tuning_training/intro.md) 负责 QLoRA 等训练时机问题。

## 项目结论

推荐的实践闭环是 `65 QLoRA 选择 → 66 推理性能比较 → 67 量化推理与部署`；若要理解具体权重量化方法，先补读 `40 GPTQ / AWQ 权重量化`。GGUF 作为独立的格式与部署路径，在 `67` 中使用对应 backend 单独验证，不能直接复用 GPTQ / AWQ 的启动参数。最终结论应同时报告精度或任务质量、显存占用、吞吐或延迟，以及模型格式和后端约束。

## 环境与验证

量化理论、误差计算和部分 W8A16 模拟可先用 CPU；真实权重量化、GPU 推理和后端部署通常需要 GPU。不同显卡、驱动、PyTorch、量化库和 serving backend 可能改变结果，必须记录环境与校准数据，并保存可复现的配置和结果文件。
