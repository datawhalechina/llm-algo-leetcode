# 多模态推理（Multimodal Inference）

> 专题类型：领域专题（建设中）　主服务目标：理解视觉语言模型如何处理图像输入，并完成可测量、可复核的推理服务优化

## 页面导语

本专题第一阶段聚焦**视觉语言模型（VLM）和图像输入**。学习者将沿着一条完整链路推进：原始图像如何变成视觉表示，视觉 token 如何进入语言模型，视觉输入如何改变 Prefill、Decode、KV Cache 和服务成本，最后如何用质量与性能证据做部署决策。

视频、音频、文档 OCR 和复杂多模态训练暂不作为第一阶段主线；它们会在图像推理路线稳定后作为扩展项目加入。这样可以先用一个固定模态建立可复用的 workload、benchmark 和调优方法，再扩展到更复杂的输入。

![多模态推理路线：从视觉表示到可复核服务](../../docs/public/topic_discussion/multimodal/multimodal_overview.svg)

## 如何开始

- **第一次学习：** 从 Task0 开始，先理解视觉编码器、连接器、视觉 token 和语言模型之间的数据流。
- **已有推理基础：** 从 Task1 进入，先建立图像请求与纯文本请求的 baseline，再学习 token 预算和缓存管理。
- **已有性能分析基础：** 从 Task5 进入，先固定模型、图片、分辨率和输出长度，完成一次端到端 profiling。
- **只想完成项目：** 从 Task6 进入，但需要先复用 Task1 的 workload 和质量指标，避免把一次演示误当成性能结论。

本专题会复用推理优化、性能优化、显存优化、量化和算子优化的方法；这些专题负责通用机制，多模态专题负责验证这些机制在视觉输入场景中的变化。

## 第一阶段范围

| 纳入主线 | 暂不纳入主线 |
|:---|:---|
| 图像输入的视觉语言模型 | 音频输入的完整推理链路 |
| 视觉编码器、连接器和视觉 token | 视频理解的完整工程链路 |
| 图像分辨率与 token budget | 多模态预训练和大规模训练 |
| 多模态 Prefill / Decode | 通用 OCR 产品建设 |
| visual / text KV Cache 与输入缓存 | 不同模型、不同后端的全面排名 |
| vLLM、SGLang 的图像服务验证 | 没有统一 workload 的性能结论 |

## 主学习线与核心问题

`Task0–6` 是第一阶段的推荐顺序。每个 Task 都要求同时保留机制解释、固定 workload、质量指标和性能证据；论文中的结果作为案例，不直接当作本教程的实测结论。

| Task | 核心问题 | 主要机制 | 预期验证出口 |
|:---|:---|:---|:---|
| Task0：VLM 架构与数据流 | 视觉信息如何进入语言模型？ | 视觉编码器、连接器、视觉 token、Cross-Attention、token 注入 | 画出从图像到文本输出的数据流，并标注各阶段的输入输出 |
| Task1：Workload 与 Baseline | 图像请求为什么比纯文本请求更复杂？ | 预处理、视觉编码、projector、Prefill、Decode、分辨率与输出长度 | 对照记录阶段耗时、TTFT、TPOT、吞吐和峰值显存 |
| Task2：视觉 Token 预算与压缩 | 如何减少视觉 token，同时控制质量损失？ | 动态分辨率、token pruning、token merging、帧采样、质量门槛 | 在固定 VQA workload 上比较 token 数、质量、延迟和显存 |
| Task3：多模态 KV Cache 与输入缓存 | 视觉 KV、文本 KV 和媒体处理缓存如何管理？ | modality-aware cache、视觉复用、多轮对话、分页与淘汰 | 对比复用前后的显存、TTFT、命中率和质量回归 |
| Task4：多模态推理引擎接入 | 不同 backend 如何接收和处理图像输入？ | chat template、processor、placeholder、输入缓存、服务配置 | 用同一模型和 workload 对照 vLLM / SGLang 的结果 |
| Task5：多模态 Profiling 与调优 | 慢在图像处理、视觉编码、Prefill 还是 Decode？ | 阶段拆解、数据传输、batch、分辨率、token budget、显存峰值 | 形成瓶颈归因、优化动作和前后对照报告 |
| Task6：图像问答综合项目 | 如何在质量、延迟、显存和成本之间做部署决策？ | 模型选择、缓存、压缩、服务配置、回归验证 | 输出可复查的图像问答服务报告和 accept / tune / reject 决策 |

## 每个 Task 的统一证据口径

多模态性能不能只记录端到端延迟。每次对照至少固定并记录：

| 类别 | 必须记录的字段 |
|:---|:---|
| 模型 | VLM 名称、视觉编码器、连接器、权重精度、backend 版本 |
| 输入 | 图片数量、分辨率、宽高比、视觉 token 数、文本长度 |
| 请求 | batch size、并发数、最大输出长度、warmup、重复次数 |
| 质量 | VQA / OCR / grounding 等任务指标、失败样例、人工抽检规则 |
| 性能 | 预处理、视觉编码、Prefill、Decode、TTFT、TPOT、吞吐 |
| 资源 | 峰值显存、权重占用、KV / 输入缓存占用、OOM 或请求失败 |

同一模型和 workload 下，先建立 baseline，再改变一个变量。token 压缩、缓存策略和 backend 对照都必须同时观察质量与性能，不能只用 token 减少率或吞吐提升率下结论。

## 共享前置与关联专题

| 需要补的能力 | 入口 | 在本专题中的作用 |
|:---|:---|:---|
| Transformer 与 Attention | [Attention（MHA / GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)、[LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) | 理解视觉 token 与文本 token 进入同一注意力计算的方式 |
| 模型架构 | [大模型架构专题](../model_architecture/intro.md) | 补充视觉塔、连接器、融合层和模型变体 |
| 推理服务 | [推理优化](../inference_optimization/intro.md) | 复用 Prefill / Decode、请求调度和服务指标 |
| 性能证据 | [性能优化](../performance_optimization/intro.md)、[显存优化](../memory_performance_tuning/intro.md) | 复用 profiling、显存账本和 benchmark 方法 |
| 算子与量化 | [算子优化](../operator_optimization/intro.md)、[量化部署](../quantization/intro.md) | 作为视觉编码器和低精度路径的扩展支撑 |
| 质量与对齐 | [后训练优化](../post_training_optimization/intro.md) | 复用任务指标、行为回归和质量门槛 |

## 参考入口

- [vLLM 多模态输入](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)：图像、视频、音频输入，以及多模态输入缓存和服务接口。
- [TensorRT-LLM 多模态支持](https://nvidia.github.io/TensorRT-LLM/features/multi-modality.html)：多模态处理器、视觉编码器和 LLM 解码器的工程组合。
- [VL-Cache](https://arxiv.org/abs/2410.23317)：视觉与文本 token 的稀疏性和 KV Cache 压缩案例。
- [TokenCarve](https://arxiv.org/abs/2503.10501)：视觉 token 压缩、质量和推理代价的案例。

这些链接用于理解公开实现和研究问题；它们的实验数字不能直接替代本教程在固定 workload 上的复测结果。

## 环境与建设状态

机制和小规模数据流验证可以先使用 CPU 与基础 PyTorch 环境完成；真实 VLM、图像预处理、GPU 显存、吞吐和 backend 对照需要单独准备 GPU 环境。第一阶段优先建设图像 VQA 的固定数据集和单模型 baseline，再扩展到多模型、视频和音频。

- **已完成：** 第一阶段范围、Task0–6 路线、统一证据字段和专题边界。
- **待补齐：** VLM 架构 Notebook、图像 workload、token budget 实验、输入缓存实验、vLLM / SGLang 对照和 GPU 项目。
- **当前不输出：** 在没有固定模型、数据集、硬件和评测协议之前，不给出模型或 backend 的性能排名。
