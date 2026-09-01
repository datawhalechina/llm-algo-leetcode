# 推理优化（Inference Optimization）

> 专题类型：主学习路线　主服务目标：请求性能与 Serving 决策

## 页面导语

本专题围绕请求链路、KV Cache、解码、量化和 Serving，最终形成可验证的推理部署决策。

## 适合谁

适合希望优化 LLM 请求延迟、吞吐和 Serving 稳定性的学习者。可以先用 CPU 完成 Attention、解码和指标机制，再根据需要进入 GPU、vLLM / SGLang backend 和多请求项目。

## 如何开始

推荐从 Part 02 的 [2.6 核心推理优化](../../02_PyTorch_Algorithms/2_6.md) 开始；需要硬件和 Attention 前置时回补 Part 01 的 GPU、显存与访存内容。

- 必读前置：Part 01 的 GPU 架构、显存访问和 KV Cache 基础；Part 02 优先完成 2.6，再进入 2.7–2.8 的 serving、量化和调度内容。
- 按需回补：如果对 Attention、带宽或 KV Cache 不熟，先回补对应 Part 00 / Part 01 Notebook。
- 真实 backend 实验前：先阅读 [使用指南](../../guide.md) 和具体 Notebook 的环境说明。

新模型架构不新增独立 Task：先完成 Attention 和请求链路，再按需阅读下方的架构补充分支，理解 Block、MoE 和 MLA 如何改变计算量、路由与状态形态。

如果目标是尽快部署服务，可以先按[使用指南](../../guide.md)完成 vLLM / SGLang 环境预检，并直接运行 66 的最小 backend smoke test；这是一条部署捷径，不替代 Task0–1 对 Attention、Prefill 和访存机制的学习。

## 主学习线与分级

`Task0-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 / 项目入口 | 学习顺序 | 专题正文 |
|:---|:---|:---|:---|:---|
| Task0 | 推理结构与请求链路基础 | [Part 02 · 04 Attention（MHA / GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)；建立请求、TTFT、TPOT 和 throughput 的共同口径 | 先理解 Attention，再建立请求和指标 | [01 请求链路与指标](./01_request_path_and_metrics.md) |
| Task1 | 硬件约束、Attention 与 Prefill | [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) → [Part 01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md) → [Part 02 · 20 FlashAttention 模拟](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)；扩展 [Part 01 · 24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md) | 先看 GPU 约束，再看访存、Tiling 和 Prefill | [02 Prefill 与 Attention Kernel](./02_prefill_and_attention_kernel.md) |
| Task2 | Decode 与生成策略 | [Part 02 · 21 解码策略](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md) → 扩展 [Part 02 · 23 投机解码](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md)、[Part 02 · 35 多 Token 解码](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.md)、[Part 02 · 36 Decode 调度](../../02_PyTorch_Algorithms/36_Decode_Scheduling.md) → 项目 [Part 02 · 68 投机解码基准](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md) | 先理解单步生成，再看 draft、multi-token 和 Decode 工作调度 | [03 解码策略](./03_decoding_strategies.md) |
| Task3 | KV Cache 表示、复用与 Chunked Prefill | [Part 01 · 11 KV Cache 增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) → [Part 02 · 22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) → [Part 02 · 24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md) → [Part 02 · 34 Prefix Cache 与 Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md) → 项目 [Part 02 · 69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)；扩展 [Part 02 · 71 MLA](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md) | 增长 → 分页 → 复用 → suffix 分块 → 架构变化 | [04 KV Cache 与调度](./04_kv_cache_and_scheduling.md) |
| Task4 | Serving 架构、PD 分离与请求调度 | [Part 02 · 37 KV Cache 调度](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.md) → [Part 02 · 38 Prefill / Decode 分离](../../02_PyTorch_Algorithms/38_Prefill_Decode_Disaggregation.md) → 项目 [Part 02 · 70 Serving 调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md)；扩展 [Part 02 · 39 推理回退与分层](../../02_PyTorch_Algorithms/39_Inference_Fallback_and_Tiers.md)、[Part 02 · 79–81 分布式项目](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md) | Cache 资源 → prefill/decode 拆池 → 请求调度 → 分布式扩展 | [04 KV Cache 与调度](./04_kv_cache_and_scheduling.md) |
| Task5 | 量化推理与部署 | [Part 01 · 21 量化理论与 INT4/INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) → [Part 02 · 40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) → [Part 02 · 41 FP8 / KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md) → 项目 [Part 02 · 67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) | 表示 → 通用量化 → 权重算法 → KV / FP8 → 真实 backend | [05 量化推理与部署](./05_quantized_inference_and_deployment.md) |
| Task6 | 综合 benchmark 与项目决策 | 项目 [Part 02 · 66 推理性能比较](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)；汇总 66–71 的结果 | 最后统一 workload、指标、质量约束和 accept / tune / reject | [06 基准测试与决策](./06_benchmark_and_decision.md) |

### 架构补充分支（不改变 Task0–6 主线）

新模型架构不按“新 Task”平铺，而按它改变的对象接入路线：

| 架构入口 | 建议位置 | 主要回答的问题 |
|:---|:---|:---|
| [05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)、[08 Architecture Tricks](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md) | Task0 后按需阅读 | Block 由哪些 Attention、Norm、MLP 组件组成，结构变化怎样改变计算图 |
| [06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md)、[07 MoE Load Balancing](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md) | Task0 后按需阅读；多卡再接 80 | expert 如何路由，active parameters、负载不均和通信代价从哪里来 |
| [71 MLA / KV Cache Architecture Benchmark](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md) | Task3 扩展项目 | MLA 如何改变 KV Cache 表示、容量和访问路径，并由 workload 验证代价 |

因此，05–08 与 71 都属于“架构交叉内容”，但不放在同一学习节点：前者先解释模型结构，后者需要先具备 KV Cache 语境。推理路线的主线仍从 04 Attention 进入，再到 Prefill、Decode、KV Cache、Serving 和量化。

### 核心与扩展分级

核心路径先建立请求链路、访存和服务指标的共同口径，再完成一次可复查的单策略或单 backend 实验；扩展路径进入真实服务、复杂 workload、多模型协作或跨策略比较。实践等级统一为：`Practice-P0` 机制或 CPU-first，`Practice-P1` 单 GPU / 单 backend，`Practice-P2` 真实 backend、多请求或复杂 workload，`Practice-P3` 多卡或生产级扩展。真实结果以项目报告中的 evidence level 为准，smoke test 不等同于稳定 benchmark。

### Part01 P1 前置

这些小节是推理路线需要掌握的共享基础，不改变它们在 Part 01 中的原有主题。表中的 `Part 01 · xx` 表示共享前置；未标注的 Part 02 小节属于本路线主线或扩展。

| P1 前置 | 在本路线中解决的问题 | 不在本路线中直接下结论 |
|:---|:---|:---|
| [Part 01 · 01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)、[Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) | dtype、带宽、容量和硬件约束 | 不替代真实 backend benchmark |
| [Part 01 · 04 Attention 与显存访问](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)、[Part 01 · 11 KV Cache 增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) | Attention 结构与 KV Cache 预算 | 不把理论账本写成实测峰值 |
| [Part 01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)、[Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) | Prefill 访存模型与量化边界 | 不保证某个 kernel 或量化格式在所有 GPU 可用 |
| [Part 01 · 22 MoE 参数与计算](../../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.md) | MoE 的 active parameters、路由和通信成本 | 不替代 79–81 的分布式实测 |

架构扩展的边界是：05/08/06/07 解释“模型怎么计算”，11/22/71 解释“状态怎么保存或路由”；只有当它们进入具体 workload，才在 66–71 或分布式项目中验证延迟、吞吐、显存和质量。

## 学习方式与项目产出

先按上面的 `Task0-6` 走 Notebook 主线；核心路径用于建立机制和完成最小实验，扩展路径用于真实 GPU、backend、并发或复杂 workload。需要连续理解概念时阅读专题正文 `01-06`，需要判断表和项目分流时阅读[推理优化正文](./casebook.md)，需要完整串联路线时阅读[推理优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[性能分析](../profiling/intro.md) 负责定位慢在哪里，[显存优化](../memory_performance_tuning/intro.md) 负责看预算与吞吐取舍，[量化与压缩](../quantization/intro.md) 负责看低比特路线；单个 kernel、访存和算子融合进入[算子优化](../operator_optimization/intro.md)，图变换、IR、lowering 和 backend 决策进入[编译与图优化](../compiler_graph_optimization/intro.md)。

### 项目产出

项目按“主题验证 → 综合决策”分层：

- **核心综合项目：** [66 推理性能对比](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)，统一比较 backend、workload、TTFT、TPOT、吞吐、P99 和峰值显存。
- **主题项目：** [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)、[69 前缀缓存](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)、[71 MLA / KV Cache 结构基准](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md)。它们分别验证量化、缓存复用和缓存表示；71 的 profiling 扩展转到 74，不在本节重复采集 trace。
- **扩展项目：** [68 推测解码](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md)、[70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md)。它们适合在具备真实 backend 或明确请求 workload 后选做。

没有 GPU 或真实 backend 时，可以先完成机制 Notebook 和 CPU-first 模板；接入 vLLM / SGLang 后，再将 66–71 升级为 Practice-P2 实验。最终统一使用 `accept / tune / reject` 输出策略判断。

## 环境与验证

基础机制可 CPU-first；真实吞吐、TTFT、TPOT 和 backend 对比需要 GPU 以及匹配的 vLLM / SGLang 运行环境。实验结论应同时记录模型、后端、数据类型、序列长度、并发度和结果文件。

开始真实实验前，先看[使用指南](../../guide.md)中的环境边界；需要逐条执行时，使用[66–70 推理项目验证清单](../../verification/inference_projects.md)。
