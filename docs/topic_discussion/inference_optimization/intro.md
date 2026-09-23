# 推理优化（Inference Optimization）

> 专题类型：主学习路线　主服务目标：请求性能与 Serving 决策

## 页面导语

本专题面向希望定位 LLM 请求瓶颈、优化延迟与吞吐，并能做出 Serving 选择的学习者。学习重点是把请求性能问题转化为可测量的指标、可验证的机制和可复查的部署决策。

想沿着一个在线服务问题连续学习时，先读[推理优化问题链：从性能症状到项目决策](./walkthrough.md)；想按知识和实验顺序学习时，从下方 Task0 开始。

![推理优化专题主图：请求、Prefill、Decode、KV Cache 与 Serving](../../public/decorative/inference_optimization_topic_hero_light.svg)

## 如何开始

按知识顺序学习时，从 Part 02 的 [2.6 核心推理优化](../../02_PyTorch_Algorithms/2_6.md) 进入，再按 Task0–6 依次学习。Task0 的核心内容建立请求与指标基础，扩展项目 66 将这套指标落实为可复用的 benchmark；想先运行真实服务时，再按[使用指南](../../guide.md)完成 vLLM / SGLang 预检。遇到 Attention、KV Cache、MoE 或 dtype 知识缺口时，回补对应 Notebook。

路线图把请求阶段、Task0–6 的学习顺序和各阶段的验证出口放在一起。

![推理优化学习路线：从请求链路到专项验证](../../public/topic_discussion/inference_optimization/inference_optimization_roadmap.svg)

## 主学习路线与验证出口

下表按“要回答的问题 → 学习入口 → 验证出口”组织；先沿学习顺序阅读机制，再根据问题选择扩展内容和项目。

| Task / 主题 | 本阶段要回答的问题 | 学习入口与验证出口 | 学习顺序 | 主要正文入口 |
|:---|:---|:---|:---|:---|
| Task0 · 请求结构与指标 | 一次请求如何经过 Prefill、Decode 和 KV 状态，并用 TTFT、TPOT 观察它？ | 共享前置：[Part 02 · 04 Attention（MHA / GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)；核心正文：[01 请求链路与指标](./01_request_path_and_metrics.md)；扩展项目：[Part 02 · 66 推理性能比较](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) | 先建立请求阶段和指标口径；只按需回看 Attention，不在本 Task 展开 Attention 实现 | [01 请求链路与指标](./01_request_path_and_metrics.md) |
| Task1 · Prefill 与 Attention Kernel | 为什么 Prefill 会受访存和中间矩阵影响，FlashAttention 改变了什么？ | 核心：[Part 02 · 04 Attention（MHA / GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md) → [Part 01 · 03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) → [Part 01 · 14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md) → [Part 02 · 20 FlashAttention 模拟与 SDPA 对照](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)；扩展：[Part 01 · 24 SRAM 优化](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md) | Attention 数据流 → GPU 约束 → 访存与 Tiling → Prefill；测量方法复用 Task0 扩展 66 | [02 Prefill 与 Attention Kernel](./02_prefill_and_attention_kernel.md) |
| Task2 · 投机式生成与 Decode 加速 | 一条请求如何通过 draft / target、多 Token 或多候选验证，在不牺牲验证语义的前提下提高每轮有效推进量？ | 基础：[Part 01 · 11 KV Cache 增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) → [Part 02 · 21 解码策略](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md)；核心：[Part 02 · 23 投机解码](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md) → [Part 02 · 35 多 Token 投机解码](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.md)；项目：[Part 02 · 68 投机解码基准](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md) | 普通 Decode → 候选提出 → 验证 / 修正 → 有效推进与成本验证；不处理多请求批次调度 | [03 解码策略](./03_decoding_strategies.md) |
| Task3 · KV Cache 状态与生命周期 | KV Cache 如何建立、追加、分页、复用、驱逐并控制容量边界？架构变化会改变每 token 的状态成本吗？ | 核心：[Part 02 · 22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) → [Part 02 · 24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md) → [Part 02 · 34 Prefix Cache 匹配与复用](../../02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.md)；项目：[Part 02 · 69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)；扩展：[Part 02 · 71 MLA](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md) | 建立 → 追加 → 分页 → 前缀匹配与复用 → 驱逐 / 重算 → 架构扩展；不处理请求公平性 | [04 KV Cache 生命周期与复用](./04_kv_cache_lifecycle_and_reuse.md) |
| Task4 · 多请求调度、异构 PD 与 Serving | 多请求如何从请求选择走向动态批次、执行重叠、Prefill / Decode 分池、状态交接和异构资源路由，并维持可控服务质量？ | 核心：[Part 02 · 36 Decode 调度](../../02_PyTorch_Algorithms/36_Decode_Scheduling.md) → [Part 02 · 37 KV Cache 调度](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.md) → [Part 02 · 38 Prefill/Decode 调度](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.md) → [Part 02 · 39 异构 PD 与服务分层](../../02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers.md)；项目：[Part 02 · 70 Serving 调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md) | 请求级选谁 → Cache 资源级能否接纳 → 迭代级动态批次 / Chunked Prefill → 执行级重叠与局部 handoff → Serving 验证；不展开多卡并行机制 | [07 Serving 调度与 PD 分离](./07_serving_scheduling_and_pd.md) |
| Task5 · 量化部署与成本 | 权重和 KV 量化如何改变显存、速度、质量与 backend 选择？ | 核心：[Part 01 · 21 量化理论与 INT4/INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [Part 02 · 25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) → [Part 02 · 40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) → [Part 02 · 41 FP8 / KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)；项目：[Part 02 · 67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)；GGUF 走独立 backend | 表示与时机 → 权重 / KV 量化 → backend 验证 | [05 量化推理与部署](./05_quantized_inference_and_deployment.md) |
| Task6 · 分布式推理、部署比较与决策 | 单实例无法满足容量或服务目标时，如何选择并行方式、估算通信代价、验证多卡/多实例扩展，并形成部署决策？ | 机制支撑：[Part 01 · 05 通信拓扑](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.md) → [Part 02 · 46 NCCL 通信 Profiling](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.md) → [Part 02 · 48 通信热点与缓解](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md) → [Part 02 · 49 并行策略选型](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md)；MoE 分支：[Part 02 · 47 专家并行](../../02_PyTorch_Algorithms/47_MoE_Expert_Parallel.md)；项目：[Part 02 · 79 分布式并行基准](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md) → [Part 02 · 80 MoE 专家并行](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md) → [Part 02 · 81 分布式推理验证](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md)；正文入口：[08 分布式推理与并行](./08_distributed_inference_and_parallelism.md)；收口：[06 端到端基准与决策](./06_benchmark_and_decision.md) | 单实例基线 → 通信观测与缓解 → 并行策略选择 → 多卡/多实例项目 → 模型、backend 与成本决策 → `accept / tune / reject` | [08 分布式推理与并行](./08_distributed_inference_and_parallelism.md) → [06 端到端基准与决策](./06_benchmark_and_decision.md) |

![推理优化知识地图：从请求对象到验证决策](../../public/topic_discussion/inference_optimization/inference_optimization_knowledge_map.svg)

## 按需回补：架构与共享前置

遇到架构、MoE、dtype 或精度方面的知识缺口时，从下表回补对应入口，再回到当前 Task 继续学习。

| 补充主题 | Notebook 入口 | 建议时机 |
|:---|:---|:---|
| 架构基础 | [05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)、[08 Architecture Tricks](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md) | Task0 后按需阅读 |
| MoE 架构 | [06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md)、[07 MoE Load Balancing](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md) | 需要 MoE 或进入多卡前 |
| 硬件与精度 | [Part 01 · 01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md) | 需要补充 dtype、精度或表示基础时 |
| MoE 计算 | [Part 01 · 22 MoE 参数与计算](../../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.md) | 学 MoE 或分布式前按需回补 |

![架构与共享前置关系图：补充内容如何接入推理优化主线](../../public/topic_discussion/inference_optimization/architecture_and_prerequisites.svg)

## 跨专题入口

项目入口已经列在上面的主学习路线表中；需要按现象分流时阅读[推理优化判断手册](./casebook.md)，需要沿请求问题链连续阅读时阅读[推理优化问题链](./walkthrough.md)。如果问题跨到其他方向，可转到[性能分析](../profiling/intro.md)、[显存优化](../memory_performance_tuning/intro.md)、[量化与压缩](../quantization/intro.md)、[算子优化](../operator_optimization/intro.md)或[编译与图优化](../operator_optimization/graph_compiler/intro.md)。

## 环境与验证

按实验目标选择环境组合：CPU 环境用于机制和指标逻辑，GPU 或 backend 环境用于真实性能与部署结果。

| 实验类型 | 环境组合 | 适用内容 |
|:---|:---|:---|
| CPU 机制 | [`base.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/base.txt) + [`torch-cpu.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/torch-cpu.txt) | Attention、解码、指标和机制模拟 |
| GPU / Transformers | [`base.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/base.txt) + [`torch-cu128.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/torch-cu128.txt) | GPU 延迟、吞吐和显存测量 |
| vLLM backend | [`torch-cu128.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/torch-cu128.txt) + [`inference-vllm.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/inference-vllm.txt) | 66–72 的 vLLM 实验 |
| SGLang backend | [`torch-cu128.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/torch-cu128.txt) + [`inference-sglang.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/inference-sglang.txt) | 66、69、70、72 的 SGLang 对照实验 |
| Profiling | [`torch-cu128.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/torch-cu128.txt) + [`profiling.txt`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/requirements/profiling.txt) | trace、TensorBoard 和性能证据 |

每次真实实验都记录模型、后端、数据类型、序列长度、并发度和结果文件；用报告中的 `evidence level` 区分 smoke test 与稳定 benchmark。

开始真实实验前，先看[使用指南](../../guide.md)中的环境边界；需要逐条执行时，使用[66–72 推理项目验证清单](../../verification/inference_projects.md)。
