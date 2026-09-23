# 推理优化判断手册

> 这份判断手册按问题组织，不属于 Task0–6 的顺序学习内容。完成机制学习后，遇到延迟、吞吐、并发或显存问题时，用它选择排查方向和验证项目。

推理优化不应从工具名称开始，而应从请求表现开始判断：问题发生在 prefill、decode、KV Cache、调度还是模型部署，应该先改计算、缓存、调度还是表示方式。

统一判断链是：

```text
现象 → 请求阶段 → 瓶颈对象 → 策略代价 → 证据出口
```

本专题重点关注 Infra-L2–L4 的执行效率；模型版本治理、灰度发布、扩缩容和流量治理属于 Infra-L5，不在这份判断手册展开。

## 推理栈与组件边界

选择推理方案时，先区分“模型是什么”和“模型如何运行”。训练或微调得到的是模型权重；量化、转换或编译后得到的是推理部署产物；vLLM、SGLang、TensorRT-LLM 和 llama.cpp 则负责加载产物、管理执行状态并处理请求。

![模型、部署产物与推理引擎关系](../../docs/public/topic_discussion/inference_optimization/model_artifact_engine_serving.svg)

| 名称 | 主要职责 | 典型例子 | 在本专题中的判断重点 |
|:---|:---|:---|:---|
| 训练模型 | 保存架构、权重和配置，可继续训练或微调 | Llama、Qwen、DeepSeek 权重 | 模型结构、参数规模、dtype 和上下文能力 |
| 推理部署产物 | 为部署准备的权重文件、量化文件或编译结果 | FP16 / BF16 checkpoint、GPTQ、AWQ、GGUF、TensorRT Engine | 文件格式、量化方式、加载要求和质量变化 |
| 推理框架 | 提供模型加载、张量计算和生成接口 | PyTorch、Transformers | 参考实现、模型兼容性和可读性 |
| 推理引擎 / Runtime | 管理 KV Cache、调度请求、调用 kernel 并提供服务接口 | vLLM、SGLang、TensorRT-LLM、llama.cpp | backend、调度、缓存、kernel 和运行时收益 |

不要把“推理模型”作为默认术语：它也可能被理解为具有推理能力的模型。本文统一使用“训练模型”和“推理部署产物”描述模型侧对象。

### 常见推理引擎如何分工

| 引擎 | 重点机制 | 适合观察的问题 | 使用边界 |
|:---|:---|:---|:---|
| vLLM | PagedAttention、KV Cache 管理、Continuous Batching | 缓存组织、批处理、TTFT、TPOT 和吞吐 | 收益依赖模型、版本、dtype 和 workload |
| SGLang | RadixAttention、Prefix Cache、结构化请求执行 | 前缀复用、缓存命中、结构化生成和调度 | 需要单独确认模型与版本适配 |
| TensorRT-LLM | TensorRT 编译、融合 kernel、量化和 NVIDIA 优化 | 编译后执行、kernel 路径和 NVIDIA 平台性能 | engine 构建和环境准备更复杂 |
| llama.cpp | GGUF 加载、CPU/GPU 混合执行 | 本地部署、GGUF 格式和资源受限场景 | GGUF 是独立 backend 路径，不沿用 GPTQ / AWQ 或 vLLM 的启动方式 |

这张表用于选择验证方向，不替代各项目的安装说明。[66](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 负责浮点推理 baseline；[67](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) 负责量化部署；[69](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb) 关注 Prefix Cache / RadixAttention；[70](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb) 关注多请求调度。TensorRT-LLM 和 llama.cpp 先作为可选 backend 路径，不强行塞入所有基础项目。

## 按现象选择机制

遇到 Cache 相关问题时，先判断改变的是“保存什么”“如何分配”“如何复用”还是“如何控制容量”，再进入对应 Notebook 或项目。

| Cache 角色 | 先回答的问题 | 典型机制 | 主要指标 | 学习与验证入口 |
|:---|:---|:---|:---|:---|
| 请求级状态 | Decode 需要保存哪些历史 K / V？ | KV Cache | 峰值显存、TPOT | [11 KV Cache](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) / [03 Decode](./03_decoding_strategies.md) |
| 物理分配 | Cache 如何减少碎片并按需增长？ | PagedAttention、Block Table | Block 利用率、并发容量、峰值显存 | [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) |
| 前缀复用 | 相同前缀是否可以避免重复 Prefill？ | RadixAttention、Prefix Cache | 命中长度、reused tokens、TTFT | [24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb) / [69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb) |
| 容量治理 | Cache 接近预算时如何控制代价？ | 驱逐、压缩、KV Cache 量化 | OOM、速度、质量、并发 | [04 KV Cache](./04_kv_cache_lifecycle_and_reuse.md) / [41 KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) |

如果还不能确定瓶颈对象，先使用下面的现象表定位阶段；已经确认属于 Cache 问题后，再使用上面的 Cache 角色表选择具体机制。

| 现象 | 先判断的对象 | 优先策略 | 主要指标 | 验证出口 |
|:---|:---|:---|:---|:---|
| 长 prompt 下首 token 明显变慢 | prefill 计算与访存 | FlashAttention、chunked prefill、prefix cache | TTFT、prefill 时间、峰值显存 | [02 Prefill](./02_prefill_and_attention_kernel.md) |
| 单请求生成阶段每 token 很慢 | decode 计算和 KV 读取 | speculative decoding、multi-token decoding | TPOT、接受率、质量 | [03 Decode](./03_decoding_strategies.md) |
| 并发增加后生成速度下降 | decode 调度和 GPU 利用率 | decode scheduling、continuous batching | TPOT、吞吐、P99、GPU 利用率 | [03 Decode](./03_decoding_strategies.md) / [07 Serving](./07_serving_scheduling_and_pd.md) |
| cache 持续增长，并发上不去 | KV Cache 容量与组织 | paging、prefix reuse、eviction | cache 容量、命中率、并发、峰值显存 | [04 KV Cache](./04_kv_cache_lifecycle_and_reuse.md) |
| 显存下降但交互体验变差 | 量化或缓存策略的代价 | 区分权重量化、KV Cache 量化和调度影响 | TTFT、TPOT、吞吐、质量、显存 | [05 Quantization](./05_quantized_inference_and_deployment.md) / [06 Benchmark](./06_benchmark_and_decision.md) |
| backend 能加载但收益不稳定 | kernel、格式或运行时路径 | 对齐 backend、dtype、workload 和版本 | 格式、kernel、延迟、吞吐、质量 | [06 Benchmark](./06_benchmark_and_decision.md) |

## 证据与项目决策

![推理优化证据与决策流程](../../docs/public/topic_discussion/inference_optimization/benchmark_decision_zh.svg)

| 指标 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| `TTFT` | 首 token 是否被 prefill 拖慢 | 只看总延迟，无法定位 prefill |
| `TPOT` | decode 阶段每 token 是否过慢 | 把 decode 慢归因于整个模型 |
| `throughput` | 单位时间能完成多少请求或 token | 只看吞吐，不看交互延迟 |
| `P99` | 长尾请求是否不稳定 | 只看平均值，忽略排队和调度 |
| `peak memory` | 当前配置能否继续增加上下文或并发 | 显存接近上限时仍盲目加 batch |
| `cache hit rate` | Prefix / Radix Cache 是否真的复用 | 只看 TTFT 变化就推断命中 |
| `acceptance rate` | speculative decoding 是否减少了验证成本 | 只看生成速度，不检查质量和接受率 |

| 证据层级 | 可以确认什么 | 不能确认什么 |
|:---|:---|:---|
| CPU 模拟 | shape、容量公式、匹配逻辑和决策逻辑 | 真实显存、TTFT、吞吐和 backend 命中 |
| GPU 探针 | CUDA 分配和基础峰值变化 | 完整 backend 机制收益 |
| backend smoke | 服务能否启动和基本指标 | 稳定 benchmark 与普遍结论 |
| repeated benchmark | 固定 workload 下的相对收益 | 其他模型或 workload 的普遍收益 |

`66` 负责建立浮点推理 baseline 和统一指标口径；`68–71` 分别验证投机解码、缓存复用、调度和架构扩展。任何候选方案都应在固定模型、backend、dtype、prompt / generated tokens、并发和请求分布下比较。

形成结论时按以下顺序检查：

1. **先定位阶段。** 先区分 prefill、decode、cache 和 serving 调度，不要用一个总延迟替代阶段指标。
2. **再定位对象。** 说明瓶颈来自计算、访存、缓存容量、排队还是模型加载路径。
3. **再选择策略。** 写清楚代价转移到了算力、带宽、显存、通信、排队延迟还是质量。
4. **最后选择证据。** CPU 只能验证 shape、容量公式和决策逻辑；真实 TTFT、TPOT、P99、吞吐和 backend 行为需要 GPU 实验。

使用 `accept / tune / reject` 时，必须同时考虑速度、显存、质量和稳定性。单次 smoke test 或单一指标改善，只能标记为待验证，不能写成完整推理优化结论。

## 阅读入口

- 想按顺序学习机制：回到[推理优化入口](./intro.md)，再读 01–07 正文。
- 想沿请求生命周期完整阅读：查看[推理优化深入阅读](./walkthrough.md)。
- 想分析显存对象和预算：转到[显存优化](../memory_performance_tuning/intro.md)。
- 想分析低比特表示和部署格式：转到[量化与压缩](../quantization/intro.md)。
- 想定位 kernel、访存和 trace 证据：转到[性能分析](../profiling/intro.md)或[算子优化](../operator_optimization/intro.md)。
