# 推理优化问题链：从性能症状到项目决策

阅读前提：默认你已经知道模型会经历 Prefill 和 Decode 两个阶段，并能把 `TTFT` 理解为首 token 延迟、把 `TPOT` 理解为后续 token 的平均生成间隔。如果这些概念还不熟，先看 [01 请求链路与指标](./01_request_path_and_metrics.md)，再回到问题链；这里重点是学习如何根据性能症状选择机制和项目，不要求先掌握具体 backend 的源码。

假设你接手的是同一个在线服务：模型已经能跑，但长 prompt 下首 token 明显变慢；并发一起来，生成速度又开始恶化；为了继续把 batch 和上下文往上推，团队开始尝试 cache 策略和量化，结果显存虽然省了，服务体验却不一定更好。

这条线最重要的是按暴露顺序判断：问题先出在哪一段，压下去以后瓶颈又转到哪里，最后哪些候选方案真的值得保留。

![推理优化从请求症状到部署决策](../../docs/public/topic_discussion/inference_optimization/inference_optimization_story.svg)

图片先给出整条判断链：从请求阶段定位问题，再选择机制和项目，最后用统一 workload 做决策。下面各段只展开其中一个阶段。

## 第一段：先确认是不是 prefill 问题

第一阶段最常见的症状是：短 prompt 还可以，一旦 prompt 拉长，`TTFT` 明显升高，但 decode 阶段单 token 速度还没完全坏掉。不要笼统说“模型太慢”，而是先把问题压到 prefill：

- Part 01 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)
- Part 01 [14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)
- Part 01 [24 SRAM Optimization Techniques](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)
- Part 02 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)
- Part 02 [34 Prefix Cache Matching and Reuse](../../02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.ipynb)
- Part 02 [38 Prefill/Decode Scheduling](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb)（长 prompt 影响并发时）

这里真正想确认的是：`TTFT` 升高是不是主要来自 attention 和 prefill，FlashAttention 或 chunked prefill 有没有空间，重复前缀是不是让无效 prefill 变多。第一个关键判断是：**首 token 慢，不等于生成慢。**

![Prefill 与 Attention 访存](../../docs/public/topic_discussion/inference_optimization/prefill_attention_zh.svg)

这张图把长 Prompt、Prefill 和 TTFT 的关系拆开，并区分访存优化、分块处理与前缀复用；具体取舍仍需回到同一 workload 验证。

## 第二段：prefill 压下去以后，瓶颈转到 decode

团队把 prefill 调过一轮以后，经常会看到第二个问题：`TTFT` 下来了，但并发一高，`TPOT` 又开始恶化，generated tokens/s 还是不高。这时问题已经从 prefill 转移到了 decode loop 和请求组织：

- Part 02 [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)
- Part 02 [23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)
- Part 02 [35 Multi-Token Speculative Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb)
- Part 02 [36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)

这里最容易出现的误判是：只要引入 speculative decoding，吞吐就一定变高。真实情况更依赖 workload；如果请求短、并发低，复杂策略不一定值得，如果调度没排顺，单独换策略也未必真能把 TPOT 压下来。

![Decode 策略对照](../../docs/public/topic_discussion/inference_optimization/decode_strategies_zh.svg)

图中关注的是生成阶段的候选策略和代价：减少单步生成成本的同时，还要观察验证开销、请求长度和调度影响。

## 第三段：吞吐还想继续上推，cache 开始决定边界

走到这一步，服务通常已经不是“完全跑不动”，而是开始被 KV cache 的容量和复用边界卡住：batch 想继续往上推，但 `peak memory` 顶住了；cache 一边跑一边涨，decode 稳定性和并发一起变差。这时先把注意力切到 cache 生命周期和复用：

- Part 01 [11 KV Cache and Memory Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb)
- Part 02 [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
- Part 02 [24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)
- Part 02 [34 Prefix Cache Matching and Reuse](../../02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.ipynb)

这里可以按四个问题理解 Cache：先看 Decode 保存的请求级状态，再看 PagedAttention 如何管理物理 Block；如果请求之间有重复前缀，再看 RadixAttention / Prefix Cache 如何减少重复 Prefill；当 Cache 接近预算时，才进入驱逐、压缩或 KV Cache 量化。推理优化关注这些机制如何影响吞吐、并发和服务稳定性；如果核心问题已经变成“装不下”，再转去显存专题。

![KV Cache 生命周期](../../docs/public/topic_discussion/inference_optimization/kv_cache_lifecycle_zh.svg)

这张图把 Cache 的增长、复用和调度边界放在一起；它用于判断瓶颈位置，不替代真实 backend 的命中率、并发和显存测量。

## 第四段：Cache 边界暴露后，Serving 调度成为系统问题（对应正文 07）

当 Cache 管理已经不能单独解决问题，新的瓶颈通常出现在请求之间：长 Prompt 的 Prefill 会带来突发计算，Decode 则需要持续获得稳定的计算和 Cache 访问。如果两类请求共享同一调度策略，TTFT、TPOT 和 P99 可能互相牵制。

- Part 02 [36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)
- Part 02 [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb)
- Part 02 [38 Prefill/Decode Scheduling](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.ipynb)
- Part 02 [39 Hetero PD and Serving Tiers](../../02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers.ipynb)
- Part 02 [70 Serving Scheduler Benchmark](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)
- 专题正文 [07 Serving 调度与 PD 分离](./07_serving_scheduling_and_pd.md)

这一段要回答的不是“哪种调度一定更快”，而是：当前请求分布是否已经需要 Continuous Batching、Chunked Prefill、PD 分离或异构路由。CPU 可以帮助理解队列、分块与决策状态；调度收益、P99、跨实例通信、KV 交接和 GPU 利用率仍需真实 backend 验证。

## 第五段：量化进入候选集，但不自动代表服务更好

当 cache、batch 和部署成本一起成为约束时，团队通常会开始引入量化。这里最危险的误判是：只要显存降了，就默认服务更好。这一步应沿量化部署线来判断：

- Part 01 [21 Quantization Theory and INT4 INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb)
- Part 02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)
- Part 02 [40 GPTQ and AWQ Weight Quantization](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)
- Part 02 [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)
- Part 02 [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)

这里真正要比较的是：这些方案改的是权重、KV cache 还是运行态张量；它们对 `TTFT / TPOT / throughput / peak memory` 的影响是否一致；当前服务目标更重在线交互还是更重离线吞吐。

![量化推理与部署](../../docs/public/topic_discussion/inference_optimization/quantized_deployment_zh.svg)

量化不是单一的“降显存”开关；图中应先区分压缩对象和运行时支持，再回到延迟、吞吐、显存和质量一起判断。

## 第六段：最后回到同一 workload 做结论

前面几段都还是局部判断，真正做结论时，必须回到同一个 benchmark 框架里。核心收口页是 66；68、69、70 是按 Decode、Cache 和 Serving 目标选择的扩展项目；当单实例已经无法满足容量或服务目标时，再进入 Task6 的 79–81 分布式机制与项目，最后用 72 做模型和部署比较：

![Benchmark 决策流程](../../docs/public/topic_discussion/inference_optimization/benchmark_decision_zh.svg)

最终比较必须固定 workload、指标和质量门槛，再将候选方案归入 `accept / tune / reject`；图片展示的是决策流程，不是某次实验的结果。

- **核心项目：** Part 02 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
- **扩展项目：** Part 02 [68 Speculative Decoding Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)
- **主题项目：** Part 02 [69 Prefix Caching Benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)
- **扩展项目：** Part 02 [70 Serving Scheduler Benchmark](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)
- **分布式机制与项目：** Part 02 [46 Communication Profiling with NCCL](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb) → [48 Communication Hotspots and Mitigation](../../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.ipynb) → [49 Parallelism Strategy Selection](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.ipynb) → [79 Distributed Parallel Benchmark](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb) → [81 Distributed Inference Logic Validation](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb)；MoE 分支进入 [47](../../02_PyTorch_Algorithms/47_MoE_Expert_Parallel.ipynb) → [80](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb)
- **部署收口：** Part 02 [72 Multi-Model Deployment Comparison](../../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.ipynb)

真正的 benchmark 收口不是“这个方法更先进”，而是 `accept / tune / reject`：它是否适合当前 workload 和服务目标。把这条故事走完以后，一个更像真实交付的结论通常是：长 prompt 下先解决 prefill，随后瓶颈转到 decode 与调度，再往后是 cache 和量化共同决定服务边界，最终被接受的不是某个单点技巧，而是一组在同一 workload 下同时站得住的链路优化组合。
