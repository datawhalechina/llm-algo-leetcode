# 04. Inference Cache and Memory Budget | 推理缓存与显存预算

## 页面目标

本节回答的是：推理显存为什么会被 KV cache 顶高，为什么分页、复用、调度和压缩都必须回到预算问题来看。

## 问题起点

推理显存问题最典型的表象是：

- 上下文一长，显存就迅速爬升；
- 并发一高，batch 就上不去；
- 延迟也变差，但问题根子未必在 decode 算法，而可能在 cache 组织。

因此，推理侧显存首先是预算问题，其次才是“让请求更快”的问题。

## 你要先确认什么

- peak memory 是否由 KV cache 主导。
- cache 增长是否来自自然长度扩展，还是碎片和复用不足。
- 当前动作是在优化复用、分页、调度还是压缩表示。

## 核心矛盾

KV cache 一方面能让 decode 不必重复计算历史 token，另一方面又会稳定吞掉显存预算。系统希望保留上下文、提高并发，但显存预算会先行成为边界。

## 与推理优化路线的边界

推理优化路线主要问“请求怎样更快、服务怎样更稳”；本节主要问“KV Cache 如何在有限预算中驻留、复用和增长”。因此同一个 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) 在推理专题中比较 TTFT、TPOT 和吞吐，在本专题中还必须记录 cache policy、命中率、峰值显存和并发容量。这样可以避免把“延迟变快”误读成“显存管理已经优化”。

这与 Task2 的训练显存不是同一个对象：训练侧主要观察 activation、梯度和 optimizer state 的生命周期，推理侧主要观察 KV Cache 的增长、复用和驻留。两者共享账本和带宽语言，但不能把 checkpoint/offload 的训练结论直接迁移到推理服务。

## 演化路径

1. 先看 cache 增长是否自然且可接受。
2. 再看 prefix reuse 和分页是否足以缓解碎片和驻留。
3. 如果预算仍然过紧，再考虑 KV cache quantization。
4. 最后回到 benchmark，看省下来的显存是否真的换到了上下文或并发收益。

## 关键取舍

- `prefix reuse` 适合重复历史明显的 workload。
- `paging` 更像 allocator 和布局优化，不一定直接改善算法。
- `KV cache quantization` 能继续压预算，但会引入表示误差和后端约束。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- PagedAttention / vLLM：理解分页为什么是推理 cache 的系统级动作。
- RadixAttention / prefix reuse 工程资料：理解前缀共享对 cache 预算的影响。
- KV cache quantization 资料：理解为什么缓存压缩和权重量化不是一回事。

## 证据边界

CPU 可以验证 KV Cache 的 shape、容量估算、分页和前缀匹配逻辑；真实命中率、并发容量、cache eviction、TTFT / TPOT 和服务显存必须由匹配的 GPU backend workload 验证。这里的缓存模型不能直接当作 vLLM 或 SGLang 的实测结果。

## 对应 Part 02

- [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.md)
- [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)
- [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)、[67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)

## 典型阅读入口

- [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 本节要点

推理侧显存问题的核心不是“cache 有没有”，而是“cache 如何组织、预算是否可接受、代价是否值得”。
