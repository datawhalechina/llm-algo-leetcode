# 01. Request Path and Metrics | 请求链路与指标口径

## 页面目标

本节回答两个问题：

- 一个请求从进来到输出 token，链路长什么样？
- TTFT、TPOT、throughput、peak memory 这些指标应该怎么一起看？

## 本节在路线中的位置

本节对应 **Task0：推理结构与请求链路基础**。它不是性能优化项目，也不要求学习者马上启动 vLLM 或 SGLang；它先把后续 Notebook 和项目共同使用的 workload、阶段划分和指标口径固定下来。

本节的核心产物不是一组性能数字，而是一份可复用的 benchmark contract：后续每次比较都应该说明同一个模型、backend、输入输出长度、batch、并发、dtype 和 cache policy。

## 问题起点

推理优化最常见的误区，不是技巧不够多，而是还没统一“在看什么”。如果 workload 没固定、指标口径没拆开，那么：

- `FlashAttention` 和 `speculative decoding` 的收益没法比较；
- `prefix reuse` 和 `KV cache quantization` 的收益会被混在一起；
- `66` 里的 benchmark report 也会退化成“换了一个配置后看起来更快”。

因此，推理优化的第一步永远不是调 kernel，而是先把请求链路和报告口径定住。

## 你要先确认什么

- workload 是否固定：模型、backend、batch、prompt tokens、generated tokens、dtype、cache policy。
- 是否拆分 prefill 和 decode，而不是只报 total latency。
- 是否同时报告 TTFT、TPOT、throughput 和 peak memory。

如果是多请求服务，还要补充请求到达模式、并发窗口、队列等待时间和 P50/P95/P99；如果是策略实验，还要补充 acceptance rate、cache hit rate 或质量约束。

## 链路骨架

```text
request
  │
  ▼
tokenize / batch assemble
  │
  ▼
prefill
  │
  ▼
KV cache
  │
  ▼
decode loop
  │
  ▼
detokenize / stream response
  │
  ▼
benchmark report
```

## 为什么这几个指标要一起看

这些指标分别对应推理链路上的不同段落，不应该互相替代：

- `TTFT` 更接近“首 token 要多久出来”，它首先受 prefill、attention kernel 和 prompt 长度影响。
- `TPOT` 更接近“后续每个 token 要多久出来”，它首先受 decode loop、KV cache 和调度影响。
- `throughput` 更接近“系统单位时间能吐多少 token”，它受 batching、调度和策略接受率影响。
- `peak memory` 是预算约束，决定 batch、上下文和 cache policy 能不能继续上去。

如果只看其中一个指标，优化方向很容易走偏。一个典型例子是：throughput 变高了，但 TTFT 也显著变差，这对在线交互往往不是好结果。

## 指标口径

| 指标 | 含义 | 主要关联 |
|:---|:---|:---|
| `TTFT` | Time To First Token，首 token 延迟 | prefill、attention kernel、chunked prefill |
| `TPOT` | Time Per Output Token | decode loop、KV cache 读写、调度 |
| `throughput` | 单位时间生成 token 数 | batching、speculative decoding、多 token 解码 |
| `peak memory` | 推理峰值显存 | 权重、KV cache、batch size、量化 |
| `prefill_share` | prefill 占总耗时比例 | prompt length、attention 访存 |
| `decode_share` | decode 占总耗时比例 | KV cache、sampling、decode scheduling |

在线服务还应区分端到端延迟的组成：

```text
e2e latency
  = queue wait
  + prefill / TTFT
  + decode time
  + detokenize / transport
```

因此，`TTFT` 变差不一定是 Attention 变慢，也可能是排队或 batch 组装时间增加；`TPOT` 变差也不一定是单个 kernel 变慢，还可能来自 cache 压力和调度等待。

## 诊断框架

把一条推理链路压成 4 个问题，会比背优化名词更稳：

1. 这个请求是 `prefill-bound` 还是 `decode-bound`？
2. 如果显存接近预算，它是不是已经变成 `memory-bound`？
3. 当前收益应该优先来自 kernel、策略、缓存管理还是量化？
4. 报告里是否能用同一 workload 证明这次变化值得保留？

可以用一个简化判断表快速分流：

| 信号 | 更可能的问题 | 下一页先看什么 |
|:---|:---|:---|
| `TTFT` 高、长 prompt 一拉长就慢 | `prefill-bound` | `02` |
| `TPOT` 高、并发时 token 吐得慢 | `decode-bound` | `03` |
| peak memory 顶到预算、batch 上不去 | `memory-bound` | `04` + `05` |
| 没有明显单点瓶颈 | 需要回到端到端判断 | `06` |

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 与 Part 02 Task0-6 的关系

本节不是简单复述 `Task0-6`。它承担的是“知识组织层”的入口作用：

- `Task0-6` 负责学习顺序，告诉读者该先读哪些 Notebook；
- `01` 负责把这些 notebook 放回同一条请求链路里，告诉读者“为什么要分 prefill、decode、KV cache、量化这几条线”；
- 因此，本节是诊断起点，而不是文件索引。

## 文献锚点

- Vaswani et al., *Attention Is All You Need*：给出 decoder 自回归推理的基础形态。
- Dao et al., *FlashAttention*：帮助理解为什么 attention 不是纯 FLOPs 问题，而是访存问题。
- Kwon et al., *vLLM / PagedAttention*：帮助理解服务系统为什么必须重新设计 cache 管理方式。

## 常见误区

- 只看 throughput，不看 TTFT。
- 不拆 prefill / decode，只报 total latency。
- workload 没固定，就比较优化结果。
- 只看单条请求，不看请求分布。

## 学习者交付物

完成本节后，至少应能写出一份最小 workload 配置和一张指标表：

| 配置或指标 | 最小记录内容 |
|:---|:---|
| workload | 模型、backend、prompt tokens、generated tokens、batch、concurrency |
| runtime | dtype、硬件、cache policy、warm-up、测量轮数 |
| latency | TTFT、TPOT、e2e latency，必要时补 P50/P95/P99 |
| capacity | throughput、peak memory、并发上限或队列等待 |
| diagnosis | prefill-bound、decode-bound、memory-bound 或 serving-bound |
| next action | 下一步进入 02、03、04、05，还是直接做项目 benchmark |

这张表不需要追求复杂，但必须能够让另一个人复现“你比较的是什么”。

## 对应项目

- **核心综合项目：** [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)，把本节的 workload contract 和指标口径用于真实对比。
- **主题项目：** [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[69 Prefix Caching Benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)。
- **扩展项目：** [68 Speculative Decoding Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)、[70 Serving Scheduler Benchmark](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)。

本节只负责统一测量口径，不替代这些项目中的策略实现和最终 `accept / tune / reject` 判断。

## 对应 Part 02

- `20` FlashAttention Sim
- `21` Decoding Strategies
- `22` vLLM PagedAttention
- `23` Speculative Decoding
- `24` SGLang RadixAttention
- `25` Quantization W8A16
- `34` Prefix Caching and Chunked Prefill
- `35` Multi-Token Decoding
- `36` Decode Scheduling
- `37` KV Cache Scheduling
- `38` Prefill-Decode Disaggregation
- `40 / 41 / 67` 量化推理与部署
- `68 / 69 / 70` 推理策略与服务扩展项目
- `66` Inference Performance Comparison

## 证据边界

CPU 可以验证请求阶段划分、指标计算和报告字段；真实 TTFT、TPOT、吞吐、P99 与峰值显存需要固定 workload 下的 GPU 或 backend 实验。本节统一测量口径，不把指标模板当成性能结论。

## 典型阅读入口

- [06 Benchmark and Decision](./06_benchmark_and_decision.md)
- [02 Prefill and Attention Kernel](./02_prefill_and_attention_kernel.md)
- [03 Decoding Strategies](./03_decoding_strategies.md)
- [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)

## 本节要点

没有统一的 workload 和指标口径，后面的推理优化都没有可比性。
