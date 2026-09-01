# 推理优化正文

这页只做推理问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## Infra 边界

推理优化主要落在 Infra-L3 框架与运行时、Infra-L4 服务与模型优化，依赖 Infra-L2 系统软件与加速库，并受 Infra-L1 GPU 算力、显存和带宽约束。Infra-L5 的版本治理、灰度发布、扩缩容和流量治理属于平台交付，不是本专题的核心实验范围。

## 同一机制的推理目标

推理专题与显存专题可以共享同一份机制 Notebook，但不共享同一套问题口径。这里把解码、KV Cache、调度和量化看成请求执行策略：先理解共同机制，再判断它是否改善服务体验。

| 共同机制 | 推理侧要回答的问题 | 主要指标 | 项目输出 |
|:---|:---|:---|:---|
| 解码 / speculative decoding | 是否减少生成阶段的有效成本 | `TPOT`、吞吐、接受率、质量 | 解码策略选型 |
| KV Cache / PagedAttention | 是否减少重复计算并支持更高并发 | `TTFT`、cache 命中率、并发、`peak memory` | cache / 调度策略选型 |
| Continuous Batching / scheduling | 是否提高服务整体利用率 | 吞吐、排队延迟、P99、GPU 利用率 | serving 配置选型 |
| 权重 / 激活 / KV Cache 量化 | 是否在质量可接受时改善服务成本 | TTFT、TPOT、吞吐、显存、质量 | backend / dtype / quantization 选型 |

量化在本专题中首先是部署策略，不因为显存下降就自动代表服务变快；必须在固定 workload、backend 和并发条件下做端到端比较。

## 判断表

先分清问题在 `prefill`、`decode`、`cache / scheduling` 还是 `deployment`，再统一 `TTFT / TPOT / throughput / peak memory` 口径，最后回到同一 workload，把候选方案收成 `accept / tune / reject`。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 长 prompt 下首 token 明显变慢 | `prefill-bound` | [02](./02_prefill_and_attention_kernel.md) | FlashAttention、chunked prefill、prefix caching |
| 并发一高，生成速度掉下去 | `decode-bound` | [03](./03_decoding_strategies.md) | speculative decoding、multi-token decoding、decode scheduling |
| cache 一边跑一边涨，batch 上不去 | `memory-bound` | [04](./04_kv_cache_and_scheduling.md) | paging、prefix reuse、eviction、KV cache quant |
| 显存降了，但交互体验变差 | `deployment trade-off` | [05](./05_quantized_inference_and_deployment.md) + [06](./06_benchmark_and_decision.md) | 区分权重量化、KV cache quant、FP8，再回 benchmark |

显存已经接近预算时，优先把它当成硬约束处理；即使 decode 也慢，继续上 batch 或上下文都不可靠。

| 指标 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| `TTFT` | 首 token 是否被 prefill 拖慢 | 只看总时延，看不出 prefill 问题 |
| `TPOT` | decode 阶段每 token 是否过慢 | 把 decode 慢误判成模型整体慢 |
| `throughput` | 系统单位时间产出是否够高 | 只看吞吐，不看交互延迟 |
| `peak memory` | 当前配置是否还能继续推 batch / context | 不把它当硬约束，只看速度 |
| `prefill_share` / `decode_share` | 主要时间花在哪一段 | 没拆阶段，无法判断下一步该改哪里 |

`66` 的价值不在于再讲机制，而在于把这些指标放回同一 workload 比较 baseline 和 candidate。

## 本节要点

这页的职责不是列技巧，而是把症状、指标和下一步动作压成一张判断表。路线入口留给 `intro`，连续故事留给 `walkthrough`，项目证据留给 `66` 及其主题 / 扩展项目，最终决策规则留给 `06`。
