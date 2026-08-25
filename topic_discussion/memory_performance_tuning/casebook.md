# 显存优化与性能调优正文

这页只做显存问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## Infra 边界

显存优化主要连接 Infra-L1 的容量与带宽、Infra-L2 的访存与 kernel、Infra-L3 的框架状态管理，并延伸到 Infra-L4 的 KV Cache、量化和 Serving 配置。每项策略都要同时检查容量、重算、搬运、通信、吞吐和质量；Infra-L5 只负责资源治理、回归和部署流程。

## 总判断链：对象 → 生命周期 → 证据 → 策略

遇到显存问题时，先沿下面四步走，不要直接从技巧名称开始：

1. **确认对象。** 先区分参数、梯度、optimizer state、activation、KV Cache 和临时 buffer；训练和推理的对象账本不能混写。
2. **确认生命周期。** 判断对象是在 forward、backward、optimizer step、prefill、decode 还是请求排队阶段驻留或增长。
3. **确认证据。** 区分公式 / toy 机制、CPU 功能验证、单 GPU benchmark 和真实 profiler trace；证据等级不足时，只保留待验证假设。
4. **选择策略。** 根据对象和瓶颈选择 accumulation、checkpoint、offload、paging、prefix reuse、量化或 kernel / graph 优化，并记录代价转移到了计算、带宽、通信、延迟还是质量。

这四步对应整条路线：Task0–1 建立对象和生命周期语言，Task2 学习训练侧策略，Task3 采集训练证据，Task4–5 处理推理与量化分支，Task6 负责跨策略的 profiling 和决策收口。

## 同一机制的显存目标

显存专题与推理专题可以共享同一份机制 Notebook，但这里把 checkpoint、offload、paging、KV Cache 和量化看成资源预算策略：先确认减少了哪类状态的驻留，再判断代价转移到了重算、带宽、通信还是质量。

| 共同机制 | 显存侧要回答的问题 | 主要指标 | 项目输出 |
|:---|:---|:---|:---|
| Checkpointing | 少保存了多少 activation，重算代价是否可接受 | `peak memory`、step time、吞吐 | 训练显存策略选型 |
| Offload | 状态搬到 CPU 或其他层级后是否仍值得 | GPU 显存、搬运时间、带宽、吞吐 | offload 范围决策 |
| PagedAttention / KV Cache | cache 碎片和驻留是否限制并发 | 每请求显存、cache 容量、并发、质量 | cache 预算决策 |
| 权重 / 激活 / KV Cache 量化 | 哪类状态被压缩，模型是否因此装得下 | 权重或 activation 占用、`peak memory`、质量、吞吐 | 量化预算决策 |

量化在本专题中首先是显存工具：先回答模型是否装得下、上下文或并发是否能提高，再验证速度和质量；不能从单项显存下降直接推出优化成功。

## 判断表

先按“对象 → 生命周期 → 证据 → 策略”的顺序分流，再判断省下来的显存有没有把时间代价一起控制住。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 训练前几步正常，中后段突然 OOM | `training activation pressure` | [02](./02_training_memory_pressure.md), [03](./03_checkpointing_and_offload.md) | 检查 batch、accumulation、checkpointing、offload |
| 推理能跑，但 cache 一直涨，batch 上不去 | `inference cache pressure` | [04](./04_inference_cache_and_memory_budget.md) | 检查 paging、prefix reuse、eviction、KV cache quant |
| 峰值显存下降了，但 benchmark 没改善 | `trade-off mismatch` | [06](./06_benchmark_and_tradeoff_decision.md) | 比较 peak memory、step time、TTFT、TPOT、throughput |
| 理论账本和实测差很多 | `ledger mismatch` | [01](./01_vram_ledger_and_metrics.md), [06](./06_benchmark_and_tradeoff_decision.md) | 对齐理论账本、运行时 buffer、碎片和流程开销 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| `activation` | 训练侧主峰值是不是来自前反向中间状态 | 把所有问题都归到 batch 太大 |
| `optimizer state` | 更新状态是不是把预算继续抬高 | 只看参数量，不看更新状态驻留 |
| `KV cache` | 推理侧预算是不是被缓存增长顶高 | 看到延迟差就直接改 decode |
| `peak memory + time` | 省显存是否把时间和吞吐一起赔掉 | 峰值降了就默认 adopt |

最终判断不该停在“省了多少显存”，而要落回“系统是不是因此更可运行、更稳定、更值得保留”。

## 本节要点

这页的职责不是列出更多省显存的方法名，而是把显存问题里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`，项目证明留给 benchmark 和项目页。
