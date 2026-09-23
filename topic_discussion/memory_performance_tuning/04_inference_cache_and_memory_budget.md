# 04. Inference Cache and Memory Budget | 推理缓存与显存预算

## 页面目标

本节对应 **Task4：推理侧 KV Cache 与容量**。这里回答为什么上下文和并发增加后 KV Cache 会成为推理显存边界，以及分页、前缀复用、缓存组织和 KV Cache 量化分别改变了什么。重点是显存对象、容量和驻留策略；请求如何排队、如何组成批次和如何满足 P99，进入[推理优化 Task4](../inference_optimization/intro.md)。

## 核心机制

### KV Cache 增长与容量账本

KV Cache 保存历史 token 对应的键和值，使 decode 不必重复计算全部历史上下文；代价是每层、每个请求和每个历史 token 都会占用运行时显存。上下文越长、并发越高，缓存增长越接近容量上限。

| 机制 | 主要改变什么 | 显存侧观察点 | 相关入口 |
|:---|:---|:---|:---|
| KV Cache 增长 | 历史 token 的运行时驻留 | 每 token、每请求和每并发的容量 | [11 KV Cache](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) |
| Paging | cache block 的组织和分配 | 碎片、可用容量和回收 | [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) |
| Prefix reuse | 重复前缀的存储复用 | 命中率、复用容量和失效 | [34 Prefix Cache](../../02_PyTorch_Algorithms/34_Prefix_Cache_Matching_and_Reuse.ipynb) |
| KV Cache 量化 | cache 的数值表示 | cache 占用、误差和容量收益 | [41 KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)；完整量化路线见 Task5 |

这里关注的是“缓存如何在有限显存中驻留和增长”，不是请求路由、扩缩容或版本治理。同一个 66 项目在推理优化路线关注服务延迟和吞吐，在本专题只提取 cache policy、命中率、峰值显存和并发容量，作为容量证据。

容量估算先使用一个统一的近似式：

```text
KV bytes / request
≈ 2 × layers × kv_heads × head_dim × tokens × dtype_bytes
```

其中 `2` 表示 K 和 V 两份状态，`tokens` 包含当前请求实际保留的上下文；并发请求数、block 元数据、对齐浪费和 backend workspace 还会继续增加实际占用。这个公式用于提出容量假设，不等于某个 backend 的最终峰值。

### 分页、复用与容量治理

Prefill 和 Decode 对 Cache 的压力也不同：Prefill 一次写入较长的历史片段，Decode 逐 token 追加并反复读取已有 Cache。分页主要改变 block 的分配和回收方式，Prefix Cache 改变重复前缀是否可以复用，eviction 则决定容量不足时保留哪些状态。它们都可能降低浪费，但不保证同时改善 TTFT、TPOT 和吞吐。

| 机制 | 主要改变 | 需要记录的证据 | 可能转移的代价 |
|:---|:---|:---|:---|
| 连续 Cache | 直接按请求保留 K/V | 容量增长、碎片、OOM | 预留浪费、扩展困难 |
| 分页 Cache | block 分配、回收和复用 | block 利用率、碎片、eviction | 管理开销、访问间接层 |
| Prefix Cache | 重复前缀的状态复用 | 命中率、复用 token、失效原因 | 失效管理、额外元数据 |
| Cache 量化 | 每个 K/V 元素的表示宽度 | dtype、误差、质量、backend kernel | 数值误差、转换或 kernel 代价 |

![KV Cache：增长、组织与预算](../../docs/public/topic_discussion/memory_performance_tuning/kv_cache_budget.svg)

## 判断与验证

推理侧显存问题按下面顺序进入实验：

1. 先用 shape 和容量估算判断 KV Cache 是否是主要压力。
2. 再观察分页或前缀复用能否减少碎片和重复驻留。
3. 预算仍不足时，再转入 Task5 判断 KV Cache 量化，或进入架构扩展，例如 [71 MLA / KV Cache](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.ipynb)。
4. 最后在匹配的 GPU backend workload 中记录 cache 容量、并发和 OOM 状态；TTFT、TPOT、吞吐和质量作为容量策略的服务侧补充证据，不在本节展开调度归因。

进入项目前，至少把下面几类观测字段分开记录：

| 观察对象 | 需要记录什么 | 主要回答的问题 |
|:---|:---|:---|
| Cache 容量 | 每 token、每请求、每并发的增长 | 当前容量边界由什么决定？ |
| Cache 组织 | block 分配、碎片、回收和 eviction | paging 是否减少了浪费？ |
| Cache 复用 | prefix 命中率、复用长度、失效原因 | 复用是否减少重复驻留和 prefill？ |
| 请求性能 | TTFT、TPOT、吞吐、P99 | 显存收益是否带来可接受的服务结果？详细调度归因转到推理优化 Task4 |
| 数值与质量 | cache dtype、误差、任务质量 | 压缩是否改变了可用性？ |

CPU 可以验证 cache shape、容量公式、分页和前缀匹配逻辑；真实命中率、eviction、并发容量和 backend 显存必须由 [66 推理 baseline](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 或 [69 Prefix Cache benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb) 验证。缓存模型不能直接写成 vLLM 或 SGLang 的实测结论；如果需要解释队列、批次和服务尾延迟，继续阅读推理优化的 Task4。
