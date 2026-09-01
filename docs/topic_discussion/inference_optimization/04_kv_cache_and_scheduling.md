# 04. KV Cache and Scheduling | KV Cache 与调度

## 页面目标

本节回答：KV Cache 怎么增长、复用、分页和驱逐，以及多请求如何组织。

## 本节在路线中的位置

本节对应 **Task3：KV Cache 表示、复用与 Chunked Prefill**。它承接 03 的 Decode 生成循环，解释并发、长上下文和重复前缀如何把服务推到显存边界；完成后进入 Task4 的 Serving 调度，或先用 69 验证缓存复用。

本节同时覆盖 Infra-L3 运行时和 Infra-L4 服务优化：KV Cache 的布局、分页和复用属于运行时执行机制；请求排队、batch 组织和服务内存边界属于服务实例内部的优化。跨模型路由、扩缩容和灰度发布仍属于 Infra-L5，不在本节主线内。

## 问题起点

只要系统开始做长上下文、多轮对话或高并发服务，KV cache 就很快从“实现细节”变成“系统边界”：

- cache 会随着层数、上下文长度和 batch 线性增长；
- 一旦 cache 顶到预算，batch、上下文和并发都上不去；
- 就算还没 OOM，碎片、分页和调度也会直接拖慢 TPOT。

这就是为什么 `KV cache` 会同时出现在推理优化和显存优化里，但两边看的目标不同。

## 你要先确认什么

- peak memory 是否接近预算。
- 长上下文和并发请求是否把 cache 撑爆。
- prefix reuse 是否有明显收益。

还要固定请求分布、prompt 长度、generated tokens、batch、并发窗口和 cache policy；否则 cache hit rate、TTFT 和显存变化没有可比性。

## 核心矛盾

KV cache 的核心矛盾是：它既是 decode 提速所需的缓存，又是推理侧最稳定增长的显存对象。系统既希望尽量保留更多上下文，又希望不要因为 cache 组织方式把吞吐和预算一起拖垮。

## 演化路径

KV cache 是推理链路里最容易成为硬约束的部分。

1. cache 会随层数、head 数、长度和 batch 持续增长。
2. prefix caching 让重复前缀尽量复用。
3. PagedAttention 把连续缓存变成块管理。
4. RadixAttention 让前缀树式复用更高效。
5. decode scheduling 决定请求怎样错峰和排序。

## 关键取舍

- `prefix caching` 更适合重复前缀明显的 workload，不是所有请求都会收益。
- `PagedAttention` 改的是 cache 管理粒度，收益常体现在碎片和并发上。
- `RadixAttention` 更强调前缀共享和树式组织，但也要求请求模式与系统实现匹配。
- `KV cache quantization` 能继续压预算，但不应替代复用、分页和调度本身。

因此，本节应先判断 cache 是否成为硬约束，再决定采用复用、分页、调度还是压缩。单请求的 cache 表示、增长和复用属于 Task3；多请求的资源池、排队、PD 分离和调度属于 Task4。

## 学习者交付物

完成本节后，至少应形成一份 KV Cache / Serving 判断：

| 项目 | 最小内容 |
|:---|:---|
| 增长边界 | 层数、KV heads、上下文长度、batch 和并发如何影响 cache |
| 请求证据 | prefix reuse、cache hit rate、TTFT、TPOT 和 peak memory |
| 管理策略 | paging、prefix reuse、cache policy、调度或容量限制 |
| 服务代价 | cache 维护、碎片、排队、batch 变化和公平性影响 |
| 下一步 | 进入 05 看量化，或用 69 / 70 做主题项目验证 |

核心结论应能够区分：当前问题是 cache 容量不够、cache 复用不足、内存碎片，还是请求调度没有把可并行的工作组织起来。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- Kwon et al., *vLLM / PagedAttention*：理解为什么服务系统必须把 cache 当块管理。
- SGLang / RadixAttention 相关资料：理解前缀树式复用的系统收益。
- prefix caching / chunked prefill 工程资料：理解复用与 prefill 分块的协同关系。

## 常见误区

- 把 KV cache 当成纯实现细节，不看它的增长曲线。
- 只看单请求，不看并发。
- 看到 cache 占用高就直接量化，不先看复用和调度。

## 对应 Part 02

- `22` vLLM PagedAttention
- `24` SGLang RadixAttention
- `34` Prefix Caching and Chunked Prefill
- `37` KV Cache Scheduling
- `41` FP8 and KV Cache Quantization

## 证据边界

CPU 可以验证 KV Cache shape、容量估算、分页逻辑和请求调度模拟；真实 cache 命中率、分页收益、并发容量、TTFT / TPOT 和服务显存需要 vLLM / SGLang 等 backend 的 GPU 实验。单请求模拟不能证明多请求服务收益。

## 经典阅读入口

- [11 KV Cache and Memory Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md)
- [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.md)

## 相关跳转

- 看 `01`，确认指标口径。
- 看 `03`，确认 decode 循环怎么耗时。
- 看 `05`，确认 cache 不够时怎么压缩。

## 对应项目

- **核心主题项目：** [69 Prefix Caching Benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)，验证请求复用模式、cache hit rate、TTFT 和维护成本。
- **扩展项目：** [70 Serving Scheduler Benchmark](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md)，验证队列、batch、TTFT、TPOT、吞吐和公平性。
- **综合项目：** [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)，把 Cache 和调度候选放回统一 workload 做最终比较。

没有真实 backend 时，可以先完成请求分布、命中行为和调度逻辑的 Practice-P1 模拟；接入 vLLM / SGLang 后，再升级为 Practice-P2，验证真实服务中的分页、复用和调度代价。

## 本节要点

KV cache 不是附属缓存，而是推理吞吐和上下文长度的核心边界。
