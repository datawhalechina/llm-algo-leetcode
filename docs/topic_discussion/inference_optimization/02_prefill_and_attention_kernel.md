# 02. Prefill and Attention Kernel | Prefill 与 Attention Kernel

## 页面目标

本节回答：长 prompt 为什么会慢，FlashAttention 和 chunked prefill 具体改变了哪一段。

## 本节在路线中的位置

本节对应 **Task1：Attention 访存瓶颈与 Prefill**。它承接 01 的请求链路和指标口径，专门解释长 Prompt 下 `TTFT` 为什么升高；完成后进入 Task2 的 Decode，或在理解完整主线后进入 Task3 判断 Prefix Cache、KV Cache 和服务调度是否成为新的瓶颈。

本节不要求学习者手写完整 CUDA kernel，也不把 FlashAttention、chunked prefill 和 prefix caching 当成同一种优化。三者分别对应访存路径、长输入分块和重复前缀复用。

## 问题起点

推理链路里，首 token 延迟往往最先暴露出 prefill 的代价。用户感受到的是“输入一大段上下文后，模型迟迟不出第一个 token”，但真正的问题常常不是模型参数量本身，而是：

- prompt 太长导致 attention 访存和中间写回膨胀；
- prefill 把大量已有 token 一次性送进模型，导致 `TTFT` 被这一段主导；
- backend 还在用对短 prompt 友好的实现，遇到长 prompt 就开始掉速。

## 你要先确认什么

- TTFT 是否在长 prompt 下明显升高。
- `prefill_share` 是否高于 decode。
- attention 是否被中间 score 矩阵和 HBM 读写拖慢。

最好同时记录 prompt length、`TTFT`、`prefill_share`、batch、dtype 和 attention backend；如果比较长短输入，还要保证 generated tokens 和其他运行条件一致。

## 核心矛盾

prefill 的核心矛盾不是“算力够不够”，而是“访存和中间结果要不要反复写回 HBM”。长上下文下，attention 的理论复杂度大家都知道，但真正把 TTFT 顶高的，往往是中间 score、softmax 和 value 聚合带来的内存路径。

## 演化路径

prefill 不是“先算一遍前向”这么简单。它要把已有 prompt 组织成上下文，同时完成 attention 计算。

1. prompt 变长后，中间矩阵和带宽压力上升。
2. naive attention 往往被 HBM 读写拖慢。
3. FlashAttention 通过 tiling 和 online softmax 减少中间写回。
4. chunked prefill 进一步把长 prompt 分块处理。
5. 最终目标是把 TTFT 压下来，而不是只看 FLOPs。

## 关键取舍

这条线的 trade-off 很明确：

- `FlashAttention` 主要换来更好的访存路径，但要求 kernel 和 backend 更匹配；
- `chunked prefill` 主要解决超长 prompt 的工程落地问题，但会改变调度和 cache 的接入方式；
- `prefix caching` 可以减少重复 prefill，但它解决的是“重复前缀”而不是“所有长 prompt 都慢”。

因此，看到 TTFT 高时，不能把这三者混成一个动作，它们处理的是不同层面的瓶颈。

## 学习者交付物

完成本节后，至少应形成一条可复查的 Prefill 判断：

| 项目 | 最小内容 |
|:---|:---|
| 症状 | prompt length 增长时 TTFT 如何变化 |
| 阶段证据 | prefill_share、decode_share、必要时的 profiler 统计 |
| 候选动作 | FlashAttention、chunked prefill 或 prefix caching |
| 适用条件 | 长 prompt、重复前缀、backend 和硬件要求 |
| 下一步 | 进入 03 看 Decode，进入 04 看 Cache，或交给 66 做端到端比较 |

核心结论应能够区分：是 Attention 访存导致 Prefill 变慢，还是排队、batch 组装或重复前缀导致 TTFT 变差。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- Dao et al., *FlashAttention*：理解 online softmax 和 tiling 为何能显著减少 HBM 写回。
- Dao, *FlashAttention-2*：关注并行分工和 kernel 落地如何继续改进吞吐。
- chunked prefill 相关工程资料：帮助理解长 prompt 在服务系统里的分块处理方式。

## 常见误区

- 只看 FLOPs，忽略 HBM/SRAM 访存。
- 把 prefill 慢简单等同于模型本身慢。
- chunked prefill 和 prefix caching 混为一谈。

## 对应 Part 02

- `20` FlashAttention Sim
- `34` Prefix Caching and Chunked Prefill
- `66` Inference Performance Comparison

## 证据边界

CPU 可以验证 Attention shape、分块逻辑和 IO 模型；真实 FlashAttention kernel、显存访问、带宽和 TTFT 变化需要匹配硬件与 backend 的 GPU 实验。chunked prefill 的服务收益也必须在固定 prompt 分布下验证。

## 经典阅读入口

- [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)
- [24 SRAM Optimization Techniques](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)
- [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)

## 相关跳转

- 看 `01`，确认指标口径。
- 看 `04`，确认 prefill 结束后 cache 怎么接。

## 对应项目

- **核心综合项目：** [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)，在固定 prompt length 和 generated tokens 下验证 TTFT 变化。
- **相关主题项目：** [69 Prefix Caching Benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)，当问题主要来自重复前缀时再进入。

本节负责 Prefill 和 Attention 的瓶颈解释，不单独裁定某个 backend 永远更快；最终仍需回到统一 workload 做比较。

## 本节要点

prefill 优化的重点是减少访存和中间写回，把首 token 延迟压下来。
