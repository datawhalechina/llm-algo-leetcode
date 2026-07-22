# 推理优化专题

## 专题概览
本专题用于沉淀 FlashAttention、解码、PagedAttention、prefix caching 和 speculative decoding 等推理加速方法，回答“怎么让推理更快、更稳、更省 cache”。

## 职责边界

这个专题只负责推理链路里的性能优化与缓存管理，不负责训练流程本身，也不负责分布式并行主线。

- `FlashAttention` 关注 attention 计算与显存模型。
- `Decoding` 关注采样、搜索和生成阶段的策略选择。
- `PagedAttention` 关注 KV cache 的分页管理与连续 batching。
- `Prefix Caching / Chunked Prefill` 关注前缀复用与预填充调度。
- `Speculative Decoding / Multi-Token Decoding / Decode Scheduling` 关注更高吞吐的生成路径。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1` | attention / memory / profiling 背景，推理优化的硬件约束 |
| `Part 2.6` | FlashAttention、Decoding Strategies、PagedAttention |
| `Part 2.7A` | Speculative Decoding、RadixAttention、Prefix Caching、Chunked Prefill、Multi-Token Decoding、Decode Scheduling |
| `Part 2.9` | 推理性能对比实验、推理路径回改和工程验证 |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `2.6` | 推理优化的三条主线：FlashAttention、解码策略、PagedAttention | [2.6 核心推理优化](../02_PyTorch_Algorithms/2_6.md) |
| `20` | FlashAttention 的分块与 online softmax 思路 | [20 FlashAttention Sim](../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) |
| `21` | temperature / top-k / top-p 的解码策略 | [21 Decoding Strategies](../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb) |
| `22` | KV cache 的分页管理与 block table | [22 vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) |
| `2.7A` | 更快生成的高级策略入口 | [2.7A 高级推理策略](../02_PyTorch_Algorithms/2_7A.md) |
| `36` | Prefix Caching 和 Chunked Prefill 的复用路径 | [36 Prefix Caching and Chunked Prefill](../02_PyTorch_Algorithms/36_Prefix_Caching_and_Chunked_Prefill.ipynb) |
| `37` | Multi-Token Decoding 的草稿-验证链路 | [37 Multi-Token Decoding](../02_PyTorch_Algorithms/37_Multi_Token_Decoding.ipynb) |
| `38` | Decode Scheduling 的排布、优先级和吞吐收益 | [38 Decode Scheduling](../02_PyTorch_Algorithms/38_Decode_Scheduling.ipynb) |
| `41` | KV Cache 调度边界与复用/驱逐策略 | [41 KV Cache Scheduling](../02_PyTorch_Algorithms/41_KV_Cache_Scheduling.ipynb) |
| `31` | 推理性能对比实验与收益验证 | [31 Inference Performance Comparison](../02_PyTorch_Algorithms/31_Inference_Performance_Comparison.ipynb) |

## 推荐入口

- 先看 `Part 2.6`，把 FlashAttention、解码和 PagedAttention 串起来。
- 再看 `Part 2.7A`，把高级推理策略和调度链路补齐。
- 最后看 `Part 2.9`，把这些能力放到项目里验证收益。

## 入口摘要

- 第一入口：`Part 1` + `2.6 -> 20 -> 21 -> 22`，先把 attention、解码和 cache 的基础主线立住。
- 第二入口：`2.7A -> 36 -> 37 -> 38 -> 41`，把前缀复用、高级生成和调度串起来。
- 验证入口：`31` + `2.9`，把推理优化的收益放到 benchmark 和项目里验证。

## 正文页

- [推理优化正文](./casebook.md)：按“场景识别 / 栈位关系 / 典型链路 / 误区 / FAQ”展开正文，适合做更细的案例和对照。
- [推理优化深入阅读](./walkthrough.md)：按完整请求链路展开，适合想看连续推演的人。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先判断慢在哪里、用什么指标证明时先看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当推理优化和 KV cache、显存账本绑在一起时先看这里。
- [编译与图优化专题](../compiler_graph_optimization/intro.md)：当问题更像 backend 选择、fusion 或调度差异时先看这里。

## Part 1 / Part 2 入口顺序

### Part 1 入口

- 先从 `Part 1` 的 attention / memory / profiling 背景进入，建立推理优化的硬件约束感。
- 再回到 `20 -> 22`，把 FlashAttention 和 KV cache 的基本行为先看清楚。

### Part 2 入口

- 先看 `2.6 -> 20 -> 21 -> 22`，把基础推理优化三条主线连起来。
- 再看 `2.7A -> 36 -> 37 -> 38 -> 41`，把前缀复用、高级生成和调度串起来。
- 最后看 `31`，把前面的机制放到 benchmark 里验证收益。

## 读法建议

- 如果你关心“为什么推理慢”，先看 `20 -> 22`。
- 如果你关心“生成时怎么选 token”，先看 `21`。
- 如果你关心“怎么把吞吐做上去”，再看 `36 -> 38 -> 41 -> 31`。
- 如果你关心“缓存怎么被复用和驱逐”，先看 `22 -> 36 -> 41`。
- 如果你关心“高级生成策略怎么组合”，先看 `2.7A -> 37 -> 38`。

## 建设方式

- 先把入口、路径和验证点讲清楚。
- 正文页再展开具体栈位、链路和典型案例。
- 后续新增内容优先沿着 `2.6 / 2.7A / 2.9` 三条线回收。

## 专题状态
当前为专题入口页，后续将逐步补充跨 Part 索引、推理优化案例和工程化拆解。
