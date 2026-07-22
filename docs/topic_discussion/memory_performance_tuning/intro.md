# 显存优化与性能调优专题

## 专题概览
本专题用于沉淀 VRAM、activation、checkpointing、offload、KV cache 和 benchmark 相关内容，回答“怎么处理显存压力并做端到端性能调优”。
`Part 0E` 也是这个专题的前置桥，因为它已经把显存观察、调试和性能判断串在了一起。

## 职责边界

这个专题负责显存压力、缓存增长和端到端性能调优，不负责并行策略本身，也不负责编译链路本体。

- `VRAM / Memory Ledger` 关注内存账本、峰值占用和资源分配。
- `Activation / Checkpointing / Offload` 关注训练侧显存压力和时间换空间。
- `KV Cache` 关注推理侧缓存增长、布局和复用。
- `Benchmark / Profiling` 关注把性能问题量化成可比较指标。
- `Deployment Tuning` 关注量化、推理和部署场景中的显存/性能权衡。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 0E` | 调试、显存和性能判断的前置桥 |
| `Part 1` | VRAM 估算、memory ledger、profiling 基础 |
| `Part 2.5` | 反向传播、activation checkpointing、offload |
| `Part 2.6` | FlashAttention、KV cache、推理侧显存观察 |
| `Part 2.9` | 训练/推理性能分析、量化部署与 benchmark 闭环 |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `Part 1B` | 单卡硬件、访存和显存估算的基础入口 | [1B 单卡硬件与访存优化](../01_Hardware_Math_and_Systems/1B.md) |
| `0E` | 调试与性能前置桥，先把显存与性能判断习惯立住 | [0E 调试与性能](../../00_Prerequisites/0E.md) |
| `0E-17` | profiling 的基础入口和瓶颈定位 | [17 PyTorch Profiling Basics](../../00_Prerequisites/17_PyTorch_Profiling_Basics.md) |
| `0E-18` | 显存账本与优化手段 | [18 Memory Profiling and Optimization](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.md) |
| `0E-19` | 最小排错和异常定位 | [19 Debugging and Anomaly Localization](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.md) |
| `0E-20` | 性能判断和优化决策 | [20 Profiling and Memory Ledger](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.md) |
| `06` | VRAM 计算与 ZeRO 的显存收益 | [06 VRAM Calculation and ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md) |
| `13` | profiling 与瓶颈定位的方法入口 | [13 Profiling and Bottleneck Analysis](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) |
| `2.5` | 反向传播与显存优化主线 | [2.5 反向传播与显存优化](../02_PyTorch_Algorithms/2_5.md) |
| `19` | checkpointing / offload 的显存 trade-off | [19 Activation Checkpointing and Activation Offload](../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md) |
| `2.6` | 推理侧缓存和显存路径 | [2.6 核心推理优化](../02_PyTorch_Algorithms/2_6.md) |
| `22` | PagedAttention 的 KV cache 管理 | [22 vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) |
| `32` | 训练性能分析与显存对比项目 | [32 训练性能分析](../02_PyTorch_Algorithms/32_Training_Performance_Analysis.md) |
| `33` | profiling 驱动的端到端优化项目 | [33 Profiling Driven End-to-End Optimization](../02_PyTorch_Algorithms/33_Profiling_Driven_End_to_End_Optimization.md) |
| `35` | 量化推理与部署中的显存权衡 | [35 Quantized Inference and Deployment](../02_PyTorch_Algorithms/35_Quantized_Inference_and_Deployment.md) |

## 推荐入口

- 先看 `Part 1B / 06 / 13`，把显存和 profiling 的基础账本立住。
- 再看 `2.5 -> 19`，理解训练侧激活和 checkpointing 的显存 trade-off。
- 再看 `2.6 -> 22`，把推理侧 KV cache 的增长和复用路径看清楚。
- 最后看 `32 -> 33 -> 35`，把性能分析、优化闭环和量化部署串起来。

## 入口摘要

- 第一入口：`Part 1B` + `06 -> 13`，先把显存账本、VRAM 计算和瓶颈定位立住。
- 第二入口：`2.5 -> 19 -> 32` / `2.6 -> 22 -> 35`，把训练侧和推理侧的显存压力看清楚。
- 验证入口：`33 -> 35`，把 profiling 驱动的优化和量化部署的收益验证收进闭环。

## 正文页

- [显存优化与性能调优正文](./casebook.md)：按“训练侧 / 推理侧 / 验证侧”展开正文，适合做更细的显存案例和调优记录。
- [显存优化与性能调优深入阅读](./walkthrough.md)：按完整调优故事展开，适合想看连续推演的人。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先把瓶颈、热点和收益先量化出来时先看这里。
- [推理优化专题](../inference_optimization/intro.md)：当显存压力主要来自推理链路里的 cache、prefill 或 decode 时先看这里。
- [通信与并行专题](../communication_parallel/intro.md)：当显存压力和多卡切分、参数分摊一起出现时先看这里。

## Part 1 / Part 2 入口顺序

### Part 1 入口

- 先看 `Part 1B`，把单卡硬件、访存和显存估算的基础账本立住。
- 再看 `06 -> 13`，把 VRAM 计算、ZeRO 收益和瓶颈定位先串起来。
- 如果想补前置桥，再从 `0E -> 17 -> 18 -> 19 -> 20` 过一遍。

### Part 2 入口

- 先看 `2.5 -> 19 -> 32`，把训练侧 activation、checkpointing 和性能分析串起来。
- 再看 `2.6 -> 22 -> 35`，把推理侧 KV cache 和量化部署的显存权衡串起来。
- 最后看 `33`，把 profiling 驱动的端到端优化补成闭环。

## 读法建议

- 如果你还没看 `0E`，建议先补它，再进这个专题。
- 如果你想先补前置桥，可以按 `0E -> 17 -> 18 -> 19 -> 20` 这条线过一遍，先把显存账本、排错习惯和性能判断立住。
- 如果你关心“训练显存为什么爆”，先看 `2.5 -> 19 -> 32`。
- 如果你关心“推理显存为什么涨”，先看 `2.6 -> 22 -> 35`。
- 如果你关心“怎么证明优化有效”，先看 `13 -> 33`。

## 建设方式

- 先把入口和路径讲清楚，再把正文页里的资源对象、案例和检查清单补深。
- 新增内容优先回收到 `2.5 / 2.6 / 33 / 35` 这几条线。
- 导读页只负责告诉读者“从哪进”，不再重复正文里的判断框架。

## 专题状态
当前为专题入口页，后续将逐步补充跨 Part 索引、显存优化案例和性能调优记录。
