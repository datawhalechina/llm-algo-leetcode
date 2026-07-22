# 通信与并行专题

## 专题概览
本专题用于沉淀 NCCL、AllReduce、ZeRO、Pipeline Parallelism 和 Tensor Parallelism 等多卡扩展方法，回答“怎么把模型扩到多卡并看懂通信代价”。

## 职责边界

这个专题只负责多卡训练和推理中的并行策略、通信代价和调度边界，不负责单机推理优化本身，也不负责编译链路。

- `NCCL / AllReduce` 关注最基础的通信原语和同步代价。
- `ZeRO` 关注参数、梯度和优化器状态的切分与显存分摊。
- `Pipeline Parallelism` 关注 micro-batch 的流水线时序和气泡问题。
- `Tensor Parallelism` 关注张量切分后的通信与计算平衡。
- `Communication Profiling` 关注通信热点、等待时间和 overlap。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1C` | 通信拓扑、显存共享、NCCL / AllReduce、并行策略判断 |
| `Part 2.8` | ZeRO、Pipeline Parallelism、Tensor Parallelism 的主线实现 |
| `Part 2.9` | 分布式并行基准项目和工程选型验证 |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `1C` | 多卡通信与显存共享的总入口，先看 Group Overview / Asset Overview / Learning Path | [1C 多卡通信与显存共享](../01_Hardware_Math_and_Systems/1C.md) |
| `05` | 通信拓扑和显存共享关系，适合先看页面里的核心职责与判断链路 | [05 Communication Topologies](../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb) |
| `06` | 显存开销与 ZeRO 收益估算，适合先看公式与对比结论 | [06 VRAM Calculation and ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) |
| `20` | NCCL 和 AllReduce 基础原语，适合先看原语定义和通信语义 | [20 NCCL and AllReduce Basics](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb) |
| `2.8` | 分布式并行策略主线，适合先看 Group Overview / Asset Overview / Learning Path | [2.8 分布式并行策略](../02_PyTorch_Algorithms/2_8.md) |
| `27` | ZeRO 的显存分摊与收益，适合看 Step 1-4 的收益与代价 | [27 ZeRO Optimizer Sim](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.ipynb) |
| `28` | Pipeline 的 micro-batch 时序，适合看 Step 1-4 的气泡和排布 | [28 Pipeline Parallelism MicroBatch](../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.ipynb) |
| `29` | Tensor Parallelism 的通信开销，适合看 Step 1-4 的切分与代价 | [29 Tensor Parallelism Sim](../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.ipynb) |
| `42` | NCCL 通信热点与等待时间，适合先看 Step 1-4 的观测流程 | [42 Communication Profiling with NCCL](../02_PyTorch_Algorithms/42_Communication_Profiling_with_NCCL.ipynb) |
| `34` | 分布式并行基准项目，适合先看 Step 1-4 的实验设置和结果汇总 | [34 Distributed Parallel Benchmark](../02_PyTorch_Algorithms/34_Distributed_Parallel_Benchmark.ipynb) |

## 推荐入口

- 先看 `Part 1C`，把“为什么多卡一定会带来通信问题”先立住。
- 再看 `2.8`，把 ZeRO、Pipeline 和 Tensor Parallelism 的策略边界串起来。
- 如果想看通信代价如何被量化，再回到 `42` 和 `34`。

## 入口摘要

- 第一入口：`Part 1C` + `05 -> 20 -> 06 -> 13`，先把通信原语、显存分摊和观测基础立住。
- 第二入口：`2.8 -> 27 -> 28 -> 29`，把 ZeRO、Pipeline 和 Tensor Parallelism 的主线补齐。
- 验证入口：`42 -> 34 -> 31`，把通信热点、分布式基准和最终收益连起来。

## 正文页

- [通信与并行正文](./casebook.md)：按“通信原语 / 并行切分 / 调度代价 / 基准验证”展开专题正文，适合做更细的选型和对照。
- [通信与并行深入阅读](./walkthrough.md)：按完整并行选型过程展开，适合想看连续推演的人。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先确认瓶颈在通信、算子还是调度时先看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当并行策略和显存分摊、cache 压力一起出现时先看这里。
- [编译与图优化专题](../compiler_graph_optimization/intro.md)：当通信策略和执行模型、backend 约束一起分析时先看这里。

## Part 1 / Part 2 入口顺序

### Part 1 入口

- 先从 `Part 1C` 进入，把通信拓扑、显存共享和多卡边界先立住。
- 再看 `05 -> 20 -> 06 -> 13`，把通信原语、显存收益和 profiling 观测串起来。

### Part 2 入口

- 先看 `2.8 -> 27 -> 28 -> 29`，把 ZeRO、Pipeline 和 Tensor Parallelism 的主线补齐。
- 再看 `42 -> 34`，把通信 profiling 和分布式 benchmark 连接起来。
- 如果需要回看收益证明，再回到 `31` 看最终验证口径。

## 典型阅读链

- 如果你想先理解多卡通信原理，先读 `05 -> 20`，把通信拓扑和 AllReduce 先讲通。
- 如果你想先理解显存是怎么被并行策略切开的，先读 `06 -> 27`，把 ZeRO 的收益和代价讲清楚。
- 如果你想先理解流水线为什么会有气泡，先读 `28 -> 34`，把 micro-batch、排布和基准结果串起来。
- 如果你想先理解张量切分的通信压力，先读 `29 -> 42`，把切分方式和通信热点串起来。
- 如果你想先看并行策略值不值，先读 `42 -> 34 -> 31`，把通信 profile、分布式 benchmark 和最终收益连起来。

## 读法建议

- 如果你关心“通信原语怎么工作”，先看 `05 -> 20`。
- 如果你关心“多卡训练怎么切”，先看 `06 -> 27 -> 28 -> 29`。
- 如果你关心“怎么证明并行策略值不值”，先看 `42 -> 34`。
- 如果你想先补前置桥，可以先看 `Part 1C` 的 Group Overview，再回到 `05 / 06 / 20`。
- 如果你关心“如何把并行策略和性能验证连起来”，先看 `06 -> 27 -> 28 -> 29 -> 42 -> 34`。

## 建设方式

- 先补通信原语和策略边界，再补正文页里的案例、对照和误区。
- 优先从 Part 1C / Part 2.8 里抽取高频、稳定、可复用的结论。
- 让正文页专注回答“通信代价来自哪里、并行策略换来了什么”。
- 后续新增内容时，优先沿着 `通信原语 -> 并行切分 -> profiling -> benchmark` 这条线放到正文页。

## 专题状态
当前为专题入口页，后续将逐步补充跨 Part 索引、并行策略案例和通信分析。
