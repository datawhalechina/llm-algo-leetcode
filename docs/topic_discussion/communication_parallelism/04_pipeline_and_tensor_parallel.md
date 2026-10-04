# 04. Model, Context, and Request Partitioning | 模型、上下文与请求切分

## 本页目标与路线位置

状态分片主要改变训练状态的驻留方式；PP、TP、CP 和 Serving Replica 则分别沿层、张量、序列和请求切分工作。四者可以组合，但切分对象、通信位置和判断指标不能混用。

![层、张量、序列与请求四种切分维度](../../public/topic_discussion/communication_parallelism/partition_axes.svg)

## 核心机制

### 四种切分维度

本页建立切分坐标系和组合约束，不替代 PP、TP、CP 或副本路由的具体机制页。判断重点是每一种方案切开什么、在哪里交换，以及应读取哪组证据。

| 机制 | 切分对象 | 主要通信 | 首先观察的代价 | 适用信号 |
|:---|:---|:---|:---|:---|
| Pipeline Parallel（PP） | 连续层 / stage | stage 间 activation 与 backward gradient | bubble、stage 不均衡 | 模型很深，单卡放不下层序列 |
| Tensor Parallel（TP） | 单层矩阵、特征维、Attention Head | All-Reduce、All-Gather、Reduce-Scatter | 高频层内同步 | 单层矩阵或 Attention 本身过大 |
| Context Parallel（CP） | sequence / context | K/V 或中间状态的环式交换 | 通信随上下文增长 | 长序列状态无法由单卡承载 |
| Serving Replica | 请求 | 请求路由、队列与缓存局部性 | 负载不均、P95/P99 | 模型可由单实例承载，但请求规模扩大 |

### 组合并行先写 Device Mesh

组合策略应先写出 world size 如何分解，再为每个 mesh 维度指定切分对象和进程组。`DP × TP × PP × CP / EP` 不是越多越好；每增加一个维度，就增加一种通信模式和配置约束。

| 组合 | 主要收益 | 新的组合约束 |
|:---|:---|:---|
| PP + TP | 同时切深度和单层宽度 | stage 内 collective 与 stage 间 P2P 可能同时暴露 |
| TP + CP | 同时扩展模型宽度和上下文 | Head 映射、序列布局和通信组必须兼容 |
| DP/FSDP + TP | 扩吞吐并切单层状态 | 数据并行组与 TP 组不能混淆 |
| Replica + TP | 单实例多卡，同时扩展请求容量 | 路由应面向实例，而不是单个 TP rank |

## 选择条件与验证证据

| 切分问题 | 机制页 | 真实证据出口 |
|:---|:---|:---|
| stage 与 micro-batch | [Part 02 · 28 Pipeline](../../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md) | [PAR-MODEL-INTERNAL](../../02_PyTorch_Algorithms/PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.md) 的 PP 分支 |
| Column / Row 与 Head 布局 | [Part 02 · 29 Tensor Parallel](../../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md) | [PAR-MODEL-INTERNAL](../../02_PyTorch_Algorithms/PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.md) 的 TP 分支 |
| 长上下文序列切分 | [PAR-CONTEXT](../../02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.md) | [PAR-MODEL-INTERNAL](../../02_PyTorch_Algorithms/PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.md) 的 CP 分支 |
| 多副本请求路由 | [PAR-REPLICA](../../02_PyTorch_Algorithms/PAR-REPLICA_Serving_Replica_Routing.md) | 81 的 Serving 分支 |
| 组合策略选择 | [Part 02 · 49](../../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md) | 固定 workload 的组合复测 |

## 相关阅读

- [PyTorch Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [PyTorch Context Parallel tutorial](https://pytorch.org/tutorials/unstable/context_parallel.html)
