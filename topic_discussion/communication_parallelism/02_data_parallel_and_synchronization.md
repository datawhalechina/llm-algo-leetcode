# 02. Data Parallel and Synchronization | 数据并行与同步

## 本页目标与路线位置

数据并行让每个 rank 处理不同样本，却要求参数更新保持一致。计算可以复制，更新语义不能分叉，因此梯度同步构成多卡训练最基础的通信基线，也提供了理解 All-Reduce、payload 与同步等待的入口。

![数据并行中的局部计算与梯度同步](../../docs/public/topic_discussion/communication_parallelism/data_parallel_sync.svg)

## 核心机制

### 从局部梯度到一致更新

每个 rank 持有完整模型并独立完成前向与反向；在 optimizer step 之前，局部梯度必须经过归约。同步频率由 optimizer step 决定，而不是简单等于 micro-batch 数量。

| 阶段 | 每个 rank 的状态 | 是否需要通信 | 关键约束 |
|:---|:---|:---|:---|
| 取样与前向 | 不同 mini-batch、相同参数版本 | 通常不需要 | 数据分片不能重复或遗漏 |
| 反向 | 产生局部梯度 | 可与 backward bucket 重叠 | bucket 大小影响启动次数和重叠机会 |
| 梯度累积 | 多个 micro-batch 累积局部梯度 | 非同步轮可延后归约 | 必须保持有效 batch 与 loss scaling 一致 |
| optimizer step | 所有 rank 使用一致梯度更新 | 需要完成同步 | 任一慢 rank 都会延迟全局更新 |

### Collective、Payload 与拓扑

All-Reduce 是数据并行的常见同步原语，但真实性能由消息大小、调用频率、world size 和拓扑共同决定。小 payload 更容易受启动延迟影响，大 payload 更依赖链路带宽。

不同并行机制需要不同的数据交换语义。先回答“结果要回到哪些 rank”，再选择 Collective，不能只按 API 名称记忆。

| Collective | 数据语义 | 常见落点 | 首要代价 |
|:---|:---|:---|:---|
| All-Reduce | 聚合后把完整结果返回所有 rank | DDP 梯度同步、部分 TP 输出归约 | 每个 rank 都参与并接收完整结果 |
| Reduce-Scatter | 聚合后把结果分片留在不同 rank | ZeRO/FSDP 梯度分片 | 结果布局必须与后续状态分片一致 |
| All-Gather | 收集各 rank 分片并恢复完整输入 | ZeRO-3/FSDP 参数恢复、部分 TP/CP 路径 | 临时完整状态可能抬高峰值 |
| All-to-All | 按目标 rank 重新分发不同数据 | EP token dispatch/combine | 消息不均与最慢 rank 决定完成时间 |
| P2P Send/Recv | 在指定 rank 间传递数据 | Pipeline stage、KV handoff | 依赖关系和流水线空转 |

| 观察项 | 说明 | 对应实验 |
|:---|:---|:---|
| payload bytes | 每次同步传输的数据规模 | Part 02 · 46 的 payload matrix |
| collective frequency | 每个 step 触发多少次同步 | backward bucket 与 accumulation 配置 |
| max-rank time | 最慢 rank 决定同步完成时间 | 每 rank 时间与负载差异 |
| overlap ratio | 通信有多少被 backward 计算覆盖 | trace 中通信与计算时间线 |
| scaling efficiency | 增加 rank 后实际吞吐相对理想值的比例 | 固定 global workload 的多卡对照 |

## 选择条件与验证证据

| 目的 | 入口 |
|:---|:---|
| 理解通信原语 | [Part 01 · 20 NCCL 与 AllReduce](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb) |
| 实测 collective | [Part 02 · 46 通信 Profiling](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb) |
| 从复制转向状态分片 | [03 状态分片与 ZeRO](./03_state_sharding_and_zero.md) |
| 进入真实训练项目 | [Part 02 · 79 分布式并行基准](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb) |

## 相关阅读

- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch Distributed collectives](https://pytorch.org/docs/stable/distributed.html)
