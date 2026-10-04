# 05. Expert Parallel and Dynamic Routing | 专家并行与动态路由

## 本页目标与路线位置

Expert Parallel（EP）把专家分到不同 rank，并把 token 动态发送给被选中的专家。它与静态 TP 的关键区别是通信量由本批次路由结果决定：热点专家会同时制造容量溢出、All-to-All 不均衡和最慢 rank 等待。

![专家路由如何形成 All-to-All 与负载热点](../../public/topic_discussion/communication_parallelism/expert_parallel_hotspots.svg)

## 核心机制

### Router 到输出的完整路径

| 阶段 | 状态变化 | 需要记录的机制证据 |
|:---|:---|:---|
| route | token 选择 top-k experts | expert counts、routing entropy |
| capacity | 每个 expert 接受有限 token | capacity factor、overflow / dropped tokens |
| dispatch | token 按目标 expert 跨 rank 重排 | send / receive counts、All-to-All bytes |
| expert compute | 各 rank 执行本地专家 | 每 rank token 数、compute time、peak memory |
| combine | 专家输出返回原 token 顺序并加权 | reverse All-to-All、combine time、正确性 |

### 负载不均决定最慢 rank

平均 token 数不能解释 EP 性能。真正限制 step time 的是热点 rank；即使总 payload 不变，不均衡也可能让部分 rank 计算和通信拥塞、其他 rank 空等。

| 现象 | 优先检查 | 可调整方向 |
|:---|:---|:---|
| overflow 升高 | capacity、top-k、router 分布 | capacity factor、路由正则、专家数 |
| All-to-All 时间抖动 | send / receive counts、拓扑 | expert placement、分层通信、路由约束 |
| expert compute 不均 | max / mean load、最慢 rank | 负载均衡、shared expert、token 重分配 |
| 吞吐增加但质量下降 | dropped tokens、router 稳定性 | 质量门槛和 overflow 策略 |

## 选择条件与验证证据

| 目的 | 入口 |
|:---|:---|
| 理解 Router 与均衡目标 | [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.md)、[07 Load Balancing](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md) |
| 验证 dispatch / combine | [Part 02 · 47 Expert Parallel](../../02_PyTorch_Algorithms/47_MoE_Expert_Parallel.md) |
| 比较均匀与偏斜负载 | [Part 02 · 80 MoE 并行基准](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md) |
| 定位通信热点 | [06 通信 Profiling 与 Overlap](./06_communication_profiling_and_overlap.md) |

## 相关阅读

- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)
