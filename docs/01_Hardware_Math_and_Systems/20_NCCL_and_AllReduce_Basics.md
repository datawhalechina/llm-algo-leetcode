# 20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础

**难度：** Medium | **环境：** CPU-first | **标签：** `并行通信`, `NCCL`, `AllReduce` | **目标人群：** 通信机制入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

多卡训练扩展不理想时，问题可能来自集合通信和同步等待，而不只是单卡算子速度。增加 GPU 数量也会增加需要协调的数据和参与者。

本节建立“局部结果 → All-Reduce → 环形传播 → 延迟与带宽成本”的关系。完成后，你应能解释 Reduce-Scatter 与 All-Gather 如何组成环形 All-Reduce，并估算每个进程的通信量和理论时间。


**关键词：** `NCCL`, `AllReduce`, `DP`

![本节概念关系](../public/01_Hardware_Math_and_Systems/20_nccl_collective_map.svg)

---
## 前置阅读

**导语：** 先把通信和并行层级对齐，再看 AllReduce 与 NCCL 如何影响同步、effective batch 和扩展效率。

- [Group 1C: Communication, Parallelism and Fault Tolerance | 1C: 通信、并行与容错](./1C.md)
- [05. Communication Topologies | 通信拓扑与分布式基石](./05_Communication_Topologies.md)
---
## Q1：NCCL 在分布式训练里到底解决什么问题？


NCCL 为 NVIDIA GPU 提供集合通信和点到点通信实现。它执行 All-Reduce、All-Gather、Reduce-Scatter 等原语，但不负责选择 DP、TP 或 PP，也不替代上层调度器。

在数据并行训练中，各 rank 先得到局部梯度，再调用集合通信形成一致结果。若 collective 没有被计算覆盖，未覆盖部分会直接进入训练步关键路径。

### Q1小验证：同步等待如何进入训练步

计算时间与 collective 时间不能按 GPU 数量线性相乘；先扣除真正被计算覆盖的通信时间，再得到训练步关键路径。

```python
def training_step_time(compute_ms, collective_ms, overlap_ms=0.0):
    """返回计算与 collective 共同形成的关键路径时间。"""
    if min(compute_ms, collective_ms, overlap_ms) < 0:
        raise ValueError('时间必须非负')
    hidden_ms = min(overlap_ms, collective_ms, compute_ms)
    exposed_ms = collective_ms - hidden_ms
    return {
        'step_ms': compute_ms + exposed_ms,
        'hidden_collective_ms': hidden_ms,
        'exposed_collective_ms': exposed_ms,
    }

reports = [training_step_time(10.0, 3.0, overlap) for overlap in [0.0, 1.5, 3.0]]
assert reports[0]['exposed_collective_ms'] == 3.0
assert reports[-1]['exposed_collective_ms'] == 0.0
for report in reports:
    print(report)

```

## Q2：AllReduce 为什么是最核心的通信原语之一？


All-Reduce 接收每个 rank 的同形状局部张量，执行归约，并把相同的完整结果返回给所有 rank。其不变量是：输出形状与局部输入相同，各 rank 输出一致。

数据并行常用它聚合局部梯度。求和与平均只差一个缩放步骤，但该缩放必须与损失归约和优化器语义一致。

### Q2小验证：AllReduce 的输出不变量是什么

先把“局部结果 -> 全局一致”这件事想清楚。

```python
def allreduce_sum(local_vectors):
    """逐元素归约，并把相同的完整结果返回给每个 rank。"""
    if not local_vectors or len({len(row) for row in local_vectors}) != 1:
        raise ValueError('所有 rank 必须提供等长向量')
    reduced = [sum(items) for items in zip(*local_vectors)]
    return [reduced.copy() for _ in local_vectors]

local_gradients = [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0], [1000.0, 2000.0]]
global_gradients = allreduce_sum(local_gradients)
print('rank 0 local:', local_gradients[0])
print('all ranks  :', global_gradients)
assert all(row == global_gradients[0] for row in global_gradients)

```

## Q3：为什么通信带宽和拓扑会直接影响训练扩展效果？


通信不是只有“能不能通”的问题，更重要的是“通得快不快”。

Ring All-Reduce 包含 `2 × (world_size - 1)` 轮通信。固定 payload 下，每个 rank 的理论通信量趋近 payload 的两倍；启动时延随轮数增长，传输项则受有效链路带宽约束。

### Q3小验证：Ring 的理论通信量与 alpha-beta 成本

对固定全量 payload，计算每个 rank 在两阶段中搬运的字节数，并把每轮启动延迟与传输成本分开。

```python
def ring_cost(payload_bytes, world_size, alpha_ms, bandwidth_gbps):
    """估算 Ring AllReduce 的轮数、每 rank 通信量和 alpha-beta 时间。"""
    if payload_bytes < 0 or world_size < 2 or alpha_ms < 0 or bandwidth_gbps <= 0:
        raise ValueError('检查 payload、world_size、alpha 和 bandwidth')
    rounds = 2 * (world_size - 1)
    bytes_per_rank = 2 * (world_size - 1) / world_size * payload_bytes
    bandwidth_bytes_per_ms = bandwidth_gbps * 1e9 / 8 / 1000
    latency_ms = rounds * alpha_ms
    transfer_ms = bytes_per_rank / bandwidth_bytes_per_ms
    return {
        'rounds': rounds,
        'bytes_per_rank': int(bytes_per_rank),
        'latency_ms': latency_ms,
        'transfer_ms': transfer_ms,
        'total_ms': latency_ms + transfer_ms,
    }

payload_bytes = 256 * 1024 * 1024
nvlink_like = ring_cost(payload_bytes, world_size=8, alpha_ms=0.01, bandwidth_gbps=900.0)
pcie_like = ring_cost(payload_bytes, world_size=8, alpha_ms=0.01, bandwidth_gbps=64.0)
assert nvlink_like['rounds'] == 14
assert nvlink_like['bytes_per_rank'] == 2 * 7 * payload_bytes // 8
assert pcie_like['transfer_ms'] > nvlink_like['transfer_ms']
print('高速链路：', nvlink_like)
print('低速链路：', pcie_like)

```

## Q4：Ring AllReduce 为什么常被用来解释 AllReduce 的实现思路？


Ring AllReduce 的关键点，不是“名字里有 ring”，而是它把一次全量聚合拆成了两个更可控的阶段：

1. **Reduce-Scatter**：先沿着环把各卡的数据逐步归约，并把结果切成若干块分散到各卡；
2. **All-Gather**：再沿着环把这些局部归约结果重新拼回完整结果。

这样做的好处是：每张卡在每一轮只发送和接收一小块数据，通信负载更均匀，不需要某一张卡一次性承担全部传输压力。

这个模拟器只验证算法状态变化。真实 NCCL 会根据拓扑、消息规模和配置选择具体算法，实际性能需要在多卡环境中测量。

### Q4小验证：逐轮追踪 chunk 所有权与归约状态

输入矩阵的第一个维度表示 rank，第二个维度表示 chunk。Reduce-Scatter 每轮只移动一个 partial chunk，All-Gather 再传播已经完成归约的 chunk。

```python
def ring_allreduce(local_chunks):
    """用同步快照模拟 Ring Reduce-Scatter 与 All-Gather。"""
    world_size = len(local_chunks)
    if world_size < 2 or any(len(row) != world_size for row in local_chunks):
        raise ValueError('需要 world_size × world_size 的 chunk 矩阵')

    partial = [list(row) for row in local_chunks]
    trace = []
    for step in range(world_size - 1):
        snapshot = [row.copy() for row in partial]
        transfers = []
        for rank in range(world_size):
            source = (rank - 1) % world_size
            chunk = (rank - step - 1) % world_size
            partial[rank][chunk] += snapshot[source][chunk]
            transfers.append({'from': source, 'to': rank, 'chunk': chunk})
        trace.append({'phase': 'reduce_scatter', 'step': step, 'transfers': transfers})

    owned = {(rank + 1) % world_size: partial[rank][(rank + 1) % world_size] for rank in range(world_size)}
    gathered = [{(rank + 1) % world_size: partial[rank][(rank + 1) % world_size]} for rank in range(world_size)]
    for step in range(world_size - 1):
        snapshot = [dict(items) for items in gathered]
        transfers = []
        for rank in range(world_size):
            source = (rank - 1) % world_size
            chunk = (rank - step) % world_size
            gathered[rank][chunk] = snapshot[source][chunk]
            transfers.append({'from': source, 'to': rank, 'chunk': chunk})
        trace.append({'phase': 'all_gather', 'step': step, 'transfers': transfers})

    outputs = [[items[index] for index in range(world_size)] for items in gathered]
    expected = [sum(local_chunks[rank][chunk] for rank in range(world_size)) for chunk in range(world_size)]
    assert all(row == expected for row in outputs)
    return {'outputs': outputs, 'owned_after_reduce_scatter': owned, 'trace': trace}

local_chunks = [[rank * 10 + chunk for chunk in range(4)] for rank in range(4)]
result = ring_allreduce(local_chunks)
print('reduce-scatter owners:', result['owned_after_reduce_scatter'])
print('rank 0 output:', result['outputs'][0])
print('rounds:', len(result['trace']))

```

---
## 相关阅读
本节只建立算法语义与理论代价。真实 NCCL 可能依据消息规模和拓扑选择 Ring、Tree 或其他实现；继续进入 Part 02 · 47 验证请求算法、实际算法和多卡时间。
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)：查阅集合通信原语、拓扑和通信环境。
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)：用固定消息大小和 GPU 数量观察集合通信带宽。
- [47. Communication Profiling with NCCL | NCCL 通信剖析](../02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL.md)

---
