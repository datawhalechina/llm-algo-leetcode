# 27. Communication Scheduling Optimization | 通信调度优化

**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `通信优化`, `Overlap` | **目标人群：** 通信机制入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

通信总时长不一定全部暴露在训练步关键路径上。消息大小、启动次数、依赖关系和可重叠计算共同决定通信是否真正拉长单步时间。

本节建立“通信负载 → 启动与带宽成本 → 调度次序 → 暴露时间”的关系。完成后，你应能比较分桶、集合通信选择和计算重叠对关键路径的影响。

**关键词：** 计算—通信重叠、`All-Reduce`、`All-to-All`

![本节概念关系](../public/01_Hardware_Math_and_Systems/27_communication_schedule_map.svg)


## 前置阅读

**导语：** 先把通信拓扑、并行策略和 NCCL 同步原语对应到一次训练 step，再分析通信放在什么位置、以什么粒度发送，以及哪些计算可以与它重叠。

- [05. Communication Topologies | 通信拓扑与分布式基石](./05_Communication_Topologies.md)
- [20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](./20_NCCL_and_AllReduce_Basics.md)
- [26. Parallel Strategy Decision Framework | 并行策略决策框架](./26_Parallel_Strategy_Decision_Framework.md)

## Q1：通信成本如何由数据量、带宽、延迟和同步共同决定？

一次通信的理想时间可拆成固定启动成本和载荷传输成本。对同样的总字节数，拆成更多消息不会降低理想传输量，却会重复支付启动延迟。

| 数量 | 单位 | 对通信时间的影响 |
| --- | --- | --- |
| 有效载荷 | MiB | 在带宽固定时线性增长 |
| 可持续带宽 | GiB/s | 决定载荷传输下限 |
| 单次启动延迟 | μs | 小消息或高频通信更敏感 |
| 启动次数 | 次 | 决定固定成本被支付多少次 |

### Q1小验证

拆开数据量、带宽、延迟和同步次数的影响。

```python
def communication_time_ms(payload_mib, bandwidth_gib_s, latency_us, launches=1):
    """用 alpha-beta 模型估算一个通信阶段的理想时间。"""
    if payload_mib < 0 or bandwidth_gib_s <= 0 or latency_us < 0 or launches <= 0:
        raise ValueError('有效载荷、带宽、延迟或启动次数非法')
    transfer_ms = payload_mib / 1024 / bandwidth_gib_s * 1000
    launch_ms = launches * latency_us / 1000
    return {
        'payload_mib': payload_mib,
        'launches': launches,
        'transfer_ms': round(transfer_ms, 4),
        'launch_ms': round(launch_ms, 4),
        'communication_ms': round(transfer_ms + launch_ms, 4),
    }

for launches in (1, 8, 64):
    print(communication_time_ms(256, bandwidth_gib_s=50, latency_us=8, launches=launches))

assert communication_time_ms(0, 50, 8, 4)['communication_ms'] == 0.032
assert communication_time_ms(256, 50, 8, 8)['transfer_ms'] == 5.0

```

## Q2：为什么把小通信合并成大通信通常更稳？

在反向传播中，梯度只有在对应层完成后才能进入通信。较小的分桶能更早启动 All-Reduce，但会增加启动次数；较大的分桶减少启动成本，却可能等待更多梯度就绪。

| 分桶变化 | 可能收益 | 可能代价 | 应观察的量 |
| --- | --- | --- | --- |
| 减小分桶 | 通信更早开始 | 启动次数增加 | 首个分桶启动时间、启动数 |
| 增大分桶 | 固定启动成本降低 | 等待就绪的时间增加 | 通信完成时间、暴露通信 |
| 就绪顺序改变 | 与反向计算形成更长重叠窗口 | 可能破坏原有依赖 | 分桶就绪时间与关键路径 |

### Q2小验证

比较合并消息后的启动开销与 overlap 空间。

```python
def schedule_gradient_buckets(layers, bucket_cap_mib, bandwidth_gib_s=50, latency_us=8):
    """根据梯度就绪时间和分桶容量构建通信时间线。"""
    if bucket_cap_mib <= 0 or not layers:
        raise ValueError('分桶容量必须为正，且层列表不能为空')
    buckets, pending, pending_mib = [], [], 0.0
    for layer in sorted(layers, key=lambda item: item['ready_ms']):
        pending.append(layer)
        pending_mib += layer['size_mib']
        if pending_mib >= bucket_cap_mib:
            buckets.append((pending, pending_mib))
            pending, pending_mib = [], 0.0
    if pending:
        buckets.append((pending, pending_mib))

    timeline, communication_available_ms = [], 0.0
    for bucket_id, (members, size_mib) in enumerate(buckets):
        ready_ms = max(member['ready_ms'] for member in members)
        start_ms = max(ready_ms, communication_available_ms)
        duration_ms = communication_time_ms(size_mib, bandwidth_gib_s, latency_us)['communication_ms']
        end_ms = start_ms + duration_ms
        timeline.append({
            'bucket_id': bucket_id,
            'layers': [member['name'] for member in members],
            'size_mib': round(size_mib, 2),
            'ready_ms': round(ready_ms, 4),
            'start_ms': round(start_ms, 4),
            'end_ms': round(end_ms, 4),
        })
        communication_available_ms = end_ms

    backward_end_ms = max(layer['ready_ms'] for layer in layers)
    return {
        'bucket_count': len(timeline),
        'backward_end_ms': backward_end_ms,
        'communication_end_ms': round(communication_available_ms, 4),
        'exposed_after_backward_ms': round(max(communication_available_ms - backward_end_ms, 0.0), 4),
        'timeline': timeline,
    }

layers = [
    {'name': 'L4', 'ready_ms': 2.0, 'size_mib': 32},
    {'name': 'L3', 'ready_ms': 5.0, 'size_mib': 32},
    {'name': 'L2', 'ready_ms': 9.0, 'size_mib': 32},
    {'name': 'L1', 'ready_ms': 14.0, 'size_mib': 32},
]
for bucket_cap_mib in (32, 64, 128):
    print(bucket_cap_mib, schedule_gradient_buckets(layers, bucket_cap_mib))

assert schedule_gradient_buckets(layers, 32)['bucket_count'] == 4
assert schedule_gradient_buckets(layers, 128)['bucket_count'] == 1

```

## Q3：为什么 All-Reduce / All-to-All 的优化目标不同？

All-Reduce 对每个 rank 的同尺寸张量做规约并广播，环形实现中每个 rank 的逻辑传输量约为 `2(P-1)/P` 倍载荷。All-to-All 则把不同数量的状态发往不同 rank；总载荷相近时，最忙 rank 的接收量仍可能决定完成时间。

| 集合通信 | 主要负载 | 主要不均衡来源 | 调度时优先观察 |
| --- | --- | --- | --- |
| All-Reduce | 梯度或分片张量 | rank 变慢、分桶就绪时间不一致 | 分桶就绪、启动次数、暴露通信 |
| All-to-All | token、专家输入或序列分片 | 路由或序列分布倾斜 | 最大 rank 载荷、长尾与空闲 rank |

### Q3小验证

区分 All-Reduce 的同步压力和 All-to-All 的路由压力。

```python
def ring_allreduce_payload_mib(tensor_mib, world_size):
    """返回环形 All-Reduce 中每个 rank 的逻辑传输量。"""
    if tensor_mib < 0 or world_size < 2:
        raise ValueError('张量容量必须非负，world_size 至少为 2')
    return tensor_mib * 2 * (world_size - 1) / world_size

def alltoall_route_ledger(token_matrix, bytes_per_token):
    """计算 All-to-All 路由矩阵的跨 rank 载荷和接收不均衡。"""
    world_size = len(token_matrix)
    if world_size < 2 or any(len(row) != world_size for row in token_matrix):
        raise ValueError('token_matrix 必须是至少包含两个 rank 的方阵')
    if bytes_per_token <= 0 or any(token < 0 for row in token_matrix for token in row):
        raise ValueError('token 数或每 token 字节数非法')
    received_tokens = [sum(token_matrix[src][dst] for src in range(world_size)) for dst in range(world_size)]
    remote_tokens = sum(
        token_matrix[src][dst]
        for src in range(world_size) for dst in range(world_size) if src != dst
    )
    mean_received = sum(received_tokens) / world_size
    return {
        'remote_payload_mib': round(remote_tokens * bytes_per_token / 2**20, 4),
        'received_tokens': received_tokens,
        'max_to_mean_receive_ratio': round(max(received_tokens) / mean_received, 4),
    }

balanced = [[16, 16, 16, 16] for _ in range(4)]
skewed = [[4, 4, 52, 4] for _ in range(4)]
print('Ring All-Reduce per-rank MiB:', ring_allreduce_payload_mib(256, 4))
print('Balanced All-to-All:', alltoall_route_ledger(balanced, bytes_per_token=4096))
print('Skewed All-to-All:', alltoall_route_ledger(skewed, bytes_per_token=4096))

assert ring_allreduce_payload_mib(256, 4) == 384
assert alltoall_route_ledger(skewed, 4096)['max_to_mean_receive_ratio'] > 3

```

## Q4：怎样把通信排进计算间隙？

通信隐藏只能来自与它没有未解决数据依赖的计算区间。多个计算区间可能相互重叠，因此要先求它们在通信区间内的并集，再计算已隐藏和暴露时间，不能直接累加重叠窗口。

| 时间量 | 含义 | 与关键路径的关系 |
| --- | --- | --- |
| 通信总时间 | 集合通信在设备上的活动区间 | 不一定全部延长 step |
| 已隐藏通信 | 被独立计算区间覆盖的通信 | 不再单独增加关键路径 |
| 暴露通信 | 通信区间中未被覆盖的部分 | 是调度优化要减少的直接对象 |
| 重叠比例 | 已隐藏时间 / 通信总时间 | 要与 step 时间一起解读 |

### Q4小验证

观察通信进入计算间隙后，关键路径成本如何变化。

```python
def communication_exposure(comm_interval, independent_compute_intervals):
    """裁剪并合并计算区间，返回已隐藏与暴露通信时间。"""
    comm_start, comm_end = comm_interval
    if comm_end < comm_start:
        raise ValueError('通信区间非法')
    clipped = []
    for start, end in independent_compute_intervals:
        if end < start:
            raise ValueError('计算区间非法')
        left, right = max(start, comm_start), min(end, comm_end)
        if left < right:
            clipped.append((left, right))

    merged = []
    for start, end in sorted(clipped):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    communication_ms = comm_end - comm_start
    hidden_ms = sum(end - start for start, end in merged)
    exposed_ms = communication_ms - hidden_ms
    return {
        'communication_ms': round(communication_ms, 4),
        'hidden_ms': round(hidden_ms, 4),
        'exposed_ms': round(exposed_ms, 4),
        'overlap_ratio': round(hidden_ms / communication_ms, 4) if communication_ms else 1.0,
        'merged_overlap_intervals': [tuple(interval) for interval in merged],
    }

cases = [
    ((0, 12), []),
    ((0, 12), [(1, 4), (3, 8)]),
    ((0, 12), [(-2, 5), (5, 20)]),
]
for case in cases:
    print(case, '->', communication_exposure(*case))

assert communication_exposure((0, 12), [(1, 4), (3, 8)])['hidden_ms'] == 7
assert communication_exposure((0, 12), [(-2, 20)])['exposed_ms'] == 0

```

## 相关阅读

**导语：** 把理论调度继续接到 CUDA 依赖和 Part 02 的真实通信时间线。

- [17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](./17_CUDA_Stream_and_Asynchrony.md)
- [29. CUDA Stream Advanced Scheduling | CUDA Stream 高级调度](./29_CUDA_Stream_Advanced_Scheduling.md)
- [Part 02 · 48 通信热点与缓解](../02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation.md)
---
