# 05. Communication Topologies | 通信拓扑与分布式基石

**难度：** Medium | **环境：** CPU-first | **标签：** `并行通信`, `分布式训练`, `通信拓扑` | **目标人群：** 需要理解多卡训练通信代价的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

大模型从单卡扩展到多卡后，每种切分都会引入数据交换。数据并行（DP）、张量并行（TP）和流水线并行（PP）切分的对象不同，因此通信负载和同步位置也不同。

本节建立“切分对象 → 集合通信接口 → 拓扑路径 → 并行组放置”的分类关系。完成后，你应能指出一种切分可能需要的通信原语，估算链路传输下界，并检查高频通信是否被放在合适的拓扑域内；All-Reduce 的算法过程在第 20 节展开，并行候选筛选由第 26 节完成。

**关键词：** `DP`, `TP`, `PP`

![本节概念关系](../public/01_Hardware_Math_and_Systems/05_communication_topology_map.svg)

---

## 前置阅读
**导语：** 先复习 GPU 内存层级和设备间连接，再用 DP、TP、PP 的示例回答三个问题：数据放在哪里、哪些数据需要交换、链路带宽如何改变并行收益。
- [03. GPU Architecture and Memory | GPU 物理架构与内存层级](./03_GPU_Architecture_and_Memory.md)

---
## Q1：DP、TP、PP 分别如何切分数据、张量和层？


先分别看清数据并行（DP）、张量并行（TP）和流水线并行（PP）各自切分什么，再说明它们组合后为什么称为 3D 并行，最后追踪切分产生的数据交换。

- **DP**：不同卡处理不同数据批次，再同步梯度。
- **TP**：把单层中的大张量切到多卡上共同计算。
- **PP**：把不同层切到不同设备或设备组上形成流水线。

将 DP、TP、PP 组合使用时，通常称为 3D 并行。它们的目标不是“越多越好”，而是让模型、算力和通信拓扑能一起匹配。先把切分对象、主要交换的数据和常见通信方式分开看：

| 并行方式 | 切分对象 | 主要交换的数据 | 常见通信方式 |
| --- | --- | --- | --- |
| DP | 不同数据批次 | 梯度或参数更新结果 | 常见 All-Reduce |
| TP | 层内张量或激活 | 局部激活、局部结果 | 可能使用 All-Gather、Reduce-Scatter 或 All-Reduce |
| PP | 不同层或阶段 | 相邻阶段的激活 | 点到点传输，并伴随流水线等待 |

### Q1小验证：对照并行维度的切分对象与通信关注点

逐项检查每种并行方式切分什么、需要交换什么，以及对应的通信方式。

```python
def three_d_parallel(dp, tp, pp):
    """组合 DP、TP、PP 三个并行维度，返回切分关系和通信关注点。"""
    if any(type(value) is not int or value <= 0 for value in (dp, tp, pp)):
        raise ValueError('dp、tp、pp须为正整数')
    # 先记录三个并行维度，再观察它们组合后的 worker 数量和通信关系。
    return {
        'dp_groups': dp,
        'tp_shards': tp,
        'pp_stages': pp,
        'effective_workers': dp * tp * pp,
        'partition_axis': {'dp': 'data', 'tp': 'tensor', 'pp': 'layers'},
        'communication_focus': {
            'dp': 'gradient synchronization',
            'tp': 'activation gather / reduce-scatter',
            'pp': 'stage activation transfer',
        },
    }

cases = [
    three_d_parallel(8, 1, 1),
    three_d_parallel(4, 2, 2),
    three_d_parallel(2, 4, 4),
]
for case in cases:
    print(case)
assert cases[0]['partition_axis'] == {'dp': 'data', 'tp': 'tensor', 'pp': 'layers'}
assert 'gradient' in cases[1]['communication_focus']['dp']
assert 'activation' in cases[2]['communication_focus']['pp']
print('3D parallelism = DP × TP × PP')

```

## Q2：集合通信怎样改变每个进程的输入与输出？

集合通信要求一组进程共同参与，进程编号称为rank，参与进程数称为world size。下面固定4个rank：All-Reduce归约后返回完整结果，All-Gather收集分片，Reduce-Scatter先归约再把结果分片。代码中的形状始终指每个rank实际传给接口的输入，不在不同原语间改变含义。

| 原语 | 每rank输入形状 | 每rank输出形状 | 数据变化 |
| --- | --- | --- | --- |
| All-Reduce | (8, 1024) | (8, 1024) | 逐元素归约，每rank获得相同完整结果 |
| All-Gather | (8, 1024) | (32, 1024) | 按第0维拼接各rank分片 |
| Reduce-Scatter | (32, 1024) | (8, 1024) | 归约完整输入，再按第0维等分 |

### Q2对照验证：统一按每rank输入推导输出

Reduce-Scatter要求第0维能均分给4个rank；归约后的具体数值继续在20节展开。


```python
def collective_shape(kind, world_size, input_shape):
    """按每rank输入形状推导输出；收集和切分均沿第0维进行。"""
    if type(world_size) is not int or world_size <= 0:
        raise ValueError('进程数须为正整数')
    if not input_shape or any(type(dim) is not int or dim <= 0 for dim in input_shape):
        raise ValueError('输入形状各维度须为正整数')
    shape = tuple(input_shape)
    if kind == 'allreduce':
        output_shape = shape
    elif kind == 'allgather':
        output_shape = (shape[0] * world_size, *shape[1:])
    elif kind == 'reducescatter':
        if shape[0] % world_size != 0:
            raise ValueError('Reduce-Scatter的第0维须能被进程数整除')
        output_shape = (shape[0] // world_size, *shape[1:])
    else:
        raise ValueError(f'未知通信原语：{kind}')
    return {'input_per_rank': shape, 'output_per_rank': output_shape}


WORLD_SIZE = 4
inputs = {'allreduce': (8, 1024), 'allgather': (8, 1024), 'reducescatter': (32, 1024)}
for kind, shape in inputs.items():
    print(kind, '→', collective_shape(kind, WORLD_SIZE, shape))
assert collective_shape('allreduce', 4, (8, 1024))['output_per_rank'] == (8, 1024)
assert collective_shape('allgather', 4, (8, 1024))['output_per_rank'] == (32, 1024)
assert collective_shape('reducescatter', 4, (32, 1024))['output_per_rank'] == (8, 1024)

```

## Q3：同一份数据经过不同链路，传输下界怎样变化？

这里统一使用MiB表示载荷、GiB/s表示单向链路带宽：256 MiB等于0.25 GiB，在100 GiB/s链路上传输下界为2.5 ms。若规格按Gb/s给出，先换算；1 GiB/s约等于8.59 Gb/s，二者不能直接代入同一公式。

下面的带宽是给定的教学条件。跨节点路径假设数据可流水传输，单份长消息的理想持续吞吐由最窄链路限制；各跳启动和转发延迟尚未计入。实际共享路径还可能受到拥塞影响。

| 路径中的链路 | 给定单向带宽（GiB/s） | 对本例的影响 |
| --- | ---: | --- |
| 机内GPU互连 | 100 | 256 MiB的下界为2.5 ms |
| GPU到网卡 | 50 | 跨节点路径的第一跳 |
| 节点间网络 | 12.5 | 比第一跳窄，形成20 ms传输下界 |

### Q3小验证：相同消息经过不同链路需要多久

在相同消息大小下，比较不同链路的理想传输时间。

```python
def path_transfer_lower_bound(payload_mib, links):
    """按流水路径的最窄单向带宽估算传输下界；载荷MiB，带宽GiB/s。"""
    if payload_mib < 0 or not links:
        raise ValueError('payload_mib 不得为负，links 不得为空')
    if any(link['bandwidth_gib_s'] <= 0 for link in links):
        raise ValueError('链路带宽必须大于 0 GiB/s')
    bottleneck = min(links, key=lambda link: link['bandwidth_gib_s'])
    lower_bound_ms = payload_mib / 1024 / bottleneck['bandwidth_gib_s'] * 1000
    return {
        'bottleneck_link': bottleneck['name'],
        'bottleneck_gib_s': bottleneck['bandwidth_gib_s'],
        'lower_bound_ms': round(lower_bound_ms, 3),
    }

intra_node = path_transfer_lower_bound(256, [{'name': 'intra-node GPU link', 'bandwidth_gib_s': 100}])
cross_node = path_transfer_lower_bound(256, [
    {'name': 'GPU-NIC', 'bandwidth_gib_s': 50},
    {'name': 'network', 'bandwidth_gib_s': 12.5},
])
assert intra_node['lower_bound_ms'] == 2.5
assert cross_node['lower_bound_ms'] == 20.0
assert cross_node['bottleneck_link'] == 'network'
print('机内路径：', intra_node)
print('跨节点路径：', cross_node)
```

## Q4：怎样检查并行组是否满足给定放置规则？

先查看当前任务允许哪些放置域，再检查并行组实际落在哪里。例如某个TP组被要求留在机内，而DP组允许机内或跨节点；检查的是这份任务规则，不由位置名称推断实际带宽。

高频通信更需要关注链路带宽、时延与共享流量。若考虑放宽机内限制，应带着消息大小和调用频率进入26节，比较目标路径的成本与容量条件。

| 当前规则或现象 | 应检查什么 |
| --- | --- |
| TP组只允许机内 | 是否有rank被放到其他节点 |
| EP组需要跨节点交换token | 任务规则是否允许，以及All-to-All负载是否可承受 |
| DP组允许两种放置域 | 拓扑约束通过后，继续比较同步成本 |
| 跨节点路径反而更快 | 按实际链路数据判断，不套用位置强弱排序 |

### Q4条件检查：对照允许域与实际放置域

保持规则不变，只改变实际放置；输出是否符合放置规则，不把通过结果当成性能结论。


```python
def check_group_placement(groups):
    """检查给定允许域；位置类别不代表带宽或时延等级。"""
    domains = {'intra_node', 'cross_node'}
    reports = []
    for group in groups:
        allowed = set(group['allowed_domains'])
        if not allowed or not allowed <= domains or group['actual_domain'] not in domains:
            raise ValueError('使用明确的机内或跨节点放置域')
        reports.append({
            'name': group['name'],
            'allowed_domains': sorted(allowed),
            'actual_domain': group['actual_domain'],
            'placement_allowed': group['actual_domain'] in allowed,
        })
    return reports


groups = [
    dict(name='TP group', allowed_domains=['intra_node'], actual_domain='intra_node'),
    dict(name='EP group', allowed_domains=['intra_node'], actual_domain='cross_node'),
    dict(name='DP group', allowed_domains=['intra_node', 'cross_node'], actual_domain='cross_node'),
]
reports = check_group_placement(groups)
assert [row['placement_allowed'] for row in reports] == [True, False, True]
for report in reports:
    print(report)

```

---
## 相关阅读
本节可以继续接到 NCCL 原语、并行策略选择和通信调度。
- [20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](./20_NCCL_and_AllReduce_Basics.md)
- [26. Parallel Strategy Decision Framework | 并行策略决策框架](./26_Parallel_Strategy_Decision_Framework.md)
- [27. Communication Scheduling Optimization | 通信调度优化](./27_Communication_Scheduling_Optimization.md)
---
