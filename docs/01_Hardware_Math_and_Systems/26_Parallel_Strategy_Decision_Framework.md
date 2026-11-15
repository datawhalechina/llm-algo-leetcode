# 26. Parallel Strategy Decision Framework | 并行策略决策框架

**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `并行策略`, `策略决策` | **目标人群：** 通信机制入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

当单设备无法满足容量或吞吐目标时，可以切分训练状态、网络层、张量、序列、专家或服务请求。不同切分对象解决的问题不同，也会引入不同通信与负载风险。

先算每个设备需要保存的状态，再检查候选切分方式、通信载荷和放置规则。完成后，你应能说明候选为什么保留或排除，并把尚未测量的通信时间与扩展效率带到通信与并行路线中验证。

**关键词：** `DP`, `TP`, `PP`, `CP`, `EP`, `Replica`

![本节概念关系](../public/01_Hardware_Math_and_Systems/26_parallel_strategy_decision_map.svg)
## 前置阅读

**导语：** 先把通信拓扑和显存切分的基础直觉接上，再看这页的并行策略选择，会更容易把“能不能放下”和“怎么切”连起来。

- [05. Communication Topologies | 通信拓扑与分布式基石](./05_Communication_Topologies.md)
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)
## Q1：要切分的是训练状态、计算还是请求？

先记录每卡状态、激活与运行预留，再选择需要分摊的对象。本例使用GiB；06节的十进制GB结果应先乘以 `10^9 / 2^30` 换算。给定状态13 GiB、激活4 GiB、预留2 GiB，总需求19 GiB，在24 GiB设备上剩余5 GiB。

互连条件取决于实际带宽、时延、消息大小和频率。机内与跨节点表示位置，不是固定的速度等级。上下文并行既可分摊长序列状态，也可用于改善计算分工；是否值得，要比较通信和执行时间。

| 策略 | 中文含义与切分对象 | 引入的工作 |
| --- | --- | --- |
| DP | 数据并行，不同进程处理不同批次 | 同步梯度；与ZeRO组合时还会分片训练状态 |
| TP | 张量并行，切分层内张量和计算 | 交换局部激活或归约局部结果 |
| PP | 流水线并行，分配不同网络层 | 阶段间传递激活，等待相邻阶段 |
| CP | 上下文并行，切分序列位置 | 交换注意力所需的状态 |
| EP | 专家并行，分配不同专家 | 分发token、合并输出并处理负载不均 |
| Replica | 服务副本并行，分发独立请求 | 管理队列、缓存局部性和尾延迟 |

### Q1小验证

建立带单位的资源账本，检查单设备容量缺口。

```python
import math


def build_resource_ledger(state_gib, activation_gib, runtime_reserve_gib, device_memory_gib):
    """汇总单设备容量需求；所有输入均以 GiB 为单位。"""
    values = (state_gib, activation_gib, runtime_reserve_gib, device_memory_gib)
    if any(value < 0 for value in values) or device_memory_gib == 0:
        raise ValueError('容量必须为非负数，且设备显存必须大于 0 GiB')
    required_gib = state_gib + activation_gib + runtime_reserve_gib
    headroom_gib = device_memory_gib - required_gib
    return {
        'required_gib': round(required_gib, 2),
        'device_memory_gib': round(device_memory_gib, 2),
        'headroom_gib': round(headroom_gib, 2),
        'fits_one_device': headroom_gib >= 0,
    }

fits = build_resource_ledger(13.0, 4.0, 2.0, 24.0)
does_not_fit = build_resource_ledger(22.0, 5.0, 2.0, 24.0)
assert fits['required_gib'] == 19
assert fits['headroom_gib'] == 5
assert fits['fits_one_device']
assert not does_not_fit['fits_one_device']
print('可单卡容纳：', fits)
print('存在容量缺口：', does_not_fit)

```

## Q2：怎样在同一载荷下比较通信频率的代价？

先把一轮训练中的总发送载荷固定，再改变发送次数。256 MiB一次发送，与32 MiB发送8次，总字节数相同；在100 GiB/s、每次固定开销5 μs的条件下，传输项都是2.5 ms，固定开销分别是0.005和0.04 ms，总计2.505和2.54 ms。

这里的载荷指每rank实际发送量，已包含所选通信算法的放大系数；20节的Ring公式可用于从梯度张量大小换算发送量。时间按调用串行相加。TP、EP等高频交换是否合算，还要在27节结合就绪时间和可重叠计算判断。

### Q2小验证

用同一单位记录单次载荷、调用次数、链路带宽和固定时延。

```python
def communication_ledger(payload_mib, calls_per_step, bandwidth_gib_s, latency_us):
    """按每rank单次发送MiB、调用次数、单向GiB/s估算串行通信时间。"""
    if payload_mib < 0 or calls_per_step < 0 or bandwidth_gib_s <= 0 or latency_us < 0:
        raise ValueError('载荷、次数和时延不得为负，带宽必须大于 0 GiB/s')
    transfer_ms_per_call = payload_mib / 1024 / bandwidth_gib_s * 1000
    latency_ms_per_call = latency_us / 1000
    total_ms = calls_per_step * (transfer_ms_per_call + latency_ms_per_call)
    return {
        'payload_per_call_mib': payload_mib,
        'calls_per_step': calls_per_step,
        'total_payload_mib': payload_mib * calls_per_step,
        'transfer_ms': round(transfer_ms_per_call * calls_per_step, 3),
        'latency_ms': round(latency_ms_per_call * calls_per_step, 3),
        'lower_bound_ms': round(total_ms, 3),
    }

low_frequency = communication_ledger(256, 1, 100, 5)
high_frequency = communication_ledger(32, 8, 100, 5)
assert low_frequency['total_payload_mib'] == high_frequency['total_payload_mib'] == 256
assert low_frequency['transfer_ms'] == high_frequency['transfer_ms'] == 2.5
assert low_frequency['lower_bound_ms'] == 2.505
assert high_frequency['lower_bound_ms'] == 2.54
print('低频大载荷：', low_frequency)
print('高频小载荷：', high_frequency)

```

## Q3：哪些条件不满足时，应先排除候选？

每个候选声明当前配置下的每卡容量需求、支持的任务与模型结构，以及允许的放置域。先按这些给定条件筛选：例如DP配置需要29 GiB，超过24 GiB容量；EP配置面向MoE，不能直接用于稠密模型。

放置域规则沿用05节的允许集合。一个跨节点候选能否更快，取决于Q2的通信成本和后续实测；筛选通过只表示列出的条件满足。容量数字也需要随批量、序列长度与分片方式更新。

### Q3条件检查：让拒绝原因对应具体约束

保持其他条件不变，分别核对容量、任务模式、模型结构和允许域。


```python
def evaluate_candidate(candidate, environment):
    """检查给定容量、任务、模型和放置规则，不按位置名称预测链路速度。"""
    required = candidate['memory_per_rank_gib']
    capacity = environment['memory_per_device_gib']
    if not all(math.isfinite(v) for v in (required, capacity)) or required < 0 or capacity <= 0:
        raise ValueError('容量需求须为有限非负数，设备容量须为有限正数')
    domains = {'none', 'intra_node', 'cross_node'}
    allowed = set(candidate['allowed_link_scopes'])
    if not allowed or not allowed <= domains or environment['link_scope'] not in domains:
        raise ValueError('需给定有效放置域及允许集合')
    reasons = []
    if required > capacity:
        reasons.append('per-rank memory exceeds device memory')
    if environment['mode'] not in candidate['modes']:
        reasons.append('workload mode is unsupported')
    if environment['model_kind'] not in candidate['model_kinds']:
        reasons.append('model structure is unsupported')
    if environment['link_scope'] not in allowed:
        reasons.append('placement domain is not allowed')
    return {'name': candidate['name'], 'conditions_met': not reasons, 'reasons': reasons}


environment = dict(memory_per_device_gib=24, mode='training', model_kind='dense', link_scope='intra_node')
candidates = [
    dict(name='DP', memory_per_rank_gib=29, modes={'training'},
         model_kinds={'dense', 'moe'}, allowed_link_scopes={'intra_node', 'cross_node'}),
    dict(name='ZeRO-3', memory_per_rank_gib=18, modes={'training'},
         model_kinds={'dense', 'moe'}, allowed_link_scopes={'intra_node', 'cross_node'}),
    dict(name='TP', memory_per_rank_gib=16, modes={'training', 'serving'},
         model_kinds={'dense', 'moe'}, allowed_link_scopes={'intra_node'}),
    dict(name='EP', memory_per_rank_gib=14, modes={'training', 'serving'},
         model_kinds={'moe'}, allowed_link_scopes={'intra_node'}),
]
reports = [evaluate_candidate(candidate, environment) for candidate in candidates]
assert reports[0]['reasons'] == ['per-rank memory exceeds device memory']
assert reports[2]['conditions_met']
assert reports[3]['reasons'] == ['model structure is unsupported']
for report in reports:
    print(report)

```

## Q4：并行度什么时候可以直接相乘？

若DP、TP、PP是彼此独立的进程网格轴，`DP=2、TP=4、PP=2`需要16个rank；下面按这个假设检查设备数量。EP或CP也可能复用已有进程组，此时应先确定实际组映射，不能把所有度数直接相乘。

模型结构与执行角色还需单独核对：EP度数大于1要求模型包含专家；服务副本与同步训练DP的任务不同。代码要求调用时明确声明轴是否独立；非独立轴返回“需要组映射”，不推算实际rank数。

### Q4小验证

检查组合所需 rank 数与结构条件，输出可定位的问题。

```python
def validate_parallel_composition(degrees, available_ranks, model_kind, *, independent_axes):
    """仅对已声明独立的进程网格轴计算乘积；复用进程组时需另给映射。"""
    supported = {'DP', 'TP', 'PP', 'CP', 'EP', 'Replica'}
    if set(degrees) - supported:
        raise ValueError('存在未知并行维度')
    if type(available_ranks) is not int or available_ranks < 1:
        raise ValueError('可用rank数须为正整数')
    if any(type(degree) is not int or degree < 1 for degree in degrees.values()):
        raise ValueError('各维度度数须为正整数')
    if type(independent_axes) is not bool:
        raise ValueError('需明确声明各轴是否独立')
    if not independent_axes:
        return {'required_ranks': None, 'valid': False, 'issues': ['需要实际进程组映射']}
    required_ranks = math.prod(degrees.values())
    issues = []
    if required_ranks > available_ranks:
        issues.append(f'requires {required_ranks} ranks but only {available_ranks} are available')
    if degrees.get('EP', 1) > 1 and model_kind != 'moe':
        issues.append('EP degree greater than 1 requires an MoE model')
    if degrees.get('Replica', 1) > 1 and degrees.get('DP', 1) > 1:
        issues.append('Replica and training DP must use distinct execution roles')
    return {'required_ranks': required_ranks, 'valid': not issues, 'issues': issues}


valid_plan = validate_parallel_composition(
    {'TP': 4, 'PP': 2, 'DP': 2}, 16, 'dense', independent_axes=True,
)
rank_shortage = validate_parallel_composition(
    {'TP': 4, 'PP': 2, 'DP': 2}, 8, 'dense', independent_axes=True,
)
wrong_structure = validate_parallel_composition({'EP': 2}, 2, 'dense', independent_axes=True)
shared_groups = validate_parallel_composition(
    {'DP': 4, 'EP': 2}, 4, 'moe', independent_axes=False,
)
assert valid_plan['valid'] and valid_plan['required_ranks'] == 16
assert len(rank_shortage['issues']) == 1
assert wrong_structure['issues'] == ['EP degree greater than 1 requires an MoE model']
assert shared_groups['required_ranks'] is None
for plan in (valid_plan, rank_shortage, wrong_structure, shared_groups):
    print(plan)

```

## 相关阅读

**导语：** 继续进入策略选择页、上下文切分页和 PyTorch 分布式文档，核对候选所需的真实证据。

- [Part 02 · 49 并行策略选择](../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md)
- [Part 02 · PAR-CONTEXT 上下文与序列并行](../02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.md)
- [PyTorch Distributed | PyTorch 分布式训练文档](https://pytorch.org/docs/stable/distributed.html)
---
