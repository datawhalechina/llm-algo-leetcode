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

本节建立“容量约束 → 切分对象 → 互连条件 → 风险门槛”的筛选顺序。完成后，你应能先排除不满足硬约束的候选，再为保留策略确定需要验证的通信和扩展证据。

**关键词：** `DP`, `TP`, `PP`, `CP`, `EP`, `Replica`

![本节概念关系](../public/01_Hardware_Math_and_Systems/26_parallel_strategy_decision_map.svg)
## 前置阅读

**导语：** 先把通信拓扑和显存切分的基础直觉接上，再看这页的并行策略选择，会更容易把“能不能放下”和“怎么切”连起来。

- [05. Communication Topologies | 通信拓扑与分布式基石](./05_Communication_Topologies.md)
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)
## Q1：当前首先受显存、通信还是模型结构约束？


并行策略不要先背名字，先登记约束：

1. **容量约束**：每个设备需要保留多少模型状态、激活和运行余量
2. **互连约束**：并行组位于机内还是跨机，链路带宽和时延是多少
3. **工作负载约束**：当前是训练还是服务，模型是稠密还是 MoE，上下文是否需要切分

从这个角度看，不同策略处理的对象并不一样：
- **DP** 解决“数据怎么铺开”，前提是模型至少接近能放下
- **TP** 解决“单层怎么拆算”，前提是机内互连足够强
- **PP** 解决“层怎么分段”，前提是模型足够深或单卡显存偏紧
- **EP** 解决“专家怎么分发”，前提是路由和 token 交换能承受
- **CP** 解决“长序列怎么切分”，前提是 Attention 状态交换可承受
- **Replica** 解决“请求怎么分发”，前提是多副本容量、队列和缓存局部性可观测

策略选择不是给候选打一个综合分，而是先记录可测量的约束，再排除不满足硬条件的候选。

| 策略 | 主要切分对象 | 主要压力 | 最关键前提 |
| --- | --- | --- | --- |
| DP / ZeRO | batch、参数、梯度或优化器状态 | 梯度同步与状态聚合 | 模型与训练状态有明确分片方案 |
| TP | 层内张量与激活 | 高频 collective | 机内互连强 |
| PP | layers / stages | pipeline bubble 与 stage 传输 | 模型足够深且 stage 可平衡 |
| CP | sequence / context | Attention 状态交换 | 长上下文收益覆盖通信代价 |
| EP | experts / tokens | dispatch、combine 与负载不均 | MoE 路由与容量可控 |
| Replica | serving requests | 队列、缓存局部性与尾延迟 | 多副本状态可观测 |

### Q1小验证

建立带单位的资源账本，检查单设备容量缺口。

```python
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
assert fits['fits_one_device']
assert not does_not_fit['fits_one_device']
print('可单卡容纳：', fits)
print('存在容量缺口：', does_not_fit)

```

## Q2：不同并行策略会引入什么通信成本？


通信成本不要只看“总量”，要同时看两个维度：

- **带宽项**：单次搬运的数据有多大
- **频率项**：这个搬运动作在整个训练路径里重复多少次

可以把它粗略写成：
$$\text{cost} \approx \text{latency} \times \text{次数} + \frac{\text{data volume}}{\text{bandwidth}}$$

这也是为什么同样叫“通信”，不同策略的脆弱点完全不同：
- **DP** 的梯度同步偏全局，频率低，但每次同步面更完整
- **TP** 的层内通信频率高，单次量通常不大，但每层都要碰一次
- **PP** 的通信发生在 stage 边界，频率低一些，但切不好就会放大气泡和等待
- **EP** 不只看 volume，还要看 token 路由和负载均衡是否稳定
- **CP** 随上下文长度交换 Attention 状态，需要同时记录序列长度和通信轮数
- **Replica** 不交换模型内部张量，但路由失衡会放大排队和尾延迟

所以，判断并行策略时，真正要问的是：
- 这个通信是不是高频出现
- 它能不能被机内互连吃掉
- 它是不是会卡在同步点上，压掉计算重叠

如果频率高、带宽弱，策略再“高级”，最后也会被通信拖回去。

### Q2小验证

用同一单位记录单次载荷、调用次数、链路带宽和固定时延。

```python
def communication_ledger(payload_mib, calls_per_step, bandwidth_gib_s, latency_us):
    """估算未计拥塞与重叠时，一个训练步内的通信下界。"""
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
high_frequency = communication_ledger(32, 24, 100, 5)
assert high_frequency['calls_per_step'] > low_frequency['calls_per_step']
print('低频大载荷：', low_frequency)
print('高频小载荷：', high_frequency)

```

## Q3：如何按约束筛选并行策略？


更实用的决策框架不是“选一个单点策略”，而是按硬约束依次排除：

1. **Fit gate**：先看单卡能不能把模型主体放下
   - 放得下，DP 才有基础；放不下，PP / ZeRO 先进入候选
2. **Interconnect gate**：再看机内互连能不能支撑高频通信
   - 强互连更支持 TP / EP；弱互连更偏向降低同步频率
3. **Structure gate**：最后看模型是不是 dense 还是 MoE
   - dense 常见组合是 DP / TP / PP
   - MoE 需要把 EP 纳入主决策
4. **Context gate**：上下文是否长到需要跨卡切分 Attention 状态
   - 单卡 Attention 状态放不下时，CP 才进入候选
5. **Serving gate**：目标是同步训练状态，还是分发独立请求
   - 多副本 Serving 优先评估 Replica 路由，不把它误写成训练 DP

候选页需要公开写出自己的设备显存需求、允许的工作负载、模型结构和链路域。只有这些要求都被当前环境满足，候选才进入后续实测。
### Q3小验证

让每个候选返回通过状态和拒绝原因，不生成含义不明的综合分数。

```python
LINK_SCOPE_ORDER = {'none': 0, 'cross_node': 1, 'intra_node': 2}

def evaluate_candidate(candidate, environment):
    """逐项核对候选声明的硬约束，并返回明确拒绝原因。"""
    reasons = []
    if candidate['memory_per_rank_gib'] > environment['memory_per_device_gib']:
        reasons.append('per-rank memory exceeds device memory')
    if environment['mode'] not in candidate['modes']:
        reasons.append('workload mode is unsupported')
    if environment['model_kind'] not in candidate['model_kinds']:
        reasons.append('model structure is unsupported')
    if LINK_SCOPE_ORDER[environment['link_scope']] < LINK_SCOPE_ORDER[candidate['required_link_scope']]:
        reasons.append('interconnect scope is insufficient')
    return {'name': candidate['name'], 'accepted': not reasons, 'reasons': reasons}

environment = {
    'memory_per_device_gib': 24,
    'mode': 'training',
    'model_kind': 'dense',
    'link_scope': 'intra_node',
}
candidates = [
    {'name': 'DP', 'memory_per_rank_gib': 29, 'modes': {'training'}, 'model_kinds': {'dense', 'moe'}, 'required_link_scope': 'cross_node'},
    {'name': 'ZeRO-3', 'memory_per_rank_gib': 18, 'modes': {'training'}, 'model_kinds': {'dense', 'moe'}, 'required_link_scope': 'cross_node'},
    {'name': 'TP', 'memory_per_rank_gib': 16, 'modes': {'training', 'serving'}, 'model_kinds': {'dense', 'moe'}, 'required_link_scope': 'intra_node'},
    {'name': 'EP', 'memory_per_rank_gib': 14, 'modes': {'training', 'serving'}, 'model_kinds': {'moe'}, 'required_link_scope': 'intra_node'},
]
reports = [evaluate_candidate(candidate, environment) for candidate in candidates]
assert not next(report for report in reports if report['name'] == 'DP')['accepted']
assert next(report for report in reports if report['name'] == 'TP')['accepted']
for report in reports:
    print(report)

```

## Q4：选择并行策略后，如何检查方案风险？


单个候选通过硬门槛，不代表组合后仍然可行。组合检查至少包括设备数、每个维度的度数、模型结构和部署域；性能风险则继续交给真实通信与负载证据验证。

例如，`TP=4, PP=2, DP=2` 至少需要 16 个 rank；EP 只能用于 MoE；服务副本与训练 DP 都复制模型，但二者的同步语义不同，不能用同一个名称代替。
### Q4小验证

检查组合所需 rank 数与结构条件，输出可定位的问题。

```python
def validate_parallel_composition(degrees, available_ranks, model_kind):
    """检查并行度乘积、可用 rank 数和模型结构。"""
    supported = {'DP', 'TP', 'PP', 'CP', 'EP', 'Replica'}
    unknown = set(degrees) - supported
    if unknown:
        raise ValueError(f'未知并行维度：{sorted(unknown)}')
    if any(not isinstance(degree, int) or degree < 1 for degree in degrees.values()):
        raise ValueError('每个并行度必须是大于等于 1 的整数')
    required_ranks = 1
    for degree in degrees.values():
        required_ranks *= degree
    issues = []
    if required_ranks > available_ranks:
        issues.append(f'requires {required_ranks} ranks but only {available_ranks} are available')
    if degrees.get('EP', 1) > 1 and model_kind != 'moe':
        issues.append('EP degree greater than 1 requires an MoE model')
    if degrees.get('Replica', 1) > 1 and degrees.get('DP', 1) > 1:
        issues.append('Replica and training DP must use distinct execution roles')
    return {'required_ranks': required_ranks, 'valid': not issues, 'issues': issues}

valid_plan = validate_parallel_composition({'TP': 4, 'PP': 2, 'DP': 2}, 16, 'dense')
invalid_plan = validate_parallel_composition({'TP': 4, 'PP': 2, 'EP': 2}, 8, 'dense')
assert valid_plan['valid']
assert not invalid_plan['valid']
print('可行组合：', valid_plan)
print('不可行组合：', invalid_plan)

```

## 相关阅读

**导语：** 继续进入策略选择页、上下文切分页和 PyTorch 分布式文档，核对候选所需的真实证据。

- [Part 02 · 49 并行策略选择](../02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection.md)
- [Part 02 · PAR-CONTEXT 上下文与序列并行](../02_PyTorch_Algorithms/PAR-CONTEXT_Context_and_Sequence_Parallelism.md)
- [PyTorch Distributed | PyTorch 分布式训练文档](https://pytorch.org/docs/stable/distributed.html)
---
