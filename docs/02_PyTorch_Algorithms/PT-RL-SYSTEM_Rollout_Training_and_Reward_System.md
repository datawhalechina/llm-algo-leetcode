# PT-RL-SYSTEM. Rollout Training and Reward System | 采样、训练与奖励系统

**难度：** Hard | **环境：** [CPU 机制练习](#runtime-environment) · [verl 单卡实验](#gpu-experiment) | **标签：** `后训练`, `强化学习系统`, `资源调度`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线后训练把轨迹生成、参考策略计算、奖励或验证、参数更新和权重分发连接成循环。单个角色跑得快不代表系统稳定：候选展开比例、批量聚合、队列增长、状态驻留和同步失败都会改变端到端容量。

系统比较先把轨迹生成、奖励和学习器的处理率换算成相同的工作单位，再检查各资源池的状态驻留、共享链路占用、版本确认和恢复条件。账本通过表示候选可以进入真实多工作进程实验，不表示已经获得吞吐收益。

**关键词：** 轨迹生成器、学习器、共置、分离部署、流水线瓶颈
<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。

- **GPU 实测：** 使用 [`PT-VERL-CUDA`](../topic_discussion/post_training_optimization/quickstart.md#pt-verl-cuda-quickstart)，从 [Step 5](#gpu-experiment) 运行单卡同步系统实验。

---
## 前置阅读

**导语：** 先理解轨迹身份和策略同步，再把算法角色放入多阶段系统。

- [Part 02 · PT-ROLLOUT-LIFECYCLE Rollout 与策略新鲜度](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)
- [Part 02 · 16 PPO 资源与状态](./16_RLHF_PPO_Memory.md)
## Step 1：一条提示会在各角色之间产生多少工作量

角色可以共享 GPU，也可以拆到不同资源池。选择拓扑前，要同时明确每个角色持有的状态、输入输出和阶段倍率。例如一个提示可能生成 `G` 条候选轨迹，奖励阶段处理 `G` 个回答，而学习器等待若干提示及其候选聚合成一次更新；这些对象不能都笼统记作一个“样本”。

因此系统账本先选定统一工作单位，例如“每个提示组”或“每条轨迹”，再把各阶段处理率换算到同一单位。例如每个提示生成 4 条轨迹时，奖励器每秒处理 80 条轨迹，只相当于每秒处理 20 个提示组；若提示组到达率为每秒 24 组，奖励队列仍会持续增长。表格列出逻辑角色的状态与倍率，图中继续展示轨迹、奖励、参考概率和新权重怎样流动；逻辑角色并不限定物理部署方式。

| 角色 | 主要驻留状态 | 输入与输出 | 数量关系 |
| --- | --- | --- | --- |
| 轨迹生成器 | 当前策略、KV Cache、采样状态 | 提示 → 回答、行为对数概率与策略版本 | 每个提示生成 `G` 条候选 |
| 奖励 / 验证器 | 奖励模型或规则版本 | 轨迹 → 奖励、验证状态 | 通常逐轨迹处理 |
| 参考策略服务 | 冻结参考策略 | 轨迹 → 参考对数概率 | 按算法需要逐轨迹处理 |
| 学习器 | 策略、优化器、梯度与训练状态 | 合格批次 → 新检查点 | 聚合多个提示组后更新 |
| 产物协调器 | 版本、队列、哈希与恢复状态 | 新产物 → 分发确认与活动版本 | 每次更新触发一次同步 |

![后训练系统角色与数据流](../public/02_PyTorch_Algorithms/pt_rl_system_topology.svg)
## Step 2：共置与分离部署分别增加哪些状态和传输

共置减少跨池边，但会让策略、参考模型、奖励模型和优化器状态竞争同一资源池；分离允许角色独立扩缩容，却新增轨迹、对数概率和权重传输。混合拓扑只在明确哪些高频边需要共置、哪些重状态必须拆分后才有意义。

| 候选 | 状态驻留 | 新增传输与同步 | 故障与调度代价 | 首先验证 |
| --- | --- | --- | --- | --- |
| 共置 | 多类模型与训练状态共享资源池 | 跨池传输少，仍需阶段间状态交接 | 显存争用、阶段切换和故障域扩大 | 分阶段峰值显存、切换空转时间 |
| 分离 | 各角色独立驻留并可分别扩缩容 | 轨迹、对数概率、奖励和权重跨池 | 队列积压、版本滞后和跨池恢复 | 每条边的载荷率、确认延迟与积压 |
| 混合 | 高频交互角色共置，重状态拆分 | 只保留选定跨池边 | 配置复杂，局部争用与传输同时存在 | 状态副本、关键跨池边和失败切换 |
## Step 3：如何用统一处理率、传输占用和硬门槛筛选拓扑

先依据阶段倍率，把采样、奖励和学习阶段的原始处理率换算为统一工作单位。令 $q_j$ 是阶段 $j$ 的原始对象处理率，$m_j$ 是每个统一工作单位需要处理的对象数，$\lambda$ 是统一工作单位到达率。阶段等价处理率 $c_j$、流水线容量 $C_{pipe}$ 和队列余量 $s_j$ 为：

$$
c_j=\frac{q_j}{m_j}, \qquad C_{pipe}=\min_j c_j, \qquad s_j=c_j-\lambda.
$$

例如以提示组为单位时，逐轨迹阶段的处理率需要除以每组候选数 $G$，学习阶段则按每次更新消费的提示组数换算。$s_j<0$ 表示该阶段的队列会持续增长。

对跨池边 $e$，令 $z_e$ 是单个对象的字节数，$m_e$ 是每个工作单位包含的对象数，$\lambda_e$ 是该边的工作单位频率，$B_e$ 是有效带宽，$o_e\in[0,1]$ 是可重叠比例。传输载荷、带宽占用和暴露占用分别为：

$$
b_e=z_e m_e\lambda_e, \qquad u_e=\frac{b_e}{B_e}, \qquad x_e=u_e(1-o_e).
$$

共享同一互连域 $f$ 的边会争用带宽，因此域内暴露占用需要相加；不同互连域可以并行，关键互连预算取各域聚合值的最大值：

$$
X_f=\sum_{e\in f}x_e, \qquad X_{critical}=\max_f X_f.
$$

$X_{critical}$ 仍是扣除可重叠部分后的传输暴露预算代理，不能直接当作端到端关键路径时间。

候选部署先形成资源池显存、阶段处理率和跨池传输三类账本，再执行两层判断：显存超限、版本未确认或恢复路径缺失属于硬失败；队列增长或传输暴露过高说明需要调优。账本全部通过只表示该部署值得进入真实多工作进程实验，不代表已经获得吞吐收益。

![候选拓扑的准入与瓶颈判断](../public/02_PyTorch_Algorithms/pt_rl_system_decision.svg)
<a id="mechanism-practice"></a>

## Step 4：实现系统账本与候选准入

题目区把阶段速率换算为统一工作单位，保留各阶段队列余量；随后逐边建立跨池传输账本并按互连域聚合；最后用结构化原因记录硬失败与调优压力。骨架保留输入和报告形状，学习者补全会改变容量、传输与准入结论的计算。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：定位流水线瓶颈<br>`pipeline_capacity` | 换算阶段等价处理率，并报告容量、瓶颈、余量和积压阶段 | 阶段集合一致；速率和倍率为正；到达率非负 | 倍率归一、并列瓶颈、稳定与积压阶段，以及非法阶段映射 |
| TODO 2：建立跨池传输账本<br>`transfer_ledger` | 计算逐边载荷与暴露占用，再按共享互连域聚合 | 同池边传输为零；同域相加；独立域取最大值；重叠比例属于 0–1 | 同池排除、共享域累加、独立域并行、关键互连域与非法重叠比例 |
| TODO 3：执行拓扑准入<br>`evaluate_topology` | 汇总显存、版本、恢复、队列和通信触发原因 | 硬失败优先于调优；同级原因不能被首个分支吞掉 | 接受、队列与链路调优，以及显存、同步、恢复的组合硬失败 |

```python
from typing import Dict, List

```


```python
def pipeline_capacity(stage_rates, arrival_rate, items_per_work_unit):
    """把阶段速率归一到同一工作单位，并保留瓶颈与积压证据。"""
    # ==========================================
    # TODO 1：计算 effective_rates、capacity、queue_margins、bottlenecks 与 unstable_stages。
    # 同一容量可能对应多个瓶颈；负余量阶段必须逐项保留。
    # ==========================================
    raise NotImplementedError("TODO 1：计算流水线容量与队列余量")

def transfer_ledger(edges, placement):
    """建立逐边账本，并按共享互连域汇总暴露占用。"""
    # ==========================================
    # TODO 2：同池边载荷记零；跨池边计算 traffic、transfer 与 exposed。
    # 同一 fabric 的 exposed 相加，critical_fabrics 保留最大值对应的全部互连域。
    # ==========================================
    raise NotImplementedError("TODO 2：建立跨池传输账本")

def evaluate_topology(capacity_report, memory_by_pool, memory_limits, transfer_report, version_sync_ok, recovery_ok, max_exposed_s_per_s=0.1):
    """按 reject > tune > accept 的优先级返回全部触发原因。"""
    # ==========================================
    # TODO 3：汇总 hard_failures 与 tuning_pressures，再决定 decision。
    # 报告同时保留超限资源池、积压阶段与过载互连域。
    # ==========================================
    raise NotImplementedError("TODO 3：执行系统拓扑准入")
```

## 测试

测试分别验证瓶颈归因、跨池载荷和系统准入。

```python
def test_capacity():
    """倍率归一、并列瓶颈和积压阶段必须同时可见。"""
    result = pipeline_capacity(
        {'rollout': 240, 'reward': 160, 'learner': 80}, arrival_rate=70,
        items_per_work_unit={'rollout': 3, 'reward': 2, 'learner': 1},
    )
    assert result['effective_rates'] == {'rollout': 80.0, 'reward': 80.0, 'learner': 80.0}
    assert result['bottlenecks'] == ['learner', 'reward', 'rollout']
    assert result['unstable_stages'] == [] and result['queue_stable'] is True
    unstable = pipeline_capacity({'rollout': 120, 'reward': 80}, 90, {'rollout': 1, 'reward': 1})
    assert unstable['unstable_stages'] == ['reward']
    try:
        pipeline_capacity({'rollout': 120}, 20, {'reward': 1})
        raise AssertionError('阶段集合不一致时应失败')
    except ValueError:
        pass
    try:
        pipeline_capacity({'rollout': float('nan')}, 20, {'rollout': 1})
        raise AssertionError('非有限阶段速率应失败')
    except ValueError:
        pass

def test_transfer():
    """同池边排除、共享域累加，独立域只参与最大值比较。"""
    edges = [
        {'source': 'rollout', 'target': 'reward', 'bytes_per_item': 100, 'items_per_work_unit': 2, 'work_units_per_second': 5, 'bandwidth_bytes_per_s': 1000, 'overlap_fraction': 0.5, 'fabric_id': 'pcie-0'},
        {'source': 'learner', 'target': 'rollout', 'bytes_per_item': 1000, 'items_per_work_unit': 1, 'work_units_per_second': 1, 'bandwidth_bytes_per_s': 2000, 'overlap_fraction': 0.0, 'fabric_id': 'pcie-1'},
        {'source': 'reward', 'target': 'learner', 'bytes_per_item': 500, 'items_per_work_unit': 1, 'work_units_per_second': 1, 'bandwidth_bytes_per_s': 2000, 'overlap_fraction': 0.0, 'fabric_id': 'pcie-1'},
        {'source': 'verifier', 'target': 'learner', 'bytes_per_item': 1000, 'items_per_work_unit': 1, 'work_units_per_second': 1, 'bandwidth_bytes_per_s': 2000, 'overlap_fraction': 0.5, 'fabric_id': 'pcie-2'},
    ]
    placement = {'rollout': 'A', 'reward': 'A', 'learner': 'B', 'verifier': 'C'}
    report = transfer_ledger(edges, placement)
    assert report['cross_pool_bytes_per_s'] == 2500
    assert report['fabric_exposed_seconds_per_s'] == {'pcie-1': 0.75, 'pcie-2': 0.25}
    assert report['critical_exposed_seconds_per_s'] == 0.75
    assert report['critical_fabrics'] == ['pcie-1']
    bad_edge = [dict(edges[1], overlap_fraction=1.2)]
    try:
        transfer_ledger(bad_edge, placement)
        raise AssertionError('非法重叠比例应失败')
    except ValueError:
        pass
    inconsistent_fabric = [edges[1], dict(edges[2], bandwidth_bytes_per_s=4000)]
    try:
        transfer_ledger(inconsistent_fabric, placement)
        raise AssertionError('同一互连域使用不同带宽口径时应失败')
    except ValueError:
        pass
    tied = transfer_ledger([edges[1], dict(edges[3], overlap_fraction=0.0)], placement)
    assert tied['critical_fabrics'] == ['pcie-1', 'pcie-2']

def test_gate():
    """拒绝优先于调优，同一级别的触发原因全部保留。"""
    multipliers = {'rollout': 1, 'reward': 1}
    stable = pipeline_capacity({'rollout': 120, 'reward': 80}, 70, multipliers)
    unstable = pipeline_capacity({'rollout': 120, 'reward': 80}, 90, multipliers)
    low_transfer = {'fabric_exposed_seconds_per_s': {'pcie-0': 0.05}, 'critical_exposed_seconds_per_s': 0.05}
    high_transfer = {'fabric_exposed_seconds_per_s': {'pcie-0': 0.20}, 'critical_exposed_seconds_per_s': 0.20}
    accepted = evaluate_topology(stable, {'A': 30}, {'A': 40}, low_transfer, True, True)
    assert accepted['decision'] == 'accept' and accepted['reasons'] == []
    tuned = evaluate_topology(unstable, {'A': 30}, {'A': 40}, high_transfer, True, True)
    assert tuned['decision'] == 'tune'
    assert tuned['reasons'] == ['unstable_queue', 'exposed_transfer']
    assert tuned['unstable_stages'] == ['reward'] and tuned['overloaded_fabrics'] == ['pcie-0']
    rejected = evaluate_topology(stable, {'A': 50}, {'A': 40}, high_transfer, False, False)
    assert rejected['decision'] == 'reject'
    assert rejected['reasons'] == ['memory_limit', 'version_sync', 'recovery_not_ready']
    try:
        evaluate_topology(stable, {'A': float('nan')}, {'A': 40}, low_transfer, True, True)
        raise AssertionError('非有限显存账本应失败')
    except ValueError:
        pass

test_capacity(); test_transfer(); test_gate()
print('✅ 后训练系统机制测试通过')
```

---

🛑 **STOP HERE** 🛑

> 请先完成题目区并运行测试，再查看参考实现。

---
## 参考代码与解析

```python
def pipeline_capacity(stage_rates, arrival_rate, items_per_work_unit):
    import math

    rates = [float(rate) for rate in stage_rates.values()]
    multipliers = [float(value) for value in items_per_work_unit.values()]
    arrival_rate = float(arrival_rate)
    if not stage_rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('stage rate 必须为正有限数')
    if not math.isfinite(arrival_rate) or arrival_rate < 0:
        raise ValueError('arrival_rate 必须是非负有限数')
    if set(stage_rates) != set(items_per_work_unit) or any(not math.isfinite(value) or value <= 0 for value in multipliers):
        raise ValueError('每个阶段都必须提供正有限工作单位倍率')
    effective_rates = {name: float(rate) / float(items_per_work_unit[name]) for name, rate in stage_rates.items()}
    capacity = min(effective_rates.values())
    bottlenecks = sorted(name for name, rate in effective_rates.items() if math.isclose(rate, capacity, rel_tol=1e-9, abs_tol=1e-12))
    queue_margins = {name: rate - arrival_rate for name, rate in effective_rates.items()}
    unstable_stages = sorted(name for name, margin in queue_margins.items() if margin < 0)
    return {'capacity': capacity, 'effective_rates': effective_rates, 'bottlenecks': bottlenecks, 'queue_margins': queue_margins,
            'unstable_stages': unstable_stages, 'queue_stable': not unstable_stages}

def transfer_ledger(edges, placement):
    import math

    rows, total_bytes, fabric_exposed, fabric_bandwidths = [], 0.0, {}, {}
    for edge in edges:
        source, target = edge['source'], edge['target']
        if source not in placement or target not in placement:
            raise ValueError('每条边的两端都必须登记资源池')
        cross_pool = placement[source] != placement[target]
        overlap = float(edge.get('overlap_fraction', 0.0))
        bandwidth = float(edge['bandwidth_bytes_per_s'])
        if not math.isfinite(overlap) or not math.isfinite(bandwidth) or not 0.0 <= overlap <= 1.0 or bandwidth <= 0:
            raise ValueError('重叠比例须属于 0–1，带宽必须为正有限数')
        bytes_per_item = float(edge['bytes_per_item'])
        items_per_work_unit = float(edge['items_per_work_unit'])
        work_units_per_second = float(edge['work_units_per_second'])
        values = (bytes_per_item, items_per_work_unit, work_units_per_second)
        if any(not math.isfinite(value) for value in values) or bytes_per_item < 0 or items_per_work_unit <= 0 or work_units_per_second < 0:
            raise ValueError('对象字节数和到达率必须有限且非负，工作单位倍率必须为正')
        traffic = bytes_per_item * items_per_work_unit * work_units_per_second if cross_pool else 0.0
        transfer = traffic / bandwidth
        exposed = transfer * (1.0 - overlap)
        total_bytes += traffic
        if cross_pool:
            fabric_id = str(edge.get('fabric_id', '')).strip()
            if not fabric_id:
                raise ValueError('跨池边必须登记 fabric_id')
            previous_bandwidth = fabric_bandwidths.setdefault(fabric_id, bandwidth)
            if not math.isclose(previous_bandwidth, bandwidth, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('同一 fabric_id 必须使用同一有效带宽口径')
            fabric_exposed[fabric_id] = fabric_exposed.get(fabric_id, 0.0) + exposed
        rows.append(edge | {'cross_pool': cross_pool, 'traffic_bytes_per_s': traffic,
                            'transfer_seconds_per_s': transfer, 'exposed_seconds_per_s': exposed})
    critical_exposed = max(fabric_exposed.values(), default=0.0)
    critical_fabrics = sorted(name for name, value in fabric_exposed.items() if value == critical_exposed) if fabric_exposed else []
    return {'cross_pool_bytes_per_s': total_bytes, 'fabric_exposed_seconds_per_s': fabric_exposed,
            'critical_exposed_seconds_per_s': critical_exposed, 'critical_fabrics': critical_fabrics, 'rows': rows}

def evaluate_topology(capacity_report, memory_by_pool, memory_limits, transfer_report, version_sync_ok, recovery_ok, max_exposed_s_per_s=0.1):
    import math

    max_exposed_s_per_s = float(max_exposed_s_per_s)
    memory_values = [float(value) for value in memory_by_pool.values()] + [float(value) for value in memory_limits.values()]
    transfer_values = [float(value) for value in transfer_report.get('fabric_exposed_seconds_per_s', {}).values()]
    if not math.isfinite(max_exposed_s_per_s) or max_exposed_s_per_s < 0:
        raise ValueError('传输暴露预算必须是非负有限数')
    if any(not math.isfinite(value) or value < 0 for value in memory_values + transfer_values):
        raise ValueError('显存与传输账本必须是非负有限数')
    over_limit_pools = sorted(pool for pool, used in memory_by_pool.items()
                              if pool not in memory_limits or used > memory_limits[pool])
    hard_failures = []
    if over_limit_pools:
        hard_failures.append('memory_limit')
    if not version_sync_ok:
        hard_failures.append('version_sync')
    if not recovery_ok:
        hard_failures.append('recovery_not_ready')

    unstable_stages = sorted(capacity_report.get('unstable_stages', []))
    overloaded_fabrics = sorted(
        name for name, value in transfer_report.get('fabric_exposed_seconds_per_s', {}).items()
        if value > max_exposed_s_per_s
    )
    tuning_pressures = []
    if unstable_stages or not capacity_report['queue_stable']:
        tuning_pressures.append('unstable_queue')
    if overloaded_fabrics or transfer_report['critical_exposed_seconds_per_s'] > max_exposed_s_per_s:
        tuning_pressures.append('exposed_transfer')

    decision = 'reject' if hard_failures else ('tune' if tuning_pressures else 'accept')
    reasons = hard_failures if hard_failures else tuning_pressures
    return {'decision': decision, 'reasons': reasons, 'over_limit_pools': over_limit_pools,
            'unstable_stages': unstable_stages, 'overloaded_fabrics': overloaded_fabrics}
```

### 解析

#### TODO 1：计算流水线容量与队列余量

- **机制：** 用每工作单位对象数把各阶段处理率归一到同一工作单位，再由最慢阶段决定流水线容量。
- **不变量：** 到达率必须与阶段容量使用同一单位；任一阶段的等价处理率低于到达率都会造成队列持续增长。
- **测试证据：** 测试覆盖倍率归一、并列瓶颈、积压阶段和阶段映射不一致。

#### TODO 2：建立跨池传输账本

- **机制：** 只为跨资源池边建立传输账本，并按共享互连域聚合扣除重叠后的暴露占用。
- **不变量：** 同一互连域的边竞争带宽，需要相加；独立互连域可以并行，关键值取最繁忙互连域而不是全边总和。
- **测试证据：** 测试核对同池边排除、共享域累加、独立域并行、关键互连域和非法重叠比例。

#### TODO 3：执行拓扑准入

- **机制：** 汇总分池显存、版本确认和恢复能力的硬失败，再记录队列与互连域的调优压力。
- **不变量：** `reject` 优先于 `tune`；同一级别的多个触发原因必须全部保留。
- **测试证据：** 测试覆盖接受、队列与链路同时调优，以及显存、同步、恢复同时失败。

<a id="gpu-experiment"></a>

## Step 5：比较共置阶段耗时与引擎状态驻留

CPU 练习换算各角色工作量并检查容量、资源池和链路；verl 单卡实验记录实际阶段事件及状态驻留。释放/保留对照保持输入和更新方式相同，关注生成引擎休眠、权重恢复、KV 恢复与学习器之间的资源交接。

| 学习问题 | CPU 已确认 | GPU 需要的真实证据 |
|:---|:---|:---|
| 各阶段实际花多少时间 | 按阶段倍率统一工作单位 | 生成、概率计算、更新、同步及状态切换事件；初始化单独记录 |
| 哪些状态竞争同一资源池 | 驻留账本与显存预算检查 | 把事件与资源采样对应，区分引擎释放/保留；整卡峰值差不是 KV 大小 |
| 哪些结果可用于后续拓扑实验 | 跨池载荷、链路占用与队列余量 | 单卡阶段与状态证据；跨池带宽和队列稳定性需另行多工作进程测量 |

### 5.1 环境与固定工作负载

使用已有的 [`PT-VERL-CUDA`](../topic_discussion/post_training_optimization/quickstart.md#pt-verl-cuda-quickstart)。与轨迹生命周期页共用 v2 标签；该页完成运行后，本页只需运行 5.1、5.3 读取结果。尚未运行时，先 `dry_run` 预检，再改为 `real_gpu` 运行 5.1–5.3。本轮只确认初始参数一致的配对，不新增训练或性能搜索。 若仅提示缺少模型缓存，其他环境检查已通过，可改为 `real_gpu` 自动准备固定版本；CPU 核对通过后才开始训练。

| 配置 | 固定内容与操作 |
|:---|:---|
| 学习者配置 | 仅设置模式、共同标签和已有 verl 目录，结果路径自动推导 |
| 模型与更新 | Qwen2.5-0.5B-Instruct、LoRA rank 16；verl 内置监督损失，一次更新，与轨迹页相同 |
| 配对变量 | 仅改变 `free_cache_engine`；保留组训练前必须匹配释放组的初始 LoRA 哈希，其余配置与阈值不变 |
| 采集范围 | 事件使用单机单调时钟；每 0.25 秒尝试采样整卡与整机占用，短阶段漏采峰值记为 null |
| 结果目标 | 检查真实生成、评分、更新、同步与状态恢复，不以节省显存或提高吞吐作为通过前提 |


```python
from pathlib import Path
from tools.project_runtime import resolve_project_root

RUN_MODE = "dry_run"  # dry_run 仅预检；通过后改为 real_gpu。
RUN_LABEL = "rtx5070ti_verl_mechanism_closure_v2"  # 两页共用；新实验换标签，已有结果只读。
VERL_ROOT = Path.home() / "verl-v0.9.1"  # 已安装的 PT-VERL-CUDA，不创建新环境。

PROJECT_ROOT = resolve_project_root()
RESULT_PATH = PROJECT_ROOT / "benchmarks/results/pt_verl_closure" / f"{RUN_LABEL}.json"
if RUN_MODE not in {"dry_run", "real_gpu"}:
    raise ValueError(f"未知运行模式：{RUN_MODE}")
print({"run_mode": RUN_MODE, "run_label": RUN_LABEL, "result_path": str(RESULT_PATH)})

```

### 5.2 执行释放与保留对照

两组各自初始化 verl actor 和生成引擎，完成一次受控更新及同步。记录实际恢复调用与状态标签，并把控制端阶段事件和资源采样按单调时钟对应；边界时刻的 actor 显存与持续采样的整卡峰值分开保存。

| 对照 | 固定或改变 | 必须保留的证据 |
|:---|:---|:---|
| 引擎释放 | 开启释放/恢复 | 休眠阶段、权重传输和恢复调用、阶段时间及资源记录 |
| 引擎保留 | 仅关闭释放/恢复 | 相同输入和更新方式的阶段与资源记录 |
| 配对核对 | 除该开关外配置相同，冻结轨迹来源相同 | 配置与来源一致性、逐项判断；不把整卡峰值差直接解释为 KV 大小 |


```python
import json
import subprocess
import sys
from tools.verl_mechanism_closure import SCHEMA

if RESULT_PATH.is_file():
    saved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if saved.get("schema_version") != SCHEMA or saved.get("run_label") != RUN_LABEL:
        raise RuntimeError("已有文件的格式或运行标签不匹配，请检查结果路径。")
    print(f"读取已有共同结果，不重复运行：{RESULT_PATH}；继续 5.3。")
else:
    command = [sys.executable, "-m", "tools.verl_mechanism_closure",
               "--mode", "preflight" if RUN_MODE == "dry_run" else "run",
               "--verl-root", str(VERL_ROOT), "--run-label", RUN_LABEL,
               "--output", str(RESULT_PATH)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode not in ({0, 4} if RUN_MODE == "dry_run" else {0}):
        print(f"结果目录：{RESULT_PATH.parent}；同标签 *_release.log / *_retain.log 保留错误详情。")
        raise RuntimeError("预检或实际执行未完成，请先查看上方具体检查与候选日志，不要直接重复运行。")
    if completed.returncode == 4:
        print("环境检查通过，仅缺模型缓存；改为 real_gpu 后会先准备固定 commit，CPU 核对通过才启动 GPU。")
    elif RUN_MODE == "dry_run":
        print("预检通过后，将 5.1 的 RUN_MODE 改为 real_gpu，再运行 5.1 → 5.2 → 5.3。")

```

### 5.3 读取阶段与状态结果

从共同结果读取本页判断：两组是否完成更新和同步、是否保留实际阶段与恢复事件，以及采样覆盖情况。逐阶段峰值为 null 表示该阶段没有可用样本，不记为零；本轮不报告稳态加速比。


```python
import json
from tools.verl_mechanism_closure import read_summary

if RESULT_PATH.is_file():
    saved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if saved.get("run_label") != RUN_LABEL:
        raise RuntimeError("结果标签与 5.1 不一致。")
    print(json.dumps(read_summary(RESULT_PATH, "system"), ensure_ascii=False, indent=2))
else:
    print("当前标签尚无 GPU 结果。dry_run 不生成实验结论；实际运行完成后再执行 5.3。")

```

### 5.4 解释阶段与引擎状态证据

[受控更新 v2](../../benchmarks/results/pt_verl_closure/rtx5070ti_verl_mechanism_closure_v2.json) 的两组初始参数及更新后参数哈希分别一致，更新与同步、阶段事件、恢复调用、配置配对及资源采样检查全部通过。本机单卡的阶段与引擎状态验证完成；监督更新阶段整卡采样峰值为释放组 6392 MiB、保留组 10489 MiB。

释放组整段峰值为 10086 MiB，出现在初次同步阶段，不能用整段峰值代替更新阶段占用。短阶段未采到资源时继续保留 null；整卡与整机采样包含其他应用，单次配对不证明稳态加速，显存差也不等于纯 KV Cache 大小。当前实验可收尾，[v1](../../benchmarks/results/pt_verl_closure/rtx5070ti_verl_mechanism_closure_v1.json) 作为未通过严格配对的历史记录保留，无需追加性能搜索。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
|:---|:---|:---|:---|
| 配对与同步 | `system_decision.checks` | 初始参数、配置、轨迹来源及两组更新同步均通过 | 仅改变引擎释放/保留配置，保留本轮单卡机制结论 |
| 状态切换 | `worker_events`、资源侧文件 `events` | 释放组记录 weights/KV 恢复；保留组不调用恢复，符合配置 | 将真实恢复调用与休眠阶段、权重传输对应，不以配置名字代替执行证据 |
| 更新阶段资源 | `system_decision.measurements.*.phases` | 释放 6392 MiB、1.367 s；保留 10489 MiB、1.345 s | 两组更新期间分别有 5/6 个采样点；保留观测值，不据一次运行宣布加速 |
| 整段资源 | `sampled_gpu_peak_mb` | 释放 10086 MiB，保留 10489 MiB | 释放组峰值来自初次同步，与更新阶段峰值分开解释 |
| 短阶段覆盖 | `phases.*.sample_count`、`sampled_gpu_peak_mb` | 释放组更新前后 rollout 评分阶段均无资源采样点 | 峰值记为 null；阶段事件仍保留，不当作零显存 |

## 相关阅读

- [Part 02 · 85 GRPO 组级对齐项目](./85_GRPO_Groupwise_Alignment_Project.md)
- [Part 02 · 86 在线后训练生命周期与采用决策](./86_DPO_Online_Benchmark.md)
- [verl](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [TRL](https://github.com/huggingface/trl)
