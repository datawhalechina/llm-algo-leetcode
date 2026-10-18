# PT-ROLLOUT-LIFECYCLE. Rollout and Policy Freshness | 轨迹与策略新鲜度

**难度：** Hard | **环境：** [CPU 机制练习](#runtime-environment) | **标签：** `后训练`, `轨迹生命周期`, `策略新鲜度`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线后训练不能把生成结果直接送进下一次更新。每条轨迹都由确定的模型、策略版本和采样配置生成，还会经历执行、奖励、准入、消费和失效等状态；任一身份或状态缺失，都可能把错误数据混入参数更新。

轨迹依次经历 `生成 → 奖励 → 准入 → 预留 → 消费`。只有必需身份字段齐全、版本与行为偏离均在阈值内且尚未消费的轨迹才能绑定到一次更新；新策略产物只有在全部文件写入完成、且所有必需工作进程确认相同版本和哈希后才能激活。

**关键词：** 轨迹身份、策略版本、新鲜度、版本同步

---
<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。

---
## 前置阅读

**导语：** 按准备进入的在线更新分支选择前置页，再观察策略版本变化如何影响轨迹准入。

- PPO / GRPO 分支：[Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势信号](./PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md)
- Online DPO 分支：[Part 02 · 50 偏好数据与对齐评测](./50_Preference_Data_and_Evaluation.md) → [Part 02 · 17 DPO 损失](./17_DPO_Loss_Tutorial.md)

---
## Step 1：一条轨迹需要记录哪些身份和处理状态

一条轨迹需要同时记录五类信息：轨迹身份与任务来源说明它从哪里来，行为策略身份说明由哪个策略产物生成，生成条件说明分词与采样方式，奖励身份说明反馈来自哪套规则，动作证据则保存策略更新需要的有效动作与行为概率。缺少任一类信息，轨迹都无法可靠去重、复现或归因。

身份信息之外，还要独立维护生命周期。正常路径是“已生成 → 已奖励 → 已预留 → 已消费”；生成失败、奖励未完成和更新失败不能统一记成零奖励。预留把轨迹独占地绑定到一次更新，提交成功后才能进入已消费状态，更新中止时则按恢复规则释放或隔离。检查可用与写入预留必须构成一次不可分割的状态迁移，否则两个学习器可能重复消费同一条轨迹。

| 信息类别 | 回答的问题 | 不满足时的动作 |
| --- | --- | --- |
| 轨迹身份与任务来源 | 这是哪次采样、对应哪个任务 | 无法去重或追溯时拒绝 |
| 行为策略身份 | 哪个不可变策略产物生成了轨迹 | 版本与产物身份矛盾时隔离 |
| 生成条件 | 使用了什么分词和采样方式 | 与当前实验条件不一致时拒绝或分组 |
| 生命周期与奖励状态 | 生成和奖励是否完成，轨迹是否已被预留或消费 | 未完成则等待，失败或终态轨迹不得再次预留 |
| 动作证据 | 有效动作和行为概率是否足以支持更新 | 证据缺失或位置错位时拒绝 |

![轨迹与策略版本生命周期](../public/02_PyTorch_Algorithms/pt_rollout_lifecycle.svg)
## Step 2：版本滞后与行为偏离如何决定轨迹准入

版本编号只能给出策略快照的新旧关系，不能说明两个版本在当前动作上的实际差异，因此轨迹准入还要检查行为偏离。令当前学习器策略版本为 $v_{current}$，轨迹的行为策略版本为 $v_{behavior}$，$m_t\in\{0,1\}$ 表示动作掩码，$\pi_{current}$ 与 $\pi_{behavior}$ 分别表示当前策略和行为策略。版本滞后 $\Delta v$ 与动作位置平均行为偏离 $D_{beh}$ 为：

$$
\Delta v=v_{current}-v_{behavior}, \qquad
D_{beh}=\frac{\sum_t m_t\left|\log\pi_{current}(a_t\mid s_t)-\log\pi_{behavior}(a_t\mid s_t)\right|}{\sum_t m_t}.
$$

其中 $s_t$、$a_t$ 是位置 $t$ 的状态与已采样动作。$\Delta v=0$ 只说明版本一致；轨迹是否满足算法假设，还取决于采样分布、行为对数概率和是否被重复消费。给定最大版本滞后 $L_{max}$ 与最大允许行为偏离 $D_{max}$，准入至少需要满足：

$$
0\leq\Delta v\leq L_{max} \quad\land\quad D_{beh}\leq D_{max}.
$$

允许有限异步时，可以在算法明确支持且行为偏离受控的条件下接纳小于阈值的旧轨迹。未来版本、缺失版本、超过版本阈值、行为偏离过大或已经被预留/消费的轨迹必须隔离。版本差因此只是首道门槛，行为证据是第二道门槛；二者都不应被单独写成普适的 on-policy 证明。

| 状态 | 条件 | 处理 |
| --- | --- | --- |
| fresh | $\Delta v=0$ | 继续检查采样契约、行为对数概率和消费状态 |
| tolerated | $0<\Delta v\leq L_{max}$ | 继续检查动作位置的行为偏离，满足阈值后才能接纳 |
| stale | $\Delta v>L_{max}$ | 拒绝或转入离线缓冲区 |
| future | $\Delta v<0$ | 版本登记错误，立即拒绝 |
## Step 3：哪些产物与工作进程确认完成后才能切换策略

学习器产出新权重后，先进入准备阶段：保存不可变产物并记录哈希、父版本和回滚指针，再经过分发、加载与工作进程确认。工作进程上报的版本号和产物哈希必须一致。只有协调器在所有必需工作进程准备完成后原子切换活动策略指针，版本才进入提交阶段；只写出检查点或只有部分工作进程加载成功，都不能声明新版本已经激活。

激活失败时，协调器保持旧策略为活动版本，隔离准备阶段误生成的新轨迹，并保留重试或回滚状态。奖励服务超时、更新失败、产物未提交和权重分发失败属于不同阶段，必须分别记录，不能用一个笼统的失败标记覆盖。

![策略更新与工作进程同步门槛](../public/02_PyTorch_Algorithms/pt_rollout_version_gate.svg)
<a id="mechanism-practice"></a>

## Step 4：实现轨迹准入、原子预留与策略激活

题目用一个带修订号的内存账本复现生命周期状态、策略兼容性、竞争预留和策略激活。输入检查、失败报告与字典装配由骨架提供；学习者只补状态迁移、行为偏离、compare-and-set 预留和激活门槛。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：推进生命周期<br>`transition_lifecycle` | 根据事件执行合法状态迁移 | 禁止跳过阶段、重用终态；更新中止只能从已预留回到已奖励 | 正常路径、释放预留、非法跳转与终态重用 |
| TODO 2：判断策略兼容性<br>`assess_policy_compatibility` | 联合版本滞后与有效动作位置的平均绝对对数概率差 | 未来/过期版本、空动作范围或行为偏离超限均拒绝 | 当前版本、容忍旧版本、掩码排除、过期与偏离超限 |
| TODO 3：竞争预留轨迹<br>`compare_and_reserve` | 按预期修订号把已奖励轨迹绑定到一次更新 | 只有一个竞争更新成功；同一更新重试幂等；冲突不覆盖账本 | 双更新竞争、修订冲突、幂等重试与状态保持 |
| TODO 4：激活策略产物<br>`activate_policy_version` | 联合产物提交状态与必需工作进程确认形成激活门槛 | 任一确认缺失或版本—哈希不一致时保持原活动策略 | 成功切换、未提交、缺失确认、错误哈希与旧指针保持 |

```python
from typing import Any, Dict, List

```


```python
LIFECYCLE_TRANSITIONS = {
    ('generated', 'reward_ready'): 'rewarded',
    ('rewarded', 'reserve'): 'reserved',
    ('reserved', 'commit'): 'consumed',
    ('reserved', 'abort'): 'rewarded',
}


def transition_lifecycle(state, event):
    """执行一次合法生命周期迁移；非法跳转必须拒绝。"""
    # TODO 1：只接受 LIFECYCLE_TRANSITIONS 明确列出的状态—事件组合。
    # next_state = ???  # 合法组合对应的下一状态；非法组合抛出 ValueError。
    raise NotImplementedError("TODO 1：推进生命周期")
    return next_state


def assess_policy_compatibility(old_log_probs, current_log_probs, action_mask, behavior_version, current_version, max_lag, max_mean_abs_log_ratio):
    """联合版本滞后与有效动作位置的行为偏离判断策略兼容性。"""
    if max_lag < 0 or max_mean_abs_log_ratio < 0:
        raise ValueError('兼容性阈值必须非负')
    if not (len(old_log_probs) == len(current_log_probs) == len(action_mask)):
        raise ValueError('行为概率、当前概率与动作掩码必须同形')
    active_indices = [index for index, active in enumerate(action_mask) if active]
    if not active_indices:
        return {'accepted': False, 'reason': 'empty_action_mask', 'freshness': None, 'mean_abs_log_ratio': None}
    # TODO 2：计算版本状态和动作位置平均绝对对数概率差。
    # lag = ???  # 当前策略版本减行为策略版本。
    # freshness = ???  # future、fresh、tolerated 或 stale。
    # mean_abs_log_ratio = ???  # 只平均 active_indices 指向的位置。
    raise NotImplementedError("TODO 2：判断策略兼容性")
    if freshness in {'future', 'stale'}:
        return {'accepted': False, 'reason': f'{freshness}_policy', 'freshness': freshness, 'mean_abs_log_ratio': mean_abs_log_ratio}
    if mean_abs_log_ratio > max_mean_abs_log_ratio:
        return {'accepted': False, 'reason': 'policy_drift', 'freshness': freshness, 'mean_abs_log_ratio': mean_abs_log_ratio}
    return {'accepted': True, 'reason': 'ok', 'freshness': freshness, 'mean_abs_log_ratio': mean_abs_log_ratio}


def compare_and_reserve(ledger, rollout_id, expected_revision, update_id):
    """按预期修订号竞争预留一条轨迹，并原子替换账本记录。"""
    if not update_id:
        raise ValueError('update_id 不能为空')
    current = ledger.get(rollout_id)
    if current is None:
        return {'reserved': False, 'reason': 'missing_rollout', 'record': None}
    if current.get('lifecycle_state') == 'reserved' and current.get('reserved_by') == update_id:
        return {'reserved': True, 'reason': 'idempotent_retry', 'record': dict(current)}
    # TODO 3：核对修订号与可预留状态，并构造下一版记录。
    # revision_matches = ???  # 当前修订号必须等于调用方看到的 expected_revision。
    # reservable = ???  # 只有 rewarded 状态可以首次预留。
    # reserved_record = ???  # 状态变为 reserved，绑定 update_id，修订号加一。
    raise NotImplementedError("TODO 3：竞争预留轨迹")
    if not revision_matches:
        return {'reserved': False, 'reason': 'revision_conflict', 'record': dict(current)}
    if not reservable:
        return {'reserved': False, 'reason': 'not_rewarded', 'record': dict(current)}
    ledger[rollout_id] = reserved_record
    return {'reserved': True, 'reason': 'reserved', 'record': dict(reserved_record)}


def activate_policy_version(active_pointer, artifact, worker_states, required_workers):
    """通过发布门槛时切换活动指针，失败时保持原策略。"""
    artifact_version = artifact.get('version')
    artifact_hash = artifact.get('artifact_hash')
    # TODO 4：检查必需工作进程确认，并形成完整激活门槛。
    # missing_workers = ???  # 尚未报告加载结果的必需工作进程。
    # mismatched_workers = ???  # 已报告但版本或产物哈希不一致的工作进程。
    # activated = ???  # 产物已提交，且不存在缺失或错误确认。
    raise NotImplementedError("TODO 4：激活策略版本")
    if not artifact.get('committed', False):
        reason = 'artifact_not_committed'
    elif missing_workers:
        reason = 'missing_worker_ack'
    elif mismatched_workers:
        reason = 'worker_mismatch'
    else:
        reason = 'activated'
    next_pointer = {'version': artifact_version, 'artifact_hash': artifact_hash} if activated else dict(active_pointer)
    return {
        'activated': activated, 'reason': reason, 'active_pointer': next_pointer,
        'missing_workers': missing_workers, 'mismatched_workers': mismatched_workers,
    }
```

## 测试

测试覆盖合法与非法状态迁移、版本与行为偏离两项门槛、竞争预留和幂等重试，以及策略发布失败时的旧指针保持。

```python
def test_lifecycle_transitions():
    """正常推进、更新中止和非法跳转必须产生不同结果。"""
    assert transition_lifecycle('generated', 'reward_ready') == 'rewarded'
    assert transition_lifecycle('rewarded', 'reserve') == 'reserved'
    assert transition_lifecycle('reserved', 'commit') == 'consumed'
    assert transition_lifecycle('reserved', 'abort') == 'rewarded'
    for state, event in [('generated', 'reserve'), ('rewarded', 'commit'), ('consumed', 'reserve')]:
        try:
            transition_lifecycle(state, event)
        except ValueError:
            continue
        raise AssertionError(f'非法迁移必须被拒绝：{state} + {event}')


def test_policy_compatibility():
    """动作掩码、版本滞后和行为偏离必须同时参与判断。"""
    old = [-0.2, -0.2, -9.0]
    current = [-0.21, -0.19, 9.0]
    mask = [True, True, False]
    fresh = assess_policy_compatibility(old, current, mask, 3, 3, 1, 0.05)
    assert fresh['accepted'] is True and fresh['freshness'] == 'fresh'
    assert abs(fresh['mean_abs_log_ratio'] - 0.01) < 1e-8
    tolerated = assess_policy_compatibility(old, current, mask, 2, 3, 1, 0.05)
    assert tolerated['accepted'] is True and tolerated['freshness'] == 'tolerated'
    assert assess_policy_compatibility(old, current, mask, 1, 3, 1, 0.05)['reason'] == 'stale_policy'
    assert assess_policy_compatibility(old, current, mask, 4, 3, 1, 0.05)['reason'] == 'future_policy'
    drifted = assess_policy_compatibility(old, [-0.8, -0.7, 9.0], mask, 3, 3, 1, 0.05)
    assert drifted['reason'] == 'policy_drift'
    empty = assess_policy_compatibility(old, current, [False, False, False], 3, 3, 1, 0.05)
    assert empty['reason'] == 'empty_action_mask'


def test_compare_and_reserve():
    """共享旧修订号的竞争更新只能有一个成功，同一更新重试保持幂等。"""
    ledger = {'r-1': {'lifecycle_state': 'rewarded', 'revision': 7, 'reserved_by': None}}
    first = compare_and_reserve(ledger, 'r-1', 7, 'update-a')
    assert first['reserved'] is True and ledger['r-1']['revision'] == 8
    competing = compare_and_reserve(ledger, 'r-1', 7, 'update-b')
    assert competing['reserved'] is False and competing['reason'] == 'revision_conflict'
    assert ledger['r-1']['reserved_by'] == 'update-a'
    retry = compare_and_reserve(ledger, 'r-1', 7, 'update-a')
    assert retry['reserved'] is True and retry['reason'] == 'idempotent_retry'
    ledger['done'] = {'lifecycle_state': 'consumed', 'revision': 2, 'reserved_by': None}
    assert compare_and_reserve(ledger, 'done', 2, 'update-c')['reason'] == 'not_rewarded'


def test_version_activation():
    """任一发布门槛失败都必须保持原活动策略指针。"""
    active = {'version': 3, 'artifact_hash': 'policy-3'}
    artifact = {'version': 4, 'artifact_hash': 'policy-4', 'committed': True}
    workers = {
        'rollout-0': {'version': 4, 'artifact_hash': 'policy-4'},
        'rollout-1': {'version': 4, 'artifact_hash': 'policy-4'},
    }
    required = ['rollout-0', 'rollout-1']
    ready = activate_policy_version(active, artifact, workers, required)
    assert ready['activated'] is True and ready['active_pointer']['version'] == 4
    missing = activate_policy_version(active, artifact, {'rollout-0': workers['rollout-0']}, required)
    assert missing['reason'] == 'missing_worker_ack' and missing['active_pointer'] == active
    wrong = workers | {'rollout-1': {'version': 4, 'artifact_hash': 'other'}}
    mismatch = activate_policy_version(active, artifact, wrong, required)
    assert mismatch['reason'] == 'worker_mismatch' and mismatch['active_pointer'] == active
    uncommitted = activate_policy_version(active, artifact | {'committed': False}, workers, required)
    assert uncommitted['reason'] == 'artifact_not_committed' and uncommitted['active_pointer'] == active
    assert active == {'version': 3, 'artifact_hash': 'policy-3'}


test_lifecycle_transitions(); test_policy_compatibility(); test_compare_and_reserve(); test_version_activation()
print('✅ 轨迹状态迁移、策略兼容性、竞争预留与版本激活测试通过')
```

---

🛑 **STOP HERE** 🛑

> 请先完成题目区并运行测试，再查看参考实现。

---
## 参考代码与解析

```python
LIFECYCLE_TRANSITIONS = {
    ('generated', 'reward_ready'): 'rewarded',
    ('rewarded', 'reserve'): 'reserved',
    ('reserved', 'commit'): 'consumed',
    ('reserved', 'abort'): 'rewarded',
}


def transition_lifecycle(state, event):
    transition = (state, event)
    if transition not in LIFECYCLE_TRANSITIONS:
        raise ValueError(f'非法生命周期迁移：{state} + {event}')
    return LIFECYCLE_TRANSITIONS[transition]

def assess_policy_compatibility(old_log_probs, current_log_probs, action_mask, behavior_version, current_version, max_lag, max_mean_abs_log_ratio):
    if max_lag < 0 or max_mean_abs_log_ratio < 0:
        raise ValueError('兼容性阈值必须非负')
    if not (len(old_log_probs) == len(current_log_probs) == len(action_mask)):
        raise ValueError('行为概率、当前概率与动作掩码必须同形')
    active_indices = [index for index, active in enumerate(action_mask) if active]
    if not active_indices:
        return {'accepted': False, 'reason': 'empty_action_mask', 'freshness': None, 'mean_abs_log_ratio': None}
    lag = current_version - behavior_version
    freshness = 'future' if lag < 0 else ('fresh' if lag == 0 else ('tolerated' if lag <= max_lag else 'stale'))
    mean_abs_log_ratio = sum(abs(float(current_log_probs[i]) - float(old_log_probs[i])) for i in active_indices) / len(active_indices)
    if freshness in {'future', 'stale'}:
        return {'accepted': False, 'reason': f'{freshness}_policy', 'freshness': freshness, 'mean_abs_log_ratio': mean_abs_log_ratio}
    if mean_abs_log_ratio > max_mean_abs_log_ratio:
        return {'accepted': False, 'reason': 'policy_drift', 'freshness': freshness, 'mean_abs_log_ratio': mean_abs_log_ratio}
    return {'accepted': True, 'reason': 'ok', 'freshness': freshness, 'mean_abs_log_ratio': mean_abs_log_ratio}

def compare_and_reserve(ledger, rollout_id, expected_revision, update_id):
    if not update_id:
        raise ValueError('update_id 不能为空')
    current = ledger.get(rollout_id)
    if current is None:
        return {'reserved': False, 'reason': 'missing_rollout', 'record': None}
    if current.get('lifecycle_state') == 'reserved' and current.get('reserved_by') == update_id:
        return {'reserved': True, 'reason': 'idempotent_retry', 'record': dict(current)}
    revision_matches = current.get('revision') == expected_revision
    reservable = current.get('lifecycle_state') == 'rewarded'
    reserved_record = current | {
        'lifecycle_state': transition_lifecycle(current.get('lifecycle_state'), 'reserve'),
        'reserved_by': update_id, 'revision': current.get('revision', 0) + 1,
    } if revision_matches and reservable else dict(current)
    if not revision_matches:
        return {'reserved': False, 'reason': 'revision_conflict', 'record': dict(current)}
    if not reservable:
        return {'reserved': False, 'reason': 'not_rewarded', 'record': dict(current)}
    ledger[rollout_id] = reserved_record
    return {'reserved': True, 'reason': 'reserved', 'record': dict(reserved_record)}

def activate_policy_version(active_pointer, artifact, worker_states, required_workers):
    artifact_version = artifact['version']
    artifact_hash = artifact['artifact_hash']
    missing_workers = sorted(worker for worker in required_workers if worker not in worker_states)
    mismatched_workers = sorted(
        worker for worker in required_workers if worker in worker_states
        and (worker_states[worker].get('version') != artifact_version
             or worker_states[worker].get('artifact_hash') != artifact_hash)
    )
    activated = bool(artifact.get('committed', False)) and not missing_workers and not mismatched_workers
    if not artifact.get('committed', False):
        reason = 'artifact_not_committed'
    elif missing_workers:
        reason = 'missing_worker_ack'
    elif mismatched_workers:
        reason = 'worker_mismatch'
    else:
        reason = 'activated'
    next_pointer = {'version': artifact_version, 'artifact_hash': artifact_hash} if activated else dict(active_pointer)
    return {
        'activated': activated, 'reason': reason, 'active_pointer': next_pointer,
        'missing_workers': missing_workers, 'mismatched_workers': mismatched_workers,
    }
```

### 解析

#### TODO 1：推进生命周期

- **机制：** 状态—事件对只沿显式状态机推进，更新中止把已预留轨迹释放回已奖励状态。
- **不变量：** 不允许跳过奖励或预留阶段，已经消费的轨迹不能再次进入更新。
- **测试证据：** 测试覆盖正常路径、释放预留、非法跳转和终态重用。

#### TODO 2：判断策略兼容性

- **机制：** 先按策略版本差识别未来、当前、可容忍滞后和过期轨迹，再只在有效动作位置计算平均绝对对数概率差。
- **不变量：** 版本差合格不代表行为仍兼容；任一门槛超限都必须拒绝轨迹。
- **测试证据：** 测试覆盖新鲜、可容忍滞后、未来、过期、掩码排除、空动作范围和行为偏离。

#### TODO 3：竞争预留轨迹

- **机制：** 调用方携带读取时看到的修订号，只有修订号和已奖励状态同时匹配时才能写入预留。
- **不变量：** 两个更新基于同一旧修订号竞争时只能有一个成功；同一更新重试不重复递增修订号。
- **测试证据：** 测试核对首次预留、竞争冲突、幂等重试和终态拒绝。

#### TODO 4：激活策略产物

- **机制：** 新策略产物先完成提交，再由所有必需工作进程确认相同版本与不可变哈希，最后原子切换活动指针。
- **不变量：** 未提交产物、缺失确认或版本—哈希不一致时都保持原活动策略。
- **测试证据：** 测试覆盖成功激活、未提交产物、缺失确认、错误哈希和旧指针保持。

## 相关阅读

- [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)
- [Part 02 · 51 Online DPO](./51_Online_DPO.md)
- [TRL 文档](https://huggingface.co/docs/trl/)
- [verl](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
