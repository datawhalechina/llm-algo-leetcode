# 51. Online DPO | 在线 DPO
**难度：** Hard | **环境：** [CPU 机制练习](#runtime-environment) | **标签：** `后训练对齐`, `DPO`, `在线优化` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/51_Online_DPO.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线 DPO 不是收到新反馈就立即更新。每轮更新前，需要确认偏好对由哪个策略版本产生、是否已被消费、是否污染独立评测，以及生成策略与当前策略的版本差是否在允许范围内；更新后，还要与固定基线比较质量和安全变化。

通过检查的偏好对按偏好对 ID 排序，组成容量固定的反馈窗口。一个窗口只触发一轮固定样本清单的 DPO 更新；候选随后接受独立评测，并根据非平局胜率、安全指标和已通过评测的检查点作出采用、延迟、回滚或停止决策。

**关键词：** `在线反馈`, `策略版本差`, `评测隔离`, `检查点回滚`

---

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。

---
## 前置阅读

**导语：** 先建立离线偏好目标、评测协议和策略版本语义，再观察持续反馈如何改变更新准入。

- [Part 02 · 50 偏好数据与对齐评测](./50_Preference_Data_and_Evaluation.md)
- [Part 02 · 17 DPO 损失](./17_DPO_Loss_Tutorial.md)
- [Part 02 · PT-ROLLOUT-LIFECYCLE Rollout 与策略新鲜度](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)

---

## Step 1：一条在线偏好如何进入本轮更新

在线偏好优化把一次离线训练扩展为持续更新。新反馈先形成同一任务下的优选 / 非优选回答对；来自工具执行的轨迹，需先根据执行结束时的成功状态和验证结果构造成对偏好。通过身份、版本差、风险与切分检查的样本进入固定窗口，完成一轮固定样本清单的更新后再接受独立评测。

| 阶段 | 核心关系 | 作用 |
|:---|:---|:---|
| 成对构造 | 同一任务下的优选与非优选回答具有明确选择依据 | 把反馈转换为 DPO 可使用的偏好对 |
| 反馈准入 | 样本身份、产生策略、来源与数据切分相互对应 | 判断偏好对能否进入当前窗口 |
| 固定窗口更新 | 一轮更新只消费排序和容量均已固定的样本清单 | 将数据变化限制在可追溯的样本集合内 |
| 独立评测 | 更新候选与固定基线使用相同评测集、指标和阈值 | 防止训练反馈替代最终评测 |
| 采用与恢复 | 采用时将候选登记为下一轮基线；退化时加载最近一个通过评测的检查点 | 明确下一轮使用哪个版本 |

![在线反馈经过准入、固定窗口更新和独立评测进入采用或回滚](../public/02_PyTorch_Algorithms/51_online_dpo_lifecycle.svg)

## Step 2：哪些偏好对能进入固定反馈窗口

每条偏好对记录生成策略版本和当前目标策略版本，并计算二者的版本差，再检查跨轮消费、当前流内重复、偏好对完整性、风险状态和评测切分。固定最大样本数、排序、策略版本和样本身份后，用这些字段生成窗口 ID，使一次 DPO 更新对应一份可重放的样本清单。

设当前目标策略版本为 $v_{current}$，偏好对的生成策略版本为 $v_{generated}$，允许的最大版本差为 $L_{max}$。版本差与新鲜度准入条件为：

$$
\Delta v=v_{current}-v_{generated}, \qquad 0\leq \Delta v\leq L_{max}.
$$

通过全部准入检查的偏好对按 ID 稳定排序，取前 $M$ 条形成第 $t$ 轮的有序身份列表 $I_t$。窗口身份由当前策略版本与 $I_t$ 共同确定：

$$
I_t=\operatorname{Sort}(\{id_i\mid pair_i\text{ 通过准入}\})[:M], \qquad
w_t=H(v_t,I_t).
$$

其中 $M$ 是窗口容量，$v_t$ 是第 $t$ 轮目标策略版本，$w_t$ 是窗口 ID，$H$ 是稳定摘要函数。相同的策略版本与入选 ID 顺序必须产生相同窗口 ID；具体序列化和摘要算法属于 Step 4 的实现契约。容量外样本只延后，不得悄悄改变本轮窗口。

| 判断维度 | 要回答的问题 | 风险响应 |
|:---|:---|:---|
| 身份完整性 | 同一反馈能否唯一识别，并与历史已消费集合区分 | 区分跨轮消费与当前流内重复 |
| 策略版本差 | 当前目标策略版本减去生成策略版本是否位于允许范围 | 接纳、拒绝或重新采样 |
| 风险检查 | 样本是否命中需要人工复核的风险规则 | 阻止自动进入更新 |
| 切分隔离 | 反馈是否污染独立评测 | 移出更新池并恢复隔离 |
| 偏好对完整性 | chosen 与 rejected 是否来自同一任务且内容不同 | 补证或拒绝 |
| 窗口样本 | 本轮消费、因容量延后的偏好对 ID 分别有哪些 | 固定顺序、容量、策略版本和窗口 ID |

![策略版本差与身份、风险、切分检查共同决定反馈准入](../public/02_PyTorch_Algorithms/51_policy_freshness_gate.svg)

## Step 3：何时采用候选，何时延迟、回滚或停止

在线更新完成后，候选必须与固定基线在相同评测条件下比较。分别观察非平局胜率、重复评测通过率、安全指标、预先登记的一票否决测试失败数和更新成本；工具任务还要检查执行成功率与验证结果。安全指标降幅超限、重复评测通过率降幅超限或一票否决测试失败时，加载最近一个通过评测的检查点；成本增幅超限时保留当前基线并补充证据。

令 $W$、$R$、$S$ 和 $C$ 分别表示非平局胜率、重复评测通过率、安全指标和更新成本。候选相对固定基线的变化量用于区分质量收益、硬退化与成本代价：

$$
\Delta W=W_{candidate}-W_{baseline}, \qquad
\Delta R=R_{candidate}-R_{baseline}, \qquad
\Delta S=S_{candidate}-S_{baseline},
$$

$$
\rho_C=\frac{C_{candidate}-C_{baseline}}{C_{baseline}}.
$$

安全或重复性越过硬门槛时回滚；没有可用检查点时停止自动更新。硬门槛通过但 $\Delta W$ 或 $\rho_C$ 未达到采用条件时延迟；质量与成本条件同时通过时，候选才成为下一轮基线。

| 证据维度 | 解释 | 决策作用 |
|:---|:---|:---|
| 非平局胜率变化 | 候选相对基线的非平局胜率提高多少 | 达到最低增幅后才可能采用 |
| 重复评测通过率 | 同一候选在重复评测中通过的比例是否下降 | 降幅超限时停止或回滚 |
| 安全与一票否决测试 | 安全指标是否下降，预先登记的安全或关键能力测试是否失败 | 任一条件越界时停止或回滚 |
| 更新成本 | 候选成本相对基线增加多少 | 超过门槛时延迟采用 |
| 工具执行结果（可选） | 工具执行成功率和验证结果是否下降 | 防止文本偏好掩盖执行失败 |

<a id="mechanism-practice"></a>

## Step 4：实现偏好对准入、固定窗口与回滚决策

代码先判断单个偏好对的准入原因，再形成样本清单与容量固定的更新窗口，最后根据非平局胜率、重复评测通过率、安全指标、一票否决测试、成本和检查点状态给出 `adopt / defer / rollback / halt`。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|:---|:---|:---|:---|
| TODO 1：判断偏好对准入<br>`classify_preference_pair` | 检查成对语义、切分、风险和策略版本差 | 每个样本只返回一个首要拒绝原因 | 完整、同答、污染、风险、当前版本与过期版本用例 |
| TODO 2：形成更新窗口<br>`build_preference_window` | 区分跨轮消费、流内重复、入选样本和容量延后样本 | 偏好对 ID 唯一，排序和窗口 ID 稳定 | 窗口身份、样本清单、延后清单、拒绝计数与入选来源集合 |
| TODO 3：选择采用或回滚<br>`recommend_online_dpo` | 联合胜率增幅、重复评测通过率、安全指标、一票否决测试、成本和检查点状态形成决策 | 任一拒绝条件越界时优先回滚或停止；胜率与成本均过线才能采用 | `adopt / defer / rollback / halt` |

```python
import hashlib

```


```python
def classify_preference_pair(
    pair: dict[str, object], current_policy_version: int, max_policy_lag: int
) -> str:
    """返回 eligible 或一个首要拒绝原因。"""
    # ==========================================
    # TODO 1：判断单个偏好对能否进入在线更新。
    # pair_id = ???  # 偏好对的稳定身份。
    # policy_lag = ???  # 当前策略与数据产生策略的版本差。
    # ==========================================
    raise NotImplementedError("TODO 1：判断偏好对准入")


def build_preference_window(
    feedback_stream: list[dict[str, object]],
    current_policy_version: int,
    max_policy_lag: int,
    max_pairs: int,
    previously_consumed_ids: set[str] | None = None,
) -> dict[str, object]:
    """去重、准入并形成顺序稳定的偏好窗口。"""
    # ==========================================
    # TODO 2：形成边界固定的在线偏好窗口。
    # consumed_ids = ???  # 历史窗口已经消费的偏好对。
    # selected_pairs, overflow_pairs = ???  # 本轮入选与因容量延后的偏好对。
    # ==========================================
    raise NotImplementedError("TODO 2：形成在线偏好窗口")


def recommend_online_dpo(
    baseline_run: dict[str, float],
    candidate_run: dict[str, float],
    min_win_rate_gain: float,
    rollback_ready: bool,
    max_repeat_pass_rate_drop: float = 0.05,
    max_safety_drop: float = 0.02,
    max_cost_increase_ratio: float = 0.5,
) -> dict[str, object]:
    """根据评测变化与检查点状态返回采用、延迟、回滚或停止。"""
    # ==========================================
    # TODO 3：根据独立评测和检查点状态决定是否采用候选。
    # comparison = ???  # 分别保留胜率、重复评测通过率、安全、一票否决测试与成本变化。
    # decision = ???  # adopt、defer、rollback 或 halt。
    # ==========================================
    raise NotImplementedError("TODO 3：形成在线更新与回滚决策")
```

## 测试

运行下方 CPU 机制测试，分别检查反馈准入、证据对照和采用或回滚决策。


```python
def _pair(pair_id, version=10, chosen='good', rejected='bad', split='train', risk=False, source='prod'):
    return {"pair_id": pair_id, "prompt_id": f"prompt-{pair_id}", "chosen": chosen,
            "rejected": rejected, "policy_version": version, "split": split,
            "risk": risk, "source": source}


def test_pair_admission_reasons():
    """成对语义、切分、风险和版本差必须分别产生明确原因。"""
    assert classify_preference_pair(_pair('ok'), 10, 2) == 'eligible'
    assert classify_preference_pair(_pair('same', chosen='x', rejected='x'), 10, 2) == 'invalid_pair'
    assert classify_preference_pair(_pair('eval', split='evaluation'), 10, 2) == 'evaluation_leakage'
    assert classify_preference_pair(_pair('risk', risk=True), 10, 2) == 'risk'
    assert classify_preference_pair(_pair('stale', version=7), 10, 2) == 'stale'
    assert classify_preference_pair(_pair('future', version=11), 10, 2) == 'future'


def test_preference_window_identity_and_capacity():
    """窗口必须区分跨轮消费和流内重复，并生成稳定身份。"""
    stream = [
        _pair('b', source='shadow'), _pair('a'), _pair('c'), _pair('b', source='shadow'),
        _pair('used'),
        _pair('stale', version=6), _pair('unsafe', risk=True),
    ]
    report = build_preference_window(stream, 10, 2, max_pairs=2, previously_consumed_ids={'used'})
    replay = build_preference_window(list(reversed(stream)), 10, 2, max_pairs=2, previously_consumed_ids={'used'})
    assert report["pair_ids"] == ['a', 'b']
    assert report["overflow_pair_ids"] == ['c']
    assert report["policy_version"] == 10
    assert report["window_id"] == replay["window_id"]
    assert report["rejection_counts"] == {'consumed': 1, 'duplicate': 1, 'risk': 1, 'stale': 1}
    assert report["eligible_sources"] == ['prod', 'shadow']


def test_adopt_defer_rollback_and_halt():
    """安全退化需要恢复路径，软门槛不足只延迟采用。"""
    baseline = {"win_rate": 0.52, "repeat_pass_rate": 0.82, "safety_score": 0.95, "update_cost": 1.0}
    adopt = {"win_rate": 0.59, "repeat_pass_rate": 0.79, "safety_score": 0.94, "update_cost": 1.2}
    unsafe = {**adopt, "safety_score": 0.90}
    low_repeatability = {**adopt, "repeat_pass_rate": 0.74}
    blocking_failure = {**adopt, "blocking_failures": 1}
    expensive = {**adopt, "update_cost": 1.8}
    assert recommend_online_dpo(baseline, adopt, 0.05, rollback_ready=True)["decision"] == "adopt"
    assert recommend_online_dpo(baseline, {**adopt, "win_rate": 0.54}, 0.05, rollback_ready=True)["decision"] == "defer"
    assert recommend_online_dpo(baseline, expensive, 0.05, rollback_ready=True)["reason"] == "cost_gate_not_met"
    assert recommend_online_dpo(baseline, low_repeatability, 0.05, rollback_ready=True)["reason"] == "repeat_pass_rate_regression"
    assert recommend_online_dpo(baseline, blocking_failure, 0.05, rollback_ready=True)["reason"] == "blocking_evaluation_failure"
    assert recommend_online_dpo(baseline, unsafe, 0.05, rollback_ready=True)["decision"] == "rollback"
    assert recommend_online_dpo(baseline, unsafe, 0.05, rollback_ready=False)["decision"] == "halt"


try:
    test_pair_admission_reasons()
    test_preference_window_identity_and_capacity()
    test_adopt_defer_rollback_and_halt()
    print("✅ 在线反馈窗口、更新对照与回滚准备测试通过。")
except (AttributeError, NameError, TypeError, ValueError, AssertionError) as error:
    raise NotImplementedError("请先完成 TODO 代码！") from error
```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
import hashlib


def classify_preference_pair(
    pair: dict[str, object], current_policy_version: int, max_policy_lag: int
) -> str:
    """返回 eligible 或一个首要拒绝原因。"""
    # TODO 1：判断偏好对准入。
    pair_id = str(pair.get('pair_id', ''))
    if not pair_id or not pair.get('prompt_id') or not pair.get('chosen') or not pair.get('rejected'):
        return 'missing_identity'
    if pair['chosen'] == pair['rejected']:
        return 'invalid_pair'
    if pair.get('split') != 'train':
        return 'evaluation_leakage'
    if bool(pair.get('risk', False)):
        return 'risk'
    policy_lag = current_policy_version - int(pair.get('policy_version', -1))
    if policy_lag < 0:
        return 'future'
    if policy_lag > max_policy_lag:
        return 'stale'
    return 'eligible'


def build_preference_window(
    feedback_stream: list[dict[str, object]],
    current_policy_version: int,
    max_policy_lag: int,
    max_pairs: int,
    previously_consumed_ids: set[str] | None = None,
) -> dict[str, object]:
    """去重、准入并形成顺序稳定的偏好窗口。"""
    # TODO 2：形成在线偏好窗口。
    if max_pairs <= 0:
        raise ValueError('max_pairs 必须为正')
    consumed_ids = set(previously_consumed_ids or set())
    seen_ids, accepted_pairs = set(), []
    rejection_counts = {}
    for pair in feedback_stream:
        pair_id = str(pair.get('pair_id', ''))
        if pair_id in consumed_ids:
            reason = 'consumed'
        elif pair_id in seen_ids:
            reason = 'duplicate'
        else:
            reason = classify_preference_pair(pair, current_policy_version, max_policy_lag)
        seen_ids.add(pair_id)
        if reason == 'eligible':
            accepted_pairs.append(pair)
        else:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    ordered_pairs = sorted(accepted_pairs, key=lambda item: str(item['pair_id']))
    selected_pairs, overflow_pairs = ordered_pairs[:max_pairs], ordered_pairs[max_pairs:]
    pair_ids = [str(item['pair_id']) for item in selected_pairs]
    identity_payload = f"policy={current_policy_version}|pairs={'|'.join(pair_ids)}"
    window_id = hashlib.sha256(identity_payload.encode('utf-8')).hexdigest()[:16]
    return {'window_id': window_id, 'policy_version': current_policy_version, 'pair_ids': pair_ids,
            'overflow_pair_ids': [str(item['pair_id']) for item in overflow_pairs],
            'eligible_sources': sorted({str(item.get('source', 'unknown')) for item in selected_pairs}),
            'rejection_counts': dict(sorted(rejection_counts.items()))}


def recommend_online_dpo(
    baseline_run: dict[str, float],
    candidate_run: dict[str, float],
    min_win_rate_gain: float,
    rollback_ready: bool,
    max_repeat_pass_rate_drop: float = 0.05,
    max_safety_drop: float = 0.02,
    max_cost_increase_ratio: float = 0.5,
) -> dict[str, object]:
    """根据更新收益、退化门槛与恢复准备形成决策。"""
    # TODO 3：选择采用、延迟或回滚。
    if baseline_run['update_cost'] <= 0:
        raise ValueError('baseline update_cost 必须为正')
    comparison = {
        'win_rate_gain': round(candidate_run['win_rate'] - baseline_run['win_rate'], 6),
        'repeat_pass_rate_delta': round(candidate_run['repeat_pass_rate'] - baseline_run['repeat_pass_rate'], 6),
        'safety_delta': round(candidate_run['safety_score'] - baseline_run['safety_score'], 6),
        'cost_increase_ratio': round((candidate_run['update_cost'] - baseline_run['update_cost']) / baseline_run['update_cost'], 6),
        'blocking_failure_count': int(candidate_run.get('blocking_failures', 0)),
    }
    if comparison["safety_delta"] < -max_safety_drop:
        decision = "rollback" if rollback_ready else "halt"
        reason = "safety_regression"
    elif comparison["blocking_failure_count"] > 0:
        decision = "rollback" if rollback_ready else "halt"
        reason = "blocking_evaluation_failure"
    elif comparison["repeat_pass_rate_delta"] < -max_repeat_pass_rate_drop:
        decision = "rollback" if rollback_ready else "halt"
        reason = "repeat_pass_rate_regression"
    elif comparison["cost_increase_ratio"] > max_cost_increase_ratio:
        decision = "defer"
        reason = "cost_gate_not_met"
    elif comparison["win_rate_gain"] >= min_win_rate_gain:
        decision = "adopt"
        reason = "quality_and_cost_pass"
    else:
        decision = "defer"
        reason = "soft_gate_not_met"
    return {"decision": decision, "reason": reason, "comparison": comparison, "rollback_ready": rollback_ready}
```

### 解析

#### TODO 1：判断偏好对准入

- 机制：先验证成对回答是否完整且不同，再依次检查训练切分、风险状态和策略版本。
- 不变量：每条数据只保留一个首要拒绝原因，使拒绝计数不会重复累计。

#### TODO 2：形成在线偏好窗口

- 机制：先排除历史窗口已经消费的偏好对 ID，再处理当前流内重复；其余样本稳定排序后分为本轮入选和容量延后两组，并由策略版本与入选 ID 生成窗口身份。
- 不变量：相同策略版本与入选偏好对 ID 始终形成相同窗口 ID；同一偏好对 ID 不得跨轮或在同一轮重复消费。

#### TODO 3：选择采用、延迟或回滚

- 机制：先检查安全指标降幅、一票否决测试失败数和重复评测通过率降幅，再检查成本增幅与非平局胜率增幅。
- 不变量：安全指标或重复评测通过率降幅超限、出现一票否决测试失败时，不能用非平局胜率增幅抵消；没有可用检查点时停止自动更新；成本超限或胜率增幅不足时保留当前基线并补充证据。
## 相关阅读

- [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](./PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)
- [Part 02 · 84 DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [Part 02 · 86 在线后训练生命周期与采用决策](./86_DPO_Online_Benchmark.md)
- [TRL: DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
