# 68. Speculative Decoding Benchmark | 投机解码机制选型基准

**难度：** Hard | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `推理优化`、`投机解码`、`机制选型` | **目标人群：** 理解投机验证、需要比较后端方案的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

同一套投机策略，可能在重复文本上减少生成时间，却在普通问答上得不偿失；有些方法还依赖特定模型结构或配对草稿权重。选型先要确认候选能运行，再回答它是否比同一目标模型单独生成更快、输出是否符合要求。

本项目固定目标模型、提示集、生成参数和硬件，把候选能力、真实执行、质量与性能分开检查。最终得到可采用的候选或明确的失败原因，供部署组合继续比较。

**关键词：** 候选家族、资产兼容、配对基准、接受统计、采用决策

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 使用 [BASE-CPU](../guide.md#base-cpu)，验证候选准入、配对与决策。
- **GPU 实验：** 使用[投机候选选型环境](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-spec025-quickstart)；历史复现与新候选所用 Profile 见 5.1。

## 前置阅读

先理解连续接受与拒绝修正，再熟悉如何固定请求和测量口径。

- [投机解码](./23_Speculative_Decoding.md)：接受概率、拒绝修正和追加词元。
- [推理性能对比](./66_Inference_Performance_Comparison.md)：同一请求集合下的延迟、吞吐与输出核对。

## Step 1：从候选来源到同目标配对比较

候选生成器（proposer）提出可能的后续词元，目标模型（target）负责验证，执行后端（backend）决定哪些实现可以运行。不同家族的差异首先是候选从哪里来：检索已有片段、使用模型内置预测层，或加载与目标模型配对的草稿权重。它们共用“提出 → 验证 → 提交”的外层流程，但验证协议仍要与具体候选结构匹配。

比较时记录目标和草稿权重的固定版本（snapshot）、候选长度与输入集合（workload）。例如对重复文本与普通问答分别配对，才能知道检索式候选的收益来自哪类输入；同一个开关名称不能代替这些身份与条件。G0 表示仅目标模型基线，G1–G5 依次对应下表五类候选。

| 家族 | 候选来源 | 需要的模型资产 | 主要检查 |
| --- | --- | --- | --- |
| N-gram | 提示或历史输出中的重复片段 | 不需额外草稿权重 | 重复输入能否覆盖检索与验证成本 |
| 原生多词元预测（MTP） | 目标模型内部的预测层 | 包含对应结构的权重文件 | 模型确有预测层且后端支持执行 |
| EAGLE | 目标模型隐藏表示与特征草稿器 | 与目标版本匹配的 EAGLE 权重 | 特征接口、词表与权重身份是否匹配 |
| DFlash | 块扩散草稿器 | 匹配的草稿权重 | 块候选能否通过后端验证 |
| DSpark | 并行主干与轻量顺序、置信度模块 | 匹配的草稿权重 | 块内依赖与候选长度控制是否启用 |

![投机候选经准入、配对比较与决策后进入部署组合](../public/02_PyTorch_Algorithms/68_speculative_benchmark_flow.svg)

## Step 2：能力与资产检查决定哪些候选能够启动

通用投机接口存在，不等于后端支持每个候选家族。还要核对模型结构或草稿资产、分词器兼容和显存条件。静态检查通过只允许尝试启动；启动后仍需用请求与原生统计确认候选路径没有静默回退。下图同时收集首词元延迟（TTFT）、每输出词元时间估计（TPOT）、吞吐和显存；基线与候选都要检查路径、输出和资源。

| 运行状态 | 判定依据 | 下一步 |
| --- | --- | --- |
| 可尝试启动（runnable） | 当前静态条件通过 | 启动服务并检查实际路径 |
| 未启用（skipped） | 当前计划关闭候选 | 保留关闭原因，不作性能比较 |
| 不支持（unsupported） | 当前后端或版本缺少所需方法 | 选择支持的实现后重新检查 |
| 缺资产（asset_missing） | 缺目标原生结构或匹配草稿权重 | 准备资产再检查 |
| 不兼容（incompatible） | 模型、词表或接口无法配对 | 更换匹配组合 |
| 容量受阻（capacity_blocked） | 已知容量估算超过设备预算 | 缩小配置或更换设备 |
| 执行失败（failed） | 启动或真实请求报错 | 检查日志，不填性能结果 |

![可运行候选顺序启动，再采集路径、输出、性能与资源证据](../public/02_PyTorch_Algorithms/68_speculative_gpu_flow.svg)

## Step 3：把运行结果转成采用、补测或拒绝

运行成功、证据可比较和候选值得采用，是三个不同判断。同目标、同请求、同采样配置的两条路径先检查输出与真实投机活动，再比较首词元延迟、端到端延迟（E2E）、输出吞吐和峰值显存。历史 TPOT 字段以首词元至请求结束的时长除以输出词元间隔数，包含流式收尾，不能当作相邻内容到达间隔。P95 是样本中第 95 百分位；接受率来自后端原生统计，只用于解释成本，不能单独证明加速。

确定性生成按请求核对输出；随机生成则另需分布与任务质量证据，不能要求不同执行路径在相同种子下逐词元一致。若候选只能配另一目标模型，要为该目标另建基线，只解释各组内部的比值，而不混排原始延迟。

| 证据与条件 | 决策 | 当前候选如何处理 |
| --- | --- | --- |
| 未运行、缺完整配对或固定条件不同 | 不评估（not_evaluated） | 保留原因，补齐配对后再判断 |
| 配对已运行，但关键指标或投机路径证据不完整 | 补测（tune） | 不晋升采用，补采缺少的证据 |
| 比较条件齐备，但输出质量不通过 | 拒绝（reject） | 保存失败证据，定位差异后另行复测 |
| 质量通过，但吞吐、尾延迟或容量门槛未全部满足 | 调整（tune） | 调整候选、预算或负载后复测 |
| 同场景的质量、路径、性能与容量均通过 | 采用（accept） | 仅把该场景的候选和证据交给部署比较 |

## Step 4：实现候选准入、配对与采用判断

题目提供候选家族表、模拟运行记录和采用门槛。你将完成家族分类、静态准入、配对证据与采用决策；测试用不支持、质量失败、指标缺失和门槛边界区分各条路径。这些记录用于 CPU 规则验证，不是真实后端测量。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
| --- | --- | --- | --- |
| TODO 1 · 分类候选家族 | 家族名与资产依赖表 | 识别别名并拒绝未知家族，不修改原表 | MTP 的内置资产与未知家族 |
| TODO 2 · 决定是否可启动 | 后端支持、资产、兼容与容量事实 | 按顺序返回首个阻断原因 | 单项失败、关闭候选与多项同时失败 |
| TODO 3 · 核对同目标配对 | 基线、候选、固定条件与指标 | 先区分未完成、不可比和缺证，再计算比值 | 目标不同、路径回退、缺指标与有效配对 |
| TODO 4 · 形成采用决策 | 配对证据与各项门槛 | 质量优先，吞吐取下限，延迟和容量取上限 | 质量拒绝、缺证补测、阈值等号与单项未过 |


```python
from collections import Counter
from typing import Any, Dict, Iterable, Mapping

```


```python
PROPOSER_FAMILIES = {
    'ngram': {'asset_kind': 'none', 'dependency': 'retrieval', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'mtp': {'asset_kind': 'target_native', 'dependency': 'model_native', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'eagle': {'asset_kind': 'paired_checkpoint', 'dependency': 'feature_autoregressive', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'dflash': {'asset_kind': 'paired_checkpoint', 'dependency': 'parallel_block', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'dspark': {'asset_kind': 'paired_checkpoint', 'dependency': 'semi_autoregressive', 'telemetry': ('proposed_tokens', 'accepted_tokens', 'verified_tokens')},
}


def classify_speculative_strategy(family: str) -> Dict[str, Any]:
    '''返回 proposer 家族的资产、依赖和遥测契约。'''
    # TODO 1：规范化 family，拒绝未知家族，并返回一份不可修改原表的副本。
    raise NotImplementedError('TODO 1：请分类 proposer 家族')


def admit_speculative_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    '''按 enabled、backend、资产、兼容和容量顺序形成准入状态。'''
    # TODO 2：状态只能来自 skipped / unsupported / asset_missing /
    # incompatible / capacity_blocked / runnable，并保留具体 reason。
    raise NotImplementedError('TODO 2：请形成候选准入状态')


def build_speculative_pair_evidence(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    '''检查同 target 配对，并计算端到端变化。'''
    # TODO 3：运行未完成、固定字段不一致、路径未生效和关键指标缺失要分开记录。
    raise NotImplementedError('TODO 3：请建立投机配对证据')


def decide_speculative_candidate(
    evidence: Mapping[str, Any], thresholds: Mapping[str, float]
) -> Dict[str, str]:
    '''根据可比性、质量、收益、尾延迟和容量形成采用决策。'''
    # TODO 4：未完成或不可比返回 not_evaluated；质量失败 reject；
    # 净收益或门槛部分未过为 tune；全部通过才 accept。
    raise NotImplementedError('TODO 4：请形成候选决策')

```

## 测试

分别检查候选类别、首个准入失败、固定条件匹配，以及质量和性能门槛的方向。

```python
def _candidate(**updates):
    base = {
        'enabled': True, 'family': 'ngram', 'backend_supported': True,
        'asset_ready': True, 'compatible': True,
        'estimated_memory_gb': 6.0, 'memory_limit_gb': 10.5,
    }
    return {**base, **updates}


def _run(role, **updates):
    base = {
        'run_status': 'completed', 'role': role,
        'target_snapshot': 'target-rev-a', 'workload_key': 'general-c1',
        'sampling_key': 't0-max128', 'execution_path': 'target_only' if role == 'baseline' else 'speculative',
        'throughput_tokens_per_s': 100.0, 'ttft_p95_ms': 50.0,
        'e2e_p95_ms': 900.0, 'peak_memory_mb': 6000.0,
        'quality_ok': True,
        'mechanism_active': True if role == 'candidate' else None,
    }
    return {**base, **updates}


def test_family_classification():
    """家族名称映射到正确资产种类，未知家族明确失败。"""
    assert classify_speculative_strategy('N-Gram')['asset_kind'] == 'none'
    assert classify_speculative_strategy('MTP')['asset_kind'] == 'target_native'
    assert classify_speculative_strategy('DSpark')['dependency'] == 'semi_autoregressive'
    try:
        classify_speculative_strategy('unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('未知 proposer 家族必须被拒绝')


def test_admission_states():
    """单项失败与多项失败都保留既定检查顺序。"""
    assert admit_speculative_candidate(_candidate())['run_status'] == 'runnable'
    assert admit_speculative_candidate(_candidate(backend_supported=False))['run_status'] == 'unsupported'
    assert admit_speculative_candidate(_candidate(asset_ready=False))['run_status'] == 'asset_missing'
    assert admit_speculative_candidate(_candidate(compatible=False))['run_status'] == 'incompatible'
    assert admit_speculative_candidate(_candidate(estimated_memory_gb=12.0))['run_status'] == 'capacity_blocked'
    assert admit_speculative_candidate(_candidate(enabled=False))['run_status'] == 'skipped'
    assert admit_speculative_candidate(_candidate(backend_supported=False, asset_ready=False))['run_status'] == 'unsupported'


def test_pair_evidence_and_decision():
    """有效配对的收益不得覆盖质量失败或缺少完整运行。"""
    baseline = _run('baseline')
    candidate = _run(
        'candidate', throughput_tokens_per_s=125.0, ttft_p95_ms=52.0,
        e2e_p95_ms=850.0, peak_memory_mb=7000.0,
    )
    evidence = build_speculative_pair_evidence(baseline, candidate)
    assert evidence['evidence_status'] == 'comparable'
    assert round(evidence['throughput_speedup'], 2) == 1.25
    thresholds = {
        'min_throughput_speedup': 1.10,
        'max_ttft_regression_ratio': 1.10,
        'max_e2e_p95_regression_ratio': 1.05,
        'max_peak_memory_mb': 10500.0,
    }
    assert decide_speculative_candidate(evidence, thresholds)['decision'] == 'accept'
    failed_quality = build_speculative_pair_evidence(baseline, {**candidate, 'quality_ok': False})
    assert decide_speculative_candidate(failed_quality, thresholds)['decision'] == 'reject'
    unsupported = build_speculative_pair_evidence(baseline, {**candidate, 'run_status': 'unsupported'})
    assert decide_speculative_candidate(unsupported, thresholds)['decision'] == 'not_evaluated'


def test_pair_boundaries_and_missing_evidence():
    """固定条件、实际路径与关键指标分别约束配对。"""
    baseline, candidate = _run('baseline'), _run('candidate')
    for key in ('target_snapshot', 'workload_key', 'sampling_key'):
        evidence = build_speculative_pair_evidence(baseline, {**candidate, key: 'different'})
        assert evidence['evidence_status'] == 'incomparable'
    for update in ({'execution_path': 'target_only'}, {'mechanism_active': False}, {'ttft_p95_ms': None}):
        evidence = build_speculative_pair_evidence(baseline, {**candidate, **update})
        assert evidence['evidence_status'] == 'incomplete'


def test_adoption_threshold_directions():
    """精确门槛可通过，任何单项越界均不能接受。"""
    thresholds = {'min_throughput_speedup': 1.1, 'max_ttft_regression_ratio': 1.1,
                  'max_e2e_p95_regression_ratio': 1.05, 'max_peak_memory_mb': 10500.0}
    evidence = {'evidence_status': 'comparable', 'quality_ok': True, 'mechanism_active': True,
                'throughput_speedup': 1.1, 'ttft_ratio': 1.1, 'e2e_p95_ratio': 1.05,
                'candidate_peak_memory_mb': 10500.0}
    assert decide_speculative_candidate(evidence, thresholds)['decision'] == 'accept'
    for key, value in (('throughput_speedup', 1.09), ('ttft_ratio', 1.11),
                       ('e2e_p95_ratio', 1.06), ('candidate_peak_memory_mb', 10501.0)):
        assert decide_speculative_candidate({**evidence, key: value}, thresholds)['decision'] == 'tune'
    assert decide_speculative_candidate({'evidence_status': 'incomplete'}, thresholds)['decision'] == 'tune'


def test_speculative_selection_template():
    """运行候选准入、比较和决策的 CPU 契约测试。"""
    test_family_classification()
    test_admission_states()
    test_pair_evidence_and_decision()
    test_pair_boundaries_and_missing_evidence()
    test_adoption_threshold_directions()
    print('测试通过：proposer 分类、准入状态、配对证据和采用决策均符合契约。')


try:
    test_speculative_selection_template()
except NotImplementedError:
    print('请先完成 TODO 部分的代码！')
    raise

```

---

🛑 **STOP HERE**

先完成题目并运行测试，再展开参考代码。

<br><br><br><br><br><br><br><br><br><br>
## 参考代码与解析

### 代码


```python
PROPOSER_FAMILIES = {
    'ngram': {'asset_kind': 'none', 'dependency': 'retrieval', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'mtp': {'asset_kind': 'target_native', 'dependency': 'model_native', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'eagle': {'asset_kind': 'paired_checkpoint', 'dependency': 'feature_autoregressive', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'dflash': {'asset_kind': 'paired_checkpoint', 'dependency': 'parallel_block', 'telemetry': ('proposed_tokens', 'accepted_tokens')},
    'dspark': {'asset_kind': 'paired_checkpoint', 'dependency': 'semi_autoregressive', 'telemetry': ('proposed_tokens', 'accepted_tokens', 'verified_tokens')},
}


def _normalize_family(family):
    return str(family).strip().lower().replace('-', '').replace('_', '')


def classify_speculative_strategy(family: str) -> Dict[str, Any]:
    aliases = {'ngram': 'ngram', 'mtp': 'mtp', 'eagle': 'eagle', 'eagle3': 'eagle',
               'dflash': 'dflash', 'dspark': 'dspark'}
    normalized = _normalize_family(family)
    if normalized not in aliases:
        raise ValueError(f'未知 speculative family：{family}')
    canonical = aliases[normalized]
    return {'family': canonical, **PROPOSER_FAMILIES[canonical]}


def admit_speculative_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    if not candidate.get('enabled', False):
        return {'run_status': 'skipped', 'reason': 'candidate_disabled'}
    if not candidate.get('backend_supported', False):
        return {'run_status': 'unsupported', 'reason': 'backend_method_unavailable'}
    if not candidate.get('asset_ready', False):
        return {'run_status': 'asset_missing', 'reason': 'required_model_asset_unavailable'}
    if not candidate.get('compatible', False):
        return {'run_status': 'incompatible', 'reason': 'target_proposer_contract_failed'}
    estimated = candidate.get('estimated_memory_gb')
    limit = candidate.get('memory_limit_gb')
    if estimated is not None and limit is not None and estimated > limit:
        return {'run_status': 'capacity_blocked', 'reason': 'estimated_memory_exceeds_limit'}
    return {'run_status': 'runnable', 'reason': 'static_admission_passed'}


def build_speculative_pair_evidence(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    completed = baseline.get('run_status') == candidate.get('run_status') == 'completed'
    if not completed:
        return {
            'evidence_status': 'not_evaluated',
            'missing_evidence': ['completed_pair'],
            'candidate_run_status': str(candidate.get('run_status', 'unknown')),
        }
    fixed_fields = ('target_snapshot', 'workload_key', 'sampling_key')
    mismatches = [name for name in fixed_fields if baseline.get(name) != candidate.get(name)]
    required = ('throughput_tokens_per_s', 'ttft_p95_ms', 'e2e_p95_ms', 'peak_memory_mb', 'quality_ok')
    missing = [name for name in required if candidate.get(name) is None or baseline.get(name) is None]
    path_ok = baseline.get('execution_path') == 'target_only' and candidate.get('execution_path') == 'speculative'
    if not path_ok:
        missing.append('speculative_execution_path')
    if candidate.get('mechanism_active') is not True:
        missing.append('observed_speculative_proposals')
    if mismatches:
        return {'evidence_status': 'incomparable', 'mismatches': mismatches, 'missing_evidence': missing}
    if missing:
        return {'evidence_status': 'incomplete', 'mismatches': [], 'missing_evidence': missing}
    throughput_speedup = candidate['throughput_tokens_per_s'] / baseline['throughput_tokens_per_s']
    return {
        'evidence_status': 'comparable', 'mismatches': [], 'missing_evidence': [],
        'throughput_speedup': throughput_speedup,
        'ttft_ratio': candidate['ttft_p95_ms'] / baseline['ttft_p95_ms'],
        'e2e_p95_ratio': candidate['e2e_p95_ms'] / baseline['e2e_p95_ms'],
        'candidate_peak_memory_mb': candidate['peak_memory_mb'],
        'quality_ok': bool(candidate['quality_ok']),
        'mechanism_active': True,
    }


def decide_speculative_candidate(
    evidence: Mapping[str, Any], thresholds: Mapping[str, float]
) -> Dict[str, str]:
    status = evidence.get('evidence_status')
    if status not in ('comparable', 'incomplete'):
        return {'decision': 'not_evaluated', 'reason': f'evidence_{status}'}
    if status == 'incomplete':
        return {'decision': 'tune', 'reason': 'collect_missing_evidence'}
    if not evidence.get('quality_ok', False):
        return {'decision': 'reject', 'reason': 'quality_gate_failed'}
    gates = {
        'throughput': evidence['throughput_speedup'] >= thresholds['min_throughput_speedup'],
        'ttft': evidence['ttft_ratio'] <= thresholds['max_ttft_regression_ratio'],
        'e2e_p95': evidence['e2e_p95_ratio'] <= thresholds['max_e2e_p95_regression_ratio'],
        'memory': evidence['candidate_peak_memory_mb'] <= thresholds['max_peak_memory_mb'],
    }
    if all(gates.values()):
        return {'decision': 'accept', 'reason': 'quality_and_net_benefit_passed'}
    return {'decision': 'tune', 'reason': 'one_or_more_performance_gates_failed'}

```

### 解析

#### TODO 1：区分候选家族的资产要求

- MTP 依赖目标模型内置结构，EAGLE 等依赖匹配权重；规范化名称后返回家族资料的副本。
- 家族测试分别核对检索式、内置式与带块内依赖的候选，并拒绝未知名称，避免将不同资产要求混为一类。

#### TODO 2：保留首个阻断原因

- 先检查是否启用，再按后端、资产、兼容和容量顺序判断；静态可启动状态不是性能决策。
- 测试逐项破坏条件，再同时破坏后端与资产，检查优先返回后端不支持，而不是丢失失败原因。

#### TODO 3：在同一比较条件下计算比值

- 目标版本、请求集合和采样配置必须一致，且两条路径已完成、候选确有投机活动；比值才有对应的比较对象。
- 配对测试分别改变固定条件、移除指标和模拟路径回退；成功配对才计算吞吐加速比与延迟比。

#### TODO 4：联合判断质量与资源性能

- 吞吐加速比需要达到下限，延迟比和峰值显存不能超过上限；质量失败不能由更好的性能抵消。
- 测试覆盖精确阈值、单项门槛未过、质量失败和缺证，确认接受率不参与采用条件。

## Step 5（可选）：GPU 与 backend 实验——多 proposer 准入与配对

| CPU 已验证 | 真实环境补充 | 本页输出 |
|:---|:---|:---|
| proposer 分类、状态机与配对规则 | backend method、模型资产、真实执行路径、端到端与资源指标 | G0–G5 状态 JSON、已执行 raw JSON、日志与 manifest |

实验分为两条轨道：同一 target 下可以直接比较绝对指标；若某个 proposer 只能配合另一 target，则必须另建 target-only 基线，只比较组内 speedup，不能横向比较原始延迟。

### 5.1 环境、候选与固定工作负载

历史基线使用 [INF-VLLM-011-CU128](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-quickstart)，新候选选型使用独立的 [INF-VLLM-SPEC025](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-spec025-quickstart)。环境安装和驱动要求由 Quickstart 维护，当前证据见[实验资产表](../topic_discussion/inference_optimization/experiment_assets.md)。每份结果保留实际运行版本，不能把新环境信息补写进历史记录。

默认关闭真实执行；复测需使用唯一标签。候选计划包含不需额外权重的 N-gram，以及依赖模型结构或草稿资产的扩展；缺资产或能力时保留状态而不启动服务。


```python
# 5.1：实验时只修改下面两个用户输入；其余字段是冻结的比较契约。
RUN_REAL_EXPERIMENT = False  # False 只做预检；True 才允许 5.4 启动真实服务。
RUN_LABEL = 'replace_with_unique_label'  # 正式复测使用新标签；不要复用 v3、v4 或 v5。
SEMANTIC_ID = 'INF-SPECULATIVE-BENCH'
RESULT_SCHEMA = 'inference-benchmark/v1'

# 环境身份：保留当前复现环境；升级实验使用仓库已登记 Profile，不原地覆盖。
VLLM_ENV = 'llm_vllm_cu128'
PROFILE_ID = 'INF-VLLM-011-CU128'
UPGRADE_ENV = 'llm_vllm_spec025_cu129'
UPGRADE_PROFILE_ID = 'INF-VLLM-SPEC025'
UPGRADE_VLLM_VERSION = '0.25.1+cu129'
OPTIONAL_RETEST_PROFILE_ID = 'INF-VLLM-MTP'  # vLLM 0.31，仅用于后续复测。
UPGRADE_MIN_DRIVER = '575.57'  # CUDA 12.9 Profile 的驱动下限；创建环境前仍需完成预检。
HARDWARE_PROFILE = 'H0_12GB_5070Ti_Laptop'
BACKEND = 'vllm'

# 固定 target 与运行参数。
TARGET_MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
MODEL_SOURCE = 'auto'  # auto / local / huggingface / modelscope。
MODEL_CACHE = 'auto'  # 自动复用仓库缓存、HF_HUB_CACHE 或 ~/.cache/huggingface/hub。
DTYPE = 'bfloat16'
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.80
ENFORCE_EAGER = True

# 两类 workload：general 检查通用生成，repetition 暴露 N-gram 的适用边界。
WORKLOADS = {
    'general': 'benchmarks/workloads/speculative_general.jsonl',
    'repetition': 'benchmarks/workloads/speculative_repetition.jsonl',
}
CONCURRENCY_LEVELS = [1]  # 首轮 smoke 固定单并发；通过后再建立独立 full 标签。
NUM_PROMPTS = 8  # 与每份冻结 workload 的 8 行一一对应，不循环 request_id。
WARMUP = 2
REPEATS = 1
MAX_TOKENS = 64
TEMPERATURE = 0.0
REQUEST_TIMEOUT_S = 180.0
COMMON_SPECULATIVE_TOKENS = 5

# candidate 配置：asset=None 表示无外置权重或尚未提供；method=None 让新版 vLLM 按模型资产自动识别。
STRATEGIES = {
    'g0_target_only': {'enabled': True, 'family': 'baseline', 'method': None, 'asset': None},
    'g1_ngram': {
        'enabled': True, 'family': 'ngram', 'method': 'ngram', 'asset': None,
        'options': {'prompt_lookup_min': 3, 'prompt_lookup_max': 5},
    },
    'g2_mtp': {'enabled': True, 'family': 'mtp', 'method': 'mtp', 'asset': None},
    'g3_eagle': {'enabled': True, 'family': 'eagle', 'method': 'eagle3', 'asset': None},
    'g4_dflash': {'enabled': True, 'family': 'dflash', 'method': None, 'asset': None, 'min_vllm': '0.25.1'},
    'g5_dspark': {'enabled': True, 'family': 'dspark', 'method': None, 'asset': None, 'min_vllm': '0.25.1'},
}

# 采用门槛。接受率是诊断项，不是独立采用门槛。
MIN_THROUGHPUT_SPEEDUP = 1.10
MAX_TTFT_REGRESSION_RATIO = 1.10
MAX_E2E_P95_REGRESSION_RATIO = 1.05
MAX_PEAK_MEMORY_MB = 10500.0
REQUIRE_EXACT_MATCH_AT_T0 = True

# 派生路径：不要逐个硬编码结果文件名。
RESULT_DIR = 'benchmarks/results'
MANIFEST_PATH = f'{RESULT_DIR}/68_manifest_{RUN_LABEL}.json'

```

### 5.2 环境与输入预检

只读检查仓库、Kernel、CUDA、vLLM、workload 和缓存目录；不下载模型、不启动服务。这里同时显示当前实验环境和建议升级环境，实际结果以当前 Kernel 及解析到的 vLLM 可执行文件为准。


```python
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tools.backend_runtime import probe_vllm_speculative_support, resolve_vllm_command
from tools.model_runtime import find_cached_snapshot, model_cache_candidates
from tools.inference_project_runtime import (
    admit_speculative_candidate, build_speculative_pair_evidence,
    decide_speculative_candidate, locate_repo_root, runtime_snapshot,
)

repo_root = locate_repo_root()
result_dir = repo_root / RESULT_DIR
result_dir.mkdir(parents=True, exist_ok=True)
workload_paths = {name: repo_root / path for name, path in WORKLOADS.items()}
missing_workloads = [str(path) for path in workload_paths.values() if not path.is_file()]
if missing_workloads:
    raise FileNotFoundError(f'缺少 speculative workload：{missing_workloads}')
if set(STRATEGIES) != {'g0_target_only', 'g1_ngram', 'g2_mtp', 'g3_eagle', 'g4_dflash', 'g5_dspark'}:
    raise ValueError('68 必须保留 G0–G5 的固定候选身份')
if TEMPERATURE != 0.0 and REQUIRE_EXACT_MATCH_AT_T0:
    raise ValueError('逐请求精确输出核对仅适用于 TEMPERATURE=0')

vllm_command = resolve_vllm_command(environment=VLLM_ENV)
capability = probe_vllm_speculative_support(environment=VLLM_ENV)
try:
    driver_version = subprocess.run(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=5,
    ).stdout.strip().splitlines()[0]
except (OSError, IndexError, subprocess.SubprocessError):
    driver_version = None
try:
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if gpu_available else 0.0
except ImportError:
    torch = None
    gpu_available = False
    gpu_name = None
    gpu_total_gb = 0.0

target_snapshot_preflight = find_cached_snapshot(TARGET_MODEL_ID, MODEL_CACHE)
result_collisions = sorted(path.name for path in result_dir.glob(f'68_*{RUN_LABEL}*'))
audit_errors = []
if not re.fullmatch(r'[A-Za-z0-9._-]+', RUN_LABEL):
    audit_errors.append('RUN_LABEL 只能包含字母、数字、点、下划线和连字符')
if not 0 < GPU_MEMORY_UTILIZATION < 1:
    audit_errors.append('GPU_MEMORY_UTILIZATION 必须位于 (0, 1)')
if any(value <= 0 for value in (*CONCURRENCY_LEVELS, NUM_PROMPTS, REPEATS, MAX_TOKENS, COMMON_SPECULATIVE_TOKENS)) or WARMUP < 0:
    audit_errors.append('并发、请求数、重复次数、生成长度和 proposal length 必须为正；预热不能为负')
ngram_options = STRATEGIES['g1_ngram'].get('options', {})
if not 1 <= ngram_options.get('prompt_lookup_min', 0) <= ngram_options.get('prompt_lookup_max', 0):
    audit_errors.append('N-gram prompt_lookup_min/max 配置无效')
if RUN_REAL_EXPERIMENT and target_snapshot_preflight is None:
    audit_errors.append(f'未在自动缓存路径中找到 target：{TARGET_MODEL_ID}')
if RUN_REAL_EXPERIMENT and not gpu_available:
    audit_errors.append('当前 Kernel 无法访问 CUDA GPU')
if RUN_REAL_EXPERIMENT and capability.get('status') != 'supported':
    audit_errors.append('当前 vLLM 未暴露可识别的 speculative 接口')
if RUN_REAL_EXPERIMENT and result_collisions:
    audit_errors.append(f'RUN_LABEL 已存在结果文件：{result_collisions}')

preflight = {
    'repo_root': str(repo_root), 'kernel_python': sys.executable,
    'configured_env': VLLM_ENV, 'profile_id': PROFILE_ID,
    'upgrade_env': UPGRADE_ENV, 'upgrade_profile_id': UPGRADE_PROFILE_ID,
    'upgrade_vllm_version': UPGRADE_VLLM_VERSION,
    'optional_retest_profile_id': OPTIONAL_RETEST_PROFILE_ID,
    'upgrade_min_driver': UPGRADE_MIN_DRIVER, 'detected_driver': driver_version,
    'vllm_command': vllm_command, 'vllm_version': capability.get('version'),
    'speculative_interface': capability.get('mode'),
    'declared_methods': capability.get('methods', []),
    'gpu_available': gpu_available, 'gpu_name': gpu_name, 'gpu_total_gb': gpu_total_gb,
    'model_cache_mode': MODEL_CACHE,
    'model_cache_candidates': [str(path) for path in model_cache_candidates(MODEL_CACHE)],
    'resolved_target_snapshot': str(target_snapshot_preflight) if target_snapshot_preflight else None,
    'result_collisions': result_collisions, 'audit_errors': audit_errors,
    'workloads': {name: str(path) for name, path in workload_paths.items()},
    'runtime': runtime_snapshot(torch),
}
print(json.dumps(preflight, ensure_ascii=False, indent=2))
if audit_errors:
    raise RuntimeError('实验前配置审计未通过：' + '；'.join(audit_errors))

```

### 5.3 候选能力、资产与容量检查

本单元按本地缓存、模型配置和后端方法声明初筛候选，不启动服务。通用投机接口、资产目录存在或版本达到下限，都不能单独确认权重配对与真实执行；还需核对目标结构、匹配版本及启动记录。容量估算未提供时，静态检查也不能证明显存足够。


```python
def _version_tuple(value):
    numbers = re.findall(r'\d+', str(value or '0'))
    return tuple(int(item) for item in numbers[:3]) + (0,) * max(0, 3 - len(numbers[:3]))


def _cached_snapshot(model_id):
    candidate = Path(str(model_id)).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return find_cached_snapshot(str(model_id), MODEL_CACHE)


def _read_hf_config(snapshot):
    config_path = snapshot / 'config.json' if snapshot else None
    if not config_path or not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding='utf-8'))


target_snapshot = _cached_snapshot(TARGET_MODEL_ID)
target_config = _read_hf_config(target_snapshot)
target_model_type = str(target_config.get('model_type', '')).lower()
native_mtp = any((
    target_model_type.endswith('_mtp'),
    bool(target_config.get('num_nextn_predict_layers')),
    bool(target_config.get('num_mtp_modules')),
))
declared_methods = set(capability.get('methods') or ())
backend_version = capability.get('version')
memory_limit_gb = MAX_PEAK_MEMORY_MB / 1024

admission = {}
for group_name, strategy in STRATEGIES.items():
    if group_name == 'g0_target_only':
        admission[group_name] = {
            'run_status': 'runnable' if target_snapshot and gpu_available else 'asset_missing',
            'reason': 'baseline_ready' if target_snapshot and gpu_available else 'target_or_gpu_missing',
            'resolved_asset': str(target_snapshot) if target_snapshot else None,
        }
        continue
    family = strategy['family']
    method = strategy.get('method')
    min_version_ok = _version_tuple(backend_version) >= _version_tuple(strategy.get('min_vllm', '0'))
    interface_ok = capability.get('status') == 'supported'
    method_ok = method in declared_methods if method else min_version_ok
    if family == 'ngram':
        asset_ready = True
        asset_path = None
    elif family == 'mtp':
        asset_ready = native_mtp
        asset_path = str(target_snapshot) if native_mtp and target_snapshot else None
    else:
        asset_path_obj = _cached_snapshot(strategy.get('asset')) if strategy.get('asset') else None
        asset_ready = asset_path_obj is not None
        asset_path = str(asset_path_obj) if asset_path_obj else None
    candidate = {
        **strategy,
        'backend_supported': interface_ok and method_ok and min_version_ok,
        'asset_ready': asset_ready,
        'compatible': bool(target_snapshot),
        'estimated_memory_gb': None,
        'memory_limit_gb': memory_limit_gb,
    }
    admission[group_name] = {
        **admit_speculative_candidate(candidate),
        'family': family, 'backend_method': method,
        'resolved_asset': asset_path, 'min_version_ok': min_version_ok,
    }

print(json.dumps({
    'target_snapshot': str(target_snapshot) if target_snapshot else None,
    'target_model_type': target_model_type, 'native_mtp_detected': native_mtp,
    'backend_version': backend_version, 'declared_methods': sorted(declared_methods),
    'admission': admission,
}, ensure_ascii=False, indent=2))

```

### 5.4 顺序执行候选并保存原始证据

对静态可启动的候选依次完成启动、就绪、预热、测量、保存与停止，避免多个服务同时占用显存。关闭、受阻和执行失败的候选也保留状态记录；目标基线没有完成时，不形成候选性能结论。


```python
import os
import urllib.request

# vLLM 0.11 以固定周期把 EngineCore 统计刷新到 Prometheus；缩短周期，避免短测得到假零值。
os.environ.setdefault('VLLM_LOG_STATS_INTERVAL', '0.5')

from tools.project_runtime import save_json_record_once

from tools.inference_project_runtime import (
    run_backend_benchmark,
    start_optional_vllm,
    start_speculative_strategy_vllm,
    stop_optional_vllm,
)


def _group_path(group_name, suffix='summary'):
    return result_dir / f'68_{group_name}_{suffix}_{RUN_LABEL}.json'


def _write_json(path, payload):
    save_json_record_once(path, payload)


def _scenario_key(workload_name, concurrency):
    return f'{workload_name}-c{concurrency}'


def _prometheus_sum(text, names):
    values = []
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        metric_name = line.split('{', 1)[0].split(None, 1)[0]
        if metric_name not in names:
            continue
        try:
            values.append(float(line.rsplit(None, 1)[-1]))
        except ValueError:
            continue
    return sum(values) if values else None


def _read_speculative_counters(base_url):
    try:
        with urllib.request.urlopen(base_url.rstrip('/') + '/metrics', timeout=5) as response:
            text = response.read().decode('utf-8', errors='replace')
    except Exception:
        return {'drafts': None, 'proposed_tokens': None, 'accepted_tokens': None}
    return {
        'drafts': _prometheus_sum(text, {
            'vllm:spec_decode_num_drafts', 'vllm:spec_decode_num_drafts_total',
        }),
        'proposed_tokens': _prometheus_sum(text, {
            'vllm:spec_decode_num_draft_tokens', 'vllm:spec_decode_num_draft_tokens_total',
        }),
        'accepted_tokens': _prometheus_sum(text, {
            'vllm:spec_decode_num_accepted_tokens', 'vllm:spec_decode_num_accepted_tokens_total',
        }),
    }


def _counter_delta(before, after, name):
    left, right = before.get(name), after.get(name)
    return right - left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None


def _measure_server(base_url, group_name, role, selected_dtype):
    scenarios = {}
    for workload_name, workload in WORKLOADS.items():
        for concurrency in CONCURRENCY_LEVELS:
            key = _scenario_key(workload_name, concurrency)
            backend_path = result_dir / f'68_{group_name}_{key}_backend_raw_{RUN_LABEL}.json'
            raw_path = result_dir / f'68_{group_name}_{key}_raw_{RUN_LABEL}.json'
            counters_before = _read_speculative_counters(base_url)
            report = run_backend_benchmark(
                project='68', base_url=base_url, model=TARGET_MODEL_ID,
                label=group_name, output=backend_path, workload=workload,
                num_prompts=NUM_PROMPTS, max_tokens=MAX_TOKENS,
                concurrency=concurrency, warmup=WARMUP, repeats=REPEATS,
                temperature=TEMPERATURE, timeout_s=REQUEST_TIMEOUT_S,
                backend=BACKEND, dtype=selected_dtype, role=role,
                include_requests=True, measure_device_peak_memory=True,
                quiet=True,
            )
            counters_after = _read_speculative_counters(base_url)
            proposed = _counter_delta(counters_before, counters_after, 'proposed_tokens')
            accepted = _counter_delta(counters_before, counters_after, 'accepted_tokens')
            drafts = _counter_delta(counters_before, counters_after, 'drafts')
            mechanism_metrics = {
                'draft_rounds': drafts,
                'proposed_tokens': proposed,
                'accepted_tokens': accepted,
                'acceptance_rate': accepted / proposed if proposed else None,
                'mean_accepted_length': 1 + accepted / drafts if drafts else None,
                'mechanism_active': proposed > 0 if proposed is not None else None,
                'source': 'vllm_prometheus_counter_delta' if proposed is not None else 'not_exported',
            }
            report.setdefault('metrics', {}).update(mechanism_metrics)
            report['mechanism_metrics'] = mechanism_metrics
            _write_json(raw_path, report)
            scenarios[key] = {'raw_path': str(raw_path.relative_to(repo_root)), 'report': report}
    return scenarios


def _compact_scenario(item):
    report = item['report']
    return {
        'raw_path': item['raw_path'],
        'workload': report.get('workload', {}),
        'metrics': report.get('metrics', {}),
        'resource_measurement': report.get('resource_measurement', {}),
    }


group_reports = {}
baseline_scenarios = None
for group_name, strategy in STRATEGIES.items():
    gate = admission[group_name]
    summary_path = _group_path(group_name)
    base_record = {
        'schema_version': 'speculative-selection/v1', 'project': '68',
        'semantic_id': SEMANTIC_ID, 'benchmark_schema': RESULT_SCHEMA,
        'run_label': RUN_LABEL, 'group': group_name, 'strategy': strategy,
        'admission': gate, 'environment': preflight,
        'run_status': gate['run_status'], 'evidence_status': 'not_evaluated',
        'decision': {'decision': 'not_evaluated', 'reason': 'not_executed'},
    }
    if not RUN_REAL_EXPERIMENT or gate['run_status'] != 'runnable':
        if not RUN_REAL_EXPERIMENT and gate['run_status'] == 'runnable':
            base_record['run_status'] = 'skipped'
            base_record['admission'] = {**gate, 'reason': 'RUN_REAL_EXPERIMENT=False'}
        if RUN_REAL_EXPERIMENT:
            _write_json(summary_path, base_record)
        group_reports[group_name] = base_record
        continue

    server = log_handle = None
    log_path = result_dir / f'68_{group_name}_backend_{RUN_LABEL}.log'
    try:
        if group_name == 'g0_target_only':
            server, log_handle, port, selected_dtype, resolved_target = start_optional_vllm(
                model_id=TARGET_MODEL_ID, model_source=MODEL_SOURCE, cache_dir=MODEL_CACHE,
                dtype=DTYPE, vllm_environment=VLLM_ENV, max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION, enforce_eager=ENFORCE_EAGER,
                served_model_name=TARGET_MODEL_ID, generation_config='vllm', log_path=log_path,
            )
            capability_snapshot = {'resolved_target_snapshot': Path(resolved_target).name}
            role = 'baseline'
        else:
            server, log_handle, port, selected_dtype, resolved_target, capability_snapshot = start_speculative_strategy_vllm(
                target_model_id=TARGET_MODEL_ID, method=strategy.get('method'),
                proposal_length=COMMON_SPECULATIVE_TOKENS,
                proposer_model_id=strategy.get('asset'), method_options=strategy.get('options'),
                model_source=MODEL_SOURCE, cache_dir=MODEL_CACHE, dtype=DTYPE,
                vllm_environment=VLLM_ENV, max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION, enforce_eager=ENFORCE_EAGER,
                served_model_name=TARGET_MODEL_ID, generation_config='vllm', log_path=log_path,
            )
            role = 'candidate'
        scenarios = _measure_server(f'http://127.0.0.1:{port}', group_name, role, selected_dtype)
        record = {
            **base_record, 'run_status': 'completed',
            'execution_path': 'target_only' if role == 'baseline' else 'speculative',
            'capability': capability_snapshot,
            'backend_log': str(log_path.relative_to(repo_root)),
            'scenarios': {key: _compact_scenario(value) for key, value in scenarios.items()},
        }
        if role == 'baseline':
            baseline_scenarios = scenarios
    except Exception as error:
        record = {
            **base_record, 'run_status': 'failed',
            'failure': {'failure_reason': str(error), 'retest_path': 'inspect_backend_log_and_asset_contract'},
            'backend_log': str(log_path.relative_to(repo_root)),
        }
    finally:
        if server is not None:
            stop_optional_vllm(server, log_handle)
    _write_json(summary_path, record)
    group_reports[group_name] = record

manifest = {
    'schema_version': 'speculative-selection-manifest/v1', 'project': '68',
    'semantic_id': SEMANTIC_ID, 'benchmark_schema': RESULT_SCHEMA,
    'run_label': RUN_LABEL, 'config': {
        'target_model': TARGET_MODEL_ID, 'workloads': WORKLOADS,
        'concurrency_levels': CONCURRENCY_LEVELS, 'num_prompts': NUM_PROMPTS,
        'warmup': WARMUP, 'repeats': REPEATS, 'max_tokens': MAX_TOKENS,
        'temperature': TEMPERATURE, 'common_speculative_tokens': COMMON_SPECULATIVE_TOKENS,
    },
    'groups': [
        {'group': name, 'summary_path': str(_group_path(name).relative_to(repo_root)),
         'run_status': report['run_status'], 'admission': report['admission']}
        for name, report in group_reports.items()
    ],
}
if RUN_REAL_EXPERIMENT:
    _write_json(repo_root / MANIFEST_PATH, manifest)
print(json.dumps({'manifest': MANIFEST_PATH if RUN_REAL_EXPERIMENT else None, 'groups': manifest['groups']}, ensure_ascii=False, indent=2))

```

### 5.5 读取结果并核对证据

本单元只展示有界摘要，不打印 raw requests。`temperature=0` 时按请求输出多重集合核对一致性；接受统计只读取 backend 原生字段，缺失时保持为空，不能从 TPOT 或 speedup 反推。


```python
from collections import Counter


def _metric(report, section, key='p95'):
    value = report.get('metrics', {}).get(section)
    return value.get(key) if isinstance(value, dict) else value


def _output_multiset(report):
    return Counter(
        (item.get('prompt'), item.get('output_text'))
        for item in (report.get('results') or []) if item.get('ok')
    )


manifest_file = repo_root / MANIFEST_PATH
if not manifest_file.is_file():
    raise FileNotFoundError(f'请先运行 5.4：{manifest_file}')
loaded_groups = {
    item['group']: json.loads((repo_root / item['summary_path']).read_text(encoding='utf-8'))
    for item in json.loads(manifest_file.read_text(encoding='utf-8'))['groups']
}
baseline_record = loaded_groups['g0_target_only']
summary_rows = []
for group_name, record in loaded_groups.items():
    if record.get('run_status') != 'completed':
        summary_rows.append({'group': group_name, 'run_status': record.get('run_status'),
                             'scenario': None, 'evidence_status': 'not_evaluated',
                             'reason': (record.get('failure') or record.get('admission') or {}).get('failure_reason')
                                       or (record.get('admission') or {}).get('reason')})
        continue
    for scenario_key, scenario in record.get('scenarios', {}).items():
        report = json.loads((repo_root / scenario['raw_path']).read_text(encoding='utf-8'))
        baseline_scenario = baseline_record.get('scenarios', {}).get(scenario_key)
        baseline_report = json.loads((repo_root / baseline_scenario['raw_path']).read_text(encoding='utf-8')) if baseline_scenario else None
        quality_ok = True if group_name == 'g0_target_only' else (
            _output_multiset(report) == _output_multiset(baseline_report)
            if REQUIRE_EXACT_MATCH_AT_T0 and baseline_report else None
        )
        metrics = report.get('metrics', {})
        summary_rows.append({
            'group': group_name, 'run_status': 'completed', 'scenario': scenario_key,
            'ttft_p95_ms': _metric(report, 'ttft_ms'),
            'tpot_p95_ms': _metric(report, 'tpot_ms'),
            'e2e_p95_ms': _metric(report, 'e2e_ms'),
            'throughput_tokens_per_s': metrics.get('output_token_throughput_per_s'),
            'peak_memory_mb': metrics.get('peak_memory_mb'),
            'quality_ok': quality_ok,
            'accepted_tokens': metrics.get('accepted_tokens'),
            'proposed_tokens': metrics.get('proposed_tokens'),
            'mechanism_active': metrics.get('mechanism_active'),
        })

for row in summary_rows:
    print(json.dumps(row, ensure_ascii=False))

```

### 5.6 形成候选决策与 72 节交接

逐场景形成配对证据，再汇总为候选结论。候选必须在至少一个声明场景完成可比实验；通用与重复 workload 结果分开保留，避免用 N-gram 的专用优势外推到所有请求。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 68 节 N-gram 候选；`rtx5070ti_speculative_ngram_smoke_v3` |
| 运行条件 | RTX 5070 Ti Laptop GPU；Qwen2.5-1.5B-Instruct；target-only 与 N-gram 配对 |
| 原始证据 | [G0 日志](../../benchmarks/results/68_g0_target_only_backend_rtx5070ti_speculative_ngram_smoke_v3.log)、[G1 日志](../../benchmarks/results/68_g1_ngram_backend_rtx5070ti_speculative_ngram_smoke_v3.log) |
| 正式证据 | [证据记录](../../benchmarks/results/68_evidence_record_rtx5070ti_speculative_ngram_smoke_v3.json) |
| 清单与交接 | [结果清单](../../benchmarks/results/68_manifest_rtx5070ti_speculative_ngram_smoke_v3.json)；质量门槛未通过，未生成 72 节交接 |
| 结论 | `reject`：N-gram 有性能信号，但输出一致性未通过；其他 proposer 家族尚无同等级实测证据 |

**后续尝试（非性能结论）**：`rtx5070ti_speculative_project_v5` 的 G0 基线成功，G1 speculative 后端未能就绪并超时；该尝试没有候选性能数据，不替代 v3 的正式结论。见 [v5 清单](../../benchmarks/results/68_manifest_rtx5070ti_speculative_project_v5.json)、[G0](../../benchmarks/results/68_g0_target_baseline_vllm_rtx5070ti_speculative_project_v5.json) 与 [G1 失败记录](../../benchmarks/results/68_g1_speculative_vllm_rtx5070ti_speculative_project_v5.json)。

```python
thresholds = {
    'min_throughput_speedup': MIN_THROUGHPUT_SPEEDUP,
    'max_ttft_regression_ratio': MAX_TTFT_REGRESSION_RATIO,
    'max_e2e_p95_regression_ratio': MAX_E2E_P95_REGRESSION_RATIO,
    'max_peak_memory_mb': MAX_PEAK_MEMORY_MB,
}
rows_by_group = {}
for row in summary_rows:
    rows_by_group.setdefault(row['group'], {})[row.get('scenario')] = row

decisions = {}
for group_name in STRATEGIES:
    if group_name == 'g0_target_only':
        continue
    candidate_record = loaded_groups[group_name]
    scenario_decisions = {}
    if candidate_record.get('run_status') == 'completed':
        for scenario_key, candidate_row in rows_by_group.get(group_name, {}).items():
            baseline_row = rows_by_group.get('g0_target_only', {}).get(scenario_key)
            baseline_input = {
                'run_status': 'completed', 'role': 'baseline',
                'target_snapshot': baseline_record.get('capability', {}).get('resolved_target_snapshot'),
                'workload_key': scenario_key, 'sampling_key': f't{TEMPERATURE}-max{MAX_TOKENS}',
                'execution_path': 'target_only',
                'throughput_tokens_per_s': baseline_row.get('throughput_tokens_per_s') if baseline_row else None,
                'ttft_p95_ms': baseline_row.get('ttft_p95_ms') if baseline_row else None,
                'e2e_p95_ms': baseline_row.get('e2e_p95_ms') if baseline_row else None,
                'peak_memory_mb': baseline_row.get('peak_memory_mb') if baseline_row else None,
                'quality_ok': True,
            }
            candidate_input = {
                'run_status': 'completed', 'role': 'candidate',
                'target_snapshot': candidate_record.get('capability', {}).get('resolved_target_snapshot'),
                'workload_key': scenario_key, 'sampling_key': f't{TEMPERATURE}-max{MAX_TOKENS}',
                'execution_path': candidate_record.get('execution_path'),
                'throughput_tokens_per_s': candidate_row.get('throughput_tokens_per_s'),
                'ttft_p95_ms': candidate_row.get('ttft_p95_ms'),
                'e2e_p95_ms': candidate_row.get('e2e_p95_ms'),
                'peak_memory_mb': candidate_row.get('peak_memory_mb'),
                'quality_ok': candidate_row.get('quality_ok'),
                'mechanism_active': candidate_row.get('mechanism_active'),
            }
            evidence = build_speculative_pair_evidence(baseline_input, candidate_input)
            if candidate_row.get('mechanism_active') is not True:
                evidence = {
                    'evidence_status': 'incomplete', 'mismatches': [],
                    'missing_evidence': ['observed_speculative_proposals'],
                }
            scenario_decisions[scenario_key] = {
                'evidence': evidence,
                'decision': decide_speculative_candidate(evidence, thresholds),
            }
    else:
        evidence = build_speculative_pair_evidence(
            {'run_status': baseline_record.get('run_status')},
            {'run_status': candidate_record.get('run_status')},
        )
        scenario_decisions['not_run'] = {
            'evidence': evidence,
            'decision': decide_speculative_candidate(evidence, thresholds),
        }
    decisions[group_name] = scenario_decisions

handoff = {
    'schema_version': 'speculative-selection-handoff/v1', 'project': '68',
    'run_label': RUN_LABEL, 'target_model': TARGET_MODEL_ID,
    'decisions': {
        group: {scenario: item for scenario, item in scenarios.items()
                if item['decision']['decision'] == 'accept'}
        for group, scenarios in decisions.items()
        if any(item['decision']['decision'] == 'accept' for item in scenarios.values())
    },
    'next_project': '72',
    'rule': '仅交接已完成、可比较且 decision=accept 的具体场景；跨 target 只比较组内 speedup。',
}
handoff_path = result_dir / f'68_handoff_to_72_{RUN_LABEL}.json'
handoff_written = RUN_REAL_EXPERIMENT and bool(handoff['decisions'])
if handoff_written:
    _write_json(handoff_path, handoff)
print(json.dumps({'handoff_path': str(handoff_path.relative_to(repo_root)) if handoff_written else None,
                  'decisions': decisions,
                  'handoff_status': 'saved' if handoff_written else 'not_written'}, ensure_ascii=False, indent=2))

```

## 相关阅读

继续追溯候选的模型结构，或把通过的局部方案放进部署约束中比较。

- [原生多词元预测](./35_Multi_Token_Decoding.md)：模型内置候选与验证。
- [动态候选与验证预算](./INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md)：接受情况、候选长度与额外成本。
- [模型架构审计](./61_Model_Architecture_Exploration.md)：检查权重包含的原生结构。
- [单卡部署与后端对比](./72_Multi_Model_Deployment_Comparison.md)：把候选放入容量、质量和性能约束。
- [投机采样论文](https://arxiv.org/abs/2211.17192)：接受与拒绝修正的推导。
