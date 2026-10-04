# 61. Model Architecture Exploration | 模型架构审计

**难度：** Hard | **环境：** CPU-first；真实 checkpoint 配置审计可选 | **标签：** `模型结构`, `架构审计`, `选型决策` | **目标人群：** 希望把模型配置转成结构判断的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

看到一个模型配置时，真正需要回答的不是“它用了哪些名词”，而是：它改动了哪个结构变量、这个变量怎样改变参数和状态账本、以及还缺少哪些真实证据才能进入训练或部署。本节把 baseline 与单变量候选整理为架构审计卡，先核对结构差异和理论账本，再决定继续做真实 checkpoint 审计、专项 benchmark，还是停止当前假设。

长序列记忆的质量和成本不在本节直接测量；当候选涉及显式 KV、递推状态或混合架构时，使用 [77 长序列记忆架构基准](./77_Long_Sequence_Memory_Architecture_Benchmark.md) 运行长度阶梯对照。

**关键词：** `baseline`, `candidate`, `architecture profile`, `single-variable change`, `evidence level`
## 前置阅读

**导语：** 先能读出 Decoder Block 的组成与配置字段，再把一个结构改动写成可比较的候选，而不是把多个模型差异混成一个结论。

- [05. LLaMA3 Block Tutorial | LLaMA3 Block 教程](./05_LLaMA3_Block_Tutorial.md)
- [08. Architecture Tricks | 架构技巧](./08_Architecture_Tricks.md)
- [ARCH-MLA. Multi-head Latent Attention | 多头潜在注意力](./ARCH-MLA_Multihead_Latent_Attention.md)
- [ARCH-HYBRID-MEMORY. Attention / SSM Hybrid | Attention 与 SSM 混合记忆](./ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md)
- [大模型架构 · 代表模型与结构对照](../topic_discussion/llm_architecture_evolution/08_representative_models.md)
### Step 1：从模型配置提出一个可检验的结构问题

架构审计从一个明确的 baseline 和一个候选改动开始。例如，把 MHA 改为 GQA 时，研究变量是 `num_key_value_heads`；改变 FFN 容量时，研究变量是 `intermediate_size`。模型名称、训练数据、tokenizer、dtype、workload 与评测协议应先固定，避免把多个变化误写成单一结构收益。

| 审计元素 | 需要写清楚 | 作用 |
| --- | --- | --- |
| baseline | 模型版本与完整 profile | 给所有差分一个参照 |
| 单变量候选 | 改动字段、改动前后值、设计动机 | 说明这轮究竟在检验什么 |
| 固定条件 | 数据、tokenizer、dtype、workload 与评测协议 | 保持结果可比较 |
| 证据出口 | config 审计、专项 benchmark、训练或服务评测 | 明确下一步需要的证据 |

![架构候选的受控比较](../public/02_PyTorch_Algorithms/61_architecture_experiment_flow.svg)
### Step 2：把结构差异落到参数、状态与访问接口

Decoder 结构差异通常会先体现在三类账本：参数量、历史状态表示和访问接口。MHA/GQA/MQA 主要改变 KV 投影与推理状态；FFN 宽度主要改变参数和计算；窗口、稀疏访问或递推状态则改变历史的保存与读取方式。账本用于提出假设，不能替代真实模型运行。

| 结构变量 | 直接变化 | 审计账本 | 后续证据入口 |
| --- | --- | --- | --- |
| `num_key_value_heads` / `latent_dim` | K/V 共享或 latent 表示 | Attention 参数、每 token KV/latent 字节 | [71 MLA / KV Cache 结构基准](./71_MLA_KV_Cache_Architecture_Benchmark.md) |
| `intermediate_size` | FFN 宽度 | MLP 参数与权重字节 | 训练或推理 workload 复测 |
| `attention_pattern` / `window_size` | 历史可见范围 | 连接范围与传播路径 | [78 Attention 访问模式基准](./78_Attention_Access_Pattern_Benchmark.md) |
| `state_representation` / `layer_schedule` | 显式 KV、递推或混合状态 | 状态增长与 Attention/SSM 层分工 | [77 长序列记忆架构基准](./77_Long_Sequence_Memory_Architecture_Benchmark.md) |
### Step 3：区分结构假设、真实配置与专项性能证据

结构账本只能说明“可能改变什么”。真实 checkpoint 的 config 审计可以确认候选是否真的采用该结构；固定 workload 下的 benchmark 才能说明质量、延迟或显存是否变化。不同模型之间即使数值不同，也可能受规模、训练数据、tokenizer 和实现路径影响。

| 证据等级 | 可以回答的问题 | 不能直接推出的结论 | 下一步 |
| --- | --- | --- | --- |
| `cpu_structure_ledger` | 参数、状态和接口是否自洽 | 真实速度或任务质量 | 读取 checkpoint config |
| `checkpoint_config_audit` | 候选实际使用了哪些结构字段 | 结构必然带来质量收益 | 选择对应专项 benchmark |
| `same_family_benchmark` | 近似同家族条件下的质量 / 成本差分 | 对其他模型家族的普遍结论 | 扩展长度或任务集 |
| `cross_model_observational` | 不同模型的部署候选差异 | 差异完全由某个结构变量造成 | 寻找匹配变体或保留为观察记录 |
### Step 4：代码设计——生成架构审计卡

题目区把 config、单变量候选、理论账本、专项证据和推荐动作连接为一张审计卡。结构账本说明候选改变了什么；质量、成本、失败状态和证据等级说明这个改变是否已经在相同 workload 下得到验证。

| TODO | 代码契约与机制责任 | 关键约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1 | 提取并校验 Decoder profile | hidden/head/KV head 与层数合法 | MHA、GQA、MQA 与非法维度 |
| TODO 2 | 核对候选是否是单变量结构改动 | `changed_field` 与真实差分一致 | 合法单变量与多变量候选 |
| TODO 3 | 建立参数与状态账本 | Q/K/V、FFN 与状态表示分别记录 | GQA 与 MHA 的 KV 账本差异 |
| TODO 4 | 生成带专项证据字段的审计卡 | 记录 workload、质量、成本、failure 与 evidence level | KV head 变更指向 71；空证据保持待测 |
| TODO 5 | 根据预算和证据状态给出下一步 | `accept` 需要可比质量与成本记录 | 结构账本、失败记录与完整 benchmark 走向不同 |

```python
from typing import Dict, List
```


```python
def extract_architecture_profile(config: Dict[str, object]) -> Dict[str, object]:
    """TODO 1：提取并校验 Decoder 的核心结构字段。"""
    # hidden_size = ???
    # attention_type = ???
    raise NotImplementedError('请先完成 TODO 1')


def validate_single_variable_candidate(baseline: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object]:
    """TODO 2：检查候选声明的 changed_field 是否与真实结构差分一致。"""
    # changed_fields = ???
    # ready = ???
    raise NotImplementedError('请先完成 TODO 2')


def build_architecture_ledger(profile: Dict[str, object], dtype_bytes: int = 2, seq_len: int = 1024) -> Dict[str, object]:
    """TODO 3：建立参数、理论权重和每 token KV / 状态账本。"""
    # attention_params = ???
    # kv_bytes_per_token = ???
    raise NotImplementedError('请先完成 TODO 3')


def build_architecture_audit_card(
    baseline: Dict[str, object], candidate: Dict[str, object], evidence: Dict[str, object] | None = None
) -> Dict[str, object]:
    """TODO 4：把结构差分、账本和对应证据入口写成审计卡。"""
    # comparison = ???
    # evidence_record = ???  # workload / quality / cost / failure / evidence_level
    # required_evidence = ???
    raise NotImplementedError('请先完成 TODO 4')


def recommend_audit_action(card: Dict[str, object], *, max_param_delta: int) -> Dict[str, object]:
    """TODO 5：按参数预算和证据状态返回 accept、tune 或 reject。"""
    # decision = ???
    # next_action = ???
    raise NotImplementedError('请先完成 TODO 5')
```


```python
# 每个测试只验证一个审计机制，便于定位 TODO 的责任。
def test_profile_contract():
    mha = extract_architecture_profile({'hidden_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 4, 'intermediate_size': 32, 'vocab_size': 100})
    gqa = extract_architecture_profile({**mha, 'num_key_value_heads': 2})
    mqa = extract_architecture_profile({**mha, 'num_key_value_heads': 1})
    assert mha['attention_type'] == 'MHA'
    assert gqa['attention_type'] == 'GQA'
    assert mqa['attention_type'] == 'MQA'
    assert not extract_architecture_profile({**mha, 'num_key_value_heads': 3})['ready']
    return mha, gqa


def test_single_variable_change(mha, gqa):
    candidate = {**gqa, 'changed_field': 'num_key_value_heads'}
    assert validate_single_variable_candidate(mha, candidate)['ready']
    assert not validate_single_variable_candidate(mha, {**candidate, 'intermediate_size': 48})['ready']


def test_architecture_ledger(mha, gqa):
    mha_ledger = build_architecture_ledger(mha, seq_len=64)
    gqa_ledger = build_architecture_ledger(gqa, seq_len=64)
    assert gqa_ledger['kv_bytes_per_token'] < mha_ledger['kv_bytes_per_token']
    assert gqa_ledger['attention_params'] < mha_ledger['attention_params']


def test_audit_card_evidence_contract(mha, gqa):
    card = build_architecture_audit_card(mha, {**gqa, 'changed_field': 'num_key_value_heads'})
    assert card['required_evidence']['primary_project'] == '71_MLA_KV_Cache_Architecture_Benchmark'
    assert set(card['required_evidence']['fields']) == {'workload', 'quality', 'cost', 'failure', 'evidence_level'}
    assert card['evidence']['evidence_level'] == 'cpu_structure_ledger'
    block_card = build_architecture_audit_card(mha, {**mha, 'intermediate_size': 48, 'changed_field': 'intermediate_size'})
    assert block_card['required_evidence']['primary_project'] == '60_Decoder_Block_Stability_Benchmark'
    moe_candidate = {**mha, 'num_local_experts': 8, 'changed_field': 'num_local_experts'}
    moe_card = build_architecture_audit_card(mha, moe_candidate)
    assert moe_card['required_evidence']['primary_project'] == '80_MoE_Expert_Parallel_Benchmark'
    assert recommend_audit_action(card, max_param_delta=0)['decision'] == 'tune'


def test_audit_decision(mha, gqa):
    candidate = {**gqa, 'changed_field': 'num_key_value_heads'}
    evidence = {
        'workload': {'context_length': 4096, 'prompt_template': 'fixed'},
        'quality': {'metric': 'retrieval_accuracy', 'value': 0.92, 'threshold': 0.90, 'passed': True},
        'cost': {'prefill_ms': 12.0, 'tpot_ms': 1.1, 'peak_memory_mb': 512.0, 'passed': True},
        'failure': None,
        'evidence_level': 'same_family_benchmark',
    }
    accepted = build_architecture_audit_card(mha, candidate, evidence)
    assert recommend_audit_action(accepted, max_param_delta=0)['decision'] == 'accept'
    failed = build_architecture_audit_card(mha, candidate, {**evidence, 'failure': 'cache interface unavailable'})
    assert recommend_audit_action(failed, max_param_delta=0)['decision'] == 'tune'


mha_profile, gqa_profile = test_profile_contract()
test_single_variable_change(mha_profile, gqa_profile)
test_architecture_ledger(mha_profile, gqa_profile)
test_audit_card_evidence_contract(mha_profile, gqa_profile)
test_audit_decision(mha_profile, gqa_profile)
print('✅ 模型架构审计项目：结构、账本、专项证据与决策测试通过。')

```

---

## 参考代码与解析

```python
def extract_architecture_profile(config: Dict[str, object]) -> Dict[str, object]:
    """TODO 1：提取并校验 Decoder 的核心结构字段。"""
    required = ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'vocab_size')
    missing = [key for key in required if key not in config]
    if missing:
        return {'ready': False, 'issues': [f'missing: {key}' for key in missing]}
    hidden_size = int(config['hidden_size'])
    attention_heads = int(config['num_attention_heads'])
    kv_heads = int(config.get('num_key_value_heads', attention_heads))
    issues = []
    if min(hidden_size, attention_heads, kv_heads, int(config['num_hidden_layers']), int(config['intermediate_size']), int(config['vocab_size'])) <= 0:
        issues.append('all dimensions must be positive')
    if not issues and hidden_size % attention_heads:
        issues.append('hidden_size must divide num_attention_heads')
    if not issues and attention_heads % kv_heads:
        issues.append('num_attention_heads must divide num_key_value_heads')
    attention_type = 'MHA' if kv_heads == attention_heads else ('MQA' if kv_heads == 1 else 'GQA')
    return {
        'ready': not issues, 'issues': issues, 'hidden_size': hidden_size,
        'num_hidden_layers': int(config['num_hidden_layers']), 'num_attention_heads': attention_heads,
        'num_key_value_heads': kv_heads, 'intermediate_size': int(config['intermediate_size']),
        'vocab_size': int(config['vocab_size']), 'head_dim': hidden_size // attention_heads if attention_heads else 0,
        'attention_type': attention_type, 'attention_pattern': str(config.get('attention_pattern', 'dense')),
        'state_representation': str(config.get('state_representation', 'kv_cache')),
        'num_local_experts': int(config.get('num_local_experts', config.get('num_experts', 0)) or 0),
        'num_experts_per_tok': int(config.get('num_experts_per_tok', config.get('top_k', 0)) or 0),
        'capacity_factor': float(config.get('capacity_factor', 0.0) or 0.0),
    }


def validate_single_variable_candidate(baseline: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object]:
    """TODO 2：检查候选声明的 changed_field 是否与真实结构差分一致。"""
    fields = ('num_key_value_heads', 'intermediate_size', 'num_hidden_layers', 'attention_pattern', 'state_representation', 'num_local_experts', 'num_experts_per_tok', 'capacity_factor')
    changes = [field for field in fields if baseline.get(field) != candidate.get(field)]
    declared = str(candidate.get('changed_field', ''))
    issues = []
    if not candidate.get('ready', False):
        issues.append('candidate profile is invalid')
    if len(changes) != 1:
        issues.append(f'expected exactly one structural change, got: {changes}')
    elif declared != changes[0]:
        issues.append(f'changed_field should be {changes[0]}')
    return {'ready': not issues, 'issues': issues, 'changed_fields': changes}


def build_architecture_ledger(profile: Dict[str, object], dtype_bytes: int = 2, seq_len: int = 1024) -> Dict[str, object]:
    """TODO 3：建立参数、理论权重和每 token KV / 状态账本。"""
    if not profile.get('ready', False) or dtype_bytes <= 0 or seq_len <= 0:
        raise ValueError('profile、dtype_bytes 或 seq_len 不合法')
    h, layers, heads, kv_heads = (int(profile[key]) for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads'))
    head_dim, intermediate, vocab = int(profile['head_dim']), int(profile['intermediate_size']), int(profile['vocab_size'])
    attention_per_layer = h * h + 2 * h * kv_heads * head_dim + h * h
    attention_params = layers * attention_per_layer
    mlp_params = layers * 3 * h * intermediate
    embedding_params = vocab * h
    norm_params = layers * 2 * h
    total_params = attention_params + mlp_params + embedding_params + norm_params
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes if profile['state_representation'] == 'kv_cache' else 0
    return {'attention_params': attention_params, 'mlp_params': mlp_params, 'embedding_params': embedding_params, 'norm_params': norm_params, 'total_params': total_params, 'weight_bytes': total_params * dtype_bytes, 'kv_bytes_per_token': kv_bytes_per_token, 'state_bytes_at_seq_len': kv_bytes_per_token * seq_len, 'evidence_level': 'cpu_structure_ledger'}


def build_architecture_audit_card(
    baseline: Dict[str, object], candidate: Dict[str, object], evidence: Dict[str, object] | None = None
) -> Dict[str, object]:
    """TODO 4：把结构差分、账本与专项证据写成审计卡。"""
    comparison = validate_single_variable_candidate(baseline, candidate)
    if not comparison['ready']:
        return {'ready': False, 'issues': comparison['issues']}
    baseline_ledger = build_architecture_ledger(baseline)
    candidate_ledger = build_architecture_ledger(candidate)
    changed_field = comparison['changed_fields'][0]
    primary_project = {
        'intermediate_size': '60_Decoder_Block_Stability_Benchmark',
        'num_hidden_layers': '60_Decoder_Block_Stability_Benchmark',
        'num_key_value_heads': '71_MLA_KV_Cache_Architecture_Benchmark',
        'attention_pattern': '78_Attention_Access_Pattern_Benchmark',
        'state_representation': '77_Long_Sequence_Memory_Architecture_Benchmark',
        'num_local_experts': '80_MoE_Expert_Parallel_Benchmark',
        'num_experts_per_tok': '80_MoE_Expert_Parallel_Benchmark',
        'capacity_factor': '80_MoE_Expert_Parallel_Benchmark',
    }.get(changed_field, 'workload_specific_training_or_inference_benchmark')
    evidence = dict(evidence or {})
    evidence_record = {
        'workload': dict(evidence.get('workload', {})),
        'quality': dict(evidence.get('quality', {})),
        'cost': dict(evidence.get('cost', {})),
        'failure': evidence.get('failure'),
        'evidence_level': str(evidence.get('evidence_level', 'cpu_structure_ledger')),
    }
    return {
        'ready': True,
        'changed_field': changed_field,
        'baseline_ledger': baseline_ledger,
        'candidate_ledger': candidate_ledger,
        'param_delta': candidate_ledger['total_params'] - baseline_ledger['total_params'],
        'required_evidence': {
            'primary_project': primary_project,
            'fields': ('workload', 'quality', 'cost', 'failure', 'evidence_level'),
        },
        'evidence': evidence_record,
    }


def recommend_audit_action(card: Dict[str, object], *, max_param_delta: int) -> Dict[str, object]:
    """TODO 5：按参数预算和专项证据返回 accept、tune 或 reject。"""
    if not card.get('ready', False):
        return {'decision': 'reject', 'reason': '候选结构契约不成立', 'next_action': 'repair_candidate'}
    if int(card['param_delta']) > max_param_delta:
        return {'decision': 'reject', 'reason': '参数增量超过预算', 'next_action': 'reduce_structure_change'}
    evidence = card.get('evidence', {})
    if evidence.get('failure'):
        return {'decision': 'tune', 'reason': f"专项证据存在失败：{evidence['failure']}", 'next_action': card['required_evidence']['primary_project']}
    if evidence.get('evidence_level') != 'same_family_benchmark':
        return {'decision': 'tune', 'reason': '尚缺同家族、同 workload 的 benchmark 证据', 'next_action': card['required_evidence']['primary_project']}
    if not evidence.get('quality', {}).get('passed') or not evidence.get('cost', {}).get('passed'):
        return {'decision': 'tune', 'reason': '质量与成本尚未同时满足目标', 'next_action': 'expand_or_remeasure'}
    return {'decision': 'accept', 'reason': '结构差分、预算与可比质量/成本证据一致', 'next_action': 'document_adoption'}

```

### 解析

审计卡先确认结构差分是否可归因，再把专项项目返回的证据按统一字段接入判断。

**TODO 1：结构 profile**：hidden size、Attention heads、KV heads 与层数构成最小结构记录；head 与 KV head 的关系区分 MHA、GQA、MQA。

**TODO 2：单变量候选**：候选一次只改变一个结构字段，才能把后续质量或成本变化与该改动对应。

**TODO 3：结构账本**：参数与每 token KV 状态账本给出候选的资源趋势，但不能替代真实测量。

**TODO 4：专项证据**：审计卡统一接收 `workload`、`quality`、`cost`、`failure` 和 `evidence_level`。KV head 的比较去 71，访问范围去 78，递推状态去 77；这些项目的结果回填后才能形成可比结论。

**TODO 5：项目决策**：结构不合法或超预算时拒绝；缺同家族 benchmark、质量/成本不达标或存在失败记录时继续调优；只有结构差分、预算与可比证据一致时才采用。
### Step 5：可选真实 checkpoint 配置审计

这一组单元只读取两个 checkpoint 的 config，生成结构 profile 和审计卡，不加载权重、不测吞吐，也不把 config 差异写成性能结论。Block 宽度或深度变化进入 60；涉及 KV heads 时进入 71；涉及 MoE 专家容量、overflow 或 dispatch 时进入 80；涉及 Attention 访问范围时进入 78，涉及长序列记忆时进入 77。

#### 5.1 配置模型候选与审计口径

| 条目 | 配置内容 | 记录目的 |
| --- | --- | --- |
| baseline / candidate | 两个 checkpoint ID、revision | 追溯 config 来源 |
| `changed_field` | 预期唯一变化的结构字段 | 检查是否确实是单变量候选 |
| 审计输出 | profile、ledger、audit card 与 decision | 保存可复查的结构判断 |

```python
# 默认关闭；只读取 checkpoint config，不下载或加载模型权重。
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

RUN_ARCH_CONFIG_AUDIT = os.environ.get('RUN_ARCH_CONFIG_AUDIT', '0') == '1'
ARCH_CONFIG_AUDIT = {
    'baseline_model': os.environ.get('ARCH_AUDIT_BASELINE_MODEL', ''),
    'candidate_model': os.environ.get('ARCH_AUDIT_CANDIDATE_MODEL', ''),
    'revision': os.environ.get('ARCH_AUDIT_REVISION', 'main'),
    'changed_field': os.environ.get('ARCH_AUDIT_CHANGED_FIELD', 'num_key_value_heads'),
    'existing_config_result': os.environ.get('ARCH_AUDIT_CONFIG_RESULT', ''),
    'producer_results': {
        'ARCH-BLOCK-STABILITY': os.environ.get('ARCH_EVIDENCE_BLOCK', ''),
        'ARCH-ATTENTION-ACCESS': os.environ.get('ARCH_EVIDENCE_ACCESS', ''),
        'ARCH-MLA-KV': os.environ.get('ARCH_EVIDENCE_MLA', ''),
        'ARCH-LONG-SEQUENCE-MEMORY': os.environ.get('ARCH_EVIDENCE_MEMORY', ''),
    },
}
print('Architecture config audit is', 'enabled' if RUN_ARCH_CONFIG_AUDIT else 'disabled')
print('Set two checkpoint IDs and RUN_ARCH_CONFIG_AUDIT=1 to read their configs.')
```

#### 5.2 读取 config，生成审计卡并保存 JSON

读取到的 config 会被映射为本节 profile。若真实差分不止一个字段，审计卡会返回 `reject`；这表示应拆分候选，不是模型不可用。

```python
if not RUN_ARCH_CONFIG_AUDIT:
    print('Config audit skipped. Enable RUN_ARCH_CONFIG_AUDIT after configuring model IDs.')
else:
    from transformers import AutoConfig
    missing = [key for key in ('baseline_model', 'candidate_model') if not ARCH_CONFIG_AUDIT[key]]
    if missing:
        raise ValueError(f'请先配置 checkpoint：{missing}')

    def profile_from_hf_config(model_id: str) -> Dict[str, object]:
        """读取常见 Decoder config 字段；缺失字段会在 profile 契约中暴露。"""
        config = AutoConfig.from_pretrained(model_id, revision=ARCH_CONFIG_AUDIT['revision'])
        raw = {
            'hidden_size': getattr(config, 'hidden_size', None),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
            'num_attention_heads': getattr(config, 'num_attention_heads', None),
            'num_key_value_heads': getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', None)),
            'intermediate_size': getattr(config, 'intermediate_size', None),
            'vocab_size': getattr(config, 'vocab_size', None),
            'attention_pattern': getattr(config, 'attention_pattern', 'dense'),
            'state_representation': 'recurrent_state' if 'mamba' in str(getattr(config, 'model_type', '')).lower() else 'kv_cache',
            'num_local_experts': getattr(config, 'num_local_experts', getattr(config, 'num_experts', 0)),
            'num_experts_per_tok': getattr(config, 'num_experts_per_tok', getattr(config, 'top_k', 0)),
            'capacity_factor': getattr(config, 'capacity_factor', 0.0),
        }
        profile = extract_architecture_profile(raw)
        profile['model_id'] = model_id
        profile['model_type'] = getattr(config, 'model_type', None)
        return profile

    baseline_profile = profile_from_hf_config(ARCH_CONFIG_AUDIT['baseline_model'])
    candidate_profile = profile_from_hf_config(ARCH_CONFIG_AUDIT['candidate_model'])
    candidate_profile['changed_field'] = ARCH_CONFIG_AUDIT['changed_field']
    card = build_architecture_audit_card(
        baseline_profile, candidate_profile,
        evidence={'workload': {'model_revision': ARCH_CONFIG_AUDIT['revision']}, 'evidence_level': 'checkpoint_config_audit'}
    )
    result = {
        'provenance': {'revision': ARCH_CONFIG_AUDIT['revision'], 'mode': 'checkpoint_config_audit'},
        'baseline': baseline_profile,
        'candidate': candidate_profile,
        'audit_card': card,
        'decision': recommend_audit_action(card, max_param_delta=0),
    }
    root = Path(os.environ.get('LLM_ALGO_REPO_DIR', Path.cwd()))
    result_dir = root / 'benchmarks' / 'results' / '61_architecture_exploration'
    result_dir.mkdir(parents=True, exist_ok=True)
    ARCH_CONFIG_AUDIT_PATH = result_dir / f"config_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    ARCH_CONFIG_AUDIT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    print(f'Saved config audit: {ARCH_CONFIG_AUDIT_PATH}')
```

#### 5.3 读取审计结果并进入专项证据

配置审计只确定“需要补哪类证据”。`60` 比较 Block 变体，`71` 比较 KV 表示，`77` 比较长序列记忆，`80` 提供 MoE 容量、overflow 与 dispatch 的交叉证据。这些项目各自保存 workload、quality、cost、failure 与 evidence level；61 只汇总返回记录形成选型结论。

```python
result_path = globals().get('ARCH_CONFIG_AUDIT_PATH')
if result_path is None and ARCH_CONFIG_AUDIT['existing_config_result']:
    result_path = Path(ARCH_CONFIG_AUDIT['existing_config_result'])
if result_path is None:
    print('No config audit result found. Run 5.2 after configuring model IDs.')
else:
    record = json.loads(Path(result_path).read_text(encoding='utf-8'))
    card, decision = record['audit_card'], record['decision']
    print('changed field:', card.get('changed_field'))
    print('evidence project:', card.get('required_evidence', {}).get('primary_project'))
    print(f"decision: {decision['decision']} — {decision['reason']}")
```

#### 5.4 登记专项 producer 结果

在 5.1 的 `producer_results` 填入已经生成的 architecture companion。61 不复制其质量或性能指标，只登记语义 ID、文件路径和哈希。MoE 的 80 仍属于通信与并行契约，不混入架构 producer 列表。

```python
import sys

ARCH_REPOSITORY_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents) if (path / 'tools' / 'architecture_result_schema.py').is_file()),
    None,
)
if ARCH_REPOSITORY_ROOT is None:
    raise RuntimeError('未找到 tools/architecture_result_schema.py，请从教程仓库中运行本节。')
if str(ARCH_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCH_REPOSITORY_ROOT))

from tools.architecture_result_schema import (
    decide_audit, make_audit_record, make_evidence_reference, save_record, validate_record,
    verify_evidence_reference,
)

ARCH_PROJECT_TO_SEMANTIC_ID = {
    '60_Decoder_Block_Stability_Benchmark': 'ARCH-BLOCK-STABILITY',
    '71_MLA_KV_Cache_Architecture_Benchmark': 'ARCH-MLA-KV',
    '77_Long_Sequence_Memory_Architecture_Benchmark': 'ARCH-LONG-SEQUENCE-MEMORY',
    '78_Attention_Access_Pattern_Benchmark': 'ARCH-ATTENTION-ACCESS',
}

```

#### 5.5 校验来源、哈希与证据状态

每个来源必须是有效的 producer companion，语义 ID 必须与登记项一致。缺失文件、字段错误或哈希变化都会进入问题列表；失败来源可以保留用于解释，但不能支持 `accept`。

```python
ARCH_PRODUCER_REFERENCES = []
ARCH_PRODUCER_RECORDS = {}
ARCH_REFERENCE_ISSUES = []
for semantic_id, configured_path in ARCH_CONFIG_AUDIT['producer_results'].items():
    if not configured_path:
        continue
    producer_path = Path(configured_path)
    try:
        producer_record = json.loads(producer_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        ARCH_REFERENCE_ISSUES.append(f'{semantic_id}: unreadable result: {exc}')
        continue
    record_errors = validate_record(producer_record)
    if producer_record.get('semantic_id') != semantic_id:
        record_errors.append('configured semantic_id does not match producer record')
    if producer_record.get('role') != 'producer':
        record_errors.append('record role is not producer')
    if record_errors:
        ARCH_REFERENCE_ISSUES.extend(f'{semantic_id}: {error}' for error in record_errors)
        continue
    reference = make_evidence_reference(semantic_id=semantic_id, result_path=producer_path)
    reference_errors = verify_evidence_reference(reference)
    if reference_errors:
        ARCH_REFERENCE_ISSUES.extend(f'{semantic_id}: {error}' for error in reference_errors)
        continue
    ARCH_PRODUCER_REFERENCES.append(reference)
    ARCH_PRODUCER_RECORDS[semantic_id] = producer_record

print({'valid_references': [item['semantic_id'] for item in ARCH_PRODUCER_REFERENCES]})
if ARCH_REFERENCE_ISSUES:
    print({'reference_issues': ARCH_REFERENCE_ISSUES})

```

#### 5.6 生成架构审计记录

最终判断同时检查配置差分和所需专项证据。来源缺失、producer 失败、producer 尚为 `tune/reject`，或当前改动只能由 80 等交叉项目解释时，61 都不会给出 `accept`。

```python
if result_path is None:
    print('No config audit result; architecture audit record was not created.')
else:
    config_record = json.loads(Path(result_path).read_text(encoding='utf-8'))
    config_card = config_record['audit_card']
    config_decision = config_record['decision']
    required_project = config_card.get('required_evidence', {}).get('primary_project')
    required_semantic_id = ARCH_PROJECT_TO_SEMANTIC_ID.get(required_project)
    external_required = required_project is not None and required_semantic_id is None
    producer_decisions = {
        semantic_id: record.get('decision', {}).get('status', 'pending')
        for semantic_id, record in ARCH_PRODUCER_RECORDS.items()
    }
    audit_gate = decide_audit(
        config_decision=config_decision,
        required_semantic_id=required_semantic_id,
        producer_records=ARCH_PRODUCER_RECORDS,
        reference_issues=ARCH_REFERENCE_ISSUES,
        external_required=external_required,
    )
    final_decision, final_reason = audit_gate['status'], audit_gate['reason']

    if not ARCH_PRODUCER_REFERENCES:
        print({'decision': 'tune', 'reason': '尚未登记有效的架构 producer companion'})
    else:
        audit_record = make_audit_record(
            source={
                'notebook': '02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb',
                'config_audit_path': str(result_path),
            },
            runtime={'mode': 'offline_architecture_audit'},
            workload=config_card.get('evidence', {}).get('workload') or {'model_revision': ARCH_CONFIG_AUDIT['revision']},
            baseline=config_record['baseline'],
            candidates=[config_record['candidate']],
            quality={'producer_decisions': producer_decisions},
            cost={'referenced_producer_count': len(ARCH_PRODUCER_REFERENCES)},
            references=ARCH_PRODUCER_REFERENCES,
            audit={
                'changed_field': config_card.get('changed_field'),
                'required_project': required_project,
                'required_semantic_id': required_semantic_id,
                'reference_issues': ARCH_REFERENCE_ISSUES,
                'config_decision': config_decision,
            },
            evidence_level='real_benchmark',
            decision={
                'status': final_decision,
                'reason': final_reason,
                'next_action': required_project if final_decision != 'accept' else 'register architecture decision',
            },
        )
        audit_path = Path(result_path).with_name(Path(result_path).stem + '_architecture.json')
        if audit_path.exists():
            print(f'Audit companion already exists and was not overwritten: {audit_path}')
        else:
            save_record(audit_path, audit_record)
            print(f'Saved architecture audit: {audit_path}')
        print({'decision': final_decision, 'reason': final_reason})

```

## 相关阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 的基础结构。
- [LLaMA](https://arxiv.org/abs/2302.13971)：现代 Decoder 的 RMSNorm、SwiGLU 与 RoPE 组合。
- [Hugging Face Transformers AutoConfig](https://huggingface.co/docs/transformers/main_classes/configuration)：读取 checkpoint 结构字段的官方接口。
- [60. Decoder Block Stability Benchmark](./60_Decoder_Block_Stability_Benchmark.md)：Block 变体的稳定性、质量与成本专项对照。
- [71. MLA / KV Cache Architecture Benchmark](./71_MLA_KV_Cache_Architecture_Benchmark.md)：KV 表示与容量专项对照。
- [77. Long-Sequence Memory Architecture Benchmark](./77_Long_Sequence_Memory_Architecture_Benchmark.md)：显式 KV、递推状态与混合记忆的长度阶梯对照。
- [78. Attention Access Pattern Benchmark](./78_Attention_Access_Pattern_Benchmark.md)：全局、窗口、块稀疏与选择性稀疏访问的专项证据。
- [80. MoE Expert Parallel Benchmark](./80_MoE_Expert_Parallel_Benchmark.md)：MoE 容量、overflow、dispatch 与跨卡通信证据；本节只引用其结论。