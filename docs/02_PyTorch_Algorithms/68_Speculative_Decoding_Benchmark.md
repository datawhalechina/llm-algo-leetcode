# 68. Speculative Decoding Benchmark | 投机解码基准
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `Speculative Decoding`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

投机解码把一次生成拆成两个协作环节：草稿模型先提出一段候选，目标模型再验证这些候选。学习时要关注两个问题：候选能连续通过多少 token，以及减少的解码轮次能否抵消额外的 draft 与 verify 成本。本节沿着 baseline、接受与回退、端到端收益逐步完成判断。

**关键词：** `acceptance rate`, `draft cost`, `verify cost`, `benchmark`

---
## 前置阅读

**导语：** 进入本节前，先理解单 token 解码如何推进，以及 draft / target 如何协作。阅读时重点观察候选提出后哪些 token 被连续接受，以及一次验证是否真的减少了目标模型的重复解码工作。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [23. Speculative Decoding | 投机解码](./23_Speculative_Decoding.md)
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
### Step 1：实验问题与比较口径

先固定模型、请求集、speculative 配置和质量门槛，明确要比较的是单轮推进效率，还是端到端生成收益。后续对照都必须沿用这组输入，才能把变化归因到 speculative 机制。

| 固定项 | 记录内容 |
|:---|:---|
| 模型 | target、draft、版本和 dtype |
| workload | prompt 分布、batch、max new tokens、temperature |
| speculative 配置 | proposal length、verify policy、acceptance 定义 |
| 质量门槛 | 输出一致性或任务指标、允许的误差 |

![投机解码 benchmark：从固定口径到项目决策](../public/02_PyTorch_Algorithms/68_speculative_benchmark_flow.svg)
### Step 2：建立 G0 baseline 与测量口径

沿用 Step1 的 workload、采样和质量门槛，先运行只包含 target 的 G0，建立后续对照所需的时间、吞吐、显存和质量基线。G0 不是可选结果；正式测量应在 warmup 后进行，并让 G0、G1、G2 使用相同的请求、重复次数和计时边界。

| 比较环节 | 运行对象或过程 | 固定条件 | 记录或目的 |
|:---|:---|:---|:---|
| G0 target-only | 仅 target 模型 | 复用 Step 1 的 workload、采样和质量门槛 | TTFT、TPOT、E2E、吞吐、P99、峰值显存、参考输出与质量结果 |
| 测量过程 | warmup 后进行正式重复测量 | G0/G1/G2 使用相同请求和计时边界 | 保证延迟与吞吐可比较 |
### Step 3：解释 acceptance 与收益来源


先比较每轮实际接受了多少 token，再结合 draft、verify 和 target 的调用成本，判断 speculative 带来的推进收益是否转化为端到端性能提升。G1 固定 workload 与采样参数对照 G0，G2 只改变 proposal length 或 draft model。下面的表格同时记录对照方式和收益指标。

| 比较项 | 它说明什么 | 用于判断 |
|:---|:---|:---|
| G1 / G2 对照 | G1 使用 draft + target；G2 只改变 proposal length 或 draft model | 控制变量，定位收益来源 |
| acceptance rate | draft 候选被接受的比例 | draft 是否足够可靠 |
| accepted tokens / round | 每轮实际推进量 | 是否减少生成轮数 |
| target forward calls | target 实际调用次数 | 是否减少主要计算 |
| draft / verify cost | 两个阶段新增的成本 | 收益是否被额外成本抵消 |
| TPOT、吞吐、质量 | 端到端变化及输出约束 | 机制收益是否转化为可接受结果 |
### Step 4：CPU 机制实现与项目决策


CPU 代码验证 proposal、verify、接受和回退；真实 backend 补充端到端延迟、吞吐和质量证据。相关字段和证据来源统一在下表核对。

| 证据类别 | 复核指标 | 验收条件或决策 |
|---|---|---|
| 证据来源 | CPU 状态转移与成本模型；backend 真实调用和时间 | 区分机制验证与真实性能证据 |
| 质量 | 输出一致性、任务指标 | 达到预设质量门槛 |
| 机制 | acceptance rate、accepted tokens / round、draft / verify cost | 收益来源可以解释 |
| 性能 | TTFT、TPOT、吞吐、P50、P99、峰值显存 | 相比 G0 有改善，尾延迟没有不可接受恶化；样本不足时仅作观察 |
| 项目决策 | 三类证据综合判断 | 同时达标则 `accept`；证据不足则 `tune`；质量不达标或性能退化则 `reject` |
| 交付记录 | baseline、candidate、对比结果和决策字段 | 保存 workload、指标、质量和证据等级，形成可复查的 benchmark 结果 |

```python
from typing import Dict, List

```


```python
# 4 个核心 TODO：逐轮验证、workload 汇总、baseline 对比、项目判断
# 目标：先在 CPU 上还原可检查的状态转移，再把结果整理成 benchmark 报告。
# CPU 练习验证 proposal/verify 的计数和成本模型；真实 acceptance rate、TPOT 和 kernel 行为属于 GPU backend 扩展。
def simulate_speculative_decode(rounds, draft_ms_per_token=1.0, verify_ms_per_round=4.0):
    """模拟多轮 draft proposal / target verify，并返回可审计的状态与成本。

    参数：
        rounds: 每轮包含 draft_tokens 与 target_tokens 的字典列表。
        draft_ms_per_token: 教学用 draft 线性成本，不是实测时间。
        verify_ms_per_round: 教学用每轮 verify 成本，不是 GPU kernel 时间。

    返回：机制指标、成本模型指标和 round_trace；不实现概率性 rejection sampling。"""
    if draft_ms_per_token < 0 or verify_ms_per_round < 0:
        raise ValueError('draft 和 verify 成本不能为负数')
    # ==========================================
    # TODO 1：只补全逐轮验证的关键状态更新；输入校验、循环结构和返回字段保持现有骨架。
    # 提示：每轮比较 draft_tokens / target_tokens 的最长相同前缀，先得到 accepted。
    # proposed_tokens = ???
    # accepted_tokens = ???
    # corrected_tokens = ???
    # round_trace = ???
    # accepted_prefix_length、rejected_token、corrected 根据 accepted 和 target_tokens 派生，保留现有字段结构。
    # 完成后再计算 acceptance_rate、effective_output_tokens、draft_cost_ms 和 verify_cost_ms；
    # rounds 为空时返回零计数摘要，不能伪造一次成功验证。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_speculative_benchmark(runs: List[Dict[str, float]]) -> Dict[str, object]:
    """汇总同一 workload 的多次运行结果，不把空输入伪装成性能结果。"""
    # TODO 2：先处理空列表；非空时汇总运行次数、平均接受率、平均吞吐和最佳 run。
    # 提示：遍历 runs，读取 name、acceptance_rate、throughput；缺失字段要明确处理。
    # run_count = ???
    # avg_acceptance_rate = ???
    # avg_throughput = ???
    # best_throughput_run = ???
    #       返回 run_count、best_throughput_run、avg_acceptance_rate、avg_throughput。
    #       缺失字段应报错或明确标记，不能当成 0 纳入均值。
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_speculative_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """计算 candidate 相对 G0 的变化，不代表真实 backend 归因。"""
    # TODO 3：补充 TTFT、吞吐、acceptance rate、verify cost 的差值，并计算 speedup。
    # 提示：从 baseline / candidate 读取 ttft_ms、throughput、acceptance_rate、verify_cost_ms。
    # ttft_delta_ms = ???
    # throughput_gain = ???
    # throughput_speedup = ???
    # verify_cost_delta = ???
    #       返回 ttft_delta_ms、throughput_gain、acceptance_rate、verify_cost_delta、throughput_speedup。
    #       throughput_speedup = candidate / baseline；baseline 吞吐必须大于 0。
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_speculative_run(baseline: Dict[str, float], candidate: Dict[str, float], min_acceptance_rate: float, quality_ok: bool = True, max_verify_cost_delta: float = 10.0) -> Dict[str, object]:
    """按吞吐、接受率、质量和验证成本给出教学决策。

    这是项目筛选模板，不是生产环境的自动调参器；门槛必须随 workload 重新校准。"""
    # TODO 4：校验两个门槛，先处理质量失败，再根据吞吐、接受率和 verify cost 输出决策。
    # 提示：使用 comparison、min_acceptance_rate、quality_ok、max_verify_cost_delta。
    # acceptance_ok = ???
    # throughput_ok = ???
    # verify_cost_ok = ???
    # decision = ???
    # next_action = ???
    #       quality_ok=False 时优先 reject；只有接受率、吞吐和 verify cost 同时过门槛才 accept。reason 随 decision 生成。
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_speculative_benchmark_template():
    rounds = [
        {'draft_tokens': [1, 2, 3], 'target_tokens': [1, 2, 9]},
        {'draft_tokens': [4, 5], 'target_tokens': [4, 5]},
    ]
    simulation = simulate_speculative_decode(rounds, draft_ms_per_token=1.0, verify_ms_per_round=4.0)
    assert simulation['rounds'] == 2, "轮数统计不正确！"
    assert simulation['proposed_tokens'] == 5, "proposal token 数统计不正确！"
    assert simulation['accepted_tokens'] == 4, "accepted token 数统计不正确！"
    assert simulation['corrected_tokens'] == 1, "分歧后的 correction token 统计不正确！"
    assert simulation['effective_output_tokens'] == 5, "有效输出 token 数统计不正确！"
    assert simulation['acceptance_rate'] == 0.8, "acceptance rate 计算不正确！"
    assert simulation['accepted_tokens_per_round'] == 2.0, "每轮接受 token 数计算不正确！"
    assert simulation['round_trace'] == [
        {'round': 0, 'proposed': 3, 'accepted_prefix_length': 2, 'accepted': 2, 'rejected_token': 9, 'corrected': 1},
        {'round': 1, 'proposed': 2, 'accepted_prefix_length': 2, 'accepted': 2, 'rejected_token': None, 'corrected': 0},
    ], "逐轮状态轨迹不正确！"
    assert simulation['mechanism_metrics']['accepted_tokens'] == 4
    assert simulation['cost_model_metrics']['verify_cost_ms'] == 8.0
    assert simulation['draft_cost_ms'] == 5.0 and simulation['verify_cost_ms'] == 8.0, "proposal / verify 成本计算不正确！"
    empty = simulate_speculative_decode([])
    assert empty['rounds'] == 0 and empty['acceptance_rate'] == 0.0, "空 rounds 应返回零计数摘要！"
    first_reject = simulate_speculative_decode([{'draft_tokens': [1, 2], 'target_tokens': [9, 2]}])
    assert first_reject['accepted_tokens'] == 0 and first_reject['corrected_tokens'] == 1, "首 token 拒绝时状态不正确！"
    try:
        simulate_speculative_decode(rounds, draft_ms_per_token=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError('非法 speculative 成本应明确拒绝！')

    baseline = {'name': 'baseline', 'ttft_ms': 120, 'throughput': 100, 'acceptance_rate': 0.0, 'verify_cost_ms': 40}
    candidate = {'name': 'spec', 'ttft_ms': 110, 'throughput': 135, 'acceptance_rate': 0.72, 'verify_cost_ms': 48}
    summary = summarize_speculative_benchmark([baseline, candidate])
    assert summary['run_count'] == 2
    assert summary['best_throughput_run'] == 'spec'
    assert summary['avg_throughput'] == 117.5
    try:
        summarize_speculative_benchmark([{'name': 'invalid', 'throughput': 1.0}])
    except KeyError:
        pass
    else:
        raise AssertionError('缺少 acceptance_rate 时应明确报错！')
    comparison = compare_speculative_to_baseline(baseline, candidate)
    assert comparison['ttft_delta_ms'] == -10
    assert comparison['throughput_gain'] == 35
    assert comparison['verify_cost_delta'] == 8
    assert comparison['throughput_speedup'] == 1.35
    decision = recommend_speculative_run(baseline, candidate, min_acceptance_rate=0.6)
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_serving_eval'
    rejected = recommend_speculative_run(baseline, candidate, min_acceptance_rate=0.6, quality_ok=False)
    assert rejected['decision'] == 'reject'
    try:
        compare_speculative_to_baseline({**baseline, 'throughput': 0}, candidate)
    except ValueError:
        pass
    else:
        raise AssertionError('baseline throughput 为 0 时应拒绝计算加速比！')
    # GPU/backend 的输出质量和实验计划由后面的可选实验区单独验证；
    # CPU TODO 1–4 测试不依赖后续 cell 才定义的 helper。


test_speculative_benchmark_template()
print('测试通过：投机解码基准模板可以工作。')

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
#### CPU 机制验证与结果解析

CPU 代码用于检查 proposal、verify、接受、分歧和回退的状态转移；`draft_cost_ms` 与 `verify_cost_ms` 是成本模型，不代表 GPU kernel 实测。运行后重点查看 `acceptance_rate`、`effective_output_tokens` 和最终决策字段。

## 参考代码与解析

### 代码

```python
def simulate_speculative_decode(rounds, draft_ms_per_token=1.0, verify_ms_per_round=4.0):
    """按轮模拟 draft proposal、target verify 和分歧后的 correction。

    参数：
        rounds: 已对齐的 draft / target token 列表；每个元素代表一轮。
        draft_ms_per_token、verify_ms_per_round: 仅用于线性成本模型。

    返回：机制计数、成本模型、逐轮 round_trace 和嵌套指标。
    边界：这是确定性的 CPU 状态转移模拟，不是 target model 的真实采样，
    也不能由此推出 vLLM / SGLang 的 kernel 延迟或线上吞吐。"""
    if draft_ms_per_token < 0 or verify_ms_per_round < 0:
        raise ValueError('draft 和 verify 成本不能为负数')
    if not isinstance(rounds, list):
        raise TypeError('rounds 必须是 list')
    proposed_tokens = accepted_tokens = corrected_tokens = 0
    accepted_rounds = 0
    round_trace = []
    for round_index, item in enumerate(rounds):
        if not isinstance(item, dict) or 'draft_tokens' not in item or 'target_tokens' not in item:
            raise ValueError(f'第 {round_index} 轮必须包含 draft_tokens 和 target_tokens')
        draft = list(item['draft_tokens'])
        target = list(item['target_tokens'])
        if not draft or not target:
            raise ValueError('每轮必须包含非空 draft_tokens 和 target_tokens')
        accepted = 0
        for draft_token, target_token in zip(draft, target):
            if draft_token != target_token:
                break
            # TODO 1 对应变量 accepted：累计当前轮连续通过的候选数量。
            accepted += 1
        # TODO 1 对应变量 proposed_tokens / accepted_tokens：分别累计提议与接受 token。
        proposed_tokens += len(draft)
        accepted_tokens += accepted
        if accepted == len(draft):
            accepted_rounds += 1
        else:
            corrected_tokens += 1
        rejected_token = target[accepted] if accepted < len(target) and accepted < len(draft) else None
        # TODO 1 对应变量 corrected_tokens：记录分歧后的 correction 数量。
        # TODO 1 对应结果 round_trace：保存接受前缀、拒绝位置和 correction 状态。
        round_trace.append({
            'round': round_index, 'proposed': len(draft),
            'accepted_prefix_length': accepted, 'accepted': accepted,
            'rejected_token': rejected_token,
            'corrected': int(accepted < len(draft)),
        })
    round_count = len(rounds)
    accepted_tokens_per_round = accepted_tokens / round_count if round_count else 0.0
    return {
        'rounds': round_count,
        'proposed_tokens': proposed_tokens,
        'accepted_tokens': accepted_tokens,
        'corrected_tokens': corrected_tokens,
        'effective_output_tokens': accepted_tokens + corrected_tokens,
        'acceptance_rate': accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
        # 这是 CPU 机制模拟的每轮推进量，不是 backend 实测值
        'accepted_tokens_per_round': accepted_tokens_per_round,
        'draft_cost_ms': proposed_tokens * draft_ms_per_token,
        'verify_cost_ms': round(round_count * verify_ms_per_round, 4),
        'fully_accepted_rounds': accepted_rounds,
        'round_trace': round_trace,
        'mechanism_metrics': {
            'proposed_tokens': proposed_tokens, 'accepted_tokens': accepted_tokens,
            'corrected_tokens': corrected_tokens, 'acceptance_rate': accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
            'accepted_tokens_per_round': accepted_tokens_per_round,
        },
        'cost_model_metrics': {
            'draft_cost_ms': proposed_tokens * draft_ms_per_token,
            'verify_cost_ms': round(round_count * verify_ms_per_round, 4),
        },
    }

# TODO 2: 汇总投机解码 workload。输入应是同一模型、同一 workload 的 run。
def summarize_speculative_benchmark(runs: List[Dict[str, float]]) -> Dict[str, object]:
    """汇总同一 workload 的多次运行。

    要求输入至少提供 throughput；acceptance_rate 缺失时应显式处理，
    不能把缺失字段静默解释为真实的 0。返回平均接受率、平均吞吐和最佳 run；
    空列表只返回空摘要，不构成性能结论。"""
    if not runs:
        return {'run_count': 0, 'best_throughput_run': None, 'avg_acceptance_rate': 0.0, 'avg_throughput': 0.0}
    required = {'name', 'acceptance_rate', 'throughput'}
    missing = [sorted(required - set(item)) for item in runs if not required.issubset(item)]
    if missing:
        raise KeyError(f'每条 run 必须包含 {sorted(required)}，缺失字段：{missing}')
    # TODO 2 对应变量 run_count：统计 workload 运行次数。
    # TODO 2 对应变量 avg_acceptance_rate / avg_throughput：计算两项均值。
    # TODO 2 对应变量 best_throughput_run：定位吞吐最高的运行。
    best = max(runs, key=lambda item: item['throughput'])
    avg_acceptance_rate = sum(item['acceptance_rate'] for item in runs) / len(runs)
    avg_throughput = sum(item['throughput'] for item in runs) / len(runs)
    return {'run_count': len(runs), 'best_throughput_run': best.get('name', 'run'),
            'avg_acceptance_rate': round(avg_acceptance_rate, 6),
            'avg_throughput': round(avg_throughput, 6)}


# TODO 3: 比较 baseline 和 speculative candidate；两者必须使用同一 workload。
def compare_speculative_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """计算 candidate 相对 G0 baseline 的指标变化。

    需要区分差值（gain / delta）与比值（speedup），并拒绝吞吐为 0 的 baseline。
    结果只表示报告层比较，不能替代真实 backend 的 TTFT、TPOT、P99 和质量校验。"""
    required = {'ttft_ms', 'throughput', 'acceptance_rate', 'verify_cost_ms'}
    for label, record in [('baseline', baseline), ('candidate', candidate)]:
        missing = sorted(required - set(record))
        if missing:
            raise KeyError(f'{label} 缺少比较字段：{missing}')
    baseline_throughput = baseline['throughput']
    if baseline_throughput <= 0:
        raise ValueError('baseline throughput 必须大于 0')
    # TODO 3 对应变量 ttft_delta_ms / throughput_gain：计算 candidate 相对 baseline 的差值。
    # TODO 3 对应变量 throughput_speedup：计算 candidate / baseline 的吞吐比值。
    # TODO 3 对应变量 acceptance_rate / verify_cost_delta：保留接受率并计算验证成本变化。
    return {
        'ttft_delta_ms': candidate['ttft_ms'] - baseline['ttft_ms'],
        'throughput_gain': candidate['throughput'] - baseline['throughput'],
        'acceptance_rate': candidate['acceptance_rate'],
        'verify_cost_delta': candidate['verify_cost_ms'] - baseline['verify_cost_ms'],
        'throughput_speedup': candidate['throughput'] / baseline_throughput,
    }


# TODO 4: 输出项目判断。阈值属于本次教学 workload 的配置，不是通用常数。
def recommend_speculative_run(baseline: Dict[str, float], candidate: Dict[str, float], min_acceptance_rate: float, quality_ok: bool = True, max_verify_cost_delta: float = 10.0) -> Dict[str, object]:
    """按吞吐、接受率、质量和验证成本给出教学决策。

    先处理质量失败，再检查吞吐收益、接受率和 verify 成本；返回
    decision / reason / next_action。该函数是可解释的筛选模板，
    不是生产调参器，也不能单凭 CPU 模拟决定启用 speculative serving。"""
    if not 0 <= min_acceptance_rate <= 1:
        raise ValueError('min_acceptance_rate 必须位于 [0, 1]')
    if max_verify_cost_delta < 0:
        raise ValueError('max_verify_cost_delta 不能为负数')
    # TODO 4 对应变量 comparison：先复用 TODO 3 的对照结果。
    # TODO 4 对应判断：依次检查 quality_ok、acceptance_rate、throughput_gain 和 verify_cost_delta。
    # TODO 4 对应分支：分别生成 decision、reason 和 next_action。
    comparison = compare_speculative_to_baseline(baseline, candidate)
    if not quality_ok:
        return {'decision': 'reject', 'reason': '质量门槛未通过', 'next_action': 'fallback_to_baseline'}
    if comparison['throughput_gain'] > 0 and comparison['acceptance_rate'] >= min_acceptance_rate and comparison['verify_cost_delta'] <= max_verify_cost_delta:
        return {'decision': 'accept', 'reason': '吞吐收益、接受率和验证成本都达标', 'next_action': 'promote_to_serving_eval'}
    if comparison['throughput_gain'] > 0 and comparison['acceptance_rate'] >= min_acceptance_rate:
        return {'decision': 'tune', 'reason': '吞吐和接受率可用，但验证成本仍偏高', 'next_action': 'refine_draft_or_verify'}
    return {'decision': 'reject', 'reason': '接受率不足或吞吐收益不明显', 'next_action': 'fallback_to_baseline'}

```

### 解析

这一页保留 `4` 个核心 TODO：逐轮状态轨迹、workload 汇总、baseline 对比和项目判断。它不要求把真实 draft model 或 GPU kernel 重写一遍，而是要求先把接受/回退机制和 benchmark 收成清晰的项目决策。

**1. TODO 1: 模拟 proposal / verify**
- **实现方式**：逐轮比较 `draft_tokens` 与 `target_tokens` 的最长相同前缀；保存 `accepted_prefix_length`、`rejected_token` 和 `round_trace`，再把结果分为机制指标与成本模型指标。
- **关键点**：`acceptance_rate = accepted_tokens / proposed_tokens`；`draft_cost_ms` 和 `verify_cost_ms` 是 CPU 成本模型，不是实测 kernel 时间；这里没有实现概率性 rejection sampling。
- **项目意义**：CPU 可以验证接受、分歧和回退的状态转移；真实 draft / target 模型、GPU kernel、TPOT 和 serving 吞吐仍需 backend 实验。
- **边界**：这里的 `effective_output_tokens` 是便于教学的“接受 token + correction token”简化计数，不等同于某个推理引擎的完整 token 生成协议。

**2. TODO 2: 汇总投机解码 workload**
- **实现方式**：空 workload 返回空摘要；非空 workload 统计 run 数、平均 acceptance rate、平均吞吐和最高吞吐 run，避免缺失字段静默变成有效结果。
- **关键点**：这一步先固定 workload 视角，后面的收益判断才不会退回成单条 run 的偶然结果。
- **项目意义**：没有 run 级摘要，就无法说明当前 speculative 配置到底是在什么 workload 下表现更好。

**3. TODO 3: 比较 baseline 和 speculative candidate**
- **实现方式**：统一比较 TTFT、吞吐、acceptance rate 和 verify cost 的变化，并计算吞吐加速比。
- **关键点**：这页现在显式补上了 `verify_cost`，不再只看吞吐和 acceptance rate。
- **项目意义**：这一步把页面从“投机解码有没有提速”推进到“提速代价是否值得保留”。

**4. TODO 4: 输出项目判断**
- **实现方式**：把 comparison 与 `quality_ok`、验证成本上限收成 `accept / tune / reject` 与下一轮动作。
- **关键点**：吞吐和 acceptance rate 可用但 verify cost 偏高时，应该走 `tune`，而不是直接 `accept`。
- **项目意义**：这一步让 `68` 真正回答“这条 speculative 链路值不值得继续采用”，而不是只给一组指标。 
### Step 5（可选）：GPU/backend 实验——真实 speculative 对照

#### 5.1 环境与固定条件

第一轮固定配置：target=`Qwen/Qwen2.5-1.5B-Instruct`，draft=`Qwen/Qwen2.5-0.5B-Instruct`，`proposal_length=5`，`concurrency=1`，`max_tokens=64`，`temperature=0`，正式请求 30–50 条、重复 3 次。显卡至少登记两档：H0 本地 12GB RTX 5070 Ti Laptop GPU，H1 24GB RTX 4090/同级 40 系算力平台；两档分别保存报告，不混合计算均值。若显存或 backend 不支持，先只运行 G0 baseline，并保留 `GPU baseline smoke` 证据等级。

![GPU speculative benchmark：从环境到项目决策](../public/02_PyTorch_Algorithms/68_speculative_gpu_flow.svg)

#### 5.2 实验条件与分组

**实验条件表**

| 项目 | H0：12GB 本地卡 | H1：24GB 算力平台 |
|---|---|---|
| GPU | RTX 5070 Ti Laptop GPU | RTX 4090/同级 40 系 |
| G0/G1/G2 | 三组均执行或标记 unsupported | 三组均执行或标记 unsupported |
| 模型与 workload | 完全固定 | 完全固定 |
| 比较原则 | 先在 H0 内比较 | 先在 H1 内比较；再观察硬件差异 |

#### 5.3 配置与结果字段

**统一项目证据字段**

68、69、70 的每条结果都保留同一组公共字段；本节再增加 acceptance、proposal 和 verify 等投机解码专属字段。

| 公共字段 | 记录内容 | 本节要求 |
|:---|:---|:---|
| `project` / `role` | 项目编号、`baseline` 或 `candidate` | G0 明确为 baseline，G1/G2 明确为 candidate 或单变量对照 |
| `workload` / `config` | 模型、backend、dtype、输入输出长度、并发、seed、repeat | baseline 与 candidate 使用同一 workload，只改变声明的实验因素 |
| `metrics` | TTFT、TPOT、throughput、P95/P99、peak memory、queue wait | 不支持的指标写 `not_available`，不能用代理值冒充实测 |
| `strategy_metrics` | acceptance / hit rate、transfer / handoff cost 等机制字段 | 本节至少记录 acceptance、accepted tokens、draft / verify cost |
| `quality` / `status` | 输出一致性、任务质量、运行状态 | 记录 `ok`、`unsupported`、`OOM` 或 `failed` 及原因 |
| `evidence_level` / `decision` | 证据等级与 `accept / tune / reject` | 失败记录和复测路径必须保留 |

**结果表模板**

| 实验组 / role | workload / config | target / draft | proposal length | acceptance rate | accepted tokens / round | target forward calls | draft / verify cost (ms) | TTFT / TPOT | throughput / P95 / P99 | peak memory / queue wait | transfer / handoff | quality / status | evidence level | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| G0 target baseline | 固定 workload / target-only | 待填写 / 不适用 | 不适用 | 不适用 | 不适用 | 待采集 | 不适用 | 待采集 | 待采集 / 待采集 / 待采集 | 待采集 / 待采集 | 不适用 | 参考输出 / ok | gpu_baseline_smoke | 待判断 |
| G1 speculative | 固定 workload / proposal 配置 | 待填写 | 5 | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 / 待采集 / 待采集 | 待采集 / 待采集 | 不适用 | 待采集 / ok | real_backend_smoke | 待判断 |
| G2 单变量对照 | 固定 workload / 单变量变化 | 待填写 | 3 或 8 | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 / 待采集 / 待采集 | 待采集 / 待采集 | 不适用 | 待采集 / ok | real_backend_smoke | 待判断 |

#### 5.4 执行与保存

先按 G0 → G1 → G2 的顺序执行，每组完成 warmup 后再进行正式重复测量。统一结果清单使用 `benchmarks/results/68_speculative_decoding.json`，分组结果使用 `68_speculative_decoding_<group>.json`；多次运行时，在文件名或报告字段中保留 seed / repeat。unsupported、OOM 或失败结果也要保存，并记录 `failure_reason` 与 `retest_path`。

#### 5.5 实测记录与证据表

CPU proposal / verify 模拟只能说明接受和回退逻辑；没有 draft model、verify 过程和真实 backend 指标时，证据等级只能记为模拟或 baseline smoke，不能宣称 speculative decoding 带来 GPU 加速。

| 检查项 | 读取内容 | 判断作用 |
|:---|:---|:---|
| 配置一致性 | model、dtype、workload、proposal length | 确认结果可比较 |
| G0/G1/G2 完整性 | 结果文件、重复记录和 unsupported 状态 | 判断实验是否完整 |
| speculative 证据 | acceptance、accepted tokens、verify cost | 确认机制确实运行 |
| 性能证据 | TTFT、TPOT、吞吐、P99、peak memory | 判断是否产生端到端收益 |
| 质量与证据等级 | output consistency、task metric、status、evidence level | 记录结果可信度 |

#### 5.6 解释结果与形成决策

先比较 G0 与 G1 的端到端收益，再用 G2 判断 proposal length 或 draft model 是否是主要影响因素。只有真实 speculative 证据、性能收益和质量结果同时成立，才输出 `accept`；证据不足为 `tune`；质量或性能不达标为 `reject`。普通 vLLM 服务不等于 speculative decoding 已启用。

如果 backend 不支持 draft model、verify 逻辑或对应启动参数，应保留 unsupported 状态，并把 strategy-specific 的 acceptance rate、verify cost 写入 `strategy_metrics`。

```python
try:
    from tools.inference_project_runtime import locate_repo_root
    REPO_ROOT = locate_repo_root()
    from tools.inference_project_runtime import (
        shared_project_config, save_project_result, start_optional_vllm,
        start_speculative_vllm, stop_optional_vllm, run_backend_benchmark,
    )
except ModuleNotFoundError:
    def shared_project_config(**kwargs): return kwargs
    def save_project_result(*args, **kwargs): raise RuntimeError('需要从仓库根目录运行真实 backend 入口')

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'  # G0/G1 的 target；第一轮正式候选。
DRAFT_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 与 target 同系列的 draft。
PROPOSAL_LENGTH = 5  # 每轮 draft 提议的 token 数；G2 再改为 3 / 8。
MIN_ACCEPTANCE_RATE = 0.6  # 教学决策门槛，不是通用生产阈值。
NUM_PROMPTS = 30  # 正式 benchmark 下限；链路 smoke 可临时使用更小值。
WARMUP = 5  # 正式测量前的预热请求数。
REPEATS = 3  # 同一配置的重复次数。
MAX_TOKENS = 64  # G0/G1/G2 必须保持一致。
TEMPERATURE = 0.0  # 确定性生成，便于比较输出质量。
TOP_P = 1.0  # 与 temperature 一起固定采样口径。
HARDWARE_PROFILE = 'auto'  # auto / H0_12GB_5070Ti_Laptop / H1_24GB_RTX4090。
EXPECTED_GPU_MEMORY_GB = {'H0_12GB_5070Ti_Laptop': 12, 'H1_24GB_RTX4090': 24}
RESULT_PATH = 'benchmarks/results/68_speculative_decoding.json'  # 统一结果文件。
project_config = shared_project_config(
    model=MODEL_ID, backend='vllm', dtype='auto', generated_tokens=MAX_TOKENS,
    cache_policy='default', draft_model=DRAFT_MODEL_ID,
    proposal_length=PROPOSAL_LENGTH, min_acceptance_rate=MIN_ACCEPTANCE_RATE,
    num_prompts=NUM_PROMPTS, warmup=WARMUP, repeats=REPEATS,
    temperature=TEMPERATURE, top_p=TOP_P, hardware_profile=HARDWARE_PROFILE,
)
print(project_config)
RUN_BACKEND_SMOKE = False  # 仅验证 baseline endpoint，不等于 speculative 已启用。
RUN_REAL_SPECULATIVE = False  # 当前 helper 不会自动启用 speculative backend。
def validate_speculative_config():
    """在启动 backend 前检查 G0/G1 的必要配置。"""
    if not MODEL_ID or not DRAFT_MODEL_ID:
        return ['target 和 draft model 都必须填写']
    if MODEL_ID == DRAFT_MODEL_ID:
        return ['target 和 draft model 不能相同']
    if PROPOSAL_LENGTH <= 0 or NUM_PROMPTS <= 0 or WARMUP < 0 or REPEATS <= 0:
        return ['proposal_length、num_prompts、repeats 必须为正数，warmup 不能为负数']
    if not 0 <= TEMPERATURE:
        return ['temperature 不能为负数']
    return []

config_errors = validate_speculative_config()
if config_errors:
    raise ValueError('speculative 配置无效：' + '；'.join(config_errors))
def build_experiment_plan():
    """生成正式采集前可复核的 G0/G1/G2 计划，不启动 backend。"""
    return [
        {'group': 'G0', 'strategy': 'target_baseline', 'enabled': False,
         'proposal_length': None, 'draft_model': None},
        {'group': 'G1', 'strategy': 'speculative', 'enabled': False,
         'proposal_length': PROPOSAL_LENGTH, 'draft_model': DRAFT_MODEL_ID},
        *[{'group': f'G2_len_{length}', 'strategy': 'speculative', 'enabled': False,
           'proposal_length': length, 'draft_model': DRAFT_MODEL_ID}
          for length in (3, 8)],
    ]

def summarize_output_quality(outputs, references=None):
    """汇总请求成功率和可选的 exact-match；不替代任务评测。"""
    outputs = list(outputs or [])
    success = [item for item in outputs if isinstance(item, str) and item.strip()]
    result = {'total': len(outputs), 'non_empty': len(success),
              'success_rate': len(success) / len(outputs) if outputs else 0.0}
    if references is not None:
        references = list(references)
        if len(references) != len(outputs):
            raise ValueError('references 与 outputs 长度必须一致')
        result['exact_match_rate'] = (
            sum(output == reference for output, reference in zip(outputs, references)) / len(outputs)
            if outputs else 0.0
        )
    return result

experiment_plan = build_experiment_plan()
print({'experiment_plan': experiment_plan})
if RUN_REAL_SPECULATIVE and not DRAFT_MODEL_ID:
    raise ValueError('真实 speculative 实验必须先配置 DRAFT_MODEL_ID。')
if RUN_REAL_SPECULATIVE:
    # 先探测 CLI 能力；不支持时保存 unsupported 报告，不加载模型。
    from tools.backend_runtime import probe_vllm_speculative_support
    capability = probe_vllm_speculative_support()
    if capability['status'] != 'supported':
        save_project_result(
            RESULT_PATH, project='68', strategy='speculative', config=project_config,
            metrics={}, quality={'status': 'unsupported', 'speculative_enabled': False},
            decision={'decision': 'tune', 'reason': '当前 vLLM CLI 未发现可识别的 speculative 参数'},
            strategy_metrics={'evidence_level': 'backend_capability_probe',
                              'capability': capability},
        )
        print({'speculative_capability': capability, 'status': 'unsupported'})
    else:
        server, log_path, port, selected_dtype, target_path, capability = start_speculative_vllm(
            target_model_id=MODEL_ID, draft_model_id=DRAFT_MODEL_ID,
            dtype='auto', proposal_length=PROPOSAL_LENGTH,
            served_model_name=MODEL_ID,
        )
        try:
            print({'speculative_capability': capability, 'port': port, 'dtype': selected_dtype})
            save_project_result(
                RESULT_PATH, project='68', strategy='speculative', config=project_config,
                metrics={}, quality={'status': 'not_evaluated', 'speculative_enabled': True},
                decision={'decision': 'tune', 'reason': 'backend 已启动，但尚未运行 G1/G2 benchmark'},
                strategy_metrics={'evidence_level': 'backend_started_no_metrics',
                                  'capability': capability},
            )
        finally:
            stop_optional_vllm(server, log_path)
if RUN_BACKEND_SMOKE:
    server, log_path, port, selected_dtype, model_path = start_optional_vllm(
        model_id=MODEL_ID, model_source='auto', dtype='auto',
        served_model_name=MODEL_ID,
    )
    try:
        report = run_backend_benchmark(
            project='68', base_url=f'http://127.0.0.1:{port}', model=MODEL_ID,
            label='vllm-baseline-for-speculative',
            output='benchmarks/results/68_backend_smoke.json',
            dtype=selected_dtype,
        )
        normalized = report.get('normalized_result', {})
        save_project_result(
            'benchmarks/results/68_speculative_decoding_target_baseline.json',
            project='68', strategy='target_baseline', config=project_config,
            metrics=normalized.get('metrics', report.get('metrics', {})),
            quality={'status': 'reference_only', 'speculative_enabled': False},
            decision={'decision': 'measure', 'reason': 'G0 target baseline only'},
            strategy_metrics={'speculative_enabled': False, 'evidence_level': 'gpu_baseline_smoke'},
        )
    finally:
        stop_optional_vllm(server, log_path)
# G1 接入真实 draft/target adapter 后，必须提供真实 metrics 再保存；strategy_metrics
# 至少记录 acceptance_rate、accepted_tokens_per_round、target_forward_calls、
# draft_cost_ms 和 verify_cost_ms，不能只保存 TTFT 或吞吐：
# save_project_result(RESULT_PATH, project='68', strategy='speculative',
#     config=project_config, metrics=metrics, quality=quality,
#     strategy_metrics={'acceptance_rate': acceptance_rate,
#                       'accepted_tokens_per_round': accepted_tokens_per_round,
#                       'target_forward_calls': target_forward_calls,
#                       'draft_cost_ms': draft_cost_ms,
#                       'verify_cost_ms': verify_cost_ms},
#     decision=decision)
```

## 相关阅读

完成本节 benchmark 后，可以继续比较缓存复用和请求调度；跨项目比较时，沿用相同的 workload、延迟、吞吐和质量记录方式。下面的论文与开源文档用于对照接受规则、真实 backend 和工程实现。
- [69. Prefix Caching Benchmark | 前缀缓存基准](./69_Prefix_Caching_Benchmark.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)
- [Fast Inference from Transformers via Speculative Decoding（论文）](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling（论文）](https://arxiv.org/abs/2302.01318)
- [vLLM Speculative Decoding（开源实现文档）](https://docs.vllm.ai/en/latest/examples/features/speculative_decoding/)
- [SGLang Speculative Decoding（开源实现文档）](https://docs.sglang.ai/advanced_features/speculative_decoding.html)