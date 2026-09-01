# 68. Speculative Decoding Benchmark | 推测解码基准
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `Speculative Decoding`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节要求你评估推测解码在固定 workload 下是否值得保留。CPU 部分验证 proposal / verify 的接受与回退逻辑；可选 GPU 部分才比较真实 draft + target backend 的 TTFT、TPOT、吞吐、acceptance rate 和质量。普通 target 服务只能作为 baseline，不能当作 speculative 已启用。
**主责与复用边界：** 本项目主责是 draft / verify 的接受率和端到端收益；66 只复用统一 workload 和性能指标，显存优化只观察峰值显存与 KV Cache 代价，不在本项目内判断 serving 调度或量化收益。

**关键词：** `acceptance rate`, `draft cost`, `verify cost`, `benchmark`

---
## 前置阅读

**导语：** 先把解码策略、推测解码机制和基础推理对比理顺，再进入这个 benchmark；本节默认你已经知道 draft / verify 的基本链路，重点转向这条链路是否值得保留。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [23. Speculative Decoding | 推测解码](./23_Speculative_Decoding.md)
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)

## 相关阅读

**导语：** 做完推测解码 benchmark 后，最自然的下一步是继续比较缓存与调度收益，或把结论推进到更完整的 serving 链路。
- [69. Prefix Caching Benchmark | 前缀缓存基准](./69_Prefix_Caching_Benchmark.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)

### Step 1–4：实验计划

先按以下顺序执行：定义口径 → 建立 G0 baseline → 运行 CPU 机制实验 → 汇总指标并决策；下面的 Step 5 是 CPU 执行区，Step 6 是可选 GPU/backend 验证区。

| 固定项 | 记录内容 |
|:---|:---|
| 模型 | target、draft、版本和 dtype |
| workload | prompt 分布、batch、max new tokens、temperature |
| speculative 配置 | proposal length、verify policy、acceptance 定义 |
| 质量门槛 | 输出一致性或任务指标、允许的误差 |

### Step 2：建立 baseline

- G0 只运行 target，记录 TTFT、TPOT、E2E、吞吐、P99、峰值显存和质量。
- G1 才运行 draft + target；两组必须使用同一 workload 和采样参数。
- 普通 vLLM endpoint 只能证明服务链路可用，不能产生 acceptance rate。

### Step 3：解释收益来源

- 同时检查 acceptance rate、draft cost、verify cost、TPOT、吞吐和质量。
- acceptance rate 低或 verify 成本高时，吞吐收益不能直接归因于 speculative。
- G2 一次只改变 proposal length 或 draft model，避免多个变量混合。

### Step 4：输出项目决策

- 只有吞吐/延迟改善、质量达标且 acceptance 与 verify 成本可解释时，才可 `accept`。
- 证据不足为 `tune`；质量不达标或性能退化为 `reject`。
- 没有真实 draft + target backend 时，结论只能停留在 CPU 机制验证。
#### 图解：20-24 如何收束到 68 推测解码基准

```text
20 FlashAttention -> 21 Decoding -> 23 Speculative
                                      ↓
                          68 Speculative Benchmark -> 66 Inference Comparison
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | TTFT、TPOT、throughput、P99、质量 | 保证比较合法 |
| candidate | acceptance rate、draft cost、verify cost | 解释收益来源 |
| 对比 | 吞吐增益、延迟变化、显存和质量 | 判断是否真的划算 |
| 决策 | accept / tune / reject | 输出 benchmark 结论 |


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
    # TODO 0：逐轮执行 verify，并保留每一轮的可审计轨迹。
    # 要求：校验每轮字段；比较最长相同前缀；记录 accepted_prefix_length、rejected_token；
    #       分歧后计入一个 correction token，并区分 mechanism_metrics / cost_model_metrics。
    # 变量提示：使用 proposed_tokens、accepted_tokens、corrected_tokens、accepted_rounds、round_trace；
    # proposed_tokens = ???；accepted_tokens = ???；corrected_tokens = ???；round_trace = ???。
    #       每轮结果至少包含 round、proposed、accepted_prefix_length、rejected_token、corrected。
    #       最终返回 acceptance_rate、effective_output_tokens、draft_cost_ms、verify_cost_ms；
    #       rounds 为空时返回可解释的零计数摘要，不能伪造一次成功验证。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_speculative_benchmark(runs: List[Dict[str, float]]) -> Dict[str, object]:
    """汇总同一 workload 的多次运行结果，不把空输入伪装成性能结果。"""
    # TODO 1：先处理空列表；非空时汇总 run_count、平均 acceptance_rate、
    #       平均吞吐和最高吞吐 run。不要把缺失字段静默当成实测 0。
    # 变量提示：遍历 runs；读取 name、acceptance_rate、throughput；
    # run_count = ???；avg_acceptance_rate = ???；avg_throughput = ???；best_throughput_run = ???。
    #       返回 run_count、best_throughput_run、avg_acceptance_rate、avg_throughput。
    #       缺失字段应报错或明确标记，不能当成 0 纳入均值。
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_speculative_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """计算 candidate 相对 G0 的变化，不代表真实 backend 归因。"""
    # TODO 2：补充 TTFT、吞吐、acceptance rate、verify cost 的差值，
    #       并计算 throughput_speedup；baseline 吞吐为 0 时应明确报错。
    # 变量提示：从 baseline / candidate 读取 ttft_ms、throughput、acceptance_rate、verify_cost_ms；
    # ttft_delta_ms = ???；throughput_gain = ???；throughput_speedup = ???。
    #       返回 ttft_delta_ms、throughput_gain、acceptance_rate、verify_cost_delta、throughput_speedup。
    #       throughput_speedup = candidate / baseline；baseline 吞吐必须大于 0。
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_speculative_run(baseline: Dict[str, float], candidate: Dict[str, float], min_acceptance_rate: float, quality_ok: bool = True, max_verify_cost_delta: float = 10.0) -> Dict[str, object]:
    """按吞吐、接受率、质量和验证成本给出教学决策。

    这是项目筛选模板，不是生产环境的自动调参器；门槛必须随 workload 重新校准。"""
    # TODO 3：校验两个门槛，先处理质量失败，再根据吞吐、接受率和
    #       verify cost 输出 decision、reason、next_action。
    # 变量提示：使用 comparison、min_acceptance_rate、quality_ok、max_verify_cost_delta；
    # acceptance_ok = ???；throughput_ok = ???；verify_cost_ok = ???；decision = ???。
    #       decision 只能是 accept / tune / reject，并给出对应的下一步动作。
    #       quality_ok=False 时优先 reject；只有接受率、吞吐和 verify cost 同时过门槛才 accept。
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
    assert simulation['round_trace'] == [
        {'round': 0, 'proposed': 3, 'accepted_prefix_length': 2, 'accepted': 2, 'rejected_token': 9, 'corrected': 1},
        {'round': 1, 'proposed': 2, 'accepted_prefix_length': 2, 'accepted': 2, 'rejected_token': None, 'corrected': 0},
    ], "逐轮状态轨迹不正确！"
    assert simulation['mechanism_metrics']['accepted_tokens'] == 4
    assert simulation['cost_model_metrics']['verify_cost_ms'] == 8.0
    assert simulation['draft_cost_ms'] == 5.0 and simulation['verify_cost_ms'] == 8.0, "proposal / verify 成本计算不正确！"
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
    quality = summarize_output_quality(['ok', '', 'answer'], ['ok', '', 'other'])
    assert quality['total'] == 3 and quality['success_rate'] == 2 / 3
    assert quality['exact_match_rate'] == 2 / 3
    plan = build_experiment_plan()
    assert [item['group'] for item in plan] == ['G0', 'G1', 'G2_len_3', 'G2_len_8']


test_speculative_benchmark_template()
print('测试通过：推测解码基准模板可以工作。')

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

### Step 5：CPU 实验——proposal / verify 机制与决策模板

CPU 代码只验证 proposal、verify、接受、分歧和回退的状态转移；`draft_cost_ms` 与 `verify_cost_ms` 是成本模型，不是 GPU kernel 实测。运行后应至少得到 `acceptance_rate`、`effective_output_tokens` 和 `accept / tune / reject`。

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
            accepted += 1
        proposed_tokens += len(draft)
        accepted_tokens += accepted
        if accepted == len(draft):
            accepted_rounds += 1
        else:
            corrected_tokens += 1
        rejected_token = target[accepted] if accepted < len(target) and accepted < len(draft) else None
        round_trace.append({
            'round': round_index, 'proposed': len(draft),
            'accepted_prefix_length': accepted, 'accepted': accepted,
            'rejected_token': rejected_token,
            'corrected': int(accepted < len(draft)),
        })
    round_count = len(rounds)
    return {
        'rounds': round_count,
        'proposed_tokens': proposed_tokens,
        'accepted_tokens': accepted_tokens,
        'corrected_tokens': corrected_tokens,
        'effective_output_tokens': accepted_tokens + corrected_tokens,
        'acceptance_rate': accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
        'draft_cost_ms': proposed_tokens * draft_ms_per_token,
        'verify_cost_ms': round(round_count * verify_ms_per_round, 4),
        'fully_accepted_rounds': accepted_rounds,
        'round_trace': round_trace,
        'mechanism_metrics': {
            'proposed_tokens': proposed_tokens, 'accepted_tokens': accepted_tokens,
            'corrected_tokens': corrected_tokens, 'acceptance_rate': accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
        },
        'cost_model_metrics': {
            'draft_cost_ms': proposed_tokens * draft_ms_per_token,
            'verify_cost_ms': round(round_count * verify_ms_per_round, 4),
        },
    }

# TODO 1: 汇总推测解码 workload。输入应是同一模型、同一 workload 的 run。
def summarize_speculative_benchmark(runs: List[Dict[str, float]]) -> Dict[str, object]:
    """汇总同一 workload 的多次运行。

    要求输入至少提供 throughput；acceptance_rate 缺失时应显式处理，
    不能把缺失字段静默解释为真实的 0。返回平均接受率、平均吞吐和最佳 run；
    空列表只返回空摘要，不构成性能结论。"""
    if not runs:
        return {'run_count': 0, 'best_throughput_run': None, 'avg_acceptance_rate': 0.0, 'avg_throughput': 0.0}
    best = max(runs, key=lambda item: item.get('throughput', 0.0))
    avg_acceptance_rate = sum(item.get('acceptance_rate', 0.0) for item in runs) / len(runs)
    avg_throughput = sum(item.get('throughput', 0.0) for item in runs) / len(runs)
    return {'run_count': len(runs), 'best_throughput_run': best.get('name', 'run'),
            'avg_acceptance_rate': round(avg_acceptance_rate, 6),
            'avg_throughput': round(avg_throughput, 6)}


# TODO 2: 比较 baseline 和 speculative candidate；两者必须使用同一 workload。
def compare_speculative_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """计算 candidate 相对 G0 baseline 的指标变化。

    需要区分差值（gain / delta）与比值（speedup），并拒绝吞吐为 0 的 baseline。
    结果只表示报告层比较，不能替代真实 backend 的 TTFT、TPOT、P99 和质量校验。"""
    baseline_throughput = baseline.get('throughput', 0.0)
    if baseline_throughput <= 0:
        raise ValueError('baseline throughput 必须大于 0')
    return {
        'ttft_delta_ms': candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0),
        'throughput_gain': candidate.get('throughput', 0.0) - baseline.get('throughput', 0.0),
        'acceptance_rate': candidate.get('acceptance_rate', 0.0),
        'verify_cost_delta': candidate.get('verify_cost_ms', 0.0) - baseline.get('verify_cost_ms', 0.0),
        'throughput_speedup': candidate.get('throughput', 0.0) / baseline_throughput,
    }


# TODO 3: 输出项目判断。阈值属于本次教学 workload 的配置，不是通用常数。
def recommend_speculative_run(baseline: Dict[str, float], candidate: Dict[str, float], min_acceptance_rate: float, quality_ok: bool = True, max_verify_cost_delta: float = 10.0) -> Dict[str, object]:
    """按吞吐、接受率、质量和验证成本给出教学决策。

    先处理质量失败，再检查吞吐收益、接受率和 verify 成本；返回
    decision / reason / next_action。该函数是可解释的筛选模板，
    不是生产调参器，也不能单凭 CPU 模拟决定启用 speculative serving。"""
    if not 0 <= min_acceptance_rate <= 1:
        raise ValueError('min_acceptance_rate 必须位于 [0, 1]')
    if max_verify_cost_delta < 0:
        raise ValueError('max_verify_cost_delta 不能为负数')
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

**0. TODO 0: 模拟 proposal / verify**
- **实现方式**：逐轮比较 `draft_tokens` 与 `target_tokens` 的最长相同前缀；保存 `accepted_prefix_length`、`rejected_token` 和 `round_trace`，再把结果分为机制指标与成本模型指标。
- **关键点**：`acceptance_rate = accepted_tokens / proposed_tokens`；`draft_cost_ms` 和 `verify_cost_ms` 是 CPU 成本模型，不是实测 kernel 时间；这里没有实现概率性 rejection sampling。
- **项目意义**：CPU 可以验证接受、分歧和回退的状态转移；真实 draft / target 模型、GPU kernel、TPOT 和 serving 吞吐仍需 backend 实验。
- **边界**：这里的 `effective_output_tokens` 是便于教学的“接受 token + correction token”简化计数，不等同于某个推理引擎的完整 token 生成协议。

**1. TODO 1: 汇总推测解码 workload**
- **实现方式**：空 workload 返回空摘要；非空 workload 统计 run 数、平均 acceptance rate、平均吞吐和最高吞吐 run，避免缺失字段静默变成有效结果。
- **关键点**：这一步先固定 workload 视角，后面的收益判断才不会退回成单条 run 的偶然结果。
- **项目意义**：没有 run 级摘要，就无法说明当前 speculative 配置到底是在什么 workload 下表现更好。

**2. TODO 2: 比较 baseline 和 speculative candidate**
- **实现方式**：统一比较 TTFT、吞吐、acceptance rate 和 verify cost 的变化，并计算吞吐加速比。
- **关键点**：这页现在显式补上了 `verify_cost`，不再只看吞吐和 acceptance rate。
- **项目意义**：这一步把页面从“推测解码有没有提速”推进到“提速代价是否值得保留”。

**3. TODO 3: 输出项目判断**
- **实现方式**：把 comparison 与 `quality_ok`、验证成本上限收成 `accept / tune / reject` 与下一轮动作。
- **关键点**：吞吐和 acceptance rate 可用但 verify cost 偏高时，应该走 `tune`，而不是直接 `accept`。
- **项目意义**：这一步让 `68` 真正回答“这条 speculative 链路值不值得继续采用”，而不是只给一组指标。 
### Step 6（可选）：GPU/backend 实验——真实 speculative 对照

第一轮固定配置：target=`Qwen/Qwen2.5-1.5B-Instruct`，draft=`Qwen/Qwen2.5-0.5B-Instruct`，`proposal_length=5`，`concurrency=1`，`max_tokens=64`，`temperature=0`，正式请求 30–50 条、重复 3 次。显卡至少登记两档：H0 本地 12GB RTX 5070 Ti Laptop GPU，H1 24GB RTX 4090/同级 40 系算力平台；两档分别保存报告，不混合计算均值。若显存或 backend 不支持，先只运行 G0 baseline，并保留 `GPU baseline smoke` 证据等级。

**实验条件表**

| 项目 | H0：12GB 本地卡 | H1：24GB 算力平台 |
|---|---|---|
| GPU | RTX 5070 Ti Laptop GPU | RTX 4090/同级 40 系 |
| G0/G1/G2 | 三组均执行或标记 unsupported | 三组均执行或标记 unsupported |
| 模型与 workload | 完全固定 | 完全固定 |
| 比较原则 | 先在 H0 内比较 | 先在 H1 内比较；再观察硬件差异 |

**结果表模板**

| 实验组 | acceptance rate | draft/verify cost | TTFT / TPOT | E2E / throughput / P99 | peak memory | quality | decision |
|---|---:|---:|---:|---:|---:|---|---|
| G0 target baseline | 不适用 | 不适用 | 待采集 | 待采集 | 待采集 | 参考输出 | 待判断 |
| G1 speculative | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 | 待判断 |
| G2 单变量对照 | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 | 待判断 |

建议文件名分别使用 `68_speculative_H0_12GB.json` 与 `68_speculative_H1_24GB.json`；若使用多次重复运行，可在文件名或报告字段中保留 seed / repeat。24GB 平台不是 12GB 平台的替代品，而是第二个可复现实验条件。

**证据边界**：CPU proposal / verify 模拟只能说明接受和回退逻辑；没有 draft model、verify 过程和真实 backend 指标时，不能宣称 speculative decoding 带来 GPU 加速。
按 66 的统一分组执行：G0 是 target baseline，G1 是 draft + target 的真实 speculative candidate，G2 才改变 draft 长度或模型组合。普通 vLLM 服务不等于 speculative decoding 已启用。

本节的 speculative candidate 需要 backend 同时提供 draft model、verify 逻辑或对应启动参数，因此共享 helper 只自动处理模型下载、dtype、空闲端口和结果保存；它不会假装普通 vLLM 服务已经完成 speculative 实验。没有这些能力时，使用本节的本地/模拟 benchmark，并把 strategy-specific 的 acceptance rate、verify cost 写入 `strategy_metrics`。

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
    server, log_path, port, selected_dtype, target_path, capability = start_speculative_vllm(
        target_model_id=MODEL_ID, draft_model_id=DRAFT_MODEL_ID,
        dtype='auto', proposal_length=PROPOSAL_LENGTH,
        served_model_name=MODEL_ID,
    )
    try:
        print({'speculative_capability': capability, 'port': port, 'dtype': selected_dtype})
        # 真实 benchmark 应在这里运行 G1/G2，再保存统一报告。
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
            'benchmarks/results/68_target_baseline.json',
            project='68', strategy='target_baseline', config=project_config,
            metrics=normalized.get('metrics', report.get('metrics', {})),
            quality={'status': 'reference_only', 'speculative_enabled': False},
            decision={'decision': 'measure', 'reason': 'G0 target baseline only'},
            strategy_metrics={'speculative_enabled': False, 'evidence_level': 'gpu_baseline_smoke'},
        )
    finally:
        stop_optional_vllm(server, log_path)
# G1 接入真实 draft/target adapter 后，必须提供真实 metrics 再保存：
# save_project_result(RESULT_PATH, project='68', strategy='speculative',
#     config=project_config, metrics=metrics, quality=quality,
#     strategy_metrics={'acceptance_rate': acceptance_rate, 'verify_cost_ms': verify_cost_ms},
#     decision=decision)
```
