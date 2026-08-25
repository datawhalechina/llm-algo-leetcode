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

本节要求你评估推测解码在固定 workload 下是否真的带来可用收益。除了比较 TTFT 和吞吐，还要记录 acceptance rate、draft cost 与 verify cost，并检查输出质量是否满足约束。最终给出是否值得继续调参或进入上线验证的 benchmark 结论。

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

### Step 1: 定义推测解码 benchmark 目标

- 固定模型、prompt 分布、batch size、max new tokens 和解码温度。
- 明确 candidate 的 draft model、verify policy 和 acceptance 统计口径。
- 先把质量约束写清楚，再比较吞吐和延迟。

### Step 2: 先确认 baseline 和质量口径合法

- baseline 至少要先跑通，并记录 TTFT、throughput 和质量约束，保证后续 speculative 方案有稳定参照。
- acceptance rate、draft cost、verify cost 和最终吞吐必须来自同一套 workload，不能把不同 prompt 分布或不同质量门槛的结果拼在一起比较。
- 如果 baseline 自己波动很大，推测解码收益就没有解释空间。

### Step 3: 用统一口径比较收益与代价

- 推测解码项目必须同时看 acceptance rate、draft cost、verify cost 和最终吞吐，不能只挑单项速度收益下结论。
- 如果 acceptance 低，吞吐提升很可能只是偶然 workload 下的结果。
- 如果 verify 太贵，推测链路即使接受率高，也不一定值得保留。

### Step 4: 输出 benchmark 结论

- 推测解码最终不是输出“吞吐有没有涨”，而是输出这条 speculative 链路在当前 workload 下是否值得继续保留、微调或放弃。
- 最终决策建议统一成 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回 draft model 大小、proposal 长度和 verify 策略。 
#### 图解：20-24 如何收束到 68 推测解码基准

```text
20 FlashAttention -> 21 Decoding -> 23 Speculative -> 66 Inference compare -> 68 Benchmark
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | TTFT、throughput、质量约束 | 保证比较合法 |
| candidate | acceptance rate、draft cost、verify cost | 解释收益来源 |
| 对比 | 吞吐增益、延迟变化、验证成本 | 判断是否真的划算 |
| 决策 | accept / tune / reject | 输出 benchmark 结论 |


```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：workload 汇总、baseline 对比、项目判断
# 目标：把推测解码结果整理成可比较的 benchmark 报告，而不是只看吞吐单指标

def summarize_speculative_benchmark(runs: List[Dict[str, float]]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_speculative_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_speculative_run(baseline: Dict[str, float], candidate: Dict[str, float], min_acceptance_rate: float) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_speculative_benchmark_template():
    baseline = {'name': 'baseline', 'ttft_ms': 120, 'throughput': 100, 'acceptance_rate': 0.0, 'verify_cost_ms': 40}
    candidate = {'name': 'spec', 'ttft_ms': 110, 'throughput': 135, 'acceptance_rate': 0.72, 'verify_cost_ms': 48}
    summary = summarize_speculative_benchmark([baseline, candidate])
    assert summary['run_count'] == 2
    assert summary['best_throughput_run'] == 'spec'
    comparison = compare_speculative_to_baseline(baseline, candidate)
    assert comparison['ttft_delta_ms'] == -10
    assert comparison['throughput_gain'] == 35
    assert comparison['verify_cost_delta'] == 8
    decision = recommend_speculative_run(baseline, candidate, min_acceptance_rate=0.6)
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_serving_eval'


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

## 参考代码与解析

### 代码


```python
# TODO 1: 汇总推测解码 workload
def summarize_speculative_benchmark(runs: List[Dict[str, float]]) -> Dict[str, object]:
    best = max(runs, key=lambda item: item.get('throughput', 0.0))
    avg_acceptance_rate = sum(item.get('acceptance_rate', 0.0) for item in runs) / len(runs) if runs else 0.0
    return {'run_count': len(runs), 'best_throughput_run': best.get('name', 'run'), 'avg_acceptance_rate': avg_acceptance_rate}


# TODO 2: 比较 baseline 和 speculative candidate
def compare_speculative_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    return {
        'ttft_delta_ms': candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0),
        'throughput_gain': candidate.get('throughput', 0.0) - baseline.get('throughput', 0.0),
        'acceptance_rate': candidate.get('acceptance_rate', 0.0),
        'verify_cost_delta': candidate.get('verify_cost_ms', 0.0) - baseline.get('verify_cost_ms', 0.0),
    }


# TODO 3: 输出项目判断
def recommend_speculative_run(baseline: Dict[str, float], candidate: Dict[str, float], min_acceptance_rate: float) -> Dict[str, object]:
    comparison = compare_speculative_to_baseline(baseline, candidate)
    if comparison['throughput_gain'] > 0 and comparison['acceptance_rate'] >= min_acceptance_rate and comparison['verify_cost_delta'] <= 10:
        return {'decision': 'accept', 'reason': '吞吐收益、接受率和验证成本都达标', 'next_action': 'promote_to_serving_eval'}
    if comparison['throughput_gain'] > 0 and comparison['acceptance_rate'] >= min_acceptance_rate:
        return {'decision': 'tune', 'reason': '吞吐和接受率可用，但验证成本仍偏高', 'next_action': 'refine_draft_or_verify'}
    return {'decision': 'reject', 'reason': '接受率不足或吞吐收益不明显', 'next_action': 'fallback_to_baseline'}

```

### 解析

这一页保留 `3` 个核心 TODO：workload 汇总、baseline 对比和项目判断。它不要求把 speculative decoding 的实现细节重写一遍，而是要求把 benchmark 收成清晰的项目决策。

**1. TODO 1: 汇总推测解码 workload**
- **实现方式**：统计 run 数、最高吞吐 run 和平均 acceptance rate。
- **关键点**：这一步先固定 workload 视角，后面的收益判断才不会退回成单条 run 的偶然结果。
- **项目意义**：没有 run 级摘要，就无法说明当前 speculative 配置到底是在什么 workload 下表现更好。

**2. TODO 2: 比较 baseline 和 speculative candidate**
- **实现方式**：统一比较 TTFT、吞吐、acceptance rate 和 verify cost 的变化。
- **关键点**：这页现在显式补上了 `verify_cost`，不再只看吞吐和 acceptance rate。
- **项目意义**：这一步把页面从“推测解码有没有提速”推进到“提速代价是否值得保留”。

**3. TODO 3: 输出项目判断**
- **实现方式**：把 comparison 收成 `accept / tune / reject` 与下一轮动作。
- **关键点**：吞吐和 acceptance rate 可用但 verify cost 偏高时，应该走 `tune`，而不是直接 `accept`。
- **项目意义**：这一步让 `68` 真正回答“这条 speculative 链路值不值得继续采用”，而不是只给一组指标。 
### 可选 Practice-P2：真实 backend 结果入口

本节的 speculative candidate 需要 backend 同时提供 draft model、verify 逻辑或对应启动参数，因此共享 helper 只自动处理模型下载、dtype、空闲端口和结果保存；它不会假装普通 vLLM 服务已经完成 speculative 实验。没有这些能力时，使用本节的本地/模拟 benchmark，并把 strategy-specific 的 acceptance rate、verify cost 写入 `strategy_metrics`。

```python
try:
    from tools.inference_project_runtime import locate_repo_root
    REPO_ROOT = locate_repo_root()
    from tools.inference_project_runtime import (
        shared_project_config, save_project_result, start_optional_vllm,
        stop_optional_vllm, run_backend_benchmark,
    )
except ModuleNotFoundError:
    def shared_project_config(**kwargs): return kwargs
    def save_project_result(*args, **kwargs): raise RuntimeError('需要从仓库根目录运行真实 backend 入口')

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # target 模型；正式实验还需配置 draft 模型。
RESULT_PATH = 'benchmarks/results/68_speculative_decoding.json'  # 统一结果文件。
project_config = shared_project_config(
    model=MODEL_ID, backend='vllm', dtype='auto', generated_tokens=64,
    cache_policy='default', draft_model=None,
)
print(project_config)
RUN_BACKEND_SMOKE = False  # 仅验证 baseline endpoint，不等于 speculative 已启用。
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
        print(report['normalized_result'])
    finally:
        stop_optional_vllm(server, log_path)
# save_project_result(RESULT_PATH, project='68', strategy='speculative',
#     config=project_config, metrics=metrics,
#     strategy_metrics={'acceptance_rate': acceptance_rate, 'verify_cost_ms': verify_cost_ms},
#     decision=decision)
```
