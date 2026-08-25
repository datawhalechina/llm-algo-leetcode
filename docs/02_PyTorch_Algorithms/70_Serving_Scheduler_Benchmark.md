# 70. Serving Scheduler Benchmark | 推理服务调度基准
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `推理服务`, `调度`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---
## 本节导读

本节要求你在固定请求流量和队列约束下比较两种 serving 调度策略。统一 workload 后，分别测量 TTFT、TPOT、吞吐、公平性和 worker 利用率，并观察不同请求之间的等待代价。最终输出调度方案的适用场景，而不是只按单一吞吐指标排序。
**层级定位：** 本项目主落在 L4，研究服务实例内部的 batching、队列和请求调度；副本扩缩容、跨模型路由、灰度发布和 SLA 治理属于 L5，只有在扩展实验中才继续接入。

**关键词：** `serving scheduler`, `TTFT`, `TPOT`, `throughput`, `fairness`, `utilization`

---
## 前置阅读

**导语：** 先把 decode 调度、KV cache 调度、PD 分离和前缀缓存 benchmark 理顺，再进入这个项目；本节默认你已经知道基本调度对象，重点转向调度策略是否值得保留。
- [36. Decode Scheduling | 解码调度](./36_Decode_Scheduling.md)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)
- [38. Prefill Decode Disaggregation | PD 分离](./38_Prefill_Decode_Disaggregation.md)
- [69. Prefix Caching Benchmark | 前缀缓存基准](./69_Prefix_Caching_Benchmark.md)

## 相关阅读

**导语：** 做完 serving 调度 benchmark 后，最自然的下一步是把服务结论推进到并行通信或分布式推理验证链路。
- [79. Distributed Parallel Benchmark | 分布式并行基准项目](./79_Distributed_Parallel_Benchmark.md)
- [81. Distributed Inference Logic Validation | 分布式推理逻辑验证](./81_Distributed_Inference_Project.md)

---
### Step 1: 定义 serving benchmark 目标

- 固定请求到达模式、prompt 长度分布、decode 长度分布和并发窗口。
- 明确 baseline 调度器与 candidate 调度器的优先级规则。
- 统一记录 TTFT、TPOT、throughput、公平性和 worker 利用率。
### Step 2: baseline 和请求分布先要合法

- serving benchmark 不能脱离 workload 讨论。
- 如果 baseline 的到达模式、队列深度或容量约束本身不稳定，后面的比较就没有解释空间。
- 至少要先确认 baseline 的 TTFT、TPOT 和吞吐是可复现的。
### Step 3: 用统一口径比较收益与代价

- serving 调度项目必须同时看 TTFT、TPOT、throughput、公平性和利用率，不能只挑单项吞吐收益下结论。
- `ttft_ms` 回答首 token 是否更快返回。
- `throughput_tps` 回答整体 token 吞吐是否更高。
- `fairness` 和 `utilization` 回答系统是否只是把部分请求压得更差来换吞吐。
### Step 4: 输出 benchmark 结论

- serving 调度最终不是输出“吞吐有没有涨”，而是输出这套调度策略在当前请求流量下是否值得继续保留、微调或放弃。
- 最终结论建议统一为 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回队列规则、batch 粒度和 worker 分池，而不是只盯住吞吐数字。
#### 图解：36-37-38-69 如何收束到 70 推理服务调度基准

```text
36 decode scheduling -> 37 KV cache scheduling -> 38 PD disaggregation -> 69 prefix serving baseline
                                          |
                                          v
                          70 serving scheduler benchmark + delivery decision
```
项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | workload、TTFT、TPOT、throughput、公平性 | 保证比较合法 |
| candidate | 调度规则、worker 利用率、缓存/队列收益 | 解释 serving 收益来源 |
| 对比 | 延迟变化、吞吐增益、公平性变化、利用率变化 | 判断是否值得 adopt |
| 决策 | accept / tune / reject | 输出 benchmark 结论 |

```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：workload 汇总、baseline 对比、项目判断
# 目标：把调度收益转成可比较的 benchmark 报告

def summarize_serving_scheduler_runs(runs: List[Dict[str, float]]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_scheduler_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_serving_scheduler_run(
    baseline: Dict[str, float], candidate: Dict[str, float], min_throughput_gain: float, min_fairness: float
) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_serving_scheduler_benchmark_template():
    baseline = {
        'name': 'fifo',
        'ttft_ms': 220,
        'tpot_ms': 42,
        'throughput_tps': 180,
        'fairness': 0.78,
        'utilization': 0.70,
    }
    candidate = {
        'name': 'priority_scheduler',
        'ttft_ms': 180,
        'tpot_ms': 36,
        'throughput_tps': 205,
        'fairness': 0.81,
        'utilization': 0.79,
    }
    summary = summarize_serving_scheduler_runs([baseline, candidate])
    assert summary['run_count'] == 2
    assert summary['best_latency_run'] == 'priority_scheduler'
    assert summary['avg_throughput_tps'] == 192.5

    comparison = compare_scheduler_to_baseline(baseline, candidate)
    assert comparison['ttft_delta_ms'] == -40
    assert comparison['tpot_delta_ms'] == -6
    assert comparison['throughput_gain_tps'] == 25
    assert comparison['fairness_delta'] == 0.03
    assert comparison['utilization_delta'] == 0.09

    decision = recommend_serving_scheduler_run(baseline, candidate, min_throughput_gain=20, min_fairness=0.8)
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_serving_rollout'

    weak_candidate = {
        'name': 'aggressive_batching',
        'ttft_ms': 195,
        'tpot_ms': 38,
        'throughput_tps': 202,
        'fairness': 0.76,
        'utilization': 0.82,
    }
    weak_decision = recommend_serving_scheduler_run(baseline, weak_candidate, min_throughput_gain=20, min_fairness=0.8)
    assert weak_decision['decision'] == 'tune'

    bad_candidate = {
        'name': 'overfit_scheduler',
        'ttft_ms': 260,
        'tpot_ms': 48,
        'throughput_tps': 170,
        'fairness': 0.60,
        'utilization': 0.66,
    }
    bad_decision = recommend_serving_scheduler_run(baseline, bad_candidate, min_throughput_gain=20, min_fairness=0.8)
    assert bad_decision['decision'] == 'reject'


test_serving_scheduler_benchmark_template()
print('测试通过：推理服务调度基准模板可以工作。')
```

🛑 **STOP HERE** 🛑

请先尝试自己完成代码并跑通测试。如果你在 Colab 中运行，并且暂时没有思路，再继续看下面的参考答案。
## 参考代码与解析

### 代码

```python
from typing import Dict, List


def summarize_serving_scheduler_runs(runs: List[Dict[str, float]]) -> Dict[str, object]:
    best = None
    run_count = len(runs)
    avg_throughput_tps = sum(item.get('throughput_tps', 0.0) for item in runs) / run_count if run_count else 0.0
    for item in runs:
        if best is None or item.get('ttft_ms', float('inf')) < best.get('ttft_ms', float('inf')):
            best = item
    return {'run_count': run_count, 'best_latency_run': best.get('name', 'run') if best else None, 'avg_throughput_tps': avg_throughput_tps}


def compare_scheduler_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    return {
        'ttft_delta_ms': round(candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0), 4),
        'tpot_delta_ms': round(candidate.get('tpot_ms', 0.0) - baseline.get('tpot_ms', 0.0), 4),
        'throughput_gain_tps': round(candidate.get('throughput_tps', 0.0) - baseline.get('throughput_tps', 0.0), 4),
        'fairness_delta': round(candidate.get('fairness', 0.0) - baseline.get('fairness', 0.0), 4),
        'utilization_delta': round(candidate.get('utilization', 0.0) - baseline.get('utilization', 0.0), 4),
    }


def recommend_serving_scheduler_run(
    baseline: Dict[str, float], candidate: Dict[str, float], min_throughput_gain: float, min_fairness: float
) -> Dict[str, object]:
    comparison = compare_scheduler_to_baseline(baseline, candidate)
    if (
        comparison['ttft_delta_ms'] < 0
        and comparison['tpot_delta_ms'] <= 0
        and comparison['throughput_gain_tps'] >= min_throughput_gain
        and candidate.get('fairness', 0.0) >= min_fairness
    ):
        return {
            'decision': 'accept',
            'reason': '延迟、吞吐和公平性都达标，适合进入真实 serving 验证',
            'next_action': 'promote_to_serving_rollout',
        }
    if comparison['throughput_gain_tps'] >= 0 and comparison['utilization_delta'] >= 0:
        return {
            'decision': 'tune',
            'reason': '吞吐和利用率已有改善，但公平性或延迟边界还不够稳',
            'next_action': 'refine_queue_rules_or_worker_split',
        }
    return {
        'decision': 'reject',
        'reason': 'candidate 没有形成可信的 serving 调度收益',
        'next_action': 'fallback_to_scheduler_audit',
    }
```

### 解析

这页现在按 `measure -> compare -> decide` 的最小 serving scheduler 项目闭环组织，不再只是罗列某种调度技巧的收益。

#### TODO 1

- 实现方式：先汇总 run 数量和平均吞吐，再找出 TTFT 最低的 run。
- 关键点：这里的 `best_latency_run` 用来定位哪种调度策略最值得优先回看，不等于最终项目结论。
- 项目意义：先把 workload 摘要做平，后面才能在同一请求流量和队列条件下比较调度收益。

#### TODO 2

- 实现方式：统一计算 `ttft_delta_ms`、`tpot_delta_ms`、`throughput_gain_tps`、`fairness_delta` 和 `utilization_delta`。
- 关键点：延迟类指标越低越好，其余三类指标越高越好，所以方向一定要统一。
- 项目意义：这一步把 serving scheduler 从“技巧演示”转成“收益和代价能否一起成立”的 benchmark 对比。

#### TODO 3

- 实现方式：先复用 baseline 对比结果，再按吞吐阈值、公平性边界和延迟收益输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应吞吐和利用率已有改善，但公平性、TTFT 或 TPOT 还没有一起收稳。
- 项目意义：serving 调度项目不是只追吞吐，而是判断这套调度策略值不值得继续上线、调优或回退。
### 可选 Practice-P2：真实 serving backend 结果入口

本节可以复用 vLLM / SGLang 的 OpenAI-compatible endpoint，但调度策略是否真正生效取决于 backend 的启动参数和版本。共享字段固定记录模型、backend、dtype、batch、并发与 cache policy；公平性、队列长度和 GPU 利用率等调度指标放入 `strategy_metrics`。

```python
try:
    from tools.inference_project_runtime import locate_repo_root
    REPO_ROOT = locate_repo_root()
    from tools.inference_project_runtime import (
        shared_project_config, save_project_result, start_optional_vllm,
        stop_optional_vllm, run_backend_benchmark,
    )
except ModuleNotFoundError:
    RUN_REAL_BACKEND = False
    def shared_project_config(**kwargs): return kwargs
    def save_project_result(*args, **kwargs): raise RuntimeError('需要从仓库根目录运行真实 backend 入口')

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 固定基座模型。
RESULT_PATH = 'benchmarks/results/70_scheduler.json'  # 统一结果文件。
project_config = shared_project_config(
    model=MODEL_ID, backend='vllm', dtype='auto', generated_tokens=64,
    concurrency=4, cache_policy='default',
)
print(project_config)
RUN_REAL_BACKEND = False  # 是否启动真实 backend；默认保持 CPU-first。
if RUN_REAL_BACKEND:
    server, log_path, port, selected_dtype, model_path = start_optional_vllm(
        model_id=MODEL_ID, model_source='auto', dtype='auto',
        served_model_name=MODEL_ID,
    )
    try:
        report = run_backend_benchmark(
            project='70', base_url=f'http://127.0.0.1:{port}', model=MODEL_ID,
            label='vllm-scheduler', output=RESULT_PATH, concurrency=4,
            dtype=selected_dtype, cache_policy='default',
        )
        print(report['normalized_result'])
    finally:
        stop_optional_vllm(server, log_path)
# save_project_result(RESULT_PATH, project='70', strategy='scheduler',
#     config=project_config, metrics=metrics,
#     strategy_metrics={'fairness': fairness, 'queue_wait_ms': queue_wait_ms,
#                       'gpu_utilization_pct': gpu_utilization_pct},
#     decision=decision)
```
