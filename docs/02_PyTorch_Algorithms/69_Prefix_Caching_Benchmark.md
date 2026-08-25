# 69. Prefix Caching Benchmark | 前缀缓存基准
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `Prefix Cache`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节要求你根据真实请求的复用模式，判断前缀缓存是否值得部署。先固定请求分布、chunk 策略和 baseline，再比较 hit rate、TTFT、prefill 节省量与缓存维护开销。最终说明缓存收益出现在哪类请求上，以及什么情况下不应启用它。

**关键词：** `prefix cache`, `hit rate`, `TTFT`, `overhead`, `deployment`

---
## 前置阅读

**导语：** 先把分页注意力、RadixAttention、前缀缓存机制和基础推理 benchmark 理顺，再进入这个项目；本节默认你已经知道缓存命中的基本机制，重点转向缓存策略是否值得保留。
- [22. vLLM PagedAttention | vLLM PagedAttention](./22_vLLM_PagedAttention.md)
- [24. SGLang RadixAttention | SGLang RadixAttention](./24_SGLang_RadixAttention.md)
- [34. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](./34_Prefix_Caching_and_Chunked_Prefill.md)
- [68. Speculative Decoding Benchmark | 推测解码基准](./68_Speculative_Decoding_Benchmark.md)

## 相关阅读

**导语：** 做完前缀缓存 benchmark 后，最自然的下一步是把缓存结论推进到更完整的 serving 调度链路。
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)

### Step 1: 定义前缀缓存 benchmark 目标

- 固定 prompt 分布、请求重用模式、batch size 和 max context length。
- 明确缓存策略、chunk size 和命中统计口径。
- 先看命中行为，再看端到端延迟收益。

### Step 2: baseline 和请求分布先要合法

- 前缀缓存 benchmark 不能脱离请求分布单独讨论。
- 如果 baseline 的请求重用模式本身不稳定，命中率数字就没有解释空间。
- 至少要先确认 baseline 的 TTFT 和复用模式是可复现的。

### Step 3: 用统一口径比较收益与代价

- 前缀缓存项目必须同时看 hit rate、TTFT 和维护开销，不能只挑命中率下结论。
- 但命中率本身不是目标，目标是命中以后能否稳定减少重复 prefill 成本。
- 如果命中率上来了，但维护开销同样显著上升，候选通常只能进入 `tune`，而不是直接 `accept`。

### Step 4: 输出 benchmark 结论

- 前缀缓存最终不是输出“命中率有没有涨”，而是输出这套缓存策略在当前请求分布下是否值得继续保留、微调或放弃。
- 最终结论建议统一为 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回调请求聚类方式、chunk 粒度和缓存失效策略，而不是盲目继续堆缓存容量。
#### 图解：22-24-34 如何收束到 69 前缀缓存基准

`69` 把缓存机制从单点技巧收成一个可部署的 benchmark。

```text
22 vLLM             paged KV and serving behavior
      │
24 RadixAttention   prefix reuse and tree lookup
      │
34 Prefix caching   chunked prefill and cache hit behavior
      │
      ▼
69 Prefix benchmark hit rate + TTFT + maintenance cost + delivery decision
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | 请求重用模式、TTFT、缓存关闭口径 | 保证比较合法 |
| candidate | hit rate、TTFT、维护开销 | 解释缓存收益来源 |
| 对比 | 命中增益、延迟变化、开销变化 | 判断是否值得 adopt |
| 决策 | accept / tune / reject | 输出 benchmark 结论 |


```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：workload 汇总、baseline 对比、项目判断
# 目标：把缓存收益转成可比较的 benchmark 报告

def summarize_prefix_cache(runs: List[Dict[str, float]]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_prefix_cache_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_prefix_cache_run(baseline: Dict[str, float], candidate: Dict[str, float], min_hit_rate: float, max_overhead: float) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_prefix_cache_benchmark_template():
    baseline = {'name': 'baseline', 'hit_rate': 0.0, 'ttft_ms': 130, 'overhead': 0.0}
    candidate = {'name': 'cache', 'hit_rate': 0.68, 'ttft_ms': 92, 'overhead': 0.08}
    summary = summarize_prefix_cache([baseline, candidate])
    assert summary['run_count'] == 2
    assert summary['best_hit_rate_run'] == 'cache'
    comparison = compare_prefix_cache_to_baseline(baseline, candidate)
    assert comparison['hit_rate_gain'] == 0.68
    assert comparison['ttft_delta_ms'] == -38
    assert comparison['overhead_delta'] == 0.08
    decision = recommend_prefix_cache_run(baseline, candidate, min_hit_rate=0.5, max_overhead=0.1)
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_serving_eval'


test_prefix_cache_benchmark_template()
print('测试通过：前缀缓存基准模板可以工作。')

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
def summarize_prefix_cache(runs: List[Dict[str, float]]) -> Dict[str, object]:
    best = None
    run_count = len(runs)
    avg_hit_rate = sum(item.get('hit_rate', 0.0) for item in runs) / run_count if run_count else 0.0
    avg_ttft_ms = sum(item.get('ttft_ms', 0.0) for item in runs) / run_count if run_count else 0.0
    for item in runs:
        if best is None or item.get('hit_rate', 0.0) > best.get('hit_rate', 0.0):
            best = item
    return {
        'run_count': run_count,
        'avg_hit_rate': avg_hit_rate,
        'best_hit_rate_run': best.get('name', 'run') if best else None,
        'avg_ttft_ms': avg_ttft_ms,
    }


def compare_prefix_cache_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    return {
        'hit_rate_gain': candidate.get('hit_rate', 0.0) - baseline.get('hit_rate', 0.0),
        'ttft_delta_ms': candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0),
        'overhead_delta': candidate.get('overhead', 0.0) - baseline.get('overhead', 0.0),
    }


def recommend_prefix_cache_run(baseline: Dict[str, float], candidate: Dict[str, float], min_hit_rate: float, max_overhead: float) -> Dict[str, object]:
    comparison = compare_prefix_cache_to_baseline(baseline, candidate)
    if comparison['hit_rate_gain'] >= min_hit_rate and comparison['ttft_delta_ms'] < 0 and comparison['overhead_delta'] <= max_overhead:
        return {'decision': 'accept', 'reason': '命中率、延迟收益和维护开销都达标', 'next_action': 'promote_to_serving_eval'}
    if comparison['hit_rate_gain'] >= min_hit_rate and comparison['ttft_delta_ms'] < 0:
        return {'decision': 'tune', 'reason': '命中率和延迟收益可用，但维护开销仍偏高', 'next_action': 'refine_chunk_or_eviction_policy'}
    return {'decision': 'reject', 'reason': '命中率不足或延迟收益不明显', 'next_action': 'fallback_to_no_cache'}

```

### 解析

这页现在按 `measure -> compare -> decide` 的最小 prefix cache 项目闭环组织，不再只是单独比较命中率。

#### TODO 1

- 实现方式：先汇总 run 数量、平均命中率和平均 TTFT，再找出命中率最高的 candidate。
- 关键点：这里的 `best_hit_rate_run` 不是最终项目结论，只是帮助定位哪组缓存配置最值得优先回看。
- 项目意义：先把 workload 摘要做平，后面才能在同一请求分布下比较缓存收益。

#### TODO 2

- 实现方式：统一计算 `hit_rate_gain`、`ttft_delta_ms` 和 `overhead_delta`。
- 关键点：命中率越高越好、TTFT 越低越好、额外维护开销越低越好，所以三类指标的方向不能混。
- 项目意义：这一步把 prefix cache 从“命中没命中”转成“收益和代价能否一起成立”的 benchmark 对比。

#### TODO 3

- 实现方式：先复用对比结果，再按命中率下限、TTFT 收益和开销上限输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应命中率和延迟收益可用，但 chunk 粒度、缓存失效或维护成本还没有收稳。
- 项目意义：prefix cache 项目不是只看 hit rate，而是判断这套策略值不值得继续部署、调优或回退。
### 可选 Practice-P2：真实 backend 结果入口

vLLM / SGLang 的 prefix-cache 开关、服务版本和 cache policy 可能不同；先用共享 helper 自动准备模型和服务，再把实际 policy 写入统一配置。普通服务启动成功不代表 prefix cache 已启用，必须在结果中保留 `cache_policy`、命中率和失效开销。

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
CACHE_POLICY = 'prefix_cache'  # candidate 策略；baseline 应使用 default。
RESULT_PATH = 'benchmarks/results/69_prefix_cache.json'  # 统一结果文件。
project_config = shared_project_config(
    model=MODEL_ID, backend='vllm', dtype='auto', generated_tokens=64,
    cache_policy=CACHE_POLICY,
)
print(project_config)
RUN_REAL_BACKEND = False  # 是否启动真实 backend；默认不下载模型、不占用 GPU。
if RUN_REAL_BACKEND:
    server, log_path, port, selected_dtype, model_path = start_optional_vllm(
        model_id=MODEL_ID, model_source='auto', dtype='auto',
        served_model_name=MODEL_ID,
    )
    try:
        report = run_backend_benchmark(
            project='69', base_url=f'http://127.0.0.1:{port}', model=MODEL_ID,
            label='vllm-prefix-cache', output=RESULT_PATH, concurrency=1,
            dtype=selected_dtype, cache_policy=CACHE_POLICY,
        )
        print(report['normalized_result'])
    finally:
        stop_optional_vllm(server, log_path)
# save_project_result(RESULT_PATH, project='69', strategy='prefix_cache',
#     config=project_config, metrics=metrics,
#     strategy_metrics={'hit_rate': hit_rate, 'maintenance_overhead_ms': overhead_ms},
#     decision=decision)
```
