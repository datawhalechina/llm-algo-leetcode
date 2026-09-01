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
**主责与复用边界：** 本项目主责是前缀复用、命中率和重复 prefill 收益；显存优化只复用 cache 容量与峰值显存观察，70 负责队列调度，71 负责 MLA 的 cache 表示，不能把命中率直接解释成通用显存优化。

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

> 注意：OpenAI-compatible API 通常不会直接返回 prefix-cache hit rate。真实 backend 实验可以用共享前缀 workload 比较 TTFT 和吞吐，但不能仅凭延迟变化反推出命中率；命中率必须来自 backend metrics、日志或专用观测接口。

### Step 1: 定义前缀缓存 benchmark 目标

- 固定 prompt 分布、请求重用模式、batch size 和 max context length。
- 明确缓存策略、chunk size 和命中统计口径。
- 先看命中行为，再看端到端延迟收益。

### Step 2: baseline 和请求分布先要合法

- 前缀缓存 benchmark 不能脱离请求分布单独讨论；共享前缀必须在 token 化后仍然一致。
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
# 4 个核心 TODO：缓存模拟、workload 汇总、baseline 对比、项目判断
# 目标：先用 CPU 验证缓存生命周期，再把结果转成可比较的 benchmark 报告。
# CPU 题目区验证 token block 的复用和驱逐；真实 KV Cache、TTFT 和 backend 命中率属于 GPU 扩展。
def simulate_prefix_cache(token_sequences: List[List[int]], block_size: int = 4, capacity_blocks: int = 8) -> Dict[str, object]:
    """模拟按完整 token block 复用的前缀缓存；不测真实 KV Cache 显存或延迟。

    token_sequences 必须已经使用同一 tokenizer 转成 token id；相同字符串但 token 序列不同不视为命中。
    """
    if block_size <= 0 or capacity_blocks <= 0:
        raise ValueError('block_size 和 capacity_blocks 必须为正数')
    # ==========================================
    # TODO 0：实现最长完整前缀命中和 LRU 驱逐。
    # 变量提示：使用 cache、clock、hit_blocks、reused_tokens、eviction_count；
    #       只复用完整 block，尾部不足 block_size 的 token 不计入命中；
    #       每个 key 应包含 block_index 和 block token 内容，避免位置错配。
    # 返回 total_requests、total_prompt_tokens、reused_tokens、miss_tokens、
    # cache_hit_rate、token_reuse_rate、prefill_work_reduction、eviction_count、cache_size_blocks。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_prefix_cache(runs: List[Dict[str, float]]) -> Dict[str, object]:
    """汇总同一请求分布下的缓存运行结果。

    输入至少应包含 name、hit_rate 和 ttft_ms；返回运行次数、平均命中率、
    平均 TTFT 和命中率最高的运行。空列表只返回空摘要，不构成性能结论。
    缺失指标不能静默解释为真实的 0；命中率是 token/block 口径时必须保持一致。"""
    # TODO 1：使用 run_count、avg_hit_rate、avg_ttft_ms、best_hit_rate_run。
    # run_count = ???；avg_hit_rate = ???；avg_ttft_ms = ???；best_hit_rate_run = ???。
    #         不要把缺失的 hit_rate 或 ttft_ms 当成实测 0；空列表应返回空摘要。
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_prefix_cache_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """计算 candidate 相对 baseline 的命中率、TTFT 和开销变化。

    命中率增益应为正，TTFT 和开销差值越低越好；两组结果必须使用同一 workload。
    结果只服务于报告比较，不能单独证明 backend 已产生真实缓存命中。"""
    # TODO 2：使用 hit_rate_gain、ttft_delta_ms、overhead_delta。
    # hit_rate_gain = ???；ttft_delta_ms = ???；overhead_delta = ???。
    #         注意增益和差值的方向不同，并要求 baseline/candidate workload 一致。
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_prefix_cache_run(baseline: Dict[str, float], candidate: Dict[str, float], min_hit_rate: float, max_overhead: float) -> Dict[str, object]:
    """按命中率、TTFT 收益和维护开销给出教学决策。

    min_hit_rate、max_overhead 是当前 workload 的门槛，不是通用常数。
    返回 decision、reason、next_action；不能替代真实 backend 验证。"""
    # TODO 3：先得到 comparison，再使用 min_hit_rate、max_overhead 判断
    # hit_rate_ok = ???；overhead_ok = ???；decision = ???；next_action = ???。
    #         accept / tune / reject，并填写 reason、next_action。
    #         命中率未达门槛时不能仅因 TTFT 改善就 accept；门槛必须先校验。
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_prefix_cache_benchmark_template():
    sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 9, 10],
        [20, 21, 22, 23, 24, 25, 26, 27],
    ]
    simulated = simulate_prefix_cache(sequences, block_size=2, capacity_blocks=4)
    assert simulated['total_requests'] == 3
    assert simulated['total_prompt_tokens'] == 24
    assert simulated['reused_tokens'] == 6
    assert simulated['miss_tokens'] == 18
    assert simulated['cache_hit_rate'] == 1 / 3
    assert simulated['token_reuse_rate'] == 6 / 24
    assert simulated['eviction_count'] > 0
    print('CPU Prefix Cache simulation:', simulated)
    for invalid in ({'block_size': 0}, {'capacity_blocks': 0}):
        try:
            simulate_prefix_cache(sequences, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('非法缓存参数应明确拒绝！')

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

### Step 5：CPU 实验——prefix hit、复用与淘汰

CPU 代码只验证 token 前缀匹配、完整 block 复用、缓存容量和淘汰逻辑；它不代表真实 KV Cache 显存、CUDA 延迟或 backend 命中率。

## 参考代码与解析

### 代码


```python
def simulate_prefix_cache(token_sequences: List[List[int]], block_size: int = 4, capacity_blocks: int = 8) -> Dict[str, object]:
    """模拟完整 token block 的前缀复用和 LRU 驱逐。

    `token_sequences` 是已经完成 tokenization 的请求序列；本函数只验证
    block 命中、复用和淘汰，不计算真实 KV Cache 的字节数或 backend 延迟。"""
    if block_size <= 0 or capacity_blocks <= 0:
        raise ValueError('block_size 和 capacity_blocks 必须为正数')
    if not isinstance(token_sequences, list):
        raise TypeError('token_sequences 必须是 list[list[int]]')
    if not all(isinstance(tokens, list) for tokens in token_sequences):
        raise TypeError('token_sequences 的每个元素必须是 token list')
    cache = {}
    clock = 0
    reused_tokens = 0
    total_prompt_tokens = 0
    hit_requests = 0
    eviction_count = 0
    for tokens in token_sequences:
        if not tokens:
            continue
        total_prompt_tokens += len(tokens)
        blocks = [tuple(tokens[start:start + block_size]) for start in range(0, len(tokens) - block_size + 1, block_size)]
        hit_blocks = 0
        for block_index, block in enumerate(blocks):
            key = (block_index, block)
            if key not in cache:
                break
            hit_blocks += 1
            clock += 1
            cache[key] = clock
        if hit_blocks:
            hit_requests += 1
        reused_tokens += hit_blocks * block_size
        for block_index, block in enumerate(blocks):
            clock += 1
            cache[(block_index, block)] = clock
            while len(cache) > capacity_blocks:
                oldest = min(cache, key=cache.get)
                del cache[oldest]
                eviction_count += 1
    miss_tokens = total_prompt_tokens - reused_tokens
    request_count = len([tokens for tokens in token_sequences if tokens])
    return {
        'total_requests': request_count,
        'total_prompt_tokens': total_prompt_tokens,
        'reused_tokens': reused_tokens,
        'miss_tokens': miss_tokens,
        'cache_hit_rate': hit_requests / request_count if request_count else 0.0,
        'token_reuse_rate': reused_tokens / total_prompt_tokens if total_prompt_tokens else 0.0,
        'prefill_work_reduction': reused_tokens / total_prompt_tokens if total_prompt_tokens else 0.0,
        'eviction_count': eviction_count,
        'cache_size_blocks': len(cache),
    }


def summarize_prefix_cache(runs: List[Dict[str, float]]) -> Dict[str, object]:
    """汇总同一请求分布下的缓存运行结果。

    每条 run 应包含 name、hit_rate 和 ttft_ms；空列表返回空摘要。
    缺失指标不能静默解释为真实的 0。"""
    if not isinstance(runs, list):
        raise TypeError('runs 必须是 list[dict]')
    required = {'name', 'hit_rate', 'ttft_ms'}
    for index, item in enumerate(runs):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f'第 {index} 条 run 必须包含 {sorted(required)}')
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
    """计算 candidate 相对 baseline 的命中率、TTFT 和开销变化。

    命中率增益越高越好，TTFT 与开销差值越低越好；两组结果必须使用同一 workload。"""
    required = {'hit_rate', 'ttft_ms', 'overhead'}
    for name, item in (('baseline', baseline), ('candidate', candidate)):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f'{name} 必须包含 {sorted(required)}')
    return {
        'hit_rate_gain': candidate.get('hit_rate', 0.0) - baseline.get('hit_rate', 0.0),
        'ttft_delta_ms': candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0),
        'overhead_delta': candidate.get('overhead', 0.0) - baseline.get('overhead', 0.0),
    }


def recommend_prefix_cache_run(baseline: Dict[str, float], candidate: Dict[str, float], min_hit_rate: float, max_overhead: float) -> Dict[str, object]:
    """按命中率、TTFT 收益和维护开销给出教学决策。

    阈值属于当前 workload 的配置，不是通用常数；该函数不能替代真实 backend 验证。"""
    if not 0 <= min_hit_rate <= 1:
        raise ValueError('min_hit_rate 必须位于 [0, 1]')
    if max_overhead < 0:
        raise ValueError('max_overhead 不能为负数')
    comparison = compare_prefix_cache_to_baseline(baseline, candidate)
    if comparison['hit_rate_gain'] >= min_hit_rate and comparison['ttft_delta_ms'] < 0 and comparison['overhead_delta'] <= max_overhead:
        return {'decision': 'accept', 'reason': '命中率、延迟收益和维护开销都达标', 'next_action': 'promote_to_serving_eval'}
    if comparison['hit_rate_gain'] >= min_hit_rate and comparison['ttft_delta_ms'] < 0:
        return {'decision': 'tune', 'reason': '命中率和延迟收益可用，但维护开销仍偏高', 'next_action': 'refine_chunk_or_eviction_policy'}
    return {'decision': 'reject', 'reason': '命中率不足或延迟收益不明显', 'next_action': 'fallback_to_no_cache'}

```

### 解析

这页现在按 `measure -> compare -> decide` 的最小 prefix cache 项目闭环组织，不再只是单独比较命中率。

#### TODO 0：模拟 Prefix Cache

- 实现方式：将 token 序列按 `block_size` 切分，只把完整 block 放入缓存；查询时从位置 0 开始寻找连续命中的最长前缀。
- 关键点：缓存 key 同时包含 block 位置和 token 内容，避免相同 token 出现在不同位置时被错误复用；容量不足时按最近最少使用顺序驱逐。
- 项目意义：CPU 可以验证命中、未命中、复用 token、prefill 工作量和驱逐逻辑，但不能据此推断真实 KV Cache 显存、CUDA 延迟或 backend 命中率。

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
### Step 6（可选）：GPU/backend 实验——真实 prefix cache 对照

**实验条件表**

| 项目 | G0 cache off | G1 prefix cache | G2 策略/backend 对照 |
|---|---|---|---|
| workload | shared-prefix 固定 | 与 G0 相同 | 与 G0 相同 |
| cache policy | default / off | 明确记录实际 policy | 分别记录 |
| backend | 固定 | 固定 | vLLM / SGLang 分开记录 |

**结果表模板**

| 实验组 | cache policy | hit rate | reused tokens | TTFT | throughput | peak VRAM | decision |
|---|---|---:|---:|---:|---:|---:|---|
| G0 | off | 不适用 | 0 | 待采集 | 待采集 | 待采集 | 待判断 |
| G1 | prefix cache | 待采集 | 待采集 | 待采集 | 待采集 | 待采集 | 待判断 |

配置代码会自动分别启动 G0/G1，并保存 `69_prefix_cache_g0_cache_off.json`、`69_prefix_cache_g1_prefix_cache.json` 和总清单 `69_prefix_cache.json`。如果 backend 没有暴露命中指标，报告必须保留 `hit_rate_evidence=backend_metrics_or_logs_required`，不能用 TTFT 变化代替命中率。

**证据边界**：CPU 模拟可以验证前缀匹配、block 复用和淘汰；真实命中率必须来自 backend metrics、日志或专用接口，不能仅凭 TTFT 或吞吐变化反推。
按 66 的统一分组执行：G0 关闭 prefix cache，G1 在相同 shared-prefix workload 下开启一种 cache policy，G2 才比较不同 backend 或缓存策略。普通服务启动成功不代表 prefix cache 已启用。

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

import json
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # G0/G1 使用同一基座模型。
WORKLOAD = 'benchmarks/workloads/prefix_reuse.jsonl'  # 共享前缀、变化后缀的请求分布。
RESULT_PATH = 'benchmarks/results/69_prefix_cache.json'  # 保存对照清单，不覆盖分组结果。
NUM_PROMPTS = 5  # smoke 规模；正式结论应提高请求数并重复运行。
CONCURRENCY = 1  # G0/G1 必须保持一致；并发扫描另存为独立实验。
WARMUP = 1
GROUPS = (
    {'name': 'g0_cache_off', 'cache_policy': 'off', 'enabled': False},
    {'name': 'g1_prefix_cache', 'cache_policy': 'prefix_cache', 'enabled': True},
)
project_config = shared_project_config(
    model=MODEL_ID, backend='vllm', dtype='auto', generated_tokens=64,
    workload=WORKLOAD, num_prompts=NUM_PROMPTS, concurrency=CONCURRENCY,
    warmup=WARMUP, groups=[item['name'] for item in GROUPS],
)
print(project_config)
RUN_REAL_BACKEND = False  # 改为 True 才启动 G0/G1；默认不下载模型、不占用 GPU。
if RUN_REAL_BACKEND:
    reports = []
    for group in GROUPS:
        result_path = f"benchmarks/results/69_prefix_cache_{group['name']}.json"
        server, log_path, port, selected_dtype, model_path = start_optional_vllm(
            model_id=MODEL_ID, model_source='auto', dtype='auto',
            served_model_name=MODEL_ID, enable_prefix_caching=group['enabled'],
        )
        try:
            report = run_backend_benchmark(
                project='69', base_url=f'http://127.0.0.1:{port}', model=MODEL_ID,
                label=f"vllm-{group['name']}", output=result_path, workload=WORKLOAD,
                num_prompts=NUM_PROMPTS, concurrency=CONCURRENCY, warmup=WARMUP,
                dtype=selected_dtype, cache_policy=group['cache_policy'],
            )
            reports.append({'group': group, 'report_path': result_path,
                            'normalized_result': report.get('normalized_result')})
            print(reports[-1])
        finally:
            stop_optional_vllm(server, log_path)
    def metric_value(report, metric, percentile='p50'):
        values = ((report.get('normalized_result') or {}).get('metrics') or {}).get(metric)
        if isinstance(values, dict):
            return values.get(percentile)
        return values

    by_group = {item['group']['name']: item for item in reports}
    g0 = by_group.get('g0_cache_off', {})
    g1 = by_group.get('g1_prefix_cache', {})
    paired_summary = {
        'g0_ttft_p50_ms': metric_value(g0, 'ttft_ms'),
        'g1_ttft_p50_ms': metric_value(g1, 'ttft_ms'),
        'g0_throughput_tokens_per_s': metric_value(g0, 'throughput_tokens_per_s', None),
        'g1_throughput_tokens_per_s': metric_value(g1, 'throughput_tokens_per_s', None),
        'hit_rate': None,
        'hit_rate_evidence': 'backend_metrics_or_logs_required',
        'decision': 'not_evaluated',
        'reason': '没有直接命中率证据，不能仅凭 TTFT 或吞吐下结论。',
    }
    manifest_path = Path(RESULT_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        'project': '69', 'config': project_config, 'groups': reports,
        'paired_summary': paired_summary,
        'hit_rate_evidence': 'backend_metrics_or_logs_required',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'对照清单已保存: {manifest_path}')
```
