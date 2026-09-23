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

多个请求如果共享相同的开头，前缀缓存可以让后续请求跳过已经完成的部分 Prefill。学习时要从请求分布出发，观察哪些前缀真的会重复、命中后节省了多少工作，以及缓存维护和失效带来了什么代价。本节沿着请求样本、baseline、对照指标和部署判断逐步完成评估。

**关键词：** `prefix cache`, `hit rate`, `TTFT`, `overhead`, `deployment`

---
## 前置阅读

**导语：** 进入本节前，先理解 KV Cache 如何保存请求状态，以及 PagedAttention、RadixAttention 如何组织这些状态。阅读时重点观察共享前缀如何被识别、命中和复用，再把这种复用与 TTFT、吞吐和缓存容量联系起来。
- [22. vLLM PagedAttention | vLLM PagedAttention](./22_vLLM_PagedAttention.md)
- [24. SGLang RadixAttention | SGLang RadixAttention](./24_SGLang_RadixAttention.md)
- [34. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](./34_Prefix_Cache_Matching_and_Reuse.md)


### Step 1: 建立前缀复用的整体视野

本项目从一组已经 token 化的请求样本出发，沿着“共享前缀 → token block → 匹配与命中 → 复用或淘汰 → 性能与资源代价”理解 prefix cache。后续实验只改变 cache policy，观察这条机制链是否带来稳定收益。

| 概念环节 | 学习时先回答的问题 |
|---|---|
| 请求输入 | 哪些请求共享相同的前缀，后缀和生成长度如何变化？ |
| 缓存机制 | 前缀如何切成 token block，哪些完整 block 可以被复用？ |
| 状态变化 | 命中、未命中、容量不足和淘汰分别如何改变缓存状态？ |
| 实验观察 | 复用减少了多少工作，维护它又增加了哪些代价？ |

![前缀缓存 benchmark：请求复用、缓存策略与可观测证据](../public/02_PyTorch_Algorithms/69_prefix_cache_benchmark_scope.svg)

### Step 2: 准备请求分布并建立 baseline

先从一组可复现的请求开始，确认 token 化后的共享前缀、生成长度和并发设置符合预期。随后运行关闭缓存的 G0，再开启一种 cache policy 运行 G1，把两组结果放在同一张对照表中：

| 固定项 | G0 baseline | G1 candidate |
|---|---|---|
| 请求输入 | 相同 shared-prefix workload | 与 G0 相同 |
| 模型与 backend | 固定版本 | 与 G0 相同 |
| batch / concurrency | 固定 | 与 G0 相同 |
| 唯一变量 | cache off | prefix cache policy |

### Step 3: 读懂复用收益与系统代价

学习时把“是否命中”“请求是否变快”和“缓存是否值得维护”放在一起观察。每组结果至少要同时记录：

| 指标组 | 字段 | 判断问题 |
|---|---|---|
| 复用与证据 | hit rate、reused tokens、prefill work reduction、backend metrics 或日志 | 重复计算减少了多少，命中是否有直接证据 |
| 性能 | TTFT、TPOT、E2E、throughput、P99 | 请求链路是否真正变快 |
| 资源 | peak memory、cache blocks、eviction | 缓存维护是否带来额外压力 |
| 质量 | status、task quality、OOM | 是否保持同一输出约束 |

### Step 4（CPU 代码练习）：实现缓存模拟、比较与决策

这一 Step 把前面的前缀匹配和分块机制写成四个相互衔接的 TODO：先模拟 token block 的命中与淘汰，再汇总运行结果、比较 baseline，最后生成项目决策。完成后先运行 CPU 测试，检查返回字段和判断方向；Step 5 再把这些字段对应到真实 backend 的实验记录。

| 类型 | 函数 | 你要完成或阅读的内容 | 运行后检查什么 |
|:---|:---|:---|:---|
| TODO 1 | `simulate_prefix_cache` | 模拟完整 token block 的命中、复用、容量和淘汰 | `hit_rate`、`reused_tokens`、`eviction_count`、`cache_size_blocks` |
| TODO 2 | `summarize_prefix_cache` | 汇总多次运行的命中率、TTFT 和最佳运行 | `run_count`、平均指标、`best_hit_rate_run` |
| TODO 3 | `compare_prefix_cache_to_baseline` | 统一 candidate 相对 baseline 的增益和差值方向 | `hit_rate_gain`、`ttft_delta_ms`、`overhead_delta` |
| TODO 4 | `recommend_prefix_cache_run` | 按当前 workload 的命中率增益、TTFT 和维护开销门槛形成项目决策 | `accept / tune / reject`、`reason`、`next_action`；命中率或收益不足时不能直接 accept |
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
    # TODO 1：补全 block 命中、复用和驱逐的关键状态更新；参数校验和返回结构保持现有骨架。
    # 提示：先生成完整 blocks，再用 key = (block_index, block) 查找连续命中。
    # blocks = ???
    # key = ???
    # hit_blocks = ???
    # reused_tokens = ???
    # eviction_count = ???
    #       只复用完整 block，尾部不足 block_size 的 token 不计入命中；
    #       返回 hit_rate、token_reuse_rate、prefill_work_reduction 等字段。
    # 返回 total_requests、total_prompt_tokens、reused_tokens、miss_tokens、
    # hit_rate、token_reuse_rate、prefill_work_reduction、eviction_count、cache_size_blocks。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_prefix_cache(runs: List[Dict[str, float]]) -> Dict[str, object]:
    """汇总同一请求分布下的缓存运行结果。

    输入至少应包含 name、hit_rate 和 ttft_ms；返回运行次数、平均命中率、
    平均 TTFT 和命中率最高的运行。空列表只返回空摘要，不构成性能结论。
    缺失指标不能静默解释为真实的 0；命中率是 token/block 口径时必须保持一致。"""
    # TODO 2：汇总多次 workload 运行结果。
    # 提示：遍历 runs，读取 name、hit_rate 和 ttft_ms。
    # run_count = ???
    # avg_hit_rate = ???
    # avg_ttft_ms = ???
    # best_hit_rate_run = ???
    # 空列表返回空摘要，缺失字段应明确报错。
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_prefix_cache_to_baseline(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """计算 candidate 相对 baseline 的命中率、TTFT 和开销变化。

    命中率增益应为正，TTFT 和开销差值越低越好；两组结果必须使用同一 workload_id。
    结果只服务于报告比较，不能单独证明 backend 已产生真实缓存命中。"""
    # TODO 3：比较 baseline 与 candidate 的收益和代价。
    # 提示：先确认 baseline 与 candidate 的 workload_id 一致。
    # hit_rate_gain = ???
    # ttft_delta_ms = ???
    # overhead_delta = ???
    # 命中率、TTFT 和开销的差值方向要保持可解释。
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_prefix_cache_run(baseline: Dict[str, float], candidate: Dict[str, float], min_hit_rate_gain: float, max_overhead: float) -> Dict[str, object]:
    """按命中率、TTFT 收益和维护开销给出教学决策。

    min_hit_rate_gain、max_overhead 是当前 workload 的门槛，不是通用常数。
    返回 decision、reason、next_action；不能替代真实 backend 验证。"""
    # TODO 4：先得到 comparison，再使用 min_hit_rate_gain、max_overhead 判断。
    # 提示：分别判断命中率增益、TTFT 和维护开销，再输出决策。
    # hit_rate_ok = ???
    # ttft_ok = ???
    # overhead_ok = ???
    # decision = ???
    # next_action = ???
    # TODO 4 对应决策：命中率增益未达门槛时不能仅因 TTFT 改善就 accept。
    #         只有命中率、TTFT 和维护开销同时满足门槛才 accept；否则区分 tune / reject。
    #         reason 和 next_action 必须解释当前证据不足的原因与下一步。
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
    assert simulated['hit_rate'] == 1 / 3
    assert simulated['token_reuse_rate'] == 6 / 24
    assert simulated['eviction_count'] > 0
    empty = simulate_prefix_cache([[]], block_size=2, capacity_blocks=1)
    assert empty['total_requests'] == 0 and empty['hit_rate'] == 0.0, "空请求应返回零计数摘要！"
    tail_case = simulate_prefix_cache([[1, 2, 3, 4, 5], [1, 2, 3, 4, 9]], block_size=4, capacity_blocks=8)
    assert tail_case['reused_tokens'] == 4 and tail_case['hit_rate'] == 0.5, "尾块不应计入命中，但完整前缀应复用！"
    print('CPU Prefix Cache simulation:', simulated)
    for invalid in ({'block_size': 0}, {'capacity_blocks': 0}):
        try:
            simulate_prefix_cache(sequences, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('非法缓存参数应明确拒绝！')

    baseline = {'name': 'baseline', 'workload_id': 'shared-prefix-v1', 'hit_rate': 0.0, 'ttft_ms': 130, 'overhead': 0.0}
    candidate = {'name': 'cache', 'workload_id': 'shared-prefix-v1', 'hit_rate': 0.68, 'ttft_ms': 92, 'overhead': 0.08}
    summary = summarize_prefix_cache([baseline, candidate])
    assert summary['run_count'] == 2
    assert summary['best_hit_rate_run'] == 'cache'
    comparison = compare_prefix_cache_to_baseline(baseline, candidate)
    assert comparison['hit_rate_gain'] == 0.68
    assert comparison['ttft_delta_ms'] == -38
    assert comparison['overhead_delta'] == 0.08
    mismatched = dict(candidate, workload_id='other-workload')
    try:
        compare_prefix_cache_to_baseline(baseline, mismatched)
    except ValueError:
        pass
    else:
        raise AssertionError('不同 workload 不应直接比较缓存收益')
    decision = recommend_prefix_cache_run(baseline, candidate, min_hit_rate_gain=0.5, max_overhead=0.1)
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

#### CPU 机制验证与结果解析

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
        # TODO 1 对应变量 blocks：只生成完整 token block，尾块不进入命中统计。
        blocks = [tuple(tokens[start:start + block_size]) for start in range(0, len(tokens) - block_size + 1, block_size)]
        hit_blocks = 0
        for block_index, block in enumerate(blocks):
            # TODO 1 对应变量 key：位置与 token 内容共同构成 cache key。
            key = (block_index, block)
            if key not in cache:
                break
            # TODO 1 对应变量 hit_blocks：只累计从位置 0 开始的连续命中 block。
            hit_blocks += 1
            clock += 1
            cache[key] = clock
        if hit_blocks:
            hit_requests += 1
        # TODO 1 对应变量 reused_tokens：把完整命中 block 转成复用 token 数。
        reused_tokens += hit_blocks * block_size
        for block_index, block in enumerate(blocks):
            clock += 1
            cache[(block_index, block)] = clock
            while len(cache) > capacity_blocks:
                oldest = min(cache, key=cache.get)
                # TODO 1 对应变量 eviction_count：容量超限时累计被驱逐的 block。
                del cache[oldest]
                eviction_count += 1
    miss_tokens = total_prompt_tokens - reused_tokens
    request_count = len([tokens for tokens in token_sequences if tokens])
    return {
        'total_requests': request_count,
        'total_prompt_tokens': total_prompt_tokens,
        'reused_tokens': reused_tokens,
        'miss_tokens': miss_tokens,
        'hit_rate': hit_requests / request_count if request_count else 0.0,
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
    # TODO 2 对应变量 run_count：统计 workload 运行次数。
    # TODO 2 对应变量 avg_hit_rate / avg_ttft_ms：计算同一口径下的两个均值。
    # TODO 2 对应变量 best_hit_rate_run：定位命中率最高的运行。
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

    命中率增益越高越好，TTFT 与开销差值越低越好；两组结果必须使用同一 workload_id。"""
    required = {'hit_rate', 'ttft_ms', 'overhead', 'workload_id'}
    for name, item in (('baseline', baseline), ('candidate', candidate)):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f'{name} 必须包含 {sorted(required)}')
    if baseline['workload_id'] != candidate['workload_id']:
        raise ValueError('baseline 和 candidate 必须使用同一 workload_id')
    # TODO 3 对应校验：先确认 baseline 与 candidate 的 workload_id 一致。
    # TODO 3 对应变量 hit_rate_gain：计算 candidate 相对 baseline 的命中率增益。
    # TODO 3 对应变量 ttft_delta_ms / overhead_delta：计算延迟和维护开销变化。
    return {
        'hit_rate_gain': candidate.get('hit_rate', 0.0) - baseline.get('hit_rate', 0.0),
        'ttft_delta_ms': candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0),
        'overhead_delta': candidate.get('overhead', 0.0) - baseline.get('overhead', 0.0),
    }


def recommend_prefix_cache_run(baseline: Dict[str, float], candidate: Dict[str, float], min_hit_rate_gain: float, max_overhead: float) -> Dict[str, object]:
    """按命中率、TTFT 收益和维护开销给出教学决策。

    阈值属于当前 workload 的配置，不是通用常数；该函数不能替代真实 backend 验证。"""
    if not 0 <= min_hit_rate_gain <= 1:
        raise ValueError('min_hit_rate_gain 必须位于 [0, 1]')
    if max_overhead < 0:
        raise ValueError('max_overhead 不能为负数')
    # TODO 4 对应实现：先校验门槛，再输出 accept / tune / reject。
    comparison = compare_prefix_cache_to_baseline(baseline, candidate)
    hit_rate_ok = comparison['hit_rate_gain'] >= min_hit_rate_gain
    ttft_ok = comparison['ttft_delta_ms'] < 0
    overhead_ok = comparison['overhead_delta'] <= max_overhead
    if hit_rate_ok and ttft_ok and overhead_ok:
        return {'decision': 'accept', 'reason': '命中率、延迟收益和维护开销都达标', 'next_action': 'promote_to_serving_eval'}
    if hit_rate_ok and ttft_ok:
        return {'decision': 'tune', 'reason': '命中率和延迟收益可用，但维护开销仍偏高', 'next_action': 'refine_chunk_or_eviction_policy'}
    return {'decision': 'reject', 'reason': '命中率不足或延迟收益不明显', 'next_action': 'fallback_to_no_cache'}

```

### 解析

这页按 `simulate -> summarize -> compare -> decide` 组织 prefix cache 的最小项目闭环。

**1. TODO 1：模拟 Prefix Cache**
- **实现方式**：将 token 序列按 `block_size` 切分，只把完整 block 放入缓存；查询时从位置 0 开始寻找连续命中的最长前缀。
- **关键点**：缓存 key 同时包含 block 位置和 token 内容，避免相同 token 出现在不同位置时被错误复用；容量不足时按最近最少使用顺序驱逐。
- **技术细节**：`hit_rate` 按请求统计，`token_reuse_rate` 和 `prefill_work_reduction` 按 token 统计；尾部不足一个 block 的 token 不进入命中统计。

**2. TODO 2：汇总 workload 运行结果**
- **实现方式**：汇总运行次数、平均命中率和平均 TTFT，再找出命中率最高的运行。
- **关键点**：每条记录都要包含 `name`、`hit_rate` 和 `ttft_ms`；空列表返回空摘要，缺失字段不能默认为实测 0。
- **技术细节**：`best_hit_rate_run` 用于定位优先回看的配置，不直接替代最终 benchmark 决策。

**3. TODO 3：比较 baseline 与 candidate**
- **实现方式**：先确认两组记录的 `workload_id` 一致，再计算 `hit_rate_gain`、`ttft_delta_ms` 和 `overhead_delta`。
- **关键点**：命中率增益使用 candidate - baseline；TTFT 和维护开销使用 candidate - baseline，正负方向要结合指标含义阅读。
- **技术细节**：这一步把缓存命中、请求延迟和维护开销放入同一个对照结果，供下一步决策函数使用。

**4. TODO 4：形成项目决策**
- **实现方式**：根据 `hit_rate_ok`、`ttft_ok` 和 `overhead_ok`，结合当前 workload 的门槛输出 `accept / tune / reject`。
- **关键点**：只有命中率和 TTFT 收益成立且维护开销可接受时才 `accept`；部分条件成立时进入 `tune`。
- **技术细节**：`min_hit_rate_gain` 和 `max_overhead` 是当前请求分布下的实验配置，不是所有 backend 都通用的常数。
### Step 5（可选）：GPU/backend 实验——真实 prefix cache 对照

#### 5.1 环境、输入与固定条件

本次实验回答一个具体问题：在同一组共享前缀请求上，开启 prefix cache 是否带来可观测收益。先记录实验契约，再让 G0 与 G1 只改变 cache policy；模型、backend、请求分布、生成长度、并发和测量流程保持一致。

| 实验要素 | 本轮契约 | G0 / G1 如何保持一致 | 证据输出 |
|:---|:---|:---|:---|
| 实验对象 | G0 cache off；G1 prefix cache | 使用同一模型 revision、backend 和服务版本 | 两组可比较结果 |
| 固定 workload | shared-prefix 请求文件、生成长度、batch / concurrency | 使用同一请求顺序和 token 化结果 | 统一 workload 记录 |
| 测量流程 | warmup、重复次数、超时和输出长度 | 两组使用相同流程 | 可复查的 JSON 结果 |
| 唯一变量 | cache policy | G0 关闭，G1 只开启 prefix cache | 命中、延迟、吞吐和资源对照 |

![前缀缓存 GPU/backend 对照实验契约](../public/02_PyTorch_Algorithms/69_prefix_cache_gpu_contract.svg)

#### 5.2 环境启动检查（可独立运行）

先确认模型、backend、GPU、驱动、CUDA、workload 文件和服务端口能够共同启动。更换 vLLM、SGLang、GPU 或 CUDA 后，应重新完成预检和 smoke test；普通服务启动成功不代表 prefix cache 已经启用。

| 检查项 | 当前记录 | 如何解读 |
|:---|:---|:---|
| 模型、backend、版本 | 待填写 | G0/G1 必须使用同一模型和服务版本 |
| GPU、驱动、CUDA、dtype | 待填写 | 更换硬件或运行时后重新采集 |
| workload、端口和缓存开关 | 待填写 | 确认 shared-prefix 请求可被服务读取 |
| 命中率证据来源 | backend metrics、日志或专用接口 | 不能只用 TTFT 或吞吐推断命中率 |

#### 5.3 配置实验条件

配置 G0、G1 和可选 G2：G0 关闭 prefix cache，G1 只开启一种 cache policy，G2 再比较不同 backend 或缓存策略。模型、请求、生成长度、并发和 dtype 先保持一致。

#### 5.4 执行实验并保存 JSON

配置代码分别启动 G0/G1，并保存 `69_prefix_cache_g0_cache_off.json`、`69_prefix_cache_g1_prefix_cache.json` 和总清单 `69_prefix_cache.json`。分组文件遵循 `69_prefix_cache_<group>.json` 命名；若 backend 没有暴露命中指标，结果中必须保留 `hit_rate_evidence=backend_metrics_or_logs_required`、`failure_reason` 或 `retest_path`。

#### 5.5 实测记录与结果表

读取 JSON 后，先核对实测环境与统一口径，再比较 G0/G1 的命中证据、延迟、吞吐、显存和淘汰开销。历史结果只能展示记录格式，不能替代当前硬件上的命中率或延迟证据。所有项目结果还要保留统一的 baseline/candidate、workload、质量、证据等级、失败原因和复测路径字段。

**实测环境与统一口径**

| 项目 | 当前记录 | 如何解读 |
|:---|:---|:---|
| 模型、backend、服务版本 | 待填写 | G0/G1 必须使用同一版本和模型 revision |
| GPU、驱动、PyTorch/CUDA、dtype | 待填写 | 更换运行环境后重新采集，不能混用结果 |
| workload、生成长度、并发、warmup / repeats | 待填写 | 两组使用同一请求分布和测量流程 |
| 命中率证据来源 | backend metrics、日志或专用接口 | 不能只用 TTFT 或吞吐推断命中率 |

**统一项目证据字段**

| 公共字段 | 记录内容 | 本节要求 |
|:---|:---|:---|
| `project` / `role` | 项目编号、`baseline` 或 `candidate` | G0/G1/G2 的角色和 cache policy 明确可追溯 |
| `workload` / `config` | 模型、backend、dtype、prompt、generated tokens、batch、concurrency、seed、repeat | G0 与 candidate 使用同一 workload |
| `metrics` | TTFT、TPOT、throughput、P95/P99、peak memory、queue wait | 缺失项写 `not_available`，不能由延迟推断 hit rate |
| `strategy_metrics` | acceptance / hit rate、reused tokens、transfer / handoff cost | 本节至少记录 hit rate 及证据来源 |
| `quality` / `status` | 输出质量、运行状态、失败原因 | 记录 `ok`、`unsupported`、`OOM` 或 `failed` |
| `evidence_level` / `decision` | 证据等级、`accept / tune / reject`、复测路径 | 结果不足时进入 `tune`，不能静默通过 |

**主线结果与复测记录**


| 实验组 / role | workload / config | model / backend / dtype | prompt / generated tokens | batch / concurrency | cache policy | hit rate / evidence | reused tokens | TTFT | TPOT | throughput / P95 / P99 | peak memory / queue wait | transfer / handoff | quality / status | evidence level | decision |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| G0 / baseline | 固定 workload / cache off | 待填写 | 待填写 | 1 / 1 | off | 不适用 | 0 | 待采集 | 待采集 | 待采集 / 待采集 / 待采集 | 待采集 / 待采集 | 不适用 | 待填写 / ok | gpu_baseline_smoke | 待判断 |
| G1 / candidate | 固定 workload / prefix cache | 待填写 | 待填写 | 1 / 1 | prefix cache | 待采集 / metrics 或日志 | 待采集 | 待采集 | 待采集 | 待采集 / 待采集 / 待采集 | 待采集 / 待采集 | 待采集 | 待填写 / ok | real_backend_smoke | 待判断 |
| G2 / candidate | 固定 workload / 单变量变化 | 待填写 | 待填写 | 待填写 | 待填写 | 待采集 / 证据来源 | 待采集 | 待采集 | 待采集 | 待采集 / 待采集 / 待采集 | 待采集 / 待采集 | 待采集 | 待填写 / ok | real_backend_smoke | 待判断 |

#### 5.6 结果解释与项目决策

vLLM / SGLang 的 prefix-cache 开关、服务版本和 cache policy 可能不同；决策时同时查看命中率证据、TTFT、吞吐、显存和失效开销。只有证据充分时才输出 `accept / tune / reject`。

```python
try:
    from tools.inference_project_runtime import locate_repo_root
    REPO_ROOT = locate_repo_root()
    from tools.inference_project_runtime import (
        shared_project_config, runtime_snapshot, start_optional_vllm,
        stop_optional_vllm, run_backend_benchmark,
    )
except ModuleNotFoundError:
    RUN_REAL_BACKEND = False
    def shared_project_config(**kwargs): return kwargs
    def save_project_result(*args, **kwargs): raise RuntimeError('需要从仓库根目录运行真实 backend 入口')

import json
from pathlib import Path
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
                backend='vllm', batch=1, max_tokens=64,
                num_prompts=NUM_PROMPTS, concurrency=CONCURRENCY, warmup=WARMUP,
                dtype=selected_dtype, cache_policy=group['cache_policy'],
            )
            normalized = report.get('normalized_result') or {}
            reports.append({
                'group': group,
                'report_path': result_path,
                'status': report.get('status', 'ok'),
                'quality': report.get('quality', {'status': 'not_evaluated'}),
                'evidence_level': 'real_backend_smoke',
                'normalized_result': normalized,
            })
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
        'quality': {'status': 'not_evaluated', 'reason': 'prefix-cache 命中率尚未取得 backend 指标'},
        'status': 'ok' if all(item.get('status') == 'ok' for item in reports) else 'failed',
        'evidence_level': 'real_backend_smoke',
        'decision': 'tune',
        'reason': '没有直接命中率证据，不能仅凭 TTFT 或吞吐下结论。',
    }
    manifest_path = Path(RESULT_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        'project': '69', 'config': project_config,
        'environment': runtime_snapshot(),
        'groups': reports,
        'paired_summary': paired_summary,
        'hit_rate_evidence': 'backend_metrics_or_logs_required',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'对照清单已保存: {manifest_path}')
```

## 相关阅读

完成前缀缓存 benchmark 后，可用 70 继续检查缓存命中与请求调度之间的关系，并保持相同的 workload 记录方式。下面的实现文档和论文用于继续追踪真实系统中的缓存组织、命中和显存管理。
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)
- [Automatic Prefix Caching（vLLM 开源实现文档）](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [Radix Cache（SGLang 开源实现）](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention（论文）](https://arxiv.org/abs/2309.06180)

> 注意：OpenAI-compatible API 通常不会直接返回 prefix-cache hit rate。真实 backend 实验可以用共享前缀 workload 比较 TTFT 和吞吐，但不能仅凭延迟变化反推出命中率；命中率必须来自 backend metrics、日志或专用观测接口。
