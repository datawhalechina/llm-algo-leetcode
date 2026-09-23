# 66. Inference Performance Comparison | 推理性能对比实验

**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `基准对比`, `性能对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节带你完成一次可复核的推理性能对比：围绕同一组请求和运行条件，比较 baseline 与候选方案在延迟、吞吐、显存和失败状态上的差异。
学习过程中，你会先建立可比较的实验条件，再理解 TTFT、TPOT、端到端延迟和吞吐分别反映什么，最后把这些证据转化为面向场景的选择：低延迟、高吞吐，或显存受限。


**关键词：** `benchmark`, `TTFT`, `TPOT`, `throughput`, `KV cache`

---
## 前置阅读

**导语：** 开始前先了解一次请求如何经历解码、KV Cache 如何增长，以及推理后端如何组织服务。阅读这些材料时，关注它们对输入处理、逐 token 生成和缓存占用的影响；这些概念随后会转化为本节可以测量的 workload 和指标。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

---
### Step 1: 把问题转成可测量的实验
先把“哪种方案更适合当前目标”转成一组可观察、可复核的实验：明确模型、请求集、运行环境和对照分组，再确定要记录的延迟、吞吐、显存与失败状态。下表和流程图把请求输入、CPU 预判、真实基线、候选对照与项目结论串起来，帮助你理解每组实验为什么存在、会留下什么证据。整个过程先固定 workload，再做单变量对照，最后根据指标形成选型结论。


| 实验阶段 | 你要做什么 | 阶段产出 |
|:---|:---|:---|
| C0 请求模拟 | 在 CPU 上模拟请求、并发和 prefill / decode | 请求轨迹与阶段耗时 |
| C1 瓶颈分类 | 在 CPU 上读取模拟指标 | prefill-bound、decode-bound 或 memory-bound |
| G0 真实基线 | 在 GPU/backend 上运行 baseline | 真实性能共同参照 |
| G1 单变量对照 | 在 GPU/backend 上运行 candidate | 单变量收益与代价 |
| G2 扩展或组合对照 | 在 G0/G1 之后比较多个候选或组合已验证策略 | 当前 workload 下的项目结论 |

![66 推理 benchmark 实验流程](../public/02_PyTorch_Algorithms/66_inference_benchmark_flow.svg)

### Step 2: 控制实验条件，选择对照变量

先把两次实验放在同一条起跑线上：模型、请求和测量方法保持一致，只选择一个主要条件进行变化。这样，后面的性能差异才有明确的解释依据。按下表准备实验。

| 实验部分 | 具体怎么做 | 目的 |
|:---|:---|:---|
| 固定条件 | 模型及权重版本、请求集、输入长度、输出长度、dtype、Cache 配置、预热次数和重复次数保持一致 | 保证两次实验可以比较 |
| 唯一变化 | 从并发数、batch、dtype、Cache 配置或 backend 中选择一个作为变量；选中的条件不再作为固定条件 | 让性能变化能够归因 |
| 记录口径 | 预先约定 TTFT、TPOT、端到端延迟、吞吐、峰值显存、成功率和 OOM 字段 | 保证不同结果可以整理到同一张表 |

### Step 3: 读懂指标，整理可比结果

把 baseline 和 candidate 的原始结果整理成同一组指标，并结合请求生命周期理解它们分别反映什么。先看首 token 等待、逐 token 生成和整次请求的差异，再观察吞吐、显存与失败状态是否出现新的代价。

| 指标 | 单位 | 主要回答的问题 |
|:---|:---|:---|
| TTFT | ms | 用户等待第一个输出 token 多久？ |
| TPOT | ms/token | decode 阶段每生成一个 token 多久？ |
| E2E latency | ms | 一次请求从开始到结束总共多久？ |
| output throughput | token/s | 系统每秒生成多少输出 token？ |
| peak memory | MB / MiB | 当前 workload 的显存峰值是否接近预算？ |
| success / OOM | 次数 / 状态 | 请求是否完成，是否发生显存不足？ |

### Step 4: 从指标定位瓶颈，形成验证计划

把测得的指标当作诊断线索：先判断时间主要消耗在输入处理、逐 token 生成，还是显存容量，再为每种判断安排针对性的复测。下表给出从现象到验证动作的映射；不要根据单个指标直接下结论，还要同时检查相关代价和失败状态。

| 你观察到的现象 | 瓶颈判断 | 优先检查方向 | 下一步验证什么 |
|:---|:---|:---|:---|
| 长输入导致首 token 等待时间明显增加 | Prefill 瓶颈（输入处理） | FlashAttention、chunked prefill | 用 G0/G1 复测 TTFT，并检查显存与启动成本 |
| 每个输出 token 生成较慢 | Decode 瓶颈（逐 token 生成） | KV Cache、投机解码、解码调度 | 用 TPOT、吞吐和质量结果验证 |
| 显存接近上限或无法提高 batch | 显存瓶颈 | PagedAttention、KV Cache 量化、GQA/MQA | 复测峰值显存、速度和 OOM 状态 |
| 各项指标没有明显短板 | 暂不明确 | 保持 baseline 或继续 Profiling | 先补充 profiler 证据，再决定是否切换 |

```python
import time

```


```python
# 补全请求模拟和推理性能对比的七个关键函数
# 目标：完成 request -> workload -> metrics -> bottleneck -> comparison -> decision 链路。
# CPU 题目区只验证离散事件和指标口径；真实 kernel、KV Cache 和 backend 行为由 GPU 扩展验证。
def simulate_inference_requests(requests, concurrency=1, prefill_ms_per_token=0.5, decode_ms_per_token=1.0, peak_mem_per_request_mb=512.0, kv_cache_mb_per_token=0.0):
    """在 CPU 上模拟请求排队和 prefill/decode 阶段，不测真实 kernel 性能。

    requests 至少包含 prompt_tokens 和 generated_tokens；返回每个请求的时间轨迹及汇总指标。
    """
    if concurrency <= 0 or prefill_ms_per_token < 0 or decode_ms_per_token < 0 or peak_mem_per_request_mb < 0 or kv_cache_mb_per_token < 0:
        raise ValueError('concurrency 和成本参数必须合法')
    # ==========================================
    # TODO 1：按 concurrency 分批，计算每个请求的 queue / prefill / decode / e2e。
    # 提示：每一批同时执行，批次耗时取该批请求 prefill+decode 的最大值；
    #       queue_ms 是等待时间，peak_mem_mb 取同时执行请求数 * 单请求预算，
    #       kv_cache_mb_per_token 只作为教学估算项，不是实际 KV Cache 分配。
    # request_results = ???  # 保留每个请求的阶段轨迹。
    # duration_ms = ???  # 汇总所有批次完成时间。
    # peak_mem_mb 和 kv_cache_tokens_peak 由同一批次轨迹派生，不单独增加 TODO。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
    """汇总推理 workload 配置，形成统一比较口径。

    prompt_tokens 和 generated_tokens 必须是非负整数；total_tokens 是两者之和。
    """
    # ==========================================
    # TODO 2：检查输入契约并汇总推理 workload 配置。
    # 提示：batch_size 必须为正，token 数必须非负；total_tokens = prompt_tokens + generated_tokens，不能把 batch 重复加进 token 数。
    # ==========================================
    # 先完成参数合法性检查，再填写 total_tokens = ???
    return {
        'model_name': model_name,
        'backend': backend,
        'batch_size': batch_size,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens,
        'total_tokens': total_tokens,
        'dtype': dtype,
        'cache_policy': cache_policy,
    }

def summarize_prefill_decode(prefill_ms, decode_ms, generated_tokens):
    """汇总 prefill / decode 延迟，形成最小延迟摘要。

    TTFT 近似为 prefill_ms，TPOT 只在 generated_tokens > 0 时计算；
    prefill_share 与 decode_share 应在 total_ms 上归一化。
    """
    # ==========================================
    # TODO 3: 汇总 prefill / decode 延迟
    # 提示：TTFT 近似等于 prefill_ms；TPOT = decode_ms / generated_tokens。
    #       generated_tokens 为 0 时应明确处理，而不是产生除零错误。
    # ==========================================
    # total_ms = ???
    # ttft_ms = ???
    # tpot_ms = ???
    # prefill_share = ???
    # decode_share = ???
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total_ms, 2),
        'ttft_ms': round(ttft_ms, 2),
        'tpot_ms': round(tpot_ms, 4),
        'prefill_share': round(prefill_share, 3),
        'decode_share': round(decode_share, 3),
    }

def compute_inference_metrics(config, latency_summary, peak_mem_mb):
    """把 workload 和延迟摘要收束成统一推理指标。

    throughput_tok_s 表示该 workload 的输出 token 吞吐；peak_mem_mb 是输入的估算或实测值，
    必须沿用其证据等级，不能在函数内改写来源。
    """
    # ==========================================
    # TODO 4: 计算推理项目核心指标
    # 提示：throughput 表示整个 batch 每秒生成 token 数，即 batch * generated_tokens / total_ms。
    # ==========================================
    # output_tokens = ???
    # throughput_tok_s = ???
    return {
        'backend': config['backend'],
        'batch_size': config['batch_size'],
        'prompt_tokens': config['prompt_tokens'],
        'generated_tokens': config['generated_tokens'],
        'ttft_ms': latency_summary['ttft_ms'],
        'tpot_ms': latency_summary['tpot_ms'],
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': latency_summary['total_ms'],
        'prefill_share': latency_summary['prefill_share'],
        'decode_share': latency_summary['decode_share'],
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def diagnose_inference_bottleneck(metrics, memory_budget_mb=None):
    """根据显存预算与 prefill/decode 占比诊断推理瓶颈。

    返回 bottleneck 和 reason；这是规则化诊断，不是 profiler 的最终归因。
    """
    # ==========================================
    # TODO 5: 诊断推理瓶颈
    # 规则：显存接近预算优先判为 memory-bound；否则按 prefill/decode 占比判断。
    # 提示：memory_budget_mb 为空时，memory_pressure 应为 False；占比达到 0.6 才算对应阶段偏重。
    # 只需要补全三个判断变量，下面的分支和报告文案已经给出。
    # ==========================================
    # memory_pressure = ???
    # prefill_heavy = ???
    # decode_heavy = ???
    if memory_pressure:
        bottleneck = 'memory-bound'
        reason = 'peak memory 接近预算，优先检查 KV cache、batch size、量化和分页策略。'
    elif prefill_heavy:
        bottleneck = 'prefill-bound'
        reason = 'prefill 占比高，优先检查 prompt length、FlashAttention、chunked prefill 和 batching。'
    elif decode_heavy:
        bottleneck = 'decode-bound'
        reason = 'decode 占比高，优先检查 KV cache 读写、decode scheduling、speculative decoding 或 multi-token decoding。'
    else:
        bottleneck = 'balanced'
        reason = 'prefill、decode 和显存压力都不突出，先保持 baseline 或继续做细粒度 profiling。'
    return {'bottleneck': bottleneck, 'reason': reason}

def compare_inference_candidates(baseline_metrics, candidate_metrics):
    """统一比较 baseline 与 candidate 的推理收益和代价。

    两者必须使用同一 workload；延迟和显存差值采用 baseline - candidate，
    throughput_gain 采用 (candidate - baseline) / baseline。
    """
    # ==========================================
    # TODO 6: 比较 baseline 和 candidate
    # 提示：latency / TTFT / TPOT / memory 的 delta 用 baseline - candidate；throughput gain 用比例增益。
    #       baseline throughput 为 0 时应显式处理，不能静默返回无穷大。
    # ==========================================
    # total_latency_delta_ms = ???  # 核心延迟差值。
    # throughput_gain = ???  # 核心吞吐比例增益。
    # ttft/tpot/peak_mem 的 delta 沿用同一 baseline - candidate 规则派生。
    return {
        'total_latency_delta_ms': round(total_latency_delta_ms, 2),
        'ttft_delta_ms': round(ttft_delta_ms, 2),
        'tpot_delta_ms': round(tpot_delta_ms, 4),
        'peak_mem_delta_mb': round(peak_mem_delta_mb, 2),
        'throughput_gain': round(throughput_gain, 4),
    }

def recommend_inference_decision(comparison, candidate_bottleneck, min_throughput_gain=0.1, max_ttft_regression_ms=20.0):
    """根据吞吐、TTFT 和瓶颈类型输出推理选型建议。

    返回 decision 和 reason；阈值属于当前教学 workload，不是通用 SLA。
    """
    # ==========================================
    # TODO 7: 输出推理选型建议
    # 规则：吞吐明显提升且 TTFT 没明显退化则 accept；有收益但仍有瓶颈则 tune；否则 reject。
    # 提示：throughput_good、ttft_ok、still_tunable 分别对应三个判断条件。
    # 下面的决策分支已经给出，只需要补全三个布尔变量。
    # ==========================================
    # throughput_good = ???
    # ttft_ok = ???
    # still_tunable = ???
    return {'decision': decision, 'reason': reason}

```


```python
# 测试目标按机制拆分：请求模拟、阶段指标、瓶颈诊断、对照差值和决策边界。
# 最后的 integration test 再检查这些结果能否串成完整的 baseline -> candidate 链路。
def test_request_simulation_contract():
    requests = [{'prompt_tokens': 100, 'generated_tokens': 20}, {'prompt_tokens': 200, 'generated_tokens': 10}]
    result = simulate_inference_requests(requests, concurrency=1, peak_mem_per_request_mb=256.0)
    assert result['request_count'] == 2
    assert result['request_results'][1]['queue_ms'] == 70.0
    assert result['peak_mem_mb'] == 256.0
    print('✅ request simulation contract')

def test_latency_and_metric_contract():
    config = build_inference_config('tiny-llama', 'pytorch-eager', 2, 128, 32, 'fp16', 'static-kv-cache')
    latency = summarize_prefill_decode(80.0, 160.0, 32)
    metrics = compute_inference_metrics(config, latency, peak_mem_mb=4096.0)
    assert config['total_tokens'] == 160
    assert latency['ttft_ms'] == 80.0 and latency['tpot_ms'] == 5.0
    assert metrics['throughput_tok_s'] == 266.67
    print('✅ latency and metric contract')

def test_bottleneck_contract():
    metrics = {'peak_mem_mb': 4096.0, 'prefill_share': 0.333, 'decode_share': 0.667}
    assert diagnose_inference_bottleneck(metrics, 4400.0)['bottleneck'] == 'memory-bound'
    assert diagnose_inference_bottleneck(metrics, 8192.0)['bottleneck'] == 'decode-bound'
    print('✅ bottleneck contract')

def test_comparison_contract():
    baseline = {'total_ms': 240.0, 'ttft_ms': 80.0, 'tpot_ms': 5.0, 'peak_mem_mb': 4096.0, 'throughput_tok_s': 266.67}
    candidate = {'total_ms': 205.0, 'ttft_ms': 85.0, 'tpot_ms': 3.75, 'peak_mem_mb': 3584.0, 'throughput_tok_s': 312.2}
    result = compare_inference_candidates(baseline, candidate)
    assert result['total_latency_delta_ms'] == 35.0 and result['peak_mem_delta_mb'] == 512.0
    assert result['throughput_gain'] > 0.15
    print('✅ comparison contract')

def test_decision_contract():
    comparison = {'throughput_gain': 0.2, 'ttft_delta_ms': -5.0}
    assert recommend_inference_decision(comparison, {'bottleneck': 'decode-bound'})['decision'] == 'accept'
    weak = {'throughput_gain': 0.02, 'ttft_delta_ms': 1.0}
    assert recommend_inference_decision(weak, {'bottleneck': 'decode-bound'})['decision'] == 'tune'
    print('✅ decision contract')

def test_inference_project_integration():
    try:
        requests = [
            {'prompt_tokens': 100, 'generated_tokens': 20},
            {'prompt_tokens': 200, 'generated_tokens': 10},
            {'prompt_tokens': 100, 'generated_tokens': 20},
        ]
        simulation = simulate_inference_requests(
            requests, concurrency=2, prefill_ms_per_token=0.5,
            decode_ms_per_token=1.0, peak_mem_per_request_mb=256.0,
        )
        assert simulation['request_count'] == 3, "请求数量统计不正确！"
        assert simulation['duration_ms'] == 180.0, "批次执行时长计算不正确！"
        assert simulation['request_results'][2]['queue_ms'] == 110.0, "排队时间计算不正确！"
        assert simulation['peak_mem_mb'] == 512.0, "并发显存预算计算不正确！"
        scaled = simulate_inference_requests(
            [{'prompt_tokens': 100, 'generated_tokens': 20}],
            peak_mem_per_request_mb=256.0, kv_cache_mb_per_token=1.0,
        )
        assert scaled['kv_cache_tokens_peak'] == 120, "KV Cache token 数计算不正确！"
        assert scaled['peak_mem_mb'] == 376.0, "KV Cache 显存随 token 增长的计算不正确！"
        empty = simulate_inference_requests([])
        assert empty['request_count'] == 0 and empty['duration_ms'] == 0.0, "空请求列表应返回空结果！"
        zero_decode = summarize_prefill_decode(prefill_ms=20.0, decode_ms=0.0, generated_tokens=0)
        assert zero_decode['tpot_ms'] == 0.0, "没有输出 token 时 TPOT 应为 0！"
        for invalid in ({'concurrency': 0}, {'prefill_ms_per_token': -1.0}, {'kv_cache_mb_per_token': -1.0}):
            try:
                simulate_inference_requests(requests, **invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('非法请求模拟参数应明确拒绝！')

        config = build_inference_config(
            model_name='tiny-llama',
            backend='pytorch-eager',
            batch_size=2,
            prompt_tokens=128,
            generated_tokens=32,
            dtype='fp16',
            cache_policy='static-kv-cache',
        )
        assert config['total_tokens'] == 160, "total_tokens 计算不正确！"
        assert config['batch_size'] == 2, "batch_size 应保留原始配置！"

        latency = summarize_prefill_decode(prefill_ms=80.0, decode_ms=160.0, generated_tokens=32)
        assert latency['total_ms'] == 240.0, "total_ms 计算不正确！"
        assert latency['ttft_ms'] == 80.0, "ttft_ms 计算不正确！"
        assert latency['tpot_ms'] == 5.0, "tpot_ms 计算不正确！"
        assert latency['prefill_share'] == 0.333, "prefill_share 计算不正确！"
        assert latency['decode_share'] == 0.667, "decode_share 计算不正确！"

        metrics = compute_inference_metrics(config, latency, peak_mem_mb=4096.0)
        assert metrics['throughput_tok_s'] == 266.67, "throughput_tok_s 计算不正确！"
        assert metrics['peak_mem_mb'] == 4096.0, "peak_mem_mb 记录不正确！"

        memory_bound = diagnose_inference_bottleneck(metrics, memory_budget_mb=4400.0)
        assert memory_bound['bottleneck'] == 'memory-bound', "显存接近预算时应优先判为 memory-bound！"

        decode_bound = diagnose_inference_bottleneck(metrics, memory_budget_mb=8192.0)
        assert decode_bound['bottleneck'] == 'decode-bound', "decode 占比高时应判为 decode-bound！"

        candidate_config = build_inference_config(
            model_name='tiny-llama',
            backend='paged-attention',
            batch_size=2,
            prompt_tokens=128,
            generated_tokens=32,
            dtype='fp16',
            cache_policy='paged-kv-cache',
        )
        candidate_latency = summarize_prefill_decode(prefill_ms=85.0, decode_ms=120.0, generated_tokens=32)
        candidate_metrics = compute_inference_metrics(candidate_config, candidate_latency, peak_mem_mb=3584.0)
        comparison = compare_inference_candidates(metrics, candidate_metrics)

        assert comparison['total_latency_delta_ms'] == 35.0, "total latency delta 计算不正确！"
        assert comparison['ttft_delta_ms'] == -5.0, "TTFT delta 计算不正确！"
        assert comparison['tpot_delta_ms'] == 1.25, "TPOT delta 计算不正确！"
        assert comparison['peak_mem_delta_mb'] == 512.0, "peak memory delta 计算不正确！"
        assert comparison['throughput_gain'] > 0.15, "throughput gain 应体现候选方案收益！"
        zero_throughput = dict(metrics, throughput_tok_s=0.0)
        try:
            compare_inference_candidates(zero_throughput, candidate_metrics)
        except ValueError:
            pass
        else:
            raise AssertionError("baseline throughput 为 0 时应明确拒绝相对增益计算！")

        decision = recommend_inference_decision(comparison, decode_bound)
        assert decision['decision'] == 'accept', "吞吐提升且 TTFT 未明显退化时应建议 accept！"

        weak_comparison = dict(comparison)
        weak_comparison['throughput_gain'] = 0.02
        weak_comparison['ttft_delta_ms'] = 1.0
        assert recommend_inference_decision(weak_comparison, decode_bound)['decision'] == 'tune', "小幅收益但仍有瓶颈时应建议 tune！"

        bad_comparison = dict(comparison)
        bad_comparison['throughput_gain'] = -0.05
        bad_comparison['ttft_delta_ms'] = -30.0
        assert recommend_inference_decision(bad_comparison, {'bottleneck': 'balanced'})['decision'] == 'reject', "没有收益且 TTFT 退化时应建议 reject！"

        print("✅ 推理性能对比项目模板代码通过基础校验。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, RuntimeError) as e:
        print(f"代码执行失败，请根据原始错误检查 TODO、输入契约或变量类型：{e}")
        raise
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


def run_inference_tests():
    for test in (test_request_simulation_contract, test_latency_and_metric_contract, test_bottleneck_contract, test_comparison_contract, test_decision_contract, test_inference_project_integration):
        test()
    print('✅ 推理性能项目：机制测试与集成测试全部通过。')

run_inference_tests()

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
# TODO 1: 模拟请求执行
def simulate_inference_requests(requests, concurrency=1, prefill_ms_per_token=0.5, decode_ms_per_token=1.0, peak_mem_per_request_mb=512.0, kv_cache_mb_per_token=0.0):
    """模拟并发请求的排队、prefill、decode 和显存占用。"""
    if concurrency <= 0 or prefill_ms_per_token < 0 or decode_ms_per_token < 0 or peak_mem_per_request_mb < 0 or kv_cache_mb_per_token < 0:
        raise ValueError('concurrency 和成本参数必须合法')
    if not requests:
        return {'request_count': 0, 'duration_ms': 0.0, 'request_results': [], 'peak_mem_mb': 0.0}
    # TODO 1 对应变量 request_results：收集每个请求的 queue/prefill/decode/e2e 轨迹。
    results = []
    # TODO 1 对应变量 duration_ms：用 clock_ms 推进批次完成时间。
    clock_ms = 0.0
    for start in range(0, len(requests), concurrency):
        wave = requests[start:start + concurrency]
        wave_results = []
        for request in wave:
            prompt_tokens = int(request['prompt_tokens'])
            generated_tokens = int(request['generated_tokens'])
            if prompt_tokens <= 0 or generated_tokens <= 0:
                raise ValueError('每个请求的 token 数必须为正数')
            prefill_ms = prompt_tokens * prefill_ms_per_token
            decode_ms = generated_tokens * decode_ms_per_token
            wave_results.append({
                'prompt_tokens': prompt_tokens,
                'generated_tokens': generated_tokens,
                'queue_ms': clock_ms,
                'prefill_ms': prefill_ms,
                'decode_ms': decode_ms,
                'ttft_ms': clock_ms + prefill_ms,
                'tpot_ms': decode_ms / generated_tokens,
                'e2e_ms': clock_ms + prefill_ms + decode_ms,
                'kv_cache_tokens': prompt_tokens + generated_tokens,
                'kv_cache_mem_mb': peak_mem_per_request_mb + (prompt_tokens + generated_tokens) * kv_cache_mb_per_token,
            })
        results.extend(wave_results)
        clock_ms += max(item['prefill_ms'] + item['decode_ms'] for item in wave_results)
    # TODO 1 的核心结果是 request_results 和 duration_ms；peak_mem_mb、kv_cache_tokens_peak 由批次轨迹派生。
    total_output_tokens = sum(item['generated_tokens'] for item in results)
    total_prompt_tokens = sum(item['prompt_tokens'] for item in results)
    return {
        'request_count': len(results),
        'total_prompt_tokens': total_prompt_tokens,
        'total_output_tokens': total_output_tokens,
        'duration_ms': round(clock_ms, 4),
        'peak_mem_mb': round(max(sum(item['kv_cache_mem_mb'] for item in results[start:start + concurrency]) for start in range(0, len(results), concurrency)), 2),
        'kv_cache_tokens_peak': max((sum(item['kv_cache_tokens'] for item in results[start:start + concurrency]) for start in range(0, len(results), concurrency)), default=0),
        'request_results': results,
    }

# TODO 2: 汇总推理 workload 配置
def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
    """汇总 workload 配置，并保留后续对照所需的全部元数据。"""
    if not model_name or not backend or batch_size <= 0 or prompt_tokens < 0 or generated_tokens < 0:
        raise ValueError('模型、backend 和 token 配置必须合法')
    # TODO 2 先检查 batch_size 和 token 字段，再计算 total_tokens；只累加 prompt 和 generated token，不重复计算 batch。
    total_tokens = prompt_tokens + generated_tokens
    return {
        'model_name': model_name,
        'backend': backend,
        'batch_size': batch_size,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens,
        'total_tokens': total_tokens,
        'dtype': dtype,
        'cache_policy': cache_policy,
    }

# TODO 3: 汇总 prefill / decode 延迟
def summarize_prefill_decode(prefill_ms, decode_ms, generated_tokens):
    """把 prefill/decode 成本整理为 TTFT、TPOT 和阶段占比。"""
    if prefill_ms < 0 or decode_ms < 0 or generated_tokens < 0:
        raise ValueError('延迟和 token 数不能为负数')
    # TODO 3 对应变量 total_ms：连接 prefill 与 decode 的总时延。
    total_ms = prefill_ms + decode_ms
    # TODO 3 对应变量 ttft_ms / tpot_ms：分别表达首 token 和输出 token 延迟。
    ttft_ms = prefill_ms
    tpot_ms = decode_ms / generated_tokens if generated_tokens else 0.0
    # TODO 3 对应变量 prefill_share / decode_share：在 total_ms 上归一化。
    prefill_share = prefill_ms / total_ms if total_ms else 0.0
    decode_share = decode_ms / total_ms if total_ms else 0.0
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total_ms, 2),
        'ttft_ms': round(ttft_ms, 2),
        'tpot_ms': round(tpot_ms, 4),
        'prefill_share': round(prefill_share, 3),
        'decode_share': round(decode_share, 3),
    }

# TODO 4: 计算推理项目核心指标
def compute_inference_metrics(config, latency_summary, peak_mem_mb):
    """将阶段延迟、吞吐和显存整理为统一项目指标。"""
    if config['batch_size'] <= 0 or peak_mem_mb < 0:
        raise ValueError('batch_size 和 peak_mem_mb 必须合法')
    # TODO 4 对应变量 output_tokens：按 batch_size 计算整个 workload 的输出 token 数。
    output_tokens = config['batch_size'] * config['generated_tokens']
    total_seconds = latency_summary['total_ms'] / 1000.0
    # TODO 4 对应变量 throughput_tok_s：用总时延换算整个 batch 的 token/s。
    throughput_tok_s = output_tokens / total_seconds if total_seconds else 0.0
    return {
        'backend': config['backend'],
        'batch_size': config['batch_size'],
        'prompt_tokens': config['prompt_tokens'],
        'generated_tokens': config['generated_tokens'],
        'ttft_ms': latency_summary['ttft_ms'],
        'tpot_ms': latency_summary['tpot_ms'],
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': latency_summary['total_ms'],
        'prefill_share': latency_summary['prefill_share'],
        'decode_share': latency_summary['decode_share'],
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 5: 诊断推理瓶颈
def diagnose_inference_bottleneck(metrics, memory_budget_mb=None):
    """依据显存压力和阶段占比给出规则化瓶颈分类。"""
    if memory_budget_mb is not None and memory_budget_mb <= 0:
        raise ValueError('memory_budget_mb 必须为正数')
    # TODO 5 对应变量 memory_pressure：显存接近预算时优先判定资源瓶颈。
    if memory_budget_mb is not None and metrics['peak_mem_mb'] >= 0.9 * memory_budget_mb:
        bottleneck = 'memory-bound'
        reason = 'peak memory 接近预算，优先检查 KV cache、batch size、量化和分页策略。'
    # TODO 5 对应变量 prefill_heavy / decode_heavy：仅在没有显存压力时判断阶段占比。
    elif metrics['prefill_share'] >= 0.6:
        bottleneck = 'prefill-bound'
        reason = 'prefill 占比高，优先检查 prompt length、FlashAttention、chunked prefill 和 batching。'
    elif metrics['decode_share'] >= 0.6:
        bottleneck = 'decode-bound'
        reason = 'decode 占比高，优先检查 KV cache 读写、decode scheduling、speculative decoding 或 multi-token decoding。'
    else:
        bottleneck = 'balanced'
        reason = 'prefill、decode 和显存压力都不突出，先保持 baseline 或继续做细粒度 profiling。'
    return {'bottleneck': bottleneck, 'reason': reason}

# TODO 6: 比较 baseline 和 candidate
def compare_inference_candidates(baseline_metrics, candidate_metrics):
    """计算 candidate 相对 baseline 的延迟、显存和吞吐变化。"""
    if baseline_metrics['throughput_tok_s'] <= 0:
        raise ValueError('baseline throughput 必须大于 0，才能计算相对增益')
    # TODO 6 的核心结果是 total_latency_delta_ms 和 throughput_gain；TTFT、TPOT、显存差值沿用同一规则派生。
    total_latency_delta_ms = baseline_metrics['total_ms'] - candidate_metrics['total_ms']
    ttft_delta_ms = baseline_metrics['ttft_ms'] - candidate_metrics['ttft_ms']
    tpot_delta_ms = baseline_metrics['tpot_ms'] - candidate_metrics['tpot_ms']
    # 显存差值同样使用 baseline - candidate，保持指标方向一致。
    peak_mem_delta_mb = baseline_metrics['peak_mem_mb'] - candidate_metrics['peak_mem_mb']
    # TODO 6 对应变量 throughput_gain：使用 candidate / baseline - 1 的比例增益。
    throughput_gain = candidate_metrics['throughput_tok_s'] / baseline_metrics['throughput_tok_s'] - 1.0
    return {
        'total_latency_delta_ms': round(total_latency_delta_ms, 2),
        'ttft_delta_ms': round(ttft_delta_ms, 2),
        'tpot_delta_ms': round(tpot_delta_ms, 4),
        'peak_mem_delta_mb': round(peak_mem_delta_mb, 2),
        'throughput_gain': round(throughput_gain, 4),
    }

# TODO 7: 输出推理选型建议
def recommend_inference_decision(comparison, candidate_bottleneck, min_throughput_gain=0.1, max_ttft_regression_ms=20.0):
    """根据吞吐收益、TTFT 退化和剩余瓶颈输出选型建议。"""
    # TODO 7 对应变量 throughput_good：检查吞吐增益门槛。
    throughput_good = comparison['throughput_gain'] >= min_throughput_gain
    # TODO 7 对应变量 ttft_ok：把正向 delta 转成回退量后检查上限。
    ttft_regression_ms = -comparison['ttft_delta_ms']
    ttft_ok = ttft_regression_ms <= max_ttft_regression_ms
    # TODO 7 对应变量 still_tunable：根据候选瓶颈判断是否还有调优方向。
    still_tunable = candidate_bottleneck['bottleneck'] != 'balanced'
    if throughput_good and ttft_ok:
        decision = 'accept'
        reason = 'candidate 吞吐提升明显，TTFT 退化在可接受范围内，值得进入正式推理方案。'
    elif comparison['throughput_gain'] > 0.0 and still_tunable:
        decision = 'tune'
        reason = 'candidate 已有收益，但瓶颈仍然存在，继续围绕诊断结果调参或换策略。'
    else:
        decision = 'reject'
        reason = 'candidate 收益不足或交互延迟退化明显，当前不值得切换。'
    return {'decision': decision, 'reason': reason}

baseline_config = build_inference_config('tiny-llama', 'pytorch-eager', 2, 128, 32, 'fp16', 'static-kv-cache')
baseline_latency = summarize_prefill_decode(prefill_ms=80.0, decode_ms=160.0, generated_tokens=32)
baseline_metrics = compute_inference_metrics(baseline_config, baseline_latency, peak_mem_mb=4096.0)
print(baseline_config)
print(baseline_metrics)
print(diagnose_inference_bottleneck(baseline_metrics, memory_budget_mb=8192.0))

candidate_config = build_inference_config('tiny-llama', 'paged-attention', 2, 128, 32, 'fp16', 'paged-kv-cache')
candidate_latency = summarize_prefill_decode(prefill_ms=85.0, decode_ms=120.0, generated_tokens=32)
candidate_metrics = compute_inference_metrics(candidate_config, candidate_latency, peak_mem_mb=3584.0)
comparison = compare_inference_candidates(baseline_metrics, candidate_metrics)
print(candidate_metrics)
print(comparison)
print(recommend_inference_decision(comparison, diagnose_inference_bottleneck(candidate_metrics, memory_budget_mb=8192.0)))

```

### 解析

**1. TODO 1: 模拟请求执行**
- **实现方式**：按 `concurrency` 将请求分成多个执行批次；同一批并行完成，批次耗时取其中最长请求的 prefill + decode 时间。
- **关键点**：后续批次会产生 queue time；`TTFT` 包含排队和 prefill，`TPOT` 只表示 decode 阶段的平均每 token 时间。
- **项目意义**：CPU 可以解释并发、排队和阶段指标的关系，但这些是教学成本模型，不是 vLLM / SGLang 的真实调度或 CUDA 测量。
- **显存扩展**：当 `kv_cache_mb_per_token > 0` 时，模拟器按 `prompt_tokens + generated_tokens` 估算每个请求的 KV Cache 增量，并按同一执行波次累加；这只能说明 token 数、并发与容量之间的关系，不能替代 backend 的 allocated/reserved 显存或 OOM 测量。

**2. TODO 2: 汇总推理 workload 配置**
- **实现方式**：把模型、backend、batch size、prompt tokens、generated tokens、dtype 和 cache policy 放进同一个配置对象。
- **关键点**：推理 benchmark 的第一原则是固定 workload。没有 workload，TTFT、TPOT、吞吐和显存都没有可比性。
- **项目意义**：后续 baseline 和 candidate 只能改一个变量，否则很难判断收益来自哪里。

**3. TODO 3: 汇总 prefill / decode 延迟**
- **实现方式**：`total_ms = prefill_ms + decode_ms`，TTFT 近似取 `prefill_ms`，TPOT 取 `decode_ms / generated_tokens`。
- **关键点**：prefill 和 decode 的瓶颈不同。总耗时下降不代表交互体验一定变好，TTFT 和 TPOT 必须拆开看。
- **项目意义**：这一步把推理性能从一个笼统 latency 拆成可诊断的两段。

**4. TODO 4: 计算推理项目核心指标**
- **实现方式**：用 `batch_size * generated_tokens / total_seconds` 计算 generated tokens/s，并和 TTFT、TPOT、total latency、peak memory 放在同一张账本里。
- **关键点**：throughput 统计的是整个 batch 的输出 token 产出，不是单条请求的 token 数。
- **项目意义**：同一个 candidate 可能吞吐更高但 TTFT 更差，指标必须一起看。

**5. TODO 5: 诊断推理瓶颈**
- **实现方式**：显存接近预算时优先判为 `memory-bound`；否则用 prefill/decode 占比判断主要瓶颈。
- **关键点**：显存预算是硬约束。如果显存已经接近上限，即使 decode 占比高，也要先处理 KV cache、batch size 或量化。
- **项目意义**：诊断结果决定下一步选 FlashAttention、chunked prefill、PagedAttention、KV cache 量化还是 decode scheduling。

**6. TODO 6: 比较 baseline 和 candidate**
- **实现方式**：latency、TTFT、TPOT 和 peak memory 使用 `baseline - candidate`，正数表示 candidate 更好；throughput 使用比例增益。
- **关键点**：delta 的方向要固定，否则报告容易把退化误写成收益；baseline throughput 为 0 时无法计算相对增益，应明确抛出输入错误。
- **项目意义**：项目报告不只写绝对值，更要说明 candidate 相比 baseline 改善或退化了多少。

**7. TODO 7: 输出推理选型建议**
- **accept**：吞吐提升达标，TTFT 退化在可接受范围内，说明候选方案值得采用。
- **tune**：candidate 有收益，但瓶颈仍然存在，需要继续沿诊断方向调参。
- **reject**：candidate 收益不足，或交互延迟退化明显，当前不值得切换。
- **项目意义**：推理选型不能只看一个指标。最终结论要同时考虑 workload、吞吐、TTFT、TPOT、显存和瓶颈类型。

**推理性能对比的实验原则**
- **变量控制**：同一轮对比中只改一个变量，例如 batch size、precision、推理后端或 cache 策略。
- **指标闭环**：每次实验至少记录 TTFT、TPOT、throughput 和 peak memory。
- **阶段拆分**：把 prefill 和 decode 分开看，避免把长 prompt 问题误判成 decode 问题。
- **结果复盘**：最终输出要回扣 Step 1 的问题：在给定约束下，哪种推理策略最划算，理由是什么。

### Step 5：GPU/backend 主实验（可选）——真实基线与候选对照

#### 5.1 环境、输入与固定条件

GPU/backend 的环境安装和平台差异见[使用指南：Part 02 环境分层与决策树](../guide.md#part-02-环境分层与决策树)。本小节先确定实验对象、输入和固定条件，不执行服务启动或 benchmark。

本节只比较端到端服务表现；量化、Prefix Cache、Speculative Decoding、Scheduler 和 MLA / KV Cache 的机制与专项实验分别在后续专题中展开。固定条件的目的，是让后续差异能够归因于明确的 backend 或 workload 变量。

| 实验要素 | 本节固定内容 | 允许改变的内容 | 输出 |
|:---|:---|:---|:---|
| 模型与请求 | 模型版本、请求集、输入长度、输出长度、生成参数 | 仅在单变量实验中改变 | 可复用 workload |
| 服务条件 | backend、dtype、最大上下文长度、显存使用上限 | 由 G1 明确选择一个变量 | 可比较的服务配置 |
| 实验分组 | baseline、单变量对照、已验证策略组合 | 单变量对照每次只改变一个主要变量 | 结果分组与元数据 |
| 核心指标 | TTFT、TPOT、E2E、请求/输出吞吐、峰值显存、成功率和 OOM | 根据目标补充质量指标 | 端到端证据 |

![66 机制、backend 与指标关系](../public/02_PyTorch_Algorithms/66_mechanism_backend_mapping.svg)

#### 公共基线输出契约

G0 的 JSON 是 68–72 项目共用的推理基线输入，不是某个专项策略的最终结论。后续项目可以在同一 workload 上读取它，再提交 candidate 结果；因此每次实测都要保留身份、环境、指标和失败状态。

| 契约块 | 至少记录 | 用途 |
|:---|:---|:---|
| 身份 | `project`、`role`、`model_revision`、`backend`、`result_json` | 判断 baseline / candidate 是否可比较 |
| workload | `workload_path`、`num_prompts`、`batch`、`concurrency`、`warmup`、`repeats` | 复现实验输入与请求压力 |
| 环境 | `dtype`、`backend_version`、`hardware`、`runtime_version`、`cache_policy` | 解释结果适用范围 |
| 证据与决策 | `evidence_level`、`quality`、`failure`、`decision` | 区分模拟、smoke、实测及 accept / tune / reject |

68–70 重点读取 TTFT、TPOT、throughput、P95/P99、queue wait、peak memory 和各自的 acceptance / hit rate；71–72 还应补充模型或部署 artifact。未采集的字段写明 `not_collected`，失败记录保留原因和复测路径。
#### 5.2 确认运行环境可启动

在开始真实 benchmark 前，先确认 Notebook client、独立 backend、GPU、驱动和 CUDA 能够协同工作。下表记录当前已验证组合；更换 vLLM、CUDA、驱动或 GPU 后，应重新完成预检和 smoke test。

表格中的最后一列说明每项检查对启动和结果解释的影响。若 backend 暂时无法启动，可以先运行 CPU-first 模拟 benchmark，但模拟结果只能用于理解指标和逻辑，不能替代真实 GPU 性能。

| 类别 | 检查项 | 当前已验证配置 | 如何解读 |
|:---|:---|:---|:---|
| 系统 | OS | Linux x86_64 | vLLM 的主要支持环境 |
| 硬件 | GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU（SM120 / Blackwell），约 12 GB | 适合小模型 smoke test |
| 驱动 | NVIDIA driver / CUDA | 570.211.01 / CUDA 12.8 | 不要与 CUDA 13.0 wheel 混用；更换后需重做预检 |
| Client | Python 环境 / PyTorch | `llm_algo` / PyTorch 2.11.0+cu128 | 运行 Notebook 和 benchmark client |
| Backend | 服务环境 / vLLM | `vllm_legacy_cu128` / Python 3.12 / PyTorch 2.8.0+cu128 / vLLM 0.11.0 | 通过 HTTP 与 client 解耦；0.11.0 是本机验证版本，不是最低版本要求 |
| 模型 | smoke test 模型 | `Qwen/Qwen2.5-0.5B-Instruct`，权重约 0.92 GiB | 需要模型仓库网络或本地缓存 |
| 启动 | dtype 与服务参数 | `bfloat16`、`max_model_len=2048`、`gpu_memory_utilization=0.8`、`--enforce-eager` | 本机需要关闭 TorchInductor/CUDAGraph |
| 运行前提 | 网络、磁盘、端口、进程权限 | 可访问 HuggingFace / ModelScope、缓存空间、默认端口 8000 | 不满足时保留 CPU-first 路径 |
| 平台差异 | Colab 参考 | T4 优先尝试 `float16`；L4/A100/H100 根据预检选择 `bfloat16` | 运行时或驱动变化后重新确认 GPU 名称与 CUDA |
| 证据范围 | 当前实测路径 | `--enforce-eager`；FlashInfer 不可用时回退 PyTorch-native sampler | 结果只代表当前 eager 配置，不等同于最新版 backend 默认性能 |

```python
"""只检查当前 Notebook 内核和 backend 命令，不启动服务、不下载模型。"""
import importlib.util
import shutil
import torch

# 1. 检查 Notebook client 使用的 PyTorch、CUDA 和 GPU。
print({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()})
if torch.cuda.is_available():
    print({'device': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0), 'bf16_supported': torch.cuda.is_bf16_supported()})
# 2. 检查 vLLM 是否安装在当前内核；独立 backend 环境可以显示 False。
print({'vllm_on_current_kernel': importlib.util.find_spec('vllm') is not None, 'vllm_command': shutil.which('vllm')})

# 如果 vLLM 在独立 conda 环境中运行，这里可以保持 vllm_on_current_kernel=False；
# Step 5 的运行单元会通过 VLLM_ENV 调用独立环境，或复用已启动的 OpenAI-compatible API。
```

#### 5.3 配置实验条件

根据 5.1 的固定条件选择模型、workload、dtype、G0/G1/G2 和结果路径；本单元只填写配置，不启动服务。

```python
# 实验配置：先运行本单元，再运行下面的环境预检和 benchmark。
# 本单元只设置变量并检查 G0/G1/G2 的填写，不下载模型、不启动 backend。
RUN_REAL_BACKEND = False  # 是否启动真实 vLLM；False 只完成 CPU-first 模板。
BACKEND = 'vllm'  # Step 5 的主基线 backend；Step 6 会显式写入 sglang。
QUANTIZATION_FORMAT = 'none'  # 66 只提供浮点 baseline；真实量化 artifact 由 67 接入。
QUANTIZATION_ARTIFACT = None
EXPERIMENT_GROUP = 'G0'  # G0=基线，G1=单变量候选，G2=扩展或组合对照。
STRATEGIES = []  # G0 为空；G1 填一个改变项；G2 填多个候选或已单独验证的组合项。
STRATEGY_NOTE = '固定 workload 的 vLLM baseline'  # 说明本次实验实际改变了什么。
MODEL_SOURCE = 'auto'  # 模型来源：auto / modelscope / huggingface / local。
MODEL_CACHE_DIR = 'model_cache'  # 模型缓存目录；通常不需要修改。
MODEL_PROFILES = {
    'qwen25_small': 'Qwen/Qwen2.5-0.5B-Instruct',
    'deepseek_r1_small': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
}
MODEL_PROFILE = 'qwen25_small'  # 先用小模型完成 smoke test。
MODEL_ID = MODEL_PROFILES[MODEL_PROFILE]  # 实际加载的模型 ID。
MODEL_REVISION = MODEL_ID  # 正式复测时建议改为 commit/tag，避免模型别名漂移。
DTYPE = 'auto'  # auto 根据 GPU 选择；也可显式写 bfloat16 / float16。
CACHE_POLICY = 'default'  # vLLM / SGLang 使用的 cache 标记；跨 backend 对比时保持一致。
VLLM_COMMAND = None  # 为空时自动查找当前环境中的 vllm
VLLM_ENV = None  # 云端保持当前 runtime；本地多环境时再填写环境名
MAX_MODEL_LEN = 2048  # 最大上下文长度；越大越占 KV Cache。
GPU_MEMORY_UTILIZATION = 0.8  # vLLM 使用显存比例；需为桌面和其他进程留余量。
ENFORCE_EAGER = True  # 先保证 RTX 50 系列等架构可复现；稳定后可尝试 False
WORKLOAD_PATH = 'benchmarks/workloads/fixed.jsonl'  # G1 workload 变量通过替换此文件实现。
NUM_PROMPTS = 5  # 请求总数；正式实验应大于 smoke test。
BATCH_SIZE = 1  # 单请求 batch 配置；修改后必须在结果中保留。
CONCURRENCY = 1  # 同时在途请求数；只做并发实验时改变它。
WARMUP = 1  # 预热请求数；正式实验建议提高到 3-10。
RUN_ID = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
REPEATS = 1  # 正式结果建议增加重复次数；smoke test 可保持 1。
BACKEND_VERSION = 'record_at_runtime'  # 运行后填写实际 backend 版本。
HARDWARE = 'record_at_runtime'  # 运行后填写 GPU 型号、显存和数量。
RUNTIME_VERSION = 'record_at_runtime'  # 运行后填写 CUDA / PyTorch / driver 等关键版本。
EVIDENCE_LEVEL = 'real_backend_smoke' if RUN_REAL_BACKEND else 'simulation'
QUALITY_STATUS = 'not_collected'  # 未采集质量指标时必须显式标记。
FAILURE_REASON = None  # 失败、unsupported 或 OOM 时填写原因。
RETEST_PATH = None  # 失败记录或待复测结果的路径。
DECISION = 'pending'  # accept / tune / reject；由结果分析后填写。
RESULT_PATH = f'benchmarks/results/66_{EXPERIMENT_GROUP.lower()}_{BACKEND}_{RUN_ID}.json'  # 自动生成，避免覆盖历史结果。
BASELINE_POINTER_PATH = 'benchmarks/results/66_g0_vllm_baseline.json'  # 仅作为 67 的最新 G0 入口，不代表历史文件。
PEAK_MEMORY_MB = None  # 可选：由外部 nvidia-smi/监控采集后填入；None 表示本次未测 GPU 峰值显存。

SUPPORTED_AUTO_STRATEGIES = {'concurrency', 'batch', 'dtype', 'workload'}  # 当前 vLLM 入口确实能执行的 G1 变量。
if EXPERIMENT_GROUP not in {'G0', 'G1', 'G2'}:
    raise ValueError('EXPERIMENT_GROUP 只能是 G0、G1 或 G2')
if EXPERIMENT_GROUP == 'G0' and STRATEGIES:
    raise ValueError('G0 baseline 不应填写 STRATEGIES')
if EXPERIMENT_GROUP == 'G1' and len(STRATEGIES) != 1:
    raise ValueError('G1 单变量对照必须填写一个策略')
if EXPERIMENT_GROUP == 'G2' and len(STRATEGIES) < 2:
    raise ValueError('G2 策略融合至少填写两个策略')
if RUN_REAL_BACKEND and EXPERIMENT_GROUP == 'G1' and not set(STRATEGIES).issubset(SUPPORTED_AUTO_STRATEGIES):
    raise ValueError('当前 GPU 入口只自动执行 concurrency / batch / dtype / workload 对照；FlashAttention、Prefix Cache、量化等请使用专项项目或手动 backend 参数。')
if RUN_REAL_BACKEND and EXPERIMENT_GROUP == 'G2':
    raise ValueError('G2 扩展或组合对照目前只有配置与报告元数据入口，尚未自动启用组合 backend；请先完成单项策略验证。')
EXPERIMENT_METADATA = {'group': EXPERIMENT_GROUP, 'role': 'baseline' if EXPERIMENT_GROUP == 'G0' else 'candidate', 'backend': BACKEND, 'strategies': list(STRATEGIES), 'note': STRATEGY_NOTE}
BASELINE_CONTRACT = {
    'schema_version': 'inference-baseline/v1',
    'project': '66_inference_performance_comparison',
    'experiment_group': EXPERIMENT_GROUP,
    'role': 'baseline' if EXPERIMENT_GROUP == 'G0' else 'candidate',
    'model_revision': MODEL_REVISION,
    'backend': BACKEND,
    'quantization_format': QUANTIZATION_FORMAT,
    'quantization_artifact': QUANTIZATION_ARTIFACT,
    'backend_version': BACKEND_VERSION,
    'hardware': HARDWARE,
    'runtime_version': RUNTIME_VERSION,
    'workload_path': WORKLOAD_PATH,
    'num_prompts': NUM_PROMPTS,
    'dtype': DTYPE,
    'cache_policy': CACHE_POLICY,
    'batch': BATCH_SIZE,
    'concurrency': CONCURRENCY,
    'warmup': WARMUP,
    'repeats': REPEATS,
    'evidence_level': EVIDENCE_LEVEL,
    'quality': {'status': QUALITY_STATUS},
    'failure': {'status': 'not_observed' if FAILURE_REASON is None else 'recorded', 'reason': FAILURE_REASON, 'retest_path': RETEST_PATH},
    'decision': DECISION,
    'result_json': RESULT_PATH,
    'latest_baseline_pointer': BASELINE_POINTER_PATH if EXPERIMENT_GROUP == 'G0' and BACKEND == 'vllm' else None,
}
print('实验配置：', EXPERIMENT_METADATA)

```

#### 5.4 执行实验、保存结果与排障

完成环境预检和配置后，依次启动 backend、运行 workload、补充实验元数据，并将每组结果独立保存为 JSON。Notebook 内核负责发起请求和保存报告；vLLM backend 可以运行在当前环境，也可以运行在单独环境中。

模型下载和服务启动会消耗显存、磁盘与时间；实验完成后需要确认结果文件已经保存，并释放 backend 子进程。自动启动失败时，再使用代码块后的手动启动与排障说明。

| 执行阶段 | 主要动作 | 阶段产出 |
|:---|:---|:---|
| 模型与 backend 启动 | 解析模型、选择端口、启动服务并等待就绪 | 可访问的 API 服务 |
| benchmark 请求 | 使用固定 workload 完成 warmup 和正式测量 | TTFT、TPOT、E2E、吞吐、成功率 |
| 元数据补充 | 写入 backend、dtype、实验组、启动参数和 evidence level | 可追溯实验记录 |
| JSON 保存与清理 | 独立保存结果，检查字段和异常，停止服务 | 可复核 JSON 与释放后的环境 |

![66 GPU/backend 实验流程](../public/02_PyTorch_Algorithms/66_gpu_backend_experiment_flow.svg)

```python
"""执行一次 vLLM 实验：解析模型、启动服务、运行 benchmark、保存 JSON 并清理进程。"""
import json
import os
import subprocess
import sys
from pathlib import Path

def write_backend_failure(error, stage):
    """Persist a failure record so startup/OOM errors remain reviewable."""
    failure_path = Path(RESULT_PATH)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure = {
        'schema_version': 'inference-benchmark/v1',
        'project': '66_inference_performance_comparison',
        'experiment': EXPERIMENT_METADATA,
        'experiment_contract': BASELINE_CONTRACT,
        'failure': {'status': 'recorded', 'stage': stage, 'reason': str(error), 'retest_path': str(failure_path)},
        'evidence_level': EVIDENCE_LEVEL,
        'decision': 'tune',
    }
    failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')

if RUN_REAL_BACKEND:
    project_root = next((path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'tools').is_dir()), None)
    if project_root is None:
        raise RuntimeError('未找到项目根目录。请从仓库根目录启动 Jupyter，或把仓库根目录加入 sys.path。')
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tools.backend_runtime import resolve_model, start_vllm, stop_backend

    try:
        # 1. 定位项目根目录和模型缓存；模型只在首次运行时下载。
        model_path = resolve_model(MODEL_ID, MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
        # 2. 启动 vLLM，自动选择可用端口并等待服务就绪。
        server, server_log, port, selected_dtype = start_vllm(
            model_path, DTYPE, vllm_command=VLLM_COMMAND,
            vllm_environment=VLLM_ENV,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=ENFORCE_EAGER,
            served_model_name=MODEL_ID,
        )
    except Exception as exc:
        write_backend_failure(exc, 'model_resolution_or_backend_start')
        raise
    try:
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        hardware_snapshot = {'device': device_name, 'cuda': torch.version.cuda, 'torch': torch.__version__}
    except Exception:
        hardware_snapshot = {'device': 'unknown', 'cuda': 'unknown', 'torch': 'unknown'}
    BASELINE_CONTRACT.update({'hardware': hardware_snapshot, 'runtime_version': hardware_snapshot})
    print({'model_path': model_path, 'dtype': selected_dtype, 'port': port, 'runtime': hardware_snapshot})

    try:
        output_path = Path(RESULT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 3. 使用固定 workload 发起请求；G0/G1 的差异来自配置单元。
        benchmark_command = [
            sys.executable, 'tools/benchmark_inference_backend.py',
            '--base-url', f'http://127.0.0.1:{port}',
            '--model', MODEL_ID,
            '--label', f'vllm-{EXPERIMENT_GROUP.lower()}',
            '--project', '66',
            '--backend', 'vllm',
            '--dtype', selected_dtype,
            '--batch', str(BATCH_SIZE),
            '--cache-policy', 'default',
            '--workload', WORKLOAD_PATH,
            '--num-prompts', str(NUM_PROMPTS),
            '--concurrency', str(CONCURRENCY),
            '--warmup', str(WARMUP),
            '--repeats', str(REPEATS),
            '--output', str(output_path),
        ]
        if PEAK_MEMORY_MB is not None:
            benchmark_command.extend(['--peak-memory-mb', str(PEAK_MEMORY_MB)])
        subprocess.run(benchmark_command, check=True)
        # 4. 追加实验元数据后保存报告，确保每组结果可以单独复核。
        saved = json.loads(output_path.read_text(encoding='utf-8'))
        saved['experiment'] = EXPERIMENT_METADATA
        saved['experiment_contract'] = BASELINE_CONTRACT
        saved['baseline_contract'] = BASELINE_CONTRACT  # 兼容已有 68–72 读取路径。
        saved.setdefault('quality', {'status': QUALITY_STATUS})
        saved.setdefault('failure', {'status': 'not_observed' if FAILURE_REASON is None else 'recorded', 'reason': FAILURE_REASON, 'retest_path': RETEST_PATH})
        saved.setdefault('decision', DECISION)
        serialized = json.dumps(saved, ensure_ascii=False, indent=2)
        output_path.write_text(serialized, encoding='utf-8')
        if EXPERIMENT_GROUP == 'G0' and BACKEND == 'vllm':
            Path(BASELINE_POINTER_PATH).write_text(serialized, encoding='utf-8')
        print(saved['metrics'])
        print('统一结果：', json.dumps(saved['normalized_result'], ensure_ascii=False, indent=2))
    except Exception as exc:
        write_backend_failure(exc, 'benchmark_or_result_write')
        raise
    finally:
        # 5. 无论 benchmark 是否成功，都停止服务并释放子进程。
        stop_backend(server, server_log)
else:
    print('跳过真实 backend：保持 CPU-first 模式。')

```

#### 手动启动与排障（可选）

如果自动启动流程因 backend 环境或模型路径问题无法运行，可以在终端手动启动服务，再执行 benchmark 脚本进行排查。Notebook 主流程不依赖手动查端口或拼接命令。若看到 `ModuleNotFoundError: No module named 'tools'`，通常是旧版本在仓库根目录尚未加入 `sys.path` 前就导入了工具模块；当前代码会先定位项目根目录。

```bash
vllm serve <model-id> --dtype bfloat16 --port 8000
python tools/benchmark_inference_backend.py
```
#### 5.5 实测记录与结果表

读取 5.4 保存的 JSON，先检查配置一致性、baseline 与单变量对照的完整性和 evidence level，再整理 TTFT、TPOT、吞吐、峰值显存、OOM 与质量。跨 backend 比较放到 Step 7。

当前本机历史记录只用于展示报告填写方式；正式结论应以学习者在自己的 GPU、模型和 workload 上重新采集的结果为准。


| 类别 | 项目 | baseline | 单变量对照 | artifact / JSON | failure | decision | 说明 |
|:---|:---|:---:|:---:|:---|:---|:---|:---|
| 环境 | GPU / 显存 | RTX 5070 Ti Laptop / 12 GB | 同左 | none / `66_vllm_real.json` | none | pending | 记录硬件差异 |
| 环境 | 模型 | `Qwen/Qwen2.5-0.5B-Instruct` | 同左 | none / 独立 JSON | none | pending | 同一对照保持一致 |
| 环境 | backend / runtime | vLLM 0.11.0 / PyTorch 2.11.0+cu128 | 同左 | none / 独立 JSON | none | pending | 记录 backend 与 client 版本 |
| 条件 | dtype / workload | `bfloat16` / `fixed.jsonl` | 同左 | none / 独立 JSON | none | pending | 生成长度 64 tokens |
| 条件 | 启动参数 | `max_model_len=2048`、`gpu_memory_utilization=0.8`、`--enforce-eager` | 同左 | none / 独立 JSON | none | pending | 改变时必须注明 |
| 变量 | concurrency | 1 | 4 | none / 独立 JSON | none | pending | 单变量对照 |
| 结果 | 成功 / 失败 | 5 / 0 | 5 / 0 | none / 独立 JSON | none | pending | 必须保留异常状态 |
| 结果 | 请求吞吐 | 3.1189 req/s | 4.2972 req/s | none / 独立 JSON | none | pending | 越高越好 |
| 结果 | 输出吞吐 | 182.1461 token/s | 250.9544 token/s | none / 独立 JSON | none | pending | 越高越好 |
| 结果 | TTFT P50 | 33.776 ms | 234.566 ms | none / 独立 JSON | none | pending | 越低越好 |
| 结果 | TPOT P50 | 4.859 ms/token | 11.528 ms/token | none / 独立 JSON | none | pending | 越低越好 |
| 结果 | E2E P50 | 337.176 ms | 956.383 ms | none / 独立 JSON | none | pending | 越低越好 |
| 结果 | peak memory | 未采集 | 未采集 | none / 独立 JSON | not_collected | tune | 不能据此下显存结论 |
| 结果 | evidence level | real backend smoke | real backend smoke | none / 独立 JSON | none | pending | 仅支持当前 workload 对照 |
| 文件 | 结果路径 | `66_vllm_real.json` | `66_vllm_concurrency4.json` | JSON 自身路径 | none | pending | 每组独立保存 |
#### 5.6 解释结果与形成决策

如何解读这组历史结果：并发提高后，输出吞吐由 182.15 提升到 250.95 token/s，但 TTFT P50 由 33.78 ms 增至 234.57 ms，E2E P50 由 337.18 ms 增至 956.38 ms。说明批处理提高吞吐的同时增加了交互延迟；当前只有少量 smoke test 请求，P99 只能作为记录，不能形成稳定线上结论。

当前结论仅支持固定 workload 下的延迟和吞吐对照；历史 JSON 没有 `peak_memory` 字段，因此不能反推出 KV Cache 或并发显存结论。若目标是交互式单请求，应优先关注 TTFT/E2E；若目标是批量吞吐，再扩大 workload 和并发 sweep，并补采 GPU 显存。

复测时至少补齐 GPU、模型、backend 版本、workload、并发、TTFT、TPOT、E2E、吞吐、峰值显存、成功率、OOM 和 evidence level；只有证据完整且目标指标达到要求，才输出 `accept`，否则进入 `tune` 或 `reject`。
### Step 6：SGLang backend 对照

#### 6.1 环境、输入与固定条件

本步要回答：在相同模型、请求集、生成长度、并发和 dtype 下，SGLang 与 vLLM 的端到端表现是否不同？相对于 vLLM G0，SGLang 是只改变 backend 的 G1 candidate。SGLang 的 RadixAttention / Radix Cache 由 backend 自己实现，本节不重新实现其内部算法，只验证服务链路并读取统一指标。

![推理引擎与统一实验关系](../public/02_PyTorch_Algorithms/66_inference_engines_comparison.svg)

#### 6.2 环境启动检查

先确认当前环境已安装并验证 SGLang，再填写启动命令模板。命令不会由 Notebook 猜测，避免把不同版本的启动参数混在一起。

#### 6.3 配置实验条件

沿用 Step 5 的模型、workload、生成长度、并发和 dtype；只替换 backend，并为 SGLang 结果使用独立 JSON 路径。

执行配置完成后，进入 6.4 启动服务并运行同一 benchmark。

输入是已验证的 `SGLANG_COMMAND_TEMPLATE`、模型、workload 和并发配置；输出是 backend 名称、dtype、请求成功率、TTFT、TPOT、E2E、吞吐和结果文件。只有 vLLM 与 SGLang 的硬件、模型、输入、输出长度和并发完全一致时，Step 7 才能进行跨 backend 对比。

<div align="center"><strong>不同引擎可以共用 benchmark 口径，但必须先确认服务接口、硬件和实验条件一致。</strong></div>

```python
# Step 6 专属配置：不影响 Step 5 的 vLLM 主实验。
RUN_SGLANG = False  # 只有在确认 SGLang 环境可用后再改为 True。
SGLANG_COMMAND_TEMPLATE = None  # 例如 python -m sglang.launch_server --model-path {model_path} --port {port}
SGLANG_READY_TIMEOUT_S = 180  # SGLang 冷启动等待时间；超时会自动停止子进程。
SGLANG_RESULT_PATH = 'benchmarks/results/66_g1_sglang_backend.json'  # G1 candidate，与 vLLM G0 分开保存。
```

#### 6.4 执行对照并保存 JSON

复用 Step 5 的 workload 和 benchmark 入口，只替换 SGLang 服务地址；完成 warmup 后运行正式请求，并将结果保存到独立 JSON。unsupported、启动失败和请求失败都要保留在结果记录中。

```python
"""可选的 SGLang 对照：复用统一 workload，独立启动服务并保存 JSON。"""
# 默认关闭，不影响 vLLM 主线。
if RUN_SGLANG:
    if not SGLANG_COMMAND_TEMPLATE:
        raise ValueError('RUN_SGLANG=True 时必须填写 SGLANG_COMMAND_TEMPLATE，并包含 {model_path} 和 {port}。')
    from tools.inference_project_runtime import (
        locate_repo_root, run_backend_benchmark, start_external_openai_backend,
    )
    from tools.backend_runtime import find_free_port, resolve_model, stop_backend
    root = locate_repo_root()
    model_path = resolve_model(MODEL_ID, MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
    sglang_port = find_free_port()
    sglang_log = root / 'benchmarks/results/66_sglang.log'
    sglang_server, sglang_log_path = start_external_openai_backend(
        SGLANG_COMMAND_TEMPLATE, model_path=str(model_path), port=sglang_port,
        log_path=sglang_log, ready_timeout_s=SGLANG_READY_TIMEOUT_S,
    )
    try:
        sglang_report = run_backend_benchmark(
            project='66', base_url=f'http://127.0.0.1:{sglang_port}',
            model=MODEL_ID, label='sglang-g1-backend', output=SGLANG_RESULT_PATH,
            workload=WORKLOAD_PATH, num_prompts=NUM_PROMPTS,
            concurrency=CONCURRENCY, warmup=WARMUP, backend='sglang',
            dtype=DTYPE, batch=BATCH_SIZE, cache_policy=CACHE_POLICY,
        )
        sglang_report['experiment'] = {**EXPERIMENT_METADATA, 'group': 'G1', 'role': 'candidate', 'backend': 'sglang', 'strategies': ['backend'], 'command_template': SGLANG_COMMAND_TEMPLATE}
        sglang_contract = {**BASELINE_CONTRACT, 'experiment_group': 'G1', 'role': 'candidate', 'backend': 'sglang', 'strategies': ['backend'], 'result_json': SGLANG_RESULT_PATH}
        sglang_report['experiment_contract'] = sglang_contract
        sglang_report['baseline_contract'] = sglang_contract  # 兼容已有 68–72 读取路径。
        sglang_report.setdefault('quality', {'status': QUALITY_STATUS})
        sglang_report.setdefault('failure', {'status': 'not_observed' if FAILURE_REASON is None else 'recorded', 'reason': FAILURE_REASON, 'retest_path': RETEST_PATH})
        sglang_report.setdefault('decision', DECISION)
        Path(SGLANG_RESULT_PATH).write_text(json.dumps(sglang_report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('SGLang 结果：', json.dumps(sglang_report.get('metrics', {}), ensure_ascii=False, indent=2))
    finally:
        stop_backend(sglang_server, sglang_log_path)
else:
    print('跳过 SGLang：默认只验证 vLLM；需要独立安装和确认 SGLang 版本后再开启。')

```

#### 6.5 实测记录与结果表

读取 SGLang JSON 后，先核对模型、GPU、dtype、workload、并发和生成长度，再整理与 vLLM 对照所需的统一字段。该表只记录同一实验口径下的 backend 结果；没有完成条件对齐时，只记录 unsupported 或待复测。

| 记录项 | vLLM | SGLang | 说明 |
|:---|:---:|:---:|:---|
| backend / 版本 | 待读取 | 待读取 | 必须保留版本信息 |
| GPU / 显存 | 待读取 | 待读取 | 尽量使用同一硬件 |
| 模型 / dtype | 待读取 | 待读取 | 模型和 dtype 保持一致 |
| workload / 并发 | 待读取 | 待读取 | 请求集、生成长度和并发一致 |
| TTFT / TPOT / E2E | 待读取 | 待读取 | 统一统计口径 |
| 吞吐 / 成功率 / OOM | 待读取 | 待读取 | 同时记录失败状态 |
| evidence level | 待读取 | 待读取 | 区分 smoke、repeated 和 unsupported |
| decision | 待判断 | 待判断 | 等 6.6 统一解释 |
#### 6.6 解释结果与形成决策

先确认两种 backend 的环境、模型、workload、并发和 dtype 完全一致，再比较 TTFT、TPOT、E2E、吞吐、成功率和 OOM。只有条件一致且证据充分时，才输出 `accept / tune / reject`；否则保留 unsupported 或待复测状态。
### Step 7：跨 backend 结果对比

只有 vLLM 和 SGLang 使用相同模型、输入分布、生成长度、并发、dtype 和硬件时，才可以进行跨 backend 比较。报告要保留 backend 名称和版本；vLLM 的 PagedAttention 与 SGLang 的 RadixAttention 不视为同一种策略，端到端差异不能直接归因给某个单一机制。

当前学习顺序：先完成 vLLM G0/G1，再完成可选的 SGLang 对照，最后比较两份统一 JSON。没有可运行的 SGLang 环境时，保留 Step 6 关闭状态即可。
## 相关阅读

完成本节后，可以沿着“机制 → 引擎 → 评测规范”继续阅读：先看推理引擎如何实现请求服务，再回到论文和官方文档核对 benchmark 中的指标含义。
- [68. Speculative Decoding Benchmark | 投机解码基准](./68_Speculative_Decoding_Benchmark.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
- [vLLM Documentation | vLLM 官方文档](https://docs.vllm.ai/en/latest/)
- [SGLang Documentation | SGLang 官方文档](https://docs.sglang.ai/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention | PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)