# 67. Quantized Inference and Deployment | 量化推理与部署

**难度：** Hard | **环境：** CPU-first | **标签：** `量化压缩`, `量化推理`, `部署` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节要求你验证一个量化推理方案是否满足部署条件。先固定 workload、运行后端和误差阈值，再比较 baseline 与量化方案的延迟、吞吐、显存和输出误差，同时记录 kernel 支持与反量化开销。最终输出一份部署建议，并明确方案适用的模型和服务条件。
**层级定位：** 本项目主落在 L4，关注量化模型如何被加载、执行和服务；低比特 kernel 属于 L2，模型版本、灰度发布和集群资源治理属于 L5，不在本项目内混为同一个结论。

**关键词：** `quantization`, `inference`, `deployment`

---

## 前置阅读

**导语：** 先把 W8A16、QLoRA、量化理论和基础推理对比看过，再进入量化部署项目会更容易判断压缩收益是否值得保留。
- [25. Quantization W8A16 | W8A16 量化](./25_Quantization_W8A16.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](./26_QLoRA_and_4bit_Quantization.md)
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)

## 相关阅读

**导语：** 完成量化部署选型后，可以继续看 profiling 是否解释了收益来源，或者把部署结论推进到真实 serving 链路。
- [74. Profiling-Driven End-to-End Optimization | profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)

### Step 1: 定义量化部署项目目标
先回答一个问题：这次量化是为了解决显存、吞吐、延迟，还是部署成本？

- 固定模型、输入集、batch size、seq len、解码策略、硬件环境和推理后端。
- 明确量化对象：只量化权重、权重和激活都量化，还是只做底座 4-bit 存储。
- 明确量化粒度：per-tensor、per-channel、per-group 或 block-wise，不同粒度会影响误差和元数据开销。
- 写清约束条件，例如最大可接受输出误差、最低 throughput、最大 latency、最大 VRAM 或是否允许依赖特定推理库。
- 这一步的目标是先定义“可部署”的标准，而不是只追求更低 bit 数。

### Step 2: 先确认 baseline 与量化口径合法

量化部署必须先确认 baseline 和量化方案可复现，不能直接把不同输入、不同后端或不同误差口径的结果拼在一起比较。

- Baseline 先记录 FP16 / FP32 的 latency、throughput、VRAM 和输出参考结果，保证后续 candidate 有稳定参照。
- Candidate 再记录 W8A16、INT8、4-bit 或其他量化方案的同一组指标，确保输入集、batch、decode 策略和运行后端一致。
- 输出误差至少要有一个可复查指标，例如 max error、cosine similarity、perplexity 变化或任务 accuracy 变化。
- 如果 baseline 自身波动很大，后面的量化收益和误差结论就没有解释空间。

### Step 3: 用统一口径比较收益与代价

量化项目必须用统一口径同时看 latency、throughput、VRAM 和误差，不能只挑显存或单项速度收益下结论。

- latency 下降说明单请求更快，但不一定代表吞吐一定提升。
- throughput 上升说明单位时间产出更高，但要结合 batch size 和服务并发解释。
- VRAM 下降可以释放 batch / context / 并发空间，但如果误差超出约束，不能直接部署。
- 精度误差需要和业务容忍度绑定：离线压缩实验可以更激进，在线服务通常要更保守。
- 这一阶段的产物应该是“收益 + 误差 + 部署条件”，而不是单项指标排行榜。

### Step 4: 输出部署选型结论

量化选型最终不是输出“哪个 bit 数更小”，而是输出这个量化方案在当前 workload 和误差约束下是否值得继续部署、微调或放弃。

- 输出 baseline / quantized 对比表，至少包含 latency、throughput、VRAM、error 和备注。
- 写清楚量化方案、量化粒度、校准数据、运行后端和硬件环境。
- 给出“什么时候用、什么时候别用”的结论。
- 如果本轮方案不可部署，要记录主要阻塞原因和下一轮尝试方向。
- 最终产物应回答：这次量化是否满足部署约束，收益来自哪里，代价是否可接受。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整量化推理与部署选型流程。下面的代码只实现其中最小、可复用的四块：测平均耗时、汇总 baseline / quantized 指标差异、生成部署报告，以及把结果收成 `accept / tune / reject` 的部署决策。真实项目中的模型量化、校准、输出误差评估和线上压测，需要基于这四步继续补充。


```python
import time

```


```python
def benchmark_fn(fn, warmup=2, iters=5):
    """测量量化部署候选的最小平均延迟。"""
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和 latency 对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    # start = ???
    for _ in range(iters):
        fn()
    # total = ???
    # avg_latency_ms = ???
    return avg_latency_ms


def summarize_quantized_result(base_metrics, quant_metrics):
    """统一比较 baseline 与量化候选的收益和误差约束。"""
    # ==========================================
    # TODO 2: 汇总 baseline / quantized 的核心指标差异
    # 提示: latency / vram / error 越低越好，throughput 越高越好
    # 正数表示 quantized 相比 baseline 有改善，error_delta 除外
    # ==========================================
    # latency_delta = ???
    # throughput_delta = ???
    # vram_delta = ???
    # error_delta = ???

    summary = {
        'latency_delta_ms': round(latency_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'vram_delta_mb': round(vram_delta, 2),
        'error_delta': round(error_delta, 4),
        'latency_improved': latency_delta > 0,
        'throughput_improved': throughput_delta > 0,
        'vram_improved': vram_delta > 0,
        'error_within_budget': error_delta <= quant_metrics['error_budget'],
    }
    return summary


def format_deployment_report(quant_name, summary, recommendation):
    """把量化候选的收益、误差和部署建议整理成报告。"""
    # ==========================================
    # TODO 3: 生成量化部署报告
    # 提示: 把指标变化、误差约束和部署建议放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        # f"| latency | {summary['latency_delta_ms']} ms | {'改善' if summary['latency_improved'] else '未改善'} |",
        # f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
        # f"| VRAM | {summary['vram_delta_mb']} MB | {'改善' if summary['vram_improved'] else '未改善'} |",
        # f"| error | {summary['error_delta']} | {'满足预算' if summary['error_within_budget'] else '超出预算'} |",
    ]
    # conclusion = ???
    return "\n".join([f"量化方案：{quant_name}", header, sep] + rows + [conclusion])


def recommend_quantized_deployment(summary, min_latency_delta_ms=5.0, min_throughput_delta=5.0, min_vram_delta_mb=256.0):
    """根据收益、误差和预算约束输出量化部署建议。"""
    # ==========================================
    # TODO 4: 输出部署决策
    # 规则：
    # - 延迟或吞吐有明显收益，VRAM 也改善，且误差在预算内：accept
    # - 误差在预算内，但收益还不够稳：tune
    # - 误差超预算，或收益不足：reject
    # ==========================================
    # strong_latency_gain = ???
    # strong_throughput_gain = ???
    # strong_vram_gain = ???
    # error_ok = ???
    # if ???:
    #     decision = ???
    #     reason = ???
    #     next_action = ???
    # elif ???:
    #     decision = ???
    #     reason = ???
    #     next_action = ???
    # else:
    #     decision = ???
    #     reason = ???
    #     next_action = ???
    # return {'decision': decision, 'reason': reason, 'next_action': next_action}

```

### 测试


```python
def test_quantized_project_template():
    try:
        counter = {'n': 0}

        def fn():
            counter['n'] += 1

        avg = benchmark_fn(fn, warmup=0, iters=2)
        assert counter['n'] == 2, "benchmark 应该运行 iters 次"
        assert avg >= 0.0, "平均耗时应该非负"

        baseline = {
            'latency_ms': 100.0,
            'throughput': 80.0,
            'vram_mb': 12000.0,
            'error': 0.0,
        }
        quantized = {
            'latency_ms': 72.0,
            'throughput': 120.0,
            'vram_mb': 7000.0,
            'error': 0.012,
            'error_budget': 0.02,
        }
        summary = summarize_quantized_result(baseline, quantized)

        assert summary['latency_delta_ms'] == 28.0
        assert summary['throughput_delta'] == 40.0
        assert summary['vram_delta_mb'] == 5000.0
        assert summary['error_delta'] == 0.012
        assert summary['latency_improved'] is True
        assert summary['throughput_improved'] is True
        assert summary['vram_improved'] is True
        assert summary['error_within_budget'] is True

        decision = recommend_quantized_deployment(summary, min_latency_delta_ms=5.0, min_throughput_delta=5.0, min_vram_delta_mb=256.0)
        assert decision['decision'] == 'accept'
        assert decision['next_action'] == 'promote_to_extended_regression'

        weak_summary = dict(summary)
        weak_summary['latency_delta_ms'] = 1.0
        weak_summary['throughput_delta'] = 2.0
        weak_summary['vram_delta_mb'] = 128.0
        weak_summary['latency_improved'] = True
        weak_summary['throughput_improved'] = True
        weak_summary['vram_improved'] = True
        weak_decision = recommend_quantized_deployment(weak_summary, min_latency_delta_ms=5.0, min_throughput_delta=5.0, min_vram_delta_mb=256.0)
        assert weak_decision['decision'] == 'tune'

        bad_summary = dict(summary)
        bad_summary['error_within_budget'] = False
        bad_decision = recommend_quantized_deployment(bad_summary, min_latency_delta_ms=5.0, min_throughput_delta=5.0, min_vram_delta_mb=256.0)
        assert bad_decision['decision'] == 'reject'

        report = format_deployment_report('W8A16', summary, decision['reason'])
        assert 'W8A16' in report
        assert '| 指标 | 变化 | 判断 |' in report
        assert '值得推进到更大样本部署回归' in report

        print("✅ 量化推理与部署项目模板代码通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_quantized_project_template()

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
def benchmark_fn(fn, warmup=2, iters=5):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和 latency 对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    avg_latency_ms = total / iters * 1000
    return avg_latency_ms


def summarize_quantized_result(base_metrics, quant_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / quantized 的核心指标差异
    # 提示: latency / vram / error 越低越好，throughput 越高越好
    # 正数表示 quantized 相比 baseline 有改善，error_delta 除外
    # ==========================================
    latency_delta = base_metrics['latency_ms'] - quant_metrics['latency_ms']
    throughput_delta = quant_metrics['throughput'] - base_metrics['throughput']
    vram_delta = base_metrics['vram_mb'] - quant_metrics['vram_mb']
    error_delta = quant_metrics['error'] - base_metrics['error']

    summary = {
        'latency_delta_ms': round(latency_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'vram_delta_mb': round(vram_delta, 2),
        'error_delta': round(error_delta, 4),
        'latency_improved': latency_delta > 0,
        'throughput_improved': throughput_delta > 0,
        'vram_improved': vram_delta > 0,
        'error_within_budget': error_delta <= quant_metrics['error_budget'],
    }
    return summary


def format_deployment_report(quant_name, summary, recommendation):
    # ==========================================
    # TODO 3: 生成量化部署报告
    # 提示: 把指标变化、误差约束和部署建议放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| latency | {summary['latency_delta_ms']} ms | {'改善' if summary['latency_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
        f"| VRAM | {summary['vram_delta_mb']} MB | {'改善' if summary['vram_improved'] else '未改善'} |",
        f"| error | {summary['error_delta']} | {'满足预算' if summary['error_within_budget'] else '超出预算'} |",
    ]
    conclusion = f"部署建议：{recommendation}。"
    return "\n".join([f"量化方案：{quant_name}", header, sep] + rows + [conclusion])


def recommend_quantized_deployment(summary, min_latency_delta_ms=5.0, min_throughput_delta=5.0, min_vram_delta_mb=256.0):
    strong_latency_gain = summary['latency_delta_ms'] >= min_latency_delta_ms
    strong_throughput_gain = summary['throughput_delta'] >= min_throughput_delta
    strong_vram_gain = summary['vram_delta_mb'] >= min_vram_delta_mb
    error_ok = summary['error_within_budget']

    if error_ok and (strong_latency_gain or strong_throughput_gain) and strong_vram_gain:
        decision = 'accept'
        reason = '收益和误差预算都达标，值得推进到更大样本部署回归。'
        next_action = 'promote_to_extended_regression'
    elif error_ok and (summary['latency_improved'] or summary['throughput_improved'] or summary['vram_improved']):
        decision = 'tune'
        reason = '误差仍在预算内，但收益还不够稳，先继续调量化粒度、校准集或后端。'
        next_action = 'refine_quant_granularity_or_backend'
    else:
        decision = 'reject'
        reason = '误差超预算，或收益不足以支撑部署切换。'
        next_action = 'fallback_to_baseline_or_rework_quant_scheme'
    return {'decision': decision, 'reason': reason, 'next_action': next_action}

```

### 解析

这页现在按 `measure -> compare -> report` 的最小量化部署项目闭环组织，不再只是单独比较 bit 数或显存收益。

#### TODO 1

- 实现方式：先做 warmup，再统计 `iters` 轮的总耗时，最后换算成单次平均 latency。
- 关键点：返回值统一用 ms，方便后面和部署报告里的延迟指标直接对齐。
- 项目意义：量化部署不是只看模型大小，先要把测量口径收平，后面才谈收益是否可信。

#### TODO 2

- 实现方式：统一计算 `latency_delta`、`throughput_delta`、`vram_delta` 和 `error_delta`。
- 关键点：前三类指标正数表示改善，`error_delta` 是部署约束，不应该误读成性能收益。
- 项目意义：这一步把量化方案从“压低 bit 数”转成“速度、显存和误差能否一起成立”的项目比较。

#### TODO 3

- 实现方式：把 latency、throughput、VRAM、error 放到同一张表，再补一句部署建议。
- 关键点：报告必须同时呈现收益和误差预算，避免只凭显存下降就直接上线。
- 项目意义：量化部署项目最后要回答的不是“能不能量化”，而是“这套方案是否满足部署约束，下一轮该扩大回归还是继续校准”。

### 可选 Practice-P2：真实 backend 与统一结果保存

默认保持 Practice-P1 的本地/模拟量化实验；需要接入 vLLM 时，将 `RUN_REAL_BACKEND` 改为 `True`。模型来源支持 `auto`、`modelscope`、`huggingface` 或本地目录，dtype 与端口由共享 helper 自动选择。不同量化格式是否被 backend 支持，仍需单独验证，不能把服务启动成功等同于量化收益成立。

Colab / ModelScope：先确保 Notebook 位于仓库根目录（或先 clone 仓库），再运行下面单元；没有 GPU 时保留 `False`，不会阻断前面的 CPU-first 练习。

```python
try:
    from tools.inference_project_runtime import locate_repo_root
    REPO_ROOT = locate_repo_root()
    from tools.inference_project_runtime import (
        shared_project_config, save_project_result, start_optional_vllm,
        stop_optional_vllm, run_backend_benchmark,
    )
except ModuleNotFoundError:
    # 题目测试或纯 CPU 环境可能没有仓库工具；真实 backend 入口保持关闭。
    RUN_REAL_BACKEND = False
    def shared_project_config(**kwargs): return kwargs
    def save_project_result(*args, **kwargs): raise RuntimeError('需要从仓库根目录运行真实 backend 入口')

RUN_REAL_BACKEND = False  # 是否启动真实 backend；默认只练习量化决策模板。
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 量化前后必须保持同一基座模型。
MODEL_SOURCE = 'auto'  # 模型来源：auto / modelscope / huggingface / local。
DTYPE = 'auto'  # 非量化计算 dtype；auto 根据当前 GPU 选择。
BACKEND = 'vllm'  # 推理运行时；更换 backend 会改变 kernel 支持范围。
CACHE_POLICY = 'default'  # KV Cache 策略；对照实验中应固定。
RESULT_PATH = 'benchmarks/results/67_quantized_deployment.json'  # 统一结果文件。

project_config = shared_project_config(
    model=MODEL_ID, backend=BACKEND, dtype=DTYPE,
    generated_tokens=64, cache_policy=CACHE_POLICY,
)
print(project_config)

# 量化候选完成本地测量后，用下面的调用保存统一结果：
# save_project_result(RESULT_PATH, project='67', strategy='w8a16',
#     config=project_config, metrics=metrics, quality=quality, decision=decision)

# 真实 backend 先验证服务链路；量化格式专用启动参数需要按实际引擎补充。
if RUN_REAL_BACKEND:
    server, log_path, port, selected_dtype, model_path = start_optional_vllm(
        model_id=MODEL_ID, model_source=MODEL_SOURCE, dtype=DTYPE,
        served_model_name=MODEL_ID,
    )
    try:
        report = run_backend_benchmark(
            project='67', base_url=f'http://127.0.0.1:{port}', model=MODEL_ID,
            label='vllm-deployment-smoke', output=RESULT_PATH,
            dtype=selected_dtype, cache_policy=CACHE_POLICY,
        )
        print({'model_path': model_path, 'dtype': selected_dtype, 'port': port})
        print(report['normalized_result'])
    finally:
        stop_optional_vllm(server, log_path)
```
