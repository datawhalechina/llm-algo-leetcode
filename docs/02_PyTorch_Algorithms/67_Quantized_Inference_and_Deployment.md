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
**主责与复用边界：** 本项目主责是量化 artifact 的加载、执行和部署判断；显存优化路线只复用权重 / KV Cache 的容量证据，推理性能路线复用 TTFT、TPOT 和吞吐口径，训练侧 QLoRA 不在本项目重复验证。

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

先固定量化对象、格式、校准数据和 backend；CPU 只验证机制，GPU 才验证部署收益。

- 固定模型、输入集、batch size、seq len、解码策略、硬件环境和推理后端。
- 0.5B 用于 smoke，1.5B 用于真实 GPU 量化；两档结果分开记录。
- 明确权重/激活对象、量化粒度和部署约束，并分开记录 calibration 与 evaluation。
- 这一步的目标是先定义“可部署”的标准，而不是只追求更低 bit 数。

先按 66 的方式划分实验组，再决定是否进入真实量化：

| 组别 | 环境 | 实验内容 | 主要回答的问题 |
|---|---|---|---|
| C0 量化机制 | CPU | 改变 bits、group size 和权重分布 | 存储量、元数据和误差如何变化？ |
| C1 决策模拟 | CPU | 输入 baseline / candidate 指标和误差预算 | 什么条件下值得继续部署？ |
| G0 浮点 baseline | GPU/backend | 固定模型和 workload，运行 FP16/BF16 | 当前硬件和 backend 的基线是什么？ |
| G1 真实量化 | GPU/backend | 只改变一种量化格式或 artifact | 量化是否真的降低显存或延迟？ |
| G2 格式/backend 对照 | GPU/backend | 比较多个已确认支持的格式或 backend | 收益是否来自量化本身且值得迁移？ |

C0/C1 是本节必做的 CPU-first 主线；G0/G1 需要真实 GPU 和量化 artifact；G2 是可选扩展。没有真实 artifact 时，G1 只能停留在部署配置检查，不能填写量化收益数字。

量化机制与实验变量的对应关系如下：权重量化主要影响模型存储和显存，激活量化还会影响运行时 kernel，GPTQ/AWQ 需要校准数据，量化 backend 决定实际执行路径。不要把 dtype、量化 bit 数和 backend 当成同一个变量同时修改。

本项目把三类候选分开处理：GPTQ 是校准后权重 artifact，重点观察误差补偿、group size 和 backend kernel；AWQ 是激活感知权重 artifact，重点观察校准集、敏感通道保护和执行路径；GGUF 是文件格式与部署生态，重点观察格式是否被目标引擎加载，以及 CPU/GPU 混合执行和量化层支持。三者不能只按文件大小排序，也不能用一种 backend 的结果代表另外两种格式。

| 候选 | 本项目要验证的机制 | 最小真实证据 |
|---|---|---|
| GPTQ | 校准后的权重量化与误差补偿 | GPTQ artifact 被目标 backend 加载，并确认量化执行路径 |
| AWQ | 激活感知与敏感权重保护 | AWQ artifact 被目标 backend 加载，并记录校准与 kernel 信息 |
| GGUF | 文件封装、格式兼容与部署加载 | GGUF artifact 被对应 GGUF backend 加载；不能沿用 GPTQ/AWQ 的启动参数 |

### Step 2: 先确认 baseline 与量化口径合法

先跑通 FP16/BF16 baseline，再在相同模型、tokenizer、workload、batch、并发、cache policy 和硬件下只改变量化 artifact。

- 记录量化格式、粒度、校准数据与版本；只修改推理 dtype 不算量化实验。
- baseline 与 candidate 使用同一组延迟、吞吐、显存和输出质量指标。
- GPTQ/AWQ 等校准方法还要记录 split、样本数和最大长度。

### Step 3: 用统一口径比较收益与代价

同时比较性能、显存、数值误差和任务质量，避免只凭低 bit 或单项速度下结论。

- latency / throughput 解释速度，peak VRAM 解释容量，error / task metric 解释质量。
- 量化收益必须结合 batch、并发和 backend；压缩比不能代替真实显存峰值。
- 任何指标改善都不能替代量化格式和 kernel 支持证据。

### Step 4: 输出部署选型结论

按性能、容量、质量和兼容性约束输出 accept / tune / reject。

- 报告保留量化格式、粒度、校准信息、backend、硬件和完整 workload。
- 未加载真实量化权重、未确认 kernel 或格式不支持时，只能报告链路检查，不能判定可部署。
- tune 记录下一步是扩大回归、调整校准、改变粒度还是切换 backend。

### Step 5：CPU 实验——量化账本与误差机制

上面的 Step 1-4 定义项目口径；下面用 CPU 代码实现量化账本、模拟耗时、指标比较和部署决策。真实量化、校准、任务评估与压测放在 Step 6。


```python
import math
import time
from typing import Dict, List

```


```python
# TODO 0：实现 per-group 量化、反量化，并统计存储与误差
def simulate_weight_quantization(weights: List[float], bits: int = 8, group_size: int = 2) -> Dict[str, float]:
    """用对称 per-group 量化模拟存储压缩和反量化误差。

    bits 决定有符号量化范围，group_size 决定每个 scale 覆盖的权重数；
    返回的是权重重构误差和存储估算，不是模型任务质量或真实 artifact 大小。
    """
    # 提示：每组 scale 使用该组最大绝对值；全零组要避免除零。
    #       量化误差按 reconstructed - original 计算，元数据也要计入 quantized_bytes。
    if bits < 2 or bits > 8 or group_size < 1 or not weights:
        raise ValueError('bits 应在 2-8 之间，group_size >= 1，weights 不能为空')
    qmax = (1 << (bits - 1)) - 1
    reconstructed, group_count = [], 0
    for start in range(0, len(weights), group_size):
        group = [float(value) for value in weights[start:start + group_size]]
        scale = max(abs(value) for value in group) / qmax
        scale = scale if scale > 0 else 1.0
        reconstructed.extend(max(-qmax, min(qmax, round(value / scale))) * scale for value in group)
        group_count += 1
    errors = [estimate - actual for estimate, actual in zip(reconstructed, weights)]
    max_abs_error = max(abs(error) for error in errors)
    mse = sum(error * error for error in errors) / len(errors)
    original_bytes = len(weights) * 4
    scale_dtype_bits = 16  # 本模拟按 FP16 scale 估算元数据；真实格式需按 artifact 记录。
    quantized_bytes = math.ceil((len(weights) * bits + group_count * scale_dtype_bits) / 8)
    return {
        'parameter_count': len(weights), 'bits': bits, 'group_size': group_size,
        'groups': group_count, 'original_bytes': original_bytes,
        'quantized_bytes': quantized_bytes, 'scale_dtype_bits': scale_dtype_bits,
        'compression_ratio': round(original_bytes / quantized_bytes, 4),
        'max_abs_error': round(max_abs_error, 8), 'mse': round(mse, 8),
    }


def benchmark_fn(fn, warmup=2, iters=5):
    """测量一个候选函数的 CPU 平均耗时，返回毫秒。

    该函数不执行 CUDA synchronize，也不代表量化 kernel 或 backend 延迟。
    """
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示：用 time.perf_counter() 记录起止时间；warmup 不计入平均值。
    #       iters 必须为正，返回单位统一为 ms，方便和 latency 对齐。
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
    """统一比较 baseline 与量化候选的收益、资源和重构误差。

    两个输入必须来自同一 workload；error_delta 表示候选误差相对 baseline 的变化，
    不能把 max_abs_error / mse 直接解释成任务质量下降。
    """
    # ==========================================
    # TODO 2: 汇总 baseline / quantized 的核心指标差异
    # 提示：latency / vram / error 越低越好，throughput 越高越好。
    #       latency_delta、vram_delta 使用 baseline - quantized；throughput_delta 使用 quantized - baseline。
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
    """把量化候选的收益、误差和部署建议整理成可读报告。

    报告必须同时展示四类指标和 decision；不能只保留压缩率或延迟。
    """
    # ==========================================
    # TODO 3: 生成量化部署报告
    # 提示：把 latency、throughput、VRAM、error 的变化和 recommendation 放在一起。
    #       量化报告中的 error 仍是模拟/重构指标，真实任务质量要另行评测。
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
    """根据收益、误差和预算约束输出量化部署建议。

    返回 decision、reason、next_action；阈值属于当前 workload，不是所有 backend 的 SLA。
    """
    # ==========================================
    # TODO 4: 输出部署决策
    # 规则：
    # - 延迟或吞吐有明显收益，VRAM 也改善，且误差在预算内：accept
    # - 误差在预算内，但收益还不够稳：tune
    # - 误差超预算，或收益不足：reject
    # 提示：先计算四个布尔变量，再按质量门槛优先的顺序返回决策。
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
        weights = [0.0, 1.0, -2.0, 3.0, 0.5]
        quant_report = simulate_weight_quantization(weights, bits=4, group_size=2)
        assert quant_report['parameter_count'] == 5
        assert quant_report['groups'] == 3
        assert quant_report['quantized_bytes'] < quant_report['original_bytes']
        assert quant_report['max_abs_error'] >= 0.0 and quant_report['mse'] >= 0.0
        try:
            simulate_weight_quantization(weights, bits=16)
        except ValueError:
            pass
        else:
            raise AssertionError('不支持的 bit 数应明确拒绝！')
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
# TODO 0: 参考实现：per-group 量化、反量化，以及存储/误差统计
def simulate_weight_quantization(weights: List[float], bits: int = 8, group_size: int = 2) -> Dict[str, float]:
    """用对称 per-group 量化模拟存储压缩和反量化误差。"""
    if bits < 2 or bits > 8 or group_size < 1 or not weights:
        raise ValueError('bits 应在 2-8 之间，group_size >= 1，weights 不能为空')
    qmax = (1 << (bits - 1)) - 1
    reconstructed, group_count = [], 0
    for start in range(0, len(weights), group_size):
        group = [float(value) for value in weights[start:start + group_size]]
        scale = max(abs(value) for value in group) / qmax
        scale = scale if scale > 0 else 1.0
        reconstructed.extend(max(-qmax, min(qmax, round(value / scale))) * scale for value in group)
        group_count += 1
    errors = [estimate - actual for estimate, actual in zip(reconstructed, weights)]
    max_abs_error = max(abs(error) for error in errors)
    mse = sum(error * error for error in errors) / len(errors)
    original_bytes = len(weights) * 4
    scale_dtype_bits = 16  # 本模拟按 FP16 scale 估算元数据；真实格式需按 artifact 记录。
    quantized_bytes = math.ceil((len(weights) * bits + group_count * scale_dtype_bits) / 8)
    return {
        'parameter_count': len(weights), 'bits': bits, 'group_size': group_size,
        'groups': group_count, 'original_bytes': original_bytes,
        'quantized_bytes': quantized_bytes, 'scale_dtype_bits': scale_dtype_bits,
        'compression_ratio': round(original_bytes / quantized_bytes, 4),
        'max_abs_error': round(max_abs_error, 8), 'mse': round(mse, 8),
    }


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

这页现在按 `simulate -> measure -> compare -> report -> decide` 的最小量化部署项目闭环组织，不再只是单独比较 bit 数或显存收益。

#### TODO 0

- 实现方式：按 group 切分权重，用对称 scale 做量化和反量化，统计量化存储、元数据开销以及最大绝对误差 / MSE。
- 关键点：压缩比不是简单的 `32 / bits`，还要计入每组 scale 等元数据；误差指标也不能直接等同于任务精度。
- 项目意义：CPU 可以验证 bit、分组粒度、存储账本与误差之间的关系；真实低比特 kernel 的速度、workspace、显存峰值和部署兼容性仍需 GPU/backend 验证。

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

### Step 6（可选）：GPU/backend 实验——真实量化部署

**实验条件表**

| 项目 | G0 baseline | G1 量化候选 | G2 格式/backend 对照 |
|---|---|---|---|
| 模型与 workload | 固定 | 与 G0 相同 | 与 G0 相同 |
| dtype / 量化格式 | FP16 或 BF16 | 只改变一种真实量化格式 | 明确列出各格式 |
| calibration / evaluation | 不适用 / 独立评估 | 记录数据集、split、样本数 | 分别记录 |
| backend / 硬件 | 固定 | 固定 | 尽量固定 |

**结果表模板**

| 实验组 | format | prompt/output | batch/concurrency | TTFT/latency | throughput | peak VRAM | error/task quality | load/kernel | decision |
|---|---|---|---|---:|---:|---:|---|---|---|
| G0 | FP16/BF16 | 固定 | 固定 | 待采集 | 待采集 | 待采集 | 参考输出 | 成功/待确认 | 待判断 |
| G1 | 真实量化格式 | 与 G0 相同 | 与 G0 相同 | 待采集 | 待采集 | 待采集 | 待采集 | 成功/待确认 | 待判断 |

**证据边界**：CPU 模拟只能支持存储量和误差机制结论。只有真实量化权重被 backend 加载、目标 kernel 或格式得到确认，并在相同 workload 下测量延迟、吞吐、显存和质量，才能支持量化部署收益结论。

本节推荐的真实 GPU 数据口径是：Qwen2.5-1.5B-Instruct，WikiText-2 train 子集用于 calibration，validation 子集用于评估；0.5B 仅作为快速 smoke 档。
数据准备单元只下载独立 split 并保存口径清单；当前的长度限制是字符级近似，真正执行 GPTQ/AWQ 前必须使用目标 tokenizer 重新截断和打包。
按 66 的统一分组执行：G0 是浮点 baseline，G1 是单一真实量化格式，G2 才比较不同量化格式或 backend。推荐把 GPTQ、AWQ、GGUF 分别作为独立 G1 运行；只有它们都通过加载和 workload 校验后，才进入 G2 对照。只有真正加载量化权重或量化启动参数，才可记录量化收益；服务启动成功本身不等于量化收益成立。

默认保持 Practice-P1 的本地/模拟量化实验；需要接入 backend 时，将 `RUN_REAL_BACKEND` 改为 `True`。模型来源支持 `auto`、`modelscope`、`huggingface` 或本地目录，dtype 与端口由共享 helper 自动选择。当前 helper 尚未把 `QUANTIZATION_FORMAT` 转换为 GPTQ/AWQ/GGUF 专用启动参数，因此检测到真实量化格式时会主动停止；不同格式必须先接入对应 backend，不能把 artifact 加载成功等同于量化收益成立。

Colab / ModelScope：先确保 Notebook 位于仓库根目录（或先 clone 仓库），再运行下面单元；没有 GPU 时保留 `False`，不会阻断前面的 CPU-first 练习。

```python
import json
from pathlib import Path

try:
    from tools.inference_project_runtime import locate_repo_root
    REPO_ROOT = locate_repo_root()
    from tools.inference_project_runtime import (
        shared_project_config, save_project_result, start_optional_vllm,
        stop_optional_vllm, start_external_openai_backend, run_backend_benchmark,
    )
    from tools.backend_runtime import (
        probe_vllm_quantization_support, build_vllm_quantization_args, find_free_port,
    )
except ModuleNotFoundError:
    # 题目测试或纯 CPU 环境可能没有仓库工具；真实 backend 入口保持关闭。
    RUN_REAL_BACKEND = False
    def shared_project_config(**kwargs): return kwargs
    def save_project_result(*args, **kwargs): raise RuntimeError('需要从仓库根目录运行真实 backend 入口')

RUN_REAL_BACKEND = False  # 是否启动真实 backend；默认只练习量化决策模板。
MODEL_PROFILES = {
    'smoke': 'Qwen/Qwen2.5-0.5B-Instruct',
    'gpu_quant': 'Qwen/Qwen2.5-1.5B-Instruct',
}
MODEL_PROFILE = 'smoke'  # smoke 适合快速链路；gpu_quant 用于真实量化收益实验。
MODEL_ID = MODEL_PROFILES[MODEL_PROFILE]  # 量化前后必须保持同一基座模型。
MODEL_SOURCE = 'auto'  # 模型来源：auto / modelscope / huggingface / local。
MODEL_CACHE_DIR = 'model_cache'  # 模型缓存目录；相对路径从仓库根目录解析。
DTYPE = 'auto'  # 非量化计算 dtype；auto 根据当前 GPU 选择。
QUANTIZATION_FORMAT = 'none'  # none / gptq / awq / gguf；none 只表示浮点 baseline。
QUANTIZATION_BACKEND = 'vllm'  # 真实量化的执行后端；GGUF 不应直接沿用 vLLM 路径。
GGUF_COMMAND_TEMPLATE = None  # 例如 ['llama-server', '-m', '{model_path}', '--port', '{port}']；需由目标引擎确认。
RUN_DATA_PREP = False  # 是否下载并准备独立的 calibration/evaluation 文本。
CALIBRATION_DATASET = 'wikitext'  # Hugging Face 数据集名称。
DATASET_CONFIG = 'wikitext-2-raw-v1'  # WikiText-2 的配置名称。
CALIBRATION_SPLIT = 'train'  # 校准只使用 train 子集。
CALIBRATION_SAMPLES = 128  # Colab/小显存先用小子集，正式实验再扩大。
CALIBRATION_MAX_LENGTH = 512  # 校准样本截断长度。
EVAL_DATASET = 'wikitext'  # 与 calibration 分开的评估数据集名称。
EVAL_SPLIT = 'validation'  # 评估使用独立 split。
EVAL_SAMPLES = 128  # 先做固定小样本 perplexity/质量评估。
QUANTIZATION_ARTIFACT = None  # 真实量化权重或目录；None 时不能宣称量化收益。
BATCH_SIZE = 1  # baseline 与量化候选必须保持一致。
CONCURRENCY = 1  # 只在单独的并发实验中修改。
NUM_PROMPTS = 5  # smoke 请求数；正式实验应扩大并重复。
MAX_TOKENS = 64  # 每个请求的生成上限。
WARMUP = 1  # 不计入正式统计的预热请求数。
MAX_MODEL_LEN = 2048  # backend 的上下文上限，影响 KV Cache 预算。
BACKEND = 'vllm'  # 推理运行时；更换 backend 会改变 kernel 支持范围。
CACHE_POLICY = 'default'  # KV Cache 策略；对照实验中应固定。
RESULT_PATH = 'benchmarks/results/67_quantized_deployment.json'  # 统一结果文件。
DATA_MANIFEST_PATH = 'benchmarks/results/67_quantization_data_manifest.json'  # 只保存数据口径，不保存数据正文。

project_config = shared_project_config(
    model=MODEL_ID, backend=BACKEND, dtype=DTYPE,
    quantization_format=QUANTIZATION_FORMAT, quantization_backend=QUANTIZATION_BACKEND,
    quantization_artifact=QUANTIZATION_ARTIFACT,
    calibration_split=CALIBRATION_SPLIT, calibration_samples=CALIBRATION_SAMPLES,
    calibration_max_length=CALIBRATION_MAX_LENGTH, eval_dataset=EVAL_DATASET,
    eval_split=EVAL_SPLIT, eval_samples=EVAL_SAMPLES,
    generated_tokens=MAX_TOKENS, batch=BATCH_SIZE, concurrency=CONCURRENCY,
    cache_policy=CACHE_POLICY,
)
print(project_config)

def validate_quantization_setup():
    """阻止把普通浮点服务误报成真实量化实验。"""
    valid_formats = {'none', 'gptq', 'awq', 'gguf'}
    if QUANTIZATION_FORMAT not in valid_formats:
        raise ValueError(f'QUANTIZATION_FORMAT 必须是 {sorted(valid_formats)} 之一。')
    if not str(QUANTIZATION_BACKEND).strip():
        raise ValueError('QUANTIZATION_BACKEND 不能为空；请明确记录实际执行后端。')
    if QUANTIZATION_FORMAT == 'none' and QUANTIZATION_ARTIFACT is not None:
        raise ValueError('QUANTIZATION_FORMAT=none 时不能填写 QUANTIZATION_ARTIFACT。')
    if QUANTIZATION_FORMAT != 'none' and not QUANTIZATION_ARTIFACT:
        raise ValueError('真实量化实验必须提供 QUANTIZATION_ARTIFACT。')
    if QUANTIZATION_ARTIFACT and not Path(QUANTIZATION_ARTIFACT).exists():
        raise FileNotFoundError(f'量化 artifact 不存在：{QUANTIZATION_ARTIFACT}')
    if QUANTIZATION_FORMAT == 'gguf' and QUANTIZATION_BACKEND == 'vllm':
        raise ValueError('GGUF 必须使用已确认支持 GGUF 的独立 backend，不能直接沿用 vLLM 启动路径。')
    if QUANTIZATION_FORMAT != 'none' and QUANTIZATION_BACKEND != BACKEND:
        raise ValueError('量化 backend 与服务 backend 不一致；请先拆成独立实验，避免把格式切换和服务栈切换混为一个变量。')
    if QUANTIZATION_FORMAT in {'gptq', 'awq'} and QUANTIZATION_BACKEND not in {'vllm', 'sglang', 'transformers'}:
        raise ValueError('GPTQ/AWQ 的 backend 需先登记为 vllm、sglang 或 transformers，并确认实际支持。')
    if MODEL_PROFILE == 'gpu_quant' and QUANTIZATION_FORMAT == 'none':
        print('提示：gpu_quant 当前仍是浮点 baseline；请设置真实量化格式后再运行 G1。')
    return {
        'artifact_required': QUANTIZATION_FORMAT != 'none',
        'artifact_configured': bool(QUANTIZATION_ARTIFACT),
        'format': QUANTIZATION_FORMAT,
        'backend': QUANTIZATION_BACKEND,
        'same_as_serving_backend': QUANTIZATION_BACKEND == BACKEND,
    }

def inspect_quantization_artifact() -> dict:
    """检查 artifact 的基本格式，不把检查结果当作 kernel 兼容性证明。"""
    if QUANTIZATION_FORMAT == 'none':
        return {'status': 'baseline', 'format': 'none'}
    artifact = Path(QUANTIZATION_ARTIFACT)
    if QUANTIZATION_FORMAT == 'gguf':
        files = [artifact] if artifact.is_file() else sorted(artifact.glob('*.gguf'))
        if not files:
            raise ValueError('GGUF artifact 必须是 .gguf 文件，或包含 .gguf 文件的目录。')
        return {'status': 'metadata_present', 'format': 'gguf', 'files': [str(item) for item in files]}
    config_path = artifact / 'config.json' if artifact.is_dir() else artifact.parent / 'config.json'
    if not config_path.exists():
        raise ValueError(f'{QUANTIZATION_FORMAT.upper()} artifact 缺少 config.json，无法确认格式元数据。')
    metadata = json.loads(config_path.read_text(encoding='utf-8'))
    quant_config = metadata.get('quantization_config') or metadata.get('quantization')
    if not isinstance(quant_config, dict):
        raise ValueError(f'{QUANTIZATION_FORMAT.upper()} artifact 未发现 quantization_config 元数据。')
    declared = json.dumps(quant_config, ensure_ascii=False).lower()
    if QUANTIZATION_FORMAT not in declared:
        raise ValueError(f'artifact 元数据未声明 {QUANTIZATION_FORMAT.upper()}，请不要把普通浮点目录当作量化模型。')
    return {'status': 'metadata_present', 'format': QUANTIZATION_FORMAT, 'config_path': str(config_path), 'quantization_config': quant_config}

QUANTIZATION_SETUP = validate_quantization_setup()
ARTIFACT_INSPECTION = inspect_quantization_artifact()
project_config['artifact_inspection'] = ARTIFACT_INSPECTION
if RUN_REAL_BACKEND and QUANTIZATION_FORMAT == 'gguf' and not GGUF_COMMAND_TEMPLATE:
    raise ValueError('GGUF 实验必须提供包含 {model_path} 和 {port} 的 GGUF_COMMAND_TEMPLATE。')

QUANTIZATION_LAUNCH_ARGS = None
if RUN_REAL_BACKEND and QUANTIZATION_FORMAT in {'gptq', 'awq'}:
    quant_capability = probe_vllm_quantization_support()
    QUANTIZATION_LAUNCH_ARGS = build_vllm_quantization_args(quant_capability, QUANTIZATION_FORMAT)

# 量化候选完成本地测量后，用下面的调用保存统一结果：
# save_project_result(RESULT_PATH, project='67', strategy='w8a16',
#     config=project_config, metrics=metrics, quality=quality, decision=decision)

def load_calibration_and_eval_texts():
    """加载独立 split 的小型文本子集；不执行量化，也不替代任务评估。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('数据准备需要 datasets，请先安装 requirements 中的依赖。') from exc
    calibration = load_dataset(
        CALIBRATION_DATASET, DATASET_CONFIG,
        split=f'{CALIBRATION_SPLIT}[:{CALIBRATION_SAMPLES}]',
    )
    evaluation = load_dataset(
        EVAL_DATASET, DATASET_CONFIG,
        split=f'{EVAL_SPLIT}[:{EVAL_SAMPLES}]',
    )
    max_chars = CALIBRATION_MAX_LENGTH * 4  # 这里只做近似字符截断；正式量化前仍需 tokenizer 截断。
    calibration_texts = [row['text'].strip()[:max_chars] for row in calibration if row.get('text', '').strip()]
    evaluation_texts = [row['text'].strip()[:max_chars] for row in evaluation if row.get('text', '').strip()]
    if not calibration_texts or not evaluation_texts:
        raise ValueError('calibration/evaluation split 没有可用文本。')
    return {'calibration': calibration_texts, 'evaluation': evaluation_texts}

if RUN_DATA_PREP:
    DATASET_BUNDLE = load_calibration_and_eval_texts()
    manifest = {
        'dataset': CALIBRATION_DATASET, 'config': DATASET_CONFIG,
        'calibration_split': CALIBRATION_SPLIT,
        'calibration_samples': len(DATASET_BUNDLE['calibration']),
        'calibration_max_length': CALIBRATION_MAX_LENGTH,
        'evaluation_split': EVAL_SPLIT,
        'evaluation_samples': len(DATASET_BUNDLE['evaluation']),
    }
    manifest_path = Path(DATA_MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(manifest)
    print(f'数据口径清单已保存：{manifest_path}')
else:
    print('跳过数据下载：RUN_DATA_PREP=False；当前仅打印量化实验配置。')

# 真实 backend 先验证服务链路；量化格式专用启动参数需要按实际引擎补充。
if RUN_REAL_BACKEND and QUANTIZATION_FORMAT == 'gguf':
    from tools.model_runtime import resolve_model
    serving_model = QUANTIZATION_ARTIFACT
    model_path = resolve_model(serving_model, MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
    port = find_free_port()
    server, log_path = start_external_openai_backend(
        GGUF_COMMAND_TEMPLATE, model_path=str(model_path), port=port,
        log_path='benchmarks/results/67_gguf_backend.log',
    )
    try:
        report = run_backend_benchmark(
            project='67', base_url=f'http://127.0.0.1:{port}', model=MODEL_ID,
            label='gguf-deployment-smoke', output=RESULT_PATH, backend=QUANTIZATION_BACKEND,
            dtype=DTYPE, cache_policy=CACHE_POLICY, batch=BATCH_SIZE,
            concurrency=CONCURRENCY, num_prompts=NUM_PROMPTS, max_tokens=MAX_TOKENS,
            warmup=WARMUP,
        )
        print(report['normalized_result'])
    finally:
        stop_optional_vllm(server, log_path)

if RUN_REAL_BACKEND and QUANTIZATION_FORMAT != 'gguf':
    serving_model = QUANTIZATION_ARTIFACT or MODEL_ID
    serving_source = 'local' if QUANTIZATION_ARTIFACT else MODEL_SOURCE
    server, log_path, port, selected_dtype, model_path = start_optional_vllm(
        model_id=serving_model, model_source=serving_source, dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        served_model_name=MODEL_ID,
        quantization_args=QUANTIZATION_LAUNCH_ARGS,
    )
    try:
        report = run_backend_benchmark(
            project='67', base_url=f'http://127.0.0.1:{port}', model=MODEL_ID,
            label='vllm-deployment-smoke', output=RESULT_PATH,
            dtype=selected_dtype, cache_policy=CACHE_POLICY,
            batch=BATCH_SIZE, concurrency=CONCURRENCY, num_prompts=NUM_PROMPTS,
            max_tokens=MAX_TOKENS, warmup=WARMUP,
        )
        print({'model_path': model_path, 'dtype': selected_dtype, 'port': port})
        print(report['normalized_result'])
    finally:
        stop_optional_vllm(server, log_path)
```
