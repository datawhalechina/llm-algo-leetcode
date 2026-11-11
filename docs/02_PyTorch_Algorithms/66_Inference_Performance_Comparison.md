# 66. Inference Performance Comparison | 单卡推理测量基线

**难度：** Hard | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `推理优化`, `请求测量`, `基线核验` | **目标人群：** 掌握请求指标的进阶者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

同一组提示词能够返回文本，并不意味着已经得到可比较的推理基线：模型版本可能没有固定，流式返回可能被缓冲，成功请求数也可能与原始记录不一致。先把输入、测量边界和输出检查连起来，才能判断一份结果是否可引用。

本节从固定模型和请求集出发，沿逐请求记录核对实际执行量，再检查逐轮汇总是否可信。CPU 题目用简化报告练习这些判断；可选 GPU 实验在真实 vLLM 服务上采集对应证据，得到可供后续比较的测量基线。输出非空说明请求确实返回了内容，回答是否正确仍需独立任务评测。

**关键词：** 流式请求、首内容延迟、终止状态、统计窗口、基线完整性

<a id="runtime-environment"></a>

## 运行环境

- **CPU 题目：** 使用 [BASE-CPU](../guide.md#base-cpu)，完成 Step 4 的基线检查。
- **真实服务：** Step 5 使用 [INF-VLLM-011-CU128](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-quickstart)，按 5.1–5.6 顺序运行；模型、请求与状态见[实验资产表](../topic_discussion/inference_optimization/experiment_assets.md#baseline-66)。

## 前置阅读

先能区分请求事件、指标边界与实验条件，再开始核验真实测量；无需提前完成缓存或注意力优化实验。

- [INF-REQUEST-LIFECYCLE：请求生命周期与指标](./INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md)
- [Part00·20：性能证据与实验设计](../00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.md)

## Step 1：固定请求与测量基线

假设同一模型名称下，两次运行分别使用较短和较长的提示词。即使后一次吞吐更高，也还无法判断执行是否变快，因为输入处理量、输出长度与停止位置可能已经改变。基线因此固定模型与分词器的实际版本、请求内容与顺序、生成规则、精度及服务配置。本次采用单并发，前一条请求结束后再发送下一条；组批、排队和共享容量带来的高负载影响，需要另用固定到达方式的服务实验观察。

正式测量在服务就绪和预热后开始，每轮保存逐请求记录，再由这些记录核对实际完成量、时间边界与汇总统计。模型准备和服务启动时间另行记录，避免将首次准备开销混入稳态请求延迟。

| 检查对象 | 固定输入与操作 | 输出与用途 |
| --- | --- | --- |
| 模型身份 | 固定权重、分词器和精度，记录实际解析的版本 | 判断是否来自同一模型条件 |
| 请求负载 | 固定提示词、顺序、生成上限、并发和重复次数 | 确认实际完成了计划工作 |
| 测量边界 | 区分首次内容、最后内容和流结束 | 解释延迟对应的可见事件 |
| 基线核验 | 对照原始请求、汇总、停止与资源记录 | 判断可引用、需补采或运行无效 |

![固定输入、测量与核验构成单卡推理基线](../public/02_PyTorch_Algorithms/66_inference_benchmark_flow.svg)

## Step 2：区分内容返回与请求结束

流式客户端直接观察到的是内容块和流结束，一个内容块可能携带多个词元，不能把内容块数当作输出词元数。设一个构造请求在 0 ms 发出，首个内容块在 20 ms 到达，最后内容块在 70 ms 到达，流结束在 80 ms。首内容延迟为 20 ms，内容跨度为 50 ms，端到端时间为 80 ms，最后 10 ms 属于内容之后的结束与传输尾部。

若后端准确报告生成四个词元，将内容跨度除以三个后续词元，得到约 16.7 ms 的内容跨度平均间隔代理；若用流结束减首内容后再除以三个后续词元，则得到 20 ms，额外计入了尾部。前者按可见内容跨度估算，后者是包含结束尾部的平均时长，两者分别命名和保存；都不是 GPU 单词元计算时间。只有一个输出词元时，两者均不定义为后续词元间隔。

请求指标按各自时间计算，吞吐则使用整轮窗口。例如，一轮成功输出共 120 个词元，从开始发送到全部请求结束经过 3 秒，输出吞吐为 40 词元／秒。把每条请求的生成速率取平均，或删掉请求之间的等待时间，都会改变这个分母；失败请求也应保留在本轮计数中。

| 观察 | 客户端边界或统计方法 | 核验要求 |
| --- | --- | --- |
| 首内容延迟 | 发送请求至首次非空内容 | 内容块可能含多个词元，不直接称作词元级 TTFT；无内容时保持缺失 |
| 内容间隔代理 | 首末内容跨度除以输出词元数减一 | 使用准确词元数，单词元为不适用 |
| 含结束尾部的平均时长 | 首内容至流结束的跨度除以输出词元数减一 | 与内容间隔代理分开；单词元为不适用 |
| 请求端到端时间 | 发送请求至流读取结束 | 与最后内容时间分别保存 |
| 输出吞吐 | 本轮成功输出词元总数除以完整测量窗口 | 分母包含整个请求批次，按轮复核 |
| 停止与长度 | 自然停止或达到生成上限 | 上限停止单列记录，不自动当作回答完整 |

## Step 3：判断一份基线能否引用

核验先区分“已有记录证明执行有误”和“记录不足以判断”。例如，原始记录只有 29 条，汇总却声称完成 30 条，这是已知的不一致，需要修复执行或统计；30 条记录均存在，但词元数只靠文本长度估算，则需要补采精确计数。已知失败优先拒绝，只有缺失时先补证据；全部检查通过才记为可引用状态 `ready`。引用仍需匹配模型、负载和测量协议，记录完整不等于优化收益或回答质量已经通过。

重复测量还要核对计划是否真正执行。计划每轮 30 条、重复 3 轮，就应保留 3 组各 30 条的记录；只完成两轮，即使每轮内部相符，也没有完成计划。轮次完整性与单轮请求计数必须分别检查。

统计与资源指标要按各自的观测范围解释。逐轮 P95 描述每一轮的尾部，三轮 P95 的平均值不是合并 90 条记录后的 P95；设备总显存包含桌面和其他进程，也不能改称服务进程独占显存。保存这些记录时应同时保留统计方法与采样对象。

| 状态 | 触发条件 | 是否可引用与下一动作 |
| --- | --- | --- |
| `reject` | 运行、输出、身份匹配或统计一致性检查失败 | 不可引用；修复后用新标签测量 |
| `tune` | 没有已知硬失败，但原始记录、计时、版本或资源证据缺失 | 先补证据；不以缺失字段判为通过 |
| `ready` | 身份匹配、请求有效、计划轮次与统计可核对、资源口径明确 | 可作为同条件比较的参照 |

## Step 4：实现基线身份、请求与统计核验

题目提供身份字典、客户端请求记录和逐轮汇总。字段读取、样本筛选、缺失记录与结果结构均已给出；补全身份匹配、时间与停止约束、统计一致性及状态优先级。测试包含缺失字段、单词元、长度上限、失败请求和汇总失配。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
| --- | --- | --- | --- |
| TODO 1：核对可引用身份 | 期望与实际身份、已提取的缺失和错配项 | 身份完整且全部匹配才通过 | 版本错配、缺少分词器身份 |
| TODO 2：核对请求有效性 | 已读出的四个时间、词元数与停止原因 | 时间有序、长度合法、停止与上限相符 | 倒序、单词元、长度上限及空输出 |
| TODO 3：核对逐轮统计 | 实际样本计数、词元总数与完整窗口 | 请求与词元统计相符，吞吐使用同一窗口 | 分母错误、漏样本、错误词元总数 |
| TODO 4：形成基线状态 | 已归集的硬失败与缺失证据 | 硬失败优先拒绝，缺失需补采，完整才可引用 | 同时失败与缺失、完整及不完整证据 |

```python
import math
```


```python
def check_baseline_identity(expected, observed):
    """核对模型、分词器、负载与计时协议是否完整且相符。"""
    keys = ("model_id", "model_revision", "tokenizer_revision", "workload_sha256",
            "profile_id", "dtype", "timing_protocol")
    missing = [key for key in keys if not expected.get(key) or not observed.get(key)]
    mismatches = [key for key in keys if expected.get(key) and observed.get(key)
                  and expected[key] != observed[key]]
    # TODO 1：身份完整并且匹配后，才允许引用。
    # identity_ok = ???  # 身份是否同时满足完整性和一致性；布尔值。
    raise NotImplementedError("请完成 TODO 1")
    return {"ok": identity_ok, "missing": missing, "mismatches": mismatches}


def check_baseline_request(row, max_tokens):
    """区分缺失证据、时间倒序、空输出与停止条件错误。"""
    if row.get("ok") is not True:
        return {"ok": False, "missing": [], "failures": ["request_failed"]}
    fields = ("submit_offset_ms", "first_content_offset_ms", "last_content_offset_ms",
              "completion_offset_ms", "output_tokens", "finish_reason",
              "estimated_output_tokens", "stream_done_received")
    missing = [key for key in fields if row.get(key) is None]
    failures = [] if str(row.get("output_text", "")).strip() else ["empty_output"]
    if missing:
        return {"ok": False, "missing": missing, "failures": failures}
    times = [row[key] for key in fields[:4]]
    finite_times = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in times)
    if not finite_times:
        return {"ok": False, "missing": [], "failures": failures + ["nonfinite_time"]}
    submitted, first, last, completed = times
    tokens = row["output_tokens"]
    exact_count = row["estimated_output_tokens"] is False
    integer_count = isinstance(tokens, int) and not isinstance(tokens, bool)
    reason = row["finish_reason"]
    # TODO 2：核对时间顺序、长度范围和停止条件。
    # time_order_ok = ???  # 四个客户端时刻是否按发生顺序排列；布尔值。
    # token_count_ok = ???  # 输出词元数是否为允许范围内的整数；布尔值。
    # stop_ok = ???  # 自然停止或达到声明长度上限是否成立；布尔值。
    raise NotImplementedError("请完成 TODO 2")
    failures.extend(name for name, passed in (
        ("time_order", time_order_ok), ("token_count", token_count_ok),
        ("stop_reason", stop_ok), ("incomplete_stream", row["stream_done_received"] is True),
    ) if not passed)
    if not exact_count:
        missing.append("exact_output_tokens")
    return {"ok": not failures and not missing, "missing": missing, "failures": failures}


def check_baseline_totals(rows, metrics, expected_requests):
    """用本轮原始样本核对计数、词元总量和完整窗口吞吐。"""
    successful = [row for row in rows if row.get("ok") is True]
    failed_count = len(rows) - len(successful)
    output_counts = [row.get("output_tokens") for row in successful]
    count_values_valid = all(isinstance(value, int) and not isinstance(value, bool)
                             and value >= 0 for value in output_counts)
    total_tokens = sum(output_counts) if count_values_valid else None
    required = ("successful_requests", "failed_requests", "output_tokens",
                "output_token_throughput_per_s", "duration_s")
    missing = [key for key in required if metrics.get(key) is None]
    duration = metrics.get("duration_s")
    throughput = metrics.get("output_token_throughput_per_s")
    finite_window = isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0
    finite_rate = isinstance(throughput, (int, float)) and math.isfinite(throughput) and throughput >= 0
    expected_rate = total_tokens / duration if finite_window and total_tokens is not None else None
    completions = [row.get("completion_offset_ms") for row in rows]
    finite_completions = [value for value in completions
                          if isinstance(value, (int, float)) and math.isfinite(value)]
    window_covers_records = (finite_window and
                             duration * 1000 + 1.0 >= max(finite_completions, default=0.0))
    relative_tolerance, absolute_tolerance = 5e-4, 0.001
    # TODO 3：请求计数、词元总量与吞吐分别核对。
    # counts_ok = ???  # 实际请求数及成功、失败计数是否与本轮声明一致；布尔值。
    # tokens_ok = ???  # 实际成功输出词元总数是否与汇总一致；布尔值。
    # throughput_ok = ???  # 窗口覆盖请求且声明吞吐符合已给定舍入容差；布尔值。
    raise NotImplementedError("请完成 TODO 3")
    failures = []
    if len(rows) != expected_requests or (
        all(metrics.get(key) is not None for key in ("successful_requests", "failed_requests"))
        and not counts_ok
    ):
        failures.append("sample_counts")
    if metrics.get("output_tokens") is not None and not tokens_ok:
        failures.append("token_total")
    if duration is not None and throughput is not None and not throughput_ok:
        failures.append("throughput_window")
    return {"ok": not failures and not missing, "missing": missing, "failures": failures,
            "observed_requests": len(rows), "observed_output_tokens": total_tokens}


def decide_baseline_readiness(evidence):
    """按硬失败、缺失与完整证据确定基线可引用状态。"""
    failures = list(evidence.get("failures", []))
    missing = list(evidence.get("missing", []))
    # TODO 4：硬失败优先于缺失证据。
    # must_reject = ???  # 当前证据是否存在必须优先拒绝的硬失败；布尔值。
    # can_reference = ???  # 当前证据是否已排除硬失败并补齐缺失；布尔值。
    raise NotImplementedError("请完成 TODO 4")
    status = "reject" if must_reject else ("ready" if can_reference else "tune")
    return {"status": status, "failures": failures, "missing": missing,
            "scope": "fixed_workload_baseline_not_task_quality",
            "next_action": "按身份和协议匹配引用" if can_reference else "修复失败或补采缺失证据"}
```


```python
def _request(**changes):
    """构造使用同一客户端时钟的完整请求记录。"""
    row = {
        'ok': True, 'submit_offset_ms': 0.0, 'first_content_offset_ms': 20.0,
        'last_content_offset_ms': 70.0, 'completion_offset_ms': 80.0,
        'output_tokens': 4, 'output_text': 'a short reply',
        'estimated_output_tokens': False, 'finish_reason': 'stop',
        'stream_done_received': True,
    }
    return {**row, **changes}


def test_identity_requires_complete_match():
    """TODO 1：完整匹配通过，版本错配或缺失身份不能引用。"""
    expected = {
        'model_id': 'model', 'model_revision': 'r1', 'tokenizer_revision': 'r1',
        'workload_sha256': 'h1', 'profile_id': 'profile', 'dtype': 'bf16',
        'timing_protocol': 'client-stream',
    }
    assert check_baseline_identity(expected, dict(expected))['ok']
    mismatch = check_baseline_identity(expected, {**expected, 'model_revision': 'r2'})
    assert not mismatch['ok'] and 'model_revision' in mismatch['mismatches']
    missing = check_baseline_identity(expected, {**expected, 'tokenizer_revision': None})
    assert not missing['ok'] and 'tokenizer_revision' in missing['missing']


def test_request_order_length_and_stop():
    """TODO 2：区分有效单词元、长度上限、错误停止与时间倒序。"""
    assert check_baseline_request(_request(), 64)['ok']
    assert check_baseline_request(_request(output_tokens=1, last_content_offset_ms=20.0), 64)['ok']
    assert check_baseline_request(_request(output_tokens=64, finish_reason='length'), 64)['ok']
    assert not check_baseline_request(_request(output_tokens=4, finish_reason='length'), 64)['ok']
    assert not check_baseline_request(_request(last_content_offset_ms=10.0), 64)['ok']
    assert not check_baseline_request(_request(output_tokens=65), 64)['ok']
    assert not check_baseline_request(_request(output_text=''), 64)['ok']
    assert 'exact_output_tokens' in check_baseline_request(_request(estimated_output_tokens=True), 64)['missing']
    assert 'request_failed' in check_baseline_request({'ok': False}, 64)['failures']
    assert 'nonfinite_time' in check_baseline_request(_request(completion_offset_ms=float('nan')), 64)['failures']


def test_totals_use_all_samples_and_the_same_window():
    """TODO 3：成功输出为分子，完整轮次窗口为分母，失败仍计入请求数。"""
    rows = [_request(), {'ok': False}]
    metrics = {
        'successful_requests': 1, 'failed_requests': 1, 'output_tokens': 4,
        'output_token_throughput_per_s': 2.0, 'duration_s': 2.0,
    }
    assert check_baseline_totals(rows, metrics, 2)['ok']
    assert not check_baseline_totals(rows, {**metrics, 'failed_requests': 0}, 2)['ok']
    assert not check_baseline_totals(rows, {**metrics, 'output_tokens': 5}, 2)['ok']
    assert not check_baseline_totals(rows, {**metrics, 'duration_s': 1.0}, 2)['ok']
    assert not check_baseline_totals(rows[:1], metrics, 2)['ok']
    assert not check_baseline_totals(rows, {**metrics, 'duration_s': 0.0}, 2)['ok']


def test_readiness_failure_precedes_missing():
    """TODO 4：完整证据可引用，缺失需补采，失败即使伴随缺失也必须拒绝。"""
    assert decide_baseline_readiness({'failures': [], 'missing': []})['status'] == 'ready'
    assert decide_baseline_readiness({'failures': [], 'missing': ['peak']})['status'] == 'tune'
    assert decide_baseline_readiness({'failures': ['runtime'], 'missing': ['peak']})['status'] == 'reject'


test_identity_requires_complete_match()
test_request_order_length_and_stop()
test_totals_use_all_samples_and_the_same_window()
test_readiness_failure_precedes_missing()
print('PASSED: 身份、请求、逐轮统计与基线状态核验通过。')
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
def check_baseline_identity(expected, observed):
    """核对模型、分词器、负载与计时协议是否完整且相符。"""
    keys = ("model_id", "model_revision", "tokenizer_revision", "workload_sha256",
            "profile_id", "dtype", "timing_protocol")
    missing = [key for key in keys if not expected.get(key) or not observed.get(key)]
    mismatches = [key for key in keys if expected.get(key) and observed.get(key)
                  and expected[key] != observed[key]]
    # TODO 1：身份完整并且匹配后，才允许引用。
    identity_ok = not missing and not mismatches
    return {"ok": identity_ok, "missing": missing, "mismatches": mismatches}


def check_baseline_request(row, max_tokens):
    """区分缺失证据、时间倒序、空输出与停止条件错误。"""
    if row.get("ok") is not True:
        return {"ok": False, "missing": [], "failures": ["request_failed"]}
    fields = ("submit_offset_ms", "first_content_offset_ms", "last_content_offset_ms",
              "completion_offset_ms", "output_tokens", "finish_reason",
              "estimated_output_tokens", "stream_done_received")
    missing = [key for key in fields if row.get(key) is None]
    failures = [] if str(row.get("output_text", "")).strip() else ["empty_output"]
    if missing:
        return {"ok": False, "missing": missing, "failures": failures}
    times = [row[key] for key in fields[:4]]
    finite_times = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in times)
    if not finite_times:
        return {"ok": False, "missing": [], "failures": failures + ["nonfinite_time"]}
    submitted, first, last, completed = times
    tokens = row["output_tokens"]
    exact_count = row["estimated_output_tokens"] is False
    integer_count = isinstance(tokens, int) and not isinstance(tokens, bool)
    reason = row["finish_reason"]
    # TODO 2：核对时间顺序、长度范围和停止条件。
    time_order_ok = 0 <= submitted <= first <= last <= completed
    token_count_ok = integer_count and 1 <= tokens <= max_tokens
    stop_ok = reason == "stop" or (reason == "length" and tokens == max_tokens)
    failures.extend(name for name, passed in (
        ("time_order", time_order_ok), ("token_count", token_count_ok),
        ("stop_reason", stop_ok), ("incomplete_stream", row["stream_done_received"] is True),
    ) if not passed)
    if not exact_count:
        missing.append("exact_output_tokens")
    return {"ok": not failures and not missing, "missing": missing, "failures": failures}


def check_baseline_totals(rows, metrics, expected_requests):
    """用本轮原始样本核对计数、词元总量和完整窗口吞吐。"""
    successful = [row for row in rows if row.get("ok") is True]
    failed_count = len(rows) - len(successful)
    output_counts = [row.get("output_tokens") for row in successful]
    count_values_valid = all(isinstance(value, int) and not isinstance(value, bool)
                             and value >= 0 for value in output_counts)
    total_tokens = sum(output_counts) if count_values_valid else None
    required = ("successful_requests", "failed_requests", "output_tokens",
                "output_token_throughput_per_s", "duration_s")
    missing = [key for key in required if metrics.get(key) is None]
    duration = metrics.get("duration_s")
    throughput = metrics.get("output_token_throughput_per_s")
    finite_window = isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0
    finite_rate = isinstance(throughput, (int, float)) and math.isfinite(throughput) and throughput >= 0
    expected_rate = total_tokens / duration if finite_window and total_tokens is not None else None
    completions = [row.get("completion_offset_ms") for row in rows]
    finite_completions = [value for value in completions
                          if isinstance(value, (int, float)) and math.isfinite(value)]
    window_covers_records = (finite_window and
                             duration * 1000 + 1.0 >= max(finite_completions, default=0.0))
    relative_tolerance, absolute_tolerance = 5e-4, 0.001
    # TODO 3：请求计数、词元总量与吞吐分别核对。
    counts_ok = (len(rows) == expected_requests
                 and metrics.get("successful_requests") == len(successful)
                 and metrics.get("failed_requests") == failed_count)
    tokens_ok = total_tokens is not None and metrics.get("output_tokens") == total_tokens
    throughput_ok = (finite_rate and window_covers_records and expected_rate is not None
                     and math.isclose(throughput, expected_rate,
                                      rel_tol=relative_tolerance, abs_tol=absolute_tolerance))
    failures = []
    if len(rows) != expected_requests or (
        all(metrics.get(key) is not None for key in ("successful_requests", "failed_requests"))
        and not counts_ok
    ):
        failures.append("sample_counts")
    if metrics.get("output_tokens") is not None and not tokens_ok:
        failures.append("token_total")
    if duration is not None and throughput is not None and not throughput_ok:
        failures.append("throughput_window")
    return {"ok": not failures and not missing, "missing": missing, "failures": failures,
            "observed_requests": len(rows), "observed_output_tokens": total_tokens}


def decide_baseline_readiness(evidence):
    """按硬失败、缺失与完整证据确定基线可引用状态。"""
    failures = list(evidence.get("failures", []))
    missing = list(evidence.get("missing", []))
    # TODO 4：硬失败优先于缺失证据。
    must_reject = bool(failures)
    can_reference = not failures and not missing
    status = "reject" if must_reject else ("ready" if can_reference else "tune")
    return {"status": status, "failures": failures, "missing": missing,
            "scope": "fixed_workload_baseline_not_task_quality",
            "next_action": "按身份和协议匹配引用" if can_reference else "修复失败或补采缺失证据"}
```

### 解析

四个判断分别对应基线的来源、请求记录、统计完整性和最终可引用状态；先确认原始事实有效，再决定能否使用结果。

#### TODO 1：核对可引用身份

将缺失与错配同时作为阻断条件，避免只比较已有字段而放过缺失版本。题目已提供两类问题列表，待补的是合取关系；测试分别移除分词器身份、改变模型版本，确认两类情况都不能通过。

#### TODO 2：核对请求有效性

发送、首次内容、最后内容和流结束使用同一时钟，必须按顺序排列。长度上限停止只有在输出词元数达到上限时才成立；估算词元数保留为缺失证据。单词元与上限停止是合法边界，空输出、时间倒序和非法长度由对应反例阻止。

#### TODO 3：核对逐轮统计

同时核对总请求数、成功失败计数和输出词元总量，再以完整窗口核对吞吐；舍入容差只吸收记录精度差异。测试保留失败请求，并分别改变窗口、计数与词元总数，区分漏样本和分母错误。

#### TODO 4：形成基线状态

只有硬失败和缺失项都为空才可引用；出现硬失败时优先拒绝，不能被同时存在的缺失项降为单纯补采。测试覆盖三种状态及失败与缺失并存，`ready` 的范围仍限定为基线完整性。
## Step 5（可选）：采集并核验真实单卡基线

固定一个小模型、同一请求集与单并发，通过 vLLM 采集客户端流式记录。运行过程与统计核验分开，首次模型准备和服务启动不计入正式请求测量。
| CPU 项目判断 | 真实服务补充的事实 | 输出资产 |
| --- | --- | --- |
| 身份匹配、请求有效性、逐轮统计与可引用状态 | 真实模型版本、逐请求时间与停止、准确词元数、重复轮次和设备总显存 | 原始报告、带固定合同的测量报告、登记回执；仅 `ready` 生成限定性引用 |
### 5.1 选择运行阶段和标签

使用 [INF-VLLM-011-CU128](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-quickstart)，面向单张 12 GB NVIDIA GPU。固定模型为 Qwen2.5-0.5B-Instruct；每轮 30 条请求、单并发、最多输出 64 个词元，预热 5 次并重复 3 轮。

每次修改阶段或标签后，按 5.1–5.6 顺序执行。

| 阶段 | 执行动作 | 写入行为 |
| --- | --- | --- |
| `prepare` | 读取固定合同和输入身份 | 不启动服务、不下载模型、不写结果 |
| `measure` | 完成 Profile 预检，启动服务并采集固定工作负载 | 新标签原始报告、测量报告和日志 |
| `read` | 只读指定标签并核对证据 | 不写文件；历史协议只读解释 |
| `close` | 读取已保存证据，登记检查结论 | 幂等写登记回执；通过时写限定性引用 |

```python
RUN_LABEL = 'replace_with_unique_label'  # 新测量用唯一标签；读取和登记使用已有标签。
RUN_STAGE = 'prepare'  # prepare / measure / read / close；只有 measure 使用 GPU。
```

### 5.2 加载合同并检查环境与输入

新测量从注册表加载[版本化合同](../../benchmarks/contracts/inference_baseline_v3.json)，核对固定参数与工作负载哈希。`read` 和 `close` 使用测量时保存的合同快照，不随默认合同升级切换参数；没有绑定快照的历史报告只读解释。

只有 `measure` 执行完整 Profile 预检；预检不下载模型或启动服务，读取与登记不使用 GPU。

```python
import importlib
import json
import subprocess
import sys
from pathlib import Path
from tools.project_runtime import resolve_project_root
from tools import inference_baseline_runtime

if RUN_STAGE not in {'prepare', 'measure', 'read', 'close'}:
    raise ValueError('未知运行阶段')
baseline_runtime = importlib.reload(inference_baseline_runtime)
PROJECT_ROOT = resolve_project_root()
plan = baseline_runtime.load_baseline_plan(PROJECT_ROOT, RUN_LABEL, RUN_STAGE)
preflight = {'ready': False, 'status': 'not_requested', 'python': sys.executable}
if RUN_STAGE == 'measure':
    checked = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'tools/environment_preflight.py'),
         '--profile', plan['contract']['profile_id']],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    if checked.returncode != 0:
        raise RuntimeError(f"Profile 预检未通过：{checked.stdout}\n{checked.stderr}")
    preflight = json.loads(checked.stdout)
    if preflight.get('ready') is not True:
        raise RuntimeError(f"Profile 尚未就绪：{preflight}")
print(json.dumps({'stage': RUN_STAGE, 'preflight': preflight,
                  'contract': plan['contract']}, ensure_ascii=False, indent=2))
```

### 5.3 确认固定工作负载与可引用范围

核对本次模型、负载、精度、并发、缓存设置与测量协议，并查看合同来源和内容 SHA256。模型与分词器的实际版本随采集记录保存，用于与这里的预期条件比较。

```python
contract = plan['contract']
if RUN_STAGE == 'measure':
    occupied = [plan[key] for key in (
        'raw_path', 'result_path', 'log_path', 'registration_path', 'reference_path'
    ) if (PROJECT_ROOT / plan[key]).exists()]
    if occupied:
        raise FileExistsError(f"标签已有资产，请换新标签：{occupied}")
print(json.dumps({
    'contract_source': plan['contract_source'],
    'contract_sha256': (plan['experiment_contract'] or {}).get('sha256'),
    'model': contract.get('model_id'), 'workload_sha256': contract.get('workload_sha256'),
    'requests_per_repeat': contract.get('num_prompts'), 'repeats': contract.get('repeats'),
    'timing_protocol': contract.get('timing_protocol'), 'result_path': plan['result_path'],
    'reference_scope': 'matching_identity_and_measurement_protocol_only',
}, ensure_ascii=False, indent=2))
```

### 5.4 执行固定请求并保存原始记录

模型解析和服务准备完成后，每轮先预热，再测量固定请求。首末内容时刻、流结束、停止原因与词元数分别保存；失败保留错误信息并清理服务。

共享采集器的旧字段 `tpot_ms` 使用首内容至流结束的区间，含结束尾部；旧 `ttft_ms` 在没有内容时还可能回退为结束时间。它们按原始报告保留，不用于本页的新口径展示。5.5 从独立时间戳重新计算首内容延迟、内容间隔代理、E2E 和结束尾部；缺少所需事件时保留缺失，只有一个输出词元时不计算间隔代理。

![服务准备、正式采集与失败清理](../public/02_PyTorch_Algorithms/66_gpu_backend_experiment_flow.svg)


```python
if RUN_STAGE == 'measure':
    baseline_runtime.collect_baseline(plan, preflight)
    print(f"测量已保存：{plan['result_path']}")
else:
    print(f"{RUN_STAGE}：跳过服务启动与测量。")
```

### 5.5 读取并核对逐轮证据

读取已保存报告，逐轮查看请求数、时间、停止原因、准确词元数与统计核验结果，同时检查合同、原始报告哈希和显存采样范围。达到长度上限的请求单列展示，以便区分正常结束与受预算截断的输出。

本页展示下列重算字段，而不是共享原始汇总中的旧 TTFT／TPOT 数值。内容间隔代理要求准确词元数大于一；有效请求的样本先通过时间顺序、计数和停止规则核验。历史报告缺少时间戳时只读解释，不补写字段，也不混入新口径统计。

| 展示名称 | 专用报告字段 | 时间边界 |
| --- | --- | --- |
| 首内容延迟 | `ttft_content_ms` | 首次内容减发送时间 |
| 内容间隔代理 | `content_tpot_proxy_ms` | 首末内容跨度除以输出词元数减一；单词元不适用 |
| 请求端到端时间 | `request_e2e_ms` | 流结束减发送时间 |
| 结束尾部 | `stream_tail_ms` | 流结束减最后内容时间 |


```python
baseline_runtime = importlib.reload(inference_baseline_runtime)
if RUN_STAGE == 'prepare':
    baseline_evidence = {
        'failures': [], 'missing': ['measurement_not_requested'],
        'decision': {'status': 'tune'}, 'runs': [],
    }
else:
    baseline_evidence = baseline_runtime.read_baseline_evidence(plan)
print(json.dumps({
    'decision': baseline_evidence['decision'],
    'historical_protocol': baseline_evidence.get('historical_protocol', False),
    'result_path': baseline_evidence.get('result_path'),
    'identity': baseline_evidence.get('identity'),
    'resource': baseline_evidence.get('resource'),
    'historical_summary': baseline_evidence.get('historical_summary'),
    'runs': [{key: value for key, value in run.items() if key != 'observations'}
             for run in baseline_evidence['runs']],
}, ensure_ascii=False, indent=2))
```

### 5.6 判断可引用状态并登记

按 Step 3 的规则读取状态及其失败项、缺失项。`close` 保存登记回执，仅在 `ready` 时生成带适用条件的基线引用；重复执行复用相同内容，同名内容冲突时停止。其他阶段不写最终登记。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 66 单卡基线；历史 `rtx5070ti_vllm_g0_v4` |
| 运行条件 | RTX 5070 Ti Laptop GPU；Qwen2.5-0.5B-Instruct；vLLM；每轮 30 个固定请求 |
| 原始证据 | [后端日志](../../benchmarks/results/66_backend_rtx5070ti_vllm_g0_v4.log) |
| 正式证据 | [历史 G0 结果](../../benchmarks/results/66_g0_vllm_rtx5070ti_vllm_g0_v4.json) |
| 清单与交接 | 历史报告未生成新协议引用；后续使用须匹配模型与测量口径 |
| 结论 | 原协议登记 `ready`，30/30 请求成功；缺少首末内容独立时间，新协议尚未实测。只读核对显示 `tune` 表示需要补采，不改写历史结论 |

```python
if RUN_STAGE == 'close':
    registration_receipt = baseline_runtime.close_baseline(plan, baseline_evidence)
else:
    registration_receipt = {
        'status': 'not_registered', 'stage': RUN_STAGE,
        'decision': baseline_evidence['decision'],
    }
print(json.dumps(registration_receipt, ensure_ascii=False, indent=2))
```

## 相关阅读

解释基线中的生成、缓存或执行现象时，再选择对应机制；专项比较使用匹配的基线身份与协议。

- [Part02·21 解码与停止](./21_Decoding_Strategies.md)：生成长度与停止规则。
- [Part02·22 分页注意力](./22_vLLM_PagedAttention.md)：状态容量与块分配。
- [Part02·68 投机解码基准](./68_Speculative_Decoding_Benchmark.md)：固定生成规则下的专项比较。
- [推理优化：基准与决策](../topic_discussion/inference_optimization/07_benchmark_and_decision.md)：单项归因、冷／热状态与组合复测。
- [vLLM 指标](https://docs.vllm.ai/en/latest/design/metrics.html)：区分服务端指标与客户端观察。
