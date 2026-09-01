# 74. Profiling Driven End to End Optimization | profiling 驱动的显存优化端到端收口

**难度：** Hard | **环境：** CPU 可完成报告收口；GPU 用于可选 trace | **标签：** `显存优化`, `性能剖析`, `端到端优化` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节把 73 / 76 / 75 的结果与 profiler 证据接起来：先确认已有结论，再用 trace 解释时间和显存代价。主线只分析一个代表性 baseline 与 candidate；其他 workload 用于扩展，不替代主线证据。推理方向可选用 71 的 MLA workload 做同样的 trace 分析，但 74 不重新讲 MLA 机制。
**层级定位：** 本项目是跨层 profiling 项目，覆盖 L1-L4 的执行证据；若采集资源调度、版本发布或服务可用性指标，才延伸到 L5。它负责定位和验证，不替代具体的算子优化、推理服务或平台治理项目。

**输入与输出：** 输入是 73 的 baseline、76 的策略 benchmark 和 75 的预算决策；输出是带 trace 证据的收口报告。CPU 路径只合并报告和检查证据，GPU 路径才采集代表性 baseline / candidate 的 CPU/CUDA trace；没有 trace 时只能报告证据缺口。
**实验分层：** CPU 验证报告链路，GPU 验证阶段耗时、kernel / 内存活动和同步证据；二者都不能代替另一方。
**主责与复用边界：** 本项目主责是用 trace 检查显存优化解释是否成立；性能分析专题复用采集和归因方法，算子优化复用 kernel 证据，推理项目只作为扩展 workload，不在本项目重新选择量化、调度或并行策略。

> 运行提示：先查看[使用指南中的项目环境预检与安装说明](../docs/guide.md#项目环境预检与安装)，再打开真实 profiling 开关。CPU 路径只合并报告；CUDA trace 必须先通过 GPU 预检。

**关键词：** `profiling`, `optimization`, `end-to-end`

---

## 前置阅读

**导语：** 先完成 73、76、75 的训练侧测量、策略比较和预算决策，再用 profiling 把显存优化方案带入端到端 workload，判断改动是否值得最终保留。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)（推理扩展）
- [71. MLA / KV Cache Architecture Benchmark | MLA / KV Cache 结构基准](./71_MLA_KV_Cache_Architecture_Benchmark.md)（推理 profiling 扩展）

## 相关阅读

**导语：** 完成优化闭环后，可以继续把瓶颈定位推进到更底层的实现手段，或回到并行/系统项目页验证是否值得迁移。
- [79. Distributed Parallel Benchmark | 分布式并行基准项目](./79_Distributed_Parallel_Benchmark.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
---
### Step 1: 定义端到端优化目标
先回答一个问题：这次优化到底要解决什么瓶颈，成功标准是什么？

- 主线固定模型、输入数据、batch size、seq len、硬件环境和运行后端，保证 trace 可比较；扩展 workload 必须另列为独立证据。
- 明确优化目标，例如降低 step time、提升 throughput、降低 peak memory 或减少通信等待，并先回到 Task 1 的账本判断问题属于容量、带宽、计算还是同步。
- 同时写清约束条件：训练任务要保留 loss / accuracy 约束，推理任务要保留精度、输出一致性或服务 SLA 约束。
- Baseline 需要能稳定复现，不能只跑一次；建议至少 warm-up 若干轮，再测多轮平均值。
- 这一步的目标是让后面的优化有判断标准，而不是只得到一组孤立数字。

### Step 2: 先确认 baseline 和 profiling 口径合法

profiling 优化必须先确认 baseline 可复现，再把“慢”拆成可解释的瓶颈类型，不能直接对着单次热点截图开刀。

- 推荐先记录总耗时、吞吐、峰值显存，再用 profiler 看热点算子和同步点；73 / 76 的表格告诉你“发生了什么”，trace 用来解释“为什么”。
- 训练场景优先拆成：数据加载、forward、backward、optimizer step、显存峰值和多卡通信，并特别观察 19 的 checkpoint 重算、42 的 CPU-GPU 搬运是否出现在时间线上。
- 推理场景优先拆成：prefill、decode、KV cache、采样逻辑、数据搬运和 kernel 开销；如果使用 71 的 DeepSeek-V2-Lite / MLA workload，还要单独观察 latent KV 读写、位置相关分量和对应 kernel。
- 不要只找“最慢的一行代码”，而要判断瓶颈属于哪一类资源：计算、显存容量、内存带宽、通信还是调度；一个算子占比高不等于它就是可优化的根因。
- 这一步的产物应该是一句话瓶颈结论，例如：`当前瓶颈主要来自 decode 阶段 KV cache 读取`。

### Step 3: 用统一口径比较收益与代价

profiling 项目必须同时看 step time、throughput、peak memory 和任务约束，不能只挑单项热点收益下结论。

- 一次只改一个方向，例如调整 batch size、开启混合精度、替换 kernel、减少同步点或改变 cache 策略；改动必须能回应 trace 中的具体证据。
- 改完后重新测同样的指标，比较 baseline / tuned 的差异。
- 如果改动影响训练 loss、推理输出、显存峰值或系统稳定性，要把代价写清楚。
- 如果某个改动只是在一项指标上变好，却让另一项变差，要把取舍写清楚。
- 这一轮修改的目标是建立因果关系，而不是一次性把所有优化开关都打开。

### Step 4: 输出端到端优化结论

端到端优化最终不是输出“某个热点是不是降了”，而是输出这次改动在当前任务约束下是否值得继续保留、微调或回退。

- 输出 baseline / tuned 对比表，至少包含 step time、throughput、peak memory 和备注。
- 附上 profiling 截图或关键统计，说明瓶颈来自哪一类资源。
- 写清楚本次改动、收益、代价和是否满足约束。
- 如果优化没有达到目标，记录失败原因和下一轮优先级。
- 将端到端结果与 75 的预算决策对照；如果 profiling 推翻了局部结论，要记录是 workload、同步、数据搬运还是系统开销造成的。
- 最终产物应回答：原始瓶颈是什么，做了什么改动，收益有多大，这个改动是否值得保留。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整 profiling 驱动优化流程。下面的代码实现其中最小、可复用的四块：测平均耗时、汇总 baseline / tuned 指标差异、生成优化报告，以及把结果收成 `accept / tune / reject` 的轻量决策。真实项目中的 profiling 截图、瓶颈证据和优化策略，需要基于这四步继续补充。

### Step 6: 按统一协议保存项目结果

74 与 73、76、75 共用实验外层字段：模型、设备、dtype、workload、baseline / tuned、step time、throughput、peak memory、质量约束和 `accept / tune / reject`。74 额外保存 profiling 证据，因此不能只复制显存策略表。

建议额外记录：`profile.tool`、`profile.top_operators`、`profile.compute_ratio`、`profile.memory_ratio`、`profile.communication_ratio`，以及 `bottleneck.category`、`bottleneck.evidence`、`bottleneck.optimization`。如果当前环境没有真实 profiler，允许这些字段为空，但不能把未测量内容写成结论。

项目结果协议由 `tools/profiling_result_schema.py` 提供。它不会覆盖原始实验数据；真实 GPU 或 Colab 环境完成 profiling 后，可将结果保存为 `benchmarks/results/74_profiling_optimization.json`。

### 提示

- 先固定 baseline，再做 profiling，再改一个变量。
- 不要只看 step time，至少同时记录 throughput 和 peak memory。
- 如果瓶颈不是单一算子，而是同步 / 调度 / cache，报告里要直接写出来。
- 真实 GPU 场景可调用答案区的 `collect_torch_profile` 保存短 trace；不要把未采集的 top operators 填成推测。
### 参数口径说明

profiling 实验必须固定 model、workload、batch、seq_len、warmup、iters、backend 和硬件；`warmup` 不计入正式测量，`iters` 决定均值稳定性。优化目标可以是 step time、throughput、peak memory 或通信等待，但一次只改一个方向，并同时保留质量/稳定性约束。

```python
import time

```


```python
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

def synchronize_if_cuda():
    """Synchronize CUDA when available; keep CPU-first execution unchanged."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_fn(fn, warmup=3, iters=10):
    """测量 CPU/GPU 函数的平均耗时；GPU 需由同步函数包住测量区间。"""
    if warmup < 0 or iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和项目报告对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    # start = ???
    for _ in range(iters):
        fn()
    # total = ???
    # avg_time_ms = ???
    return avg_time_ms

def summarize_optimization_result(base_metrics, tuned_metrics):
    """统一计算 baseline - tuned 的时间、显存和吞吐差异。"""
    # ==========================================
    # TODO 2: 汇总 baseline / tuned 的核心指标差异
    # 提示: 正数表示 tuned 相比 baseline 有改善
    # step time / memory 越低越好，throughput 越高越好
    # ==========================================
    # time_delta = ???
    # memory_delta = ???
    # throughput_delta = ???

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    """输出指标变化、瓶颈证据边界和下一步动作。"""
    # ==========================================
    # TODO 3: 生成一段最小优化报告
    # 提示: 把指标变化、瓶颈结论和下一步动作放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    # rows = ???
    # conclusion = ???
    return "\n".join([header, sep] + rows + [conclusion])


def recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0):
    """使用显式阈值输出 accept / tune / reject，不把 CPU 结果写成 GPU 结论。"""
    # ==========================================
    # TODO 4: 给出轻量优化决策
    # 规则：
    # - 时间和吞吐都改善：accept
    # - 时间改善，且显存或吞吐至少有一项为正收益：tune
    # - 否则：reject
    # ==========================================
    # strong_time_gain = ???
    # strong_memory_gain = ???
    # strong_throughput_gain = ???
    # if ???:
    #     decision = ???
    #     reason = ???
    # positive_memory_gain = summary['peak_mem_delta_mb'] > 0
    # positive_throughput_gain = summary['throughput_delta'] > 0
    # elif ???:
    #     decision = ???
    #     reason = ???
    # else:
    #     decision = ???
    #     reason = ???
    # return {'decision': decision, 'reason': reason}
    raise NotImplementedError

```

### 测试


```python
def test_optimization_project_template():
    try:
        counter = {'n': 0}

        def fn():
            counter['n'] += 1

        result = benchmark_fn(fn, warmup=0, iters=2)
        assert counter['n'] == 2, "benchmark 应该运行 iters 次"
        assert result >= 0.0, "平均耗时应该非负"
        for invalid in ({'warmup': -1, 'iters': 2}, {'warmup': 0, 'iters': 0}):
            try:
                benchmark_fn(fn, **invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('非法 warmup / iters 应明确拒绝')

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0, 'throughput': 80.0}
        tuned = {'step_time_ms': 96.0, 'peak_mem_mb': 7168.0, 'throughput': 100.0}
        summary = summarize_optimization_result(baseline, tuned)

        assert summary['step_time_delta_ms'] == 24.0
        assert summary['peak_mem_delta_mb'] == 1024.0
        assert summary['throughput_delta'] == 20.0
        assert summary['time_improved'] is True
        assert summary['memory_improved'] is True
        assert summary['throughput_improved'] is True

        report = format_optimization_report(summary, 'backward kernel 占比过高', '保留混合精度并继续检查 optimizer')
        assert '| 指标 | 变化 | 判断 |' in report
        assert 'backward kernel 占比过高' in report
        assert '保留混合精度并继续检查 optimizer' in report

        decision = recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert decision['decision'] == 'accept'

        mixed_summary = {'step_time_delta_ms': 12.0, 'peak_mem_delta_mb': 128.0, 'throughput_delta': 2.0, 'time_improved': True, 'memory_improved': True, 'throughput_improved': True}
        mixed_decision = recommend_optimization_decision(mixed_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert mixed_decision['decision'] == 'tune'

        weak_summary = {'step_time_delta_ms': -3.0, 'peak_mem_delta_mb': 64.0, 'throughput_delta': 1.0, 'time_improved': False, 'memory_improved': True, 'throughput_improved': True}
        weak_decision = recommend_optimization_decision(weak_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert weak_decision['decision'] == 'reject'

        print("✅ profiling 驱动的端到端优化项目模板代码通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_optimization_project_template()

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
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

def synchronize_if_cuda():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_torch_profile(train_step_fn, output_dir='benchmarks/results/74_profile', warmup=2, iters=5):
    """Collect a short CPU/CUDA trace and return report metadata."""
    if torch is None:
        raise RuntimeError('需要安装 PyTorch 才能采集 profiler trace')
    if warmup < 0 or iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        train_step_fn()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    schedule = torch.profiler.schedule(wait=0, warmup=1 if iters > 1 else 0, active=max(1, iters - 1), repeat=1)
    trace_handler = torch.profiler.tensorboard_trace_handler(str(output_path))
    with torch.profiler.profile(activities=activities, schedule=schedule, on_trace_ready=trace_handler, record_shapes=True, profile_memory=True, with_stack=False) as prof:
        for _ in range(iters):
            train_step_fn()
            prof.step()
    return {
        'status': 'collected',
        'tool': 'torch.profiler',
        'activities': [activity.name for activity in activities],
        'trace_dir': str(output_path),
        'warmup': warmup,
        'iters': iters,
    }


def benchmark_fn(fn, warmup=3, iters=10):
    if warmup < 0 or iters <= 0:
        raise ValueError('warmup must be >= 0 and iters must be > 0')
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和项目报告对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    synchronize_if_cuda()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize_if_cuda()
    total = time.perf_counter() - start
    avg_time_ms = total / iters * 1000
    return avg_time_ms


def summarize_optimization_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / tuned 的核心指标差异
    # 提示: 正数表示 tuned 相比 baseline 有改善
    # step time / memory 越低越好，throughput 越高越好
    # ==========================================
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    memory_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    throughput_delta = tuned_metrics['throughput'] - base_metrics['throughput']

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    # ==========================================
    # TODO 3: 生成一段最小优化报告
    # 提示: 把指标变化、瓶颈结论和下一步动作放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| step time | {summary['step_time_delta_ms']} ms | {'改善' if summary['time_improved'] else '未改善'} |",
        f"| peak memory | {summary['peak_mem_delta_mb']} MB | {'改善' if summary['memory_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
    ]
    conclusion = f"瓶颈判断：{bottleneck}。下一步：{next_action}。"
    return "\n".join([header, sep] + rows + [conclusion])


def recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0):
    strong_time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
    strong_memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
    strong_throughput_gain = summary['throughput_delta'] >= min_throughput_delta
    positive_memory_gain = summary['peak_mem_delta_mb'] > 0
    positive_throughput_gain = summary['throughput_delta'] > 0
    if strong_time_gain and strong_throughput_gain:
        decision = 'accept'
        reason = '时间与吞吐改善都达标，当前优化值得保留。'
    elif strong_time_gain and (positive_memory_gain or positive_throughput_gain):
        decision = 'tune'
        reason = '时间改善成立，但资源或吞吐收益还不够稳，建议继续微调。'
    else:
        decision = 'reject'
        reason = '当前优化没有形成稳定的端到端收益，建议回退或重新定位瓶颈。'
    return {'decision': decision, 'reason': reason}

```

### 解析

这一版题目区保留 `4` 个核心 TODO：测量、汇总、报告和轻量决策。这里不把 profiling 页做成重型项目审计器，而是让读者先掌握 `measure -> summarize -> report -> decide` 的最小项目闭环。

- **这一题要解决什么**：把 profiling 优化流程压缩成一个最小可复用模板，保证每次优化都能留下可比较的指标和明确结论。
- **为什么这样做**：性能优化不能只看单次运行结果，必须固定 baseline、测量同一组指标，并把改动前后的差异收敛成项目报告。
- **带走的直觉**：profiling 的价值不是“找到一个慢点”，而是建立 `测量 -> 定位 -> 修改 -> 复测 -> 复盘` 的闭环。

**1. TODO 1 (benchmark_fn)**

- **warmup**：先运行若干轮，不计入统计，避免初始化、缓存和调度抖动影响结果。
- **计时范围**：只把正式测量的 `iters` 轮放进 `start / total` 之间。
- **单位统一**：返回 ms，而不是秒，方便和 step time、latency 表格放在一起比较。
- **工程注意**：答案代码在 CUDA 可用时会在计时前后调用 `torch.cuda.synchronize()`；CPU 环境下自动跳过同步。

**2. TODO 2 (summarize_optimization_result)**

- **step time 差值**：`baseline - tuned`，正数表示 tuned 更快。
- **peak memory 差值**：`baseline - tuned`，正数表示 tuned 更省显存。
- **throughput 差值**：`tuned - baseline`，正数表示 tuned 吞吐更高。
- **布尔判断**：`time_improved / memory_improved / throughput_improved` 把数值变化变成可读结论，方便项目报告直接引用。

**3. TODO 3 (format_optimization_report)**

- **表格部分**：把核心指标变化放到同一张 Markdown 表里，便于复盘和横向比较。
- **瓶颈判断**：不要只输出数字，还要写清楚瓶颈来自哪里，例如数据加载、backward kernel、KV cache 或通信同步。
- **下一步动作**：每轮优化结束都应该留下后续优先级，否则下一轮很容易重新从零开始定位。

**4. TODO 4 (recommend_optimization_decision)**

- **accept**：时间改善和吞吐改善都达标，说明这次改动对端到端目标确实有帮助。
- **tune**：时间改善成立，但显存或吞吐收益还不够稳，说明这次优化方向可能对，但还没到可直接保留的程度。
- **reject**：没有形成稳定的端到端收益，应该回退或重新定位瓶颈，而不是继续堆优化开关。

**项目化原则**

- **一次只改一个变量**：否则收益不可归因。
- **指标要成组出现**：只看变快不够，还要看显存、吞吐、loss / 精度或输出一致性。
- **结论要回扣目标**：最终判断必须回答 Step 1 的问题：这次优化是否达成目标，是否值得保留。

### Step 7（可选）：真实 GPU trace 采集

本 Step 使用 76 的 `seq_len=768` FP32 workload，实际运行 baseline / checkpoint 的短训练 step，并保存 `torch.profiler` 的 CPU/CUDA trace。默认关闭；没有 GPU 时不要运行。它只负责采集证据，最终报告仍由后面的收口代码生成。

**可选的 MLA profiling 扩展：** 如果要验证 71 的 MLA / KV Cache 结构，应使用同一模型、输入长度和 backend，单独记录 prefill / decode、latent KV 读写、位置相关分量、kernel 时间和显存变化。71 负责回答“缓存表示如何变化”，74 负责回答“该表示在真实执行中带来什么时间与带宽代价”；当前训练 trace 代码不能仅通过替换模型名变成 MLA 推理 trace，因此没有对应 collector 时应标记为未采集。

```python
from pathlib import Path

RUN_REAL_PROFILE = False  # 默认先完成 CPU 报告收口；真实 GPU 采集时显式改为 True。
PROFILE_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
PROFILE_BATCH_SIZE = 1  # 必须与 76 的代表性 workload 对齐。
PROFILE_SEQ_LEN = 768  # 只 profile 一个代表性 workload；其他长度另存 trace。
PROFILE_WARMUP = 2  # 不计入 trace 结论的预热轮数。
PROFILE_ITERS = 5  # 短 trace 采集轮数；只用于归因，不替代 73 / 76 的正式均值。
PROFILE_STRATEGIES = ['baseline', 'checkpoint']  # 先采最小对照；candidate 应来自 76 的可行方案。
if PROFILE_BATCH_SIZE <= 0 or PROFILE_SEQ_LEN <= 0 or PROFILE_WARMUP < 0 or PROFILE_ITERS <= 0:
    raise ValueError('PROFILE_BATCH_SIZE/PROFILE_SEQ_LEN/PROFILE_ITERS 必须 > 0，PROFILE_WARMUP 不能为负数。')
if not PROFILE_STRATEGIES or len(PROFILE_STRATEGIES) != len(set(PROFILE_STRATEGIES)):
    raise ValueError('PROFILE_STRATEGIES 不能为空且不能包含重复策略。')
if any(strategy not in {'baseline', 'checkpoint', 'offload', 'hybrid'} for strategy in PROFILE_STRATEGIES):
    raise ValueError('PROFILE_STRATEGIES 只能使用 baseline/checkpoint/offload/hybrid。')
PROFILE_76_RELATIVE_PATH = Path('benchmarks/results/76_real_gpu_memory.json')
PROFILE_TRACE_RELATIVE_DIR = Path('benchmarks/results/74_profile')
PROFILE_OUTPUT_RELATIVE_PATH = Path('benchmarks/results/74_real_gpu_profile.json')

```


```python
import gc
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

if RUN_REAL_PROFILE:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    project_root = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
    for candidate in (project_root, *project_root.parents):
        if (candidate / 'benchmarks').is_dir() and (candidate / '02_PyTorch_Algorithms').is_dir():
            project_root = candidate
            break
    else:
        colab_root = Path('/content/llm-algo-leetcode')
        if colab_root.is_dir() and (colab_root / 'tools/project_runtime.py').is_file():
            project_root = colab_root
        elif Path('/content').is_dir() and not colab_root.exists():
            subprocess.run([
                'git', 'clone',
                'https://github.com/datawhalechina/llm-algo-leetcode.git',
                str(colab_root),
            ], check=True)
            project_root = colab_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tools.model_runtime import resolve_model
    from tools.profiling_runtime import collect_training_trace
    from tools.project_runtime import environment_preflight, standard_experiment_config, standard_training_metrics
    preflight = environment_preflight(torch, required_packages=('transformers',), require_gpu=True)
    print({'environment_preflight': preflight})
    if not preflight['ready']:
        raise RuntimeError('环境预检未通过，请先按 next_actions 修复；没有开始 profiling。')

    upstream_path = project_root / PROFILE_76_RELATIVE_PATH
    if not upstream_path.exists():
        raise FileNotFoundError(f'找不到 76 结果：{upstream_path}')
    upstream = json.loads(upstream_path.read_text(encoding='utf-8'))
    upstream_config = upstream.get('config', {})
    for key, value in {'model_id': PROFILE_MODEL_ID, 'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN}.items():
        if upstream_config.get(key) != value:
            raise ValueError(f'74 与 76 的 profiling workload 不一致：{key}={upstream_config.get(key)} != {value}')

    model_path = resolve_model(PROFILE_MODEL_ID, source='auto', cache_dir='model_cache')
    model_config = AutoConfig.from_pretrained(model_path)
    generator = torch.Generator(device='cpu').manual_seed(42)
    input_ids_cpu = torch.randint(0, model_config.vocab_size, (PROFILE_BATCH_SIZE, PROFILE_SEQ_LEN), generator=generator)
    trace_root = project_root / PROFILE_TRACE_RELATIVE_DIR
    trace_root.mkdir(parents=True, exist_ok=True)
    candidates = []

    for strategy in PROFILE_STRATEGIES:
        trace_dir = trace_root / strategy
        trace_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        model.config.use_cache = False
        if strategy == 'checkpoint':
            model.gradient_checkpointing_enable()
        model.to('cuda').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        input_ids = input_ids_cpu.to('cuda')
        labels = input_ids.clone()

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            return loss

        measured = collect_training_trace(
            train_step, torch_module=torch, output_dir=trace_dir,
            warmup=PROFILE_WARMUP, iters=PROFILE_ITERS,
            batch_size=PROFILE_BATCH_SIZE,
        )
        trace_files = sorted(trace_dir.glob('*.pt.trace.json'))
        if not trace_files:
            raise RuntimeError(f'{strategy} 未生成 .pt.trace.json；请检查 profiler schedule 和输出目录：{trace_dir}')
        candidates.append({
            'name': strategy, 'status': 'ok', 'step_time_ms': measured['step_time_ms'],
            'samples_per_s': measured['samples_per_s'], 'loss': measured['loss'],
            'peak_memory_mb': measured['peak_memory_mb'], 'peak_reserved_mb': measured['peak_reserved_mb'],
            'trace_dir': str(trace_dir.relative_to(project_root)), 'top_operators': measured['top_operators'],
        })
        del optimizer, model, input_ids, labels
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        'task': 'task3_training_memory_optimization', 'stage': 'real_profiler_trace',
        'config': {'model_id': PROFILE_MODEL_ID, 'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN, 'dtype': 'float32', 'strategies': PROFILE_STRATEGIES, 'warmup': PROFILE_WARMUP, 'iters': PROFILE_ITERS},
        'source_76': str(upstream_path.relative_to(project_root)), 'trace_root': str(trace_root.relative_to(project_root)),
        'candidates': candidates, 'profile': {'tool': 'torch.profiler', 'activities': ['CPU', 'CUDA'], 'status': 'collected', 'trace_files_by_strategy': {item['name']: [str(path) for path in sorted((trace_root / item['name']).glob('*.pt.trace.json'))] for item in candidates}},
    }
    report['experiment'] = standard_experiment_config({
        'model_id': PROFILE_MODEL_ID, 'backend': 'torch.profiler',
        'dtype': 'float32', 'optimizer': 'AdamW',
        'batch_size': PROFILE_BATCH_SIZE, 'seq_len': PROFILE_SEQ_LEN,
        'warmup': PROFILE_WARMUP, 'iters': PROFILE_ITERS, 'seed': 42,
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
    })
    report['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in candidates}
    output_path = project_root / PROFILE_OUTPUT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
else:
    print('跳过真实 profiler：保持 CPU-first 模式。')

```

### Step 8（可选）：读取 73 / 76 / 75 并生成收口报告

这一单元读取已经保存的上游 JSON 和真实 profiler 报告，检查项目证据是否齐全，并生成 `benchmarks/results/74_profiling_optimization.json`。如果没有真实 profiling 证据，报告会明确标记为 `tune`，不会把 75 的局部预算结论直接升级为端到端结论。

```python
import json
from pathlib import Path
import sys

project_root = Path.cwd().resolve()
for candidate_root in (project_root, *project_root.parents):
    if (candidate_root / 'tools/project_runtime.py').is_file():
        if str(candidate_root) not in sys.path:
            sys.path.insert(0, str(candidate_root))
        break
from tools.project_runtime import resolve_project_root, standard_experiment_config, standard_training_metrics

RUN_REAL_PROJECT = True  # True：读取已有 73/76/75 JSON 并生成 74 报告。
PROJECT_ROOT = resolve_project_root()

UPSTREAM_PATHS = {
    '73': PROJECT_ROOT / 'benchmarks/results/73_real_gpu_training.json',
    '76': PROJECT_ROOT / 'benchmarks/results/76_real_gpu_memory.json',
    '75': PROJECT_ROOT / 'benchmarks/results/75_memory_budget_decision.json',
}
OUTPUT_PATH = PROJECT_ROOT / 'benchmarks/results/74_profiling_optimization.json'
PROFILE_TRACE_DIR = PROJECT_ROOT / 'benchmarks/results/74_profile'  # torch.profiler 输出目录。
PROFILE_REPORT_PATH = PROJECT_ROOT / 'benchmarks/results/74_real_gpu_profile.json'

def load_required_reports(paths):
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('缺少上游报告：' + ', '.join(missing))
    return {name: json.loads(path.read_text(encoding='utf-8')) for name, path in paths.items()}

def inspect_profile_trace(trace_dir, expected_strategies=None):
    """只确认 trace 是否存在，不把文件存在误写成瓶颈结论。"""
    trace_dir = Path(trace_dir)
    expected_strategies = list(expected_strategies or [])
    trace_files = sorted(
        path for path in trace_dir.rglob('*')
        if path.is_file() and path.name.endswith('.pt.trace.json')
    ) if trace_dir.is_dir() else []
    if not trace_files:
        return {
            'status': 'not_collected', 'tool': None, 'trace_dir': str(trace_dir),
            'trace_files': [], 'evidence': [],
        }
    strategy_dirs = {name: sorted((trace_dir / name).glob('*.pt.trace.json')) for name in expected_strategies}
    missing_strategies = [name for name, files in strategy_dirs.items() if not files]
    return {
        'status': 'collected', 'tool': 'torch.profiler', 'trace_dir': str(trace_dir),
        'trace_files': [str(path) for path in trace_files],
        'evidence': ['trace_file_exists'] if not missing_strategies else ['trace_file_exists', 'missing_expected_strategy_trace'],
        'expected_strategies': expected_strategies,
        'missing_strategies': missing_strategies,
        'trace_files_by_strategy': {name: [str(path) for path in files] for name, files in strategy_dirs.items()},
    }

def build_upstream_report(reports):
    if PROFILE_REPORT_PATH.is_file():
        profile_seed = json.loads(PROFILE_REPORT_PATH.read_text(encoding='utf-8'))
        expected_strategies = [item.get('name') for item in profile_seed.get('candidates', []) if item.get('name')]
    else:
        expected_strategies = ['baseline', 'checkpoint']
    profiling = inspect_profile_trace(PROFILE_TRACE_DIR, expected_strategies=expected_strategies)
    if PROFILE_REPORT_PATH.is_file():
        profile_report = json.loads(PROFILE_REPORT_PATH.read_text(encoding='utf-8'))
        profiling['report_path'] = str(PROFILE_REPORT_PATH)
        profiling['top_operators'] = {item['name']: item.get('top_operators', '') for item in profile_report.get('candidates', [])}
    if profiling['status'] == 'collected' and not profiling.get('missing_strategies'):
        profile_decision = {'decision': 'tune', 'reason': 'profiler_trace_collected_but_requires_bottleneck_interpretation', 'next_action': 'interpret_trace_and_validate_one_change'}
    else:
        profile_decision = {'decision': 'tune', 'reason': 'profiler_trace_incomplete_for_declared_candidates', 'next_action': 'collect_missing_strategy_trace_and_rerun_same_workload'}
    report = {
        'schema_version': 'profiling-project/v1',
        'project': '74_profiling_driven_end_to_end_optimization',
        'stage': 'upstream_report_merge',
        'sources': {name: str(path) for name, path in UPSTREAM_PATHS.items()},
        'upstream': {
            '73_training_baseline': reports['73'].get('baseline'),
            '76_strategy_summary': reports['76'].get('summary'),
            '76_decision': reports['76'].get('decision'),
            '75_budget_summary': reports['75'].get('summary'),
            '75_decision': reports['75'].get('decision'),
        },
        'profiling': profiling,
        'decision': profile_decision,
    }
    report['experiment'] = standard_experiment_config(reports['76'].get('config', {}))
    report['standard_metrics'] = {
        item.get('name', f'candidate_{index}'): standard_training_metrics(item)
        for index, item in enumerate(reports['76'].get('candidates', []))
        if isinstance(item, dict)
    }
    return report

if RUN_REAL_PROJECT:
    upstream_reports = load_required_reports(UPSTREAM_PATHS)
    project_report = build_upstream_report(upstream_reports)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(project_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(project_report, ensure_ascii=False, indent=2))
else:
    print('跳过 74 上游报告收口：保持 CPU-first 模式。')

```


```python
# 运行本单元，自动整理真实 profiler 数据和结论。
import json
from pathlib import Path

project_root = Path.cwd().resolve()
for candidate_root in (project_root, *project_root.parents):
    if (candidate_root / 'benchmarks').is_dir() and (candidate_root / '02_PyTorch_Algorithms').is_dir():
        project_root = candidate_root
        break
else:
    colab_root = Path('/content/llm-algo-leetcode')
    if (colab_root / 'benchmarks').is_dir():
        project_root = colab_root
PROJECT_ROOT = project_root
profile_report_path = PROJECT_ROOT / 'benchmarks/results/74_real_gpu_profile.json'
close_report_path = PROJECT_ROOT / 'benchmarks/results/74_profiling_optimization.json'
if not profile_report_path.is_file():
    raise FileNotFoundError(f'找不到真实 profiler 报告：{profile_report_path}')

profile_report = json.loads(profile_report_path.read_text(encoding='utf-8'))
rows = []
for item in profile_report.get('candidates', []):
    rows.append({
        'strategy': item.get('name'),
        'step_time_ms': item.get('step_time_ms'),
        'samples_per_s': item.get('samples_per_s'),
        'peak_memory_mb': item.get('peak_memory_mb'),
        'peak_reserved_mb': item.get('peak_reserved_mb'),
        'status': item.get('status'),
        'trace_dir': item.get('trace_dir'),
    })

print('74 真实 profiler 数据（仅 baseline / checkpoint）')
print(json.dumps({
    'config': profile_report.get('config'),
    'candidates': rows,
    'trace': profile_report.get('profile'),
}, ensure_ascii=False, indent=2))

if len(rows) == 2 and all(row['status'] == 'ok' for row in rows):
    base = next(row for row in rows if row['strategy'] == 'baseline')
    tuned = next(row for row in rows if row['strategy'] == 'checkpoint')
    print('\n可比较指标')
    print(f"显存变化：{base['peak_memory_mb']:.2f} → {tuned['peak_memory_mb']:.2f} MB，减少 {base['peak_memory_mb'] - tuned['peak_memory_mb']:.2f} MB")
    print(f"步耗时变化：{base['step_time_ms']:.2f} → {tuned['step_time_ms']:.2f} ms，增加 {tuned['step_time_ms'] - base['step_time_ms']:.2f} ms")
    print(f"吞吐变化：{base['samples_per_s']:.3f} → {tuned['samples_per_s']:.3f} samples/s")
    print('解释：checkpoint 降低了峰值显存，但增加了计算时间；这是用重算换显存。')
    print('证据边界：当前 trace 可以支持 baseline / checkpoint 的时间线对比，不能单独证明某个 kernel 的硬件效率或最终 accept。')

if close_report_path.is_file():
    close_report = json.loads(close_report_path.read_text(encoding='utf-8'))
    print('\n项目收口')
    print(json.dumps(close_report.get('decision', {}), ensure_ascii=False, indent=2))
    print('下一步：在 Perfetto 中分别打开 baseline / checkpoint trace，确认 forward、backward 和 optimizer step 的时间线，再验证一个针对瓶颈的改动。')

```

## 本次真实 GPU 结果

实验条件：`Qwen/Qwen2.5-0.5B-Instruct`、RTX 5070 Ti Laptop GPU、FP32、`batch_size=1`、`seq_len=768`、`warmup=2`、`iters=5`、AdamW。74 节只对 baseline 和 checkpoint 采集 trace；offload / hybrid 的策略指标见 76 节。

| 策略 | 单步耗时（ms） | 吞吐（samples/s） | 峰值显存（MB） | reserved（MB） | last loss | Trace |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 690.375 | 1.448 | 9782.73 | 10750.00 | 11.477463 | 已采集 |
| checkpoint | 825.059 | 1.212 | 9450.75 | 10896.00 | 11.477463 | 已采集 |

| 对比项 | 结果 |
| --- | --- |
| 峰值显存 | checkpoint 减少 331.98 MB，约 3.39% |
| 单步耗时 | 增加 134.68 ms，约 19.51% |
| 吞吐 | 从 1.448 降至 1.212 samples/s，约下降 16.30% |
| 当前决策 | `tune`：已取得 trace，但还没有完成瓶颈解释和改动后的复验 |

这组数据支持“checkpoint 用重计算换显存”的结论，但不支持“checkpoint 已经是最终最优方案”。Profiler 会增加额外开销，因此表中的耗时用于观察相对变化，最终性能仍以 73 / 76 的非 profiler 测量为准。

## 真实 trace 怎么看

74 节当前生成的是 `torch.profiler` 的 Chrome trace，不是 TensorBoard event 文件。可以打开 <https://ui.perfetto.dev>，分别加载：

- `benchmarks/results/74_profile/baseline/*.pt.trace.json`
- `benchmarks/results/74_profile/checkpoint/*.pt.trace.json`

每次只看一个文件，并在相同的 `ProfilerStep` 区间比较。先定位 forward、backward、optimizer step，再观察 checkpoint 是否在 backward 中增加了重复计算，以及 GPU 时间线上是否出现等待空洞。

`Command Buffer Full`、`cudaLaunchKernel` 和 CUDA 总时间可能包含异步活动或重叠，不能仅凭表格中的百分比认定为瓶颈；需要结合时间线和相同采集配置判断。Perfetto 的 slice overlap 提示通常只影响显示布局，不代表 trace 数据丢失。

本次结果支持的结论是：checkpoint 用约 `331.98 MB` 峰值显存下降换取约 `134.68 ms` 的单步时间代价。若要把 `tune` 升级为 `accept`，还需要针对时间线中确认的瓶颈验证一次改动，并重新比较相同 workload 的时间、吞吐、显存和质量。