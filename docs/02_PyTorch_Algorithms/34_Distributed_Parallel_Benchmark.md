# 34. Distributed Parallel Benchmark | 分布式并行基准

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战` | **目标人群：** 工程实践

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/34_Distributed_Parallel_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


**关键词：** `distributed training, benchmark, parallelism`

这个项目把 ZeRO、Pipeline 和 Tensor Parallelism 放到同一套 benchmark 里比较，形成并行策略选型依据。它承接 `2.8` 的分布式并行策略，也方便和 `Part 1` 的 profiling 方法串成一个可复用的评测框架。

## 前置阅读

**导语：** 先把最小前置看完，再进入项目会更容易把问题拆开。
- [2.8. Distributed Parallel Strategy | 分布式并行策略](./2_8.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 05. Communication Topologies | 通信拓扑与分布式基石](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)

## 相关阅读

**导语：** 如果想继续往更底层的工程背景延伸，可以看这些页面。
- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](./27_ZeRO_Optimizer_Sim.md)
- [28. Pipeline Parallelism MicroBatch | Pipeline 并行微批次](./28_Pipeline_Parallelism_MicroBatch.md)
- [29. Tensor Parallelism Sim | Tensor 并行模拟](./29_Tensor_Parallelism_Sim.md)

### Step 1: 统一实验设置
先固定模型、输入、设备和评测指标，让不同方案站在同一条起跑线上。

- 模型结构、参数量、输入长度和 batch size 保持一致。
- 明确比较目标，例如 peak memory、throughput、latency 和通信开销。
- 如果需要，也可以先写清楚约束条件，比如显存上限或最小吞吐要求。

### Step 2: 运行策略对比

比较不同并行方案的显存、吞吐和延迟，把差异拆成可观察的指标。

- 分别运行 ZeRO、Pipeline 和 Tensor Parallelism 的 baseline。
- 记录每种方案的峰值显存、吞吐、延迟和通信等待时间。
- 关注它们在同一 workload 下的收益与代价，而不是只看单项速度。

### Step 3: 输出选型结论

整理出适用场景和推荐策略，把实验结果转成决策建议。

- 说明每种并行策略更适合哪类模型和资源条件。
- 给出“什么时候选它、什么时候别选它”的结论。
- 如果有取舍，把显存、吞吐和通信之间的平衡写清楚。

### Step 4: 复盘与沉淀

把并行 benchmark 的结果沉淀成后续可复用的评测记录。

- 回看这次 benchmark 的关键参数和实验设置。
- 把不同策略的结果整理成对比表，方便后续复查。
- 如果后面要扩展新的并行方案，就沿用同一套评测框架。
- 一张并行策略对比表，至少包含 peak memory、throughput、latency 和通信开销。
- 一组 baseline 和优化后的指标对比。
- 一段结论，说明每种策略适合什么模型和资源条件。
- 一份可复用的评测记录，方便后续扩展新的并行方案。

### 代码


```python
# ==========================================
# TODO: 完成分布式并行基准模板的两个函数
# 1. 统计 benchmark_fn(fn, warmup=2, iters=5)
# 2. 汇总 baseline 和 tuned 的 memory / throughput / latency 差值
# ==========================================
def benchmark_fn(fn, warmup=2, iters=5):
    # TODO 1: 先做 warmup，再返回平均耗时
    pass

def summarize_parallel_result(base_metrics, tuned_metrics):
    # TODO 2: 汇总并行策略差异
    pass

raise NotImplementedError("请先完成 TODO 代码！")

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 统计平均 benchmark 耗时
import time

def benchmark_fn(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters

# TODO 2: 汇总并行策略差异
def summarize_parallel_result(base_metrics, tuned_metrics):
    return {
        'memory_delta_mb': round(base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb'], 2),
        'throughput_delta': round(tuned_metrics['throughput'] - base_metrics['throughput'], 2),
        'latency_delta_ms': round(base_metrics['latency_ms'] - tuned_metrics['latency_ms'], 2),
    }

baseline = {'peak_mem_mb': 12000.0, 'throughput': 1.0, 'latency_ms': 100.0}
parallel = {'peak_mem_mb': 9000.0, 'throughput': 1.5, 'latency_ms': 80.0}
print(summarize_parallel_result(baseline, parallel))

```

### 解析

- `benchmark_fn` 把不同并行策略的运行时间标准化成平均值。
- `summarize_parallel_result` 负责比较显存、吞吐和延迟的差值。
- 这些差值帮助把并行策略选择转成具体决策。

### 测试


```python
def test_parallel_project_template():
    counter = {'n': 0}

    def fn():
        counter['n'] += 1

    avg = benchmark_fn(fn, warmup=0, iters=2)
    assert counter['n'] == 2
    assert avg >= 0.0

    summary = summarize_parallel_result(
        {'peak_mem_mb': 10.0, 'throughput': 1.0, 'latency_ms': 5.0},
        {'peak_mem_mb': 8.0, 'throughput': 1.2, 'latency_ms': 4.0},
    )
    assert summary['memory_delta_mb'] == 2.0
    print("✅ 分布式并行基准项目模板代码通过基础校验。")


test_parallel_project_template()

```
