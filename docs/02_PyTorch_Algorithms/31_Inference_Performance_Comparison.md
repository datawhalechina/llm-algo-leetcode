# 31. Inference Performance Comparison | 推理性能对比实验

**难度：** Hard | **环境：** CPU-first | **标签：** `推理`, `benchmark`, `profiling` | **目标人群：** 推理工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/31_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*



**关键词：** `benchmark`, `latency`, `throughput`, `memory`

这个项目围绕同一个模型、同一批输入，把不同推理策略的延迟、吞吐和显存占用拉到一张表里比较，最后形成可复现的推理选型结论。它承接 `2.6`、`2.7` 和 `2.8` 的内容，也借用 `Part 1` 的 profiling 方法来判断瓶颈和取舍。

### Step 1: 定义问题与固定 baseline
选择一个 baseline，明确想优化的对象、指标和约束。

- 先固定模型、数据、batch size、seq len 和评测方式。
- 只看一组核心指标，例如 latency / throughput / peak memory / step time。
- 如果是训练任务，再补一条精度或 loss 约束，避免只追求更快。

### Step 2: 测量与定位

记录 profiling 结果，分清数据、算子、通信和显存瓶颈。

- 先跑一轮 baseline，再看时间分布、显存曲线和热点算子。
- 把问题拆成数据等待、前向 / 反向算子、通信同步和峰值显存四类。
- 这一步的目标是把“慢”具体化，而不是先急着改代码。

### Step 3: 修改与复测

针对瓶颈做最小修改，再次测量验证收益。

- 一次只改一个方向，避免优化结果不可归因。
- 改完后重新测同样的指标，比较改前 / 改后差异。
- 如果某个改动只是在一项指标上变好，却让另一项变差，要把取舍写清楚。

### Step 4: 复盘与沉淀

输出改动前后对比表、profiling 截图和最终判断，把这次经验收成可复用的优化记录。

- 记录本次瓶颈来自哪里，以及下次优先看哪一层。
- 把这次优化的取舍和结论写成可复用的排障路径。
- 如果还有后续优化空间，就把下一轮优先级列出来。


```python
import time

```


```python

def benchmark_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    return total / iters


def summarize_inference_result(prefill_ms, decode_ms, peak_mem_mb):
    total = prefill_ms + decode_ms
    decode_share = decode_ms / total if total else 0.0
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total, 2),
        'decode_share': round(decode_share, 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }


example = summarize_inference_result(42.5, 18.0, 5120.0)
print(example)

```


```python
# ==========================================
# TODO: 完成推理性能统计的两个函数
# 1. 计算 benchmark_fn(fn, warmup=3, iters=10)
# 2. 汇总 prefill / decode / total / decode_share / peak_mem_mb
# ==========================================
def benchmark_fn(fn, warmup=3, iters=10):
    # TODO 1: 先做 warmup，再返回平均耗时
    pass

def summarize_inference_result(prefill_ms, decode_ms, peak_mem_mb):
    # TODO 2: 汇总 prefill / decode / total / decode_share / peak_mem_mb
    pass

raise NotImplementedError("请先完成 TODO 代码！")

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 统计平均 benchmark 耗时
def benchmark_fn(fn, warmup=3, iters=10):
    """Measure average runtime after warmup."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    return total / iters

# TODO 2: 汇总推理指标
def summarize_inference_result(prefill_ms, decode_ms, peak_mem_mb):
    total = prefill_ms + decode_ms
    decode_share = decode_ms / total if total else 0.0
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total, 2),
        'decode_share': round(decode_share, 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

for prefill_ms, decode_ms, peak_mem_mb in [(42.5, 18.0, 5120.0)]:
    print(summarize_inference_result(prefill_ms, decode_ms, peak_mem_mb))

```

### 解析

- `benchmark_fn` 负责在 warmup 之后测平均耗时，便于稳定比较。
- `summarize_inference_result` 负责把 prefill / decode / total / decode_share / peak_mem_mb 统一收口。
- `decode_share` 可以帮助判断 decode 是否成为主要瓶颈。

### 测试


```python
def test_inference_project_template():
    summary = summarize_inference_result(10.0, 5.0, 256.0)
    assert summary['total_ms'] == 15.0
    assert summary['decode_share'] == 0.333
    assert summary['peak_mem_mb'] == 256.0

    counter = {'n': 0}
    def fn():
        counter['n'] += 1

    avg = benchmark_fn(fn, warmup=0, iters=3)
    assert counter['n'] == 3
    assert avg >= 0.0
    print("✅ 推理性能对比项目模板代码通过基础校验。")


test_inference_project_template()

```
