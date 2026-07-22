# 32. Training Performance Analysis | 训练性能分析

**难度：** Hard | **环境：** CPU-first | **标签：** `训练`, `profiling`, `显存` | **目标人群：** 训练工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/32_Training_Performance_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*



**关键词：** `training`, `profiling`, `memory`, `step time`

这个项目把训练链路里的性能问题拆开：数据准备、前向反向和显存压力，判断到底哪个环节拖慢了系统。它接住 `2.3` 的训练闭环、`2.5` 的显存优化，以及 `Part 1` 的 profiling 入口。

### Step 1: 定义问题与固定 baseline
选择一个 baseline，明确想优化的对象、指标和约束。

- 先固定模型、数据、batch size、seq len 和评测方式。
- 只看一组核心指标，例如 step time / peak memory / throughput / loss。
- 如果需要，再补一条精度或收敛约束，避免只追求更快。

### Step 2: 测量与定位

记录 profiling 结果，分清数据准备、前向反向和显存瓶颈。

- 先跑一轮 baseline，再看时间分布、显存曲线和热点算子。
- 把问题拆成数据等待、前向 / 反向算子和峰值显存三类。
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
import time
import torch


def measure_train_step(train_step_fn, warmup=2, iters=8):
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        train_step_fn()
    elapsed = (time.perf_counter() - start) / iters

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }


def summarize_training_result(base_metrics, tuned_metrics):
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }


baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0}
tuned = {'step_time_ms': 98.0, 'peak_mem_mb': 6144.0}
print(summarize_training_result(baseline, tuned))

```


```python
# ==========================================
# TODO: 完成训练性能统计的两个函数
# 1. 统计 measure_train_step(train_step_fn, warmup=2, iters=8)
# 2. 汇总 baseline 和 tuned 的 step_time / peak_mem 差值
# ==========================================
def measure_train_step(train_step_fn, warmup=2, iters=8):
    # TODO 1: 记录平均 step time 和 peak memory
    pass

def summarize_training_result(base_metrics, tuned_metrics):
    # TODO 2: 比较 baseline 和 tuned 的指标差值
    pass

raise NotImplementedError("请先完成 TODO 代码！")

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 测量训练 step 的平均耗时和峰值显存
def measure_train_step(train_step_fn, warmup=2, iters=8):
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        train_step_fn()
    elapsed = (time.perf_counter() - start) / iters

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 2: 汇总 baseline 和 tuned 的差异
def summarize_training_result(base_metrics, tuned_metrics):
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }

counter = {'n': 0}
def train_step():
    counter['n'] += 1
print(measure_train_step(train_step, warmup=0, iters=2))

```

### 解析

- `measure_train_step` 负责把训练一步的平均耗时和峰值显存抽出来。
- `summarize_training_result` 负责比较 baseline 和 tuned 的差值。
- 这页的核心不是单看快慢，而是看性能和显存的取舍。

### 测试


```python
def test_training_project_template():
    counter = {'n': 0}

    def train_step():
        counter['n'] += 1

    result = measure_train_step(train_step, warmup=0, iters=2)
    assert counter['n'] == 2
    assert 'step_time_ms' in result and 'peak_mem_mb' in result
    assert result['step_time_ms'] >= 0.0
    assert result['peak_mem_mb'] >= 0.0
    print("✅ 训练性能分析项目模板代码通过基础校验。")


test_training_project_template()

```
