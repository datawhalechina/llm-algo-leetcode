# 32. Training Performance Analysis | 训练性能分析

**难度：** Hard | **环境：** CPU-first | **标签：** `训练`, `profiling`, `显存` | **目标人群：** 训练工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/32_Training_Performance_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这个项目把训练链路里的性能问题拆开：数据准备、前向反向和显存压力，判断到底哪个环节拖慢了系统。它应该能接住 `2.3` 的训练闭环、`2.5` 的显存优化，以及 `Part 1` 的 profiling 入口。

**关键词：** `training`, `profiling`, `memory`, `step time`

## 前置阅读

**导语：** 先把训练闭环、显存优化和项目实战看完，再做训练性能分析会更容易定位瓶颈。
- [Part 2: 13 End-to-End Fine-Tuning Experiment](./13_End_to_End_Fine_Tuning_Experiment.md)
- [19. Activation Checkpointing and Activation Offload | 激活检查点与激活卸载](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [30. LoRA Fine-Tuning Project | LoRA 微调项目](./30_LoRA_Fine_Tuning_Project.md)

## 相关阅读

**导语：** 如果想继续往更底层的性能分析延伸，可以回看 Part 1 的 profiling 章节。
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)


## 项目目标

这个项目的目标是把训练链路里的性能问题拆开：到底是数据准备慢、前向反向慢，还是显存压力把系统拖慢。它应该能接住 `2.3` 的训练闭环、`2.5` 的显存优化，以及 `Part 1` 的 profiling 入口。

- 观察训练 step 的耗时构成。
- 记录峰值显存、梯度累积和 checkpointing 对训练开销的影响。
- 输出一个“改前 / 改后”的训练性能对照结论。

## 实验对象

建议沿用一个尽量稳定的小训练任务，例如微型因果语言模型、分类模型或前面章节已经跑通的 SFT 样本。关键是保持输入、批大小和优化器设置尽量固定，避免比较对象不一致。

1. **训练输入**：固定一批样本，尽量复用同一数据切片。
2. **训练配置**：固定 optimizer、lr、batch size 和 accumulation 策略。
3. **对照变量**：只改一个变量，例如是否开启 checkpointing、是否使用 offload、是否改变 accumulation。

## 实现步骤

1. **建立基线**：先跑一个最简单的训练循环，记录 step time 和 peak memory。
2. **引入变量**：逐个切换 gradient accumulation、activation checkpointing、offload 或更小 batch。
3. **记录变化**：对比每种设置下的吞吐、显存和 loss 下降速度。
4. **找出瓶颈**：判断是数据、计算、通信，还是显存回收机制在限制训练速度。
5. **回到正文**：把结果和 `13` 的 profiling 方法一起解释清楚。


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

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


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

```

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
