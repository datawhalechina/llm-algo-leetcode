# 35. Quantized Inference and Deployment | 量化推理与部署

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战` | **目标人群：** 工程实践

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/35_Quantized_Inference_and_Deployment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


**关键词：** `quantization, inference, deployment`

这个项目把 W8A16、4-bit 量化和 QLoRA 放进一个可复现的推理与部署流程里，比较精度、速度和显存之间的权衡。它承接 `2.7` 的高级推理与压缩优化，也能直接接到 `Part 1` 的 profiling 方法。

## 前置阅读

**导语：** 先把最小前置看完，再进入项目会更容易把问题拆开。
- [2.7. Advanced Inference and Compression | 高级推理与压缩优化](./2_7.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)

## 相关阅读

**导语：** 如果想继续往更底层的工程背景延伸，可以看这些页面。
- [25. Quantization W8A16 | W8A16 量化](./25_Quantization_W8A16.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](./26_QLoRA_and_4bit_Quantization.md)
- [31. Inference Performance Comparison | 推理性能对比实验](./31_Inference_Performance_Comparison.md)

### Step 1: 明确量化对象
先决定要量化权重、激活还是两者都量化，并写清目标场景。

- 明确是做离线推理、在线服务还是模型压缩实验。
- 先选定量化粒度，例如 W8A16、INT8、4-bit 或更低比特。
- 记录约束条件，比如最小精度、最大显存占用或最低吞吐。

### Step 2: 跑通最小推理链路

在同一批输入上比较量化前后结果，把链路先跑通再谈优化。

- 固定输入、prompt 和解码设置，保证比较公平。
- 记录量化前后的输出差异、latency、throughput 和 VRAM 占用。
- 如果结果有偏差，先确认是数值误差、缓存还是实现细节导致的。

### Step 3: 总结部署结论

给出精度、速度和显存之间的权衡建议，把实验结果收成部署方案。

- 说明不同量化方案更适合什么硬件和使用场景。
- 给出“什么时候用、什么时候别用”的结论。
- 如果某种量化速度更快但精度损失更大，要把取舍写清楚。

### Step 4: 复盘与扩展

把量化实验的结果沉淀成可复用的部署记录。

- 回看这次实验的量化配置、输入设置和评测指标。
- 把量化前后结果整理成一页对比表，方便后续复查。
- 如果后面要换模型或换硬件，就沿用同一套对比方法。
- 一张量化前后对比表，至少包含 latency、throughput、VRAM 和精度。
- 一组 baseline 和优化后的指标对比。
- 一段结论，说明不同量化方案更适合什么硬件和场景。
- 一份可复用的部署记录，方便后续复查。

### 代码


```python
# ==========================================
# TODO: 完成量化推理与部署模板的两个函数
# 1. 统计 benchmark_fn(fn, warmup=2, iters=5)
# 2. 汇总 baseline 和 tuned 的 latency / throughput / vram 差值
# ==========================================
def benchmark_fn(fn, warmup=2, iters=5):
    # TODO 1: 先做 warmup，再返回平均耗时
    pass

def summarize_quantized_result(base_metrics, tuned_metrics):
    # TODO 2: 汇总量化前后的差异
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

# TODO 2: 汇总量化前后的差异
def summarize_quantized_result(base_metrics, tuned_metrics):
    return {
        'latency_delta_ms': round(base_metrics['latency_ms'] - tuned_metrics['latency_ms'], 2),
        'throughput_delta': round(tuned_metrics['throughput'] - base_metrics['throughput'], 2),
        'vram_delta_mb': round(base_metrics['vram_mb'] - tuned_metrics['vram_mb'], 2),
    }

baseline = {'latency_ms': 100.0, 'throughput': 1.0, 'vram_mb': 12000.0}
quantized = {'latency_ms': 72.0, 'throughput': 1.3, 'vram_mb': 7000.0}
print(summarize_quantized_result(baseline, quantized))

```

### 解析

- `benchmark_fn` 把量化前后的运行时间标准化成平均值。
- `summarize_quantized_result` 负责比较 latency / throughput / VRAM 的差值。
- 这些指标帮助把量化方案的部署取舍说清楚。

### 测试


```python
def test_quantized_project_template():
    counter = {'n': 0}

    def fn():
        counter['n'] += 1

    avg = benchmark_fn(fn, warmup=0, iters=2)
    assert counter['n'] == 2
    assert avg >= 0.0

    summary = summarize_quantized_result(
        {'latency_ms': 10.0, 'throughput': 1.0, 'vram_mb': 20.0},
        {'latency_ms': 8.0, 'throughput': 1.2, 'vram_mb': 12.0},
    )
    assert summary['vram_delta_mb'] == 8.0
    print("✅ 量化推理与部署项目模板代码通过基础校验。")


test_quantized_project_template()

```
