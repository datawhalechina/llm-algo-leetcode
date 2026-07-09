# 31. Inference Performance Comparison | 推理性能对比实验

**难度：** Hard | **环境：** CPU-first | **标签：** `推理`, `benchmark`, `profiling` | **目标人群：** 推理工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/31_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这个项目围绕同一个模型、同一批输入，把不同推理策略的延迟、吞吐和显存占用拉到一张表里比较。它的作用是把 `2.6`、`2.7` 和 `2.8` 的内容落到一个可复现的工程判断上。

**关键词：** `benchmark`, `latency`, `throughput`, `memory`

## 前置阅读

**导语：** 先把核心推理优化、量化和分布式并行看完，再做推理性能对比会更有意义。
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [25. Quantization W8A16 | W8A16 量化](./25_Quantization_W8A16.md)

## 相关阅读

**导语：** 推理性能对比之后，建议继续看训练性能分析。
- [32. Training Performance Analysis | 训练性能分析](./32_Training_Performance_Analysis.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)


## 项目目标

这个项目不是单纯跑一个推理 demo，而是围绕同一个模型、同一批输入，把不同推理策略的延迟、吞吐和显存占用拉到一张表里比较。它的作用是把 `2.6`、`2.7` 和 `2.8` 的内容落到一个可复现的工程判断上。

- 对比预填充阶段和解码阶段的耗时差异。
- 比较不同 batch size、不同上下文长度、不同 cache 策略下的吞吐变化。
- 把推理上的性能结论和 `Part 1` 的 profiling 入口接起来。

## 实验对象

建议固定一个小型因果语言模型或已经训练好的微型检查点，避免模型本身变化掩盖策略差异。输入也尽量固定成一组短、中、长三档 prompt，以便比较：

1. **短输入**：看启动开销和 prefill 的基础代价。
2. **中等输入**：看缓存是否开始发挥作用。
3. **长输入**：看 KV cache、分页和推理调度是否成为主要瓶颈。

## 实现步骤

1. **建立基线**：先跑一个最简单的 greedy decoding 或 teacher-forcing 风格推理，记录基础 latency 和 throughput。
2. **逐项替换**：在同一模型上比较 `FlashAttention / PagedAttention / 量化 / batch` 等不同策略对性能的影响。
3. **拆分阶段**：把 prefill 和 decode 分开统计，避免只看总耗时而丢掉阶段差异。
4. **记录资源**：至少记录 `token/s`、`latency`、`peak memory` 和 `decode stability`。
5. **做结论表**：输出一张“策略 -> 代价 -> 收益”的对照表，给后面的训练性能分析做参照。


```python
import time

```


```python
import time


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

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
import time


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

```

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
