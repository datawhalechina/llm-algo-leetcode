# 20. Profiling and Memory Ledger | 性能剖析与显存账本

**难度：** Medium | **环境：** CPU-first | **标签：** `profiling`, `显存`, `性能分析` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页把性能指标、热点和显存账本放到同一条排查路径中：先看 latency、throughput 和热点，再拆参数、梯度、优化器状态和激活。本页的阈值和账本是 CPU-first 的教学示例；真实 GPU 峰值、吞吐、OOM 和 profiler 归因需要在 Part 2 的 73 / 74 / 76 中验证。

> **证据边界：** profiler 热点是下一步排查的线索，不等于已经确认根因；显存账本是对象级估算，不等同于 CUDA allocator 的完整占用。

**关键词：** `profiler`, `latency`, `memory`

## 前置阅读
**导语：** 先看 0E 组页，把性能剖析和显存账本的边界对齐，再进入这一页会更顺。
- [19. Debugging and Anomaly Localization | 调试与异常定位](./19_Debugging_and_Anomaly_Localization.md)
- [0E 组页](./0E.md)
## 相关阅读
**导语：** 如果想把性能剖析和显存账本接上 Part 1，可以顺着看下面这一页。
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)

## Q1：性能问题先看哪几个指标？

先分清 latency、throughput、GPU 利用率和 CPU 等待时间，再提出问题到底可能是算子慢、数据慢还是调度慢的假设；阈值只是本例的教学参数，不是通用标准。


```python
def summarize_step_metrics(step_ms, tokens, gpu_busy_pct, cpu_wait_ms):
    tok_per_s = tokens / (step_ms / 1000.0)
    return {
        'latency_ms': step_ms,
        'throughput_tok_s': round(tok_per_s, 1),
        'gpu_busy_pct': gpu_busy_pct,
        'cpu_wait_ms': cpu_wait_ms,
    }


summary = summarize_step_metrics(step_ms=180, tokens=360, gpu_busy_pct=42, cpu_wait_ms=55)
print('summary:', summary)
print('bottleneck:', 'data' if summary['cpu_wait_ms'] > 40 and summary['gpu_busy_pct'] < 60 else 'compute')

# 输出示例: summary 中 latency_ms=180, throughput_tok_s=2000.0, bottleneck=data

```

## Q2：怎么用 profiler 找热点？

先看前向、反向、优化器更新和数据加载各占多少时间，再决定下一步先验证数据管线、算子还是训练骨架；热点本身不等于根因。


```python
def top_hotspots(events, topk=2):
    ordered = sorted(events, key=lambda x: x[1], reverse=True)
    return ordered[:topk]


events = [('forward', 38), ('backward', 74), ('optimizer', 18), ('dataloader', 52)]
hotspots = top_hotspots(events)
print('hotspots:', hotspots)
print('focus:', hotspots[0][0])

# 输出示例: hotspots -> [('backward', 74), ('dataloader', 52)], focus -> backward

```

## Q3：显存账本怎么拆？

先分清参数、梯度、优化器状态、激活和临时缓冲，再提出是缩 batch、做 accumulation，还是上 checkpoint 的候选方案；最终选择要交给固定 workload 的 benchmark。


```python
def memory_ledger(param_count, param_bytes, grad_bytes, adam_state_bytes, activation_bytes):
    total = param_count * (param_bytes + grad_bytes + adam_state_bytes) + activation_bytes
    return {
        'parameters_mb': round(param_count * param_bytes / 1024**2, 2),
        'grads_mb': round(param_count * grad_bytes / 1024**2, 2),
        'adam_state_mb': round(param_count * adam_state_bytes / 1024**2, 2),
        'activation_mb': round(activation_bytes / 1024**2, 2),
        'total_mb': round(total / 1024**2, 2),
    }


ledger = memory_ledger(param_count=1_000_000, param_bytes=2, grad_bytes=2, adam_state_bytes=8, activation_bytes=24 * 1024**2)
print('ledger:', ledger)
print('dominant:', 'activation' if ledger['activation_mb'] > ledger['adam_state_mb'] else 'state')

# 输出示例: ledger 中 activation_mb 往往是主要开销

```

## Q4：显存压力出现时，先用 batch、accumulation 还是 checkpoint？

先判断是不是显存真的放不下；如果是，可先评估缩 batch，再看 accumulation 是否能保住有效 batch，最后比较 checkpoint、offload 或混合精度。这个顺序是排查起点，不是所有 workload 的固定答案。


```python
def choose_memory_action(can_fit, hotspot, communication_heavy):
    """返回下一步排查方向，不直接替代真实 benchmark 决策。"""
    if not can_fit:
        return 'reduce_batch_or_compare_checkpoint_offload'
    if hotspot == "dataloader":
        return 'optimize_input_pipeline'
    if communication_heavy:
        return 'inspect_communication_frequency_and_overlap'
    return 'profile_more'


print('case1:', choose_memory_action(False, 'backward', False))
print('case2:', choose_memory_action(True, 'dataloader', False))
print('case3:', choose_memory_action(True, 'backward', True))

# 输出示例: 这些结果只是下一步排查方向，不能直接当成最终优化结论

```

## Q5：latency 高但 throughput 低，通常先怀疑什么？

先判断是单步慢、整体产出低，还是两者同时发生；把症状拆开后，再用 profiler 或 benchmark 验证数据、算子、同步还是训练骨架是否是真正瓶颈。


```python
def performance_signal(latency_ms, tokens, gpu_busy_pct):
    throughput = round(tokens / (latency_ms / 1000.0), 1)
    signal = {
        'latency_ms': latency_ms,
        'throughput_tok_s': throughput,
        'gpu_busy_pct': gpu_busy_pct,
    }
    if latency_ms > 200 and throughput < 2000:
        signal['zone'] = 'slow_and_low_throughput'
        signal['next'] = 'inspect_batch_and_dataloader'
    elif gpu_busy_pct < 50:
        signal['zone'] = 'likely_input_or_sync_bound'
        signal['next'] = 'inspect_input_pipeline_or_sync'
    else:
        signal['zone'] = 'compute_or_kernel_bound'
        signal['next'] = 'inspect_forward_or_kernel'
    return signal


signal = performance_signal(latency_ms=220, tokens=360, gpu_busy_pct=42)
print('signal:', signal)
# 输出示例: zone 和 next 会一起指出下一步该查哪一层

```

## Q6：profiler 看到热点后，先改数据、算子还是训练骨架？

先看热点落在 data / forward / backward / optimizer 哪一层，再决定下一步验证数据管线、算子实现、通信同步还是训练骨架；不要只凭单个热点直接改代码。


```python
def choose_first_fix(hotspot):
    mapping = {
        'dataloader': 'data',
        'forward': 'operator',
        'backward': 'training_step',
        'optimizer': 'optimizer_loop',
    }
    if hotspot in mapping:
        return {'hotspot': hotspot, 'first_fix': mapping[hotspot]}
    return {'hotspot': hotspot, 'first_fix': 'profile_more'}


for hotspot in ['dataloader', 'forward', 'backward', 'optimizer']:
    print(choose_first_fix(hotspot))
# 输出示例: first_fix 会直接映射回 data / operator / training_step / optimizer_loop

```
