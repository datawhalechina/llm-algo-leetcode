# 66. Inference Performance Comparison | 推理性能对比实验

**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `基准对比`, `性能对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节要求你比较一个推理 baseline 与候选优化方案在固定 workload 下的表现。先统一 batch、输入长度、生成长度和 warm-up 方式，再分别测量 TTFT、端到端延迟、吞吐和峰值显存。最终输出一张对比表，并说明该方案适合低延迟、高吞吐还是显存受限场景。
**层级定位：** 本项目主落在 L4，关注单个模型实例如何执行请求；会调用 L2 的算子/后端能力和 L3 的运行时，但不负责 L5 的多模型发布、集群扩缩容或流量治理。

> 环境提示：如果你在 Colab、ModelScope 或本地 GPU 上运行真实 backend，请先阅读 [Part02 Intro 的环境说明](./intro.md#environment-notes-环境说明)。默认只需要当前 Notebook runtime；本地只有在 vLLM 与 PyTorch 依赖冲突时才需要额外的 vLLM 环境。

**关键词：** `benchmark`, `TTFT`, `TPOT`, `throughput`, `KV cache`

---
## 前置阅读

**导语：** 先把解码、KV cache 和推理后端的最小口径理顺，再做推理性能对比；本节不重复讲每个优化机制，而是把它们放到同一个 benchmark 口径里比较。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

## 相关阅读

**导语：** 做完基础推理对比后，最自然的下一步是继续拆具体优化收益，或把结论推进到量化部署。
- [68. Speculative Decoding Benchmark | 推测解码基准](./68_Speculative_Decoding_Benchmark.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)

---
### Step 1: 定义推理对比项目目标
先回答一个问题：在同一模型、同一输入集和同一硬件环境下，哪种推理策略更划算？

- 固定模型、backend、batch size、prompt tokens、generated tokens、dtype、cache policy 和评测轮数。
- Baseline 建议从 `PyTorch eager + batch=1 + 固定 prompt/output length + warm-up + 多轮测量` 开始。
- 明确 candidate 只改一个变量，例如 FlashAttention、batch size、KV cache 策略、量化精度或推理后端。
- 统一核心指标：TTFT、TPOT、generated tokens/s、total latency、peak memory。
- 这节的目标不是证明某个方案“能跑”，而是在相同约束下输出可解释的推理选型结论。

### Step 2: 先确认 workload 和 baseline 口径合法

推理对比必须先确认 workload 和 baseline 可复现，不能直接把不同 prompt、不同 batch 或不同 backend 的数字放在一起比较。

- 先固定模型、backend、batch size、prompt tokens、generated tokens、dtype、cache policy 和 warm-up / 多轮测量方式。
- Baseline 建议从 `PyTorch eager + batch=1 + 固定 prompt/output length` 开始，保证后续 candidate 的改动边界清晰。
- TTFT、TPOT、throughput、total latency 和 peak memory 必须来自同一套 workload，避免把不同实验口径拼成一张表。
- 如果 baseline 自身波动很大，后面的 candidate 结果就没有解释空间。

### Step 3: 用统一口径比较收益与成本

推理项目必须用统一口径同时看 latency、throughput 和 memory，不能只挑单一指标下结论。

| 瓶颈类型 | 典型信号 | 候选方向 |
| --- | --- | --- |
| prefill-bound | prefill 占比高，长 prompt 变慢明显 | FlashAttention、chunked prefill、batching |
| decode-bound | TPOT 高，decode 占比高 | speculative decoding、multi-token decoding、decode scheduling |
| memory-bound | peak memory 接近预算，batch 上不去 | KV cache quantization、PagedAttention、GQA/MQA |
| balanced | 各项都不突出 | 保持 baseline 或做小步 profiling |

### Step 4: 输出推理选型结论

推理选型最终不是输出“哪个 benchmark 更好看”，而是输出哪种方案值得在当前 workload 下继续保留、微调或切换。

- 输出 baseline vs candidate 对比表，至少包含 TTFT、TPOT、throughput、total latency、peak memory 和瓶颈判断。
- 如果 candidate 只提升吞吐但明显拉高 TTFT，要说明适合离线批处理还是在线交互。
- 如果 candidate 显存更省但 TPOT 变差，要说明是否为了更大 batch 或更长上下文让路。
- 最终决策统一使用 `accept / tune / reject`：候选方案值得采用、还需继续调优、或当前不值得切换。
- 报告结论必须回扣 Step 1 的 workload，不能泛化成“某方案永远更好”。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整推理性能对比项目流程。下面的代码实现六块最小能力：配置 workload、汇总 prefill/decode、计算指标、诊断瓶颈、比较候选方案和输出决策。

完成 Step 5 后，如果当前环境具备 GPU 和 vLLM，可以继续运行文末的真实 backend Notebook 实验单元；没有这些条件时，停留在 CPU-first 主线即可。
#### 图解：推理项目如何从 workload 走到选型结论

`66` 不重复讲所有推理优化机制，而是把它们放进同一套 benchmark 口径里比较。

```text
workload config
      │
      ▼
baseline run ──► prefill/decode metrics ──► bottleneck diagnosis
      │                                               │
      ▼                                               ▼
candidate run ─► candidate comparison ───────────► accept / tune / reject
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| Workload | backend、batch、prompt tokens、generated tokens、dtype、cache policy | 保证可复现 |
| 指标 | TTFT、TPOT、throughput、total latency、peak memory | 保证同口径比较 |
| 诊断 | prefill-bound、decode-bound、memory-bound、balanced | 解释为什么优化有效或无效 |
| 对比 | latency / throughput / memory delta | 判断 candidate 是否值得保留 |
| 决策 | accept / tune / reject | 输出推理选型结论 |

### Colab / ModelScope Notebook 工作流

两类云端 Notebook 都遵循同一条链路：先探测 GPU 和 CUDA，再安装 backend，下载模型，启动本地 OpenAI-compatible 服务，最后运行本节 benchmark。云端 GPU 型号可能变化，因此不能直接复制本机的版本和 dtype。

**Colab**

1. 选择 GPU runtime，并确认 `nvidia-smi`、`torch.cuda.is_available()` 和 GPU 型号。
2. 在同一个 Notebook kernel 中安装 vLLM；此时 `VLLM_ENV = None`、`VLLM_COMMAND = None`。
3. 通常使用 `MODEL_SOURCE = 'huggingface'`；网络受限时先用 ModelScope 下载到本地路径。
4. T4 优先 `float16`；L4/A100/H100 通常可用 `bfloat16`；第一次建议 `ENFORCE_EAGER = True`。
5. 设置 `RUN_REAL_BACKEND = True`，运行 Step 6；服务端口由 helper 自动选择，不需要公开 Colab 端口。
6. 将 `benchmarks/results/*.json` 复制到 Google Drive 或下载到本地，因为 Colab runtime 释放后文件会消失。

**ModelScope Notebook**

1. 先探测平台提供的 GPU、驱动、CUDA 和 Python 环境。
2. 安装 `modelscope` 与匹配的 vLLM；如果 vLLM 与 Notebook kernel 不在同一环境，设置 `VLLM_ENV` 为实际环境名。
3. 设置 `MODEL_SOURCE = 'modelscope'`，helper 会调用 `snapshot_download`，然后把本地模型目录传给 vLLM。
4. 先用小模型和 `CONCURRENCY = 1` 完成 smoke test，再测试并发 4。
5. 将 JSON 结果保存到持久化工作目录，并记录 GPU 型号、驱动、PyTorch、vLLM 和启动参数。

两种平台都不应直接把结果与本机 RTX 5070 Ti 的结果横向比较；先比较同一平台内的 baseline / candidate，再把硬件和软件栈作为实验条件写入报告。

```python
import time

```


```python
# 补全推理性能对比的六个关键函数
# 目标：完成 workload -> metrics -> bottleneck -> comparison -> decision 的最小项目链路

def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
    """汇总推理 workload 配置，形成统一比较口径。"""
    # ==========================================
    # TODO 1: 汇总推理 workload 配置
    # 提示：total_tokens = prompt_tokens + generated_tokens
    # ==========================================
    # total_tokens = ???
    return {
        'model_name': model_name,
        'backend': backend,
        'batch_size': batch_size,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens,
        'total_tokens': total_tokens,
        'dtype': dtype,
        'cache_policy': cache_policy,
    }

def summarize_prefill_decode(prefill_ms, decode_ms, generated_tokens):
    """汇总 prefill / decode 延迟，形成最小延迟摘要。"""
    # ==========================================
    # TODO 2: 汇总 prefill / decode 延迟
    # 提示：TTFT 近似等于 prefill_ms；TPOT = decode_ms / generated_tokens。
    # ==========================================
    # total_ms = ???
    # ttft_ms = ???
    # tpot_ms = ???
    # prefill_share = ???
    # decode_share = ???
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total_ms, 2),
        'ttft_ms': round(ttft_ms, 2),
        'tpot_ms': round(tpot_ms, 4),
        'prefill_share': round(prefill_share, 3),
        'decode_share': round(decode_share, 3),
    }

def compute_inference_metrics(config, latency_summary, peak_mem_mb):
    """把 workload 和延迟摘要收束成统一推理指标。"""
    # ==========================================
    # TODO 3: 计算推理项目核心指标
    # 提示：throughput 表示整个 batch 每秒生成 token 数。
    # ==========================================
    # output_tokens = ???
    # throughput_tok_s = ???
    return {
        'backend': config['backend'],
        'batch_size': config['batch_size'],
        'prompt_tokens': config['prompt_tokens'],
        'generated_tokens': config['generated_tokens'],
        'ttft_ms': latency_summary['ttft_ms'],
        'tpot_ms': latency_summary['tpot_ms'],
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': latency_summary['total_ms'],
        'prefill_share': latency_summary['prefill_share'],
        'decode_share': latency_summary['decode_share'],
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def diagnose_inference_bottleneck(metrics, memory_budget_mb=None):
    """根据显存预算与 prefill/decode 占比诊断推理瓶颈。"""
    # ==========================================
    # TODO 4: 诊断推理瓶颈
    # 规则：显存接近预算优先判 memory-bound；否则按 prefill/decode 占比判断。
    # ==========================================
    # memory_pressure = ???
    # prefill_heavy = ???
    # decode_heavy = ???
    # if ???:
    #     bottleneck = ???
    #     reason = ???
    # elif ???:
    #     bottleneck = ???
    #     reason = ???
    # elif ???:
    #     bottleneck = ???
    #     reason = ???
    # else:
    #     bottleneck = ???
    #     reason = ???
    return {'bottleneck': bottleneck, 'reason': reason}

def compare_inference_candidates(baseline_metrics, candidate_metrics):
    """统一比较 baseline 与 candidate 的推理收益和代价。"""
    # ==========================================
    # TODO 5: 比较 baseline 和 candidate
    # 提示：latency / TTFT / TPOT / memory 的 delta 用 baseline - candidate；throughput gain 用比例增益。
    # ==========================================
    # total_latency_delta_ms = ???
    # ttft_delta_ms = ???
    # tpot_delta_ms = ???
    # peak_mem_delta_mb = ???
    # throughput_gain = ???
    return {
        'total_latency_delta_ms': round(total_latency_delta_ms, 2),
        'ttft_delta_ms': round(ttft_delta_ms, 2),
        'tpot_delta_ms': round(tpot_delta_ms, 4),
        'peak_mem_delta_mb': round(peak_mem_delta_mb, 2),
        'throughput_gain': round(throughput_gain, 4),
    }

def recommend_inference_decision(comparison, candidate_bottleneck, min_throughput_gain=0.1, max_ttft_regression_ms=20.0):
    """根据吞吐、TTFT 和瓶颈类型输出推理选型建议。"""
    # ==========================================
    # TODO 6: 输出推理选型建议
    # 规则：吞吐明显提升且 TTFT 没明显退化则 accept；有收益但仍有瓶颈则 tune；否则 reject。
    # ==========================================
    # throughput_good = ???
    # ttft_ok = ???
    # still_tunable = ???
    # if ???:
    #     decision = ???
    #     reason = ???
    # elif ???:
    #     decision = ???
    #     reason = ???
    # else:
    #     decision = ???
    #     reason = ???
    return {'decision': decision, 'reason': reason}

```


```python
# 测试你的实现
def test_inference_project_template():
    try:
        config = build_inference_config(
            model_name='tiny-llama',
            backend='pytorch-eager',
            batch_size=2,
            prompt_tokens=128,
            generated_tokens=32,
            dtype='fp16',
            cache_policy='static-kv-cache',
        )
        assert config['total_tokens'] == 160, "total_tokens 计算不正确！"
        assert config['batch_size'] == 2, "batch_size 应保留原始配置！"

        latency = summarize_prefill_decode(prefill_ms=80.0, decode_ms=160.0, generated_tokens=32)
        assert latency['total_ms'] == 240.0, "total_ms 计算不正确！"
        assert latency['ttft_ms'] == 80.0, "ttft_ms 计算不正确！"
        assert latency['tpot_ms'] == 5.0, "tpot_ms 计算不正确！"
        assert latency['prefill_share'] == 0.333, "prefill_share 计算不正确！"
        assert latency['decode_share'] == 0.667, "decode_share 计算不正确！"

        metrics = compute_inference_metrics(config, latency, peak_mem_mb=4096.0)
        assert metrics['throughput_tok_s'] == 266.67, "throughput_tok_s 计算不正确！"
        assert metrics['peak_mem_mb'] == 4096.0, "peak_mem_mb 记录不正确！"

        memory_bound = diagnose_inference_bottleneck(metrics, memory_budget_mb=4400.0)
        assert memory_bound['bottleneck'] == 'memory-bound', "显存接近预算时应优先判为 memory-bound！"

        decode_bound = diagnose_inference_bottleneck(metrics, memory_budget_mb=8192.0)
        assert decode_bound['bottleneck'] == 'decode-bound', "decode 占比高时应判为 decode-bound！"

        candidate_config = build_inference_config(
            model_name='tiny-llama',
            backend='paged-attention',
            batch_size=2,
            prompt_tokens=128,
            generated_tokens=32,
            dtype='fp16',
            cache_policy='paged-kv-cache',
        )
        candidate_latency = summarize_prefill_decode(prefill_ms=85.0, decode_ms=120.0, generated_tokens=32)
        candidate_metrics = compute_inference_metrics(candidate_config, candidate_latency, peak_mem_mb=3584.0)
        comparison = compare_inference_candidates(metrics, candidate_metrics)

        assert comparison['total_latency_delta_ms'] == 35.0, "total latency delta 计算不正确！"
        assert comparison['ttft_delta_ms'] == -5.0, "TTFT delta 计算不正确！"
        assert comparison['tpot_delta_ms'] == 1.25, "TPOT delta 计算不正确！"
        assert comparison['peak_mem_delta_mb'] == 512.0, "peak memory delta 计算不正确！"
        assert comparison['throughput_gain'] > 0.15, "throughput gain 应体现候选方案收益！"

        decision = recommend_inference_decision(comparison, decode_bound)
        assert decision['decision'] == 'accept', "吞吐提升且 TTFT 未明显退化时应建议 accept！"

        weak_comparison = dict(comparison)
        weak_comparison['throughput_gain'] = 0.02
        weak_comparison['ttft_delta_ms'] = 1.0
        assert recommend_inference_decision(weak_comparison, decode_bound)['decision'] == 'tune', "小幅收益但仍有瓶颈时应建议 tune！"

        bad_comparison = dict(comparison)
        bad_comparison['throughput_gain'] = -0.05
        bad_comparison['ttft_delta_ms'] = -30.0
        assert recommend_inference_decision(bad_comparison, {'bottleneck': 'balanced'})['decision'] == 'reject', "没有收益且 TTFT 退化时应建议 reject！"

        print("✅ 推理性能对比项目模板代码通过基础校验。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了数值错误")
        elif isinstance(e, AssertionError):
            print(f"❌ 测试失败: {e}")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_inference_project_template()

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
# TODO 1: 汇总推理 workload 配置
def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
    total_tokens = prompt_tokens + generated_tokens
    return {
        'model_name': model_name,
        'backend': backend,
        'batch_size': batch_size,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens,
        'total_tokens': total_tokens,
        'dtype': dtype,
        'cache_policy': cache_policy,
    }

# TODO 2: 汇总 prefill / decode 延迟
def summarize_prefill_decode(prefill_ms, decode_ms, generated_tokens):
    total_ms = prefill_ms + decode_ms
    ttft_ms = prefill_ms
    tpot_ms = decode_ms / generated_tokens if generated_tokens else 0.0
    prefill_share = prefill_ms / total_ms if total_ms else 0.0
    decode_share = decode_ms / total_ms if total_ms else 0.0
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total_ms, 2),
        'ttft_ms': round(ttft_ms, 2),
        'tpot_ms': round(tpot_ms, 4),
        'prefill_share': round(prefill_share, 3),
        'decode_share': round(decode_share, 3),
    }

# TODO 3: 计算推理项目核心指标
def compute_inference_metrics(config, latency_summary, peak_mem_mb):
    output_tokens = config['batch_size'] * config['generated_tokens']
    total_seconds = latency_summary['total_ms'] / 1000.0
    throughput_tok_s = output_tokens / total_seconds if total_seconds else 0.0
    return {
        'backend': config['backend'],
        'batch_size': config['batch_size'],
        'prompt_tokens': config['prompt_tokens'],
        'generated_tokens': config['generated_tokens'],
        'ttft_ms': latency_summary['ttft_ms'],
        'tpot_ms': latency_summary['tpot_ms'],
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': latency_summary['total_ms'],
        'prefill_share': latency_summary['prefill_share'],
        'decode_share': latency_summary['decode_share'],
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 4: 诊断推理瓶颈
def diagnose_inference_bottleneck(metrics, memory_budget_mb=None):
    if memory_budget_mb is not None and metrics['peak_mem_mb'] >= 0.9 * memory_budget_mb:
        bottleneck = 'memory-bound'
        reason = 'peak memory 接近预算，优先检查 KV cache、batch size、量化和分页策略。'
    elif metrics['prefill_share'] >= 0.6:
        bottleneck = 'prefill-bound'
        reason = 'prefill 占比高，优先检查 prompt length、FlashAttention、chunked prefill 和 batching。'
    elif metrics['decode_share'] >= 0.6:
        bottleneck = 'decode-bound'
        reason = 'decode 占比高，优先检查 KV cache 读写、decode scheduling、speculative decoding 或 multi-token decoding。'
    else:
        bottleneck = 'balanced'
        reason = 'prefill、decode 和显存压力都不突出，先保持 baseline 或继续做细粒度 profiling。'
    return {'bottleneck': bottleneck, 'reason': reason}

# TODO 5: 比较 baseline 和 candidate
def compare_inference_candidates(baseline_metrics, candidate_metrics):
    total_latency_delta_ms = baseline_metrics['total_ms'] - candidate_metrics['total_ms']
    ttft_delta_ms = baseline_metrics['ttft_ms'] - candidate_metrics['ttft_ms']
    tpot_delta_ms = baseline_metrics['tpot_ms'] - candidate_metrics['tpot_ms']
    peak_mem_delta_mb = baseline_metrics['peak_mem_mb'] - candidate_metrics['peak_mem_mb']
    throughput_gain = (
        candidate_metrics['throughput_tok_s'] / baseline_metrics['throughput_tok_s'] - 1.0
        if baseline_metrics['throughput_tok_s'] else 0.0
    )
    return {
        'total_latency_delta_ms': round(total_latency_delta_ms, 2),
        'ttft_delta_ms': round(ttft_delta_ms, 2),
        'tpot_delta_ms': round(tpot_delta_ms, 4),
        'peak_mem_delta_mb': round(peak_mem_delta_mb, 2),
        'throughput_gain': round(throughput_gain, 4),
    }

# TODO 6: 输出推理选型建议
def recommend_inference_decision(comparison, candidate_bottleneck, min_throughput_gain=0.1, max_ttft_regression_ms=20.0):
    ttft_regression_ms = -comparison['ttft_delta_ms']
    if comparison['throughput_gain'] >= min_throughput_gain and ttft_regression_ms <= max_ttft_regression_ms:
        decision = 'accept'
        reason = 'candidate 吞吐提升明显，TTFT 退化在可接受范围内，值得进入正式推理方案。'
    elif comparison['throughput_gain'] > 0.0 and candidate_bottleneck['bottleneck'] != 'balanced':
        decision = 'tune'
        reason = 'candidate 已有收益，但瓶颈仍然存在，继续围绕诊断结果调参或换策略。'
    else:
        decision = 'reject'
        reason = 'candidate 收益不足或交互延迟退化明显，当前不值得切换。'
    return {'decision': decision, 'reason': reason}

baseline_config = build_inference_config('tiny-llama', 'pytorch-eager', 2, 128, 32, 'fp16', 'static-kv-cache')
baseline_latency = summarize_prefill_decode(prefill_ms=80.0, decode_ms=160.0, generated_tokens=32)
baseline_metrics = compute_inference_metrics(baseline_config, baseline_latency, peak_mem_mb=4096.0)
print(baseline_config)
print(baseline_metrics)
print(diagnose_inference_bottleneck(baseline_metrics, memory_budget_mb=8192.0))

candidate_config = build_inference_config('tiny-llama', 'paged-attention', 2, 128, 32, 'fp16', 'paged-kv-cache')
candidate_latency = summarize_prefill_decode(prefill_ms=85.0, decode_ms=120.0, generated_tokens=32)
candidate_metrics = compute_inference_metrics(candidate_config, candidate_latency, peak_mem_mb=3584.0)
comparison = compare_inference_candidates(baseline_metrics, candidate_metrics)
print(candidate_metrics)
print(comparison)
print(recommend_inference_decision(comparison, diagnose_inference_bottleneck(candidate_metrics, memory_budget_mb=8192.0)))

```

### 解析

**1. TODO 1: 汇总推理 workload 配置**
- **实现方式**：把模型、backend、batch size、prompt tokens、generated tokens、dtype 和 cache policy 放进同一个配置对象。
- **关键点**：推理 benchmark 的第一原则是固定 workload。没有 workload，TTFT、TPOT、吞吐和显存都没有可比性。
- **项目意义**：后续 baseline 和 candidate 只能改一个变量，否则很难判断收益来自哪里。

**2. TODO 2: 汇总 prefill / decode 延迟**
- **实现方式**：`total_ms = prefill_ms + decode_ms`，TTFT 近似取 `prefill_ms`，TPOT 取 `decode_ms / generated_tokens`。
- **关键点**：prefill 和 decode 的瓶颈不同。总耗时下降不代表交互体验一定变好，TTFT 和 TPOT 必须拆开看。
- **项目意义**：这一步把推理性能从一个笼统 latency 拆成可诊断的两段。

**3. TODO 3: 计算推理项目核心指标**
- **实现方式**：用 `batch_size * generated_tokens / total_seconds` 计算 generated tokens/s，并和 TTFT、TPOT、total latency、peak memory 放在同一张账本里。
- **关键点**：throughput 统计的是整个 batch 的输出 token 产出，不是单条请求的 token 数。
- **项目意义**：同一个 candidate 可能吞吐更高但 TTFT 更差，指标必须一起看。

**4. TODO 4: 诊断推理瓶颈**
- **实现方式**：显存接近预算时优先判为 `memory-bound`；否则用 prefill/decode 占比判断主要瓶颈。
- **关键点**：显存预算是硬约束。如果显存已经接近上限，即使 decode 占比高，也要先处理 KV cache、batch size 或量化。
- **项目意义**：诊断结果决定下一步选 FlashAttention、chunked prefill、PagedAttention、KV cache 量化还是 decode scheduling。

**5. TODO 5: 比较 baseline 和 candidate**
- **实现方式**：latency、TTFT、TPOT 和 peak memory 使用 `baseline - candidate`，正数表示 candidate 更好；throughput 使用比例增益。
- **关键点**：delta 的方向要固定，否则报告容易把退化误写成收益。
- **项目意义**：项目报告不只写绝对值，更要说明 candidate 相比 baseline 改善或退化了多少。

**6. TODO 6: 输出推理选型建议**
- **accept**：吞吐提升达标，TTFT 退化在可接受范围内，说明候选方案值得采用。
- **tune**：candidate 有收益，但瓶颈仍然存在，需要继续沿诊断方向调参。
- **reject**：candidate 收益不足，或交互延迟退化明显，当前不值得切换。
- **项目意义**：推理选型不能只看一个指标。最终结论要同时考虑 workload、吞吐、TTFT、TPOT、显存和瓶颈类型。

**推理性能对比的实验原则**
- **变量控制**：同一轮对比中只改一个变量，例如 batch size、precision、推理后端或 cache 策略。
- **指标闭环**：每次实验至少记录 TTFT、TPOT、throughput 和 peak memory。
- **阶段拆分**：把 prefill 和 decode 分开看，避免把长 prompt 问题误判成 decode 问题。
- **结果复盘**：最终输出要回扣 Step 1 的问题：在给定约束下，哪种推理策略最划算，理由是什么。

## Step 6（可选）：Notebook 中运行真实 backend

下面的单元把模型准备、dtype、端口、服务生命周期和 benchmark 串起来。默认 `RUN_REAL_BACKEND = False`，因此没有 GPU 或 vLLM 时仍可以顺利完成本节；在 Colab / ModelScope GPU 环境中，把它改为 `True` 后按需填写模型配置即可。

模型下载和真实服务启动会消耗显存、磁盘与时间，完成实验后务必运行清理单元。
### 真实 backend 实验的环境要求

Step 6 不是只安装一个 Python 包就能保证复现；vLLM 的 CUDA 扩展、PyTorch CUDA wheel、NVIDIA 驱动和 GPU 架构需要同时匹配。当前已验证的本机兼容组合如下：

| 项目 | 本机已验证配置 | 说明 |
|---|---|---|
| OS | Linux x86_64 | vLLM 的主要支持环境 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU（SM120 / Blackwell） | 约 12 GB 显存；小模型可运行 |
| NVIDIA driver | 570.211.01，CUDA 12.8 | 不要与 CUDA 13.0 wheel 混用 |
| client 环境 | `llm_algo`，PyTorch 2.11.0+cu128 | 运行 Notebook 和 benchmark client |
| backend 环境 | `vllm_legacy_cu128`，Python 3.12，PyTorch 2.8.0+cu128，vLLM 0.11.0 | 单独运行 vLLM 服务；通过 HTTP 与 client 解耦 |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` | 权重约 0.92 GiB，适合 smoke test |
| 启动约束 | `bfloat16`、`max_model_len=2048`、`gpu_memory_utilization=0.8`、`--enforce-eager` | 本机需要关闭 TorchInductor/CUDAGraph |

这里的 vLLM 0.11.0 不是教程要求的最低版本，而是本机经过验证的兼容版本。较新的 vLLM 版本可能自动选择 CUDA 13.0 runtime，或在 RTX 5070 Ti 的 SM120 kernel 路径上启动失败；因此教程应固定已验证版本，而不是无条件安装最新版。vLLM 官方的旧版安装文档也提供了 CUDA 12.8 预编译组合。

真实 backend 实验还需要：可访问 HuggingFace 或 ModelScope 的网络、足够的模型缓存磁盘空间、可用的本地端口（默认 8000），以及允许启动本地进程。没有这些条件时，仍可完成本节的 CPU-first 模拟 benchmark。

**重要限制**：当前实测使用 `--enforce-eager`，并且 FlashInfer 不可用时回退到 PyTorch-native sampler。因此本节实测代表“vLLM eager + Triton/原生采样”的可复现结果，不应直接宣称为最新版 vLLM 默认优化配置的性能。

### 不同运行环境的配置边界

| 环境 | 建议策略 | dtype | 注意事项 |
|---|---|---|---|
| 本机 RTX 5070 Ti + 驱动 570 | 固定已验证的 `vLLM 0.11.0 + cu128` | `bfloat16` | 保留 `--enforce-eager`，不要直接升级到默认 CUDA 13 wheel |
| 本机 RTX 5070 Ti + 驱动 580 | 可以重新测试较新的 vLLM / CUDA wheel | 先用 `bfloat16` | 580 只解决驱动 runtime 兼容性，不保证所有 SM120 kernel 都可用；仍需实际 smoke test |
| Colab T4 | 使用 Colab 当前预装 CUDA，再安装匹配的 vLLM | `float16` | T4 通常不适合直接照搬本机的 `bfloat16` 配置 |
| Colab L4 / A100 / H100 | 先探测驱动和 GPU，再选择 vLLM CUDA backend | 通常可用 `bfloat16` | 不要假定每次 Colab 分配到同一种 GPU |

Colab 中应先运行 `!nvidia-smi` 和 `torch.cuda.get_device_name(0)`，再安装与运行时匹配的 vLLM；Colab GPU 型号、驱动和预装 PyTorch 可能随 runtime 改变。若使用当前 vLLM 官方安装方式，可优先让安装器根据 CUDA backend 选择依赖，而不是把本机的 `vllm_legacy_cu128` 环境直接复制过去。

驱动升级到 580 后，首先验证 `nvidia-smi`、`torch.version.cuda` 和 `torch.cuda.is_available()`，再验证 vLLM 服务；驱动版本变新不等于 vLLM 的 Blackwell/SM120 自定义 kernel 一定可用。

```python
import importlib.util
import shutil
import torch

print({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()})
if torch.cuda.is_available():
    print({'device': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0), 'bf16_supported': torch.cuda.is_bf16_supported()})
print({'vllm_on_current_kernel': importlib.util.find_spec('vllm') is not None, 'vllm_command': shutil.which('vllm')})

# 如果 vLLM 在独立 conda 环境中运行，这里可以保持 vllm_on_current_kernel=False，
# 改用手动启动服务，再把 BASE_URL 指向已启动的 OpenAI-compatible API。
```


```python
# 只需要修改这一格
RUN_REAL_BACKEND = False  # 是否启动真实 vLLM；False 只完成 CPU-first 模板。
MODEL_SOURCE = 'auto'  # 模型来源：auto / modelscope / huggingface / local。
MODEL_CACHE_DIR = 'model_cache'  # 模型缓存目录；通常不需要修改。
MODEL_PROFILES = {
    'qwen25_small': 'Qwen/Qwen2.5-0.5B-Instruct',
    'deepseek_r1_small': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
}
MODEL_PROFILE = 'qwen25_small'  # 先用小模型完成 smoke test。
MODEL_ID = MODEL_PROFILES[MODEL_PROFILE]  # 实际加载的模型 ID。
DTYPE = 'auto'  # auto 根据 GPU 选择；也可显式写 bfloat16 / float16。
VLLM_COMMAND = None  # 为空时自动查找当前环境中的 vllm
VLLM_ENV = None  # 云端保持当前 runtime；本地多环境时再填写环境名
MAX_MODEL_LEN = 2048  # 最大上下文长度；越大越占 KV Cache。
GPU_MEMORY_UTILIZATION = 0.8  # vLLM 使用显存比例；需为桌面和其他进程留余量。
ENFORCE_EAGER = True  # 先保证 RTX 50 系列等架构可复现；稳定后可尝试 False
NUM_PROMPTS = 5  # 请求总数；正式实验应大于 smoke test。
CONCURRENCY = 1  # 同时在途请求数；只做并发实验时改变它。
WARMUP = 1  # 预热请求数；正式实验建议提高到 3-10。

```


```python
import json
import os
import subprocess
import sys
from pathlib import Path

if RUN_REAL_BACKEND:
    project_root = next((path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'tools').is_dir()), None)
    if project_root is None:
        raise RuntimeError('未找到项目根目录。请从仓库根目录启动 Jupyter，或把仓库根目录加入 sys.path。')
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tools.backend_runtime import resolve_model, start_vllm, stop_backend

    model_path = resolve_model(MODEL_ID, MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
    server, server_log, port, selected_dtype = start_vllm(
        model_path, DTYPE, vllm_command=VLLM_COMMAND,
        vllm_environment=VLLM_ENV,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=ENFORCE_EAGER,
        served_model_name=MODEL_ID,
    )
    print({'model_path': model_path, 'dtype': selected_dtype, 'port': port})

    try:
        output_path = Path('benchmarks/results/66_vllm_real.json')
        subprocess.run([
            sys.executable, 'tools/benchmark_inference_backend.py',
            '--base-url', f'http://127.0.0.1:{port}',
            '--model', MODEL_ID,
            '--label', 'vllm-real',
            '--project', '66',
            '--backend', 'vllm',
            '--dtype', selected_dtype,
            '--batch', '1',
            '--cache-policy', 'default',
            '--workload', 'benchmarks/workloads/fixed.jsonl',
            '--num-prompts', str(NUM_PROMPTS),
            '--concurrency', str(CONCURRENCY),
            '--warmup', str(WARMUP),
            '--output', str(output_path),
        ], check=True)
        saved = json.loads(output_path.read_text(encoding='utf-8'))
        print(saved['metrics'])
        print('统一结果：', json.dumps(saved['normalized_result'], ensure_ascii=False, indent=2))
    finally:
        stop_backend(server, server_log)
else:
    print('跳过真实 backend：保持 CPU-first 模式。')

```

### 本机真实 backend 实测记录

以下结果来自 RTX 5070 Ti Laptop GPU，Qwen/Qwen2.5-0.5B-Instruct，vLLM 0.11.0，PyTorch 2.11.0+cu128，`bfloat16`，`max_model_len=2048`，`gpu_memory_utilization=0.8`，并启用 `--enforce-eager`。workload 为 `benchmarks/workloads/fixed.jsonl`，共 5 条请求，`max_tokens=64`。

| 并发 | 成功/失败 | 请求吞吐（req/s） | 输出吞吐（token/s） | TTFT P50 | TPOT P50 | E2E P50 | 结果文件 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5/0 | 3.1189 | 182.1461 | 33.776 ms | 4.859 ms | 337.176 ms | `66_vllm_real.json` |
| 4 | 5/0 | 4.2972 | 250.9544 | 234.566 ms | 11.528 ms | 956.383 ms | `66_vllm_concurrency4.json` |

**如何解读**：并发从 1 提升到 4 后，输出吞吐由 182.15 提升到 250.95 token/s，但 TTFT P50 由 33.78 ms 增至 234.57 ms，E2E P50 由 337.18 ms 增至 956.38 ms。说明本次配置通过批处理提高了吞吐，同时增加了交互延迟；在只有 5 条请求的 smoke test 中，P99 只作记录，不作为稳定结论。

**当前结论**：真实 backend 链路已打通。若目标是交互式单请求，优先关注并发 1 的 TTFT/E2E；若目标是批量吞吐，再继续测试更大的 workload 和并发 sweep，并同时采集 GPU 显存。
### 手动启动方式（可选附录）

如果需要单独调试服务，也可以在终端运行 `vllm serve <model-id> --dtype bfloat16 --port 8000`，再运行 `tools/benchmark_inference_backend.py`。Notebook 主流程不依赖手动查端口或拼接命令。