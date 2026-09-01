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
本节的浮点 baseline 用于给其他推理项目提供共同参照，不实现 GPTQ/AWQ/GGUF 的格式转换、校准或加载。量化 artifact 与专用 backend 由 `67` 验证，再将已确认的结果带回本节做统一 workload 对照。
**层级定位：** 本项目主落在 L4，关注单个模型实例如何执行请求；会调用 L2 的算子/后端能力和 L3 的运行时，但不负责 L5 的多模型发布、集群扩缩容或流量治理。

> 运行提示：运行真实 backend 前，先查看[使用指南中的项目环境预检与安装说明](../docs/guide.md#项目环境预检与安装)。默认使用当前 Notebook runtime；本地只有在 vLLM 与 PyTorch 依赖冲突时才需要额外环境。

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
### Step 1: 定义问题、workload 和实验分组
先回答一个问题：在同一模型、同一输入集和同一硬件环境下，哪种推理策略更划算？

- 固定模型、backend、batch size、prompt tokens、generated tokens、dtype、cache policy 和评测轮数。
- Baseline 建议从 `PyTorch eager + batch=1 + 固定 prompt/output length + warm-up + 多轮测量` 开始。
- 明确 candidate 只改一个变量，例如 FlashAttention、batch size、KV cache 策略、量化精度或推理后端。
- 统一核心指标：TTFT、TPOT、generated tokens/s、total latency、peak memory。
- 这节的目标不是证明某个方案“能跑”，而是在相同约束下输出可解释的推理选型结论。

先做 CPU 机制验证，再按条件升级 GPU/backend：

| 组别 | 环境 | 实验内容 | 主要回答的问题 |
| --- | --- | --- | --- |
| C0 机制模拟 | CPU | 改变请求数量、并发、prefill/decode 成本模型 | 排队、阶段耗时和指标如何变化？ |
| C1 瓶颈诊断 | CPU | 用模拟指标区分 prefill-bound、decode-bound、memory-bound | 应该优先选择哪类策略？ |
| G0 真实 baseline | GPU/backend | 固定模型和 workload，运行 vLLM baseline | 当前硬件和软件栈的真实基线是什么？ |
| G1 单变量对照 | GPU/backend | 一次只改变 batch、dtype、cache 或 attention backend | 某个策略是否带来真实收益？ |
| G2 策略融合 | GPU/backend | 先有单项证据，再比较两种或多种策略组合 | 组合后是否仍值得采用？ |

C0/C1 是必做的 CPU-first 主线；G0/G1 是有 GPU 和 backend 时的真实验证；G2 是可选收口实验。没有 GPU 时，不需要伪造 G0-G2 的数值。

#### G1 单变量矩阵

| 变量 | 类型 | 对应 Task | 66 的职责 |
| --- | --- | --- | --- |
| 并发度 / batch | workload / serving | Task0、Task4 | 当前 vLLM 入口可自动执行 |
| dtype | 运行配置 | Task1、Task5 | 当前 vLLM 入口可自动执行 |
| prompt / output 长度 | workload | Task0、Task2、Task3 | 可自动切换 workload，观察阶段瓶颈 |
| attention backend | 算子策略 | Task2 | 需要 backend 或 profiler 证据 |
| KV Cache / Prefix Cache | 内存策略 | Task4 | 由 69 专项验证后接入对比 |
| speculative decoding | 解码策略 | Task3 | 由 68 验证 draft / verify |
| quantization backend | 压缩策略 | Task5 | 由 67 验证量化格式和 kernel |
| serving scheduler | 服务策略 | Task4 | 由 70 验证队列和公平性 |

其中 prompt / output 长度是 workload 变量，不应和 FlashAttention、Cache 或量化一起当作一个“优化策略”。当前 66 自动 G1 只覆盖并发度、batch、dtype 和 workload 切换；其余变量必须先在专项项目中确认真实开关，再放入 66 做统一端到端对比。

### Step 2: 建立 baseline 并控制变量

baseline 必须先跑通并可复现；后续 candidate 只改变一个主要变量，不能把不同 workload、batch 或 backend 的数字直接混在一起。

- 固定模型、backend、batch、prompt tokens、generated tokens、dtype、cache policy、warm-up 和重复次数。
- baseline 建议从 PyTorch eager、batch=1、固定 prompt/output length 开始。
- TTFT、TPOT、throughput、total latency 和 peak memory 必须来自同一套 workload。

### Step 3: 运行候选、分析指标并诊断瓶颈

对 baseline 和 candidate 同时记录 latency、throughput、memory，再根据阶段占比和预算判断瓶颈。

| 瓶颈类型 | 典型信号 | 候选方向 |
| --- | --- | --- |
| prefill-bound | prefill 占比高，长 prompt 变慢明显 | FlashAttention、chunked prefill、batching |
| decode-bound | TPOT 高，decode 占比高 | speculative decoding、multi-token decoding、decode scheduling |
| memory-bound | peak memory 接近预算，batch 上不去 | KV cache quantization、PagedAttention、GQA/MQA |
| balanced | 各项都不突出 | 保持 baseline 或做小步 profiling |

### Step 4: 输出综合决策

推理选型最终不是输出“哪个 benchmark 更好看”，而是判断方案在当前 workload 下是否值得保留、微调或切换。

- 输出 baseline vs candidate 对比表，至少包含 TTFT、TPOT、throughput、total latency、peak memory 和瓶颈判断。
- 如果 candidate 只提升吞吐但明显拉高 TTFT，要说明适合离线批处理还是在线交互。
- 如果 candidate 显存更省但 TPOT 变差，要说明是否为了更大 batch 或更长上下文让路。
- 最终决策统一使用 `accept / tune / reject`：候选方案值得采用、还需继续调优、或当前不值得切换。
- 报告结论必须回扣 Step 1 的 workload，不能泛化成“某方案永远更好”。

### Step 5：CPU 实验——机制与决策模板

上面的 Step 1-4 是完整推理性能对比项目流程。下面的代码实现七块最小能力：模拟请求执行、配置 workload、汇总 prefill/decode、计算指标、诊断瓶颈、比较候选方案和输出决策。

完成 Step 5 后，如果当前环境具备 GPU 和 vLLM，可以继续运行文末的 GPU/backend 实验单元；没有这些条件时，停留在 CPU-first 主线即可。
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
| 策略矩阵 | baseline、单项策略、组合策略及 `strategies` 字段 | 防止多变量收益无法归因 |
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
# 补全请求模拟和推理性能对比的七个关键函数
# 目标：完成 request -> workload -> metrics -> bottleneck -> comparison -> decision 链路。
# CPU 题目区只验证离散事件和指标口径；真实 kernel、KV Cache 和 backend 行为由 GPU 扩展验证。
def simulate_inference_requests(requests, concurrency=1, prefill_ms_per_token=0.5, decode_ms_per_token=1.0, peak_mem_per_request_mb=512.0, kv_cache_mb_per_token=0.0):
    """在 CPU 上模拟请求排队和 prefill/decode 阶段，不测真实 kernel 性能。

    requests 至少包含 prompt_tokens 和 generated_tokens；返回每个请求的时间轨迹及汇总指标。
    """
    if concurrency <= 0 or prefill_ms_per_token < 0 or decode_ms_per_token < 0 or peak_mem_per_request_mb < 0 or kv_cache_mb_per_token < 0:
        raise ValueError('concurrency 和成本参数必须合法')
    # ==========================================
    # TODO 0: 按 concurrency 分批，计算每个请求的 queue / prefill / decode / e2e
    # 提示：每一批同时执行，批次耗时取该批请求 prefill+decode 的最大值；
    #       queue_ms 是等待时间，peak_memory 取同时执行请求数 * 单请求预算，
    #       kv_cache_mb_per_token 只作为教学估算项，不是实际 KV Cache 分配。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
    """汇总推理 workload 配置，形成统一比较口径。

    prompt_tokens 和 generated_tokens 必须是非负整数；total_tokens 是两者之和。
    """
    # ==========================================
    # TODO 1: 汇总推理 workload 配置
    # 提示：total_tokens = prompt_tokens + generated_tokens；不要把 batch 重复加进 token 数。
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
    """汇总 prefill / decode 延迟，形成最小延迟摘要。

    TTFT 近似为 prefill_ms，TPOT 只在 generated_tokens > 0 时计算；
    prefill_share 与 decode_share 应在 total_ms 上归一化。
    """
    # ==========================================
    # TODO 2: 汇总 prefill / decode 延迟
    # 提示：TTFT 近似等于 prefill_ms；TPOT = decode_ms / generated_tokens。
    #       generated_tokens 为 0 时应明确处理，而不是产生除零错误。
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
    """把 workload 和延迟摘要收束成统一推理指标。

    throughput_tok_s 表示该 workload 的输出 token 吞吐；peak_mem_mb 是输入的估算或实测值，
    必须沿用其证据等级，不能在函数内改写来源。
    """
    # ==========================================
    # TODO 3: 计算推理项目核心指标
    # 提示：throughput 表示整个 batch 每秒生成 token 数，即 batch * generated_tokens / total_ms。
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
    """根据显存预算与 prefill/decode 占比诊断推理瓶颈。

    返回 bottleneck 和 reason；这是规则化诊断，不是 profiler 的最终归因。
    """
    # ==========================================
    # TODO 4: 诊断推理瓶颈
    # 规则：显存接近预算优先判 memory-bound；否则按 prefill/decode 占比判断。
    # 提示：memory_budget_mb 为空时不要计算显存压力；占比相等时返回 balanced。
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
    """统一比较 baseline 与 candidate 的推理收益和代价。

    两者必须使用同一 workload；延迟和显存差值采用 baseline - candidate，
    throughput_gain 采用 (candidate - baseline) / baseline。
    """
    # ==========================================
    # TODO 5: 比较 baseline 和 candidate
    # 提示：latency / TTFT / TPOT / memory 的 delta 用 baseline - candidate；throughput gain 用比例增益。
    #       baseline throughput 为 0 时应显式处理，不能静默返回无穷大。
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
    """根据吞吐、TTFT 和瓶颈类型输出推理选型建议。

    返回 decision 和 reason；阈值属于当前教学 workload，不是通用 SLA。
    """
    # ==========================================
    # TODO 6: 输出推理选型建议
    # 规则：吞吐明显提升且 TTFT 没明显退化则 accept；有收益但仍有瓶颈则 tune；否则 reject。
    # 提示：throughput_good、ttft_ok、still_tunable 分别对应三个判断条件。
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
        requests = [
            {'prompt_tokens': 100, 'generated_tokens': 20},
            {'prompt_tokens': 200, 'generated_tokens': 10},
            {'prompt_tokens': 100, 'generated_tokens': 20},
        ]
        simulation = simulate_inference_requests(
            requests, concurrency=2, prefill_ms_per_token=0.5,
            decode_ms_per_token=1.0, peak_mem_per_request_mb=256.0,
        )
        assert simulation['request_count'] == 3, "请求数量统计不正确！"
        assert simulation['duration_ms'] == 180.0, "批次执行时长计算不正确！"
        assert simulation['request_results'][2]['queue_ms'] == 110.0, "排队时间计算不正确！"
        assert simulation['peak_mem_mb'] == 512.0, "并发显存预算计算不正确！"
        scaled = simulate_inference_requests(
            [{'prompt_tokens': 100, 'generated_tokens': 20}],
            peak_mem_per_request_mb=256.0, kv_cache_mb_per_token=1.0,
        )
        assert scaled['kv_cache_tokens_peak'] == 120, "KV Cache token 数计算不正确！"
        assert scaled['peak_mem_mb'] == 376.0, "KV Cache 显存随 token 增长的计算不正确！"
        for invalid in ({'concurrency': 0}, {'prefill_ms_per_token': -1.0}, {'kv_cache_mb_per_token': -1.0}):
            try:
                simulate_inference_requests(requests, **invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('非法请求模拟参数应明确拒绝！')

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
def simulate_inference_requests(requests, concurrency=1, prefill_ms_per_token=0.5, decode_ms_per_token=1.0, peak_mem_per_request_mb=512.0, kv_cache_mb_per_token=0.0):
    """模拟并发请求的排队、prefill、decode 和显存占用。"""
    if concurrency <= 0 or prefill_ms_per_token < 0 or decode_ms_per_token < 0 or peak_mem_per_request_mb < 0 or kv_cache_mb_per_token < 0:
        raise ValueError('concurrency 和成本参数必须合法')
    if not requests:
        return {'request_count': 0, 'duration_ms': 0.0, 'request_results': [], 'peak_mem_mb': 0.0}
    results = []
    clock_ms = 0.0
    for start in range(0, len(requests), concurrency):
        wave = requests[start:start + concurrency]
        wave_results = []
        for request in wave:
            prompt_tokens = int(request['prompt_tokens'])
            generated_tokens = int(request['generated_tokens'])
            if prompt_tokens <= 0 or generated_tokens <= 0:
                raise ValueError('每个请求的 token 数必须为正数')
            prefill_ms = prompt_tokens * prefill_ms_per_token
            decode_ms = generated_tokens * decode_ms_per_token
            wave_results.append({
                'prompt_tokens': prompt_tokens,
                'generated_tokens': generated_tokens,
                'queue_ms': clock_ms,
                'prefill_ms': prefill_ms,
                'decode_ms': decode_ms,
                'ttft_ms': clock_ms + prefill_ms,
                'tpot_ms': decode_ms / generated_tokens,
                'e2e_ms': clock_ms + prefill_ms + decode_ms,
                'kv_cache_tokens': prompt_tokens + generated_tokens,
                'kv_cache_mem_mb': peak_mem_per_request_mb + (prompt_tokens + generated_tokens) * kv_cache_mb_per_token,
            })
        results.extend(wave_results)
        clock_ms += max(item['prefill_ms'] + item['decode_ms'] for item in wave_results)
    total_output_tokens = sum(item['generated_tokens'] for item in results)
    total_prompt_tokens = sum(item['prompt_tokens'] for item in results)
    return {
        'request_count': len(results),
        'total_prompt_tokens': total_prompt_tokens,
        'total_output_tokens': total_output_tokens,
        'duration_ms': round(clock_ms, 4),
        'peak_mem_mb': round(max(sum(item['kv_cache_mem_mb'] for item in results[start:start + concurrency]) for start in range(0, len(results), concurrency)), 2),
        'kv_cache_tokens_peak': max((sum(item['kv_cache_tokens'] for item in results[start:start + concurrency]) for start in range(0, len(results), concurrency)), default=0),
        'request_results': results,
    }

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

**0. TODO 0: 模拟请求执行**
- **实现方式**：按 `concurrency` 将请求分成多个执行批次；同一批并行完成，批次耗时取其中最长请求的 prefill + decode 时间。
- **关键点**：后续批次会产生 queue time；`TTFT` 包含排队和 prefill，`TPOT` 只表示 decode 阶段的平均每 token 时间。
- **项目意义**：CPU 可以解释并发、排队和阶段指标的关系，但这些是教学成本模型，不是 vLLM / SGLang 的真实调度或 CUDA 测量。
- **显存扩展**：当 `kv_cache_mb_per_token > 0` 时，模拟器按 `prompt_tokens + generated_tokens` 估算每个请求的 KV Cache 增量，并按同一执行波次累加；这只能说明 token 数、并发与容量之间的关系，不能替代 backend 的 allocated/reserved 显存或 OOM 测量。

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

## Step 6（可选）：GPU/backend 实验——真实 baseline 与候选对照

下面的单元对应实验组 G0：把模型准备、dtype、端口、服务生命周期和 benchmark 串起来。完成 G0 后，固定模型和 workload，再分别运行 G1 的单变量对照；只有单项策略已有结果，才进入 G2 的策略融合。默认 `RUN_REAL_BACKEND = False`，因此没有 GPU 或 vLLM 时仍可以顺利完成本节；在 Colab / ModelScope GPU 环境中，把它改为 `True` 后按需填写模型配置即可。

66 不要求在一个 Notebook 中重新实现量化、Prefix Cache、Speculative Decoding 或 Scheduler；这些机制分别由 67–70 负责。66 只读取它们的统一结果，比较端到端指标，并记录组合配置。

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
EXPERIMENT_GROUP = 'G0'  # G0=baseline，G1=单变量对照，G2=策略融合。
STRATEGIES = []  # G0 为空；G1 填一个策略名；G2 填两个或更多已单独验证的策略名。
STRATEGY_NOTE = '固定 workload 的 vLLM baseline'  # 说明本次实验实际改变了什么。
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
RUN_SGLANG = False  # 可选：只在已安装并确认版本兼容时启动 SGLang。
SGLANG_COMMAND_TEMPLATE = None  # 例如 python -m sglang.launch_server --model-path {model_path} --port {port}
SGLANG_READY_TIMEOUT_S = 180  # SGLang 冷启动等待时间；超时会自动停止子进程。
SGLANG_RESULT_PATH = 'benchmarks/results/66_g0_sglang.json'  # 与 vLLM 结果分开保存。
MAX_MODEL_LEN = 2048  # 最大上下文长度；越大越占 KV Cache。
GPU_MEMORY_UTILIZATION = 0.8  # vLLM 使用显存比例；需为桌面和其他进程留余量。
ENFORCE_EAGER = True  # 先保证 RTX 50 系列等架构可复现；稳定后可尝试 False
WORKLOAD_PATH = 'benchmarks/workloads/fixed.jsonl'  # G1 workload 变量通过替换此文件实现。
NUM_PROMPTS = 5  # 请求总数；正式实验应大于 smoke test。
BATCH_SIZE = 1  # 单请求 batch 配置；修改后必须在结果中保留。
CONCURRENCY = 1  # 同时在途请求数；只做并发实验时改变它。
WARMUP = 1  # 预热请求数；正式实验建议提高到 3-10。
RESULT_PATH = 'benchmarks/results/66_g0_vllm_baseline.json'  # 每组实验使用独立路径，避免覆盖。
PEAK_MEMORY_MB = None  # 可选：由外部 nvidia-smi/监控采集后填入；None 表示本次未测 GPU 峰值显存。

SUPPORTED_AUTO_STRATEGIES = {'concurrency', 'batch', 'dtype', 'workload'}  # 当前 vLLM 入口确实能执行的 G1 变量。
if EXPERIMENT_GROUP not in {'G0', 'G1', 'G2'}:
    raise ValueError('EXPERIMENT_GROUP 只能是 G0、G1 或 G2')
if EXPERIMENT_GROUP == 'G0' and STRATEGIES:
    raise ValueError('G0 baseline 不应填写 STRATEGIES')
if EXPERIMENT_GROUP == 'G1' and len(STRATEGIES) != 1:
    raise ValueError('G1 单变量对照必须填写一个策略')
if EXPERIMENT_GROUP == 'G2' and len(STRATEGIES) < 2:
    raise ValueError('G2 策略融合至少填写两个策略')
if RUN_REAL_BACKEND and EXPERIMENT_GROUP == 'G1' and not set(STRATEGIES).issubset(SUPPORTED_AUTO_STRATEGIES):
    raise ValueError('当前 GPU 入口只自动执行 concurrency / batch / dtype / workload 对照；FlashAttention、Prefix Cache、量化等请使用专项项目或手动 backend 参数。')
if RUN_REAL_BACKEND and EXPERIMENT_GROUP == 'G2':
    raise ValueError('G2 策略融合目前只有配置与报告元数据入口，尚未自动启用组合 backend；请先完成单项策略验证。')
EXPERIMENT_METADATA = {'group': EXPERIMENT_GROUP, 'strategies': list(STRATEGIES), 'note': STRATEGY_NOTE}
print('实验配置：', EXPERIMENT_METADATA)

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
        output_path = Path(RESULT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_command = [
            sys.executable, 'tools/benchmark_inference_backend.py',
            '--base-url', f'http://127.0.0.1:{port}',
            '--model', MODEL_ID,
            '--label', f'vllm-{EXPERIMENT_GROUP.lower()}',
            '--project', '66',
            '--backend', 'vllm',
            '--dtype', selected_dtype,
            '--batch', str(BATCH_SIZE),
            '--cache-policy', 'default',
            '--workload', WORKLOAD_PATH,
            '--num-prompts', str(NUM_PROMPTS),
            '--concurrency', str(CONCURRENCY),
            '--warmup', str(WARMUP),
            '--output', str(output_path),
        ]
        if PEAK_MEMORY_MB is not None:
            benchmark_command.extend(['--peak-memory-mb', str(PEAK_MEMORY_MB)])
        subprocess.run(benchmark_command, check=True)
        saved = json.loads(output_path.read_text(encoding='utf-8'))
        saved['experiment'] = EXPERIMENT_METADATA
        output_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
        print(saved['metrics'])
        print('统一结果：', json.dumps(saved['normalized_result'], ensure_ascii=False, indent=2))
    finally:
        stop_backend(server, server_log)
else:
    print('跳过真实 backend：保持 CPU-first 模式。')

```


```python
# Step 7：可选 SGLang 对照；默认关闭，不影响 vLLM 主线。
if RUN_SGLANG:
    if not SGLANG_COMMAND_TEMPLATE:
        raise ValueError('RUN_SGLANG=True 时必须填写 SGLANG_COMMAND_TEMPLATE，并包含 {model_path} 和 {port}。')
    from tools.inference_project_runtime import (
        locate_repo_root, run_backend_benchmark, start_external_openai_backend,
    )
    from tools.backend_runtime import find_free_port, resolve_model, stop_backend
    root = locate_repo_root()
    model_path = resolve_model(MODEL_ID, MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
    sglang_port = find_free_port()
    sglang_log = root / 'benchmarks/results/66_sglang.log'
    sglang_server, sglang_log_path = start_external_openai_backend(
        SGLANG_COMMAND_TEMPLATE, model_path=str(model_path), port=sglang_port,
        log_path=sglang_log, ready_timeout_s=SGLANG_READY_TIMEOUT_S,
    )
    try:
        sglang_report = run_backend_benchmark(
            project='66', base_url=f'http://127.0.0.1:{sglang_port}',
            model=MODEL_ID, label='sglang-g0', output=SGLANG_RESULT_PATH,
            workload=WORKLOAD_PATH, num_prompts=NUM_PROMPTS,
            concurrency=CONCURRENCY, warmup=WARMUP, backend='sglang',
            dtype=DTYPE, batch=BATCH_SIZE, cache_policy=CACHE_POLICY,
        )
        sglang_report['experiment'] = {**EXPERIMENT_METADATA, 'backend': 'sglang', 'command_template': SGLANG_COMMAND_TEMPLATE}
        Path(SGLANG_RESULT_PATH).write_text(json.dumps(sglang_report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('SGLang 结果：', json.dumps(sglang_report.get('metrics', {}), ensure_ascii=False, indent=2))
    finally:
        stop_backend(sglang_server, sglang_log_path)
else:
    print('跳过 SGLang：默认只验证 vLLM；需要独立安装和确认 SGLang 版本后再开启。')

```

### 本机真实 backend 实测记录

下面的记录对应 Step 6 的 G0/G1：并发 1 是固定 workload 的 baseline，并发 4 是只改变 concurrency 的单变量对照。它们不是两个不同的推理 backend，也不是两种已经验证的优化策略。

**实验条件**

| 项目 | 配置 |
|---|---|
| GPU | RTX 5070 Ti Laptop GPU |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Backend | vLLM 0.11.0 |
| PyTorch / CUDA | 2.11.0+cu128 |
| dtype | `bfloat16` |
| 启动参数 | `max_model_len=2048`、`gpu_memory_utilization=0.8`、`--enforce-eager` |
| workload | `benchmarks/workloads/fixed.jsonl`，5 条请求，`max_tokens=64` |

**实验结果**

| 实验组 | 改变的变量 | 并发 | 成功/失败 | 请求吞吐（req/s） | 输出吞吐（token/s） | TTFT P50 | TPOT P50 | E2E P50 | 结果文件 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| G0 baseline | 无 | 1 | 5/0 | 3.1189 | 182.1461 | 33.776 ms | 4.859 ms | 337.176 ms | `66_vllm_real.json` |
| G1 concurrency | concurrency | 4 | 5/0 | 4.2972 | 250.9544 | 234.566 ms | 11.528 ms | 956.383 ms | `66_vllm_concurrency4.json` |

**如何解读**：并发从 1 提升到 4 后，输出吞吐由 182.15 提升到 250.95 token/s，但 TTFT P50 由 33.78 ms 增至 234.57 ms，E2E P50 由 337.18 ms 增至 956.38 ms。说明本次配置通过批处理提高了吞吐，同时增加了交互延迟；在只有 5 条请求的 smoke test 中，P99 只作记录，不作为稳定结论。

**当前结论**：真实 backend 链路已打通。若目标是交互式单请求，优先关注并发 1 的 TTFT/E2E；若目标是批量吞吐，再继续测试更大的 workload 和并发 sweep，并同时采集 GPU 显存。现有两份历史 JSON 没有 `peak_memory` 字段，因此这里只能下延迟/吞吐结论，不能反推出 KV Cache 或并发显存结论。

### Step 7（可选）：SGLang backend 对照

SGLang 不与 vLLM 共用启动代码。它的 RadixAttention / Radix Cache 机制、版本和启动参数都有自己的边界；本节只复用统一 benchmark 客户端。Step 7 已提供可选的外部 OpenAI-compatible backend 启动适配器，但不会猜测 SGLang 版本或命令参数；学习者必须填写本环境已验证的命令模板。

### Step 8（可选）：跨 backend 结果对比

跨 backend 只能在模型、输入分布、生成长度、并发、dtype 和硬件一致时比较。报告必须保留 backend 名称和版本；vLLM 的 PagedAttention 与 SGLang 的 RadixAttention 不视为同一种策略，端到端差异不能直接归因给某个单一机制。

当前学习顺序：先完成 vLLM G0/G1，再阅读 24 和 69 理解 SGLang 的缓存机制，最后开启 Step 7–8；没有可运行的 SGLang 环境时保留关闭状态即可。
### 手动启动方式（可选附录）

如果需要单独调试服务，也可以在终端运行 `vllm serve <model-id> --dtype bfloat16 --port 8000`，再运行 `tools/benchmark_inference_backend.py`。Notebook 主流程不依赖手动查端口或拼接命令。