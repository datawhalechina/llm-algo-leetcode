# 81. Distributed Inference Project | 分布式推理逻辑验证
**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `分布式推理`, `推理服务` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---
## 本节导读

本节用同一组请求比较完整副本路由、模型并行实例与 Prefill / Decode 分离三种服务形态。先在 CPU 上拆开请求阶段、队列与 handoff 成本，再把最有价值的候选带入多卡 backend，形成从服务选择到部署证据的连续项目。

**关键词：** `distributed inference`, `routing`, `load balance`, `communication cost`, `migration decision`

---
## 前置阅读

**导语：** 先建立单卡推理基线，再理解请求调度、模型内部切分与副本路由。本项目把这些机制放到同一请求 workload 下，判断是否值得迁移到分布式部署。
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)
- [49. Parallelism Strategy Selection | 并行策略选择](./49_Parallelism_Strategy_Selection.md)
- [PAR-REPLICA. Serving Replica Routing | Serving 副本路由](./PAR-REPLICA_Serving_Replica_Routing.md)


---
### Step 1：从请求阶段选择分布式服务形态

一条请求先在 Prefill 阶段处理 prompt，再在 Decode 阶段逐 token 生成。最直接的扩展方式是增加完整副本并把请求路由到较空闲副本；当长 prompt 与长 decode 争抢同一组 GPU 时，也可以让 Prefill 和 Decode 位于不同 worker。后一种设计需要把 KV 状态从 Prefill worker 交给 Decode worker，收益取决于阶段计算差异是否大于 handoff、排队和协调成本。

| 服务形态 | 分工对象 | 主要收益 | 首先观察的代价 |
|:---|:---|:---|:---|
| 完整副本路由 | 整个请求 | 提高并发、降低排队 | 副本负载不均、模型副本容量 |
| TP / PP 服务实例 | 单个模型执行 | 让大模型跨卡运行 | collective / stage 传输、长尾延迟 |
| Prefill / Decode 分离 | 请求的两个阶段 | 隔离不同计算特征 | KV handoff bytes、网络等待、状态一致性 |
### Step 2：定义可比较的请求 workload

每个请求至少记录 prompt tokens、生成 tokens 和到达顺序。若比较 PD 分离，还要固定 KV 表示的每 token 字节数与 Prefill / Decode worker 的数量；若比较副本路由，则固定副本数、路由策略和并发上限。这样才能判断时间变化来自服务形态，而不是请求长度或输入分布改变。

| 字段 | 用途 | 对哪些策略必须一致 |
|:---|:---|:---|
| `prompt_tokens` / `generate_tokens` | 决定 Prefill 与 Decode 工作量 | 全部对照 |
| 到达顺序与并发 | 决定 queue wait 和路由压力 | 副本路由、PD 分离 |
| `kv_bytes_per_token` | 估算 PD handoff 传输量 | PD 分离 |
| TP / PP / replicas | 说明服务实例的切分与副本形状 | 分布式 candidate |
| backend、dtype、模型 revision | 保证运行时与模型产物可比 | 真实 GPU benchmark |
### Step 3：用阶段指标判断收益来自哪里

副本扩展首先看 queue wait、负载和 makespan；PD 分离还必须单独看 TTFT、TPOT 与 KV handoff。把通信成本合并到总时延会掩盖问题：TTFT 改善但 handoff 过慢，或吞吐提升却导致长尾上升，都是需要继续调优而非直接接受的信号。

| 指标 | 主要回答的问题 | 适用策略 |
|:---|:---|:---|
| TTFT / Prefill time | prompt 是否更快进入首 token | 全部，尤其 PD 分离 |
| TPOT / Decode time | 生成阶段是否被计算或 KV 访存拖慢 | 全部，尤其 Decode worker |
| queue wait / makespan | 路由和副本是否真的提升并发 | 副本路由、PD 分离 |
| handoff bytes / handoff time | KV 状态传输是否暴露在关键路径 | PD 分离 |
| request / token throughput、P95/P99 | 系统是否稳定扩大服务能力 | 真实 GPU benchmark |
### Step 4：CPU 模拟——比较路由、并行与 PD handoff

题目区把请求时间拆成 Prefill、Decode、TP/PP 通信代理与 PD handoff，再用同一请求集合比较完整副本路由、模型并行实例和 PD 分离。CPU 账本不预测真实 GPU 时间，而是帮助你识别下一轮真实 benchmark 应测哪一组候选。

| 函数 | 机制职责 | 保留的中间证据 |
|:---|:---|:---|
| `estimate_request_cost` | 分别估算 Prefill、Decode 与 TP/PP 通信代理 | TTFT / TPOT proxy、communication time |
| `simulate_distributed_inference` | 按策略分配请求并累积 worker 负载 | assignments、queue proxy、makespan |
| `recommend_distributed_inference_run` | 同时判断时延、负载、通信与吞吐 | accept / tune / reject 与下一步实验 |
#### 机制证据如何进入分布式推理项目

| 上游证据 | 带入 81 的内容 |
| --- | --- |
| 66 推理 baseline | 固定模型、请求与单实例延迟口径 |
| 70 Serving 调度 | queue wait、并发和长尾证据 |
| 49 并行策略选择 | TP / PP 实例形状与通信约束 |
| PAR-REPLICA 副本路由 | 请求分配、负载和缓存局部性 |
| 81 项目出口 | 比较副本、模型并行实例与 PD 分离并形成迁移决策 |
项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | workload、single-replica time、route policy | 保证迁移比较合法 |
| candidate | TP/PP 逻辑度、副本数、通信成本 | 解释分布式收益来源 |
| 对比 | makespan、throughput、imbalance、comm cost | 判断是否值得迁移 |
| 决策 | accept / tune / reject | 输出逻辑验证结论 |

```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：阶段成本账本、分布式路由模拟、迁移决策
# 目标：用同一 workload 比较副本路由与通信成本；PD handoff 的真实时间在 GPU backend 结果中验证。

def estimate_request_cost(request: Dict[str, int], config: Dict[str, float]) -> Dict[str, float]:
    """估算单请求的 Prefill、Decode 与通信代理成本。"""
    # TODO 1：阶段成本账本。
    # 变量提示：prompt_tokens = request['prompt_tokens']；generate_tokens = request['generate_tokens']。
    # 计算 prefill_ms、decode_ms、TP 通信代理和 PP stage 惩罚，再合成 total_ms。
    raise NotImplementedError("请先完成 TODO 代码！")

def simulate_distributed_inference(
    requests: List[Dict[str, int]], num_replicas: int, config: Dict[str, float]
) -> Dict[str, object]:
    """在 least-loaded 路由下汇总 worker 负载、队列与通信成本。"""
    # TODO 2：请求路由与系统账本。
    # 变量提示：replica_loads = [0.0 for _ in range(num_replicas)]；chosen_replica = min(...)。
    # 保留 assignments、makespan、queue/communication 成本和吞吐代理。
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_distributed_inference_run(
    baseline: Dict[str, float], candidate: Dict[str, float], max_imbalance_ratio: float, max_comm_ratio: float
) -> Dict[str, object]:
    """根据时延、负载与通信代价形成下一步实验决策。"""
    # TODO 3：策略决策。
    # 变量提示：candidate_makespan、comm_ratio、imbalance_ratio、decision。
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_distributed_inference_project():
    requests = [
        {'prompt_tokens': 128, 'generate_tokens': 64},
        {'prompt_tokens': 64, 'generate_tokens': 32},
    ]
    config = {
        'prefill_ms_per_token': 0.01,
        'decode_ms_per_token': 0.02,
        'comm_ms_per_token': 0.03,
        'pipeline_penalty_ms': 0.25,
        'tp_degree': 2,
        'pp_stages': 1,
    }

    cost = estimate_request_cost(requests[0], config)
    assert round(cost['prefill_ms'], 2) == 1.28
    assert round(cost['decode_ms'], 2) == 1.28
    assert round(cost['comm_ms'], 2) == 0.06
    assert round(cost['total_ms'], 2) == 2.62

    pd_config = {**config, 'serving_mode': 'pd_disaggregated', 'kv_bytes_per_token': 16, 'handoff_base_ms': 0.2, 'handoff_bytes_per_ms': 1024}
    pd_cost = estimate_request_cost(requests[0], pd_config)
    assert pd_cost['handoff_bytes'] == 2048
    assert round(pd_cost['handoff_ms'], 2) == 2.2
    assert round(pd_cost['total_ms'], 2) == 4.82

    summary = simulate_distributed_inference(requests, num_replicas=2, config=config)
    assert summary['routing_strategy'] == 'least_loaded'
    assert summary['assignments'] == [0, 1]
    assert round(summary['single_replica_time_ms'], 2) == 3.93
    assert round(summary['makespan_ms'], 2) == 2.62
    assert round(summary['throughput_req_per_ms'], 4) == 0.7634
    assert round(summary['imbalance_ratio'], 4) == 0.6667
    assert round(summary['comm_cost_ms'], 2) == 0.09

    pd_summary = simulate_distributed_inference(requests, num_replicas=2, config=pd_config)
    assert pd_summary['serving_mode'] == 'pd_disaggregated'
    assert pd_summary['handoff_total_bytes'] == 3072
    assert pd_summary['handoff_total_ms'] > 0

    decision = recommend_distributed_inference_run(
        {'single_replica_time_ms': 3.93},
        summary,
        max_imbalance_ratio=0.8,
        max_comm_ratio=0.05,
    )
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_real_cluster_check'

    weak_candidate = {
        'single_replica_time_ms': 3.93,
        'makespan_ms': 3.10,
        'imbalance_ratio': 0.9,
        'comm_cost_ms': 0.09,
    }
    weak_decision = recommend_distributed_inference_run(
        {'single_replica_time_ms': 3.93},
        weak_candidate,
        max_imbalance_ratio=0.8,
        max_comm_ratio=0.05,
    )
    assert weak_decision['decision'] == 'tune'

    bad_candidate = {
        'single_replica_time_ms': 3.93,
        'makespan_ms': 4.30,
        'imbalance_ratio': 1.2,
        'comm_cost_ms': 0.40,
    }
    bad_decision = recommend_distributed_inference_run(
        {'single_replica_time_ms': 3.93},
        bad_candidate,
        max_imbalance_ratio=0.8,
        max_comm_ratio=0.05,
    )
    assert bad_decision['decision'] == 'reject'


test_distributed_inference_project()
print('测试通过：分布式推理逻辑验证模板可以工作。')
```

🛑 **STOP HERE** 🛑

请先尝试自己完成代码并跑通测试。如果你在 Colab 中运行，并且暂时没有思路，再继续看下面的参考答案。
## 参考代码与解析

```python
from typing import Dict, List


def estimate_request_cost(request: Dict[str, int], config: Dict[str, float]) -> Dict[str, float]:
    prompt_tokens = request['prompt_tokens']
    generate_tokens = request['generate_tokens']
    prefill_ms = prompt_tokens * config['prefill_ms_per_token']
    decode_ms = generate_tokens * config['decode_ms_per_token']
    comm_ms = max(config['tp_degree'] - 1, 0) * config['comm_ms_per_token'] * (
        prompt_tokens / 128.0 + generate_tokens / 64.0
    )
    pipeline_ms = max(config['pp_stages'] - 1, 0) * config['pipeline_penalty_ms']
    handoff_bytes = prompt_tokens * config.get('kv_bytes_per_token', 0.0)
    handoff_ms = 0.0
    if config.get('serving_mode', 'replica') == 'pd_disaggregated':
        handoff_ms = config.get('handoff_base_ms', 0.0) + handoff_bytes / config.get('handoff_bytes_per_ms', 1.0)
    total_ms = prefill_ms + decode_ms + comm_ms + pipeline_ms + handoff_ms
    return {
        'prefill_ms': round(prefill_ms, 4),
        'decode_ms': round(decode_ms, 4),
        'comm_ms': round(comm_ms, 4),
        'handoff_bytes': round(handoff_bytes, 4),
        'handoff_ms': round(handoff_ms, 4),
        'total_ms': round(total_ms, 4),
    }


def simulate_distributed_inference(
    requests: List[Dict[str, int]], num_replicas: int, config: Dict[str, float]
) -> Dict[str, object]:
    replica_loads = [0.0 for _ in range(num_replicas)]
    assignments = []
    single_replica_time_ms = 0.0
    comm_cost_ms = 0.0
    handoff_total_ms = 0.0
    handoff_total_bytes = 0.0
    queue_wait_total_ms = 0.0

    for request in requests:
        cost = estimate_request_cost(request, config)
        single_replica_time_ms += cost['total_ms']
        comm_cost_ms += cost['comm_ms']
        handoff_total_ms += cost['handoff_ms']
        handoff_total_bytes += cost['handoff_bytes']
        chosen_replica = min(range(num_replicas), key=lambda idx: replica_loads[idx])
        assignments.append(chosen_replica)
        queue_wait_total_ms += replica_loads[chosen_replica]
        replica_loads[chosen_replica] += cost['total_ms']

    makespan_ms = max(replica_loads) if replica_loads else 0.0
    throughput_req_per_ms = len(requests) / makespan_ms if makespan_ms else 0.0
    min_load = min(replica_loads) if replica_loads else 0.0
    avg_load = sum(replica_loads) / len(replica_loads) if replica_loads else 0.0
    imbalance_ratio = (makespan_ms - min_load) / avg_load if avg_load else 0.0

    return {
        'routing_strategy': 'least_loaded',
        'serving_mode': config.get('serving_mode', 'replica'),
        'assignments': assignments,
        'single_replica_time_ms': round(single_replica_time_ms, 4),
        'makespan_ms': round(makespan_ms, 4),
        'throughput_req_per_ms': round(throughput_req_per_ms, 4),
        'imbalance_ratio': round(imbalance_ratio, 4),
        'comm_cost_ms': round(comm_cost_ms, 4),
        'handoff_total_ms': round(handoff_total_ms, 4),
        'handoff_total_bytes': round(handoff_total_bytes, 4),
        'queue_wait_total_ms': round(queue_wait_total_ms, 4),
    }


def recommend_distributed_inference_run(
    baseline: Dict[str, float], candidate: Dict[str, float], max_imbalance_ratio: float, max_comm_ratio: float
) -> Dict[str, object]:
    baseline_time = baseline.get('single_replica_time_ms', 0.0)
    candidate_makespan = candidate.get('makespan_ms', 0.0)
    comm_ratio = candidate.get('comm_cost_ms', 0.0) / baseline_time if baseline_time else 0.0

    if (
        candidate_makespan < baseline_time
        and candidate.get('imbalance_ratio', 1.0) <= max_imbalance_ratio
        and comm_ratio <= max_comm_ratio
    ):
        return {
            'decision': 'accept',
            'reason': '逻辑时延下降，且负载与通信代价都在可控范围内',
            'next_action': 'promote_to_real_cluster_check',
        }
    if candidate_makespan < baseline_time:
        return {
            'decision': 'tune',
            'reason': '已有逻辑收益，但负载均衡或通信比例还不够稳',
            'next_action': 'refine_routing_or_parallel_degree',
        }
    return {
        'decision': 'reject',
        'reason': 'candidate 没有形成可信的分布式推理收益',
        'next_action': 'fallback_to_single_replica_audit',
    }
```

### Step 5（可选）：GPU / 多副本推理 benchmark

#### 5.1 环境、输入与固定条件

固定模型、tokenizer、请求集合、生成长度、dtype、GPU 数、backend 和路由策略。G0 使用单副本 baseline；G1 比较多副本路由；G2 比较 PD 分离。不同服务形态分别保存 JSON，不把配置差异隐藏在同一结果中。

| 固定对象 | 本节记录 | 作用 |
| --- | --- | --- |
| model contract | model / tokenizer revision、dtype、artifact | 避免模型与编码输入漂移 |
| request workload | prompt / output tokens、concurrency、warmup、repeats | 保证三种服务形态可比 |
| distributed layout | replicas、TP/PP、GPU 与 backend | 解释收益来自路由还是模型切分 |
| output | 每组 JSON、failure、evidence level、decision | 支持独立复测与项目决策 |

```python
# 5.1：默认关闭。每个服务形态各自保存 JSON，避免把副本与 PD 结果混在一起。
RUN_GPU_EXPERIMENT = False
WORLD_SIZE = 2
REPLICAS = 2
TP_DEGREE = 1
PP_STAGES = 1
# 5.3：选择已安装的服务 backend，并为每个实验组提供启动命令。
BACKEND_ADAPTER = 'vllm'  # 可改为 'sglang'；命令语法随 adapter 调整。
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
CONCURRENCY = 1
PROMPT_TOKENS = 512
MAX_TOKENS = 64
KV_BYTES_PER_TOKEN = 16384
RESULT_DIR = 'benchmarks/results/81_distributed_inference'
# 为 G0/G1/G2 填入当前 backend 的启动与压测命令；每组必须写入同名 JSON。
EXPERIMENTS = [
    {'name': 'g0_single_replica', 'mode': 'replica', 'replicas': 1, 'tp': 1, 'pp': 1, 'command': []},
    {'name': 'g1_multi_replica', 'mode': 'replica', 'replicas': REPLICAS, 'tp': TP_DEGREE, 'pp': PP_STAGES, 'command': []},
    {'name': 'g2_pd_disaggregated', 'mode': 'pd_disaggregated', 'replicas': REPLICAS, 'tp': TP_DEGREE, 'pp': PP_STAGES, 'command': []},
]
print({'run': RUN_GPU_EXPERIMENT, 'world_size': WORLD_SIZE, 'backend_adapter': BACKEND_ADAPTER, 'experiments': [item['name'] for item in EXPERIMENTS]})

```

#### 5.2 backend 与通信预检

检查 CUDA、可见 GPU、`torchrun` 与所选 adapter；PD 实验还应确认 Prefill / Decode 服务端点可访问。

```python
# 5.2：默认关闭；开启时检查真实服务 benchmark 的多卡与 adapter 前提。
GPU_READY = False
if RUN_GPU_EXPERIMENT:
    import shutil, torch
    GPU_READY = torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE and shutil.which('torchrun') is not None
    print({'adapter': BACKEND_ADAPTER, 'visible_gpus': torch.cuda.device_count(), 'world_size': WORLD_SIZE, 'ready': GPU_READY})
else:
    print({'run': False, 'adapter': BACKEND_ADAPTER})

```

#### 5.3 配置 G0 / G1 / G2

在每组的 `command` 中填入当前 adapter 的启动与压测命令；命令必须在对应 JSON 路径写入结果。

```python
# 5.3：命令为空时保持默认关闭；填入命令后再开启 RUN_GPU_EXPERIMENT。
for experiment in EXPERIMENTS:
    result_path = f"{RESULT_DIR}/{experiment['name']}.json"
    print({'group': experiment['name'], 'mode': experiment['mode'], 'result_path': result_path, 'command_configured': bool(experiment['command'])})

```

#### 5.4 执行并保存 JSON

按 G0、G1、G2 顺序执行已配置的真实服务实验；通信和 KV handoff 直接从 backend 结果读取。

```python
# 5.4：按服务形态运行已配置的 backend 命令；默认关闭。
import subprocess
from pathlib import Path

if RUN_GPU_EXPERIMENT:
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        if not experiment['command']:
            raise ValueError(f"请为 {experiment['name']} 填写 backend benchmark command，并写入 {RESULT_DIR}/{experiment['name']}.json")
        subprocess.run(experiment['command'], check=True)
        result_path = Path(RESULT_DIR) / f"{experiment['name']}.json"
        if not result_path.exists():
            raise FileNotFoundError(f'真实 benchmark 未写入结果文件：{result_path}')
if not RUN_GPU_EXPERIMENT:
    print('GPU 实验默认关闭；先完成后端启动与通信预检，再采集端到端结果。')

```

#### 5.5 读取结果与记录证据

读取每组 JSON，并将服务延迟、队列、KV handoff 与成功率放在同一份比较记录中。

```python
# 5.5：只接受包含服务、队列与 KV handoff 证据的真实 backend 结果。
import json
from pathlib import Path

required_metric_fields = {'ttft_ms', 'tpot_ms', 'e2e_ms', 'throughput_tokens_per_s', 'p95_ms', 'p99_ms', 'queue_wait_ms', 'kv_handoff_bytes', 'kv_handoff_ms', 'peak_memory_mb', 'success_rate'}
for experiment in EXPERIMENTS:
    result_path = Path(RESULT_DIR) / f"{experiment['name']}.json"
    if not result_path.exists():
        print(f'尚无真实 backend 结果：{result_path}')
        continue
    result = json.loads(result_path.read_text())
    required = {'backend', 'workload', 'hardware', 'metrics', 'evidence_level', 'failure', 'decision'}
    missing = required - set(result)
    if missing:
        raise ValueError(f'{result_path} 缺少项目证据字段：{sorted(missing)}')
    metric_missing = required_metric_fields - set(result['metrics'])
    if metric_missing:
        raise ValueError(f'{result_path} 缺少服务指标：{sorted(metric_missing)}')
    print({'name': experiment['name'], 'metrics': result['metrics'], 'failure': result['failure'], 'decision': result['decision']})

```

#### 5.6 解释结果与形成决策

| 读取项 | 判断与下一步 |
| --- | --- |
| 运行状态 | `failure` 非空时先记录 backend、端口、模型或多卡启动信息 |
| 可比口径 | 核对 model revision、token workload、concurrency、dtype、replicas、TP/PP 与 backend adapter |
| 核心指标 | 联合查看 TTFT、TPOT、E2E、P95/P99、queue wait、KV handoff、吞吐、显存与成功率 |
| 项目决策 | 副本、模型并行或 PD 分离的收益与服务目标匹配时 `accept`；路由、并行度或 handoff 仍可调整时 `tune`；失败或长尾/传输代价吞掉收益时 `reject` |
### 解析

这页现在按 `estimate -> simulate -> decide` 的最小分布式推理迁移闭环组织，不再只是做单卡逻辑模拟。

#### TODO 1

- 实现方式：先分别估算 prefill、decode、TP 通信和 PP 流水惩罚，再合成单请求总时间。
- 关键点：这里的通信成本和流水惩罚不是精确运行时，而是帮助判断迁移是否可能被系统代价吃掉的逻辑估算。
- 项目意义：先把请求成本拆开，后面才能解释分布式推理的收益到底来自并行加速还是只是估算假象。

#### TODO 2

- 实现方式：在 least-loaded 路由下模拟请求分配，汇总 single-replica 总时间、makespan、吞吐和不均衡。
- 关键点：`single_replica_time_ms` 是单机顺序跑完这批请求的基线，`makespan_ms` 才是迁移后的真正系统完成时间。
- 项目意义：这一步把分布式推理从“能不能切分”转成“负载、吞吐和通信能否一起成立”的迁移比较。

#### TODO 3

- 实现方式：先比较基线与 makespan 的加速，再按 imbalance ratio 和 comm ratio 输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应已有迁移收益，但路由策略、并行切分度或副本配置还没有一起收稳。
- 项目意义：分布式推理项目最后要回答的是“值不值得迁移到真实集群”，而不是只看单次逻辑模拟有没有加速。

## 相关阅读

以下资料按“分布式推理架构 → 推理系统实现 → 多卡证据”排列，用于把本节的路由、makespan 和通信成本连接到真实部署。

- [Megatron-LM 论文：大规模 Transformer 并行训练与推理基础](https://arxiv.org/abs/1909.08053)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [PyTorch Distributed 官方文档](https://pytorch.org/docs/stable/distributed.html)
- [79 分布式并行基准](./79_Distributed_Parallel_Benchmark.md)
- [80 MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
