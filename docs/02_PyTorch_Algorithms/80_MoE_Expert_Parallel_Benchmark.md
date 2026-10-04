# 80. MoE Expert Parallel Benchmark | MoE 专家并行基准
**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `MoE`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节是通信与并行的 MoE benchmark，也为模型架构演进 Task4 提供系统专项证据。它比较三类专家路由：本地路由、均衡跨卡路由和偏斜跨卡路由。固定 expert 数量、token workload 与 router 规则后，观察吞吐、all-to-all 通信量、expert 负载分布、overflow 与训练稳定性如何共同影响专家并行选择。

**关键词：** `MoE`, `expert parallel`, `communication`, `imbalance`, `delivery`

---
## 前置阅读

**导语：** 先理解 MoE 路由、负载均衡与专家并行数据流，再比较本地、均衡跨卡和偏斜跨卡三种路由形态。
- [06. MoE Router | MoE 路由](./06_MoE_Router.md)
- [07. MoE Load Balancing Loss | MoE 负载均衡损失](./07_MoE_Load_Balancing_Loss.md)
- [47. MoE Expert Parallel | MoE 专家并行](./47_MoE_Expert_Parallel.md)
- [46. Communication Profiling with NCCL | NCCL 通信分析](./46_Communication_Profiling_with_NCCL.md)

---
### Step 1：从 Router 选择到跨 rank 专家执行

06、07 分别回答 token 选谁、为什么不能长期偏向少数专家；当选中的专家位于不同 rank 时，token 还要先 dispatch、完成 expert 计算、再 combine 回原顺序。本节比较本地命中、均衡跨卡与偏斜跨卡三种路由形态，观察同一 workload 如何产生不同系统结果。

| 路由形态 | token 主要去向 | 先出现的风险 | 需要记录的证据 |
|---|---|---|---|
| local | 本 rank 的专家 | 局部容量不足 | overflow、专家负载、吞吐 |
| cross-rank balanced | 多个 rank 均匀接收 token | all-to-all 固定开销 | dispatch / combine 字节、最大 rank 延迟 |
| cross-rank skewed | 少数远端专家接收大量 token | 热点、长尾与 overflow 叠加 | 最大负载、overflow、P95 step time |

### Step 2：容量与 overflow 如何改变有效工作量

每个 expert 在一个 step 中只能接收有限 token。Router 选中某 expert 的次数超过它的 capacity 后，额外 token 会被丢弃、回退或延迟处理；因此“路由更均匀”不仅是一个训练 loss 的目标，也会改变真正完成的 token 数。

| 量 | 计算口径 | 它解释什么 |
|---|---|---|
| assigned tokens | 路由到该 expert 的 token 数 | 路由的实际分配 |
| capacity | 单 expert 在当前 step 的上限 | 可被正常执行的工作量 |
| overflow | `max(assigned - capacity, 0)` | 未被当前 expert 正常处理的 token |
| effective tokens | `assigned - overflow` 的总和 | 可用于计算有效吞吐的 token 数 |

### Step 3：把通信、负载与吞吐放到同一对照中

跨 rank 路由带来 dispatch 与 combine 两次状态移动。均衡路由可能增加固定通信，却避免单 expert 热点；偏斜路由即使平均通信量相近，也可能因最长 rank 延迟和 overflow 让 step 变慢。比较时要把有效 token 吞吐、通信时间、最大负载和稳定性一起读。

偏斜路由的原始吞吐可能更高，但 overflow 会减少真正完成的 token。项目比较应使用有效吞吐，并保留最慢 rank、dispatch / combine 与稳定性，避免把丢弃工作换来的速度误判为收益。
#### 项目证据如何衔接

本页把机制页的观察量转成项目证据：06 提供完整概率与 Top-K 分配，07 提供负载均衡的训练信号，46/47 提供通信与 dispatch 的测量方式。项目 baseline 来自同一 EP workload 的本地路由，而不是训练状态并行项目。

| 输入页 | 带入本项目的量 | 在 80 中如何使用 |
|---|---|---|
| 06 Router | full probability、Top-K 分配 | 解释 token 为什么去某些 expert |
| 07 Load balancing | 平均偏好、分配比例 | 判断拥塞是否源于路由偏斜 |
| 46 / 47 | 通信时间、dispatch / combine | 判断跨 rank 路由的代价 |
| local EP baseline | 本地 token、容量与吞吐 | 与均衡/偏斜跨 rank 路由保持同一 workload |

### Step 4：CPU 代码练习——汇总多策略并作出项目判断

题目区将本地、均衡跨卡和偏斜跨卡的运行记录作为输入：先汇总 workload、容量与 overflow，再与 baseline 对照通信和有效吞吐，最后给出 `accept / tune / reject`。测试覆盖收益、overflow 和稳定性三类相互制约的条件。

| TODO | 机制责任 | 测试观察 |
|---|---|---|
| TODO 1 | 汇总 token、通信与 overflow | 有效工作量和最高吞吐候选 |
| TODO 2 | 计算吞吐、通信、负载与 overflow 的变化 | 收益是否被系统代价抵消 |
| TODO 3 | 按容量、稳定性和收益作出决策 | `accept / tune / reject` 与下一步 |


```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：运行摘要、baseline 对比、项目判断
# 目标：把容量、overflow、通信和有效吞吐整理成 MoE benchmark 报告

def summarize_moe_parallel_runs(runs: list[dict[str, float]]) -> dict[str, object]:
    """TODO 1：汇总多种路由形态的有效工作量与系统代价。"""
    # total_effective_tokens = ???
    # mean_comm_ms = ???
    # max_overflow = ???
    # best_effective_throughput_run = ???
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_moe_parallel_to_baseline(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    """TODO 2：计算 candidate 相对 baseline 的收益和代价。"""
    # effective_throughput_gain = ???
    # comm_delta_ms = ???
    # overflow_delta = ???
    # imbalance_delta = ???
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_moe_parallel_run(
    baseline: dict[str, float],
    candidate: dict[str, float],
    max_imbalance: float,
    min_stability: float,
    max_overflow: float = 0.0,
    max_comm_increase_ms: float = 12.0,
) -> dict[str, object]:
    """TODO 3：按有效吞吐、容量、通信和稳定性形成项目决策。"""
    # capacity_ok = ???
    # communication_ok = ???
    # decision = ???
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_moe_parallel_benchmark_template():
    baseline = {'name': 'local', 'comm_ms': 40, 'imbalance': 0.25, 'effective_throughput': 100, 'stability': 0.82, 'overflow': 0, 'effective_tokens': 1000}
    candidate = {'name': 'cross_rank_balanced', 'comm_ms': 48, 'imbalance': 0.12, 'effective_throughput': 128, 'stability': 0.80, 'overflow': 0, 'effective_tokens': 1000}
    skewed = {'name': 'cross_rank_skewed', 'comm_ms': 47, 'imbalance': 0.72, 'effective_throughput': 132, 'stability': 0.80, 'overflow': 280, 'effective_tokens': 720}
    summary = summarize_moe_parallel_runs([baseline, candidate, skewed])
    assert summary['run_count'] == 3
    assert summary['best_effective_throughput_run'] == 'cross_rank_skewed'
    assert summary['total_effective_tokens'] == 2720
    assert summary['max_overflow'] == 280
    comparison = compare_moe_parallel_to_baseline(baseline, candidate)
    assert comparison['comm_delta_ms'] == 8
    assert comparison['imbalance_delta'] == -0.13
    assert comparison['effective_throughput_gain'] == 28
    assert comparison['overflow_delta'] == 0
    assert comparison['stability_delta'] == -0.02
    decision = recommend_moe_parallel_run(baseline, candidate, max_imbalance=0.15, min_stability=0.78)
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_cluster_eval'
    skewed_decision = recommend_moe_parallel_run(baseline, skewed, max_imbalance=0.15, min_stability=0.78)
    assert skewed_decision['decision'] == 'tune'


test_moe_parallel_benchmark_template()
print('测试通过：MoE 专家并行基准模板可以工作。')

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
def summarize_moe_parallel_runs(runs: list[dict[str, float]]) -> dict[str, object]:
    """TODO 1：汇总多种路由形态的有效工作量与系统代价。"""
    if not runs:
        return {'run_count': 0, 'total_effective_tokens': 0, 'mean_comm_ms': 0.0, 'max_overflow': 0.0, 'best_effective_throughput_run': None}
    best = max(runs, key=lambda item: item.get('effective_throughput', item.get('throughput', 0.0)))
    return {
        'run_count': len(runs),
        'total_effective_tokens': sum(item.get('effective_tokens', item.get('tokens', 0.0)) for item in runs),
        'mean_comm_ms': sum(item.get('comm_ms', 0.0) for item in runs) / len(runs),
        'max_overflow': max(item.get('overflow', 0.0) for item in runs),
        'best_effective_throughput_run': best.get('name', 'run'),
    }


def compare_moe_parallel_to_baseline(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        'comm_delta_ms': candidate.get('comm_ms', 0.0) - baseline.get('comm_ms', 0.0),
        'imbalance_delta': round(candidate.get('imbalance', 0.0) - baseline.get('imbalance', 0.0), 4),
        'effective_throughput_gain': candidate.get('effective_throughput', candidate.get('throughput', 0.0)) - baseline.get('effective_throughput', baseline.get('throughput', 0.0)),
        'overflow_delta': candidate.get('overflow', 0.0) - baseline.get('overflow', 0.0),
        'stability_delta': round(candidate.get('stability', 0.0) - baseline.get('stability', 0.0), 4),
    }


def recommend_moe_parallel_run(
    baseline: dict[str, float],
    candidate: dict[str, float],
    max_imbalance: float,
    min_stability: float,
    max_overflow: float = 0.0,
    max_comm_increase_ms: float = 12.0,
) -> dict[str, object]:
    comparison = compare_moe_parallel_to_baseline(baseline, candidate)
    if (
        comparison['effective_throughput_gain'] > 0
        and candidate.get('imbalance', 10**9) <= max_imbalance
        and candidate.get('stability', -10**9) >= min_stability
        and candidate.get('overflow', 10**9) <= max_overflow
        and comparison['comm_delta_ms'] <= max_comm_increase_ms
    ):
        return {'decision': 'accept', 'reason': '吞吐收益、负载均衡和稳定性都达标', 'next_action': 'promote_to_cluster_eval'}
    if comparison['effective_throughput_gain'] > 0 and candidate.get('stability', -10**9) >= min_stability:
        return {'decision': 'tune', 'reason': '有效吞吐有收益，但容量、负载或通信代价仍需调整', 'next_action': 'refine_router_capacity_or_placement'}
    return {'decision': 'reject', 'reason': '收益不足或训练稳定性不达标', 'next_action': 'fallback_to_parallel_baseline'}

```

### Step 5（可选）：GPU / 多卡 Expert Parallel 机制基准

#### 5.1 环境、输入与固定条件

固定 token 数、hidden size、rank-local expert 计算、dtype、GPU 数、warmup 与 repeats。基准实际执行 All-to-All dispatch、本地 expert GEMM 与反向 All-to-All combine，并比较均匀和偏斜路由。

| 固定对象 | 本节记录 | 作用 |
| --- | --- | --- |
| workload | token 数、hidden size、dtype | 保证两种路由接收相同工作量 |
| distributed environment | world size、GPU、NCCL、PyTorch | 解释通信路径和 rank 差异 |
| execution | rank-local expert、warmup、repeats | 让最慢 rank 时间可复测 |
| output | source JSON、summary JSON | 保留失败、证据等级和项目决策 |

```python
# 5.1：默认关闭；真实 EP 机制基准至少需要两张 CUDA GPU。
RUN_GPU_EXPERIMENT = False
WORLD_SIZE = 2
EXPERTS_PER_RANK = 1  # 每个 rank 持有一个本地 expert；项目 backend 可扩展多个 expert。
TOKENS_PER_RANK = 256
HIDDEN_SIZE = 1024
DTYPE = 'bf16'
WARMUP = 2
REPEATS = 5
RESULT_DIR = 'benchmarks/results/80_moe_dispatch'
ROUTE_MODES = ['balanced', 'skewed']  # 只改变路由负载分布。
SUMMARY_PATH = f'{RESULT_DIR}/summary.json'
print({'run': RUN_GPU_EXPERIMENT, 'world_size': WORLD_SIZE, 'experts_per_rank': EXPERTS_PER_RANK, 'route_modes': ROUTE_MODES, 'summary': SUMMARY_PATH})

```

#### 5.2 多卡环境预检

检查 CUDA、可见 GPU 数与 `torchrun`，再开始 all-to-all 路由实验。

```python
# 5.2：默认关闭；开启时检查多卡条件。
GPU_READY = False
if RUN_GPU_EXPERIMENT:
    import shutil, torch
    GPU_READY = torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE and shutil.which('torchrun') is not None
    print({'visible_gpus': torch.cuda.device_count(), 'world_size': WORLD_SIZE, 'ready': GPU_READY})
else:
    print({'run': False, 'required_gpus': WORLD_SIZE})

```

#### 5.3 配置实验组与 JSON 路径

G0 与 G1 运行相同的 EP 路径，只改变 token 是否集中到热点 rank。`G2` 是真实 MoE backend 的结果槽位，用于补齐 capacity、overflow 和质量证据。

```python
# 5.3：每组的 JSON 由 5.4 写入 RESULT_DIR；G2 留给真实 MoE backend。
EXPERIMENT_GROUPS = {'G0': 'balanced', 'G1': 'skewed', 'G2': 'real_moe_backend'}
print(EXPERIMENT_GROUPS)

```

#### 5.4 执行并保存 JSON

按均匀与偏斜两组路由运行完整 EP 机制 benchmark，并汇总 source JSON。

```python
# 5.4：运行均匀与偏斜 EP 对照并汇总结果；默认关闭。
import subprocess
from pathlib import Path

if RUN_GPU_EXPERIMENT and GPU_READY:
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    result_paths = []
    for route_mode in ROUTE_MODES:
        output = Path(RESULT_DIR) / f'{route_mode}.json'
        command = ['torchrun', '--standalone', f'--nproc_per_node={WORLD_SIZE}', 'tools/run_parallel_mechanism_benchmark.py', '--mode', 'ep_dispatch', '--routing', route_mode, '--output', str(output), '--tokens-per-rank', str(TOKENS_PER_RANK), '--hidden-dim', str(HIDDEN_SIZE), '--warmup', str(WARMUP), '--repeats', str(REPEATS), '--dtype', 'bfloat16' if DTYPE == 'bf16' else 'float32']
        print(' '.join(command))
        subprocess.run(command, check=False)
        result_paths.append(str(output))
    summary_command = ['python', 'tools/aggregate_distributed_evidence.py', '--project', '80', '--output', SUMMARY_PATH] + sum((['--input', path] for path in result_paths), [])
    subprocess.run(summary_command, check=True)
elif RUN_GPU_EXPERIMENT:
    print('多卡预检未通过；请先检查 5.2 输出。')
else:
    print('GPU benchmark 默认关闭。')

```

#### 5.5 读取结果与记录证据

读取汇总 JSON，先确认均匀与偏斜两组共享同一 token workload，再联合比较负载分布、最慢 rank 时间、通信字节和显存；真实 backend 结果另行补充 capacity、overflow 与质量。

```python
# 5.5：读取汇总结果；它保留两组真实 EP 机制基准的指标与 evidence level。
import json
from pathlib import Path

if Path(SUMMARY_PATH).exists():
    result = json.loads(Path(SUMMARY_PATH).read_text())
    required = {'workload', 'hardware', 'metrics', 'evidence_level', 'failure', 'decision'}
    missing = required - set(result)
    if missing:
        raise ValueError(f'结果 JSON 缺少项目证据字段：{sorted(missing)}')
    print({'decision': result['decision'], 'failure': result['failure'], 'sources': [(item['path'], item['metrics'], item['evidence_level']) for item in result.get('sources', [])]})
else:
    print(f'尚无 dispatch benchmark 结果：{SUMMARY_PATH}')

```

#### 5.6 解释结果与形成决策

| 读取项 | 判断与下一步 |
| --- | --- |
| 运行状态 | `failure` 非空时先记录启动、OOM 或 all-to-all 问题，再调整运行条件 |
| 可比口径 | 核对 token workload、world size、rank-local expert、dtype 与 route mode |
| 机制基准 | 联合比较接收 token 分布、负载偏斜、最慢 rank 时间、dispatch bytes 与峰值显存 |
| backend 复测 | 补齐 expert load、capacity、overflow、质量、稳定性与端到端有效吞吐 |
| 项目决策 | 机制基准先判断负载与通信代价；真实 backend 再以 capacity、overflow、质量和端到端收益决定 `accept / tune / reject` |
### 解析

这页现在按 `measure -> compare -> decide` 的最小 MoE 专家并行项目闭环组织，不再只是单独比较吞吐和通信代价。

#### TODO 1

- 实现方式：汇总有效 token、平均通信时间和最大 overflow，再找出有效吞吐最高的 run。
- 关键点：`best_effective_throughput_run` 只是帮助定位最值得回看的 candidate，不等于最终项目结论。
- 项目意义：先把 workload 摘要做平，后面才能在同一 expert 并行设置下比较收益与代价。

#### TODO 2

- 实现方式：统一计算 `effective_throughput_gain`、`comm_delta_ms`、`imbalance_delta`、`overflow_delta` 和 `stability_delta`。
- 关键点：有效吞吐和稳定性越高越好，通信代价、负载不均衡和 overflow 越低越好，所以指标方向必须统一。
- 项目意义：这一步把 MoE 专家并行从“技巧演示”转成“吞吐、通信和稳定性能否一起成立”的 benchmark 对比。

#### TODO 3

- 实现方式：先复用 baseline 对比结果，再按吞吐收益、imbalance 边界和稳定性输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应吞吐收益已出现，但 router、capacity factor 或通信拓扑还没有一起收稳。
- 项目意义：MoE 专家并行项目最后要回答的是“这套并行方案值不值得继续扩到真实集群”，而不是只看某个吞吐数字。

## 相关阅读

以下资料按“MoE 路由机制 → 专家并行实现 → 通信证据”排列，用于把本节的负载均衡、all-to-all 和吞吐权衡连接到真实系统。

- [Switch Transformers 论文：稀疏专家路由](https://arxiv.org/abs/2101.03961)
- [DeepSpeed-MoE 论文：Scaling Inference with MoE](https://arxiv.org/abs/2201.05596)
- [MegaBlocks 官方仓库](https://github.com/databricks/megablocks)
- [DeepSpeed 官方仓库](https://github.com/microsoft/DeepSpeed)
- [79 分布式并行基准](./79_Distributed_Parallel_Benchmark.md)
- [81 分布式推理项目](./81_Distributed_Inference_Project.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
