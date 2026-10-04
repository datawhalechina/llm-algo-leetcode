# 79. Distributed Parallel Benchmark | 训练状态并行基准

**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `分布式`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节沿训练状态这条主线比较 DDP、FSDP 与 ZeRO：参数、梯度和优化器状态怎样驻留在各个 rank，状态分片节省的显存是否值得新增的通信与重组代价。你将先建立每 rank 状态账本，再用显存、吞吐和通信门槛筛选候选，最后形成可复测的训练并行决策。

**关键词：** `distributed training`, `benchmark`, `parallelism`

---

## 前置阅读

**导语：** 本项目以训练状态驻留为判断起点。先理解 ZeRO/FSDP 的分片对象和底层通信，再比较真实训练中的显存与扩展效率。
- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](./27_ZeRO_Optimizer_Sim.md)
- [46. Communication Profiling with NCCL | NCCL 通信分析](./46_Communication_Profiling_with_NCCL.md)
- [P1: 05. Communication Topologies | 通信拓扑与分布式基石](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)


---
### Step 1：从复制状态走向分片状态

数据并行首先复制模型和训练状态。DDP 保留完整参数、梯度与优化器状态；ZeRO/FSDP 逐步分片优化器状态、梯度和参数。world size 增大时，常驻状态下降，但参数临时聚合、通信 buffer 和 activation 仍会参与峰值。

| 策略 | 参数 | 梯度 | 优化器状态 | 关键通信 |
| --- | --- | --- | --- | --- |
| DDP | 复制 | 复制 | 复制 | 梯度 All-Reduce |
| ZeRO-1 | 复制 | 复制 | 分片 | 梯度归约与参数同步 |
| ZeRO-2 | 复制 | 分片 | 分片 | Reduce-Scatter 与参数同步 |
| ZeRO-3 / FSDP full shard | 分片 | 分片 | 分片 | 参数 All-Gather、梯度归约与 reshard |

### Step 2：沿训练生命周期识别峰值与通信

状态分片改变的不只是静态字节数。forward 前的参数聚合、backward 中的梯度归约、optimizer step 的分片更新以及 checkpoint 重组，会在不同时间形成临时峰值和同步点。

| 生命周期位置 | 主要状态 | 应观察的代价 |
| --- | --- | --- |
| 加载与初始化 | 参数、主机缓冲区 | 启动峰值、rank 0 放大 |
| forward / backward | activation、聚合参数、梯度 bucket | peak memory、All-Gather / Reduce-Scatter |
| optimizer step | 参数、梯度、优化器状态 | step time、吞吐、同步等待 |
| checkpoint | full / local / sharded state dict | 保存峰值、产物大小、恢复兼容性 |

### Step 3：用门槛筛选训练状态策略

候选首先要满足单 rank 显存上限和正确性要求，再比较吞吐与通信占比。显存最低的策略不一定最好：更高阶段的状态分片可能用频繁聚合换取容量，因此应保留 `peak_memory_mb`、`throughput`、`step_time_ms`、`communication_ratio` 和 failure。

### Step 4：实现状态账本、准入与选型

题目区把项目拆成三个机制责任：计算不同阶段的每 rank 常驻训练状态；按显存、吞吐和通信门槛标记候选是否准入；在合格候选中优先选择吞吐更高、通信占比更低的策略。

| 函数 | 机制责任 | 重点测试 |
| --- | --- | --- |
| `training_state_ledger` | 按策略计算参数、梯度和优化器状态的每 rank 驻留量 | world size、未知策略、分片顺序 |
| `evaluate_candidate` | 应用显存、吞吐、通信与 failure 门槛 | 临界值、失败运行、多个不满足原因 |
| `select_training_state_strategy` | 从合格候选中形成策略结论 | 无候选、吞吐排序、通信 tie-break |


```python
from copy import deepcopy

```


```python
def training_state_ledger(parameter_mb, gradient_mb, optimizer_mb, world_size, strategy):
    """计算指定训练状态策略下的每 rank 常驻状态账本。"""
    if world_size < 1:
        raise ValueError('world_size 必须至少为 1')

    shard_factors = {
        'ddp': (1, 1, 1),
        'zero1': (1, 1, world_size),
        'zero2': (1, world_size, world_size),
        'zero3': (world_size, world_size, world_size),
        'fsdp_full_shard': (world_size, world_size, world_size),
    }
    if strategy not in shard_factors:
        raise ValueError(f'未知训练状态策略：{strategy}')

    # TODO 1：训练状态分片机制——分别计算参数、梯度和优化器状态的每 rank 驻留量。
    # parameter_factor, gradient_factor, optimizer_factor = ???
    # parameter_per_rank_mb = ???
    # gradient_per_rank_mb = ???
    # optimizer_per_rank_mb = ???
    resident_state_mb = parameter_per_rank_mb + gradient_per_rank_mb + optimizer_per_rank_mb
    return {
        'strategy': strategy,
        'parameter_per_rank_mb': parameter_per_rank_mb,
        'gradient_per_rank_mb': gradient_per_rank_mb,
        'optimizer_per_rank_mb': optimizer_per_rank_mb,
        'resident_state_mb': resident_state_mb,
    }


def evaluate_candidate(metrics, memory_limit_mb, min_throughput, max_communication_ratio):
    """按容量、吞吐、通信和运行状态判断候选是否可以进入决策。"""
    result = deepcopy(metrics)
    # TODO 2：候选准入机制——逐项检查 failure、显存、吞吐与通信占比。
    # failures = ???
    result['eligible'] = not failures
    result['gate_failures'] = failures
    return result


def select_training_state_strategy(candidates):
    """在通过准入门槛的候选中选择训练状态策略。"""
    eligible = [item for item in candidates if item['eligible']]
    if not eligible:
        return {'decision': 'reject', 'strategy': None, 'reason': '没有候选通过容量、吞吐与通信门槛'}

    # TODO 3：策略选择机制——优先吞吐，其次通信占比，再比较峰值显存。
    # selected = ???
    return {
        'decision': 'accept',
        'strategy': selected['strategy'],
        'reason': '候选通过准入门槛，并在吞吐、通信占比和峰值显存排序中最优',
    }

```

### 测试


```python
def test_training_state_ledger():
    ddp = training_state_ledger(400, 400, 800, world_size=4, strategy='ddp')
    zero2 = training_state_ledger(400, 400, 800, world_size=4, strategy='zero2')
    zero3 = training_state_ledger(400, 400, 800, world_size=4, strategy='zero3')
    assert ddp['resident_state_mb'] == 1600
    assert zero2['resident_state_mb'] == 700
    assert zero3['resident_state_mb'] == 400
    assert ddp['resident_state_mb'] > zero2['resident_state_mb'] > zero3['resident_state_mb']
    for kwargs in (
        {'world_size': 0, 'strategy': 'ddp'},
        {'world_size': 4, 'strategy': 'pipeline'},
    ):
        try:
            training_state_ledger(400, 400, 800, **kwargs)
            raise AssertionError('非法 world size 或未知策略必须被拒绝')
        except ValueError:
            pass


def test_candidate_gate():
    metrics = {'strategy': 'zero2', 'peak_memory_mb': 9000, 'throughput': 110, 'communication_ratio': 0.22, 'failure': None}
    accepted = evaluate_candidate(metrics, memory_limit_mb=10000, min_throughput=100, max_communication_ratio=0.25)
    rejected = evaluate_candidate(metrics, memory_limit_mb=8000, min_throughput=120, max_communication_ratio=0.20)
    assert accepted['eligible'] is True
    assert rejected['eligible'] is False
    assert set(rejected['gate_failures']) == {'memory', 'throughput', 'communication'}
    runtime_failed = evaluate_candidate({**metrics, 'failure': 'oom'}, 10000, 100, 0.25)
    assert runtime_failed['eligible'] is False
    assert runtime_failed['gate_failures'] == ['runtime']


def test_strategy_selection():
    candidates = [
        {'strategy': 'ddp', 'eligible': False, 'throughput': 130, 'communication_ratio': 0.12, 'peak_memory_mb': 15000},
        {'strategy': 'zero2', 'eligible': True, 'throughput': 112, 'communication_ratio': 0.20, 'peak_memory_mb': 9000},
        {'strategy': 'zero3', 'eligible': True, 'throughput': 105, 'communication_ratio': 0.31, 'peak_memory_mb': 7000},
    ]
    decision = select_training_state_strategy(candidates)
    assert decision['strategy'] == 'zero2'
    assert select_training_state_strategy([])['decision'] == 'reject'


def run_training_state_tests():
    try:
        test_training_state_ledger()
        test_candidate_gate()
        test_strategy_selection()
        print("✅ 训练状态账本、候选准入与策略选择通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


run_training_state_tests()

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
def training_state_ledger(parameter_mb, gradient_mb, optimizer_mb, world_size, strategy):
    """计算指定训练状态策略下的每 rank 常驻状态账本。"""
    if world_size < 1:
        raise ValueError('world_size 必须至少为 1')
    shard_factors = {
        'ddp': (1, 1, 1),
        'zero1': (1, 1, world_size),
        'zero2': (1, world_size, world_size),
        'zero3': (world_size, world_size, world_size),
        'fsdp_full_shard': (world_size, world_size, world_size),
    }
    if strategy not in shard_factors:
        raise ValueError(f'未知训练状态策略：{strategy}')
    # TODO 1：训练状态分片机制。
    parameter_factor, gradient_factor, optimizer_factor = shard_factors[strategy]
    parameter_per_rank_mb = parameter_mb / parameter_factor
    gradient_per_rank_mb = gradient_mb / gradient_factor
    optimizer_per_rank_mb = optimizer_mb / optimizer_factor
    resident_state_mb = parameter_per_rank_mb + gradient_per_rank_mb + optimizer_per_rank_mb
    return {
        'strategy': strategy,
        'parameter_per_rank_mb': parameter_per_rank_mb,
        'gradient_per_rank_mb': gradient_per_rank_mb,
        'optimizer_per_rank_mb': optimizer_per_rank_mb,
        'resident_state_mb': resident_state_mb,
    }


def evaluate_candidate(metrics, memory_limit_mb, min_throughput, max_communication_ratio):
    """按容量、吞吐、通信和运行状态判断候选是否可以进入决策。"""
    result = deepcopy(metrics)
    # TODO 2：候选准入机制。
    failures = []
    if metrics.get('failure'):
        failures.append('runtime')
    if metrics['peak_memory_mb'] > memory_limit_mb:
        failures.append('memory')
    if metrics['throughput'] < min_throughput:
        failures.append('throughput')
    if metrics['communication_ratio'] > max_communication_ratio:
        failures.append('communication')
    result['eligible'] = not failures
    result['gate_failures'] = failures
    return result


def select_training_state_strategy(candidates):
    """在通过准入门槛的候选中选择训练状态策略。"""
    eligible = [item for item in candidates if item['eligible']]
    if not eligible:
        return {'decision': 'reject', 'strategy': None, 'reason': '没有候选通过容量、吞吐与通信门槛'}
    # TODO 3：策略选择机制。
    selected = sorted(
        eligible,
        key=lambda item: (-item['throughput'], item['communication_ratio'], item['peak_memory_mb']),
    )[0]
    return {
        'decision': 'accept',
        'strategy': selected['strategy'],
        'reason': '候选通过准入门槛，并在吞吐、通信占比和峰值显存排序中最优',
    }

```

### Step 5（可选）：GPU / 多卡并行 benchmark

#### 5.1 环境与固定 workload

本项目固定模型结构、global batch、micro-batch、序列长度、dtype、GPU 数与重复次数，再比较 DDP 与不同训练状态分片阶段。G0、G1、G2 的策略和 JSON 路径在 5.3 集中声明。

| 类别 | 固定并记录的字段 | 用途 |
| --- | --- | --- |
| 运行环境 | GPU 数、PyTorch / CUDA、NCCL、互联拓扑 | 解释通信与每 rank 显存 |
| workload | 模型宽度 / 层数、global / micro batch、序列长度、dtype、warmup、repeats | 让策略结果可比较 |
| 对照 | G0 baseline、G1 单策略候选、G2 可选扩展 | 逐组解释策略带来的变化 |
| 输出 | `RESULT_DIR`、每组 JSON、`summary.json` | 保留复测、failure 与决策 |

```python
# 5.1：默认关闭；只比较 DDP、FSDP/ZeRO 训练状态策略。
RUN_GPU_EXPERIMENT = False
WORLD_SIZE = 2
MODEL_WIDTH = 1024
MODEL_LAYERS = 8
MICRO_BATCH = 1
GRADIENT_ACCUMULATION = 2
SEQ_LEN = 256
DTYPE = 'bf16'
WARMUP = 2
REPEATS = 5
RESULT_DIR = 'benchmarks/results/79_distributed_parallel'
ZERO_STRATEGIES = ['ddp', 'zero1', 'zero2', 'zero3']
SUMMARY_PATH = f'{RESULT_DIR}/summary.json'
print({'run': RUN_GPU_EXPERIMENT, 'world_size': WORLD_SIZE, 'strategies': ZERO_STRATEGIES, 'summary': SUMMARY_PATH})

```

#### 5.2 多卡环境预检

开启实验前检查 CUDA、可见 GPU 数、NCCL 与 `torchrun`。预检结果帮助你确认后续记录来自可启动的多卡环境。

```python
# 5.2：默认关闭；开启时检查运行多卡实验所需条件。
GPU_READY = False
if RUN_GPU_EXPERIMENT:
    import shutil
    import torch
    GPU_READY = torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE and shutil.which('torchrun') is not None
    print({'cuda': torch.cuda.is_available(), 'visible_gpus': torch.cuda.device_count(), 'world_size': WORLD_SIZE, 'torchrun': bool(shutil.which('torchrun')), 'ready': GPU_READY})
else:
    print({'run': False, 'required_gpus': WORLD_SIZE, 'ready': GPU_READY})

```

#### 5.3 配置实验组与 JSON 路径

G0 是 DDP 复制基线，G1 比较优化器与梯度分片，G2 比较参数也参与分片的 ZeRO-3/FSDP 路径。每种策略保留独立 source JSON，再由项目汇总生成 `summary.json`。

```python
# 5.3：实验组与预期 source；实际执行在 5.4。
EXPERIMENT_GROUPS = {
    'G0': {'purpose': 'ddp_baseline', 'sources': ['zero_ddp.json']},
    'G1': {'purpose': 'optimizer_and_gradient_sharding', 'sources': ['zero_zero1.json', 'zero_zero2.json']},
    'G2': {'purpose': 'full_state_sharding', 'sources': ['zero_zero3.json']},
}
print({'groups': EXPERIMENT_GROUPS, 'result_dir': RESULT_DIR})

```

#### 5.4 执行并保存 JSON

依次运行 DDP 与 ZeRO 各阶段，在相同 workload 下保存每 rank 显存、step time、吞吐和通信证据，再汇总 source JSON。

```python
# 5.4：运行机制实验并汇总证据；默认关闭。
import subprocess
from pathlib import Path

if RUN_GPU_EXPERIMENT and GPU_READY:
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    result_paths = []
    for strategy in ZERO_STRATEGIES:
        output = Path(RESULT_DIR) / f'zero_{strategy}.json'
        command = ['torchrun', '--standalone', f'--nproc_per_node={WORLD_SIZE}', 'tools/run_zero_stage_benchmark.py', '--strategy', strategy, '--output', str(output), '--width', str(MODEL_WIDTH), '--layers', str(MODEL_LAYERS), '--micro-batch', str(MICRO_BATCH), '--gradient-accumulation', str(GRADIENT_ACCUMULATION), '--seq-len', str(SEQ_LEN), '--warmup', str(WARMUP), '--repeats', str(REPEATS), '--dtype', DTYPE]
        print(' '.join(command))
        subprocess.run(command, check=False)
        result_paths.append(str(output))
    summary_command = ['python', 'tools/aggregate_distributed_evidence.py', '--project', '79', '--output', SUMMARY_PATH] + sum((['--input', path] for path in result_paths), [])
    subprocess.run(summary_command, check=True)
elif RUN_GPU_EXPERIMENT:
    print('多卡预检未通过；请先检查 5.2 的 CUDA、GPU 数和 torchrun 输出。')
else:
    print('GPU benchmark 默认关闭。')

```

#### 5.5 读取结果与记录证据

读取项目汇总后，先确认每个 source 的运行状态和统一 workload，再把策略收益与通信代价放在同一张记录中。

```python
# 5.5：读取项目汇总。每个 source 的 evidence level 会保留在 sources 中。
import json
from pathlib import Path

if Path(SUMMARY_PATH).exists():
    result = json.loads(Path(SUMMARY_PATH).read_text())
    required = {'strategy', 'workload', 'hardware', 'metrics', 'evidence_level', 'failure', 'decision'}
    missing = required - set(result)
    if missing:
        raise ValueError(f'结果 JSON 缺少项目证据字段：{sorted(missing)}')
    print({'decision': result['decision'], 'failure': result['failure'], 'sources': [(item['path'], item['evidence_level'], item['decision']) for item in result.get('sources', [])]})
else:
    print(f'尚无项目汇总结果：{SUMMARY_PATH}')

```

#### 5.6 解释结果与形成决策

| 读取项 | 判断与下一步 |
| --- | --- |
| 运行状态 | `failure` 非空时先记录启动、OOM 或通信问题，再调整实验条件 |
| 可比口径 | 对照 `workload / hardware / strategy`，确认策略变化来自预期实验组 |
| 核心指标 | 联合查看每 rank peak memory、最慢 rank step time、throughput 与 communication time |
| 项目决策 | 收益与训练压力匹配时 `accept`；需要切分或 batch 调整时 `tune`；运行失败或收益不足时 `reject` |
### 解析

**1. TODO 1：训练状态账本**

DDP 复制三类训练状态；ZeRO-1 先分片优化器状态，ZeRO-2 再分片梯度，ZeRO-3/FSDP full shard 继续分片参数。账本描述常驻状态，不把临时参数聚合、activation 或通信 buffer 错算成已经消失。

**2. TODO 2：候选准入**

候选必须同时通过运行状态、单 rank 显存、最低吞吐与最高通信占比门槛。保留全部失败原因，可以区分 OOM、吞吐不足和通信暴露，而不是只返回一个布尔值。

**3. TODO 3：策略选择**

先排除未通过门槛的策略，再按吞吐、通信占比和峰值显存排序。这个顺序表达的是当前项目偏好；改变业务目标时应显式改变排序规则，而不是修改测量结果。

## 相关阅读

以下资料按“并行训练机制 → 通信实现 → 多卡项目验证”排列，用于把本节的显存、吞吐和通信证据连接到真实多卡环境。

- [Megatron-LM 论文：高效大规模 Transformer 训练](https://arxiv.org/abs/1909.08053)
- [ZeRO 论文：Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [PyTorch Distributed 官方文档](https://pytorch.org/docs/stable/distributed.html)
- [NCCL 官方仓库](https://github.com/NVIDIA/nccl)
- [80 MoE 专家并行基准](./80_MoE_Expert_Parallel_Benchmark.md)
- [81 分布式推理项目](./81_Distributed_Inference_Project.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
