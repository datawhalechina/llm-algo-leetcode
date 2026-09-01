# 76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目

**难度：** Hard | **环境：** CPU 可完成正确性验证；GPU 用于策略 benchmark | **标签：** `显存优化`, `Checkpoint/Offload`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节承接 73 的 baseline，比较 checkpoint、offload 和组合方案在同一训练 workload 下的实际代价。主线报告固定模型、输入和训练口径；高压力 workload、不同 seq_len 或 dtype 属于扩展，必须单独记录。结果交给 75 做预算决策，本节不直接裁决。
**主责与复用边界：** 本项目主责是训练侧 activation 策略的同口径比较；73 提供 baseline，75 负责预算决策，74 负责 trace 解释。推理 KV Cache、量化格式和分布式切分不在本项目内重复实现。

> 运行提示：先查看[使用指南中的项目环境预检与安装说明](../docs/guide.md#项目环境预检与安装)，再打开真实 GPU 开关。CPU 路径只检查正确性；真实 GPU 路径必须先通过预检。

**关键词：** `activation`, `checkpoint`, `offload`, `memory`, `benchmark`

---
## 前置阅读

**导语：** 先完成 checkpoint、offload 等训练侧显存机制学习，并阅读 73 了解统一测量口径，再进入这个项目；本节默认你已经知道这些技巧各自怎么省显存，重点转向在同一训练任务下测量哪种方案更值。
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [43. Unified Memory Management | 统一内存管理](./43_Unified_Memory_Management.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)


## 相关阅读

**导语：** 做完这页后，先把结果交给 75 完成训练侧预算决策，再由 74 使用 profiling 对显存优化方案做端到端最终验证。
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [74. Profiling Driven End-to-End Optimization | Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)

---
### Step 1：显存策略与机制假设
先明确每种策略改变了哪一段 activation 生命周期，以及代价转移到了哪里。

| 方案 | forward 结束后 activation | backward 时的动作 | 主要代价 |
|:---|:---|:---|:---|
| baseline | 留在 GPU，等待 backward | 直接读取 | 显存驻留 |
| checkpoint | 只保留检查点边界 | 重新执行部分 forward | 额外计算 |
| offload | 保存到 CPU | backward 前搬回 GPU | 带宽、同步和 CPU 内存 |
| hybrid | 同时使用两种机制 | 重算并搬运部分状态 | 综合代价 |

这些是待验证的机制预期，不是实测结论。
### Step 2：比较口径与实验协议
固定比较条件，只改变显存策略；不同 workload 或 dtype 必须另存报告。

| 项目 | 固定内容 | 目的 |
|:---|:---|:---|
| 模型与输入 | 同一模型、随机输入、标签和初始化 seed | 保证策略间训练任务一致 |
| 训练配置 | optimizer、学习率、训练步数 | 排除训练过程差异 |
| workload | batch、seq_len、warmup、iters、seed | 保证压力和统计口径一致 |
| 候选策略 | baseline、checkpoint、offload、hybrid | 形成可比较集合 |

先用 CPU 检查 loss、gradient 和 step 语义，再用 GPU 采集真实显存、吞吐和 OOM。固定随机输入只保证策略间可比，不代表真实数据集上的训练质量。Gradient Accumulation 不放入本节核心四策略；如果要比较，必须固定 effective batch 并另存扩展报告。
### Step 3：指标与项目判定
显存最低的方案不一定最值得采用；必须同时检查容量、速度和训练状态。

| 指标 | 含义 | 用于判断 |
|:---|:---|:---|
| peak allocated | 张量实际达到的显存峰值 | 观察显存收益 |
| peak reserved | allocator 保留的显存峰值 | 观察缓存和碎片影响 |
| step time / samples/s | 重算、搬运和同步的时间代价 | 观察吞吐损失 |
| val loss / OOM | 训练质量和可运行性 | 过滤不可接受方案 |

报告中可用 `memory_saving = baseline_peak - candidate_peak` 和 `throughput_ratio = candidate_throughput / baseline_throughput` 表达取舍。`accept` 还要求候选未 OOM、显存和吞吐满足预算、质量不越过阈值，并达到有效显存收益；否则根据问题进入 `tune` 或 `reject`。最终预算裁决交给 75。
### Step 4：动手实战（CPU-first）

**要求：** 请补全下方三个函数：检查预算与质量阈值、汇总 baseline / checkpoint / offload / hybrid 候选，并输出 `accept / tune / reject`。先运行 CPU 正确性测试，再将同一比较口径用于 Step 5 的真实 GPU benchmark。

```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：预算检查、候选汇总、项目结论
# 目标：把 baseline / checkpoint / offload / hybrid 的方案比较收束成一份项目报告。
# CPU 题目区只验证预算与决策逻辑；真实策略、saved tensors、重算和搬运代价属于 GPU 实验。

def validate_strategy_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """检查显存上限、吞吐下限和质量上限是否完整且合法。

    返回 is_valid、missing_keys 和 invalid_keys；不负责判断某个策略是否最优。
    """
    # TODO 1：定义 required_budget_keys、required_quality_keys。
    # 提示：至少检查 memory_cap_mb、min_samples_per_s、max_val_loss。
    #       变量示例：budget_missing = ???、quality_missing = ???、invalid_keys = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_memory_strategy_candidates(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """按预算和质量门槛筛选四类显存策略候选。

    返回候选数量、可行数量、可行名称、最佳候选和显存节省；OOM 或缺失指标
    不能被当作普通的零值。
    """
    # TODO 2：遍历 candidates，分别计算每个候选的可行性。
    # 提示：变量示例：memory_ok = ???、throughput_ok = ???、quality_ok = ???、
    #       is_feasible = ???；baseline_peak = ???；memory_saving = ???。
    #       checkpoint 节省 activation 驻留，offload 还要承担 CPU-GPU 搬运代价。
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
    """根据可行候选和显存收益输出 accept、tune 或 reject。

    accept 仅表示当前 workload 和预算下值得继续保留，不表示普遍有效。
    """
    # TODO 3：读取 feasible_count、best_candidate、memory_saving 和吞吐保留率。
    # 提示：变量示例：has_feasible = ???、meaningful_saving = ???、
    #       decision = ???、reason = ???、next_action = ???。
    #       没有可行候选时 reject；可行但节省不显著时 tune。
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_memory_strategy_project():
    try:
        budget = {'memory_cap_mb': 12000.0, 'min_samples_per_s': 6.0}
        quality_floor = {'max_val_loss': 1.15}
        check = validate_strategy_budget(budget, quality_floor)
        assert check['is_valid'] is True, '预算检查应通过'
        assert check['missing_keys'] == [], '完整预算不应缺字段'
        missing = validate_strategy_budget({'memory_cap_mb': 12000.0}, quality_floor)
        assert missing['is_valid'] is False, '缺少吞吐门槛时应拒绝预算配置'
        assert 'min_samples_per_s' in missing['missing_keys'], '应指出缺少的预算字段'

        candidates = [
            {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
            {'name': 'checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.6, 'val_loss': 1.08},
            {'name': 'offload', 'peak_memory_mb': 10500.0, 'samples_per_s': 5.2, 'val_loss': 1.09},
            {'name': 'hybrid', 'peak_memory_mb': 9800.0, 'samples_per_s': 6.1, 'val_loss': 1.11},
        ]
        summary = summarize_memory_strategy_candidates(candidates, budget, quality_floor)
        assert summary['feasible_count'] == 2, '应有两个方案满足预算与质量'
        assert summary['best_candidate'] == 'hybrid', 'hybrid 应成为最省显存的可行方案'

        decision = decide_memory_strategy_project(summary)
        assert decision['decision'] == 'accept', '可行且最优的方案应被接受'

        hard_summary = summarize_memory_strategy_candidates(
            [
                {'name': 'checkpoint', 'peak_memory_mb': 13000.0, 'samples_per_s': 6.4, 'val_loss': 1.10},
                {'name': 'offload', 'peak_memory_mb': 11000.0, 'samples_per_s': 4.8, 'val_loss': 1.12},
            ],
            budget,
            quality_floor,
        )
        hard_decision = decide_memory_strategy_project(hard_summary)
        assert hard_decision['decision'] == 'reject', '没有满足预算与质量时应 reject'

        edge_summary = summarize_memory_strategy_candidates(
            [
                {'name': 'oom_candidate', 'status': 'oom'},
                {'name': 'incomplete', 'peak_memory_mb': 10000.0},
            ],
            budget,
            quality_floor,
        )
        assert edge_summary['oom_count'] == 1, 'OOM 候选应单独计数'
        assert edge_summary['invalid_count'] == 1, '缺少指标的候选应标记为 invalid'
        assert decide_memory_strategy_project(edge_summary)['decision'] == 'reject', '没有可行策略时应 reject'
        print('所有测试通过！')
    except NotImplementedError:
        print('请先完成 TODO 代码！')
        raise
    except AssertionError as e:
        print(f'测试失败: {e}')
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except Exception as e:
        print(f'发生错误: {e}')
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_memory_strategy_project()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
from typing import Dict, List
def validate_strategy_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    required_budget_keys = ['memory_cap_mb', 'min_samples_per_s']
    required_quality_keys = ['max_val_loss']
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    return {
        'is_valid': len(missing_keys) == 0,
        'missing_keys': missing_keys,
    }


def summarize_memory_strategy_candidates(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    feasible: List[Dict[str, float]] = []
    quality_failed = 0
    invalid_count = 0
    oom_count = 0

    for candidate in candidates:
        if candidate.get('status', 'ok') == 'oom':
            oom_count += 1
            continue
        memory = candidate.get('peak_memory_mb')
        throughput = candidate.get('samples_per_s')
        eval_loss = candidate.get('eval_loss', candidate.get('val_loss'))
        if not all(isinstance(value, (int, float)) for value in (memory, throughput, eval_loss)):
            invalid_count += 1
            continue
        memory_ok = memory <= budget['memory_cap_mb']
        speed_ok = throughput >= budget['min_samples_per_s']
        quality_ok = eval_loss <= quality_floor['max_val_loss']
        if not quality_ok:
            quality_failed += 1
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    feasible.sort(key=lambda x: (x['peak_memory_mb'], -x['samples_per_s'], x.get('eval_loss', x.get('val_loss'))))
    best_candidate = feasible[0]['name'] if feasible else None
    baseline = next((item for item in candidates if item.get('name') == 'baseline' and item.get('status', 'ok') == 'ok' and all(isinstance(item.get(key), (int, float)) for key in ('peak_memory_mb', 'samples_per_s'))), None)
    best = feasible[0] if feasible else None
    return {
        'candidate_count': len(candidates),
        'measured_count': len(candidates) - oom_count - invalid_count,
        'oom_count': oom_count,
        'invalid_count': invalid_count,
        'feasible_count': len(feasible),
        'best_candidate': best_candidate,
        'quality_failed_count': quality_failed,
        'feasible_names': [item['name'] for item in feasible],
        'baseline_peak_memory_mb': baseline['peak_memory_mb'] if baseline else None,
        'best_peak_memory_mb': best['peak_memory_mb'] if best else None,
        'memory_saving_mb': (baseline['peak_memory_mb'] - best['peak_memory_mb']) if baseline and best else 0.0,
        'throughput_ratio': (best['samples_per_s'] / baseline['samples_per_s']) if baseline and best else None,
    }


def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
    feasible_count = summary['feasible_count']
    best_candidate = summary['best_candidate']
    quality_failed_count = summary['quality_failed_count']

    if feasible_count == 0:
        return {
            'decision': 'reject',
            'reason': 'no_strategy_meets_budget_and_quality',
            'next_action': 'rework_checkpoint_or_offload_scope',
        }
    meaningful_memory_gain = summary.get('memory_saving_mb', 0.0) >= 512.0
    acceptable_throughput = summary.get('throughput_ratio') is None or summary.get('throughput_ratio') >= 0.70
    if best_candidate in {'checkpoint', 'offload', 'hybrid'} and meaningful_memory_gain and acceptable_throughput:
        return {
            'decision': 'accept',
            'reason': 'strategy_is_best_feasible_option',
            'next_action': 'promote_to_training_run',
        }
    if quality_failed_count > 0:
        return {
            'decision': 'tune',
            'reason': 'strategy_needs_quality_recovery',
            'next_action': 'adjust_checkpoint_granularity_or_offload_scope',
        }
    if not meaningful_memory_gain:
        return {
            'decision': 'tune',
            'reason': 'memory_saving_below_meaningful_threshold',
            'next_action': 'test_pressure_or_offload_scope',
        }
    if not acceptable_throughput:
        return {
            'reason': 'throughput_loss_exceeds_budget',
            'decision': 'tune',
            'next_action': 'reduce_checkpoint_or_offload_scope',
        }
    return {
        'decision': 'tune',
        'reason': 'baseline_still_best_under_current_budget',
        'next_action': 'revisit_strategy_mix',
    }

```

### 解析

**1. TODO 1: 检查预算与质量阈值**
- **实现方式**：先把显存上限、吞吐下限和验证损失上限检查齐，再进入方案比较。
- **关键点**：没有统一预算口径时，checkpoint / offload / hybrid 之间的比较都没有解释力。
- **项目意义**：这一步把 `76` 固定成预算约束下的显存策略对比页，而不是泛技巧列表。

**2. TODO 2: 汇总显存策略候选**
- **实现方式**：按 peak memory、samples/s 和 val loss 统一过滤候选，再选出最省显存的可行方案。
- **关键点**：显存收益只有在质量和吞吐都没有跌出边界时，才值得被保留。
- **项目意义**：这一步把 `19 / 42 / 43 / 73` 的机制与测量知识收成真正可比较的工程候选。

**3. TODO 3: 输出项目结论**
- **实现方式**：把候选可行性和最优方案统一收成 `accept / tune / reject`。
- **关键点**：项目结论必须回答“当前预算下哪种显存策略值得继续采用”，而不是只输出一个峰值显存最小值。
- **项目意义**：这一步把 `76` 收成显存优化路线中的正式策略对比项目。

## CPU 正确性检查：先验证策略不会改变训练结果

没有 GPU 时，不能验证真实 peak memory、CUDA kernel 或 CPU-GPU 搬运速度，但仍可以用一个小型网络检查策略语义：baseline、checkpoint、offload 和 hybrid 应该得到一致的 loss 与参数梯度。测试还会观察 baseline / checkpoint 的 saved tensors；`save_on_cpu` 会接管 offload 的保存钩子，因此 offload / hybrid 不伪造张量数量。这个测试验证的是 autograd 正确性，不是显存收益；真实显存结论仍由下面的 GPU benchmark 给出。

```python
import contextlib
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

VALID_STRATEGIES = {'baseline', 'checkpoint', 'offload', 'hybrid'}

def run_cpu_strategy(strategy, observe_saved_tensors=False):
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f'未知策略: {strategy}')
    torch.manual_seed(42)
    first = nn.Linear(8, 16)
    second = nn.Linear(16, 4)
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)
    offload_context = (
        torch.autograd.graph.save_on_cpu(pin_memory=False)
        if strategy in {'offload', 'hybrid'} else contextlib.nullcontext()
    )
    saved, unpacked = [], []
    def pack(tensor):
        saved.append({'shape': tuple(tensor.shape), 'numel': tensor.numel(), 'device': str(tensor.device)})
        return tensor
    def unpack(tensor):
        unpacked.append({'shape': tuple(tensor.shape), 'numel': tensor.numel(), 'device': str(tensor.device)})
        return tensor
    observe_context = (
        torch.autograd.graph.saved_tensors_hooks(pack, unpack)
        if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else contextlib.nullcontext()
    )
    with observe_context:
        with offload_context:
            if strategy in {'checkpoint', 'hybrid'}:
                hidden = checkpoint(first, x, use_reentrant=False)
            else:
                hidden = first(x)
            loss = ((second(hidden) - target) ** 2).mean()
            loss.backward()
    gradients = [first.weight.grad.detach().clone(), second.weight.grad.detach().clone()]
    observation = {
        'saved_count': len(saved) if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else None,
        'unpacked_count': len(unpacked) if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else None,
        'saved_numel': sum(item['numel'] for item in saved) if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else None,
        'note': 'save_on_cpu owns the autograd hooks; inspect offload placement with GPU profiling.' if strategy in {'offload', 'hybrid'} else 'saved_tensors_hooks observation',
    }
    return (second(hidden).detach(), loss.detach(), gradients, observation)

baseline_output, baseline_loss, baseline_grads, baseline_observation = run_cpu_strategy('baseline', observe_saved_tensors=True)
assert torch.isfinite(baseline_output).all() and torch.isfinite(baseline_loss).all()
assert baseline_observation['saved_count'] > 0
assert baseline_observation['saved_count'] == baseline_observation['unpacked_count']
print(f"baseline saved tensors: {baseline_observation['saved_count']}, saved elements: {baseline_observation['saved_numel']}")
for strategy in ('checkpoint', 'offload', 'hybrid'):
    output, loss, grads, observation = run_cpu_strategy(strategy, observe_saved_tensors=strategy == 'checkpoint')
    assert torch.isfinite(output).all() and torch.isfinite(loss).all()
    assert torch.allclose(output, baseline_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(loss, baseline_loss, atol=1e-6, rtol=1e-5)
    assert all(torch.allclose(g, b, atol=1e-6, rtol=1e-5) for g, b in zip(grads, baseline_grads))
    if strategy == 'checkpoint':
        assert observation['saved_count'] > 0
        assert observation['saved_count'] == observation['unpacked_count']
    else:
        assert observation['saved_count'] is None
    print(f'{strategy}: loss/gradient check passed')
try:
    run_cpu_strategy('unknown')
except ValueError:
    pass
else:
    raise AssertionError('未知策略应明确报错')
print('CPU correctness test passed; this does not measure real GPU memory saving.')

```

## Step 5（可选）：真实 GPU 显存策略 benchmark

本 Step 复用 73 的训练口径，在同一模型、固定输入和 FP32 + AdamW 配置下比较 baseline、activation checkpoint、CPU offload 和 hybrid。运行前会自动读取并校验 `benchmarks/results/73_real_gpu_training.json`；如果模型、batch、seq_len、dtype 或 optimizer 不一致，会要求先用相同 workload 重跑 73。提供 `smoke` 与 `pressure` 两档 workload：smoke 用于快速校验，pressure 用于提高 activation 压力。结果保存到 `benchmarks/results/76_real_gpu_memory.json`。

这里的 offload 使用 PyTorch 的 `torch.autograd.graph.save_on_cpu` 保存 backward 所需张量，重点是教学 benchmark，不等同于生产训练框架中的完整 offload 调度。`eval_loss` 只是固定随机输入上的质量代理指标，不等同于真实数据集验证结果。`REPEATS=1` 用于快速 smoke；改为 3 后会把每种策略的独立运行结果和均值一起写入报告，用于检查收益稳定性。

本节主线保持全参数训练口径：`Qwen2.5-0.5B + FP32 + AdamW`。如果要在 12GB 显存上测试更大模型或更长序列，应另设 LoRA / QLoRA 扩展实验：LoRA 主要减少可训练参数、梯度和 optimizer state，QLoRA 进一步压缩基座权重；它们不能与本节主线结果直接横向比较，但可以用于观察更大模型下 activation checkpoint / offload 的适用边界。

| 策略 | 主要换取什么 | 适用判断 |
|---|---|---|
| baseline | 不引入额外重算或搬运 | 显存有余量、优先吞吐时作为参照 |
| checkpoint | 用额外 forward 计算换 activation 驻留 | activation 主导峰值且能接受吞吐下降时优先尝试 |
| offload | 用 CPU-GPU 带宽和同步换 GPU 驻留空间 | GPU 显存成为硬约束、速度代价可接受时使用 |
| hybrid | 同时承担部分重算和搬运代价 | 单一策略不够或需要折中时再测试 |

本节的 CPU 结果证据等级是 `autograd_correctness`，GPU 结果是 `fixed_workload_strategy_benchmark`；两者都不能单独推出真实任务收敛或生产训练性能。

```python
from pathlib import Path

RUN_REAL_GPU = False  # 默认先运行 CPU 正确性检查；GPU benchmark 时显式改为 True。
DTYPE_MODE = 'fp32'  # fp32：主线；bf16：显式扩展实验，启用 CUDA autocast。
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 固定基座模型。
MODEL_SOURCE = 'auto'  # 模型来源：auto / huggingface / modelscope / local。
MODEL_CACHE_DIR = 'model_cache'  # 模型缓存目录。
BATCH_SIZE = 1  # 实际值由 WORKLOADS 覆盖；增大它会提高 activation 压力。
WORKLOAD = 'pressure'  # pressure(seq_len=768)：与 73 主线对齐；pressure_1024 是长序列扩展。
WORKLOADS = {
    'smoke': {'batch_size': 1, 'seq_len': 512, 'warmup': 2, 'iters': 5},
    'pressure': {'batch_size': 1, 'seq_len': 768, 'warmup': 2, 'iters': 5},
    'pressure_1024': {'batch_size': 1, 'seq_len': 1024, 'warmup': 2, 'iters': 5},
}
WARMUP = 2  # 预热轮数，不计入正式平均值；正式报告建议至少 2。
ITERS = 5  # 每种策略的正式测量轮数；这是教学 smoke，稳定性要求高时应增加。
SEED = 42  # 固定输入，保证策略间 workload 一致；不代表真实数据集质量。
STRATEGIES = ['baseline', 'checkpoint', 'offload', 'hybrid']  # 四种策略在同一 workload 下比较；资源不足时可先删 offload，但要记录。

# 完成 smoke 后，可改为：['baseline', 'checkpoint', 'offload', 'hybrid']
# BF16 / 长序列扩展允许先比较子集，但报告必须保留实际 strategies。
REPEATS = 1  # 每种策略的独立运行次数；正式稳定性检查建议改为 3。
MEMORY_CAP_MB = 11200.0  # 硬显存预算；需为系统和桌面进程留余量。
MIN_SAMPLES_PER_S = 1.0  # 最低吞吐；低于此值的策略判为不可行。
# None 表示按当前固定 workload 的 baseline eval_loss 自动生成质量上限
MAX_VAL_LOSS = None  # 只是质量代理门槛，不是完整任务质量门槛。
# 主线与 73 的 FP32 / pressure 报告对齐；BF16 / seq1024 扩展时显式修改这两条路径。
BASELINE_73_RELATIVE_PATH = Path('benchmarks/results/73_real_gpu_training.json')
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/76_real_gpu_memory.json')

```


```python
import json
import gc
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# 先定位仓库并加入 sys.path，再导入项目工具；避免 Colab 直接打开时找不到 tools。
PROJECT_ROOT = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
if not (PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    colab_root = Path('/content/llm-algo-leetcode')
    if (colab_root / 'tools/project_runtime.py').is_file():
        PROJECT_ROOT = colab_root
    else:
        for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
            if (candidate / 'tools/project_runtime.py').is_file():
                PROJECT_ROOT = candidate
                break
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, resolve_project_root, environment_preflight, runtime_snapshot, standard_experiment_config, standard_training_metrics, validate_training_config
from tools.training_memory_runtime import measure_training_run
from tools.memory_strategy_runtime import (
    summarize_memory_strategy_candidates,
    decide_memory_strategy_project,
)

PROJECT_ROOT = resolve_project_root(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_PATH = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
print(f'项目根目录: {PROJECT_ROOT}')
print(f'结果保存路径: {OUTPUT_PATH}')

if RUN_REAL_GPU:
    import torch
    valid_dtype_modes = {'fp32', 'bf16'}
    valid_strategies = {'baseline', 'checkpoint', 'offload', 'hybrid'}
    if DTYPE_MODE not in valid_dtype_modes:
        raise ValueError(f'DTYPE_MODE 必须是 {sorted(valid_dtype_modes)} 之一，当前为 {DTYPE_MODE!r}')
    if not STRATEGIES or len(set(STRATEGIES)) != len(STRATEGIES) or not set(STRATEGIES).issubset(valid_strategies):
        raise ValueError(f'STRATEGIES 必须是 valid_strategies 的非空子集且不能重复，当前为 {STRATEGIES!r}')
    if MEMORY_CAP_MB <= 0 or MIN_SAMPLES_PER_S < 0:
        raise ValueError('MEMORY_CAP_MB 必须大于 0，MIN_SAMPLES_PER_S 不能小于 0。')
    if REPEATS < 1:
        raise ValueError('REPEATS 必须至少为 1。')
    from tools.model_runtime import resolve_model
    from transformers import AutoConfig, AutoModelForCausalLM

    if WORKLOAD not in WORKLOADS:
        raise ValueError(f'未知 workload: {WORKLOAD}，可选值：{sorted(WORKLOADS)}')
    workload_config = WORKLOADS[WORKLOAD]
    BATCH_SIZE = workload_config['batch_size']
    SEQ_LEN = workload_config['seq_len']
    WARMUP = workload_config['warmup']
    ITERS = workload_config['iters']
    validate_training_config({'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN, 'warmup': WARMUP, 'iters': ITERS, 'seed': SEED})
    preflight = environment_preflight(torch, required_packages=('transformers',), require_gpu=True, output_path=OUTPUT_PATH)
    print({'environment_preflight': preflight})
    if not preflight['ready']:
        raise RuntimeError('环境预检未通过，请先按 next_actions 修复；没有加载模型。')
    print({'runtime': runtime_snapshot(torch)})
    if not torch.cuda.is_available():
        raise RuntimeError('RUN_REAL_GPU=True 但 CUDA 不可用。')

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda')
    baseline_report_path = PROJECT_ROOT / BASELINE_73_RELATIVE_PATH
    if not baseline_report_path.exists():
        raise FileNotFoundError(f'找不到 73 baseline：{baseline_report_path}，请先运行 73 的相同 workload。')
    baseline_report = json.loads(baseline_report_path.read_text(encoding='utf-8'))
    baseline_config = baseline_report.get('config', {})
    expected_baseline = {
        'model_id': MODEL_ID, 'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
        'dtype': 'float32', 'optimizer': 'AdamW', 'workload': WORKLOAD,
        'warmup': WARMUP, 'iters': ITERS,
    }
    if DTYPE_MODE == 'bf16':
        expected_baseline['amp_dtype'] = 'torch.bfloat16'
        if baseline_config.get('mode') != 'bf16_probe':
            raise ValueError('BF16 扩展实验需要 73 的 bf16_probe 报告。')
    mismatches = {
        key: {'baseline': baseline_config.get(key), 'current': value}
        for key, value in expected_baseline.items()
        if baseline_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f'73 与 76 的 workload 口径不一致，请先重跑 73：{mismatches}')
    model_path = resolve_model(MODEL_ID, source=MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
    print(f'模型路径: {model_path}')
    model_config = AutoConfig.from_pretrained(model_path)
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    shared_input_ids_cpu = torch.randint(
        0, model_config.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator
    )
    eval_input_ids_cpu = torch.randint(
        0, model_config.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator
    )

    def strategy_context(name):
        if name in {'offload', 'hybrid'}:
            return torch.autograd.graph.save_on_cpu(pin_memory=False)
        return nullcontext()

    def run_strategy(name):
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        model.config.use_cache = False
        if name in {'checkpoint', 'hybrid'}:
            model.gradient_checkpointing_enable()
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        input_ids = shared_input_ids_cpu.to(device)
        labels = input_ids.clone()
        eval_input_ids = eval_input_ids_cpu.to(device)
        eval_labels = eval_input_ids.clone()

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with strategy_context(name):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DTYPE_MODE == 'bf16'):
                    loss = model(input_ids=input_ids, labels=labels).loss
                loss.backward()
            optimizer.step()
            return float(loss.detach().item())

        result = measure_training_run(
            train_step, torch_module=torch, batch_size=BATCH_SIZE,
            warmup=WARMUP, iters=ITERS,
        )
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DTYPE_MODE == 'bf16'):
                eval_loss = float(model(input_ids=eval_input_ids, labels=eval_labels).loss.item())
        result = {
            'name': name,
            'status': 'ok',
            'step_time_ms': result['step_time_ms'],
            'samples_per_s': result['samples_per_s'],
            'loss': result['loss'],
            'eval_loss': round(eval_loss, 6),
            'peak_memory_mb': result['peak_mem_mb'],
            'peak_reserved_mb': result['peak_reserved_mb'],
        }
        del optimizer, model, input_ids, labels, eval_input_ids, eval_labels
        gc.collect()
        torch.cuda.empty_cache()
        return result

    raw_run_strategy = run_strategy
    def run_strategy(name):
        try:
            return raw_run_strategy(name)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            return {
                'name': name,
                'status': 'oom',
                'error': str(exc).split('\n')[0],
            }

    def run_strategy_repeated(name):
        runs = [run_strategy(name) for _ in range(REPEATS)]
        successful = [item for item in runs if item.get('status') == 'ok']
        if not successful:
            return {**runs[0], 'runs': runs}
        numeric_keys = ('step_time_ms', 'samples_per_s', 'loss', 'eval_loss', 'peak_memory_mb', 'peak_reserved_mb')
        aggregated = {
            key: round(sum(item[key] for item in successful) / len(successful), 3)
            for key in numeric_keys
        }
        status = 'ok' if len(successful) == len(runs) else 'partial_oom'
        return {**successful[0], **aggregated, 'status': status, 'runs': runs, 'successful_runs': len(successful), 'oom_runs': len(runs) - len(successful)}

    candidates = [run_strategy_repeated(name) for name in STRATEGIES]
    budget = {'memory_cap_mb': MEMORY_CAP_MB, 'min_samples_per_s': MIN_SAMPLES_PER_S}
    baseline_candidate = next((item for item in candidates if item.get('name') == 'baseline' and item.get('status', 'ok') == 'ok'), None)
    if MAX_VAL_LOSS is None and baseline_candidate is not None:
        max_eval_loss = baseline_candidate['eval_loss'] * 1.02
    elif MAX_VAL_LOSS is None:
        max_eval_loss = float('inf')
    else:
        max_eval_loss = MAX_VAL_LOSS
    quality_floor = {'max_val_loss': round(max_eval_loss, 6)}
    summary = summarize_memory_strategy_candidates(candidates, budget, quality_floor)
    decision = decide_memory_strategy_project(summary)
    result = {
        'task': 'task3_training_memory_optimization',
        'environment_preflight': preflight,
        'stage': 'activation_checkpoint_offload_benchmark',
        'source_baseline': str(baseline_report_path.relative_to(PROJECT_ROOT)),
        'baseline_validation': {'status': 'matched', 'config': expected_baseline},
        'evidence_level': 'fixed_workload_strategy_smoke' if DTYPE_MODE == 'fp32' else 'bf16_capacity_strategy_smoke',
        'config': {
            'model_id': MODEL_ID, 'workload': WORKLOAD, 'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
            'dtype': 'float32', 'amp_dtype': 'torch.bfloat16' if DTYPE_MODE == 'bf16' else None, 'mode': DTYPE_MODE, 'optimizer': 'AdamW',
            'warmup': WARMUP, 'iters': ITERS, 'strategies': STRATEGIES,
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'seed': SEED,
        },
        'budget': budget, 'quality_floor': quality_floor,
        'candidates': candidates, 'summary': summary, 'decision': decision,
    }
    result['experiment'] = standard_experiment_config(result['config'])
    result['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in candidates}
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print('跳过真实 GPU benchmark：保持 CPU-first 模式。')

```

## 实测记录：本地 RTX 5070 Ti GPU

下面记录 76 的真实 GPU benchmark，作为本节的可读实验报告；原始 JSON 由 Step 5 保存到 `benchmarks/results/76_real_gpu_memory.json`。

### 环境与统一配置

| 项目 | 配置 |
|:---|:---|
| Linux 内核 | `6.8.0-138-generic` |
| GPU / 显存 | NVIDIA GeForce RTX 5070 Ti Laptop GPU / 12227 MiB |
| NVIDIA 驱动 | `570.211.01` |
| PyTorch / CUDA | `2.11.0+cu128` / `12.8` |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 训练配置 | FP32、AdamW、batch=1、warmup=2、iters=5、seed=42 |
| Pressure workload | seq_len=768 |
| 预算 | 显存上限 11200 MiB，吞吐下限 1 sample/s |

### Pressure 实测结果

| 策略 | step time | throughput | peak allocated | peak reserved | eval loss | 状态 |
|:---|---:|---:|---:|---:|---:|:---|
| baseline | 497.839 ms | 2.009 | 9782.74 MiB | 10750.00 MiB | 12.356503 | ok |
| checkpoint | 567.740 ms | 1.761 | 9450.76 MiB | 10896.00 MiB | 12.356503 | ok |
| offload | 1877.559 ms | 0.533 | 9448.33 MiB | 10404.00 MiB | 12.356503 | ok |
| hybrid | 783.478 ms | 1.276 | 9454.64 MiB | 10444.00 MiB | 12.356503 | ok |

### 结果解读

checkpoint 相比 baseline 节省 331.98 MiB 显存，吞吐下降约 12.3%；offload 只多节省约 2.4 MiB，却使吞吐下降约 73.4%；hybrid 的显存收益与 checkpoint 接近，但吞吐下降约 36.5%。四种策略的 eval loss 完全一致，说明本次策略切换没有造成质量退化。

当前没有方案达到 512 MiB 的有效显存收益阈值，因此项目决策为 `tune`，不是 `accept`。checkpoint 是当前最值得保留的候选；offload 和 hybrid 的速度代价不值得当前 workload 采用。这里的结论只适用于本次模型、序列长度和全参数训练口径。

### BF16 / seq_len=1024 扩展实验

该实验不替代 FP32/seq_len=768 主线，而是验证低精度是否能先解决长序列容量边界。配置为 Qwen2.5-0.5B-Instruct、batch=1、seq_len=1024、BF16 autocast、AdamW；只比较 baseline 与 checkpoint。

| 策略 | step time | throughput | peak allocated | peak reserved | eval loss | 状态 |
|:---|---:|---:|---:|---:|---:|:---|
| baseline | 297.060 ms | 3.366 samples/s | 10037.52 MiB | 10542.00 MiB | 12.202896 | ok |
| checkpoint | 342.564 ms | 2.919 samples/s | 10010.36 MiB | 10876.00 MiB | 12.202565 | ok |

BF16 使 seq_len=1024 在当前 12GB GPU 上成功运行；checkpoint 只额外节省约 27.16 MiB，吞吐保留约 86.7%，因此当前 BF16 workload 下不构成明显显存收益。这个结果说明：本次长序列压力更可能由 logits、参数或其他固定状态主导，不能据此断言 checkpoint 在所有长序列任务中都无效。由于当前显存容量有限，本实验没有形成更高 activation 主导 workload 的证据；更高压力需要 LoRA / QLoRA、分块 loss 或独立 activation-only benchmark 扩展。

### 与 75 节的衔接

76 负责提供候选策略的实测证据，75 负责把这些结果放入显存上限、吞吐下限和质量下限，输出最终的 `accept / tune / reject` 预算决策。