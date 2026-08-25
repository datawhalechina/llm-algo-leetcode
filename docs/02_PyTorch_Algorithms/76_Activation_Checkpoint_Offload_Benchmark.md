# 76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `Checkpoint/Offload`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节是 Task 3 的核心方案 benchmark，承接 73 建立的测量口径，在相同训练任务下比较 checkpoint、offload 和组合方案。固定显存预算、吞吐目标与质量下限后，分别记录 peak memory、step time、数据搬运或重算开销，并与 baseline 对照。最终把结果交给 75，形成预算决策；本节本身负责测量方案代价，不负责最终预算裁决。

> 环境提示：运行真实 GPU benchmark 前，请先阅读 [Part02 Intro 的环境说明](./intro.md#environment-notes-环境说明)。本节只需要当前 GPU PyTorch runtime，不需要 vLLM，也不要求学习者预先创建两套虚拟环境。

**关键词：** `activation`, `checkpoint`, `offload`, `memory`, `benchmark`

## Task 0–2 如何在本节体现

76 不重新讲解前置机制，而是把三层知识放进同一个训练实验中：

| 前置任务 | 在 76 中体现的内容 | 实验观察 |
|:---|:---|:---|
| Task 0：反向传播基础 | checkpoint 通过反向传播时的重计算恢复激活；offload 必须保证 backward 需要的张量仍可取回 | loss、梯度和训练 step 是否正确 |
| Task 1：显存与性能认知 | 区分 allocated / reserved，理解显存容量、计算时间和数据搬运之间的关系 | peak memory、step time、吞吐 |
| Task 2：训练侧显存机制 | 把 checkpoint、offload、hybrid 作为候选策略，在同一训练任务下进行控制变量比较 | 节省多少显存、付出多少重算或搬运代价 |

因此，76 的实验链路是：`Task 0 保证训练正确性 → Task 1 定义测量口径 → Task 2 提供优化候选 → 76 做方案 benchmark`。73 提供统一 baseline，75 再根据 76 的结果形成预算决策。

---
## 前置阅读

**导语：** 先完成 Task 2 的 checkpoint、offload 等训练侧显存机制学习，并阅读 73 了解统一测量口径，再进入这个项目；本节默认你已经知道这些技巧各自怎么省显存，重点转向在同一训练任务下测量哪种方案更值。
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [43. Unified Memory Management | 统一内存管理](./43_Unified_Memory_Management.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)


## 相关阅读

**导语：** 做完这页后，先把结果交给 75 完成训练侧预算决策，再由 74 使用 profiling 对显存优化方案做端到端最终验证。
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [74. Profiling Driven End-to-End Optimization | Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)

### Step 1: 定义显存策略对比目标
先回答一个问题：当前训练任务下，你比较这些方案是为了压进预算、保住吞吐，还是在二者之间找最稳的折中？

- 固定模型、数据、batch size、seq len、训练步数、评测指标和质量下限，保证后面的策略比较都在同一口径下进行。
- 明确候选方案：baseline、checkpoint、offload，以及必要时的 hybrid 组合。
- 把预算边界先写清楚：显存上限、最低可接受吞吐和最大允许的 val loss 退化。
- 这一步的目标不是立刻挑一个方案，而是先把“什么叫值得采用”定义清楚。

### Step 2: 先确认 baseline 和方案口径合法
显存策略对比项目必须先确认 baseline 和候选方案的比较口径一致，否则后面的结果没有解释力。

- 先记录 baseline 的 peak memory、step time、samples/s 和 val loss。
- 再确认候选方案只改显存策略，不要把优化器、数据和训练步数一起改掉。
- 对组合方案，还要单独说明 checkpoint 颗粒度和 offload 范围。
- 如果 baseline 本身不稳定，或者方案口径前后不一致，后面的对比结论都不可信。

### Step 3: 用统一口径比较收益与代价
显存策略对比不能只看哪组显存最低，还要把吞吐、step time 和质量约束一起算进去。

- 至少统一比较 peak memory、step time、samples/s 和 val loss。
- 如果某个方案显存收益很大，但吞吐掉得太多或质量越过阈值，它通常只能进入 `tune` 或 `reject`。
- 如果某个方案显著压低显存，同时速度和质量都还在交付边界内，就可以进入 `accept`。
- 这一步的目标是把显存收益、执行代价和训练风险收成一张可比较的方案表。

### Step 4: 输出显存策略项目结论
显存策略对比项目最终不是输出“哪个技巧最省显存”，而是输出当前预算下最值得继续采用的策略组合。

- 项目结论建议统一成 `accept / tune / reject`。
- 输出最小报告时，至少包含候选方案、核心指标差异、是否满足预算与质量、以及下一轮动作。
- 若进入 `tune`，下一轮优先回调 checkpoint 颗粒度、offload 范围或组合方式，而不是直接再叠更多技巧。

#### 图解：Task 2 + 73 如何收束到 76 显存策略对比项目

`76` 不重复解释单个显存技巧，而是把前面几节的机制和预算口径收成一份方案对比报告。

```text
19 Checkpoint / offload   memory saving intuition
      │
42 Activation offload     transfer and runtime trade-off
      │
43 Unified memory         system-side memory coordination
      │
73 Training analysis      fixed measurement protocol
      ▼
76 Activation / Checkpoint / Offload Benchmark
      ├─ strategy candidates
      ├─ baseline vs checkpoint vs offload vs hybrid
      ├─ quality floor review
      └─ benchmark report → 75 budget decision
```

项目页最小产物：

| 产物 | 你至少要记录什么 | 作用 |
|:---|:---|:---|
| 候选方案 | baseline / checkpoint / offload / hybrid | 固定比较对象 |
| 预算边界 | 显存上限、吞吐下限、质量阈值 | 固定方案判断边界 |
| 结果对比 | peak memory、step time、samples/s、val loss | 统一看收益与代价 |
| 项目结论 | accept / tune / reject | 输出策略选择 |


```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：预算检查、候选汇总、项目结论
# 目标：把 baseline / checkpoint / offload / hybrid 的方案比较收束成一份项目报告

def validate_strategy_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_memory_strategy_candidates(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
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

## Step 6（可选）：真实 GPU 显存策略 benchmark

本 Step 复用 73 的训练口径，在同一模型、固定输入和 FP32 + AdamW 配置下比较 baseline、activation checkpoint、CPU offload 和 hybrid。运行前会自动读取并校验 `benchmarks/results/73_real_gpu_training.json`；如果模型、batch、seq_len、dtype 或 optimizer 不一致，会要求先用相同 workload 重跑 73。提供 `smoke` 与 `pressure` 两档 workload：smoke 用于快速校验，pressure 用于提高 activation 压力。结果保存到 `benchmarks/results/76_real_gpu_memory.json`。

这里的 offload 使用 PyTorch 的 `torch.autograd.graph.save_on_cpu` 保存 backward 所需张量，重点是教学 benchmark，不等同于生产训练框架中的完整 offload 调度。`eval_loss` 只是固定随机输入上的质量代理指标，不等同于真实数据集验证结果。

本节主线保持全参数训练口径：`Qwen2.5-0.5B + FP32 + AdamW`。如果要在 12GB 显存上测试更大模型或更长序列，应另设 LoRA / QLoRA 扩展实验：LoRA 主要减少可训练参数、梯度和 optimizer state，QLoRA 进一步压缩基座权重；它们不能与本节主线结果直接横向比较，但可以用于观察更大模型下 activation checkpoint / offload 的适用边界。

```python
from pathlib import Path

RUN_REAL_GPU = True  # 是否运行真实 GPU；实测时显式改为 True。
DTYPE_MODE = 'bf16'  # fp32：主线；bf16：扩展实验，启用 CUDA autocast。
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 固定基座模型。
MODEL_SOURCE = 'auto'  # 模型来源：auto / huggingface / modelscope / local。
MODEL_CACHE_DIR = 'model_cache'  # 模型缓存目录。
BATCH_SIZE = 1  # 默认 batch；实际值由 WORKLOADS 覆盖。
WORKLOAD = 'pressure_1024'  # pressure 是 768 主线；pressure_1024 是长序列扩展。
WORKLOADS = {
    'smoke': {'batch_size': 1, 'seq_len': 512, 'warmup': 2, 'iters': 5},
    'pressure': {'batch_size': 1, 'seq_len': 768, 'warmup': 2, 'iters': 5},
    'pressure_1024': {'batch_size': 1, 'seq_len': 1024, 'warmup': 2, 'iters': 5},
}
WARMUP = 2  # 预热轮数，不计入正式平均值。
ITERS = 5  # 每种策略的正式测量轮数。
SEED = 42  # 固定输入，保证策略间 workload 一致。
STRATEGIES = ['baseline', 'checkpoint'] if DTYPE_MODE == 'bf16' else ['baseline', 'checkpoint', 'offload', 'hybrid']

# 完成 smoke 后，可改为：['baseline', 'checkpoint', 'offload', 'hybrid']
MEMORY_CAP_MB = 11200.0  # 硬显存预算；需为系统和桌面进程留余量。
MIN_SAMPLES_PER_S = 1.0  # 最低吞吐；低于此值的策略判为不可行。
# None 表示按当前固定 workload 的 baseline eval_loss 自动生成质量上限
MAX_VAL_LOSS = None  # 只是质量代理门槛，不是完整任务质量门槛。
BASELINE_73_RELATIVE_PATH = Path('benchmarks/results/73_real_gpu_training_bf16.json')
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/76_real_gpu_memory_bf16_seq1024.json')

```


```python
import json
import gc
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

def resolve_project_root():
    override = os.environ.get('LLM_ALGO_PROJECT_ROOT')
    if override:
        return Path(override).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'benchmarks').is_dir() and (candidate / '02_PyTorch_Algorithms').is_dir():
            return candidate
    return current

PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, runtime_snapshot, validate_training_config
from tools.memory_strategy_runtime import (
    summarize_memory_strategy_candidates,
    decide_memory_strategy_project,
)
OUTPUT_PATH = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
print(f'项目根目录: {PROJECT_ROOT}')
print(f'结果保存路径: {OUTPUT_PATH}')

if RUN_REAL_GPU:
    import torch
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

        for _ in range(WARMUP):
            train_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        losses = [train_step() for _ in range(ITERS)]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DTYPE_MODE == 'bf16'):
                eval_loss = float(model(input_ids=eval_input_ids, labels=eval_labels).loss.item())
        result = {
            'name': name,
            'status': 'ok',
            'step_time_ms': round(elapsed * 1000 / ITERS, 3),
            'samples_per_s': round(BATCH_SIZE * ITERS / elapsed, 3),
            'loss': round(losses[-1], 6),
            'eval_loss': round(eval_loss, 6),
            'peak_memory_mb': round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            'peak_reserved_mb': round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
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

    candidates = [run_strategy(name) for name in STRATEGIES]
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
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print('跳过真实 GPU benchmark：保持 CPU-first 模式。')

```

## 实测记录：本地 RTX 5070 Ti GPU

下面记录 76 的真实 GPU benchmark，作为本节的可读实验报告；原始 JSON 由 Step 6 保存到 `benchmarks/results/76_real_gpu_memory.json`。

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

| 策略 | step time | throughput | peak allocated | eval loss | 状态 |
|:---|---:|---:|---:|---:|:---|
| baseline | 497.839 ms | 2.009 | 9782.74 MiB | 12.356503 | ok |
| checkpoint | 567.740 ms | 1.761 | 9450.76 MiB | 12.356503 | ok |
| offload | 1877.559 ms | 0.533 | 9448.33 MiB | 12.356503 | ok |
| hybrid | 783.478 ms | 1.276 | 9454.64 MiB | 12.356503 | ok |

### 结果解读

checkpoint 相比 baseline 节省 331.98 MiB 显存，吞吐下降约 12.3%；offload 只多节省约 2.4 MiB，却使吞吐下降约 73.4%；hybrid 的显存收益与 checkpoint 接近，但吞吐下降约 36.5%。四种策略的 eval loss 完全一致，说明本次策略切换没有造成质量退化。

当前没有方案达到 512 MiB 的有效显存收益阈值，因此项目决策为 `tune`，不是 `accept`。checkpoint 是当前最值得保留的候选；offload 和 hybrid 的速度代价不值得当前 workload 采用。

### BF16 / seq_len=1024 扩展实验

该实验不替代 FP32/seq_len=768 主线，而是验证低精度是否能先解决长序列容量边界。配置为 Qwen2.5-0.5B-Instruct、batch=1、seq_len=1024、BF16 autocast、AdamW；只比较 baseline 与 checkpoint。

| 策略 | 峰值显存 | throughput | eval loss | 状态 |
|:---|---:|---:|---:|:---|
| baseline | 10037.52 MiB | 3.366 samples/s | 12.202896 | ok |
| checkpoint | 10010.36 MiB | 2.919 samples/s | 12.202565 | ok |

BF16 使 seq_len=1024 在当前 12GB GPU 上成功运行；checkpoint 只额外节省约 27.16 MiB，吞吐保留约 86.7%，因此当前 BF16 workload 下不构成明显显存收益。这个结果说明：本次长序列压力更可能由 logits、参数或其他固定状态主导，不能据此断言 checkpoint 在所有长序列任务中都无效。由于当前显存容量有限，本实验没有形成更高 activation 主导 workload 的证据；更高压力需要 LoRA / QLoRA、分块 loss 或独立 activation-only benchmark 扩展。

### 与 75 节的衔接

76 负责提供候选策略的实测证据，75 负责把这些结果放入显存上限、吞吐下限和质量下限，输出最终的 `accept / tune / reject` 预算决策。