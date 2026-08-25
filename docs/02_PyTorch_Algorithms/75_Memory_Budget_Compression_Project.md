# 75. Memory Budget Compression Project | 显存预算压缩项目

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `预算规划`, `压缩策略` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节是 Task 3 的决策收口项目。你需要在训练质量不退化的前提下压缩显存预算。先明确硬件上限和质量下限，再汇总 73 的 baseline 与 76 的方案 benchmark，比较 checkpoint、offload、batch 调整等候选策略对 peak memory、step time、吞吐和 loss 的影响。最终输出一份预算表，并判断哪种方案值得保留。

**关键词：** `memory`, `budget`, `checkpoint`, `offload`, `project`

---
## 前置阅读

**导语：** 先完成 73 的训练性能测量和 76 的 checkpoint / offload 方案 benchmark，再进入这个项目；本节默认你已经知道显存为什么会爆，重点转向在预算约束下哪些压缩方案值得保留。本节输出的是 Task 3 的最终预算决策，不替代 Task 6 的端到端 profiling。
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)


## 相关阅读

**导语：** 完成本节后进入 74，用 profiling 对训练侧显存决策做端到端最终验证；如果决策为 `tune`，再回到 76 调整候选策略，必要时回到 73 复测 baseline。
- [74. Profiling Driven End-to-End Optimization | Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)

### Step 1: 定义显存预算压缩目标
先回答一个问题：当前训练任务最硬的约束到底是显存上限、质量下限，还是可接受的训练时长？

- 固定模型、数据、batch size、seq len、训练步数、评测指标和质量下限，保证后面的候选策略比较都在同一口径下进行。
- 明确显存预算，例如单卡可用显存上限、最低可接受吞吐和最大允许的 val loss 退化。
- 把候选策略先写清楚，例如 baseline、checkpoint、offload 或 batch 压缩。
- 这一步的目标不是立刻选策略，而是先把“什么叫压缩成功”定义清楚。

### Step 2: 先确认 baseline 和预算口径合法
显存预算项目必须先确认 baseline 和预算口径稳定，否则后面的压缩收益没有解释力。

- 先记录 baseline 的 peak memory、step time、samples/s 和 val loss。
- 再确认候选策略只改显存相关变量，不要把优化器、数据和训练步数一起改掉。
- 对预算本身，也要先确认显存上限、吞吐下限和质量阈值都写清楚。
- 如果 baseline 本身就不稳定，或者预算边界不明确，后面的压缩结论都不可信。

### Step 3: 用统一口径比较收益与代价
显存预算项目不能只看峰值显存是否下降，还要把速度、质量和工程代价一起算进去。

- 至少统一比较 peak memory、step time、throughput 和 val loss。
- 如果某个方案显著压低显存，但吞吐掉得太多或质量越过阈值，它通常只能进入 `tune` 或 `reject`。
- 如果某个方案满足预算、质量可接受，而且速度代价还在交付边界内，就可以进入 `accept`。
- 这一步的目标是把显存收益、性能代价和质量风险收成一张预算判断表。

### Step 4: 输出显存预算项目结论
显存预算项目最终不是输出“哪个方案最省显存”，而是输出当前预算下最值得继续保留的压缩方案。

- 项目结论建议统一成 `accept / tune / reject`。
- 输出最小报告时，至少包含预算口径、候选策略、核心指标差异和下一轮动作。
- 若进入 `tune`，下一轮优先回调 batch、checkpoint 颗粒度、offload 范围或通信方式，而不是一次性叠加更多优化手段。

#### 图解：19 / 42 / 73 / 76 如何收束到 75 显存预算压缩项目

`75` 不重复解释单个显存技巧，而是把前面几节的机制和证据口径收成一份预算下的压缩决策报告。

```text
19 Checkpoint / offload   activation savings intuition
      │
42 Activation offload     memory / transfer trade-off
      │
73 Training analysis      peak memory / step time / loss
      │
76 Strategy benchmark     checkpoint / offload / hybrid cost
      ▼
75 Memory Budget Compression Project
      ├─ budget ledger
      ├─ baseline vs compression strategies
      ├─ quality floor review
      └─ accept / tune / reject
```

项目页最小产物：

| 产物 | 你至少要记录什么 | 作用 |
|:---|:---|:---|
| 预算账本 | 显存上限、吞吐下限、质量阈值 | 固定压缩边界 |
| 候选策略 | baseline / checkpoint / offload 等方案 | 保证比较口径一致 |
| 结果对比 | peak memory、step time、throughput、val loss | 统一看收益与代价 |
| 项目结论 | accept / tune / reject | 输出预算判断 |

**本节交付物：** 一份可被 74 读取的预算决策记录，至少包含预算边界、候选策略、peak memory、step time、throughput、val loss、可行方案和下一步动作。75 负责训练侧局部预算裁决，最终是否保留还要由 74 的端到端 profiling 证据确认。

```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：预算检查、候选汇总、项目结论
# 目标：把 baseline 与压缩策略的显存预算比较收束成一份项目报告

from typing import Dict, List

def validate_memory_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_memory_strategies(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_memory_budget_project():
    try:
        budget = {'memory_cap_mb': 12000.0, 'min_samples_per_s': 6.0}
        quality_floor = {'max_val_loss': 1.15}
        check = validate_memory_budget(budget, quality_floor)
        assert check['is_valid'] is True, '预算检查应通过'
        assert check['missing_keys'] == [], '完整预算不应缺字段'

        candidates = [
            {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
            {'name': 'checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.5, 'val_loss': 1.09},
            {'name': 'offload', 'peak_memory_mb': 9800.0, 'samples_per_s': 4.5, 'val_loss': 1.08},
        ]
        summary = summarize_memory_strategies(candidates, budget, quality_floor)
        assert summary['feasible_count'] == 1, '只应有一个方案满足预算与质量'
        assert summary['best_candidate'] == 'checkpoint', 'checkpoint 应成为最优可行方案'

        decision = decide_memory_budget_project(summary)
        assert decision['decision'] == 'accept', '可行且最优的方案应被接受'

        hard_summary = summarize_memory_strategies(
            [
                {'name': 'checkpoint', 'peak_memory_mb': 13000.0, 'samples_per_s': 6.2, 'val_loss': 1.10},
                {'name': 'offload', 'peak_memory_mb': 11000.0, 'samples_per_s': 5.0, 'val_loss': 1.20},
            ],
            budget,
            quality_floor,
        )
        hard_decision = decide_memory_budget_project(hard_summary)
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


test_memory_budget_project()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
from typing import Dict, List

def validate_memory_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    required_budget_keys = ['memory_cap_mb', 'min_samples_per_s']
    required_quality_keys = ['max_val_loss']
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    return {
        'is_valid': len(missing_keys) == 0,
        'missing_keys': missing_keys,
    }


def summarize_memory_strategies(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
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
        'throughput_ratio': (best['samples_per_s'] / baseline['samples_per_s']) if baseline and best else None,
        'best_peak_memory_mb': best['peak_memory_mb'] if best else None,
        'memory_saving_mb': (baseline['peak_memory_mb'] - best['peak_memory_mb']) if baseline and best else 0.0,
    }


def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    feasible_count = summary['feasible_count']
    best_candidate = summary['best_candidate']
    quality_failed_count = summary['quality_failed_count']

    if feasible_count == 0:
        return {
            'decision': 'reject',
            'reason': 'no_strategy_meets_budget_and_quality',
            'next_action': 'tighten_batch_or_rework_memory_plan',
        }
    meaningful_memory_gain = summary.get('memory_saving_mb', 0.0) >= 512.0
    acceptable_throughput = summary.get('throughput_ratio') is None or summary.get('throughput_ratio') >= 0.70
    if best_candidate in {'checkpoint', 'offload', 'hybrid'} and meaningful_memory_gain and acceptable_throughput:
        return {
            'decision': 'accept',
            'reason': 'compression_strategy_is_best_feasible_option',
            'next_action': 'promote_to_training_run',
        }
    if quality_failed_count > 0:
        return {
            'decision': 'tune',
            'reason': 'compression_needs_quality_recovery',
            'next_action': 'adjust_checkpoint_scope_or_batch_plan',
        }
    if not meaningful_memory_gain:
        return {
            'decision': 'tune',
            'reason': 'memory_saving_below_meaningful_threshold',
            'next_action': 'test_larger_pressure_or_optimizer_compression',
        }
    if not acceptable_throughput:
        return {
            'decision': 'tune',
            'reason': 'throughput_loss_exceeds_budget',
            'next_action': 'reduce_compression_scope',
        }
    return {
        'decision': 'tune',
        'reason': 'baseline_still_best_under_current_budget',
        'next_action': 'revisit_memory_strategy_mix',
    }

```

### 解析

**1. TODO 1: 检查预算与质量阈值**
- **实现方式**：先把显存上限、吞吐下限和验证损失上限检查齐，再进入方案比较。
- **关键点**：没有统一预算口径时，显存压缩方案之间的比较都没有解释力。
- **项目意义**：这一步把 `75` 固定成预算约束下的显存决策页，而不是泛显存技巧页。

**2. TODO 2: 汇总显存压缩策略**
- **实现方式**：按 peak memory、samples/s 和 val loss 统一过滤候选，再选出最省显存的可行方案。
- **关键点**：显存收益只有在质量和吞吐都没有跌出边界时，才值得被保留。
- **项目意义**：这一步把 `19 / 42 / 73 / 76` 的机制与 benchmark 证据收成真正可比较的工程候选。

**3. TODO 3: 输出项目结论**
- **实现方式**：把候选可行性和最优方案统一收成 `accept / tune / reject`。
- **关键点**：项目结论必须回答“当前预算下哪种显存压缩方案值得继续采用”，而不是只输出一个峰值显存最小值。
- **项目意义**：这一步把 `75` 收成显存优化路线中的正式预算项目。

## 真实 GPU 结果解读：当前应继续 tune

以下结果来自 76 节的 RTX 5070 Ti Laptop GPU 实验，并由本节自动生成预算决策 JSON。

| 策略 | 峰值显存（MB） | 吞吐（samples/s） | 相对 baseline | 是否可行 |
|:---|---:|---:|---:|:---|
| baseline | 9782.74 | 2.009 | 100% | 是（11200 MiB） |
| checkpoint | 9450.76 | 1.761 | 87.7% | 是 |
| offload | 9448.33 | 0.533 | 26.5% | 否：吞吐低于下限 |
| hybrid | 9454.64 | 1.276 | 63.5% | 是 |

`checkpoint` 是当前最佳可行方案：显存节省约 332 MB，验证损失没有变化，吞吐保留约 87.7%。但项目设定的有意义显存收益阈值为 512 MB，因此本节输出 `tune`，而不是 `accept`。这表示方案有效但收益还不足以定版，不表示实验失败。

后续可以提高序列长度、调整 checkpoint 粒度，或进入优化器状态压缩实验；不要把 `offload` 仅凭更低峰值显存直接判为最佳方案。

### 预算敏感性与 BF16 扩展结果

| 实验 | 预算 | 可行策略 | 最佳策略 | 显存节省 | 决策 |
|:---|---:|:---|:---|---:|:---|
| FP32 / seq768 | 11200 MiB | baseline / checkpoint / hybrid | checkpoint | 331.98 MiB | tune |
| FP32 / seq768 | 9600 MiB | checkpoint / hybrid | checkpoint | 331.98 MiB | tune |
| BF16 / seq1024 | 9600 MiB | 无 | — | — | reject |
| BF16 / seq1024 | 11200 MiB | baseline / checkpoint | checkpoint | 27.16 MiB | tune |

这些结果说明：9600 MiB 预算下，FP32 baseline 和 BF16 长序列方案分别呈现不同的容量边界；11200 MiB 预算下 BF16 可以运行 seq_len=1024，但 checkpoint 只带来约 27 MiB 的额外显存收益。因此 BF16 在这个 workload 下比 checkpoint 更直接地解决了容量问题，checkpoint 仍需通过更高 activation 压力或 profiling 进一步判断。受当前 12GB GPU 容量限制，本项目不能把更高 activation 主导场景的结论写成已验证事实。
## Step 5（可选）：接入 73 / 76 的真实实验结果

先运行下面的“75 项目配置”代码块，再运行后面的真实结果代码。它读取 73 的训练 baseline 和 76 的策略 benchmark，把真实测量结果转换为显存预算项目的最终决策。若结果文件不存在，先完成 73 和 76 的真实 GPU 实验。本节本身不下载模型，也不需要 vLLM；环境准备请先阅读 [Part02 Intro 的环境说明](./intro.md#environment-notes-环境说明)。

### 75 项目配置

75 只读取 76 的 JSON，不重新训练模型，也不需要 GPU。默认关闭真实项目决策；需要生成报告时，将 `RUN_REAL_PROJECT` 改为 `True`。

```python
from pathlib import Path

# 75 只读取 76 的 JSON，不重新训练模型，也不需要 GPU。
RUN_REAL_PROJECT = True  # 设为 True 后运行预算决策。
BUDGET = {
    'memory_cap_mb': 11200.0,  # 当前实验允许的 GPU 显存上限。
    'min_samples_per_s': 1.0,  # 可接受的最低训练吞吐。
}
MAX_VAL_LOSS = None  # None：使用 76 baseline eval_loss 的 1.02 倍作为代理门槛。
RESULT_76_RELATIVE_PATH = Path('benchmarks/results/76_real_gpu_memory_bf16_seq1024.json')
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/75_memory_budget_decision_bf16_seq1024_11200.json')

```


```python
import json
import os
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
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, require_input_file
from tools.memory_budget_runtime import (
    summarize_memory_strategies,
    decide_memory_budget_project,
)
RESULT_76_PATH = PROJECT_ROOT / RESULT_76_RELATIVE_PATH

if RUN_REAL_PROJECT:
    require_input_file(RESULT_76_PATH, '76 结果')
    raw = json.loads(RESULT_76_PATH.read_text(encoding='utf-8'))
    candidates = []
    for item in raw['candidates']:
        if item.get('status', 'ok') != 'ok':
            continue
        candidates.append({
            'name': item['name'],
            'peak_memory_mb': item['peak_memory_mb'],
            'samples_per_s': item['samples_per_s'],
            'eval_loss': item.get('eval_loss', item.get('val_loss')),
            'val_loss': item.get('eval_loss', item.get('val_loss')),
        })
    baseline = next((x for x in candidates if x['name'] == 'baseline'), None)
    if baseline is None:
        raise RuntimeError('76 没有可用 baseline，无法形成预算决策。')
    max_val_loss = baseline['eval_loss'] * 1.02 if MAX_VAL_LOSS is None else MAX_VAL_LOSS
    quality_floor = {'max_val_loss': round(max_val_loss, 6)}
    summary = summarize_memory_strategies(candidates, BUDGET, quality_floor)
    decision = decide_memory_budget_project(summary)
    project_result = {
        'task': 'task3_training_memory_optimization',
        'stage': 'memory_budget_decision',
        'source': str(RESULT_76_PATH),
        'budget': BUDGET,
        'quality_floor': quality_floor,
        'candidates': candidates,
        'summary': summary,
        'decision': decision,
    }
    output_path = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(project_result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(project_result, ensure_ascii=False, indent=2))
else:
    print('跳过真实项目决策：保持 CPU-first 模式。')

```
