# 75. Memory Budget Compression Project | 显存预算压缩项目

**难度：** Hard | **环境：** CPU-only；读取 GPU 项目报告 | **标签：** `显存优化`, `预算规划`, `压缩策略` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节读取 73 / 76 的报告，把显存上限、吞吐下限和质量门槛转成可执行的预算决策。这里不加载模型、不训练、不调用 CUDA；CPU 只做候选筛选和阈值敏感性分析，因此不会产生新的性能或显存实测。
**实验分层：** 73 / 76 负责采集数据，75 负责解释数据。若改变模型、workload 或硬件，应先重新获得上游报告，再运行本节。
**主责与复用边界：** 本项目主责是预算阈值、候选筛选和敏感性分析；它不重新训练、不采集 GPU 性能，也不替代 76 的策略 benchmark 或 74 的 profiling 归因。

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

### Step 1: 定义任务与资源约束
先把需求写成可检查的资源规格，而不是直接问哪个策略最省显存。至少明确模型、任务、数据规模、训练配置、目标 GPU 和交付指标。

| 规划输入 | 示例 | 作用 |
|:---|:---|:---|
| 模型与任务 | Qwen2.5-0.5B / causal LM training | 确定参数、激活和质量指标的含义 |
| 数据规模 | 样本数、平均 token、最大 seq_len | 判断 workload 是否具有代表性 |
| 硬件档案 | 12GB Laptop GPU、RTX 4090 24GB、40GB GPU | 定义可用显存预算 |
| 交付约束 | peak memory、吞吐、eval loss、训练时长 | 定义可接受方案 |


- 固定模型、数据、batch size、seq len、训练步数、评测指标和质量下限，保证后面的候选策略比较都在同一口径下进行。
- 明确显存预算，例如单卡可用显存上限、最低可接受吞吐和最大允许的 val loss 退化；预算对应的是目标硬件的可用空间，不是理论显存下界。
- 把候选策略先写清楚，例如 baseline、checkpoint、offload 或 batch 压缩，并标记它们分别是在减少 activation 驻留、转移存储位置，还是减少一次处理的 token 数。
- 这一步的目标不是立刻选策略，而是先把“什么叫压缩成功”定义清楚。

### Step 2: 区分实测证据与容量规划
75 支持两种模式：`measured` 读取 76 在目标硬件上的真实 benchmark；`projected` 只根据账本和硬件预算做规划，不能证明吞吐、kernel 或 OOM 边界。换 GPU、模型、dtype、数据或 seq_len 后，必须重新运行 73 / 76。

- `measured`：候选指标来自同一份 76 报告，适合输出当前 workload 下的 accept / tune / reject。
- `projected`：适合回答“24GB 设备理论上能否容纳更大的配置”，结果必须标为估算，并列出待补的 GPU 验证。
- 不把 5070 Ti 的实测吞吐直接迁移到 RTX 4090；不同硬件只能共享预算分析框架，不能共享性能结论。

**硬件规划示例：**

| 目标硬件 | 适合先规划的任务 | 首要调整项 | 需要补的实测 |
|:---|:---|:---|:---|
| 12GB Laptop GPU | 0.5B 级短上下文 LoRA / activation 压力 | micro-batch、seq_len、checkpoint | 73 / 76 同硬件 benchmark |
| RTX 4090 24GB | 更长上下文、较大 micro-batch、7B 级 LoRA/量化 | 可用显存、dtype、batch 和吞吐目标 | 目标模型在 4090 上的 73 / 76 |
| 40GB Data-center GPU | 更大模型或更高并发训练 | 优化器状态、并行方式和数据吞吐 | 目标模型与并行配置的 GPU benchmark |

这张表是规划入口，不是兼容性承诺；75 不能仅凭显存容量推导模型一定能训练，也不能推导 RTX 4090 的 step time。

### Step 3: 先确认 baseline 和预算口径合法
显存预算项目必须先确认 baseline 和预算口径稳定，否则后面的压缩收益没有解释力。

- 先记录 baseline 的 peak memory、step time、samples/s 和 val loss；它是 73 的测量结果，也是 76 的策略收益参照。
- 再确认候选策略只改显存相关变量，不要把优化器、数据和训练步数一起改掉；否则无法判断收益来自 checkpoint、offload 还是 workload 变小。
- 对预算本身，也要先确认显存上限、吞吐下限和质量阈值都写清楚。
- 如果 baseline 本身就不稳定，或者预算边界不明确，后面的压缩结论都不可信。

### Step 4: 用统一口径比较收益与代价
显存预算项目不能只看峰值显存是否下降，还要把速度、质量和工程代价一起算进去。

- 至少统一比较 peak memory、peak reserved、step time、throughput 和 val loss，并计算相对 baseline 的显存节省与吞吐保留率。
- 如果某个方案显著压低显存，但吞吐掉得太多或质量越过阈值，它通常只能进入 `tune` 或 `reject`；75 不重新解释 Task 2 的机制，只判断代价是否符合预算。
- 如果某个方案满足预算、质量可接受，而且速度代价还在交付边界内，就可以进入 `accept`。
- 这一步的目标是把显存收益、性能代价和质量风险收成一张预算判断表。

### Step 5: 输出显存预算项目结论
显存预算项目最终不是输出“哪个方案最省显存”，而是输出当前预算下最值得继续保留的压缩方案。

- 项目结论建议统一成 `accept / tune / reject`。
- 输出最小报告时，至少包含预算口径、候选策略、核心指标差异、每个候选的淘汰原因和下一轮动作。
- 若进入 `tune`，下一轮优先回调 batch、checkpoint 颗粒度、offload 范围或通信方式，而不是一次性叠加更多优化手段。
- 最后运行预算敏感性：分别改变显存上限、吞吐下限和质量阈值，观察可行集合与最佳候选是否稳定。只有在阈值附近仍保持同一结论，才可以称为较稳定的预算决策。
- 如果目标硬件与 76 报告中的 GPU 不一致，报告只能作为迁移前参考；最终结论应标为 `needs_target_hardware_validation`。

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
| 淘汰原因 | 显存超限 / 吞吐不足 / 质量超限 / OOM | 解释为什么不能选某方案 |
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
    # TODO：检查预算字段，不读取 GPU 状态，也不为缺失字段补默认值。
    # required_budget_keys = ???；required_quality_keys = ???；
    # budget_missing = ???；quality_missing = ???；missing_keys = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_memory_strategies(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    # TODO：按显存、吞吐和质量门槛汇总同一 workload 的候选策略。
    # memory_ok = ???；throughput_ok = ???；quality_ok = ???；is_feasible = ???；
    # baseline_peak = ???；best_peak = ???；memory_saving = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    # TODO：根据可行候选和节省幅度输出 accept / tune / reject。
    # has_feasible = ???；meaningful_saving = ???；decision = ???；
    # reason = ???；next_action = ???。
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
        missing = validate_memory_budget({'memory_cap_mb': 12000.0}, quality_floor)
        assert missing['is_valid'] is False, '缺少吞吐门槛时应拒绝预算配置'
        assert 'min_samples_per_s' in missing['missing_keys'], '应指出缺少的预算字段'

        candidates = [
            {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
            {'name': 'checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.5, 'val_loss': 1.09},
            {'name': 'offload', 'peak_memory_mb': 9800.0, 'samples_per_s': 4.5, 'val_loss': 1.08},
        ]
        summary = summarize_memory_strategies(candidates, budget, quality_floor)
        assert summary['feasible_count'] == 1, '只应有一个方案满足预算与质量'
        assert summary['best_candidate'] == 'checkpoint', 'checkpoint 应成为最优可行方案'
        evaluations = {item['name']: item for item in summary['evaluations']}
        assert evaluations['baseline']['reasons'] == ['memory_over_budget'], '应记录 baseline 的显存淘汰原因'
        assert evaluations['offload']['reasons'] == ['throughput_below_floor'], '应记录 offload 的吞吐淘汰原因'

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

        edge_summary = summarize_memory_strategies(
            [
                {'name': 'oom_candidate', 'status': 'oom'},
                {'name': 'incomplete', 'peak_memory_mb': 10000.0},
            ],
            budget,
            quality_floor,
        )
        assert edge_summary['oom_count'] == 1, 'OOM 候选应单独计数'
        assert edge_summary['invalid_count'] == 1, '缺少指标的候选应标记为 invalid'
        assert decide_memory_budget_project(edge_summary)['decision'] == 'reject', '没有可行策略时应 reject'
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
    evaluations = []

    for candidate in candidates:
        if candidate.get('status', 'ok') == 'oom':
            oom_count += 1
            evaluations.append({'name': candidate.get('name'), 'status': 'oom', 'reasons': ['oom']})
            continue
        memory = candidate.get('peak_memory_mb')
        throughput = candidate.get('samples_per_s')
        eval_loss = candidate.get('eval_loss', candidate.get('val_loss'))
        if not all(isinstance(value, (int, float)) for value in (memory, throughput, eval_loss)):
            invalid_count += 1
            evaluations.append({'name': candidate.get('name'), 'status': 'invalid', 'reasons': ['missing_or_non_numeric_metric']})
            continue
        memory_ok = memory <= budget['memory_cap_mb']
        speed_ok = throughput >= budget['min_samples_per_s']
        quality_ok = eval_loss <= quality_floor['max_val_loss']
        if not quality_ok:
            quality_failed += 1
        reasons = []
        if not memory_ok:
            reasons.append('memory_over_budget')
        if not speed_ok:
            reasons.append('throughput_below_floor')
        if not quality_ok:
            reasons.append('quality_over_floor')
        evaluations.append({'name': candidate.get('name'), 'status': 'feasible' if not reasons else 'rejected', 'reasons': reasons})
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
        'min_memory_saving_mb': budget.get('min_memory_saving_mb', 512.0),
        'min_throughput_ratio': budget.get('min_throughput_ratio', 0.70),
        'evaluations': evaluations,
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
    meaningful_memory_gain = summary.get('memory_saving_mb', 0.0) >= summary.get('min_memory_saving_mb', 512.0)
    acceptable_throughput = summary.get('throughput_ratio') is None or summary.get('throughput_ratio') >= summary.get('min_throughput_ratio', 0.70)
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

| 策略 | step time（ms） | 吞吐（samples/s） | peak allocated（MB） | peak reserved（MB） | eval loss | 状态 |
|:---|---:|---:|---:|---:|---:|:---|
| baseline | 497.839 | 2.009 | 9782.74 | 10750.00 | 12.356503 | ok |
| checkpoint | 567.740 | 1.761 | 9450.76 | 10896.00 | 12.356503 | ok |
| offload | 1877.559 | 0.533 | 9448.33 | 10404.00 | 12.356503 | ok |
| hybrid | 783.478 | 1.276 | 9454.64 | 10444.00 | 12.356503 | ok |

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
## Step 6（可选）：读取 73 / 76 的真实实验结果

先运行下面的“75 项目配置”代码块，再运行后面的结果代码。它读取 73 的训练 baseline 和 76 的策略 benchmark，把真实测量结果转换为显存预算项目的决策。若结果文件不存在，先完成 73 和 76 的真实 GPU 实验。本节不下载模型，也不需要 vLLM；环境准备请先查看[使用指南中的项目环境预检与安装说明](../docs/guide.md#项目环境预检与安装)。

### 75 项目配置

75 只读取 76 的 JSON，不重新训练模型，也不需要 GPU。当前可执行路径是 `measured`：默认关闭真实项目决策；需要生成报告时，将 `RUN_REAL_PROJECT` 改为 `True`。`projected` 目前只作为规划标签，不能生成未经验证的性能预测。

```python
from pathlib import Path

# 75 只读取 76 的 JSON，不重新训练模型，也不需要 GPU。
RUN_REAL_PROJECT = True  # 设为 True 后运行预算决策。
EVIDENCE_MODE = 'measured'  # 当前可执行模式；projected 仅用于规划标记，不产生性能预测。
MODEL_PROFILE = {
    'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
    'task': 'causal_lm_training',
    'dataset_name': '固定小型教学数据集',
    'dataset_samples': 4,
}
HARDWARE_PROFILES = {
    'laptop_12gb': {'name': '12GB Laptop GPU', 'total_memory_mb': 12227, 'usable_memory_mb': 11200},
    'rtx4090_24gb': {'name': 'RTX 4090 24GB', 'total_memory_mb': 24576, 'usable_memory_mb': 22500},
    'datacenter_40gb': {'name': '40GB Data-center GPU', 'total_memory_mb': 40960, 'usable_memory_mb': 38000},
}
HARDWARE_PROFILE = 'laptop_12gb'  # 规划目标；measured 模式仍以 76 报告中的 GPU 为证据。
USE_PROFILE_MEMORY_CAP = True  # True：自动使用硬件档案的 usable_memory_mb；False：使用下面手动填写的预算。
BUDGET = {
    'memory_cap_mb': 11200.0,  # 当前实验允许的 GPU 显存上限。
    'min_samples_per_s': 1.0,  # 可接受的最低训练吞吐。
    'min_memory_saving_mb': 512.0,  # 相对 baseline 的最低显存收益；教学阈值，可调整。
    'min_throughput_ratio': 0.70,  # 相对 baseline 的最低吞吐保留率。
}
if USE_PROFILE_MEMORY_CAP:
    BUDGET['memory_cap_mb'] = float(HARDWARE_PROFILES[HARDWARE_PROFILE]['usable_memory_mb'])
# 这里的 cap 是决策预算，不是整张卡可分配的理论总显存。
MAX_VAL_LOSS = None  # None：使用 76 baseline eval_loss 的 1.02 倍作为教学代理门槛，不等于真实任务质量。
SENSITIVITY_MEMORY_CAPS = [9600.0, 11200.0]  # 只改变预算约束，不重新训练；可按设备显存调整。
SENSITIVITY_THROUGHPUT_FLOORS = [1.0, 1.5, 2.0]  # 观察 accept / tune / reject 是否依赖吞吐门槛。
SENSITIVITY_QUALITY_FLOORS = [12.20, 12.45, 12.80]  # 质量代理阈值；不是完整任务评测。
# 主线读取 76 的 FP32 / pressure 报告；BF16 或其他 workload 必须改成对应的独立文件。
RESULT_76_RELATIVE_PATH = Path('benchmarks/results/76_real_gpu_memory.json')
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/75_memory_budget_decision.json')

```


```python
import json
import os
import sys
from pathlib import Path

# 75 不需要 GPU，但仍需先定位仓库，才能读取 76 报告并导入项目工具。
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
from tools.project_runtime import ensure_output_path, require_input_file, resolve_project_root, standard_experiment_config, standard_training_metrics
from tools.memory_budget_runtime import (
    summarize_memory_strategies,
    decide_memory_budget_project,
)

PROJECT_ROOT = resolve_project_root(PROJECT_ROOT)
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))
RESULT_76_PATH = PROJECT_ROOT / RESULT_76_RELATIVE_PATH

if RUN_REAL_PROJECT:
    if EVIDENCE_MODE != 'measured':
        raise ValueError('当前 75 代码只执行 measured 决策；projected 仅记录规划标签，不能生成性能预测。')
    require_input_file(RESULT_76_PATH, '76 结果')
    raw = json.loads(RESULT_76_PATH.read_text(encoding='utf-8'))
    candidates = []
    for item in raw['candidates']:
        candidates.append({
            'name': item['name'],
            'status': item.get('status', 'ok'),
            'step_time_ms': item.get('step_time_ms'),
            'samples_per_s': item.get('samples_per_s'),
            'peak_memory_mb': item.get('peak_memory_mb', item.get('peak_mem_mb')),
            'peak_reserved_mb': item.get('peak_reserved_mb'),
            'loss': item.get('loss'),
            'eval_loss': item.get('eval_loss', item.get('val_loss')),
            'val_loss': item.get('val_loss', item.get('eval_loss')),
            'error': item.get('error'),
        })
    baseline = next((x for x in candidates if x['name'] == 'baseline'), None)
    if baseline is None:
        raise RuntimeError('76 没有可用 baseline，无法形成预算决策。')
    max_val_loss = baseline['eval_loss'] * 1.02 if MAX_VAL_LOSS is None else MAX_VAL_LOSS
    quality_floor = {'max_val_loss': round(max_val_loss, 6)}
    summary = summarize_memory_strategies(candidates, BUDGET, quality_floor)
    decision = decide_memory_budget_project(summary)
    sensitivity = []
    for memory_cap_mb in SENSITIVITY_MEMORY_CAPS:
        for throughput_floor in SENSITIVITY_THROUGHPUT_FLOORS:
            for quality_floor_value in SENSITIVITY_QUALITY_FLOORS:
                sensitivity_budget = {
                    'memory_cap_mb': memory_cap_mb,
                    'min_samples_per_s': throughput_floor,
                    'min_memory_saving_mb': BUDGET['min_memory_saving_mb'],
                    'min_throughput_ratio': BUDGET['min_throughput_ratio'],
                }
                sensitivity_quality_floor = {'max_val_loss': quality_floor_value}
                sensitivity_summary = summarize_memory_strategies(candidates, sensitivity_budget, sensitivity_quality_floor)
                sensitivity_decision = decide_memory_budget_project(sensitivity_summary)
                sensitivity.append({
                    'memory_cap_mb': memory_cap_mb,
                    'min_samples_per_s': throughput_floor,
                    'max_val_loss': quality_floor_value,
                    'feasible_names': sensitivity_summary['feasible_names'],
                    'best_candidate': sensitivity_summary['best_candidate'],
                    'decision': sensitivity_decision['decision'],
                })
    project_result = {
        'task': 'task3_training_memory_optimization',
        'stage': 'memory_budget_decision',
        'source': str(RESULT_76_PATH),
        'budget': BUDGET,
        'quality_floor': quality_floor,
        'evidence_mode': EVIDENCE_MODE,
        'planning_context': {
            'model': MODEL_PROFILE,
            'hardware_target': HARDWARE_PROFILES[HARDWARE_PROFILE],
            'note': '硬件目标用于预算规划；只有 76 报告中的硬件和 workload 属于 measured evidence。',
        },
        'candidates': candidates,
        'summary': summary,
        'decision': decision,
        'sensitivity': sensitivity,
    }
    source_config = raw.get('config', {})
    project_result['planning_context']['measured_hardware'] = source_config.get('device')
    project_result['planning_context']['hardware_comparison'] = 'same_hardware_required_for_performance_claim'
    project_result['experiment'] = standard_experiment_config({**source_config, 'dtype': source_config.get('dtype', source_config.get('amp_dtype'))})
    project_result['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in candidates}
    output_path = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(project_result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(project_result, ensure_ascii=False, indent=2))
    print('\n预算敏感性：')
    for row in sensitivity:
        print(f"memory_cap={row['memory_cap_mb']:.0f} MB, throughput_floor={row['min_samples_per_s']:.1f}, feasible={row['feasible_names']}, decision={row['decision']}")
else:
    print('跳过真实项目决策：保持 CPU-first 模式。')

```
