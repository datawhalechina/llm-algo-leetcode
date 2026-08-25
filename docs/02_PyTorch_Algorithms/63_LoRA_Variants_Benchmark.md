# 63. LoRA Variants Benchmark | LoRA 变体对比项目
**难度：** Hard | **环境：** CPU-first | **标签：** `训练微调`, `LoRA`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节要求你在统一训练预算下比较多种 LoRA 配置。固定数据、步数和评测口径后，分别记录 rank、alpha、dropout、target modules 对效果、显存、训练步时和可训练参数量的影响。最终输出一张 benchmark 排名与推荐表，并说明推荐结果依赖哪些约束。

**关键词：** `LoRA`, `variant`, `benchmark`, `budget`, `decision`

---
## 前置阅读

**导语：** 先把 LoRA 机制、有效 batch 口径、端到端微调闭环和基础 LoRA 项目理顺，再进入这个 benchmark；本节默认你已经知道单个 LoRA 项目怎么做，重点转向不同变体之间的比较和选型。

- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](./60_LoRA_Fine_Tuning_Project.md)

## 相关阅读

**导语：** 做完 LoRA 变体 benchmark 后，最自然的下一步是回看指令微调项目如何使用这些配置，或继续看训练性能分析是否支持当前选型。

- [62. Instruction Fine-Tuning Project | 指令微调项目](./62_Instruction_Fine_Tuning_Project.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
---
### Step 1: 定义 LoRA 变体 benchmark 目标

- 固定底座模型、数据集、batch size、seq len、优化器和训练步数。
- 明确候选 LoRA 变体，例如不同 rank、alpha、dropout、target modules 或初始化策略。
- 统一记录 train loss、val loss、step time、peak memory、可训练参数量和参数占比。

### Step 2: 先确认 baseline 和预算口径合法

LoRA 变体 benchmark 必须先确认 baseline 和预算口径稳定，不能脱离基线项目页单独存在。
- 至少要先知道基础 LoRA 配置的资源口径和效果基线，再去比较不同变体。
- 如果预算本身不清楚，排序结果再漂亮也没有部署意义。

### Step 3: 用统一口径比较收益与成本

LoRA 变体比较必须用统一口径同时看收益与成本，单一分数只能帮助排序，不能直接替代项目结论。
- 真正的判断至少要同时看效果、显存、步时和训练参数占比。
- 如果某个变体效果更好，但资源代价明显更高，它通常只能进入 `tune`，而不是直接 `accept`。

### Step 4: 输出项目结论

- 这页最终要输出 `accept / tune / reject`，而不是只给一个“推荐第一名”。
- 若进入 `tune`，下一轮优先回调 rank、target modules 和 dropout，而不是盲目增加更多变体。
#### 图解：10-60 如何收束到 63 LoRA Benchmark

`63` 把 LoRA 机制和项目经验收成一张统一的 benchmark 表。

```text
10 LoRA          target modules / rank / alpha / dropout
      │
12 Accumulation  micro batch -> effective batch
      │
13 E2E report    train loss / val loss / step time / memory
      │
60 LoRA project  baseline vs LoRA artifact and ledger
      │
      ▼
63 LoRA bench    variant ranking + budget-aware delivery decision
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | 基础 LoRA 口径、预算上限 | 保证比较合法 |
| candidate | rank、target modules、资源变化 | 解释变体收益来源 |
| 对比 | 效果、显存、步时、参数占比 | 判断是否值得 adopt |
| 决策 | accept / tune / reject | 输出 benchmark 结论 |

### 参数口径说明

`rank` 控制 LoRA 低秩容量，`alpha` 控制缩放，`dropout` 影响正则化，`target_modules` 决定 adapter 挂载位置。benchmark 时固定模型、数据、split、batch、seq_len、学习率和 steps，只改变这些 LoRA 变量；`train_loss / val_loss / step_time_ms / memory_mb / trainable_ratio` 分别用于效果、资源和参数效率比较。

```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：变体评分、排序、项目推荐
# 目标：把不同 LoRA 变体转成统一 benchmark 结果，而不是只给一张排名表

def score_lora_variant(variant: Dict[str, float]) -> Dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def rank_lora_variants(variants: List[Dict[str, float]]) -> List[Dict[str, float]]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_lora_variant(baseline: Dict[str, float], variants: List[Dict[str, float]], memory_budget_mb: int) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_lora_benchmark_template():
    baseline = {'name': 'baseline_lora', 'val_loss': 1.28, 'step_time_ms': 100, 'memory_mb': 1250, 'trainable_ratio': 0.06}
    variants = [
        {'name': 'rank4', 'train_loss': 1.2, 'val_loss': 1.4, 'step_time_ms': 90, 'memory_mb': 1100, 'trainable_ratio': 0.04},
        {'name': 'rank8', 'train_loss': 1.1, 'val_loss': 1.2, 'step_time_ms': 110, 'memory_mb': 1350, 'trainable_ratio': 0.08},
        {'name': 'rank16', 'train_loss': 1.0, 'val_loss': 1.15, 'step_time_ms': 140, 'memory_mb': 1700, 'trainable_ratio': 0.16},
    ]
    assert 'composite_cost' in score_lora_variant(variants[0])
    ranked = rank_lora_variants(variants)
    assert isinstance(ranked, list) and ranked[0]['name'] == 'rank8'
    decision = recommend_lora_variant(baseline, variants, memory_budget_mb=1500)
    assert decision['decision'] == 'accept'
    assert decision['recommended_name'] == 'rank8'
    assert decision['next_action'] == 'promote_to_extended_eval'

    tight_budget_variants = [
        {'name': 'budget_rank4', 'train_loss': 1.18, 'val_loss': 1.28, 'step_time_ms': 92, 'memory_mb': 1180, 'trainable_ratio': 0.04},
        {'name': 'budget_rank8', 'train_loss': 1.12, 'val_loss': 1.22, 'step_time_ms': 108, 'memory_mb': 1520, 'trainable_ratio': 0.08},
    ]
    tight_budget_decision = recommend_lora_variant(baseline, tight_budget_variants, memory_budget_mb=1200)
    assert tight_budget_decision['decision'] == 'tune'
    assert tight_budget_decision['recommended_name'] == 'budget_rank4'
    assert tight_budget_decision['next_action'] == 'refine_rank_or_target_modules'

    weak_variants = [
        {'name': 'worse_rank4', 'train_loss': 1.3, 'val_loss': 1.35, 'step_time_ms': 95, 'memory_mb': 1120, 'trainable_ratio': 0.04},
        {'name': 'worse_rank8', 'train_loss': 1.25, 'val_loss': 1.31, 'step_time_ms': 108, 'memory_mb': 1300, 'trainable_ratio': 0.08},
    ]
    reject_decision = recommend_lora_variant(baseline, weak_variants, memory_budget_mb=1500)
    assert reject_decision['decision'] == 'reject'
    assert reject_decision['recommended_name'] == 'worse_rank8'
    assert reject_decision['next_action'] == 'fallback_to_baseline_lora'


test_lora_benchmark_template()
print('测试通过：LoRA 变体 benchmark 模板可以工作。')

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
# TODO 1: 计算 LoRA 变体的综合成本
def score_lora_variant(variant: Dict[str, float]) -> Dict[str, float]:
    train_loss = float(variant.get('train_loss', 0.0))
    val_loss = float(variant.get('val_loss', 0.0))
    step_time_ms = float(variant.get('step_time_ms', 0.0))
    memory_mb = float(variant.get('memory_mb', 0.0))
    trainable_ratio = float(variant.get('trainable_ratio', 0.0))
    composite_cost = val_loss * 100 + train_loss * 10 + step_time_ms * 0.1 + memory_mb * 0.01 + trainable_ratio * 100
    return {
        'name': variant.get('name', 'variant'),
        'composite_cost': composite_cost,
        'memory_mb': memory_mb,
        'trainable_ratio': trainable_ratio,
        'val_loss': val_loss,
    }


# TODO 2: 对 LoRA 变体排序
def rank_lora_variants(variants: List[Dict[str, float]]) -> List[Dict[str, float]]:
    return sorted([score_lora_variant(variant) for variant in variants], key=lambda item: item['composite_cost'])


# TODO 3: 输出项目推荐结论
def recommend_lora_variant(baseline: Dict[str, float], variants: List[Dict[str, float]], memory_budget_mb: int) -> Dict[str, object]:
    feasible = [variant for variant in variants if float(variant.get('memory_mb', 10**9)) <= memory_budget_mb]
    if not feasible:
        return {
            'decision': 'reject',
            'recommended_name': None,
            'reason': '没有候选满足显存预算',
            'next_action': 'reduce_rank_or_scope',
        }

    best = min(
        feasible,
        key=lambda item: (
            float(item.get('val_loss', 10**9)),
            float(item.get('memory_mb', 10**9)),
            float(item.get('trainable_ratio', 10**9)),
            float(item.get('step_time_ms', 10**9)),
        ),
    )
    baseline_val_loss = float(baseline.get('val_loss', 10**9))
    baseline_memory = float(baseline.get('memory_mb', 10**9))

    if float(best.get('val_loss', 10**9)) < baseline_val_loss and float(best.get('memory_mb', 10**9)) <= memory_budget_mb:
        return {
            'decision': 'accept',
            'recommended_name': best.get('name', 'variant'),
            'reason': '在预算内带来更好的效果表现',
            'next_action': 'promote_to_extended_eval',
        }
    if float(best.get('val_loss', 10**9)) <= baseline_val_loss:
        return {
            'decision': 'tune',
            'recommended_name': best.get('name', 'variant'),
            'reason': '效果可用，但显存或训练参数代价仍偏高',
            'next_action': 'refine_rank_or_target_modules',
        }
    return {
        'decision': 'reject',
        'recommended_name': best.get('name', 'variant'),
        'reason': '候选未带来稳定效果收益',
        'next_action': 'fallback_to_baseline_lora',
    }

```

### 解析

这一页保留 `3` 个核心 TODO：变体评分、统一排序和项目推荐。它不要求把 LoRA 训练过程重写一遍，而是要求把 benchmark 决策补完整。

**1. TODO 1: 计算 LoRA 变体的综合成本**
- **实现方式**：把 `val_loss`、`train_loss`、`step_time_ms`、`memory_mb` 和 `trainable_ratio` 折算成统一的 `composite_cost`。
- **关键点**：这一步的目标不是追求完美公式，而是把效果和资源放进同一排序口径里。
- **项目意义**：没有统一评分口径，就只能看单项指标，无法支撑后面的 benchmark 结论。

**2. TODO 2: 对 LoRA 变体排序**
- **实现方式**：先对每个变体调用 `score_lora_variant`，再按 `composite_cost` 从低到高排序。
- **关键点**：排序只是候选筛选，不等于最终 `accept`；真正结论还要回到 baseline 和预算边界。
- **项目意义**：这一步让不同 rank、alpha 或 target modules 进入同一候选池，而不是零散比较。

**3. TODO 3: 输出项目推荐结论**
- **实现方式**：结合 baseline、显存预算和候选效果，输出 `accept / tune / reject` 与下一轮动作。
- **关键点**：预算内效果更好时才 `accept`；效果可用但预算边界偏紧时走 `tune`；没有稳定收益时 `reject`。
- **项目意义**：这一步把页面从“变体排序”推进到“项目选型”，回答的是哪种 LoRA 配置值得继续采用。

### 可选：统一项目报告导出
默认关闭。只有完成 baseline、LoRA 变体、预算和质量比较后，才导出统一 JSON。报告模板见 `docs/verification/fine_tuning_projects.md`。

```python
try:
    from tools.fine_tuning_project_runtime import runtime_snapshot, save_project_report, validate_project_config
except ModuleNotFoundError:
    runtime_snapshot = lambda: {'device': 'unknown'}
    validate_project_config = lambda config: []
    save_project_report = None
PROJECT_ID = '63_lora_variants_benchmark'
PROJECT_RESULT_PATH = 'benchmarks/results/63_lora_variants.json'
PROJECT_CONFIG = {'project': PROJECT_ID, 'model': 'template', 'dtype': 'fp32', 'batch_size': 1, 'seq_len': 128, 'steps': 1, 'seed': 42}
RUN_PROJECT_EXPORT = False  # True 只保存已完成的 benchmark 报告。
config_errors = validate_project_config(PROJECT_CONFIG)
if config_errors:
    raise ValueError('; '.join(config_errors))
print('runtime:', runtime_snapshot())
if RUN_PROJECT_EXPORT:
    if 'PROJECT_REPORT' not in globals():
        raise RuntimeError('请先组装完整的 PROJECT_REPORT')
    PROJECT_REPORT.setdefault('project', PROJECT_ID)
    PROJECT_REPORT.setdefault('config', PROJECT_CONFIG)
    PROJECT_REPORT.setdefault('environment', runtime_snapshot())
    save_project_report(PROJECT_RESULT_PATH, PROJECT_REPORT)

```
