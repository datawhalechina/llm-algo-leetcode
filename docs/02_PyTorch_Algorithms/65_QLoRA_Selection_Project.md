# 65. QLoRA Selection Project | QLoRA 选型项目

**难度：** Hard | **环境：** CPU-first | **标签：** `量化压缩`, `QLoRA`, `选型` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节要求你在显存预算和质量下限明确的情况下，比较全参数、LoRA 与 QLoRA 三种方案。统一记录显存、吞吐、训练稳定性和验证质量，确认量化收益是否足以覆盖额外误差与实现成本。最终输出当前预算下的方案选择及其适用边界。

本项目只负责低比特训练适配的预算选型：重点是 NF4/QLoRA、LoRA 参数、训练显存和验证质量。它不验证 GPTQ/AWQ 的后训练部署，也不验证 GGUF 的文件格式与推理 backend；这些边界分别由 `40` 和 `67` 承担。

**关键词：** `QLoRA`, `budget`, `memory`, `selection`, `project`

---
## 前置阅读

**导语：** 先把 LoRA、有效 batch、端到端微调和量化基础理顺，再进入这个项目；本节默认你已经知道低比特与适配器怎么接起来，重点转向在预算约束下该不该用 QLoRA。
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [40. GPTQ and AWQ | GPTQ 与 AWQ](./40_GPTQ_and_AWQ_Weight_Quantization.md)（交叉参考：部署侧权重量化，不是本项目候选）

## 相关阅读

**导语：** 做完这页后，最自然的下一步是把低资源训练结论继续推向量化部署，或者回到显存预算侧继续做更细的资源压缩判断。
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
---
### Step 1: 定义 QLoRA 选型目标与实验矩阵
先回答一个问题：当前预算约束下，哪个方案能在训练可行、质量达标和吞吐可接受之间取得平衡？本节的 CPU 代码负责预算推演；真实显存与量化 kernel 仍需 GPU 验证。

| 实验层级 | 候选与变量 | 主要指标 | 结论边界 |
|:---|:---|:---|:---|
| CPU 主实验 | baseline / LoRA / QLoRA 的账本与规则 | 估算显存、可行性、候选排序 | 只能筛选方案，不能证明真实速度 |
| GPU 验证（可选） | 固定数据、步数和评测，仅改变适配器或量化配置 | peak memory、step time、吞吐、val loss、OOM | 验证真实训练代价与质量 |

固定任务、数据集、训练步数、batch size、seq len、评测指标和质量下限；候选只改变 LoRA / QLoRA 配置，不同时改变数据和训练口径。

### Step 2（CPU 项目设计）：建立预算与 baseline 口径
低资源微调项目必须先确认 baseline 和预算口径稳定，否则显存收益无法解释。

- 记录显存上限、最低吞吐和验证损失上限。
- 拆分冻结底座、LoRA 参数、梯度、optimizer state、activation 和量化元数据。
- 对 QLoRA 核对位宽、NF4 / double quant、target modules 和 rank。
- 估算字段标记为 `estimated`；不要把账本数字写成 CUDA 峰值。

### Step 3（GPU 验证设计）：固定 workload，只改变候选变量
GPU 实验用于验证 CPU 预算推演是否接近真实训练，不是为了让每个学习者都跑完整训练。

- 最小对照是同一 workload 下的 LoRA 与 QLoRA；全参数方案只有在显存足够时才加入。
- 统一记录 peak memory、peak reserved、step time、tokens/s、train / val loss 和 OOM 状态。
- 量化模型必须记录实际量化格式、加载是否成功和使用的 backend / kernel。
- GPU 结果只能说明当前模型、数据、硬件和软件栈下的表现，不能外推到所有模型。

### Step 4: 用统一口径比较收益与代价
QLoRA 选型不能只看显存是否下降，还要把训练质量和工程代价一起算进去。

- 至少统一比较 peak memory、step time、val loss 和是否满足质量下限。
- 如果 QLoRA 只节省少量显存，却显著拉低质量或拖慢步时，它通常只能进入 `tune` 或 `reject`。
- 如果 QLoRA 明显节省预算，同时质量退化可接受、吞吐也没有恶化到不可交付，就可以进入 `accept`。
- 这一步的目标是把预算收益、训练代价和质量风险收成同一张项目判断表。

### Step 5: 输出 QLoRA 选型结论
低资源微调项目最终不是输出“哪个方案最省显存”，而是输出当前预算下最值得继续采用的方案。

- 项目结论建议统一成 `accept / tune / reject`。
- 输出最小报告时，至少包含候选配置、显存与步时差异、质量下限判断和下一轮动作。
- 若进入 `tune`，下一轮优先回调 rank、target modules、量化位宽或 batch 策略，而不是先扩更多候选方案。

#### 图解：10 / 12 / 13 / 25 / 26 如何收束到 65 QLoRA 选型项目

`65` 不重复实现 LoRA 或量化原理，而是把前面几节的训练与压缩口径收成一份预算下的方案对比报告。

```text
10 LoRA            target modules / rank / adapter config
      │
12 Accumulation    effective batch / update cadence
      │
13 End-to-end      train / val loop and minimal report
      │
25 W8A16          weight-only representation intuition
26 QLoRA / NF4    frozen base + trainable adapter
      ▼
65 QLoRA Selection Project
      ├─ budget ledger
      ├─ baseline vs LoRA vs QLoRA
      ├─ quality floor review
      └─ accept / tune / reject
```

显存证据边界：本节的 CPU 代码只负责预算筛选、候选排序和决策逻辑；如果候选显存来自估算，报告必须标记为 `estimated`，不能写成 CUDA 峰值。要证明底座权重、LoRA 参数、梯度、optimizer state 或 activation 的实际占用，需在固定 workload 下采集 GPU 的 `peak_memory`、`peak_reserved` 和 OOM 状态。`73–76` 提供训练侧通用测量与策略对照，`65` 只补充 QLoRA/NF4 的专项选型。

`40` 的 GPTQ/AWQ 和 `67` 的量化部署属于推理侧交叉路线，不是本项目的候选方案；`GGUF` 也不应被当作 QLoRA 的训练格式。

项目页最小产物：

| 产物 | 你至少要记录什么 | 作用 |
|:---|:---|:---|
| 预算账本 | 显存上限、目标吞吐、质量下限 | 固定选型边界 |
| 候选配置 | baseline / LoRA / QLoRA 的关键配置 | 保证比较口径一致 |
| 结果对比 | peak memory、step time、val loss | 统一看收益与代价 |
| 项目结论 | accept / tune / reject | 输出方案选择 |

### 参数口径说明

`memory_cap_mb` 是显存上限，`min_tokens_per_s` 是最低吞吐，`max_val_loss` 是质量上限；三者共同定义 candidate 是否可行。`bit_width`、量化格式、double quant、LoRA rank 和 target modules 属于候选配置，比较时只能改变量化/适配器变量，不能同时改变数据、训练步数和质量评测口径。

显存账本建议至少拆成：`base_weight_mb`（冻结底座）、`trainable_param_mb`（LoRA 参数）、`gradient_mb`、`optimizer_state_mb`、`activation_mb` 和 `quant_metadata_mb`。这些字段用于解释显存来源；`estimated_total_mb` 是账本估算，`peak_memory_mb` / `peak_reserved_mb` 只有在 CUDA 训练中采集后才是实测值。

```python
from typing import Dict, List

```


```python
# TODO: 完成 QLoRA 选型项目的预算检查、候选汇总和项目结论
# 目标：把 baseline / LoRA / QLoRA 的低资源微调比较收束成一份选型报告

def build_memory_ledger(base_weight_mb: float, trainable_param_mb: float, gradient_mb: float, optimizer_state_mb: float, activation_mb: float, quant_metadata_mb: float = 0.0, peak_memory_mb: float = None, peak_reserved_mb: float = None, evidence: str = 'estimated') -> Dict[str, object]:
    """汇总训练显存对象，并区分估算值与 CUDA 峰值。"""
    values = {
        'base_weight_mb': base_weight_mb, 'trainable_param_mb': trainable_param_mb,
        'gradient_mb': gradient_mb, 'optimizer_state_mb': optimizer_state_mb,
        'activation_mb': activation_mb, 'quant_metadata_mb': quant_metadata_mb,
    }
    if any(float(value) < 0 for value in values.values()):
        raise ValueError('显存账本中的各项不能为负数。')
    estimated_total_mb = sum(float(value) for value in values.values())
    report = {**values, 'estimated_total_mb': round(estimated_total_mb, 3), 'evidence': evidence}
    if peak_memory_mb is not None:
        report['peak_memory_mb'] = float(peak_memory_mb)
        report['reconciliation_gap_mb'] = round(float(peak_memory_mb) - estimated_total_mb, 3)
    if peak_reserved_mb is not None:
        report['peak_reserved_mb'] = float(peak_reserved_mb)
    return report


def validate_qlora_candidate(candidate: Dict[str, float]) -> List[str]:
    # TODO 0：检查候选名称、显存、吞吐和验证损失
    # 提示：required = ('name', 'memory_mb', 'tokens_per_s', 'val_loss')。
    # 对数值字段检查可转换、有限且 memory_mb / tokens_per_s 不为负。
    # 返回错误列表；空列表表示候选可以进入排序。
    raise NotImplementedError("请先完成 TODO 代码！")

def validate_budget_and_quality(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """检查低资源微调项目的预算口径和质量下限是否完整。"""
    # ==========================================
    # TODO 1：检查预算与质量下限是否完整
    # 提示：required_budget_keys = ['memory_cap_mb', 'min_tokens_per_s']。
    #       required_quality_keys = ['max_val_loss']。
    # 依次计算 budget_missing、quality_missing，再合并为 missing_keys。
    # 数值上限必须为正；max_val_loss 不能为负。
    # 没有统一预算口径时，后面的显存、吞吐和质量比较都没有解释力。
    # ==========================================
    required_budget_keys = ['memory_cap_mb', 'min_tokens_per_s']
    required_quality_keys = ['max_val_loss']
    # budget_missing = ???
    # quality_missing = ???
    # missing_keys = ???
    return {
        'is_valid': len(missing_keys) == 0,
        'missing_keys': missing_keys,
    }


def summarize_low_resource_candidates(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """按显存、吞吐和验证损失筛选 QLoRA 候选。

    只返回通过全部门槛的候选名称；非法或缺失测量不能静默进入排序。
    """
    """汇总低资源微调候选，筛出满足预算与质量的方案。"""
    # ==========================================
    # TODO 2: 汇总低资源微调候选
    # 提示：统计满足预算与质量的候选，并输出最省显存的可用方案。
    # 这里至少要同时看 memory、tokens/s 和 val_loss，不能只看单一指标。
    # ==========================================
    feasible = []
    rejected = []
    for candidate in candidates:
        # TODO 2：对每个候选分别计算以下布尔变量：
        # within_memory = candidate['memory_mb'] <= budget['memory_cap_mb']
        # enough_throughput = candidate['tokens_per_s'] >= budget['min_tokens_per_s']
        # within_val_loss = candidate['val_loss'] <= quality_floor['max_val_loss']
        # is_feasible = within_memory and enough_throughput and within_val_loss
        # is_feasible 时追加 candidate，否则把 candidate['name'] 放入 rejected。
        pass
    best_candidate = min(feasible, key=lambda item: item['memory_mb'])['name'] if feasible else None
    return {
        'candidate_count': len(candidates),
        'feasible_count': len(feasible),
        'best_candidate': best_candidate,
        'feasible_candidates': [item['name'] for item in feasible],
        'rejected_candidates': rejected,
    }


def decide_qlora_project(summary: Dict[str, object]) -> Dict[str, object]:
    """根据可行候选汇总给出 QLoRA 选型结论。"""
    # 返回 decision、reason 和 next_action；accept 只表示当前预算下值得继续验证。
    # ==========================================
    # TODO 3：输出项目结论
    # 提示：读取 feasible_count、best_candidate、quality_failed_count。
    # 返回 decision / reason / next_action 三个字段。
    # 没有可行候选时 reject；QLoRA 是最优可行方案时 accept；否则通常进入 tune。
    # ==========================================
    feasible_count = summary.get('feasible_count', 0)
    best_candidate = summary.get('best_candidate')
    # TODO：没有可行候选时 reject。
    # TODO：best_candidate == 'qlora' 时 accept。
    # TODO：其余仍有可行候选时返回 tune。
    return {
        'decision': 'reject',
        'reason': '',
        'next_action': '',
    }

```


```python
# 测试你的实现
def test_qlora_selection_project():
    try:
        budget = {'memory_cap_mb': 12000.0, 'min_tokens_per_s': 18.0}
        quality_floor = {'max_val_loss': 1.20}
        assert validate_qlora_candidate({'name': 'broken', 'memory_mb': -1})
        check = validate_budget_and_quality(budget, quality_floor)
        assert check['is_valid'] is True, '预算检查应通过'
        assert check['missing_keys'] == [], '完整预算不应缺字段'
        ledger = build_memory_ledger(100.0, 10.0, 10.0, 40.0, 200.0, quant_metadata_mb=5.0, peak_memory_mb=380.0, peak_reserved_mb=420.0)
        assert ledger['estimated_total_mb'] == 365.0, '账本估算总量应等于各显存对象之和'
        assert ledger['peak_memory_mb'] == 380.0 and ledger['peak_reserved_mb'] == 420.0
        assert ledger['reconciliation_gap_mb'] == 15.0, '应保留估算与 CUDA 峰值的差值'
        try:
            build_memory_ledger(-1.0, 0.0, 0.0, 0.0, 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError('显存账本不应接受负数')

        candidates = [
            {'name': 'full_ft', 'memory_mb': 22000.0, 'tokens_per_s': 10.0, 'val_loss': 1.05},
            {'name': 'lora', 'memory_mb': 14500.0, 'tokens_per_s': 20.0, 'val_loss': 1.10},
            {'name': 'qlora', 'memory_mb': 9800.0, 'tokens_per_s': 19.0, 'val_loss': 1.16},
        ]
        summary = summarize_low_resource_candidates(candidates, budget, quality_floor)
        assert summary['feasible_count'] == 1, '只应有一个方案满足预算与质量'
        assert summary['best_candidate'] == 'qlora', 'QLoRA 应成为最优可行方案'

        decision = decide_qlora_project(summary)
        assert decision['decision'] == 'accept', '可行且最优的 QLoRA 应被接受'

        hard_summary = summarize_low_resource_candidates(
            [
                {'name': 'lora', 'memory_mb': 13000.0, 'tokens_per_s': 17.0, 'val_loss': 1.18},
                {'name': 'qlora', 'memory_mb': 11000.0, 'tokens_per_s': 19.0, 'val_loss': 1.28},
            ],
            budget,
            quality_floor,
        )
        hard_decision = decide_qlora_project(hard_summary)
        assert hard_decision['decision'] == 'reject', '没有满足质量下限时应 reject'
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


test_qlora_selection_project()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
def build_memory_ledger(base_weight_mb: float, trainable_param_mb: float, gradient_mb: float, optimizer_state_mb: float, activation_mb: float, quant_metadata_mb: float = 0.0, peak_memory_mb: float = None, peak_reserved_mb: float = None, evidence: str = 'estimated') -> Dict[str, object]:
    """汇总训练显存对象，并区分估算值与 CUDA 峰值。"""
    values = {
        'base_weight_mb': base_weight_mb, 'trainable_param_mb': trainable_param_mb,
        'gradient_mb': gradient_mb, 'optimizer_state_mb': optimizer_state_mb,
        'activation_mb': activation_mb, 'quant_metadata_mb': quant_metadata_mb,
    }
    if any(float(value) < 0 for value in values.values()):
        raise ValueError('显存账本中的各项不能为负数。')
    estimated_total_mb = sum(float(value) for value in values.values())
    report = {**values, 'estimated_total_mb': round(estimated_total_mb, 3), 'evidence': evidence}
    if peak_memory_mb is not None:
        report['peak_memory_mb'] = float(peak_memory_mb)
        report['reconciliation_gap_mb'] = round(float(peak_memory_mb) - estimated_total_mb, 3)
    if peak_reserved_mb is not None:
        report['peak_reserved_mb'] = float(peak_reserved_mb)
    return report

import math


# TODO 0: 检查候选字段，避免无效测量进入排序
def validate_qlora_candidate(candidate: Dict[str, float]) -> List[str]:
    errors = []
    required = ('name', 'memory_mb', 'tokens_per_s', 'val_loss')
    for key in required:
        if key not in candidate:
            errors.append(f'missing:{key}')
    if errors:
        return errors
    for key in ('memory_mb', 'tokens_per_s', 'val_loss'):
        try:
            value = float(candidate[key])
        except (TypeError, ValueError):
            errors.append(f'non_numeric:{key}')
            continue
        if not math.isfinite(value):
            errors.append(f'non_finite:{key}')
        if key in ('memory_mb', 'tokens_per_s') and value < 0:
            errors.append(f'negative:{key}')
    return errors


# TODO 1: 检查预算与质量下限
def validate_budget_and_quality(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    required_budget_keys = ['memory_cap_mb', 'min_tokens_per_s']
    required_quality_keys = ['max_val_loss']
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    for key in required_budget_keys:
        if key in budget and float(budget[key]) <= 0:
            missing_keys.append(f'invalid:{key}')
    if 'max_val_loss' in quality_floor and float(quality_floor['max_val_loss']) < 0:
        missing_keys.append('invalid:max_val_loss')
    return {
        'is_valid': len(missing_keys) == 0,
        'missing_keys': missing_keys,
    }


# TODO 2: 汇总低资源微调候选
def summarize_low_resource_candidates(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, object]) -> Dict[str, object]:
    feasible: List[Dict[str, float]] = []
    quality_failed = 0

    for candidate in candidates:
        errors = validate_qlora_candidate(candidate)
        if errors:
            raise ValueError(f'非法 QLoRA 候选 {candidate.get("name", "unknown")}: {errors}')
        memory_ok = candidate['memory_mb'] <= budget['memory_cap_mb']
        speed_ok = candidate['tokens_per_s'] >= budget['min_tokens_per_s']
        quality_ok = candidate['val_loss'] <= quality_floor['max_val_loss']
        if not quality_ok:
            quality_failed += 1
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    feasible.sort(key=lambda x: (x['memory_mb'], -x['tokens_per_s'], x['val_loss']))
    best_candidate = feasible[0]['name'] if feasible else None
    return {
        'candidate_count': len(candidates),
        'feasible_count': len(feasible),
        'best_candidate': best_candidate,
        'quality_failed_count': quality_failed,
        'feasible_names': [item['name'] for item in feasible],
    }


# TODO 3: 输出项目结论
def decide_qlora_project(summary: Dict[str, object]) -> Dict[str, object]:
    feasible_count = summary['feasible_count']
    best_candidate = summary['best_candidate']
    quality_failed_count = summary['quality_failed_count']

    if feasible_count == 0:
        return {
            'decision': 'reject',
            'reason': 'no_candidate_meets_budget_and_quality',
            'next_action': 'relax_budget_or_improve_quality',
        }
    if best_candidate == 'qlora':
        return {
            'decision': 'accept',
            'reason': 'qlora_is_best_feasible_option',
            'next_action': 'promote_to_training_run',
        }
    if quality_failed_count > 0:
        return {
            'decision': 'tune',
            'reason': 'qlora_needs_rank_or_quant_tuning',
            'next_action': 'adjust_rank_or_quantization_bits',
        }
    return {
        'decision': 'tune',
        'reason': 'qlora_not_best_under_current_budget',
        'next_action': 'revisit_target_modules_or_batch_plan',
    }

```

### 解析

这一页保留 `3` 个核心 TODO：预算检查、候选汇总和项目结论。它不要求把量化训练过程重写一遍，而是要求把低资源微调的预算约束收成清晰的选型判断。

**1. TODO 1: 检查预算与质量下限**
- **实现方式**：先把显存上限、吞吐下限和验证损失上限检查齐，再进入方案比较。
- **关键点**：没有统一预算口径时，候选方案的显存或吞吐比较都没有解释力。
- **项目意义**：这一步把 `65` 固定成预算约束下的选型页，而不是泛量化实验页。

**2. TODO 2: 汇总低资源微调候选**
- **实现方式**：按显存、吞吐和验证损失统一过滤候选，再选出最省显存的可行方案。
- **关键点**：QLoRA 只有在质量和吞吐都没跌出边界时，显存收益才有意义。
- **项目意义**：这一步把 `10 / 40 / 41` 的机制知识收成真正可比较的工程候选。

**3. TODO 3: 输出项目结论**
- **实现方式**：把候选可行性和最优方案统一收成 `accept / tune / reject`。
- **关键点**：项目结论必须回答“当前预算下 QLoRA 是否值得继续采用”，而不是只输出一个候选名字。
- **项目意义**：这一步把 `65` 收成低资源微调路线中的正式选型项目。

### 可选：统一项目报告导出
默认关闭。完成预算、吞吐和质量筛选后，再导出 QLoRA 选型报告。报告模板见 `docs/verification/fine_tuning_projects.md`。

```python
try:
    from tools.fine_tuning_project_runtime import preflight_runtime, runtime_snapshot, save_project_report, validate_project_config
except ModuleNotFoundError:
    preflight_runtime = lambda torch_module, run_mode='cpu', **kwargs: {'run_mode': run_mode, 'ready': False, 'reasons': ['共享运行时工具不可用']}
    runtime_snapshot = lambda: {'device': 'unknown'}
    validate_project_config = lambda config: []
    save_project_report = None
RUN_MODE = 'cpu'  # cpu / dry_run / real_gpu；真实 QLoRA 训练作为后续扩展。
PROJECT_ID = '65_qlora_selection'
PROJECT_RESULT_PATH = 'benchmarks/results/65_qlora_selection.json'
PROJECT_CONFIG = {'project': PROJECT_ID, 'model': 'template', 'dtype': 'fp32', 'batch_size': 1, 'seq_len': 128, 'steps': 1, 'seed': 42, 'run_mode': RUN_MODE}
RUN_PROJECT_EXPORT = False  # True 只保存已完成的 QLoRA 选型报告。
config_errors = validate_project_config(PROJECT_CONFIG)
if config_errors:
    raise ValueError('; '.join(config_errors))
print('runtime:', runtime_snapshot())
if RUN_MODE == 'dry_run':
    import importlib.util
    try:
        import torch
        preflight = preflight_runtime(torch, run_mode='dry_run')
    except ImportError as exc:
        preflight = {'run_mode': 'dry_run', 'ready': False, 'reasons': [f'缺少 torch：{exc}']}
    preflight['bitsandbytes_available'] = importlib.util.find_spec('bitsandbytes') is not None
    print('dry_run:', preflight)
if RUN_PROJECT_EXPORT:
    if 'PROJECT_REPORT' not in globals():
        raise RuntimeError('请先组装完整的 PROJECT_REPORT')
    PROJECT_REPORT.setdefault('project', PROJECT_ID)
    PROJECT_REPORT.setdefault('config', PROJECT_CONFIG)
    PROJECT_REPORT.setdefault('environment', runtime_snapshot())
    save_project_report(PROJECT_RESULT_PATH, PROJECT_REPORT)

```
