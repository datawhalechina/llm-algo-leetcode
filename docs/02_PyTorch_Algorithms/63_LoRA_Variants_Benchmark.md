# 63. LoRA Variants Benchmark | LoRA 变体选型项目

**难度：** Hard | **环境：** CPU + 单卡 GPU | **标签：** `监督微调`, `LoRA`, `候选选型` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节读取 64 节准入的数据清单，在固定模型、数据、训练预算和独立评测下比较 LoRA 候选。首先检查候选是否可比以及是否通过质量和资源门槛，再通过配对复测选出供 62 节训练使用的适配器配置。

**关键词：** `LoRA`, `单变量对照`, `独立评测`, `资源门槛`, `选型报告`

---
## 前置阅读

**导语：** 先理解 LoRA 更新范围、有效 batch 和真实数据准入，再比较候选配置。

- [Part 02 · 14 LoRA 目标模块与低秩更新](./14_LoRA_Tutorial.md)
- [Part 02 · 31 LoRA 变体与参数预算](./31_LoRA_Variants_Theory.md)
- [Part 02 · 64 SFT 数据质量项目](./64_SFT_Data_Quality_Project.md)

---
### Step 1：定义单变量候选与固定实验条件

LoRA 候选可以改变低秩容量、缩放、正则化或更新范围，但一轮实验只能选择其中一个维度作为变量。模型版本、数据切分、训练预算、独立评测集和随机种子集合保持一致，才能把结果差异归因到该变量；本项目的真实 GPU 对照只改变秩。

| 候选维度 | 机制作用 | 本轮处理 | 必须固定的相关条件 |
|:---|:---|:---|:---|
| 秩 `rank` | 改变低秩分支的容量和可训练参数量 | 作为唯一变量 | 缩放、正则化和目标模块 |
| 缩放 `alpha` | 改变低秩更新的缩放强度 | 保持不变 | 秩、正则化和目标模块 |
| 丢弃率 `dropout` | 改变低秩分支的正则化强度 | 保持不变 | 秩、缩放和目标模块 |
| 目标模块 `target_modules` | 改变模型中允许更新的参数范围 | 保持不变 | 秩、缩放和正则化 |
| 公共实验条件 | 决定候选是否能够直接比较 | 全部固定 | 模型与数据版本、训练步数、评测集和随机种子集合 |

![LoRA 变体的单变量实验](../public/02_PyTorch_Algorithms/63_lora_variant_experiment_flow.svg)

### Step 2：先检查可比性，再执行逐项准入

候选首先要通过可比性检查：固定条件不能漂移，并且只能改变声明的一个 LoRA 维度。可比候选再分别接受执行、质量和资源检查；这些都是硬门槛，任一失败都不能由另一项优势抵消。更新前评测用于发现候选起点异常，固定更新预算后的独立评测才用于比较训练结果。

| 判断层 | 检查对象 | 通过条件 | 失败动作 |
|:---|:---|:---|:---|
| 可比性 | 模型与数据版本、训练预算、评测集、种子集合和 LoRA 配置 | 固定条件一致，且恰好改变一个声明维度 | 整组对照失效，修正后重跑 |
| 执行准入 | 运行状态、失败记录和非有限值 | 必要运行全部有效 | 保留失败证据并补测 |
| 质量准入 | 固定预算后的独立评测损失 | 不超过预先声明的上限 | 淘汰候选 |
| 资源准入 | 峰值显存与监督词元吞吐 | 同时满足设备容量和速度下限 | 淘汰候选 |
| 门槛余量 | 指标与对应阈值的距离 | 每项分别记录，不合成为加权总分 | 根据失败项调整候选或预算 |

### Step 3：用共享随机种子完成配对选择

首轮准入只能排除明显不合格的候选，不能证明细小质量差能够复现。最终比较要求每个合格候选使用完全相同的随机种子集合；缺失、重复或失败的运行都会破坏配对关系。对每个共享种子计算候选间的评测损失差，再结合候选自身的重复极差判断结果是否稳定。

| 选择阶段 | 使用的证据 | 判断规则 | 输出状态 |
|:---|:---|:---|:---|
| 配对完整性 | 各候选的成功运行与随机种子 | 种子集合完全一致、无重复且达到最少次数 | 不完整则 `tune` |
| 重复稳定性 | 同一候选跨种子的评测损失 | 最大值与最小值之差不超过预设上限 | 波动过大则 `tune` |
| 配对质量差 | 每个共享种子上的候选损失差 | 损失差的中位数落入预设质量等价区间 | 进入资源比较 |
| 资源排序 | 峰值显存、步时和可训练参数量 | 质量等价时依次选择代价更低的候选 | 唯一候选则 `accept` |
| 无合格候选 | 逐项准入结果 | 没有候选同时通过全部硬门槛 | `reject` |

![LoRA 变体的 CPU 决策实现](../public/02_PyTorch_Algorithms/63_lora_cpu_todo_flow.svg)

### Step 4：实现可比性、准入与选择

CPU 题目使用一个基线配置、多个候选记录和多随机种子复测记录。学习者需要找出固定条件与 LoRA 配置的变化，逐项检查执行、质量和资源门槛，再根据共享随机种子的配对损失差判断质量是否等价。测试覆盖固定条件漂移、单项门槛失败、配对缺失、运行失败和重复波动。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：检查候选是否真正可比 | 固定字段与允许变化的配置字段 | 只允许一个声明变量发生变化 | 模型、数据、预算漂移与多变量变化 |
| TODO 2：逐项执行候选准入 | 质量、显存、吞吐与执行状态 | 保留每项门槛余量，不使用加权总分 | 阈值边界、显存越界与执行失败 |
| TODO 3：在质量等价区间内选择候选 | 共享随机种子上的配对损失差与资源记录 | 配对完整且稳定后，选择低代价方案 | 种子缺失、运行失败、波动过大与质量等价 |

```python
import statistics
from typing import Any, Dict, List
```


```python
# ==========================================
# TODO 1：检查候选是否保持单变量可比
# ==========================================
def check_candidate_comparability(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    fixed_fields: List[str],
    allowed_variant_fields: List[str],
) -> Dict[str, Any]:
    """报告固定字段漂移和未声明的 LoRA 配置变化。"""
    fixed_mismatches = [field for field in fixed_fields if baseline[field] != candidate[field]]
    changed_variant_fields = [
        field for field in allowed_variant_fields if baseline[field] != candidate[field]
    ]

    # 一轮实验必须恰好改变一个已声明的 LoRA 变量，且固定字段不能漂移。
    # comparable = ???  # 固定条件无漂移，且恰好改变一个声明的候选维度。
    raise NotImplementedError("TODO 1：请判断候选是否可比")
    return {
        "comparable": comparable,
        "fixed_mismatches": fixed_mismatches,
        "changed_variant_fields": changed_variant_fields,
    }


# ==========================================
# TODO 2：检查质量与资源硬门槛
# ==========================================
def evaluate_lora_candidate(candidate: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """逐项检查候选，不允许用加权分数抵消失败门槛。"""
    checks = {
        "execution": candidate["status"] == "ok",
        "quality": candidate["final_eval_loss"] <= gates["maximum_eval_loss"],
        "memory": candidate["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
        "throughput": candidate["supervised_tokens_per_second"] >= gates["minimum_tokens_per_second"],
    }

    margins = {
        "quality": gates["maximum_eval_loss"] - candidate["final_eval_loss"],
        "memory": gates["maximum_peak_allocated_bytes"] - candidate["peak_allocated_bytes"],
        "throughput": candidate["supervised_tokens_per_second"] - gates["minimum_tokens_per_second"],
    }
    # eligible = ???  # 所有硬门槛必须同时通过，不能相互抵消。
    raise NotImplementedError("TODO 2：请判断候选是否通过硬门槛")
    return {
        "name": candidate["name"],
        "eligible": eligible,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "margins": margins,
    }


# ==========================================
# TODO 3：用共享随机种子选择质量等价候选
# ==========================================
def select_lora_candidate(candidates: List[Dict[str, Any]], repeat_results: List[Dict[str, Any]], gates: Dict[str, float]) -> Dict[str, Any]:
    """检查配对复测完整性与稳定性，再从质量等价候选中选择低代价方案。"""
    evaluations = [evaluate_lora_candidate(candidate, gates) for candidate in candidates]
    eligible_names = {item["name"] for item in evaluations if item["eligible"]}
    eligible = [candidate for candidate in candidates if candidate["name"] in eligible_names]
    if not eligible:
        return {"decision": "reject", "selected": None, "evaluations": evaluations, "failed_checks": ["no_eligible_candidate"]}

    relevant_repeats = [item for item in repeat_results if item["name"] in eligible_names]
    if any(item["status"] != "ok" for item in relevant_repeats):
        return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["repeat_execution_failure"]}

    losses_by_candidate = {}
    for candidate in eligible:
        records = [item for item in relevant_repeats if item["name"] == candidate["name"]]
        seeds = [item["seed"] for item in records]
        if len(seeds) != len(set(seeds)):
            return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["duplicate_repeat_seed"]}
        if len(seeds) < gates["minimum_repeat_count"]:
            return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["paired_repeat_incomplete"]}
        losses_by_candidate[candidate["name"]] = {item["seed"]: item["final_eval_loss"] for item in records}

    seed_sets = [set(seed_losses) for seed_losses in losses_by_candidate.values()]
    if any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
        return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["paired_repeat_incomplete"]}
    shared_seeds = sorted(seed_sets[0])

    summaries = []
    for candidate in eligible:
        losses = [losses_by_candidate[candidate["name"]][seed] for seed in shared_seeds]
        summaries.append({**candidate, "repeat_median_loss": statistics.median(losses), "repeat_spread": max(losses) - min(losses)})
    if any(item["repeat_spread"] > gates["maximum_repeat_eval_loss_spread"] for item in summaries):
        return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["repeat_instability"]}

    best = min(summaries, key=lambda item: (item["repeat_median_loss"], item["name"]))
    best_losses = losses_by_candidate[best["name"]]
    for item in summaries:
        deltas = [losses_by_candidate[item["name"]][seed] - best_losses[seed] for seed in shared_seeds]
        item["paired_delta_median"] = statistics.median(deltas)
    equivalent = [item for item in summaries if abs(item["paired_delta_median"]) <= gates["quality_equivalence_band"]]
    # 质量等价时，优先选择峰值显存和步时更低的候选。
    # selected = ???  # 在质量等价候选中依次最小化显存、步时和名称。
    raise NotImplementedError("TODO 3：请选择低代价的稳定候选")
    return {"decision": "accept", "selected": selected, "evaluations": evaluations, "repeat_summaries": summaries, "shared_seeds": shared_seeds, "failed_checks": []}
```


```python
def _lora_selection_fixture():
    """构造候选、门槛和配对复测的共享样例。"""
    baseline = {
        "model_commit": "m1", "data_commit": "d1", "steps": 12, "seed": 42,
        "rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    gates = {
        "maximum_eval_loss": 1.30,
        "maximum_peak_allocated_bytes": 3_000,
        "minimum_tokens_per_second": 100.0,
        "minimum_repeat_count": 2,
        "maximum_repeat_eval_loss_spread": 0.02,
        "quality_equivalence_band": 0.03,
    }
    candidates = [
        {"name": "rank4", "status": "ok", "final_eval_loss": 1.24, "peak_allocated_bytes": 2_000, "supervised_tokens_per_second": 130.0, "median_step_time_ms": 50.0},
        {"name": "rank8", "status": "ok", "final_eval_loss": 1.18, "peak_allocated_bytes": 2_500, "supervised_tokens_per_second": 120.0, "median_step_time_ms": 55.0},
        {"name": "rank16", "status": "ok", "final_eval_loss": 1.10, "peak_allocated_bytes": 3_500, "supervised_tokens_per_second": 110.0, "median_step_time_ms": 60.0},
    ]
    repeats = [
        {"name": "rank4", "seed": 11, "status": "ok", "final_eval_loss": 1.20},
        {"name": "rank4", "seed": 29, "status": "ok", "final_eval_loss": 1.21},
        {"name": "rank8", "seed": 11, "status": "ok", "final_eval_loss": 1.18},
        {"name": "rank8", "seed": 29, "status": "ok", "final_eval_loss": 1.19},
    ]
    return baseline, gates, candidates, repeats


def test_candidate_comparability():
    """验证单变量变化通过，固定预算漂移被拒绝。"""
    baseline, _, _, _ = _lora_selection_fixture()
    rank16 = {**baseline, "rank": 16}
    report = check_candidate_comparability(
        baseline, rank16,
        fixed_fields=["model_commit", "data_commit", "steps", "seed"],
        allowed_variant_fields=["rank", "alpha", "dropout", "target_modules"],
    )
    assert report["comparable"] and report["changed_variant_fields"] == ["rank"]
    assert not check_candidate_comparability(
        baseline, {**rank16, "steps": 20},
        ["model_commit", "data_commit", "steps", "seed"],
        ["rank", "alpha", "dropout", "target_modules"],
    )["comparable"]
    assert not check_candidate_comparability(
        baseline, {**baseline, "rank": 16, "alpha": 32},
        ["model_commit", "data_commit", "steps", "seed"],
        ["rank", "alpha", "dropout", "target_modules"],
    )["comparable"]



def test_candidate_hard_gates():
    """验证资源失败不能被更好的质量值抵消。"""
    _, gates, candidates, _ = _lora_selection_fixture()
    assert evaluate_lora_candidate(candidates[2], gates)["failed_checks"] == ["memory"]


def test_quality_equivalent_selection():
    """验证共享种子配对、失败处理和质量等价带内的资源选择。"""
    _, gates, candidates, repeats = _lora_selection_fixture()
    selection = select_lora_candidate(candidates, repeats, gates)
    assert selection["decision"] == "accept"
    assert selection["selected"]["name"] == "rank4"
    assert selection["shared_seeds"] == [11, 29]
    assert select_lora_candidate([{**candidates[0], "status": "failed"}], [], gates)["decision"] == "reject"
    missing_pair = [item for item in repeats if not (item["name"] == "rank8" and item["seed"] == 29)]
    assert select_lora_candidate(candidates, missing_pair, gates)["failed_checks"] == ["paired_repeat_incomplete"]
    failed_repeat = [{**item, "status": "failed"} if item["name"] == "rank8" and item["seed"] == 29 else item for item in repeats]
    assert select_lora_candidate(candidates, failed_repeat, gates)["failed_checks"] == ["repeat_execution_failure"]
    unstable = [
        {**repeats[0], "final_eval_loss": 1.10}, {**repeats[1], "final_eval_loss": 1.20},
        {**repeats[2], "final_eval_loss": 1.08}, {**repeats[3], "final_eval_loss": 1.18},
    ]
    assert select_lora_candidate(candidates, unstable, gates)["failed_checks"] == ["repeat_instability"]


test_candidate_comparability()
test_candidate_hard_gates()
test_quality_equivalent_selection()
print("所有测试通过！")
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
# TODO 1：检查候选是否保持单变量可比
def check_candidate_comparability(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    fixed_fields: List[str],
    allowed_variant_fields: List[str],
) -> Dict[str, Any]:
    """报告固定字段漂移和未声明的 LoRA 配置变化。"""
    fixed_mismatches = [field for field in fixed_fields if baseline[field] != candidate[field]]
    changed_variant_fields = [field for field in allowed_variant_fields if baseline[field] != candidate[field]]
    comparable = not fixed_mismatches and len(changed_variant_fields) == 1
    return {"comparable": comparable, "fixed_mismatches": fixed_mismatches, "changed_variant_fields": changed_variant_fields}


# TODO 2：检查质量与资源硬门槛
def evaluate_lora_candidate(candidate: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """逐项检查候选，不允许用加权分数抵消失败门槛。"""
    checks = {
        "execution": candidate["status"] == "ok",
        "quality": candidate["final_eval_loss"] <= gates["maximum_eval_loss"],
        "memory": candidate["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"],
        "throughput": candidate["supervised_tokens_per_second"] >= gates["minimum_tokens_per_second"],
    }
    margins = {
        "quality": gates["maximum_eval_loss"] - candidate["final_eval_loss"],
        "memory": gates["maximum_peak_allocated_bytes"] - candidate["peak_allocated_bytes"],
        "throughput": candidate["supervised_tokens_per_second"] - gates["minimum_tokens_per_second"],
    }
    eligible = all(checks.values())
    return {"name": candidate["name"], "eligible": eligible, "failed_checks": [name for name, passed in checks.items() if not passed], "margins": margins}


# TODO 3：用共享随机种子选择质量等价候选
def select_lora_candidate(candidates: List[Dict[str, Any]], repeat_results: List[Dict[str, Any]], gates: Dict[str, float]) -> Dict[str, Any]:
    """检查配对复测完整性与稳定性，再从质量等价候选中选择低代价方案。"""
    evaluations = [evaluate_lora_candidate(candidate, gates) for candidate in candidates]
    eligible_names = {item["name"] for item in evaluations if item["eligible"]}
    eligible = [candidate for candidate in candidates if candidate["name"] in eligible_names]
    if not eligible:
        return {"decision": "reject", "selected": None, "evaluations": evaluations, "failed_checks": ["no_eligible_candidate"]}
    relevant_repeats = [item for item in repeat_results if item["name"] in eligible_names]
    if any(item["status"] != "ok" for item in relevant_repeats):
        return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["repeat_execution_failure"]}
    losses_by_candidate = {}
    for candidate in eligible:
        records = [item for item in relevant_repeats if item["name"] == candidate["name"]]
        seeds = [item["seed"] for item in records]
        if len(seeds) != len(set(seeds)):
            return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["duplicate_repeat_seed"]}
        if len(seeds) < gates["minimum_repeat_count"]:
            return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["paired_repeat_incomplete"]}
        losses_by_candidate[candidate["name"]] = {item["seed"]: item["final_eval_loss"] for item in records}
    seed_sets = [set(seed_losses) for seed_losses in losses_by_candidate.values()]
    if any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
        return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["paired_repeat_incomplete"]}
    shared_seeds = sorted(seed_sets[0])
    summaries = []
    for candidate in eligible:
        losses = [losses_by_candidate[candidate["name"]][seed] for seed in shared_seeds]
        summaries.append({**candidate, "repeat_median_loss": statistics.median(losses), "repeat_spread": max(losses) - min(losses)})
    if any(item["repeat_spread"] > gates["maximum_repeat_eval_loss_spread"] for item in summaries):
        return {"decision": "tune", "selected": None, "evaluations": evaluations, "failed_checks": ["repeat_instability"]}
    best = min(summaries, key=lambda item: (item["repeat_median_loss"], item["name"]))
    best_losses = losses_by_candidate[best["name"]]
    for item in summaries:
        deltas = [losses_by_candidate[item["name"]][seed] - best_losses[seed] for seed in shared_seeds]
        item["paired_delta_median"] = statistics.median(deltas)
    equivalent = [item for item in summaries if abs(item["paired_delta_median"]) <= gates["quality_equivalence_band"]]
    selected = min(equivalent, key=lambda item: (item["peak_allocated_bytes"], item["median_step_time_ms"], item["name"]))
    return {"decision": "accept", "selected": selected, "evaluations": evaluations, "repeat_summaries": summaries, "shared_seeds": shared_seeds, "failed_checks": []}
```

### 解析

#### TODO 1：检查候选可比性

- **判断依据**：模型、数据、预算或随机种子变化都会破坏归因；同一轮同时改变多个 LoRA 字段也无法说明收益来源。
- **实现动作**：分别列出固定字段漂移和 LoRA 字段变化，要求前者为空且后者恰好一个。
- **测试证据**：测试覆盖合法秩对照和训练步数变化。

#### TODO 2：执行硬门槛检查

- **判断依据**：执行、质量、显存和吞吐任一失败都不能被其他指标抵消。
- **实现动作**：逐项检查门槛并保留距离阈值的余量，避免只留下通过或失败。
- **测试证据**：质量最好的秩 16 候选因显存越界被排除。

#### TODO 3：用共享随机种子选择质量等价候选

- **判断依据**：只有共享同一随机种子的运行才能形成配对差；缺失、重复或失败的运行都不能支持候选排名。
- **实现动作**：先核对种子集合和重复极差，再计算相对最佳候选的逐种子损失差；差值中位数进入质量等价区间后，才按显存、步时和名称稳定选择。
- **测试证据**：测试覆盖质量等价但资源不同、无合格候选、配对缺失、运行失败和重复波动过大。
## Step 5：真实 LoRA 候选选型

CPU 题目区验证候选可比性、质量等价带和资源排序；真实实验在固定模型、数据、更新预算与共享随机种子下比较三个 LoRA 秩候选。

| 对照层 | 输入 | 主要证据 | 输出与证据边界 |
|:---|:---|:---|:---|
| CPU 选型器 | 人工构造的候选与重复结果 | 可比性、质量等价区间、资源排序 | 只确认选型规则，不判断真实候选 |
| 真实候选矩阵 | 固定数据、模型与多组配对随机种子 | 评测损失分布、显存、步时、吞吐与适配器 | 输出选型清单；不外推到其他模型或任务 |

### 5.1 环境、输入资产与固定条件

按[后训练优化 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart)准备 `PT-SFT-CUDA` 环境。12 GB 单卡使用 Qwen2.5-0.5B、长度 256、三个秩候选和相同更新预算；选中候选实测峰值分配约 1.77 GiB。

```python
from pathlib import Path
import json

from tools.project_runtime import resolve_project_root

RUN_MODE = "dry_run"  # 默认值 dry_run：只检查配置；改为 real_gpu 执行候选矩阵。
RUN_LABEL = "lora_variant_selection_v2"  # 已登记标签；正式复测时改为唯一标签并保留旧结果。
PROFILE_ID = "PT-SFT-CUDA"  # 与 64、65、62 共用 SFT 项目环境。
SEMANTIC_ID = "PT-LORA-VARIANTS"  # 项目稳定语义身份。
SCHEMA_VERSION = "fine-tuning-project/v1"

PROJECT_ROOT = resolve_project_root()
UPSTREAM_MANIFEST = PROJECT_ROOT / "benchmarks/results/64_sft_data_quality/sft_data_manifest_alpaca_qwen25_v2.json"  # 64 节带切分身份的交付清单。
MODEL_ID = "Qwen/Qwen2.5-0.5B"  # 12 GB 单卡默认模型。
MODEL_REVISION = "main"  # 运行时解析为不可变提交哈希。
MAX_LENGTH = 256  # 与上游审计长度一致。
TRAIN_EXAMPLES = 48  # 固定候选训练样本数。
EVALUATION_EXAMPLES = 16  # 固定独立评测样本数。
UPDATE_STEPS = 12  # 所有候选使用相同更新步数。
LEARNING_RATE = 2e-4  # 所有候选固定学习率。
DTYPE = "bfloat16"  # 不支持 BF16 时显式改为 float16。
RUN_SEEDS = [42, 123, 2024]  # 每个候选使用相同随机种子集合，形成配对重复。

BASE_VARIANT = {"name": "rank8", "rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
CANDIDATES = [
    {"name": "rank4", "rank": 4, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    BASE_VARIANT,
    {"name": "rank16", "rank": 16, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
]
GATES = {
    "maximum_eval_loss": 2.0,
    "maximum_peak_allocated_bytes": 10 * 1024**3,
    "minimum_tokens_per_second": 1.0,
    "minimum_repeat_count": 3,
    "maximum_repeat_eval_loss_spread": 0.01,
    "quality_equivalence_band": 0.01,
}

RESULTS_DIR = PROJECT_ROOT / "benchmarks/results/63_lora_variants"
SWEEP_PATH = RESULTS_DIR / f"lora_variant_sweep_{RUN_LABEL}.json"
DELIVERY_PATH = RESULTS_DIR / f"lora_variant_delivery_{RUN_LABEL}.json"
ADAPTERS_DIR = RESULTS_DIR / f"adapters_{RUN_LABEL}"
print({"run_mode": RUN_MODE, "sweep_path": str(SWEEP_PATH), "candidate_count": len(CANDIDATES)})
```

### 5.2 预检环境与数据准入清单

预检要求 CUDA、Transformers、PEFT 与 Datasets 可用，并拒绝状态不是 `accept` 的 64 节结果。

```python
import torch
from tools.project_runtime import environment_preflight, require_input_file
from tools.post_training_project_runtime import hash_directory, run_lora_candidate, save_json_record

preflight = environment_preflight(
    torch,
    required_packages=("transformers", "peft", "datasets", "huggingface_hub"),
    require_gpu=RUN_MODE == "real_gpu",
    output_path=SWEEP_PATH,
    environment_profile=PROFILE_ID,
)
if not preflight["ready"]:
    raise RuntimeError(preflight["reasons"])
upstream_path = require_input_file(UPSTREAM_MANIFEST, "64 节数据准入清单")
upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
if upstream["decision"] != "accept":
    raise RuntimeError(f"上游数据未通过准入：{upstream['decision']}")
for path in (DELIVERY_PATH, ADAPTERS_DIR):
    if RUN_MODE == "real_gpu" and path.exists():
        raise FileExistsError(f"输出已存在，请修改 RUN_LABEL：{path}")
print({"preflight": preflight["status"], "upstream_decision": upstream["decision"], "existing_sweep": SWEEP_PATH.exists()})
```

### 5.3 固定候选与门槛

| 项目 | 固定条件 |
|:---|:---|
| 上游数据 | 64 节 `accept` 数据准入清单及其数据集提交哈希 |
| 候选变量 | 仅改变秩（`rank`），其余 LoRA 字段固定 |
| 比较预算 | 相同样本、更新步数、学习率、长度和配对随机种子集合 |
| 硬门槛 | 每次执行成功，评测损失、峰值显存和监督词元吞吐通过 |
| 稳定性 | 每个候选重复次数达标，评测损失极差不超过阈值 |
| 选择规则 | 先进入最佳质量的等价带，再比较显存和步时 |
### 5.4 执行候选筛选并保存完整证据

每个候选都使用同一组随机种子重新加载基础模型。第一组重复同时保存该候选的适配器，其余重复采集质量与资源证据，使候选比较与最终交付使用各自独立的记录。

```python
if RUN_MODE == "real_gpu":
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    model_commit = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    dataset_info = upstream["dataset"]
    source_count = upstream["workload"]["sample_count"]
    dataset = load_dataset(dataset_info["id"], split=f"train[:{source_count}]", revision=dataset_info["resolved_commit"])
    snapshot = upstream["snapshot"]
    train_records = [dict(dataset[index]) for index in snapshot["train_record_indices"][:TRAIN_EXAMPLES]]
    evaluation_records = [dict(dataset[index]) for index in snapshot["evaluation_record_indices"][:EVALUATION_EXAMPLES]]
    if len(train_records) != TRAIN_EXAMPLES or len(evaluation_records) != EVALUATION_EXAMPLES:
        raise RuntimeError("64 节准入快照不足以满足固定训练/评测工作负载")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_commit)
    candidate_results = []
    for candidate in CANDIDATES:
        for repeat_index, run_seed in enumerate(RUN_SEEDS):
            result = run_lora_candidate(
                model_id=MODEL_ID, revision=model_commit, tokenizer=tokenizer,
                train_records=train_records, evaluation_records=evaluation_records,
                candidate=candidate, max_length=MAX_LENGTH, update_steps=UPDATE_STEPS,
                learning_rate=LEARNING_RATE, seed=run_seed, dtype_name=DTYPE,
                save_directory=ADAPTERS_DIR / candidate["name"] if repeat_index == 0 else None,
            )
            result["run_seed"] = run_seed
            candidate_results.append(result)
    sweep = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "environment_profile": PROFILE_ID, "run_mode": RUN_MODE, "run_label": RUN_LABEL,
        "upstream_manifest": str(UPSTREAM_MANIFEST), "upstream_sample_ids_hash": upstream["audit"]["sample_ids_hash"],
        "model": {"id": MODEL_ID, "requested_revision": MODEL_REVISION, "resolved_commit": model_commit},
        "workload": {"max_length": MAX_LENGTH, "train_examples": TRAIN_EXAMPLES, "evaluation_examples": EVALUATION_EXAMPLES, "update_steps": UPDATE_STEPS, "learning_rate": LEARNING_RATE, "run_seeds": RUN_SEEDS},
        "gates": GATES, "candidates": candidate_results,
    }
    save_json_record(SWEEP_PATH, sweep)
    print({"sweep_path": str(SWEEP_PATH), "run_count": len(candidate_results), "candidate_names": [item["name"] for item in CANDIDATES]})
else:
    print({"run_mode": RUN_MODE, "next_action": "确认候选和门槛后改为 real_gpu"})
```

### 5.5 独立读取候选证据并形成选择

从保存的 JSON 重新加载候选名称、运行状态和共享随机种子集合，再检查质量与资源硬门槛。通过门槛的候选按共享随机种子计算配对损失差；差值落入质量等价区间后，再比较峰值显存和步时。

```python
def select_lora_candidate_for_run(results, gates, expected_names, expected_seeds):
    """核对候选与共享种子矩阵，再按质量等价和资源成本选择候选。"""
    import statistics

    grouped = {}
    for result in results:
        grouped.setdefault(result["name"], []).append(result)
    if set(grouped) != set(expected_names):
        return {"decision": "tune", "selected": None, "failed_checks": ["candidate_set_mismatch"], "summaries": []}

    summaries = []
    losses_by_candidate = {}
    for name in expected_names:
        repeats = grouped[name]
        successful = [item for item in repeats if item["status"] == "ok"]
        seeds = [item["run_seed"] for item in successful]
        if len(successful) != len(repeats):
            return {"decision": "tune", "selected": None, "failed_checks": ["repeat_execution_failure"], "summaries": summaries}
        if len(seeds) != len(set(seeds)) or set(seeds) != set(expected_seeds):
            return {"decision": "tune", "selected": None, "failed_checks": ["paired_repeat_incomplete"], "summaries": summaries}
        losses_by_candidate[name] = {item["run_seed"]: item["final_eval_loss"] for item in successful}
        losses = [item["final_eval_loss"] for item in successful]
        checks = {
            "repeat_count": len(successful) >= gates["minimum_repeat_count"],
            "quality": bool(losses) and all(loss <= gates["maximum_eval_loss"] for loss in losses),
            "memory": bool(successful) and all(item["peak_allocated_bytes"] <= gates["maximum_peak_allocated_bytes"] for item in successful),
            "throughput": bool(successful) and all(item["supervised_tokens_per_second"] >= gates["minimum_tokens_per_second"] for item in successful),
            "stability": len(losses) >= 2 and max(losses) - min(losses) <= gates["maximum_repeat_eval_loss_spread"],
        }
        summaries.append({
            "name": name, "eligible": all(checks.values()),
            "failed_checks": [key for key, passed in checks.items() if not passed],
            "repeat_count": len(successful), "repeat_seeds": sorted(seeds),
            "median_eval_loss": statistics.median(losses) if losses else None,
            "eval_loss_spread": max(losses) - min(losses) if len(losses) >= 2 else None,
            "median_peak_allocated_bytes": statistics.median(item["peak_allocated_bytes"] for item in successful) if successful else None,
            "median_step_time_ms": statistics.median(item["median_step_time_ms"] for item in successful) if successful else None,
        })
    eligible = [item for item in summaries if item["eligible"]]
    if not eligible:
        return {"decision": "reject", "selected": None, "failed_checks": ["no_eligible_candidate"], "summaries": summaries}
    best = min(eligible, key=lambda item: (item["median_eval_loss"], item["name"]))
    best_losses = losses_by_candidate[best["name"]]
    for item in eligible:
        deltas = [losses_by_candidate[item["name"]][seed] - best_losses[seed] for seed in expected_seeds]
        item["paired_delta_median"] = statistics.median(deltas)
    equivalent = [item for item in eligible if abs(item["paired_delta_median"]) <= gates["quality_equivalence_band"]]
    selected = min(equivalent, key=lambda item: (item["median_peak_allocated_bytes"], item["median_step_time_ms"], item["name"]))
    return {"decision": "accept", "selected": selected, "failed_checks": [], "shared_seeds": list(expected_seeds), "quality_floor": best["median_eval_loss"], "equivalent_candidates": [item["name"] for item in equivalent], "summaries": summaries}

if SWEEP_PATH.exists():
    sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    selection = select_lora_candidate_for_run(sweep["candidates"], sweep["gates"], [item["name"] for item in CANDIDATES], sweep["workload"]["run_seeds"])
    print({
        "decision": selection["decision"],
        "selected": selection["selected"]["name"] if selection["selected"] else None,
        "quality_equivalent": selection.get("equivalent_candidates", []),
        "summaries": selection["summaries"],
    })
```

### 5.6 形成选型清单并保存适配器配置

根据 5.4 保存的配对重复证据和适配器，生成包含选择理由、候选摘要和适配器哈希的 JSON 清单。62 节以该清单中的唯一候选启动正式训练。

| 决策项 | 实测证据 | 结果 | 后续动作 |
|:---|:---|:---|:---|
| 配对重复 | rank 4 / 8 / 16 均使用种子 42、123、2024 | 通过 | 保留完整 3 × 3 矩阵 |
| 质量等价 | 三个候选的配对损失差中位数均位于 0.01 等价带内 | 通过 | 再比较资源成本 |
| 资源排序 | rank 4 的峰值分配显存和中位步时最低 | 选中 rank 4 | 交给 62 正式训练 |
| 项目决策 | 候选矩阵、适配器与哈希完整 | `accept` | 保存选型清单 |

`lora_variant_selection_v2` 已完成三个秩候选、三个共享随机种子的配对重复。三个候选均通过质量、稳定性、吞吐和显存门槛，并进入 0.01 的质量等价带；rank 4 的中位峰值分配显存约 1.76 GiB，中位步时约 35.05 ms，因此在当前固定负载下选中 rank 4，结论为 `accept`。该选择只适用于本次模型、数据和更新预算。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 63 节 LoRA 秩选型；`lora_variant_selection_v2` |
| 运行条件 | rank 4 / 8 / 16；共享种子 42、123、2024；固定模型、数据和更新预算 |
| 原始证据 | [候选扫描](../../benchmarks/results/63_lora_variants/lora_variant_sweep_lora_variant_selection_v2.json) |
| 正式证据 | 候选扫描内保存配对质量、稳定性、吞吐和显存指标 |
| 清单与交接 | [选型清单](../../benchmarks/results/63_lora_variants/lora_variant_delivery_lora_variant_selection_v2.json)；本地 adapters 由清单记录 |
| 结论 | `accept`：当前固定负载选择 rank 4，交给 62 节正式训练 |

```python
if RUN_MODE == "real_gpu":
    sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    selection = select_lora_candidate_for_run(sweep["candidates"], sweep["gates"], [item["name"] for item in CANDIDATES], sweep["workload"]["run_seeds"])
    if selection["decision"] != "accept":
        failed_delivery = {
            "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
            "run_label": RUN_LABEL, "decision": selection["decision"],
            "decision_details": {"decision": selection["decision"], "failed_checks": selection["failed_checks"], "summaries": selection["summaries"]},
            "selection_evidence": str(SWEEP_PATH), "upstream_manifest": str(UPSTREAM_MANIFEST),
            "next_consumer": None,
        }
        save_json_record(DELIVERY_PATH, failed_delivery)
        raise RuntimeError(f"候选选择未通过：{selection['decision']}，{selection['failed_checks']}；交付记录已保存")
    selected_name = selection["selected"]["name"]
    selected_config = next(item for item in CANDIDATES if item["name"] == selected_name)
    selected_adapter_dir = ADAPTERS_DIR / selected_name
    adapter_hash = hash_directory(selected_adapter_dir) if selected_adapter_dir.is_dir() else None
    delivery_decision = "accept" if adapter_hash else "reject"
    delivery = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "run_label": RUN_LABEL, "decision": delivery_decision, "selected_config": selected_config,
        "decision_details": {"decision": delivery_decision, "failed_checks": [] if adapter_hash else ["selected_adapter_missing"], "quality_equivalent": selection["equivalent_candidates"], "candidate_summaries": selection["summaries"]},
        "selection_evidence": str(SWEEP_PATH), "upstream_manifest": str(UPSTREAM_MANIFEST),
        "model": sweep["model"], "workload": sweep["workload"], "repeat_seeds": sweep["workload"]["run_seeds"],
        "candidates": sweep["candidates"],
        "quality": {"median_eval_loss": selection["selected"]["median_eval_loss"], "repeat_eval_loss_spread": selection["selected"]["eval_loss_spread"]},
        "resources": {"selected_peak_allocated_bytes": selection["selected"]["median_peak_allocated_bytes"], "selected_median_step_time_ms": selection["selected"]["median_step_time_ms"]},
        "repeat_results": [item for item in sweep["candidates"] if item["name"] == selected_name], "artifact": {"role": "training.lora_selection", "adapter_path": str(selected_adapter_dir), "sha256": adapter_hash},
        "next_consumer": "PT-SFT-DELIVERY" if delivery_decision == "accept" else None,
    }
    save_json_record(DELIVERY_PATH, delivery)
    print({"delivery_path": str(DELIVERY_PATH), "decision": delivery_decision, "adapter_path": str(selected_adapter_dir), "adapter_sha256": adapter_hash})
```

## 相关阅读

- [LoRA 原论文](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT 官方文档](https://huggingface.co/docs/peft)
- [Part 02 · 62 SFT 训练交付项目](./62_Instruction_Fine_Tuning_Project.md)
- [Part 02 · 65 QLoRA 低资源选型](./65_QLoRA_Selection_Project.md)
