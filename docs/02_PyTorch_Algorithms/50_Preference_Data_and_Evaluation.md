# 50. Preference Data and Evaluation | 偏好数据与对齐评测

**难度：** Medium | **环境：** [CPU 机制练习](#runtime-environment) | **标签：** `后训练对齐`, `偏好数据`, `评测协议` | **目标人群：** 希望把 DPO / GRPO 结果解释清楚的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

DPO 和 Online DPO 直接使用同一任务下的优选 / 非优选回答对；PPO、GRPO 与 RLVR 使用轨迹和奖励更新策略。两类训练信号不同，但训练损失下降或奖励上升都不能直接说明候选模型更有帮助、更安全或更可靠，最终仍要与固定基线在隔离评测集上比较。

可信的偏好数据需要回答三个问题：两个回答是否真的可比，训练数据是否与评测数据隔离，自动评审器重复判断和交换回答顺序后是否稳定。候选决策还要同时检查预先规定的安全指标、有效样本量、评测覆盖率、明确判决数和非平局胜率。

**关键词：** `优选 / 非优选回答`, `轨迹记录`, `评测切分`, `非平局胜率`, `评审一致性`

---
<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。
- **框架核对：** 使用 [`PT-CPU`](../topic_discussion/post_training_optimization/quickstart.md#pt-cpu-quickstart)，运行 [框架一致性核对](#framework-check)。

---
## 前置阅读

**导语：** 先判断当前问题是否需要偏好优化，再建立训练和评测共用的数据身份。

- [Part 02 · 33 微调准入与最小干预](./33_Fine_Tuning_Readiness.md)

---
## Step 1：一条偏好对怎样通过可比性与切分检查

偏好学习的最小单位不是两段孤立文本，而是同一任务与约束下的一次可比选择。比较对象、偏好依据、来源身份和数据切分必须同时成立；训练集与最终评测集还要使用规范化后的任务输入检查重复。工具任务需要把动作、观察、终态和验证结论绑定到同一次执行。任一关系失效，偏好信号就可能混入题目差异、标注噪声或评测泄漏。

| 契约层次 | 需要成立的关系 | 失效后的影响 |
|:---|:---|:---|
| 比较对象 | 两个回答来自同一任务、输入和约束 | 比较结果反映题目差异而非回答质量 |
| 偏好语义 | 选择理由与目标能力一致 | 标注噪声被当作训练信号 |
| 来源身份 | 生成来源、标注规范和判分依据可追溯 | 争议样本无法复核 |
| 切分隔离 | 训练、验证与最终评测之间不存在规范化后重复的任务输入 | 记忆效应被误判为能力提升 |
| 轨迹证据（可选） | 动作、观察、终态与验证结论属于同一次执行 | 奖励与任务完成情况失去因果联系 |

![偏好样本从回答对、身份审计和切分隔离进入训练与评测](../public/02_PyTorch_Algorithms/50_preference_data_contract.svg)

## Step 2：如何区分评测缺失、平局与候选胜负

基线与候选需要面对相同评测样本、生成参数和判分规则。回答展示顺序应随机化，并通过交换顺序后的重映射结果检查位置偏差。统计时分别保留预期样本数、实际完成数、明确判决数、平局数和非平局胜率，避免把平局误当成评测缺失。

设计划评测数为 $n_{expected}$，实际完成数为 $n_{evaluated}$，候选获胜、基线获胜和平局数分别为 $n_{candidate}$、$n_{baseline}$ 与 $n_{tie}$。覆盖率检查评测是否完成，明确判决率区分平局与缺失，非平局胜率只在已经形成胜负的样本中比较候选：

$$
n_{evaluated}=n_{candidate}+n_{baseline}+n_{tie}, \qquad
C_{eval}=\frac{n_{evaluated}}{n_{expected}}, \qquad
R_{decision}=\frac{n_{candidate}+n_{baseline}}{n_{evaluated}}, \qquad
W_{non\text{-}tie}=\frac{n_{candidate}}{n_{candidate}+n_{baseline}}.
$$

计算前必须确认相应分母大于零；$n_{tie}$ 属于已完成评测，因此进入覆盖率和明确判决率口径，但不进入非平局胜率分母。重复判分检查同一输入下的稳定性，交换回答位置检查位置偏差。令 $j_i^{(1)}$、$j_i^{(2)}$ 分别表示第 $i$ 个偏好对的首次与重复判分，$\widetilde{j}_i^{(swap)}$ 表示交换位置后还原到原方向的判分，$\mathbf 1(\cdot)$ 是指示函数，则：

$$
C_{repeat}=\frac{1}{N}\sum_{i=1}^{N}\mathbf 1\!\left(j_i^{(1)}=j_i^{(2)}\right), \qquad
C_{swap}=\frac{1}{N}\sum_{i=1}^{N}\mathbf 1\!\left(j_i^{(1)}=\widetilde{j}_i^{(swap)}\right), \qquad
C_{judge}=\min(C_{repeat},C_{swap}).
$$

这里 $N$ 是参与一致性核对的偏好对数量，$C_{judge}$ 取两项检查的较低值，作为进入候选门禁的保守一致性证据。

| 观察量 | 需要保留的证据 | 决策作用 |
|:---|:---|:---|
| 评测覆盖率 | 计划数、完成数与未完成原因 | 判断评测是否按计划完成 |
| 明确判决率 | 候选胜、基线胜与平局计数 | 区分大量平局与评测缺失 |
| 非平局胜率 | 候选胜与基线胜计数 | 衡量候选在明确判决样本中的相对胜负 |
| 安全与失败分布 | 分别统计安全检查失败，以及各任务类别和风险类别的指标下降 | 任一安全指标低于阈值时拒绝候选 |
| 判分一致性 | 比较重复判分和交换顺序后的重映射结果 | 不稳定时进入复核而非直接准入 |
| 工具执行结果（可选） | 检查工具执行结束时的成功状态、输出和验证结果 | 防止文本偏好掩盖执行失败 |

## Step 3：哪些证据不足会触发复核或拒绝

数据检查失败，或任一预先规定的安全指标低于阈值时直接拒绝。前两项检查通过后，只要实际完成样本数、评测覆盖率、明确判决数、非平局胜率或判分一致性有一项不足，就进入复核并记录具体原因；这些条件全部达到阈值时才允许准入。

| 证据组合 | 状态 | 下一步 |
|:---|:---|:---|
| 数据与安全通过，样本数、覆盖率、明确判决数、胜率和一致性均达到门槛 | 准入 | 保留评测身份、样本规模与失败案例 |
| 数据与安全检查通过，但样本规模、覆盖率、明确判决数、胜率或一致性不足 | 复核 | 根据失败原因补样本、人工复核或重复评测 |
| 数据契约失效或任一安全指标低于阈值 | 拒绝 | 修复数据或候选后重新评测 |
| 文本质量改善，但工具执行成功率或验证结果下降 | 复核 | 根据轨迹记录定位失败阶段 |

![固定比较条件通过数据、安全、样本与胜率检查形成评测结论](../public/02_PyTorch_Algorithms/50_preference_evidence_gate.svg)

<a id="mechanism-practice"></a>

## Step 4：实现偏好数据审计、成对统计与候选门禁

题目区使用固定偏好对与评测记录复现数据准入和离线评测。骨架提供记录结构与决策顺序；学习者补全数据审计、指标汇总、判分一致性和评测门禁。测试覆盖空字段、重复与污染、平局与部分完成、顺序翻转，以及通过、复核和拒绝三类结果。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|:---|:---|:---|:---|
| TODO 1：审计偏好数据<br>`audit_preference_dataset` | 审计完整性、来源、偏好依据、切分、重复和训练—评测污染 | 任务输入先规范化，各类风险分别计数 | 空 pair、来源缺失、非法切分、重复身份与规范化后污染 |
| TODO 2：汇总成对结果<br>`compute_pairwise_outcomes` | 汇总完成数、胜负平、覆盖率与明确判决率 | 平局不计为胜负，评测缺失与平局分开记录 | 重复 ID、部分完成、平局和空评测 |
| TODO 3：检查评审一致性<br>`compute_judge_consistency` | 检查同一偏好对的重复判分 | 按样本身份比较，不能只比较总体比例 | 一致、翻转与长度不匹配 |
| TODO 4：判定候选状态<br>`classify_preference_evidence` | 先检查数据与安全，再检查样本数、覆盖率、明确判决数、胜率和一致性 | 返回状态及全部失败原因 | `qualified / review / reject` 与原因列表 |

```python
from dataclasses import dataclass

```


```python
@dataclass
class PreferenceSample:
    sample_id: str
    split: str
    prompt: str
    chosen: str
    rejected: str
    source_id: str
    preference_reason: str


def normalize_task_input(text: str) -> str:
    """统一大小写与空白，供训练—评测重复检查使用。"""
    return " ".join(text.casefold().split())


def audit_preference_dataset(samples: list[PreferenceSample]) -> dict[str, int]:
    """分别统计空字段、无效回答对、重复身份和训练—评测污染。"""
    # ==========================================
    # TODO 1：分别审计空字段、来源、偏好依据、切分、重复身份和跨切分污染。
    # normalized_prompts = ???  # 规范化后再比较训练与评测任务输入。
    # contamination_count = ???  # 两个切分共有的规范化任务输入数量。
    # ==========================================
    raise NotImplementedError("TODO 1：审计偏好数据")


def compute_pairwise_outcomes(
    pair_scores: list[dict[str, float]], expected_pair_count: int | None = None
) -> dict[str, float | int]:
    """分别汇总评测覆盖率、明确判决率与非平局胜率。"""
    if not pair_scores:
        raise ValueError("pair_scores 不能为空")
    # ==========================================
    # TODO 2：验证 pair ID 与预期样本数，再分开统计评测缺失、平局和胜负。
    # evaluated_pairs = ???  # 实际完成评测的样本数。
    # decided_pairs = ???  # 具有明确胜负的样本数。
    # ==========================================
    raise NotImplementedError("TODO 2：汇总成对评测结果")


def compute_judge_consistency(first_pass: list[int], repeated_pass: list[int], remapped_swapped_pass: list[int]) -> dict[str, float]:
    """比较重复判分与交换回答顺序后的重映射结果。"""
    if not first_pass or len({len(first_pass), len(repeated_pass), len(remapped_swapped_pass)}) != 1:
        raise ValueError("三次判分必须非空且长度一致")
    # ==========================================
    if any(value not in {-1, 0, 1} for values in (first_pass, repeated_pass, remapped_swapped_pass) for value in values):
        raise ValueError("判分只能使用 -1、0、1")
    # TODO 3：分别计算重复稳定性与顺序交换后的重映射稳定性。
    # repeated_agreements = ???  # 重复判分与首次判分一致的数量。
    # swapped_agreements = ???  # 交换顺序并重映射后仍一致的数量。
    # judge_consistency = ???  # 两项一致性中较低的一项。
    # ==========================================
    raise NotImplementedError("TODO 3：计算判分一致性")


def classify_preference_evidence(
    outcomes: dict[str, float | int],
    data_pass: bool,
    safety_pass: bool,
    judge_consistency: float,
    min_win_rate: float = 0.6,
    min_evaluation_coverage: float = 0.8,
    min_evaluated_pairs: int = 3,
    min_decided_pairs: int = 2,
    min_consistency: float = 0.8,
) -> dict[str, object]:
    """先检查数据与安全，再返回样本和质量条件的全部失败原因。"""
    # ==========================================
    # TODO 4：数据或安全检查失败时拒绝；否则收集其余未通过条件。
    # rejection_reasons = ???  # 数据检查与安全检查的失败原因。
    # review_reasons = ???  # 样本数、覆盖率、明确判决数、胜率与一致性的失败原因。
    # ==========================================
    raise NotImplementedError("TODO 4：判定候选状态并返回原因")
```

## 测试

运行下方 CPU 测试，分别检查数据隔离、成对统计、判分一致性和候选决策。

```python
def _preference_samples() -> list[PreferenceSample]:
    return [
        PreferenceSample("train-1", "train", "Explain LoRA", "Low-rank adapters.", "All weights change.", "generator-v1", "correctness"),
        PreferenceSample("train-1", "holdout", "Explain LoRA", "Adapters.", "", "", ""),
        PreferenceSample("eval-1", "eval", "  explain   lora  ", "Low-rank adapters.", "All weights change.", "frozen-eval-v1", "correctness"),
    ]


def test_preference_data_audit() -> None:
    """空字段、重复身份和规范化后的跨切分污染必须分别暴露。"""
    assert audit_preference_dataset(_preference_samples()) == {
        "total": 3, "empty_count": 1, "invalid_pair_count": 0,
        "missing_source_count": 1, "missing_reason_count": 1, "invalid_split_count": 1,
        "duplicate_id_count": 1, "contamination_count": 1,
    }


def test_pairwise_outcome_protocol() -> None:
    """评测缺失、平局、明确判决和非平局胜率必须分开记录。"""
    report = compute_pairwise_outcomes([
        {"pair_id": "p1", "candidate_score": 0.9, "baseline_score": 0.1},
        {"pair_id": "p2", "candidate_score": 0.2, "baseline_score": 0.2},
        {"pair_id": "p3", "candidate_score": 0.4, "baseline_score": 0.5},
    ], expected_pair_count=4)
    assert report == {
        "expected_pairs": 4, "evaluated_pairs": 3, "decided_pairs": 2,
        "wins": 1, "losses": 1, "ties": 1,
        "evaluation_coverage": 0.75, "decision_rate": 2 / 3, "decided_win_rate": 0.5,
    }
    try:
        compute_pairwise_outcomes([
            {"pair_id": "p1", "candidate_score": 1.0, "baseline_score": 0.0},
            {"pair_id": "p1", "candidate_score": 0.0, "baseline_score": 1.0},
        ])
    except ValueError:
        pass
    else:
        raise AssertionError("重复 pair ID 必须被拒绝")


def test_judge_consistency() -> None:
    """同一批 pair 的重复判分应逐项比较。"""
    assert compute_judge_consistency([1, 0, -1, 1], [1, 0, 1, 1], [1, 0, -1, -1]) == {
        "total": 4, "repeat_consistency": 0.75, "swap_consistency": 0.75, "judge_consistency": 0.75,
    }


def test_preference_evidence_gate() -> None:
    """数据或安全检查失败时拒绝，其余条件不足时返回全部复核原因。"""
    qualified = {
        "expected_pairs": 4, "evaluated_pairs": 4, "decided_pairs": 3,
        "evaluation_coverage": 1.0, "decided_win_rate": 2 / 3,
    }
    assert classify_preference_evidence(qualified, True, True, 0.9) == {"status": "qualified", "reasons": []}
    assert classify_preference_evidence(qualified, True, False, 0.9) == {"status": "reject", "reasons": ["safety_gate_failed"]}
    incomplete = {
        "expected_pairs": 8, "evaluated_pairs": 2, "decided_pairs": 1,
        "evaluation_coverage": 0.25, "decided_win_rate": 0.5,
    }
    result = classify_preference_evidence(incomplete, True, True, 0.7)
    assert result["status"] == "review"
    assert set(result["reasons"]) == {
        "insufficient_evaluated_pairs", "low_evaluation_coverage",
        "insufficient_decided_pairs", "low_decided_win_rate", "low_judge_consistency",
    }


def test_invalid_preference_evidence_domains() -> None:
    """空身份、非有限分数和不一致计数必须被拒绝。"""
    invalid_scores = [
        [{"pair_id": "", "candidate_score": 1.0, "baseline_score": 0.0}],
        [{"pair_id": "p1", "candidate_score": float("nan"), "baseline_score": 0.0}],
    ]
    for rows in invalid_scores:
        try:
            compute_pairwise_outcomes(rows)
        except ValueError:
            pass
        else:
            raise AssertionError("无效 pair 身份或分数必须被拒绝")
    invalid_outcomes = {"evaluated_pairs": 2, "decided_pairs": 3, "evaluation_coverage": 1.0, "decided_win_rate": 0.5}
    try:
        classify_preference_evidence(invalid_outcomes, True, True, 0.9)
    except ValueError:
        pass
    else:
        raise AssertionError("不一致评测计数必须被拒绝")


test_preference_data_audit()
test_pairwise_outcome_protocol()
test_judge_consistency()
test_preference_evidence_gate()
test_invalid_preference_evidence_domains()
print("✅ 偏好数据隔离、成对统计与候选决策测试通过。")
```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
from dataclasses import dataclass

@dataclass
class PreferenceSample:
    sample_id: str
    split: str
    prompt: str
    chosen: str
    rejected: str
    source_id: str
    preference_reason: str


def normalize_task_input(text: str) -> str:
    """统一大小写与空白，供训练—评测重复检查使用。"""
    return " ".join(text.casefold().split())


def audit_preference_dataset(samples: list[PreferenceSample]) -> dict[str, int]:
    """分别统计空字段、无效回答对、重复身份和训练—评测污染。"""
    # TODO 1：审计偏好数据。
    allowed_splits = {"train", "validation", "eval", "test"}
    ids = [sample.sample_id for sample in samples]
    train_prompts = {normalize_task_input(sample.prompt) for sample in samples if sample.split == "train"}
    eval_prompts = {normalize_task_input(sample.prompt) for sample in samples if sample.split in {"validation", "eval", "test"}}
    return {
        "total": len(samples),
        "empty_count": sum(not sample.prompt.strip() or not sample.chosen.strip() or not sample.rejected.strip() for sample in samples),
        "invalid_pair_count": sum(
            bool(sample.chosen.strip())
            and normalize_task_input(sample.chosen) == normalize_task_input(sample.rejected)
            for sample in samples
        ),
        "missing_source_count": sum(not sample.source_id for sample in samples),
        "missing_reason_count": sum(not sample.preference_reason for sample in samples),
        "invalid_split_count": sum(sample.split not in allowed_splits for sample in samples),
        "duplicate_id_count": len(ids) - len(set(ids)),
        "contamination_count": len(train_prompts & eval_prompts),
    }


def compute_pairwise_outcomes(
    pair_scores: list[dict[str, float]], expected_pair_count: int | None = None
) -> dict[str, float | int]:
    """分别汇总评测覆盖率、明确判决率与非平局胜率。"""
    if not pair_scores:
        raise ValueError("pair_scores 不能为空")
    # TODO 2：汇总成对结果。
    import math

    required_fields = {"pair_id", "candidate_score", "baseline_score"}
    if any(not required_fields.issubset(row) for row in pair_scores):
        raise ValueError("每条评测必须包含 pair_id 与双方分数")
    pair_ids = [str(row['pair_id']).strip() for row in pair_scores]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_id 必须非空且唯一")
    if any(
        not math.isfinite(float(row[name]))
        for row in pair_scores for name in ("candidate_score", "baseline_score")
    ):
        raise ValueError("评测分数必须为有限数")
    evaluated_pairs = len(pair_scores)
    expected_pairs = evaluated_pairs if expected_pair_count is None else expected_pair_count
    if expected_pairs < evaluated_pairs or expected_pairs <= 0:
        raise ValueError("expected_pair_count 不能小于实际评测数")
    wins = sum(row['candidate_score'] > row['baseline_score'] for row in pair_scores)
    losses = sum(row['candidate_score'] < row['baseline_score'] for row in pair_scores)
    ties = evaluated_pairs - wins - losses
    decided_pairs = wins + losses
    return {
        "expected_pairs": expected_pairs,
        "evaluated_pairs": evaluated_pairs,
        "decided_pairs": decided_pairs,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "evaluation_coverage": evaluated_pairs / expected_pairs,
        "decision_rate": decided_pairs / evaluated_pairs,
        "decided_win_rate": wins / decided_pairs if decided_pairs else 0.0,
    }


def compute_judge_consistency(first_pass: list[int], repeated_pass: list[int], remapped_swapped_pass: list[int]) -> dict[str, float]:
    """比较重复判分与交换回答顺序后的重映射结果。"""
    if not first_pass or len({len(first_pass), len(repeated_pass), len(remapped_swapped_pass)}) != 1:
        raise ValueError("三次判分必须非空且长度一致")
    if any(value not in {-1, 0, 1} for values in (first_pass, repeated_pass, remapped_swapped_pass) for value in values):
        raise ValueError("判分只能使用 -1、0、1")
    # TODO 3：检查判分一致性。
    repeated_agreements = sum(first == repeated for first, repeated in zip(first_pass, repeated_pass))
    swapped_agreements = sum(first == swapped for first, swapped in zip(first_pass, remapped_swapped_pass))
    repeat_consistency = repeated_agreements / len(first_pass)
    swap_consistency = swapped_agreements / len(first_pass)
    return {"total": len(first_pass), "repeat_consistency": repeat_consistency,
            "swap_consistency": swap_consistency, "judge_consistency": min(repeat_consistency, swap_consistency)}


def classify_preference_evidence(
    outcomes: dict[str, float | int],
    data_pass: bool,
    safety_pass: bool,
    judge_consistency: float,
    min_win_rate: float = 0.6,
    min_evaluation_coverage: float = 0.8,
    min_evaluated_pairs: int = 3,
    min_decided_pairs: int = 2,
    min_consistency: float = 0.8,
) -> dict[str, object]:
    """先检查数据与安全，再返回样本和质量条件的全部失败原因。"""
    # TODO 4：判定候选状态。
    import math

    required_outcomes = {"evaluated_pairs", "decided_pairs", "evaluation_coverage", "decided_win_rate"}
    if not required_outcomes.issubset(outcomes):
        raise ValueError("outcomes 缺少候选门禁字段")
    evaluated_pairs = int(outcomes["evaluated_pairs"])
    decided_pairs = int(outcomes["decided_pairs"])
    coverage = float(outcomes["evaluation_coverage"])
    win_rate = float(outcomes["decided_win_rate"])
    rates = [coverage, win_rate, float(judge_consistency), float(min_win_rate),
             float(min_evaluation_coverage), float(min_consistency)]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in rates):
        raise ValueError("覆盖率、胜率、一致性及其门槛必须位于 [0, 1]")
    if evaluated_pairs < 0 or decided_pairs < 0 or decided_pairs > evaluated_pairs:
        raise ValueError("评测计数必须非负，且明确判决数不能超过已评测数")
    if min_evaluated_pairs < 0 or min_decided_pairs < 0:
        raise ValueError("样本数门槛必须非负")
    rejection_reasons = []
    if not data_pass:
        rejection_reasons.append("data_gate_failed")
    if not safety_pass:
        rejection_reasons.append("safety_gate_failed")
    if rejection_reasons:
        return {"status": "reject", "reasons": rejection_reasons}
    review_reasons = []
    if outcomes["evaluated_pairs"] < min_evaluated_pairs:
        review_reasons.append("insufficient_evaluated_pairs")
    if outcomes["evaluation_coverage"] < min_evaluation_coverage:
        review_reasons.append("low_evaluation_coverage")
    if outcomes["decided_pairs"] < min_decided_pairs:
        review_reasons.append("insufficient_decided_pairs")
    if outcomes["decided_win_rate"] < min_win_rate:
        review_reasons.append("low_decided_win_rate")
    if judge_consistency < min_consistency:
        review_reasons.append("low_judge_consistency")
    status = "review" if review_reasons else "qualified"
    return {"status": status, "reasons": review_reasons}
```

### 解析

#### TODO 1：审计偏好数据

- 机制：分别统计空字段、相同回答、来源与偏好依据缺失、非法切分、重复身份和规范化任务输入的跨切分重复。
- 不变量：各类风险独立记录；大小写或连续空白变化不能绕过训练—评测污染检查。

#### TODO 2：汇总成对结果

- 机制：预期样本数与实际完成数形成评测覆盖率，胜负数与完成数形成明确判决率，非平局胜率只使用胜负样本。
- 不变量：平局属于已完成评测，但不进入胜率分母；缺失评测既不算平局，也不进入胜负统计。

#### TODO 3：检查判分一致性

- 机制：逐个偏好对比较重复判分，并把交换回答顺序后的结果重映射到原始候选身份。
- 不变量：三次判分长度一致且只使用 `-1 / 0 / 1`；重复稳定性与换序稳定性分别报告。

#### TODO 4：判定候选状态

- 机制：数据检查或安全检查失败时拒绝候选；两项检查通过后，再判断样本数、覆盖率、明确判决数、胜率和一致性。
- 不变量：数据检查失败或任一安全指标低于阈值时返回 `reject`；其余条件不足时返回 `review`；全部通过时才返回 `qualified`。
<a id="framework-check"></a>

## 框架一致性核对（可选）

下面把同一批偏好记录交给 Hugging Face `Dataset`，核对字段结构、样本身份、切分筛选和规范化任务输入的跨切分重复。候选胜率与安全门禁仍由 Step 4 负责，Dataset 只复核数据接口和切分结果。

| 核对对象 | Step 4 教学实现 | 成熟框架 CPU 路径 | 共同输入与通过条件 |
|:---|:---|:---|:---|
| 字段与身份 | `PreferenceSample` 与 `audit_preference_dataset` | `Dataset.from_list` / `Dataset.features` | 使用同一批记录；字段完整且行顺序、`sample_id` 保持不变 |
| 切分筛选 | 按 `split` 区分训练与评测记录 | `Dataset.filter` | 训练与评测样本 ID 分别一致 |
| 污染计数 | `normalize_task_input` 后比较任务输入集合 | `Dataset.map` 后比较两个切分 | 使用相同规范化函数；跨切分重复数量一致 |

```python
from dataclasses import asdict
from importlib.metadata import version

try:
    from datasets import Dataset, Features, Value
except ImportError:
    framework_check = {
        "status": "skipped",
        "reason": "请使用 PT-CPU 环境运行本单元",
    }
else:
    def check_preference_dataset_interface():
        """核对偏好记录在 Dataset 中的字段、身份、切分与污染计数。"""
        samples = [
            PreferenceSample("train-1", "train", " Explain PPO " , "Use clipping.", "Ignore ratios.", "src-a", "correctness"),
            PreferenceSample("train-2", "train", "Explain DPO", "Compare a pair.", "Use one answer.", "src-b", "correctness"),
            PreferenceSample("eval-1", "eval", "explain   ppo", "Check held-out behavior.", "Reuse training text.", "src-c", "isolation"),
        ]
        local_audit = audit_preference_dataset(samples)
        fields = {name: Value("string") for name in asdict(samples[0])}
        dataset = Dataset.from_list([asdict(sample) for sample in samples], features=Features(fields))
        normalized = dataset.map(lambda row: {"normalized_prompt": normalize_task_input(row["prompt"])})
        train_rows = normalized.filter(lambda row: row["split"] == "train")
        eval_rows = normalized.filter(lambda row: row["split"] in {"validation", "eval", "test"})
        train_prompts = set(train_rows["normalized_prompt"])
        eval_prompts = set(eval_rows["normalized_prompt"])
        checks = {
            "schema_preserved": set(dataset.features) == set(fields),
            "identity_preserved": dataset["sample_id"] == [sample.sample_id for sample in samples],
            "split_filter_matches": train_rows["sample_id"] == ["train-1", "train-2"] and eval_rows["sample_id"] == ["eval-1"],
            "contamination_count_matches": len(train_prompts & eval_prompts) == local_audit["contamination_count"] == 1,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise AssertionError(f"偏好数据框架核对失败：{failed_checks}")
        return {
            "status": "passed",
            "check_kind": "data_interface",
            "versions": {"datasets": version("datasets")},
            "checks": checks,
            "row_count": len(dataset),
            "contamination_count": local_audit["contamination_count"],
        }

    framework_check = check_preference_dataset_interface()

framework_check
```

## 相关阅读

- [Part 02 · 17 DPO 损失](./17_DPO_Loss_Tutorial.md)
- [Part 02 · 18 GRPO 损失](./18_GRPO_Loss_Tutorial.md)
- [Part 02 · 51 Online DPO](./51_Online_DPO.md)
- [Part 02 · 84 DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [Part 02 · 85 GRPO 组级对齐项目](./85_GRPO_Groupwise_Alignment_Project.md)
- [TRL: DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL: GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
