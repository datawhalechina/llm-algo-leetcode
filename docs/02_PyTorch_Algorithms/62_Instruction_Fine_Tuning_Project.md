# 62. Instruction Fine Tuning Project | 指令微调项目
**难度：** Hard | **环境：** CPU-first | **标签：** `训练微调`, `指令微调`, `数据工程` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节承接第 13 节的端到端 SFT 小项目，把“训练结果是否可信”推进为“指令微调是否可以交付”。第 64 节先回答数据是否具备训练条件，本节进一步固定 instruction、input、response 的任务格式、评测集与资源预算，再结合 train / val 指标和生成样例检查结果是否满足要求。它与第 60 节的分工是：本节判断指令微调任务能否交付，第 60 节在任务口径稳定后比较全参数与 LoRA 的更新成本。本节的 Hard 主要来自数据、格式、任务质量、资源和产物等多类证据的整合与决策。

**关键词：** `instruction tuning`, `data audit`, `evaluation`, `delivery`

---
## 前置阅读

**导语：** 进入本项目前，先能运行 SFT 训练闭环、LoRA 适配和学习率调度，再把注意力转向数据模板、格式稳定性和交付判断。

- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [11. Optimizer Updates and Learning Rate Scheduling | 优化器更新与学习率调度](./11_Optimizer_Updates_and_Learning_Rate_Scheduling.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
---

### Step 1：先把交付问题和对照组定下来

本步只回答一个问题：这批指令数据是否已经具备进入 SFT 训练和评测的条件。最小任务是把 `instruction / input` 组成 prompt，把 `output` 作为 response，再检查模型回答和项目报告。

| 项目内容 | 本节设定 | 需要确认什么 |
|:---|:---|:---|
| 任务与输入 | 使用 `instruction / input → output`；CPU 默认使用 Notebook 内置的最小样例 | 字段映射和 prompt 模板正确 |
| 数据与评测 | 固定训练 / 验证划分，并准备 `eval_cases` 评测样例；GPU 实验时再替换为明确的数据集和模型 | 训练和评测使用可复查的输入 |
| CPU 检查 | 审计字段、模板、长度、评测输入和决策逻辑 | 数据能否进入训练 |
| GPU SFT（可选） | 在配置单元中填写模型、数据、dtype、batch、seq_len、steps 和 seed | 比较基础模型与指令微调模型的 loss、生成质量和资源 |
| 输出 | 保存数据审计、模型回答、loss、资源记录和项目报告 | 判断结果是否具备交付条件 |


![指令微调的交付判断链](../public/02_PyTorch_Algorithms/62_instruction_finetuning_flow.svg)
<div align="center"><strong>指令微调项目实验流程：</strong>先定义任务并固定条件，再建立模型对照，最后汇总数据、质量、资源和产物证据。</div>

### Step 2（CPU）：完成数据与格式准入实验

输入是 instruction、input、response 记录和最小评测集合。先把训练前风险转成可复查的指标，再决定数据是否可以进入 GPU 训练。实验动作：固定字段和阈值 → 审计原始数据 → 检查模板格式 → 检查评测集合 → 记录 blocker 与普通问题 → 生成 CPU 数据报告。

| 检查对象 | 记录指标 | 实验目的 | 失败影响 |
|:---|:---|:---|:---|
| 数据字段 | 样本数、空字段、重复数 | 判断是否存在可用监督 | 可能形成 blocker |
| prompt 长度 | 超长样本数、长度分位数 | 判断是否可能被截断 | 可能丢失任务信息 |
| 模板格式 | 缺字段、拼接异常、role 错误 | 判断训练输入是否稳定 | 形成格式 blocker |
| 评测集合 | 样例数、任务类型、输出格式 | 判断训练后是否可验证 | 形成评测缺口 |


### Step 3（CPU）：实现并运行项目报告
在题目区实现数据摘要、格式检查、样例抽检和项目决策；四个函数依次形成 CPU 报告检查链。完成后先运行题目区测试，再对照参考答案与解析检查字段口径和决策分支。

![62 Step 3：CPU 题目区的最小实现](../public/02_PyTorch_Algorithms/62_instruction_cpu_todo_flow.svg)
<div align="center"><strong>CPU 报告检查链：</strong>数据摘要 → 格式检查 → 样例抽检 → 项目决策。</div>

需要保存 CPU 报告时，将 `RUN_PROJECT_EXPORT` 改为 `True`。需要进行真实模型验证时，再进入 Step 4。

### Step 4（GPU，可选）：运行真实模型对照
GPU 实验验证微调后的模型是否满足任务和资源要求；数据审计结果作为实验准入依据。基础模型和指令微调模型使用相同的评测样例、模板和统计口径。

| 实验对象 | 固定条件 | 主要指标 | 主要用途 |
|:---|:---|:---|:---|
| 基础模型 | 同一模型、评测样例和模板 | 格式稳定性、任务完成度 | 建立未微调基线 |
| 指令微调模型 | 同一数据 split、dtype、batch、seq_len、steps 和 seed | train / val loss、生成质量 | 判断微调是否有效 |
| 资源记录 | 与对照实验一致的 workload | step time、peak memory | 判断训练代价是否可接受 |
| 项目验收 | 固定数据、格式和任务规则 | 数据、格式、任务指标 | 交给 Step 5 输出决策 |

实验流程：固定数据和模板 → 评测基础模型 → 完成指令微调 → 在同一评测集生成结果 → 对比 loss、格式和任务完成度 → 记录资源 → 进入 Step 5。

SFT 通过后要保存可复用的 checkpoint 或 adapter，而不是只保留项目报告。进入 DPO/GRPO 时，使用本节确定的 SFT checkpoint、tokenizer/chat template 和数据版本初始化 policy/reference，并把本节评测结果作为后续对齐的 baseline。

### Step 5：把证据合成交付决策
将 Step 2 的数据准入结果和 Step 4 的模型对照结果放在同一张报告中：先检查数据准入，再检查验证损失、格式/任务指标和资源是否达到预设门槛。项目决策综合数据准入、任务指标、资源记录和可复现产物；train loss 作为其中一项训练信号。

| 决策 | 必须满足 | 下一步 |
|:---|:---|:---|
| `accept` | 数据无 blocker，任务指标达标，资源可接受 | 保存配置、评测集和模型产物 |
| `tune` | 数据可用，但指标或资源仍需调整 | 回到数据模板、评测样例或训练配置 |
| `reject` | 数据、格式或任务质量无法满足当前目标 | 更换数据或任务定义后重新审计 |

#### 09-13 如何收束到 62 指令微调项目

`62` 把 SFT 数据工程和训练闭环组合成一个项目交付模板。

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| 数据 | 样本数、空 response、重复样本、超长样本 | 判断数据是否值得训 |
| 格式 | 缺字段、空字段、模板拼接问题 | 判断输入是否稳定 |
| 训练 | train / val 指标、step time | 判断训练是否可信 |
| 样例 | 训练后最小回答抽检 | 判断输出是否可交付 |
| 决策 | accept / tune / reject | 输出项目结论 |

### 参数口径说明

本节主要是数据与交付模板。`max_prompt_chars` 是 prompt 长度审计阈值，不是模型的 token 上限；`instruction / input / response` 是数据字段，必须固定字段映射；`eval_cases` 是训练后样例评测集合。真实训练时还要固定 model、dtype、batch、seq_len、steps 和验证集，不能只凭格式检查宣布项目完成。

```python
from typing import Dict, List

```


```python
# 4 个核心 TODO：数据审计、格式检查、样例抽检、项目总结
# 目标：把 instruction / input / response 数据整理成统一项目报告，而不是只看训练指标。
# CPU 题目区验证数据与格式口径；GPU 扩展才验证真实模型的 loss、生成质量和资源。

# TODO 1：统计指令数据集摘要
def summarize_instruction_dataset(records: List[Dict[str, str]], max_prompt_chars: int) -> Dict[str, float]:
    """统计样本数、空 response、重复样本和 prompt 长度风险。

    每条记录使用 instruction、input、response 字段；返回摘要字典，
    不计算模型 loss，也不把字符数当成 tokenizer token 数。
    """
    # 提示：重复键使用三字段组合；空 response 只按 response 判定。
    # total_samples = ???；empty_response_count = ???；duplicate_count = ???；over_length_count = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

# TODO 2：检查格式是否合法
def check_instruction_format(batch: List[Dict[str, str]]) -> Dict[str, int]:
    """检查 instruction / response 必填字段和非空约束。

    返回 valid_count、missing_field_count 和 format_issue_count；
    缺字段与字段存在但为空必须分开统计。
    """
    # 提示：只有 instruction、response 都存在且非空时才计入 valid_count。
    # missing_field_count = ???；format_issue_count = ???；valid_count = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

# TODO 3：汇总训练后样例抽检结果
def review_instruction_outputs(outputs: List[Dict[str, object]]) -> Dict[str, object]:
    """汇总固定样例的格式通过率和任务通过率。

    输入记录至少包含 format_ok、task_ok；空列表时通过率应为 0.0。
    `format_ok` 检查输出格式，`task_ok` 检查最小任务目标；二者都是样例级诊断，
    不等同于完整评测集的模型能力。GPU 生成结果可以沿用这些字段进入项目报告。
    """
    # 提示：分别统计 format_pass_count / task_pass_count，并保留样例总数。
    #       不要把 format_ok 当成 task_ok，也不要用一个总通过率掩盖两类失败。
    # sample_count = ???；format_pass_count = ???；task_pass_count = ???；pass_rate = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

# TODO 4：输出项目交付结论
def build_instruction_project_report(summary: Dict[str, float], format_check: Dict[str, int], output_review: Dict[str, object]) -> Dict[str, object]:
    """把数据摘要、格式检查和样例抽检收成项目决策。

    存在字段/格式 blocker 时应 reject；输入合规但样例任务不稳定时可 tune；
    仅凭 CPU 审计不能输出真实训练效果的 accept。
    """
    # 提示：返回 decision、project_ready 和 next_action 三个核心字段。
    #       先判断数据/格式 blocker，再判断样例格式和任务通过率；CPU 审计不能伪造 GPU accept。
    # blockers = ???；project_ready = ???；decision = ???；next_action = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_instruction_project_template():
    records = [
        {'instruction': '解释 LoRA。', 'input': '', 'response': 'LoRA 是低秩适配。'},
        {'instruction': '解释 LoRA。', 'input': '', 'response': 'LoRA 是低秩适配。'},
        {'instruction': '给出答案。', 'input': '', 'response': ''},
        {'instruction': '   ', 'input': '', 'response': '有回答但没有指令'},
    ]
    summary = summarize_instruction_dataset(records, max_prompt_chars=20)
    assert summary['total_samples'] == 4
    assert summary['empty_response_count'] == 1
    assert summary['duplicate_count'] == 1

    format_check = check_instruction_format(records)
    assert format_check['valid_count'] == 2
    assert format_check['format_issue_count'] == 2

    output_review = review_instruction_outputs([
        {'format_ok': True, 'task_ok': True},
        {'format_ok': True, 'task_ok': False},
    ])
    assert output_review['format_pass_count'] == 2
    assert output_review['task_pass_count'] == 1
    assert output_review['format_pass_rate'] == 1.0
    assert output_review['task_pass_rate'] == 0.5

    report = build_instruction_project_report(summary, format_check, output_review)
    assert report['decision'] == 'reject'
    assert report['project_ready'] is False
    assert report['next_action'] == 'fix_data_or_format'

    clean_records = [
        {'instruction': '总结 LoRA。', 'input': '一句话', 'response': 'LoRA 是一种参数高效微调方法。'},
        {'instruction': '解释 QLoRA。', 'input': '', 'response': 'QLoRA 在量化底座上进行低秩适配。'},
    ]
    clean_summary = summarize_instruction_dataset(clean_records, max_prompt_chars=40)
    clean_format = check_instruction_format(clean_records)
    clean_review = review_instruction_outputs([
        {'format_ok': True, 'task_ok': True},
        {'format_ok': True, 'task_ok': True},
    ])
    accept_report = build_instruction_project_report(clean_summary, clean_format, clean_review)
    assert accept_report['decision'] == 'accept'
    assert accept_report['project_ready'] is True
    assert accept_report['next_action'] == 'promote_to_delivery'

    tune_review = review_instruction_outputs([
        {'format_ok': True, 'task_ok': True},
        {'format_ok': True, 'task_ok': False},
    ])
    tune_report = build_instruction_project_report(clean_summary, clean_format, tune_review)
    assert tune_report['decision'] == 'tune'
    assert tune_report['project_ready'] is False
    assert tune_report['next_action'] == 'refine_eval_or_training'


test_instruction_project_template()
print('测试通过：指令微调项目模板可以工作。')

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
def summarize_instruction_dataset(records: List[Dict[str, str]], max_prompt_chars: int) -> Dict[str, float]:
    seen = set()
    empty_response_count = 0
    duplicate_count = 0
    over_length_count = 0
    total_prompt_chars = 0
    for record in records:
        instruction = str(record.get('instruction', ''))
        input_text = str(record.get('input', ''))
        response = str(record.get('response', ''))
        prompt = instruction + input_text
        total_prompt_chars += len(prompt)
        key = (instruction, input_text, response)
        if not response.strip():
            empty_response_count += 1
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
        if len(prompt) > max_prompt_chars:
            over_length_count += 1
    total_samples = len(records)
    return {
        'total_samples': total_samples,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_length_count': over_length_count,
        'avg_prompt_chars': total_prompt_chars / total_samples if total_samples else 0.0,
    }


def check_instruction_format(batch: List[Dict[str, str]]) -> Dict[str, int]:
    valid_count = 0
    missing_field_count = 0
    format_issue_count = 0
    for record in batch:
        if 'instruction' not in record or 'response' not in record:
            missing_field_count += 1
            continue
        if not str(record.get('instruction', '')).strip() or not str(record.get('response', '')).strip():
            format_issue_count += 1
            continue
        valid_count += 1
    return {'valid_count': valid_count, 'missing_field_count': missing_field_count, 'format_issue_count': format_issue_count}


def review_instruction_outputs(outputs: List[Dict[str, object]]) -> Dict[str, object]:
    format_pass_count = sum(1 for item in outputs if item.get('format_ok', False))
    task_pass_count = sum(1 for item in outputs if item.get('task_ok', False))
    sample_count = len(outputs)
    return {
        'sample_count': sample_count,
        'format_pass_count': format_pass_count,
        'task_pass_count': task_pass_count,
        'format_pass_rate': round(format_pass_count / sample_count, 4) if sample_count else 0.0,
        'task_pass_rate': round(task_pass_count / sample_count, 4) if sample_count else 0.0,
        'sample_ready': bool(outputs) and format_pass_count == sample_count,
    }


def build_instruction_project_report(summary: Dict[str, float], format_check: Dict[str, int], output_review: Dict[str, object]) -> Dict[str, object]:
    blockers = []
    soft_issues = []
    if summary['empty_response_count'] > 0:
        blockers.append('存在空 response 样本')
    if summary['over_length_count'] > 0:
        blockers.append('存在超长 prompt 样本')
    if format_check['missing_field_count'] > 0:
        blockers.append('存在字段缺失样本')
    if format_check['format_issue_count'] > 0:
        blockers.append('存在格式不稳定样本')
    if output_review['task_pass_count'] < output_review['format_pass_count']:
        soft_issues.append('训练后样例任务完成度不足')

    if not blockers and output_review['sample_ready'] and not soft_issues:
        decision = 'accept'
        next_action = 'promote_to_delivery'
    elif not blockers and (output_review['sample_ready'] or output_review['format_pass_count'] > 0):
        decision = 'tune'
        next_action = 'refine_eval_or_training'
    else:
        decision = 'reject'
        next_action = 'fix_data_or_format'

    return {
        'decision': decision,
        'blockers': blockers + soft_issues,
        'next_action': next_action,
        'project_ready': decision == 'accept',
    }

```

### 解析

这一页保留 `4` 个核心 TODO：数据审计、格式检查、样例抽检和项目总结。它不要求把训练循环重写一遍，而是要求把“这一轮指令微调能不能交付”补成完整判断链。

**1. TODO 1: 统计指令数据集摘要**
- **实现方式**：遍历 `instruction / input / response` 记录，统计总样本数、空 response、重复样本、超长 prompt 和平均 prompt 长度。
- **关键点**：`prompt` 长度按 `instruction + input` 口径处理；空 response、重复样本和超长样本都应该在训练前被发现。
- **项目意义**：这一步先回答“数据值不值得训”，而不是先跑训练再看结果。

**2. TODO 2: 检查格式是否合法**
- **实现方式**：区分 `valid_count`、`missing_field_count` 和 `format_issue_count`，把缺字段和空 instruction / response 分开统计。
- **关键点**：格式检查不是在找模型效果问题，而是在找模板和样本结构问题；这类问题属于训练前 blocker。
- **项目意义**：如果模板拼接不稳，后面的 train / val 指标再漂亮也没有交付意义。

**3. TODO 3: 汇总训练后样例抽检结果**
- **实现方式**：统计 `format_pass_count`、`task_pass_count`，并用 `sample_ready` 表示样例是否足够进入交付判断。
- **关键点**：`sample_ready` 只表示样例格式层面可继续看，不等于项目已经可以 `accept`。
- **项目意义**：这一步把训练结果从纯指标表推进到可读样例验证，避免“loss 变好但输出不可用”。

**4. TODO 4: 输出项目交付结论**
- **实现方式**：把数据摘要、格式检查和样例抽检统一收成 `accept / tune / reject`，同时给出 `next_action`。
- **关键点**：数据或格式硬问题走 `reject`；数据和格式过关但样例任务完成度不稳时走 `tune`；只有样例格式和任务完成度都稳定时才 `accept`。
- **项目意义**：这一步让页面真正回答“这一轮指令微调能不能交付”，而不是只回答“训练有没有跑通”。

### 可选：统一项目报告导出
默认不导出。完成数据审计、格式检查和样例评测后，再开启导出，避免把模板演示结果当成真实项目结论。报告模板见 `docs/verification/fine_tuning_projects.md`。

```python
try:
    from tools.fine_tuning_project_runtime import preflight_runtime, runtime_snapshot, save_project_report, validate_project_config
except ModuleNotFoundError:
    preflight_runtime = lambda torch_module, run_mode='cpu', **kwargs: {'run_mode': run_mode, 'ready': False, 'reasons': ['共享运行时工具不可用']}
    runtime_snapshot = lambda: {'device': 'unknown'}
    validate_project_config = lambda config: []
    save_project_report = None
RUN_MODE = 'cpu'  # cpu / dry_run / real_gpu；默认不启动真实训练。
PROJECT_ID = '62_instruction_fine_tuning'
PROJECT_RESULT_PATH = 'benchmarks/results/62_instruction_fine_tuning.json'
PROJECT_CONFIG = {'project': PROJECT_ID, 'model': 'template', 'dtype': 'fp32', 'batch_size': 1, 'seq_len': 128, 'steps': 1, 'seed': 42, 'run_mode': RUN_MODE}
RUN_PROJECT_EXPORT = False  # True 只保存已完成的项目报告。
config_errors = validate_project_config(PROJECT_CONFIG)
if config_errors:
    raise ValueError('; '.join(config_errors))
print('runtime:', runtime_snapshot())
if RUN_MODE == 'dry_run':
    try:
        import torch
        print('dry_run:', preflight_runtime(torch, run_mode='dry_run'))
    except ImportError as exc:
        print({'run_mode': 'dry_run', 'ready': False, 'reasons': [f'缺少 torch：{exc}']})
if RUN_PROJECT_EXPORT:
    if 'PROJECT_REPORT' not in globals():
        raise RuntimeError('请先组装完整的 PROJECT_REPORT')
    PROJECT_REPORT.setdefault('project', PROJECT_ID)
    PROJECT_REPORT.setdefault('config', PROJECT_CONFIG)
    PROJECT_REPORT.setdefault('environment', runtime_snapshot())
    save_project_report(PROJECT_RESULT_PATH, PROJECT_REPORT)

```

## 相关阅读

以下资料按“指令微调论文 → 开源训练实现 → 后续项目”排列，用于把数据模板、训练接口和交付指标连接到真实训练流程。

- [FLAN 论文：Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)

- [Hugging Face TRL 官方仓库](https://github.com/huggingface/trl)
- [Hugging Face Transformers 官方仓库](https://github.com/huggingface/transformers)
- [Transformers Trainer 官方文档](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)
- [63. LoRA Variants Benchmark | LoRA 变体对比项目](./63_LoRA_Variants_Benchmark.md)
- [84. DPO Preference Project | DPO 偏好优化项目](./84_DPO_Preference_Project.md)
