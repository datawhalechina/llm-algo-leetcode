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

本节承接第 13 节的端到端 SFT 小项目，把“训练结果是否可信”推进为“指令微调是否可以交付”。你需要先审计 instruction、input、response 的字段和格式，再固定训练步数、评测集与资源预算，最后结合 train / val 指标和生成样例检查结果是否满足任务要求。最终给出可以交付、需要调整还是应该停止的判断。

**关键词：** `instruction tuning`, `data audit`, `evaluation`, `delivery`

---
## 前置阅读

**导语：** 先把 SFT 训练闭环、LoRA 适配和训练调度理顺，再进入这个项目；本节默认你已经知道训练怎么跑，重点转向数据模板、格式稳定性和交付判断。

- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [11. LR Schedulers WSD Cosine | WSD 余弦学习率调度器](./11_LR_Schedulers_WSD_Cosine.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)

## 相关阅读

**导语：** 做完指令微调项目后，最自然的下一步是继续比较参数高效微调方案，或把这一轮训练结果推进到偏好优化与对齐项目。

- [63. LoRA Variants Benchmark | LoRA 变体对比项目](./63_LoRA_Variants_Benchmark.md)
- [84. DPO Preference Project | DPO 偏好优化项目](./84_DPO_Preference_Project.md)

### Step 1: 定义指令微调目标

- 固定底座模型、训练数据、prompt 模板、batch size、seq len 和训练步数。
- 明确 evaluation set 的构成，保证训练集和验证集的分工清晰。
- 记录 instruction、input、response 的字段约定，避免样本格式漂移。
- baseline 至少要回答两个问题：不用微调时格式是否稳定，已有微调方案是否已经能满足任务。

| 实验组 | 环境 | 固定内容 | 主要验证 |
|:---|:---|:---|:---|
| CPU 机制 | CPU 或 GPU | 人工样例、字段和评测输入 | 数据、模板和决策逻辑 |
| GPU SFT | 单卡 GPU | 模型、数据 split、dtype、batch、steps | train/val loss、资源和生成结果 |

CPU 结果不能证明模型效果；GPU 结果也必须同时保留数据审计和评测样例。

### Step 2（CPU 项目设计）: 先做数据与格式合法性检查

指令微调必须先确认数据和格式口径可信，训练结论才有交付意义。
- 训练前先做数据审计：样本数、空 response、重复样本、超长 prompt。
- 再做格式抽检：是否存在缺字段、空 instruction、空 response 或模板拼接异常。
- 如果数据和格式检查不通过，这一轮实验最多只能产出 blocker，而不是有效训练结论。

### Step 3（GPU 项目设计，可选）: 用统一口径收训练与评测结果

训练和评测结果必须放在统一口径下比较，不能把指标改善和格式稳定性割裂开看。
- 统一记录 train / val 指标、step time、资源消耗和最小样例抽检结果。
- 输出至少一个“训练后回答样例”，验证格式、语气和任务完成度。
- 如果训练指标变好，但格式抽检仍然失败，这一轮仍然不能直接 adopt。

### Step 4: 输出项目交付结论

- 最终结论不只回答“能不能训”，而要回答“能不能交付”。
- 项目结论建议统一成 `accept / tune / reject` 三档。
- 若进入 `tune`，下一轮应优先回到数据模板、评测样本或训练配置，而不是盲目继续加步数。

### Step 5（CPU 代码练习）: 实现数据、格式和输出评测
下面的题目区只实现可在 CPU 检查的函数；真实 SFT 训练和 GPU 资源采集不放进题目区。
#### 图解：09-13 如何收束到 62 指令微调项目

`62` 把 SFT 数据工程和训练闭环组合成一个项目交付模板。

```text
09 SFT data       instruction / input / response / labels
      │
10 LoRA           optional adapter tuning for instruction task
      │
11 Scheduler      lr schedule counted by optimizer update
      │
13 E2E report     train loss / val loss / instruction quality
      │
      ▼
62 Instruction    data audit + format check + sample review + delivery decision
```

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
    通过率是样例级诊断，不等同于完整评测集的模型能力。
    """
    # 提示：分别统计 format_pass_count / task_pass_count，并保留样例总数。
    # sample_count = ???；format_pass_count = ???；task_pass_count = ???；pass_rate = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

# TODO 4：输出项目交付结论
def build_instruction_project_report(summary: Dict[str, float], format_check: Dict[str, int], output_review: Dict[str, object]) -> Dict[str, object]:
    """把数据摘要、格式检查和样例抽检收成项目决策。

    存在字段/格式 blocker 时应 reject；输入合规但样例任务不稳定时可 tune；
    仅凭 CPU 审计不能输出真实训练效果的 accept。
    """
    # 提示：返回 decision、project_ready 和 next_action 三个核心字段。
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
