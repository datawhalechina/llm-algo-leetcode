# 64. SFT Data Quality Project | SFT 数据质量项目

**难度：** Hard | **环境：** CPU-first | **标签：** `训练微调`, `数据质量`, `评测` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节先不训练模型，而是判断一批 SFT 数据是否具备进入正式训练的条件。你需要检查空回答、字段缺失、模板一致性、重复样本和评测覆盖，并把发现的问题按风险和修复成本整理出来。最终给出数据可直接训练、需要回修，还是应当暂缓使用的结论。

**关键词：** `SFT`, `data quality`, `template`, `evaluation`, `project`

---
## 前置阅读

**导语：** 先把 SFT 样本构造、端到端训练闭环和数据工程里的关键风险理顺，再进入这个项目；本节默认你已经知道数据如何进 loss，重点转向这批数据是否值得进入正式训练。
- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [30. Long Context Fine-Tuning | 长上下文微调](./30_Long_Context_Fine_Tuning.md)
- [32. Data Engineering for SFT | SFT 数据工程](./32_Data_Engineering_for_SFT.md)

## 相关阅读

**导语：** 做完这页后，最自然的下一步是把可训练数据送入正式微调项目，或继续看受限预算下的数据与配置该怎么取舍。
- [62. Instruction Fine-Tuning Project | 指令微调项目](./62_Instruction_Fine_Tuning_Project.md)
- [65. QLoRA Selection Project | QLoRA 选型项目](./65_QLoRA_Selection_Project.md)
---
### Step 1: 定义项目闸门与实验口径
先回答一个问题：这批数据是否具备进入训练的资格？本节只负责数据质量闸门，不负责证明某个模型训练得更快或效果更好。

| 实验层级 | 输入 | 主要操作 | 主要产物 |
|:---|:---|:---|:---|
| CPU 主实验 | 原始 SFT 样本 | 字段、模板、重复、长度和评测覆盖审计 | 清洗前后报告与 `accept / tune / reject` |
| GPU 下游验证（可选） | 通过闸门的数据 | 交给 60 / 62 做固定口径训练或推理验证 | 训练质量与资源报告，不归因给 64 |

先固定任务目标、数据来源、模板格式、长度口径和最小评测集合，避免把“样本更干净”与“模型效果提升”混为一谈。

### Step 2（CPU 项目设计）：建立数据质量基线
先确认原始数据合法，再决定是否清洗；不能拿几条样本的直观印象代替数据集审计。

- 统计样本数、空 prompt、空 response、重复样本、超长样本和长度分布。
- 检查必填字段、`messages` 结构、role 顺序和最后一个 assistant turn。
- 检查评测样例是否覆盖核心任务和主要输出格式。
- 记录阈值及其含义；阈值是准入规则，不是模型质量指标。

### Step 3（CPU 项目设计）：比较清洗前后变化
清洗实验比较的是数据风险是否下降，以及损失了多少样本；它不能单独证明训练质量提升。

- 对清洗前、清洗后数据使用同一套审计函数。
- 报告保留样本数、移除样本数、问题类型变化、重复率和长度分位数。
- 字段缺失、空 response 或模板错位属于 blocker；轻微重复和长度异常可进入 `tune`。
- 如需验证 loss、任务指标、吞吐或显存，把固定版本数据交给 60 / 62 的下游实验。

### Step 4（CPU 项目设计）：输出数据准入结论
项目结论要回答“现在能不能交给训练项目”，而不是只输出一张统计表。

- `accept`：必填字段、模板和评测覆盖满足当前规则。
- `tune`：没有致命 blocker，但仍需处理重复、长度或评测缺口。
- `reject`：存在会破坏训练目标或评测解释力的 blocker。
- 报告至少包含审计结果、规则版本、清洗动作、保留样本数和下一轮动作。

### Step 5（CPU 代码实践）：把审计规则实现为可复用函数
本节的代码练习聚焦审计、清洗前后对比和决策，不要求重复实现 60 / 62 的训练循环。

#### 图解：09 / 13 / 30 / 32 如何收束到 64 数据质量项目

`64` 不重复实现训练循环，而是把前面几节的数据与训练口径收成一份训练前的质量审计报告。

```text
09 SFT data        input_ids / labels / loss mask
      │
13 End-to-end      train / val loop and minimal report
      │
30 Long context    truncation / length budget / task fit
      │
32 Data engineering template / fields / eval coverage / cleaning
      ▼
64 SFT Data Quality Project
      ├─ sample audit
      ├─ template audit
      ├─ eval coverage review
      └─ accept / tune / reject
```

项目页最小产物：

| 产物 | 你至少要记录什么 | 作用 |
|:---|:---|:---|
| 样本审计 | 样本数、空 response、重复率、超长样本 | 判断数据是否可信 |
| 模板审计 | 必填字段、role 顺序、assistant 收束位置 | 判断模板是否可训练 |
| 评测覆盖 | 核心任务样例数、格式覆盖、缺口 | 判断训练后是否可验证 |
| 项目结论 | accept / tune / reject | 输出训练前决策 |

### 参数口径说明

`max_prompt_chars` 和 `max_response_chars` 是数据审计阈值，用于发现风险，不等于 tokenizer 的实际截断长度；`required_keys` 定义模板必须存在的字段；`eval_cases` 决定评测覆盖。正式数据准入还应记录清洗前后样本数、重复率、长度分位数和实际 token 截断数。

```python
from typing import Dict, List

```


```python
# TODO: 完成 SFT 数据质量项目的样本审计、模板审计和项目结论
# 目标：把训练前的数据检查收束成一份可用于 accept / tune / reject 的项目报告。
# CPU 题目区只验证数据规则；它不能证明清洗后模型的训练效果或 GPU 资源收益。

def audit_sft_samples(examples: List[Dict[str, str]], max_prompt_chars: int, max_response_chars: int) -> Dict[str, float]:
    """统计空字段、重复样本、字符长度风险和平均 response 长度。

    max_prompt_chars / max_response_chars 是字符审计阈值，不是 tokenizer 截断长度；
    返回摘要字段必须能支持清洗前后对比。
    """
    # TODO 1：按 prompt/response 组合识别重复；分别统计空 prompt 与空 response。
    # sample_count = ???；empty_prompt_count = ???；empty_response_count = ???；duplicate_count = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def audit_chat_template(records: List[Dict[str, object]], required_keys: List[str]) -> Dict[str, int]:
    """检查必填字段、messages 是否为空以及 assistant 收尾是否正确。

    records 中的 messages 应为 turn 列表；缺字段、空列表和错误收尾分别计数。
    """
    # TODO 2：required_keys 缺失计入 missing_key_count；最后一个 role 非 assistant
    # 时计入 assistant_tail_error_count，不要把空 messages 当作正确模板。
    # missing_key_count = ???；empty_messages_count = ???；assistant_tail_error_count = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def review_sft_data_project(sample_audit: Dict[str, float], template_audit: Dict[str, int], eval_cases: List[Dict[str, str]]) -> Dict[str, object]:
    """根据样本、模板和评测覆盖审计输出数据准入决策。

    返回 decision、eval_case_count、blockers 和 next_action；blocker 优先于普通问题。
    """
    # TODO 3：空 response、字段缺失、模板收尾错误等直接影响训练目标的问题，
    # 应进入 blockers；不要仅按 issue 总数给出结论。
    # blockers = ???；decision = ???；project_ready = ???；next_action = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_cleaning_effect(before_audit: Dict[str, float], after_audit: Dict[str, float]) -> Dict[str, float]:
    """比较清洗前后的样本保留量和审计问题变化。

    返回 removed_samples、issue_delta 和 issue_reduction_ratio；问题减少不等于模型质量提升。
    """
    # TODO 4：使用 sample_count 与 issue_count；缺失字段不能静默当作 0。
    # removed_samples = ???；issue_delta = ???；issue_reduction_ratio = ???。
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_sft_data_quality_project():
    try:
        examples = [
            {'prompt': '介绍 LoRA。', 'response': 'LoRA 是一种低秩适配方法。'},
            {'prompt': '介绍 LoRA。', 'response': 'LoRA 是一种低秩适配方法。'},
            {'prompt': '给出训练建议。', 'response': ''},
            {'prompt': 'x' * 20, 'response': '可训练'},
        ]
        sample_audit = audit_sft_samples(examples, max_prompt_chars=12, max_response_chars=20)
        assert sample_audit['sample_count'] == 4, '样本数统计错误'
        assert sample_audit['duplicate_count'] == 1, '重复样本统计错误'
        assert sample_audit['empty_response_count'] == 1, '空 response 统计错误'
        assert sample_audit['over_prompt_limit_count'] == 1, '超长 prompt 统计错误'

        records = [
            {'messages': [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}], 'task': 'chat'},
            {'messages': [], 'task': 'chat'},
            {'messages': [{'role': 'user', 'content': 'ask'}]},
        ]
        template_audit = audit_chat_template(records, required_keys=['messages', 'task'])
        assert template_audit['missing_key_count'] == 1, '缺失字段统计错误'
        assert template_audit['empty_messages_count'] == 1, '空 messages 统计错误'
        assert template_audit['assistant_tail_error_count'] == 1, 'assistant 收尾统计错误'

        eval_cases = [
            {'task': 'chat', 'expected_format': 'answer'},
            {'task': 'summarization', 'expected_format': 'bullet'},
        ]
        decision = review_sft_data_project(sample_audit, template_audit, eval_cases)
        assert decision['decision'] == 'reject', '高风险数据不应直接通过'
        assert decision['eval_case_count'] == 2, '评测样例数统计错误'
        assert decision['blockers'], '应给出 blocker 列表'

        clean_audit = audit_sft_samples(
            [
                {'prompt': '问：什么是 SFT？', 'response': '答：监督微调。'},
                {'prompt': '问：为什么要评测？', 'response': '答：为了验证质量。'},
            ],
            max_prompt_chars=20,
            max_response_chars=20,
        )
        clean_template = audit_chat_template(
            [
                {'messages': [{'role': 'user', 'content': '问'}, {'role': 'assistant', 'content': '答'}], 'task': 'chat'},
            ],
            required_keys=['messages', 'task'],
        )
        ready = review_sft_data_project(clean_audit, clean_template, eval_cases)
        assert ready['decision'] == 'accept', '干净数据应可进入训练'
        cleaning = summarize_cleaning_effect(sample_audit, clean_audit)
        assert cleaning['removed_sample_count'] == 2, '清洗删除样本数统计错误'
        assert cleaning['issue_delta'] > 0, '清洗前后问题数变化错误'
        print('所有测试通过！')
    except AssertionError as e:
        print(f'测试失败: {e}')
        raise
    except Exception as e:
        print(f'发生错误: {e}')
        raise


test_sft_data_quality_project()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 审计 SFT 样本
def audit_sft_samples(examples: List[Dict[str, str]], max_prompt_chars: int, max_response_chars: int) -> Dict[str, float]:
    seen = set()
    empty_prompt_count = 0
    empty_response_count = 0
    duplicate_count = 0
    over_prompt_limit_count = 0
    over_response_limit_count = 0
    total_response_chars = 0

    for example in examples:
        prompt = example.get('prompt', '')
        response = example.get('response', '')
        pair = (prompt, response)
        if pair in seen:
            duplicate_count += 1
        else:
            seen.add(pair)

        if not prompt.strip():
            empty_prompt_count += 1
        if not response.strip():
            empty_response_count += 1
        if len(prompt) > max_prompt_chars:
            over_prompt_limit_count += 1
        if len(response) > max_response_chars:
            over_response_limit_count += 1

        total_response_chars += len(response)

    sample_count = len(examples)
    average_response_chars = total_response_chars / sample_count if sample_count else 0.0
    issue_count = empty_prompt_count + empty_response_count + duplicate_count + over_prompt_limit_count + over_response_limit_count
    return {
        'sample_count': sample_count,
        'empty_prompt_count': empty_prompt_count,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_prompt_limit_count': over_prompt_limit_count,
        'over_response_limit_count': over_response_limit_count,
        'average_response_chars': average_response_chars,
        'issue_count': issue_count,
    }


# TODO 2: 审计模板与字段
def audit_chat_template(records: List[Dict[str, object]], required_keys: List[str]) -> Dict[str, int]:
    missing_key_count = 0
    empty_messages_count = 0
    assistant_tail_error_count = 0

    for record in records:
        if any(key not in record for key in required_keys):
            missing_key_count += 1

        messages = record.get('messages', [])
        if not messages:
            empty_messages_count += 1
            continue

        last_role = messages[-1].get('role')
        if last_role != 'assistant':
            assistant_tail_error_count += 1

    return {
        'record_count': len(records),
        'missing_key_count': missing_key_count,
        'empty_messages_count': empty_messages_count,
        'assistant_tail_error_count': assistant_tail_error_count,
        'issue_count': missing_key_count + empty_messages_count + assistant_tail_error_count,
    }


# TODO 3: 输出项目结论
def review_sft_data_project(sample_audit: Dict[str, float], template_audit: Dict[str, int], eval_cases: List[Dict[str, str]]) -> Dict[str, object]:
    blockers: List[str] = []

    if sample_audit['empty_response_count'] > 0:
        blockers.append('empty_response')
    if template_audit['missing_key_count'] > 0:
        blockers.append('missing_template_keys')
    if template_audit['assistant_tail_error_count'] > 0:
        blockers.append('assistant_tail_error')
    if not eval_cases:
        blockers.append('missing_eval_cases')

    total_issues = sample_audit['issue_count'] + template_audit['issue_count']
    if blockers:
        decision = 'reject'
        next_action = 'fix_template_or_labels'
    elif total_issues > 0:
        decision = 'tune'
        next_action = 'clean_duplicates_or_length_outliers'
    else:
        decision = 'accept'
        next_action = 'promote_to_finetuning'

    return {
        'decision': decision,
        'blockers': blockers,
        'eval_case_count': len(eval_cases),
        'total_issue_count': total_issues,
        'next_action': next_action,
    }


# TODO 4: 比较清洗前后的数据质量变化
def summarize_cleaning_effect(before_audit: Dict[str, float], after_audit: Dict[str, float]) -> Dict[str, float]:
    before_samples = int(before_audit['sample_count'])
    after_samples = int(after_audit['sample_count'])
    before_issues = int(before_audit['issue_count'])
    after_issues = int(after_audit['issue_count'])
    issue_delta = before_issues - after_issues
    return {
        'before_sample_count': before_samples,
        'after_sample_count': after_samples,
        'removed_sample_count': before_samples - after_samples,
        'before_issue_count': before_issues,
        'after_issue_count': after_issues,
        'issue_delta': issue_delta,
        'issue_reduction_ratio': round(issue_delta / before_issues, 4) if before_issues else 0.0,
    }

```

### 解析

这一页保留 `3` 个核心 TODO：样本审计、模板审计和项目结论。它不要求把数据清洗流水线全部重写，而是要求把训练前的数据质量判断收成可执行的项目闸门。

**1. TODO 1: 审计 SFT 样本**
- **实现方式**：遍历 `prompt / response` 样本，统计空字段、重复样本、超长样本和平均 response 长度。
- **关键点**：训练前先确认数据可信。空 response 会让样本没有有效监督，重复样本会放大小数据过拟合风险，超长样本会改变截断和预算口径。
- **项目意义**：这一步把第 `09` 节的单条样本正确性扩展成项目级数据集审计。

**2. TODO 2: 审计模板与字段**
- **实现方式**：检查必填字段、`messages` 是否为空，以及最后一个 turn 是否收束到 `assistant`。
- **关键点**：模板不稳定时，训练 loss 即使下降，也可能对应错误的监督目标。
- **项目意义**：这一步把第 `32` 节的数据工程风险提前暴露在训练前，而不是把问题拖到项目后期。

**3. TODO 3: 输出项目结论**
- **实现方式**：把样本审计、模板审计和评测覆盖统一收成 `accept / tune / reject`。
- **关键点**：blocker 要先于总 issue 数判断。像空 response、缺模板字段、assistant 收尾错误这类问题会直接破坏训练与评测解释力，应优先走 `reject`；只有在没有 blocker、但仍有重复样本或长度异常时，才进入 `tune`。
- **项目意义**：这一步把 `64` 固定成训练前的数据质量闸门，而不是一组分散的清洗脚本。
- **关键点**：项目结论不只依赖样本数，还要看 blocker 是否会直接破坏训练与评测解释力。
- **项目意义**：这一步把 `64` 收成训练前的数据质量闸门，而不是单纯的清洗脚本集合。

### 可选：统一项目报告导出
默认关闭。完成样本审计、模板审计和评测样例检查后，再导出项目报告。报告模板见 `docs/verification/fine_tuning_projects.md`。

```python
try:
    from tools.fine_tuning_project_runtime import preflight_runtime, runtime_snapshot, save_project_report, validate_project_config
except ModuleNotFoundError:
    preflight_runtime = lambda torch_module, run_mode='cpu', **kwargs: {'run_mode': run_mode, 'ready': False, 'reasons': ['共享运行时工具不可用']}
    runtime_snapshot = lambda: {'device': 'unknown'}
    validate_project_config = lambda config: []
    save_project_report = None
RUN_MODE = 'cpu'  # cpu / dry_run / real_gpu；本节默认只运行 CPU 数据审计。
PROJECT_ID = '64_sft_data_quality'
PROJECT_RESULT_PATH = 'benchmarks/results/64_sft_data_quality.json'
PROJECT_CONFIG = {'project': PROJECT_ID, 'model': 'template', 'dtype': 'fp32', 'batch_size': 1, 'seq_len': 128, 'steps': 1, 'seed': 42, 'run_mode': RUN_MODE}
RUN_PROJECT_EXPORT = False  # True 只保存已完成的数据质量报告。
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
