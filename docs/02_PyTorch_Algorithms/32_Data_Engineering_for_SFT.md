# 32. Data Engineering for SFT | SFT 数据契约与泄漏审计
**难度：** Medium | **环境：** [CPU 主线](#runtime-environment) | **标签：** `后训练`, `SFT`, `数据治理` | **目标人群：** 后训练学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

当后训练准入判断确认问题需要进入 SFT 后，候选示范仍不能直接送入模板与标签构造。来源不可追溯、训练样本重复或训练任务泄漏到独立评测集，都会让后续损失下降失去可信解释。

本节沿着“字段契约 → 双指纹审计 → 准入清单”处理这些风险：任务输入指纹用于检查训练—评测泄漏，完整样本指纹用于检查训练集重复，所有拒绝原因都随样本保留。完成后，你应该能生成可追溯的数据准入结果，并把通过审计的记录交给监督边界与标签构造环节。

**关键词：** `data contract`, `deduplication`, `split isolation`, `decontamination`

---
<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。
- **框架核对：** 如需检查准入记录能否无损进入 Hugging Face Datasets，使用 [`PT-CPU`](../topic_discussion/post_training_optimization/quickstart.md#pt-cpu-quickstart)；完成 Step 4 后运行[框架一致性核对](#framework-check)。

---
## 前置阅读

**导语：** 33 节先确定是否需要 SFT 并生成数据需求单；本节再用结构化样本与稳定身份检查候选示范能否进入训练。
- [33. Post-Training Admission and Stage Selection | 后训练准入与阶段选择](./33_Fine_Tuning_Readiness.md)
- [04. Python Config and Data Entry | Python 配置与数据组织](../00_Prerequisites/04_Python_Config_and_Data_Entry.md)

---

## Step 1：一条 SFT 样本需要具备哪些来源、切分与监督字段

一条 SFT 样本既要能够追溯来源，也要明确它属于哪个数据切分、模型接收什么任务条件，以及哪部分内容提供监督。只有身份、切分、任务输入和监督目标都满足契约，样本才能进入后续的数据审计。

先规范化身份、切分、指令、上下文和回答字段，以下用 $x$ 表示规范化后的记录。基础契约可以写成布尔合取：

$$
C(x)=I_{\mathrm{id}}(x)\land I_{\mathrm{source}}(x)\land I_{\mathrm{split}}(x)\land I_{\mathrm{instruction}}(x)\land I_{\mathrm{response}}(x).
$$

其中，$I_j(x)\in\{0,1\}$ 表示字段 $j$ 是否满足契约，$\land$ 表示所有字段条件必须同时成立。$C(x)$ 只说明单条记录字段完整，不说明它是否与其他训练样本重复或与评测任务重合。贯穿示例沿用 33 节的三步流程任务：候选数据中同时放入一条合格示范、一条内容重复示范和一条与冻结评测任务重合的示范，观察它们如何得到不同准入结果。

| 契约维度 | 接受条件 | 不满足时说明什么 |
|:---|:---|:---|
| 可追溯身份 | 样本标识和来源标识非空 | 无法回查数据来源与失败样本 |
| 切分角色 | 明确属于训练、验证或测试切分 | 无法判断样本应进入哪个数据集合 |
| 任务输入 | 指令非空，可选上下文与指令共同定义任务 | 模型没有完整的任务条件 |
| 监督目标 | 回答非空，并与任务输入保持字段分离 | 无有效监督，或无法正确构造标签范围 |

![SFT 数据准入链](./../public/02_PyTorch_Algorithms/32_sft_data_admission_flow.svg)
## Step 2：如何识别训练集重复与评测数据泄漏

身份重复只能发现同一样本身份被重复使用，无法发现更换身份后的同一内容。训练集去重需要比较完整样本，训练与评测之间的泄漏检查只比较任务输入；如果泄漏检查混入回答，同一任务换一个回答就可能逃过检查。两类指纹采用相同的文本规范化规则，但比较范围不同。

设规范化后的指令、上下文和回答分别为 $i(x)$、$c(x)$、$r(x)$，则两类比较键为

$$
f_{\mathrm{task}}(x)=\big(i(x),c(x)\big),\qquad
f_{\mathrm{example}}(x)=\big(i(x),c(x),r(x)\big).
$$

两类比较键共享规范化规则，但字段组合不同：任务输入指纹刻画“模型要回答什么”，完整样本指纹还包含“希望模型怎样回答”。

| 比较键 | 组成 | 比较范围 | 发现的问题 |
|:---|:---|:---|:---|
| 样本身份 | 数据来源声明的稳定标识 | 训练集内部 | 同一身份被重复使用 |
| 任务输入指纹 | 规范化后的完整任务条件，不包含目标回答 | 训练集与独立评测集 | 训练任务泄漏到评测集 |
| 完整样本指纹 | 规范化后的任务条件与目标回答 | 训练集内部 | 不同身份指向相同监督样本 |

![SFT 数据风险矩阵](./../public/02_PyTorch_Algorithms/32_sft_data_risk_matrix.svg)
## Step 3：哪些样本可以准入，拒绝原因如何保留

契约失败、样本身份重复、完整样本重复和训练—评测泄漏可能同时发生，审计不能在发现第一项风险后停止。每条训练样本都应保留来源、两类指纹、准入结果和全部拒绝原因；只有拒绝原因集合为空的样本，才能进入模板、标签、长度与序列拼接处理。

把所有命中的拒绝原因记为集合 $R(x)$：

$$
R(x)=R_{\mathrm{contract}}(x)\cup R_{\mathrm{id}}(x)\cup R_{\mathrm{duplicate}}(x)\cup R_{\mathrm{leak}}(x),\qquad
A(x)=\mathbf{1}[R(x)=\varnothing].
$$

准入变量 $A(x)$ 由完整原因集合决定，因此同一样本同时命中的风险都必须保留。准入结果不是本节终点：通过的记录需要连同来源与切分信息交给 13 节构造监督标签，被拒绝的记录保留全部原因供数据修复。

| 拒绝原因 | 触发条件 | 修复方向 |
|:---|:---|:---|
| 身份字段缺失 | `sample_id`、`source_id` 或 `split` 为空 | 补齐来源字段或移除样本 |
| 切分值非法 | `split` 不属于 train、validation、test | 改正切分值并重新审计 |
| 任务或回答缺失 | `instruction` 或 `response` 为空 | 补齐任务与监督目标，或移除样本 |
| 身份重复 | 训练集中重复出现相同样本标识 | 核对来源和数据版本 |
| 完整样本重复 | 不同身份具有相同任务输入与回答 | 去重，并保留来源标识完整且通过契约检查的样本 |
| 评测任务泄漏 | 训练任务输入与独立评测任务重合 | 重新切分或移除污染样本 |

| 本节输出 | 进入 13 节后的用途 |
|:---|:---|
| `instruction`、`input` | 构造模型输入与提示模板 |
| `response` | 构造需要参与损失计算的监督标签 |
| `sample_id`、`source_id`、`split` | 保留样本追踪和训练—评测隔离证据 |
| 准入样本身份集合 | 限定可以进入训练批次的记录 |
| 拒绝原因 | 修复候选数据，不进入标签构造 |
<a id="mechanism-practice"></a>
## Step 4：实现 SFT 数据契约与泄漏审计

题目区使用规范化样本、切分身份和追踪字段复现 SFT 数据准入。骨架提供记录结构；学习者补全任务输入与完整样本指纹、重复和泄漏审计，以及下游训练接口检查。测试覆盖回答变化、重复身份、跨切分污染、缺失监督和追踪字段保留。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|:---|:---|:---|:---|
| TODO 1：构造两类指纹<br>`build_sft_fingerprints` | 分离任务输入与完整监督样本的比较键 | 回答变化只能改变完整样本指纹 | 修改回答后，任务输入指纹保持不变，完整样本指纹随之改变 |
| TODO 2：审计重复与泄漏<br>`audit_sft_dataset` | 分别判断完整样本重复和任务输入泄漏 | 风险独立记录，同一样本可同时命中 | 重复身份、重复样本、评测泄漏和缺失监督被分别记录；只有准入样本进入训练接口并保留追踪字段 |

```python
from typing import Dict, List

```


```python
def normalize_sft_record(record: Dict[str, object]) -> Dict[str, str]:
    """统一字段类型并移除字符串首尾空白。"""
    return {
        "sample_id": str(record.get("sample_id", "")).strip(),
        "source_id": str(record.get("source_id", "")).strip(),
        "split": str(record.get("split", "")).strip(),
        "instruction": str(record.get("instruction", "")).strip(),
        "input": str(record.get("input", "")).strip(),
        "response": str(record.get("response", "")).strip(),
    }


def assess_sft_contract(record: Dict[str, str]) -> Dict[str, object]:
    """判定身份与监督契约，并保留可解释的缺失字段。"""
    required_identity_fields = ("sample_id", "source_id", "split")
    missing_identity_fields = [name for name in required_identity_fields if not record[name]]
    split_allowed = record["split"] in {"train", "validation", "test"}
    supervision_complete = bool(record["instruction"] and record["response"])
    contract_issues = [f"missing_{name}" for name in missing_identity_fields]
    if record["split"] and not split_allowed:
        contract_issues.append("invalid_split")
    if not record["instruction"]:
        contract_issues.append("missing_instruction")
    if not record["response"]:
        contract_issues.append("missing_response")
    admitted = not contract_issues
    return {"missing_identity_fields": missing_identity_fields, "supervision_complete": supervision_complete, "contract_issues": contract_issues, "admitted": admitted}


def build_sft_fingerprints(record: Dict[str, str]) -> Dict[str, tuple]:
    """分别构造任务输入与完整监督样本的比较键。"""
    instruction = record["instruction"].casefold()
    input_text = record["input"].casefold()
    response = record["response"].casefold()
    # ==========================================
    # TODO 1：分离任务输入指纹与完整样本指纹。
    # task_fingerprint = ???
    # example_fingerprint = ???
    # ==========================================
    raise NotImplementedError("TODO 1：构造两类指纹")
    return {"task": task_fingerprint, "example": example_fingerprint}


def audit_sft_dataset(records: List[Dict[str, str]], evaluation_task_fingerprints: set[tuple[str, str]]) -> Dict[str, object]:
    """输出训练重复、跨切分泄漏和准入原因清单。"""
    seen_sample_ids = set()
    seen_training_examples = set()
    rows = []
    admitted_sample_ids = []
    for record in records:
        if record["split"] in {"validation", "test"}:
            continue
        contract = assess_sft_contract(record)
        fingerprints = build_sft_fingerprints(record)
        is_duplicate_identity = contract["admitted"] and record["sample_id"] in seen_sample_ids
        # ==========================================
        # TODO 2：分别判断完整样本重复和任务输入泄漏。
        # is_duplicate_example = ???
        # is_evaluation_leak = ???
        # ==========================================
        raise NotImplementedError("TODO 2：审计重复与泄漏")
        reasons = []
        reasons.extend(contract["contract_issues"])
        if is_duplicate_identity:
            reasons.append("duplicate_identity")
        if is_duplicate_example:
            reasons.append("duplicate_example")
        if is_evaluation_leak:
            reasons.append("evaluation_task_leak")
        admitted = not reasons
        rows.append({"sample_id": record["sample_id"], "source_id": record["source_id"], "task_fingerprint": fingerprints["task"], "example_fingerprint": fingerprints["example"], "decision": "admit" if admitted else "reject", "reasons": reasons})
        if admitted:
            admitted_sample_ids.append(record["sample_id"])
        if contract["admitted"]:
            seen_sample_ids.add(record["sample_id"])
            seen_training_examples.add(fingerprints["example"])
    return {"rows": rows, "admitted_sample_ids": admitted_sample_ids}


def build_sft_training_examples(records: List[Dict[str, str]], admitted_sample_ids: List[str], template_prefix: str) -> List[Dict[str, str]]:
    """只把风险审计通过的样本装配为训练示例。"""
    admitted_ids = set(admitted_sample_ids)
    examples = []
    for record in records:
        if record["sample_id"] not in admitted_ids:
            continue
        prompt = template_prefix + record["instruction"]
        if record["input"]:
            prompt += "\n### Input\n" + record["input"]
        examples.append({"sample_id": record["sample_id"], "source_id": record["source_id"], "split": record["split"], "prompt": prompt, "response": record["response"]})
    return examples
```

## 测试

运行下方 CPU 机制测试，检查样本字段、重复与泄漏风险、最终准入清单，以及交给 13 节的训练记录合同。


```python
def _sample_sft_records():
    """返回包含身份重复、内容重复、评测污染、空监督与干净样本的样例。"""
    raw = [
        {"sample_id": "train-1", "source_id": "source-a", "split": "train", "instruction": " 按三步处理退款请求 ", "input": "订单已签收", "response": " 申请、核验、确认。 "},
        {"sample_id": "train-2", "source_id": "source-b", "split": "train", "instruction": "按三步处理退款请求", "input": "订单已签收", "response": "申请、核验、确认。"},
        {"sample_id": "train-3", "source_id": "source-c", "split": "train", "instruction": "说明退款进度", "input": "已经提交申请", "response": ""},
        {"sample_id": "train-1", "source_id": "source-d", "split": "train", "instruction": "解释退款时限", "input": "订单已签收", "response": "核验后告知处理时限。"},
        {"sample_id": "train-4", "source_id": "source-e", "split": "train", "instruction": "说明修改地址步骤", "input": "订单尚未发货", "response": "核对订单、更新地址、确认结果。"},
    ]
    return [normalize_sft_record(record) for record in raw]


def _assert_raises(error_type, fn) -> None:
    """断言调用抛出指定异常。"""
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"预期抛出 {error_type.__name__}")


def test_contract_assessment():
    """身份与监督必须分别判定，且同时满足才准入。"""
    valid, duplicate, empty = _sample_sft_records()[:3]
    assert assess_sft_contract(valid) == {"missing_identity_fields": [], "supervision_complete": True, "contract_issues": [], "admitted": True}
    assert assess_sft_contract(empty) == {"missing_identity_fields": [], "supervision_complete": False, "contract_issues": ["missing_response"], "admitted": False}
    missing_source = {**duplicate, "source_id": ""}
    assert assess_sft_contract(missing_source)["missing_identity_fields"] == ["source_id"]
    invalid_split = {**duplicate, "split": "holdout"}
    assert assess_sft_contract(invalid_split)["contract_issues"] == ["invalid_split"]


def test_fingerprint_semantics():
    """回答变化只改变完整样本指纹，不改变任务输入指纹。"""
    first = normalize_sft_record({"instruction": "解释 LoRA", "input": "", "response": "答案 A"})
    second = normalize_sft_record({"instruction": "解释 LoRA", "input": "", "response": "答案 B"})
    first_keys = build_sft_fingerprints(first)
    second_keys = build_sft_fingerprints(second)
    assert first_keys["task"] == second_keys["task"]
    assert first_keys["example"] != second_keys["example"]


def test_dataset_audit():
    """完整重复和评测任务泄漏必须独立记录。"""
    records = _sample_sft_records()
    evaluation_tasks = {build_sft_fingerprints(records[0])["task"]}
    report = audit_sft_dataset(records, evaluation_tasks)
    assert report["admitted_sample_ids"] == ["train-4"]
    assert report["rows"][0]["reasons"] == ["evaluation_task_leak"]
    assert report["rows"][1]["reasons"] == ["duplicate_example", "evaluation_task_leak"]
    assert report["rows"][2]["reasons"] == ["missing_response"]
    assert report["rows"][3]["reasons"] == ["duplicate_identity"]


def test_admission_handoff():
    """只有准入清单通过的样本才能进入模板化。"""
    records = _sample_sft_records()
    evaluation_tasks = {build_sft_fingerprints(records[0])["task"]}
    report = audit_sft_dataset(records, evaluation_tasks)
    examples = build_sft_training_examples(records, report["admitted_sample_ids"], "### Instruction\n")
    assert [example["sample_id"] for example in examples] == ["train-4"]


def test_training_interface_contract():
    """交给 13 节的记录必须同时保留监督内容和审计身份。"""
    records = _sample_sft_records()
    evaluation_tasks = {build_sft_fingerprints(records[0])["task"]}
    report = audit_sft_dataset(records, evaluation_tasks)
    examples = build_sft_training_examples(records, report["admitted_sample_ids"], "### Instruction\n")
    assert set(examples[0]) == {"sample_id", "source_id", "split", "prompt", "response"}
    assert examples[0]["source_id"] == "source-e" and examples[0]["split"] == "train"
    rejected_ids = {row["sample_id"] for row in report["rows"] if row["decision"] == "reject"}
    assert not ({example["sample_id"] for example in examples} & rejected_ids)


def test_rejected_record_does_not_poison_seen_state():
    """字段不完整的记录不能让后续合法样本被误判为重复身份。"""
    records = [
        normalize_sft_record({"sample_id": "same", "source_id": "", "split": "train", "instruction": "解释 SFT", "response": "坏记录"}),
        normalize_sft_record({"sample_id": "same", "source_id": "source-ok", "split": "train", "instruction": "解释 SFT", "response": "监督微调"}),
    ]
    report = audit_sft_dataset(records, set())
    assert report["rows"][0]["reasons"] == ["missing_source_id"]
    assert report["rows"][1]["decision"] == "admit"


test_contract_assessment()
test_fingerprint_semantics()
test_dataset_audit()
test_admission_handoff()
test_training_interface_contract()
test_rejected_record_does_not_poison_seen_state()
print("✅ SFT 契约、指纹、泄漏审计与准入清单测试通过。")
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
def normalize_sft_record(record: Dict[str, object]) -> Dict[str, str]:
    return {
        "sample_id": str(record.get("sample_id", "")).strip(),
        "source_id": str(record.get("source_id", "")).strip(),
        "split": str(record.get("split", "")).strip(),
        "instruction": str(record.get("instruction", "")).strip(),
        "input": str(record.get("input", "")).strip(),
        "response": str(record.get("response", "")).strip(),
    }


def assess_sft_contract(record: Dict[str, str]) -> Dict[str, object]:
    required_identity_fields = ("sample_id", "source_id", "split")
    missing_identity_fields = [name for name in required_identity_fields if not record[name]]
    split_allowed = record["split"] in {"train", "validation", "test"}
    supervision_complete = bool(record["instruction"] and record["response"])
    contract_issues = [f"missing_{name}" for name in missing_identity_fields]
    if record["split"] and not split_allowed:
        contract_issues.append("invalid_split")
    if not record["instruction"]:
        contract_issues.append("missing_instruction")
    if not record["response"]:
        contract_issues.append("missing_response")
    admitted = not contract_issues
    return {"missing_identity_fields": missing_identity_fields, "supervision_complete": supervision_complete, "contract_issues": contract_issues, "admitted": admitted}


def build_sft_fingerprints(record: Dict[str, str]) -> Dict[str, tuple]:
    instruction = record["instruction"].casefold()
    input_text = record["input"].casefold()
    response = record["response"].casefold()
    # TODO 1：构造任务输入指纹与完整样本指纹。
    task_fingerprint = (instruction, input_text)
    example_fingerprint = (instruction, input_text, response)
    return {"task": task_fingerprint, "example": example_fingerprint}


def audit_sft_dataset(records: List[Dict[str, str]], evaluation_task_fingerprints: set[tuple[str, str]]) -> Dict[str, object]:
    seen_sample_ids = set()
    seen_training_examples = set()
    rows = []
    admitted_sample_ids = []
    for record in records:
        if record["split"] in {"validation", "test"}:
            continue
        contract = assess_sft_contract(record)
        fingerprints = build_sft_fingerprints(record)
        is_duplicate_identity = contract["admitted"] and record["sample_id"] in seen_sample_ids
        # TODO 2：独立判断完整样本重复和评测任务泄漏。
        is_duplicate_example = contract["admitted"] and fingerprints["example"] in seen_training_examples
        is_evaluation_leak = contract["admitted"] and fingerprints["task"] in evaluation_task_fingerprints
        reasons = []
        reasons.extend(contract["contract_issues"])
        if is_duplicate_identity:
            reasons.append("duplicate_identity")
        if is_duplicate_example:
            reasons.append("duplicate_example")
        if is_evaluation_leak:
            reasons.append("evaluation_task_leak")
        admitted = not reasons
        rows.append({"sample_id": record["sample_id"], "source_id": record["source_id"], "task_fingerprint": fingerprints["task"], "example_fingerprint": fingerprints["example"], "decision": "admit" if admitted else "reject", "reasons": reasons})
        if admitted:
            admitted_sample_ids.append(record["sample_id"])
        if contract["admitted"]:
            seen_sample_ids.add(record["sample_id"])
            seen_training_examples.add(fingerprints["example"])
    return {"rows": rows, "admitted_sample_ids": admitted_sample_ids}


def build_sft_training_examples(records: List[Dict[str, str]], admitted_sample_ids: List[str], template_prefix: str) -> List[Dict[str, str]]:
    admitted_ids = set(admitted_sample_ids)
    examples = []
    for record in records:
        if record["sample_id"] not in admitted_ids:
            continue
        prompt = template_prefix + record["instruction"]
        if record["input"]:
            prompt += "\n### Input\n" + record["input"]
        examples.append({"sample_id": record["sample_id"], "source_id": record["source_id"], "split": record["split"], "prompt": prompt, "response": record["response"]})
    return examples
```

### 解析

#### TODO 1：构造两类指纹

- 机制：任务输入指纹只包含指令与上下文，完整样本指纹再加入回答；两者采用相同的表示规范化。
- 不变量：回答变化不能改变任务输入指纹，但必须改变完整样本指纹。测试用同一问题的两个不同答案验证这一差异。

#### TODO 2：审计重复与泄漏

- 机制：完整样本指纹查询训练已见集合，任务输入指纹查询冻结评测集合；缺失身份字段、非法切分、空任务、空回答、身份重复和内容风险分别记录。
- 不变量：同一条样本可以同时命中完整重复和评测泄漏；只有 `contract_issues` 与内容风险均为空时才能进入模板化。交给 13 节的训练记录必须继续携带 `sample_id`、`source_id` 与 `split`，被拒绝的样本不得进入训练记录。测试同时检查风险组合、非法切分和训练接口合同。
<a id="framework-check"></a>

## 框架一致性核对（可选）

使用相同的原始记录与冻结评测指纹，比较 Step 4 审计结果和 Hugging Face Datasets 中的字段结构、准入集合与样本血缘。

| 核对对象 | Step 4 教学实现 | 成熟框架 CPU 路径 | 共同输入与通过条件 |
| --- | --- | --- | --- |
| 字段结构 | `build_sft_training_examples` 生成训练记录 | `Features` 与 `Dataset.from_list` | 使用相同准入样本；身份、切分、提示和回答字段完整且类型一致 |
| 准入集合 | `audit_sft_dataset` 排除合同错误、重复和泄漏样本 | `Dataset.to_list` 读取准入行 | 使用相同原始记录与评测指纹；准入 ID 集合一致，被拒样本不进入 Dataset |
| 样本血缘 | 保留 `sample_id`、`source_id`、`split` | Dataset 行与特征定义 | 使用相同准入样本；转换后身份字段值保持不变 |
| 下游监督字段 | 输出 `prompt` 与 `response` | Dataset 字符串列 | 使用相同准入样本；两列非空且可供 SFT 批处理器读取 |

```python
from importlib.metadata import version

try:
    from datasets import Dataset, Features, Value
except ImportError:
    framework_check = {
        "status": "skipped",
        "reason": "请使用 PT-CPU 环境运行本单元",
    }
else:
    def check_sft_data_interface():
        """核对准入后的 SFT 记录能否无损进入 Hugging Face Dataset。"""
        raw_records = [
            {"sample_id": "train-1", "source_id": "source-a", "split": "train", "instruction": "按三步处理退款请求", "input": "订单已签收", "response": "申请、核验、确认。"},
            {"sample_id": "train-2", "source_id": "source-b", "split": "train", "instruction": "说明修改地址步骤", "input": "订单尚未发货", "response": "核对订单、更新地址、确认结果。"},
            {"sample_id": "train-3", "source_id": "source-c", "split": "train", "instruction": "说明退款进度", "input": "已经提交申请", "response": ""},
        ]
        records = [normalize_sft_record(record) for record in raw_records]
        evaluation_tasks = {build_sft_fingerprints(records[0])["task"]}
        audit = audit_sft_dataset(records, evaluation_tasks)
        examples = build_sft_training_examples(
            records, audit["admitted_sample_ids"], "### Instruction\n"
        )
        features = Features({
            "sample_id": Value("string"),
            "source_id": Value("string"),
            "split": Value("string"),
            "prompt": Value("string"),
            "response": Value("string"),
        })
        dataset = Dataset.from_list(examples, features=features)
        dataset_rows = dataset.to_list()
        rejected_ids = {
            row["sample_id"] for row in audit["rows"]
            if row["decision"] == "reject"
        }
        checks = {
            "schema_preserved": set(dataset.features) == set(features),
            "admitted_rows_only": [row["sample_id"] for row in dataset_rows] == audit["admitted_sample_ids"],
            "rejected_rows_excluded": not ({row["sample_id"] for row in dataset_rows} & rejected_ids),
            "lineage_preserved": all(row["source_id"] and row["split"] == "train" for row in dataset_rows),
            "supervision_fields_readable": all(row["prompt"] and row["response"] for row in dataset_rows),
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise AssertionError(f"框架数据接口核对失败：{failed_checks}")
        return {
            "status": "passed",
            "check_kind": "data_interface",
            "versions": {"datasets": version("datasets")},
            "checks": checks,
            "admitted_rows": len(dataset),
            "rejected_rows": len(rejected_ids),
        }

    framework_check = check_sft_data_interface()

framework_check
```

## 相关阅读

完成样本身份、重复和训练—评测泄漏审计后，可以继续检查长度处理与真实训练交付。

- [Hugging Face Datasets 官方文档](https://huggingface.co/docs/datasets)
- [30. Long-Context Fine-Tuning Mechanics | 长上下文微调机制](./30_Long_Context_Fine_Tuning.md)
- [62. Instruction Fine-Tuning Project | 指令微调项目](./62_Instruction_Fine_Tuning_Project.md)
- [64. SFT Data Quality Project | SFT 数据质量项目](./64_SFT_Data_Quality_Project.md)
