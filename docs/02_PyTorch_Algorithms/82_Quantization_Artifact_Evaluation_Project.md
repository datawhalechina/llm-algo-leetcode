# 82. Quantization Artifact Evaluation Project | 量化产物评估项目

**难度：** Hard | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `量化产物`、`质量核对`、`加载验证` | **目标人群：** 理解量化产物与质量指标的读者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

复制一份量化权重目录后，目标后端可能找不到分词器，也可能无法识别压缩配置。即使服务能返回文本，仍要知道质量记录对应哪份权重、用过哪些数据，以及文件传输后是否完整。

把模型版本、量化配置、数据和文件清单放在一起，先核对记录与实际文件，再尝试加载和生成。通过后得到的是可以继续做浮点—量化部署对照的候选，速度与服务容量由后续请求实验测量。

<a id="runtime-environment"></a>

## 运行环境

- **CPU 练习：** 使用 [`BASE-CPU`](../guide.md#base-cpu)，核对候选声明、质量方向与准入状态。
- **真实加载：** 使用 [`INF-VLLM-011-CU128`](../topic_discussion/quantization_compression/quickstart.md#q-vllm-quickstart)，需要单张兼容的 NVIDIA GPU；输入来自40节生成的本地候选包。
- **结果与交接：** [Step 5](#gpu-experiment)按 prepare、measure、close 执行；默认 prepare 不启动模型或后端。

## 前置阅读

先了解量化候选怎样生成，再检查它是否具备进入部署对照的证据。

- [40 GPTQ / AWQ 权重量化](40_GPTQ_and_AWQ_Weight_Quantization.md)：生成本项目读取的标准量化候选包，并记录模型、数据、质量和文件身份。

## Step 1：怎样确定正在检查哪份量化产物

量化产物通常包含权重、压缩配置、模型配置与分词器文件。候选包是这些文件及其来源记录的集合，用产物编号、基座版本和内容摘要确定身份。内容摘要由文件字节计算；复制后重新计算并比较，可以发现文件缺失、替换或损坏。

同一候选还要说明量化工具、位宽和分组方式，以及质量评测的输入与规则。这些信息让接收方能找到同一份文件、理解格式，并重查已有质量判断。

| 候选记录 | 保存什么 | 接收方用来检查什么 |
| --- | --- | --- |
| 产物身份 | 产物编号、基座模型与版本 | 权重是否来自预期模型 |
| 量化配置 | 方法、格式、位宽、分组和工具版本 | 目标加载器能否理解配置 |
| 数据来源 | 数据版本、划分和抽样记录 | 质量评测使用了哪些输入 |
| 文件清单 | 文件角色、大小和 SHA256 摘要 | 实际文件是否完整、一致 |
| 质量结果 | 指标方向、基线、候选值和阈值 | 按同一规则重算是否通过 |

![量化产物检查后再进入部署比较](../public/02_PyTorch_Algorithms/82_quant_artifact_evidence_flow.svg)

## Step 2：文件核对、质量记录与加载各回答什么

文件核对确定检查的是同一份产物；质量记录说明它在指定样本和门槛下的表现；真实加载说明目标后端能读取它并完成一次往返生成。往返生成是发送输入并收到非空输出，任务是否答对还需对应的评测规则。

上游验证集参与过候选选择，本项目复核这份选择记录。校准与验证使用不同划分和抽样记录，仍需在关心数据重叠时比较样本集合或文本内容；两个摘要不同仅说明记录不同。更严格的最终质量评估使用未参与选择的测试集。

| 检查层 | 实际输入 | 通过条件 | 所支持的判断 |
| --- | --- | --- | --- |
| 文件与配置 | 身份、文件角色、大小及摘要 | 记录完整，实际文件与清单一致 | 产物可找到并核对 |
| 数据与质量 | 版本、划分、样本、指标方向与数值 | 记录可追溯，有限数值按方向达标 | 既定选择协议下质量合格 |
| 真实加载 | 后端版本、加载状态、生成与失败 | 加载和往返完成 | 可以继续做部署对照 |

![同一产物的三类检查结果](../public/02_PyTorch_Algorithms/82_quant_artifact_evidence_layers.svg)

## Step 3：检查结果怎样决定下一动作

先处理文件、质量或加载失败，再判断是否缺少测量。尚未加载时保留待验证；启用显存预算后，需要相同计量范围的有效记录，缺测与超预算分别处理。全部通过表示产物可进入部署对照，接下来用实际请求比较低精度执行与资源表现。

| 当前结果 | 项目状态 | 下一动作 |
| --- | --- | --- |
| 文件、质量或加载失败 | 拒绝（`reject`） | 修复文件、量化配置或运行问题后重测 |
| 尚无真实加载记录 | 待验证（`pending`） | 执行目标后端的加载与生成 |
| 已启用预算但缺少有效显存记录 | 待验证（`pending`） | 按预算计量范围补采 |
| 加载完成但有效显存超预算 | 调整（`tune`） | 调整参数、产物或预算后复测 |
| 文件、质量、加载和已启用预算通过 | 接受（`accept`） | 生成产物引用，进入部署对照 |

## Step 4：实现候选检查与加载后的决策

本题提供候选声明、数据与质量检查事实、加载结果及可选预算，骨架负责字段提取和问题列表。请组合候选完整性、数据记录与质量条件，再区分未加载、加载失败、缺少显存测量和超预算等状态。磁盘文件内容校验由真实项目流程执行，CPU 样例检查的是声明与判断规则。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
| --- | --- | --- | --- |
| TODO 1：组合候选完整性 | 身份、量化配置和文件角色检查 | 三项均通过 | 缺少分词器或摘要时失败 |
| TODO 2：组合数据与质量条件 | 登记划分、样本摘要和指标方向 | 来源条件与质量同时通过 | 相同划分／摘要、质量失败和非法数值 |
| TODO 3：判断加载与预算状态 | 两项检查、加载结果与可选预算 | 分开处理失败、缺测、超预算和通过 | 四种状态及缺测／边界情况 |


```python
from typing import Any, Dict
from tools.quantization_project_runtime import (
    candidate_contract_facts, candidate_evidence_facts, artifact_loader_facts,
)

```


```python
def validate_candidate_contract(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """TODO 1：组合身份、量化配置和文件角色门槛；不访问真实文件。"""
    facts = candidate_contract_facts(candidate)
    identity_ready, quantization_ready, files_ready = (facts[key] for key in ('identity_ready', 'quantization_ready', 'files_ready'))
    # TODO 1：组合静态门槛。
    # contract_ready = ???  # 三类声明都满足，不能互相替代。
    raise NotImplementedError('请完成 TODO 1：contract_ready')
    return {'ready': contract_ready, **facts}


def evaluate_candidate_evidence(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """TODO 2：按指标方向重算质量，并核对登记的数据划分。"""
    facts = candidate_evidence_facts(candidate)
    split_isolated = facts['split_isolated']
    sample_hashes_distinct = facts['sample_hashes_distinct']
    direction, ratio, limit = facts['direction'], facts['observed_ratio'], facts['quality_limit']
    # TODO 2：质量数值先通过有限性和一致性检查，再按指标方向比较。
    # quality_ready = ???   # 声明通过、数值有效，且比率满足所声明方向的门槛。
    # evidence_ready = ???  # 登记划分、样本摘要与质量判断同时成立。
    raise NotImplementedError('请完成 TODO 2：quality_ready、evidence_ready')
    issues = [name for name, ready in (('split_overlap', split_isolated), ('sample_identity_missing_or_equal', sample_hashes_distinct), ('quality', quality_ready)) if not ready]
    return {'ready': evidence_ready, **facts, 'quality_ready': quality_ready, 'issues': issues}


def decide_artifact_adoption(contract_check: Dict[str, Any], evidence_check: Dict[str, Any], loader_result: Dict[str, Any] | None, *, max_memory_mb: float = 0.0) -> Dict[str, str]:
    """TODO 3：组合真实加载和预算，缺测与超预算形成不同状态。"""
    if contract_check.get('ready') is not True or evidence_check.get('ready') is not True:
        issues = [*contract_check.get('issues', []), *evidence_check.get('issues', [])]
        return {'decision': 'reject', 'reason': '; '.join(issues) or '静态或质量门禁未通过'}
    if loader_result is None:
        return {'decision': 'pending', 'reason': 'CPU 门禁已通过，尚未执行真实加载'}
    facts = artifact_loader_facts(loader_result, max_memory_mb)
    # TODO 3：组合加载与资源判断，缺测分支由骨架保留为 pending。
    # loader_ready = ???  # 加载完成，且全部固定提示都返回非空生成。
    # memory_ready = ???  # 预算合法，且未启用上限或有效记录没有超预算。
    raise NotImplementedError('请完成 TODO 3：loader_ready、memory_ready')
    if not loader_ready:
        return {'decision': 'reject', 'reason': '真实加载与往返生成未完成'}
    if not facts['budget_valid'] or (not facts['memory_missing'] and not facts['memory_valid']):
        return {'decision': 'reject', 'reason': '显存预算或测量数值无效'}
    if max_memory_mb > 0 and facts['memory_missing']:
        return {'decision': 'pending', 'reason': '已启用预算但尚无有效显存证据'}
    if not memory_ready:
        return {'decision': 'tune', 'reason': '加载成功，但显存记录超过预算'}
    return {'decision': 'accept', 'reason': '候选可进入浮点—量化部署基准实验'}

```

## 测试

核对候选声明、数据与质量条件，以及加载和预算形成的状态。


```python
def _candidate_fixture() -> Dict[str, Any]:
    """构造三个测试共同使用的完整候选包。"""
    return {
        "schema_version": "quantization-artifact-candidate/v1",
        "artifact_id": "toy-gptq-v1",
        "model": {"id": "toy/base", "revision": "abc123"},
        "quantization": {"method": "gptq", "format": "pack-quantized", "bits": 4, "group_size": 128},
        "calibration": {"dataset": "toy/data", "revision": "data-v1", "split": "train", "sample_ids_hash": "cal-hash"},
        "evaluation": {
            "dataset": "toy/data", "revision": "data-v1",
            "split": "validation",
            "sample_ids_hash": "eval-hash",
            "direction": "lower_is_better",
            "baseline_value": 10.0,
            "candidate_value": 10.5,
            "max_degradation_ratio": 1.10,
            "passed": True,
        },
        "files": {"weights": ["model.safetensors"], "config": "config.json", "tokenizer": ["tokenizer.json"], "inventory": [{"path": name, "sha256": "a" * 64} for name in ("model.safetensors", "config.json", "tokenizer.json")]},
    }


def test_candidate_contract() -> None:
    """验证 TODO 1：完整身份与文件角色通过，缺少分词器时失败。"""
    candidate = _candidate_fixture()
    assert validate_candidate_contract(candidate)["ready"]
    incomplete = {**candidate, "files": {**candidate["files"], "tokenizer": []}}
    result = validate_candidate_contract(incomplete)
    assert not result["ready"] and "files" in result["issues"]
    bad_inventory = {**candidate, "files": {**candidate["files"], "inventory": [{"path": "model.safetensors"}]}}
    assert not validate_candidate_contract(bad_inventory)["ready"]


def test_candidate_evidence() -> None:
    """验证 TODO 2：数据隔离和带方向质量证据必须同时成立。"""
    candidate = _candidate_fixture()
    assert evaluate_candidate_evidence(candidate)["ready"]
    overlapping = {**candidate, "evaluation": {**candidate["evaluation"], "split": "train", "sample_ids_hash": "cal-hash"}}
    assert not evaluate_candidate_evidence(overlapping)["ready"]
    failed_quality = {
        **candidate,
        "evaluation": {**candidate["evaluation"], "candidate_value": 12.0},
    }
    assert not evaluate_candidate_evidence(failed_quality)["ready"]
    higher_is_better = {
        **candidate,
        "evaluation": {
            **candidate["evaluation"],
            "direction": "higher_is_better",
            "candidate_value": 9.6,
            "min_retention_ratio": 0.95,
        },
    }
    assert evaluate_candidate_evidence(higher_is_better)["ready"]


def test_adoption_decision() -> None:
    """验证 TODO 3：未执行、加载失败、超预算和通过形成不同状态。"""
    ready = {"ready": True, "issues": []}
    loader = {
        "load_status": "loaded",
        "roundtrip_quality": {"non_empty_generations": 2, "prompt_count": 2},
        "peak_memory_mb": 1200.0,
        "failure": None,
    }
    assert decide_artifact_adoption(ready, ready, None)["decision"] == "pending"
    assert decide_artifact_adoption(ready, ready, {**loader, "load_status": "failed"})["decision"] == "reject"
    assert decide_artifact_adoption(ready, ready, loader, max_memory_mb=1000)["decision"] == "tune"
    assert decide_artifact_adoption(ready, ready, loader, max_memory_mb=1500)["decision"] == "accept"
    assert decide_artifact_adoption(ready, ready, loader, max_memory_mb=0)["decision"] == "accept"


def test_invalid_numbers_roles_and_evidence_scope() -> None:
    """缺少角色清单、非整数配置与非有限质量不能进入真实加载。"""
    candidate = _candidate_fixture()
    for bad_bits in (True, 4.5, '4'):
        assert not validate_candidate_contract({**candidate, 'quantization': {**candidate['quantization'], 'bits': bad_bits}})['ready']
    missing = {**candidate, 'files': {**candidate['files'], 'inventory': candidate['files']['inventory'][:1]}}
    assert not validate_candidate_contract(missing)['ready']
    for value in (float('nan'), float('inf'), True):
        evidence = {**candidate, 'evaluation': {**candidate['evaluation'], 'direction': 'higher_is_better', 'candidate_value': value, 'min_retention_ratio': .95}}
        assert not evaluate_candidate_evidence(evidence)['ready']
    zero = {**candidate, 'evaluation': {**candidate['evaluation'], 'candidate_value': 0.0}}
    assert evaluate_candidate_evidence(zero)['ready']
    stale_ratio = {**candidate, 'evaluation': {**candidate['evaluation'], 'ratio_vs_baseline': .5}}
    assert not evaluate_candidate_evidence(stale_ratio)['ready']
    checked = evaluate_candidate_evidence(candidate)
    assert checked['sample_hashes_distinct'] and checked['sample_disjointness_proven'] is False
    foreign = {**candidate, 'evaluation': {**candidate['evaluation'], 'revision': 'other'}}
    assert not evaluate_candidate_evidence(foreign)['ready']


def test_missing_memory_and_invalid_roundtrip() -> None:
    """缺测不伪装超预算，布尔计数和非有限预算不能通过。"""
    ready = {'ready': True, 'issues': []}
    loader = {'load_status': 'loaded', 'roundtrip_quality': {'prompt_count': 2, 'non_empty_generations': 2}, 'peak_memory_mb': None, 'failure': None}
    assert decide_artifact_adoption(ready, ready, loader, max_memory_mb=1000)['decision'] == 'pending'
    assert decide_artifact_adoption(ready, ready, loader, max_memory_mb=float('nan'))['decision'] == 'reject'
    invalid = {**loader, 'roundtrip_quality': {'prompt_count': True, 'non_empty_generations': True}}
    assert decide_artifact_adoption(ready, ready, invalid)['decision'] == 'reject'


test_candidate_contract()
test_candidate_evidence()
test_adoption_decision()
test_invalid_numbers_roles_and_evidence_scope()
test_missing_memory_and_invalid_roundtrip()
print("测试通过：候选契约、数据与质量证据、真实加载决策相互一致。")

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难，可以继续向下查看参考代码与解析。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
def validate_candidate_contract(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """TODO 1：组合身份、量化配置和文件角色门槛；不访问真实文件。"""
    facts = candidate_contract_facts(candidate)
    identity_ready, quantization_ready, files_ready = (facts[key] for key in ('identity_ready', 'quantization_ready', 'files_ready'))
    contract_ready = identity_ready and quantization_ready and files_ready
    return {'ready': contract_ready, **facts}


def evaluate_candidate_evidence(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """TODO 2：按指标方向重算质量，并核对登记的数据划分。"""
    facts = candidate_evidence_facts(candidate)
    split_isolated = facts['split_isolated']
    sample_hashes_distinct = facts['sample_hashes_distinct']
    direction, ratio, limit = facts['direction'], facts['observed_ratio'], facts['quality_limit']
    quality_ready = bool(facts['quality_inputs_ready'] and (
        (direction == 'lower_is_better' and ratio <= limit)
        or (direction == 'higher_is_better' and ratio >= limit)
    ))
    evidence_ready = split_isolated and sample_hashes_distinct and quality_ready
    issues = [name for name, ready in (('split_overlap', split_isolated), ('sample_identity_missing_or_equal', sample_hashes_distinct), ('quality', quality_ready)) if not ready]
    return {'ready': evidence_ready, **facts, 'quality_ready': quality_ready, 'issues': issues}


def decide_artifact_adoption(contract_check: Dict[str, Any], evidence_check: Dict[str, Any], loader_result: Dict[str, Any] | None, *, max_memory_mb: float = 0.0) -> Dict[str, str]:
    """TODO 3：组合真实加载和预算，缺测与超预算形成不同状态。"""
    if contract_check.get('ready') is not True or evidence_check.get('ready') is not True:
        issues = [*contract_check.get('issues', []), *evidence_check.get('issues', [])]
        return {'decision': 'reject', 'reason': '; '.join(issues) or '静态或质量门禁未通过'}
    if loader_result is None:
        return {'decision': 'pending', 'reason': 'CPU 门禁已通过，尚未执行真实加载'}
    facts = artifact_loader_facts(loader_result, max_memory_mb)
    loader_ready = facts['load_complete'] and facts['roundtrip_complete']
    memory_ready = facts['budget_valid'] and (max_memory_mb == 0 or facts['memory_within_budget'])
    if not loader_ready:
        return {'decision': 'reject', 'reason': '真实加载与往返生成未完成'}
    if not facts['budget_valid'] or (not facts['memory_missing'] and not facts['memory_valid']):
        return {'decision': 'reject', 'reason': '显存预算或测量数值无效'}
    if max_memory_mb > 0 and facts['memory_missing']:
        return {'decision': 'pending', 'reason': '已启用预算但尚无有效显存证据'}
    if not memory_ready:
        return {'decision': 'tune', 'reason': '加载成功，但显存记录超过预算'}
    return {'decision': 'accept', 'reason': '候选可进入浮点—量化部署基准实验'}

```

### 解析

#### TODO 1：组合候选完整性

- 身份、量化配置与文件角色各提供一个必要条件，任一项不满足都会阻止继续加载。
- 答案对三个状态取合取。测试删除分词器角色或摘要，核对声明缺失；实际磁盘哈希由项目流程另行比较。

#### TODO 2：组合数据与质量条件

- 登记划分和样本摘要用于追溯数据，质量按指标方向重算比值；满足这些检查才能复用已有候选选择证据。
- 答案合并数据条件与质量结果。测试制造相同划分、摘要及质量失败，核对失败项；集合交集检查需要实际样本而非摘要比较。

#### TODO 3：判断加载与预算状态

- 未加载或启用预算后缺测时待验证；加载失败及非法数值拒绝，有效记录超过预算时调整。
- 答案先查加载与往返，再检查预算及测量。测试核对四种状态，保留缺测和超限的不同原因。

<a id="gpu-experiment"></a>

## Step 5：验证量化产物并生成部署准入记录

使用40节获选候选的同一份文件和质量记录，先检查内容身份，再执行vLLM最小往返生成。`prepare`只读预检，`measure`保存加载与原始准入证据，`close`重读证据并生成报告、登记回执；只有接受的报告才交给部署对照。

| CPU 练习 | 真实环境补充 | 项目输出 |
|---|---|---|
| 组合候选声明、质量方向与准入状态 | 磁盘内容哈希、固定vLLM调用与非空生成 | 原始加载记录、准入报告、登记回执；接受时生成稳定引用 |

### 5.1 本次运行配置

真实加载使用 [`INF-VLLM-011-CU128`](../topic_discussion/quantization_compression/quickstart.md#q-vllm-quickstart)。准备40节的获选候选及对应结果，再填写唯一上游标签，按 `prepare → measure → close` 分阶段顺序运行本节；`close`无需重新启动GPU或后端。


```python
RUN_LABEL = 'replace_with_unique_label'  # 本次82节标签；measure/close前填写，已有不同内容时不得覆盖。
RUN_STAGE = 'prepare'  # prepare只读；measure执行或恢复采集；close校验并补齐报告与交接。
UPSTREAM_RUN_LABEL = 'replace_with_upstream_label'  # 40节已完成real_gpu并接受候选的RUN_LABEL；只配置这一份直接来源。

```

### 5.2 固定合同、来源与环境预检

从公共运行合同解析路径和固定加载参数，检查40节的获选身份、数据与质量声明、候选包和实际文件哈希。来源缺失或JSON损坏时输出具体原因。只有新`measure`才执行环境GPU预检；`prepare`和`close`不初始化设备。


```python
import importlib
import json
import subprocess
import sys

from tools.project_runtime import resolve_project_root
import tools.quantization_project_runtime as quantization_project

# 集中刷新公共模块；后续从模块对象取函数，避免热Kernel继续使用旧绑定。
quantization_project = importlib.reload(quantization_project)
plan = quantization_project.artifact_gate_plan(
    resolve_project_root(), RUN_LABEL, UPSTREAM_RUN_LABEL, RUN_STAGE,
)
prepared = quantization_project.prepare_artifact_gate(plan)
profile_report = {'status': 'not_requested', 'ready': None}
gpu_preflight_executed = False
preflight_issues = list(prepared['issues'])
if RUN_STAGE == 'measure' and prepared['ready'] and not plan['loader_path'].exists():
    gpu_preflight_executed = True
    try:
        process = subprocess.run(
            [sys.executable, str(plan['root'] / 'tools/environment_preflight.py'),
             '--profile', plan['contract']['environment_profile']],
            cwd=plan['root'], capture_output=True, text=True, timeout=60,
        )
        profile_report = json.loads(process.stdout)
        if not isinstance(profile_report, dict):
            raise ValueError('环境预检输出必须是JSON对象')
        if process.returncode != 0 or profile_report.get('ready') is not True:
            preflight_issues.extend(profile_report.get('reasons') or ['环境Profile未通过'])
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        preflight_issues.append(f'环境预检失败：{error}')
preflight = {'ready': not preflight_issues, 'issues': preflight_issues,
             'profile': plan['contract']['environment_profile'],
             'gpu_preflight_executed': gpu_preflight_executed,
             'inputs': prepared['snapshot']}
print(json.dumps(preflight, ensure_ascii=False, indent=2))

```

### 5.3 候选与质量门槛

复核登记的数据快照、不同划分与样本摘要，再按指标方向重算质量。40节验证集已参与候选选择，这里不把它当作新的独立任务测试；真实加载前的判断保持`pending`。


```python
if prepared['ready']:
    candidate_contract = quantization_project.validate_candidate_contract(prepared['bundle'])
    candidate_evidence = quantization_project.evaluate_candidate_evidence(prepared['bundle'])
    cpu_decision = quantization_project.decide_artifact_adoption(candidate_contract, candidate_evidence, None)
else:
    candidate_contract = {'ready': False, 'issues': prepared['issues']}
    candidate_evidence = {'ready': False, 'issues': prepared['issues']}
    cpu_decision = {'decision': 'pending', 'reason': '先按5.2的问题修复来源'}
print({'contract': candidate_contract, 'evidence': candidate_evidence, 'decision': cpu_decision})

```

### 5.4 执行或恢复加载采集

`measure`执行固定vLLM调用，只保存加载记录和原始准入证据。已有同身份加载记录时复用，不再次启动后端；若在生成原始准入记录前中断，可用相同配置继续。兼容字段`peak_memory_mb`是设备空闲显存差，不是过程峰值，当前合同不启用峰值预算。


```python
if RUN_STAGE == 'measure':
    if not preflight['ready']:
        raise RuntimeError('预检未通过：' + '; '.join(preflight['issues']))
    if plan['loader_path'].exists():
        loader_result = json.loads(plan['loader_path'].read_text(encoding='utf-8'))
    else:
        invocation = quantization_project.artifact_loader_invocation(plan, prepared)
        command = [
            sys.executable, str(plan['root'] / 'tools/validate_quantized_artifact.py'),
            '--artifact', prepared['snapshot']['artifact_path'], '--output', str(plan['loader_path']),
            '--invocation-json', json.dumps(invocation, ensure_ascii=False),
            '--max-model-len', str(plan['contract']['max_model_len']),
            '--max-tokens', str(plan['contract']['max_tokens']),
            '--gpu-memory-utilization', str(plan['contract']['gpu_memory_utilization']),
        ]
        completed = subprocess.run(command, cwd=plan['root'], capture_output=True, text=True)
        if plan['loader_path'].is_file():
            loader_result = json.loads(plan['loader_path'].read_text(encoding='utf-8'))
        else:
            loader_result = {
                'schema_version': 'quantization-artifact-loader/v2', 'backend': 'vllm',
                'invocation': invocation, 'status': 'failed', 'load_status': 'failed',
                'failure': {'stage': 'loader_process', 'returncode': completed.returncode,
                            'reason': completed.stderr[-2000:] or '后端未生成完整加载记录'},
            }
    measurement = quantization_project.save_artifact_gate_measurement(plan, prepared, loader_result)
    print(measurement)
else:
    print({'status': 'skipped', 'stage': RUN_STAGE, 'reason': '本阶段不执行真实加载'})

```

### 5.5 只读核对已保存证据

按当前标签重新定位记录，核对上游与候选包SHA256、实际文件、固定调用、加载记录、质量判断和准入结论；已有报告、交接与回执也须一致。读取不重建候选、不改写文件；缺少测量时显示待采集，身份冲突则停止。


```python
if prepared['ready']:
    checked = quantization_project.read_artifact_gate(plan)
else:
    checked = {'status': 'blocked', 'decision': {'decision': 'pending'}, 'issues': prepared['issues']}
print({'status': checked['status'], 'decision': checked['decision'],
       'evidence': checked.get('summary'), 'issues': checked.get('issues', []),
       'next_action': checked.get('next_action')})

```

### 5.6 形成结论、登记与交接

`close`重新校验磁盘证据，再生成准入报告和登记回执；`accept`才生成供67消费的稳定引用。同标签内容一致时只读复用，中断后补齐缺失资产；已有内容不一致时停止，不覆盖历史记录。

| 已核对的证据 | 决策 | 下一动作 |
|---|---|---|
| 尚无完整测量 | pending | 执行measure；若已有加载记录则复用恢复 |
| 契约、质量或真实加载未通过 | reject | 保留失败报告与回执，修复后使用新标签 |
| 加载完成但有效资源记录超过门槛 | tune | 检查资源条件；当前合同未启用峰值预算 |
| 契约、质量与真实加载全部通过 | accept | 生成稳定引用，进入浮点／量化部署对照 |

**当前登记结果**：尚无本项目真实加载结果。后续按固定模型与候选包采集加载记录，再登记标签与决策；当前状态见[实验资产表](../topic_discussion/quantization_compression/experiment_assets.md#项目验证顺序)。


```python
if RUN_STAGE == 'close':
    registration = quantization_project.close_artifact_gate(plan)
    print(registration)
else:
    print({'status': 'skipped', 'stage': RUN_STAGE, 'reason': '只有close生成最终报告、交接和登记回执'})

```

## 相关阅读

检查通过后进入实际请求比较；遇到格式兼容问题时对照目标后端支持文档。

- [量化推理与部署](./67_Quantized_Inference_and_Deployment.md)：比较同源浮点与量化服务。
- [vLLM 量化文档](https://docs.vllm.ai/en/latest/features/quantization/)：核对所选版本的格式与硬件支持。
