# 62. Instruction Fine Tuning Project | SFT 训练交付项目

**难度：** Hard | **环境：** CPU + 单卡 GPU | **标签：** `监督微调`, `断点恢复`, `adapter 交付` | **目标人群：** 已完成数据与配置选型的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节读取 64 节的数据准入结果和 63 节的 LoRA 选型方案，按固定配置完成一次可恢复训练。训练交付包括检查点（checkpoint）、最终适配器（adapter）、分词器（tokenizer）、重载评测和不可变产物清单（manifest），并作为 DPO 或 GRPO 的基线。

**关键词：** `SFT`, `检查点恢复`, `适配器`, `产物清单`

---
## 前置阅读

**导语：** 先完成数据准入和更新方案选型，再进入固定配置的训练交付。

- [Part 02 · 15 端到端微调实验](./15_End_to_End_Fine_Tuning_Experiment.md)
- [Part 02 · 64 SFT 数据质量项目](./64_SFT_Data_Quality_Project.md)
- [Part 02 · 63 LoRA 变体选型项目](./63_LoRA_Variants_Benchmark.md)
- [Part 02 · 65 QLoRA 低资源选型项目](./65_QLoRA_Selection_Project.md)

---
### Step 1：从选定方案走到可交付训练产物

通过准入的数据快照和已选定的 LoRA 配置共同构成训练输入。模型版本、数据身份、更新配置和训练预算固定后，执行可恢复、可重载、可独立评测的 SFT 训练，并保存适配器、分词器、检查点、更新动力学摘要、评测结果和产物清单。

| 阶段 | 输入 | 关键问题 | 输出 |
|:---|:---|:---|:---|
| 启动检查 | 数据准入与方案选型结果 | 上游证据是否完整且相互一致 | 固定训练身份 |
| 训练与保存 | 固定配置和目标步数 | 梯度、参数增量和有限值是否可解释；中断前是否保存了继续训练所需状态 | 阶段检查点与更新摘要 |
| 恢复与完成 | 检查点和剩余训练预算 | 是否从正确步数继续且达到目标步数 | 最终候选 |
| 重载与交付 | 保存产物和独立评测集 | 离开训练进程后能否复现质量 | SFT 基线产物清单 |

![指令微调的交付判断链](../public/02_PyTorch_Algorithms/62_instruction_finetuning_flow.svg)

### Step 2：把恢复训练视为同一次训练的延续

检查点只有在训练身份、可训练状态、调度器与随机数状态、以及进度连续性同时成立时才能恢复。本项目保存适配器、优化器、调度器、随机数和训练进度；恢复前核对模型、数据与更新配置，恢复后核对首个新步是否紧接保存步。身份变化时启动新运行；状态缺失或步数断裂时修复检查点，防止把两次不同实验记录为同一次训练。

| 恢复条件 | 使用的证据 | 通过条件 | 失败动作 |
|:---|:---|:---|:---|
| 训练身份 | 模型版本、数据哈希和更新配置哈希 | 三项均与当前训练一致 | 启动新运行，不复用旧优化器状态 |
| 可训练状态 | 适配器、优化器、调度器与随机数状态 | 四类状态都存在且可读取 | 修复或重新生成检查点 |
| 保存进度 | 已保存步数与目标步数 | 保存步位于起点和目标步之间 | 拒绝无效检查点 |
| 恢复连续性 | 保存步与恢复后的首个新步 | 首个新步紧接保存步 | 停止恢复并检查进度记录 |
| 预算完成 | 最终步数与原定目标步数 | 恢复后达到同一训练预算 | 继续训练，不进入交付 |

### Step 3：让独立重载决定产物能否交付

训练进程中的损失下降只能证明参数发生了更新。交付前还要确认更新过程始终有限，且梯度范数和参数增量已随结果保存；这些量只能解释训练过程，不替代独立质量评测或通用能力保留判断。训练结束后，在新模型实例中重载基础模型与适配器，并在同一独立评测集上比较保存前后的结果。交付门槛采用合取关系：任一门槛失败都进入 `reject`（拒绝）并记录失败项；全部通过才进入 `accept`（接受），随后冻结 SFT 基线产物清单。

| 交付门槛 | 检查证据 | 通过条件 | 失败后的处理 |
|:---|:---|:---|:---|
| 执行状态 | 训练运行状态 | 运行正常结束且没有未处理失败 | `reject`，修复执行失败后重跑 |
| 恢复位置 | 保存步与恢复起点 | 恢复起点等于保存步，后续更新连续 | `reject`，检查进度与恢复入口 |
| 训练预算 | 最终步数与目标步数 | 完成固定更新预算 | `reject`，继续训练至目标步数 |
| 更新过程 | 有限值、梯度范数与参数增量摘要 | 全部更新有限且摘要完整 | `reject`，先定位数值或更新异常 |
| 独立质量 | 更新前后独立评测损失 | 退化不超过预设上限 | `reject`，调整数据、配置或训练预算 |
| 重载一致性 | 保存前结果与新实例重载结果的绝对差 | 偏差不超过序列化容差 | `reject`，检查保存、加载与数值精度 |
| 产物完整性 | 适配器、分词器、检查点和目录哈希 | 必需产物存在且具有稳定身份 | `reject`，补齐产物后重新核验 |

![SFT 交付的恢复与重载检查](../public/02_PyTorch_Algorithms/62_instruction_cpu_todo_flow.svg)

### Step 4：实现选型清单检查、连续恢复与交付判断

CPU 题目依次检查更新方案清单、检查点摘要和独立评测结果。学习者需要判断复测记录是否属于同一候选且随机种子互不重复，检查训练身份与恢复步数是否连续，并在全部交付门槛通过后生成接受决定。测试覆盖候选混入、重复种子、数据漂移、状态缺失、恢复错位、质量退化和重载偏差。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：验证选型结果能否作为训练输入 | 唯一配置、配对复测和适配器信息 | 复测属于同一候选且随机种子不重复 | 状态失败、候选混入、复测不足与字段缺失 |
| TODO 2：验证恢复确实延续原训练 | 训练身份、状态文件、保存步与恢复首步 | 身份一致、状态齐全且步数连续 | 数据漂移、缺优化器、重新从零开始 |
| TODO 3：判断重载产物能否交付 | 初始/最终质量、目标步数、重载和哈希 | 完成预算且质量不退化，重载结果一致 | 步数不足、质量退化、重载偏差与产物缺失 |

```python
from typing import Any, Dict, List
```


```python
# ==========================================
# TODO 1：校验上游选型清单
# ==========================================
def validate_selection_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """检查已接受的唯一配置、复测证据和 artifact 是否完整。"""
    required_config = {"name", "rank", "alpha", "dropout", "target_modules"}
    selected_config = manifest.get("selected_config", {})
    repeat_results = manifest.get("repeat_results", [])
    selected_name = selected_config.get("name")
    repeat_seeds = [item.get("seed") for item in repeat_results]
    checks = {
        "accepted": manifest.get("decision") == "accept",
        "config_complete": required_config.issubset(selected_config),
        "repeat_complete": len(repeat_results) >= 2 and all(item.get("status") == "ok" for item in repeat_results),
        "same_candidate": False,
        "independent_seeds": False,
        "artifact_present": bool(manifest.get("artifact", {}).get("adapter_path")),
    }

    # checks["same_candidate"] = ???  # 每次复测必须属于 selected_config 指定的候选。
    # checks["independent_seeds"] = ???  # 随机种子必须存在且不能重复。
    # failed_checks, consumable = ???  # 保留失败原因；无失败项时才能进入训练。
    raise NotImplementedError("TODO 1：请验证候选身份、复测独立性与训练输入条件")
    return {"consumable": consumable, "failed_checks": failed_checks}


# ==========================================
# TODO 2：检查检查点恢复条件
# ==========================================
def check_resume_contract(checkpoint: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """检查训练身份、状态文件和恢复步数。"""
    identity_keys = ("model_commit", "data_hash", "config_hash")
    state_complete = all(checkpoint.get(key, False) for key in ("adapter_present", "optimizer_state_present", "scheduler_state_present", "rng_state_present"))
    progress_valid = 0 < checkpoint.get("global_step", -1) < expected["total_steps"]
    identity_matches = False
    continuation_valid = False

    # identity_matches = ???  # 模型、数据和配置身份必须全部与当前训练一致。
    # continuation_valid = ???  # 恢复后的第一步必须紧接检查点保存步。
    # resumable = ???  # 身份、状态、进度和步数连续性必须同时成立。
    raise NotImplementedError("TODO 2：请验证检查点身份与连续恢复关系")
    return {"resumable": resumable, "identity_matches": identity_matches, "state_complete": state_complete, "progress_valid": progress_valid, "continuation_valid": continuation_valid}


# ==========================================
# TODO 3：判断 SFT 产物能否交付
# ==========================================
def decide_sft_delivery(evidence: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """依据恢复、独立评测、重载一致性和产物哈希形成交付决策。"""
    checks = {
        "execution": evidence.get("status") == "ok",
        "resume": evidence.get("checkpoint", {}).get("saved_step") == evidence.get("checkpoint", {}).get("resumed_from_step"),
        "completed_budget": evidence["final_step"] == evidence["target_step"],
        "quality": evidence["final_eval_loss"] <= evidence["initial_eval_loss"] + gates["maximum_eval_loss_regression"],
        "reload": evidence["reload_loss_abs_diff"] <= gates["maximum_reload_loss_abs_diff"],
        "update_dynamics": evidence.get("update_dynamics", {}).get("finite_all") is True,
        "artifact": bool(evidence.get("artifact", {}).get("sha256")),
    }

    # 任一硬门槛失败都不能交付；返回失败字段供下一轮修复。
    # decision = ???  # 任一交付门槛失败都应拒绝。
    raise NotImplementedError("TODO 3：请形成 SFT 交付决策")
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}
```


```python
def test_selection_manifest_contract():
    """验证同一候选的独立复测和产物信息缺一不可。"""
    selection = {
        "decision": "accept",
        "selected_config": {"name": "rank8", "rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "repeat_results": [{"name": "rank8", "seed": 42, "status": "ok"}, {"name": "rank8", "seed": 43, "status": "ok"}],
        "artifact": {"adapter_path": "adapter"},
    }
    assert validate_selection_manifest(selection)["consumable"]
    assert not validate_selection_manifest({**selection, "repeat_results": [{"name": "rank8", "seed": 42, "status": "ok"}]})["consumable"]
    mixed = {**selection, "repeat_results": [{"name": "rank8", "seed": 42, "status": "ok"}, {"name": "rank16", "seed": 43, "status": "ok"}]}
    assert validate_selection_manifest(mixed)["failed_checks"] == ["same_candidate"]
    duplicated_seed = {**selection, "repeat_results": [{"name": "rank8", "seed": 42, "status": "ok"}, {"name": "rank8", "seed": 42, "status": "ok"}]}
    assert validate_selection_manifest(duplicated_seed)["failed_checks"] == ["independent_seeds"]


def test_resume_contract():
    """验证恢复身份、状态文件和训练步连续性。"""
    expected = {"model_commit": "m1", "data_hash": "d1", "config_hash": "c1", "total_steps": 20, "first_resumed_step": 11}
    checkpoint = {**expected, "adapter_present": True, "optimizer_state_present": True, "scheduler_state_present": True, "rng_state_present": True, "global_step": 10}
    assert check_resume_contract(checkpoint, expected)["resumable"]
    assert not check_resume_contract({**checkpoint, "data_hash": "d2"}, expected)["resumable"]
    assert not check_resume_contract({**checkpoint, "scheduler_state_present": False}, expected)["resumable"]
    assert not check_resume_contract(checkpoint, {**expected, "first_resumed_step": 12})["continuation_valid"]


def test_sft_delivery_decision():
    """验证预算、质量、重载和产物门槛采用合取关系。"""
    gates = {"maximum_eval_loss_regression": 0.0, "maximum_reload_loss_abs_diff": 1e-4}
    evidence = {
        "status": "ok", "initial_eval_loss": 1.9, "final_eval_loss": 1.8, "final_step": 20, "target_step": 20, "reload_loss_abs_diff": 1e-6,
        "checkpoint": {"saved_step": 10, "resumed_from_step": 10},
        "update_dynamics": {"finite_all": True},
        "artifact": {"sha256": "abc"},
    }
    assert decide_sft_delivery(evidence, gates)["decision"] == "accept"
    assert decide_sft_delivery({**evidence, "reload_loss_abs_diff": 0.01}, gates)["decision"] == "reject"
    assert decide_sft_delivery({**evidence, "artifact": {}}, gates)["decision"] == "reject"


test_selection_manifest_contract()
test_resume_contract()
test_sft_delivery_decision()
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
# TODO 1：校验上游选型清单
def validate_selection_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """检查已接受的唯一配置、复测证据和产物是否完整。"""
    required_config = {"name", "rank", "alpha", "dropout", "target_modules"}
    selected_config = manifest.get("selected_config", {})
    repeat_results = manifest.get("repeat_results", [])
    selected_name = selected_config.get("name")
    repeat_seeds = [item.get("seed") for item in repeat_results]
    checks = {
        "accepted": manifest.get("decision") == "accept",
        "config_complete": required_config.issubset(selected_config),
        "repeat_complete": len(repeat_results) >= 2 and all(item.get("status") == "ok" for item in repeat_results),
        "same_candidate": False,
        "independent_seeds": False,
        "artifact_present": bool(manifest.get("artifact", {}).get("adapter_path")),
    }
    checks["same_candidate"] = bool(selected_name) and all(item.get("name") == selected_name for item in repeat_results)
    checks["independent_seeds"] = None not in repeat_seeds and len(repeat_seeds) == len(set(repeat_seeds))
    failed_checks = [name for name, passed in checks.items() if not passed]
    consumable = not failed_checks
    return {"consumable": consumable, "failed_checks": failed_checks}


# TODO 2：检查检查点恢复条件
def check_resume_contract(checkpoint: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """检查训练身份、状态文件和恢复步数。"""
    identity_keys = ("model_commit", "data_hash", "config_hash")
    state_complete = all(checkpoint.get(key, False) for key in ("adapter_present", "optimizer_state_present", "scheduler_state_present", "rng_state_present"))
    progress_valid = 0 < checkpoint.get("global_step", -1) < expected["total_steps"]
    identity_matches = all(checkpoint.get(key) == expected.get(key) for key in identity_keys)
    continuation_valid = expected.get("first_resumed_step") == checkpoint.get("global_step", -1) + 1
    resumable = identity_matches and state_complete and progress_valid and continuation_valid
    return {"resumable": resumable, "identity_matches": identity_matches, "state_complete": state_complete, "progress_valid": progress_valid, "continuation_valid": continuation_valid}


# TODO 3：判断 SFT 产物能否交付
def decide_sft_delivery(evidence: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """依据恢复、独立评测、重载一致性和产物哈希形成交付决策。"""
    checks = {
        "execution": evidence.get("status") == "ok",
        "resume": evidence.get("checkpoint", {}).get("saved_step") == evidence.get("checkpoint", {}).get("resumed_from_step"),
        "completed_budget": evidence["final_step"] == evidence["target_step"],
        "quality": evidence["final_eval_loss"] <= evidence["initial_eval_loss"] + gates["maximum_eval_loss_regression"],
        "reload": evidence["reload_loss_abs_diff"] <= gates["maximum_reload_loss_abs_diff"],
        "update_dynamics": evidence.get("update_dynamics", {}).get("finite_all") is True,
        "artifact": bool(evidence.get("artifact", {}).get("sha256")),
    }
    decision = "accept" if all(checks.values()) else "reject"
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}
```

### 解析

#### TODO 1：校验上游选型

- **判断依据**：正式训练只接受一个已经通过选型的配置；复测记录必须属于同一候选并来自不同随机种子。
- **实现动作**：根据候选身份验证每次复测的归属，检查随机种子是否重复，再汇总是否可以进入训练及具体失败项。
- **测试证据**：测试覆盖复测不足、候选混入和随机种子重复，避免仅凭 `accept` 状态进入训练。

#### TODO 2：检查恢复契约

- **判断依据**：模型、数据或配置变化时，旧优化器状态不再属于当前训练；恢复首步还必须紧接保存步。
- **实现动作**：分别检查身份、状态文件、保存进度和恢复首步是否连续。
- **测试证据**：数据哈希漂移、缺失优化器状态和恢复首步错位都会阻止恢复。

#### TODO 3：判断交付

- **判断依据**：执行成功、训练预算完成、质量不退化、恢复位置一致、重载误差在容差内且产物有哈希，缺一不可。
- **实现动作**：保留每项硬门槛和失败字段，不使用综合分数。
- **测试证据**：测试覆盖正常交付、重载偏差和缺失产物。
## Step 5：真实 SFT 训练交付

CPU 题目区验证选型、恢复和交付条件；GPU 实验读取 63 节的 LoRA 选型清单，执行“训练到中点 → 保存恢复状态 → 重建模型、优化器与调度器 → 继续训练 → 保存并重载适配器”的完整训练。

| 对照层 | 输入 | 主要证据 | 输出与证据边界 |
|:---|:---|:---|:---|
| CPU 交付状态机 | 人工构造的选型、检查点与评测摘要 | 选型一致性、恢复连续性、采用状态 | 只确认交付逻辑，不证明真实训练完成 |
| 真实 SFT 交付 | 已接受的数据和 LoRA 选型清单 | 中断恢复、更新动力学、相对评测、重载一致性与产物哈希 | 输出 SFT 基线清单；不证明偏好、安全或通用能力保留 |
### 5.1 环境、输入资产与固定条件

按[后训练优化 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart)准备 `PT-SFT-CUDA` 环境。12 GB 单卡使用 Qwen2.5-0.5B、长度 256 和 LoRA 适配器；已登记实验的峰值分配约为 1.77 GiB。

```python
from pathlib import Path
import json

from tools.project_runtime import resolve_project_root

RUN_MODE = "dry_run"  # 默认值 dry_run：只检查配置；改为 real_gpu 执行真实训练。
RUN_LABEL = "rtx5070ti_sft_delivery_v3"  # 新交付契约的推荐标签；保留 v2_1 作为历史证据。
PROFILE_ID = "PT-SFT-CUDA"
SEMANTIC_ID = "PT-SFT-DELIVERY"
SCHEMA_VERSION = "fine-tuning-project/v1"

PROJECT_ROOT = resolve_project_root()
SELECTION_MANIFEST = PROJECT_ROOT / "benchmarks/results/63_lora_variants/lora_variant_delivery_lora_variant_selection_v2.json"  # 63 节配对重复选型结果。
MODEL_ID = "Qwen/Qwen2.5-0.5B"  # 必须与选型清单一致。
MAX_LENGTH = 256  # 必须与数据审计和选型实验负载一致。
TRAIN_EXAMPLES = 48  # 默认训练样本数；增大会提高训练时间。
EVALUATION_EXAMPLES = 16  # 默认独立评测样本数；增大会提高评测时间。
TOTAL_STEPS = 24  # 完整交付预算，高于候选筛选预算。
CHECKPOINT_STEP = 12  # 在中点保存并强制恢复。
LEARNING_RATE = 2e-4  # 默认学习率；修改后必须使用新标签重新形成基线。
DTYPE = "bfloat16"  # 默认计算精度；可改为 float16，但需重新验证稳定性。
SEED = 42  # 默认随机种子；用于复现训练与评测。
GATES = {"maximum_eval_loss": 2.0, "maximum_eval_loss_regression": 0.0, "maximum_reload_loss_abs_diff": 1e-4, "require_finite_updates": True}

RESULTS_DIR = PROJECT_ROOT / "benchmarks/results/62_sft_delivery"
EVIDENCE_PATH = RESULTS_DIR / f"sft_delivery_evidence_{RUN_LABEL}.json"
MANIFEST_PATH = RESULTS_DIR / f"sft_delivery_manifest_{RUN_LABEL}.json"
CHECKPOINT_DIR = RESULTS_DIR / f"checkpoint_{RUN_LABEL}"
ARTIFACT_DIR = RESULTS_DIR / f"adapter_{RUN_LABEL}"
print({"run_mode": RUN_MODE, "evidence_path": str(EVIDENCE_PATH), "total_steps": TOTAL_STEPS})
```

### 5.2 预检环境与选型清单

预检拒绝未接受、复测不足或缺少唯一 LoRA 配置的上游结果，并确认所有输出路径尚未存在。

```python
import torch
from tools.project_runtime import environment_preflight, require_input_file
from tools.post_training_project_runtime import run_lora_delivery, save_json_record

def validate_selection_manifest_for_run(manifest):
    required_config = {"name", "rank", "alpha", "dropout", "target_modules"}
    repeat_results = manifest.get("repeat_results", [])
    checks = {
        "accepted": manifest.get("decision") == "accept",
        "config_complete": required_config.issubset(manifest.get("selected_config", {})),
        "repeat_complete": len(repeat_results) >= 2 and all(item.get("status") == "ok" for item in repeat_results),
        "artifact_present": bool(manifest.get("artifact", {}).get("adapter_path")),
    }
    return {"consumable": all(checks.values()), "failed_checks": [name for name, passed in checks.items() if not passed]}

preflight = environment_preflight(
    torch, required_packages=("transformers", "peft", "datasets", "huggingface_hub"),
    require_gpu=RUN_MODE == "real_gpu", output_path=EVIDENCE_PATH,
    environment_profile=PROFILE_ID,
)
if not preflight["ready"]:
    raise RuntimeError(preflight["reasons"])
selection_path = require_input_file(SELECTION_MANIFEST, "63 节 LoRA 选型清单")
selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
selection_check = validate_selection_manifest_for_run(selection_manifest)
if not selection_check["consumable"]:
    raise RuntimeError(selection_check["failed_checks"])
for path in (EVIDENCE_PATH, MANIFEST_PATH, CHECKPOINT_DIR, ARTIFACT_DIR):
    if RUN_MODE == "real_gpu" and path.exists():
        raise FileExistsError(f"输出已存在，请修改 RUN_LABEL：{path}")
print({"preflight": preflight["status"], "selected": selection_manifest["selected_config"]["name"]})
```


```python
# 显式记录选型结果和训练身份，运行期间保持不变。
training_identity = {
    "selection_manifest": str(SELECTION_MANIFEST),
    "selected_config": selection_manifest["selected_config"],
    "total_steps": TOTAL_STEPS,
    "checkpoint_step": CHECKPOINT_STEP,
    "max_length": MAX_LENGTH,
    "learning_rate": LEARNING_RATE,
    "seed": SEED,
    "update_dynamics_fields": ["finite_all", "gradient_norm_median", "parameter_delta_norm_median", "learning_rate_start", "learning_rate_end"],
}
print(training_identity)
```

### 5.3 固定恢复与交付门槛

| 检查 | 固定条件 | 失败结果 |
|:---|:---|:---|
| 检查点身份 | 模型提交哈希、数据哈希、配置哈希不变 | 禁止恢复 |
| 可恢复状态 | 适配器、优化器、调度器与随机数状态齐全 | 禁止恢复 |
| 训练进度 | 从 `CHECKPOINT_STEP` 继续到 `TOTAL_STEPS`，更新总数完整 | `reject` |
| 更新过程 | 全部更新有限，并记录梯度范数与参数增量 | `reject` |
| 独立评测 | 最终损失不超过绝对门槛，且相对初始评测不退化 | `reject` |
| 重载一致性 | 保存前后评测损失差异不超过容差 | `reject` |
| 产物完整性 | 适配器、分词器与目录哈希存在 | `reject` |

```python
# 本单元输出固定门槛；完整训练在 5.4 执行。
print({"gates": GATES, "checkpoint_step": CHECKPOINT_STEP, "total_steps": TOTAL_STEPS})
```

### 5.4 执行中断恢复训练并保存完整证据

本单元执行固定配置的完整实验负载，并保存训练、恢复、评测与产物证据。

```python
if RUN_MODE == "real_gpu":
    from datasets import load_dataset
    from transformers import AutoTokenizer

    selection_manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    sweep = json.loads(Path(selection_manifest["selection_evidence"]).read_text(encoding="utf-8"))
    data_manifest = json.loads(Path(selection_manifest["upstream_manifest"]).read_text(encoding="utf-8"))
    model_commit = sweep["model"]["resolved_commit"]
    dataset_info = data_manifest["dataset"]
    source_count = data_manifest["workload"]["sample_count"]
    dataset = load_dataset(dataset_info["id"], split=f"train[:{source_count}]", revision=dataset_info["resolved_commit"])
    snapshot = data_manifest["snapshot"]
    train_records = [dict(dataset[index]) for index in snapshot["train_record_indices"][:TRAIN_EXAMPLES]]
    evaluation_records = [dict(dataset[index]) for index in snapshot["evaluation_record_indices"][:EVALUATION_EXAMPLES]]
    if len(train_records) != TRAIN_EXAMPLES or len(evaluation_records) != EVALUATION_EXAMPLES:
        raise RuntimeError("64 节准入快照不足以满足固定训练/评测工作负载")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_commit)
    run_evidence = run_lora_delivery(
        model_id=MODEL_ID, revision=model_commit, tokenizer=tokenizer,
        train_records=train_records, evaluation_records=evaluation_records,
        lora_config=selection_manifest["selected_config"], max_length=MAX_LENGTH,
        total_steps=TOTAL_STEPS, checkpoint_step=CHECKPOINT_STEP,
        learning_rate=LEARNING_RATE, seed=SEED, dtype_name=DTYPE,
        checkpoint_directory=CHECKPOINT_DIR, artifact_directory=ARTIFACT_DIR,
    )
    record = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "environment_profile": PROFILE_ID, "run_mode": RUN_MODE, "run_label": RUN_LABEL,
        "selection_manifest": str(SELECTION_MANIFEST),
        "model": {"id": MODEL_ID, "resolved_commit": model_commit},
        "data": {"manifest": selection_manifest["upstream_manifest"], "sample_ids_hash": data_manifest["audit"]["sample_ids_hash"]},
        "selected_config": selection_manifest["selected_config"],
        "workload": {"max_length": MAX_LENGTH, "train_examples": TRAIN_EXAMPLES, "evaluation_examples": EVALUATION_EXAMPLES, "total_steps": TOTAL_STEPS, "checkpoint_step": CHECKPOINT_STEP, "learning_rate": LEARNING_RATE, "seed": SEED},
        "gates": GATES, "evidence": run_evidence,
    }
    save_json_record(EVIDENCE_PATH, record)
    print({"evidence_path": str(EVIDENCE_PATH), "artifact": run_evidence["artifact"]})
else:
    print({"run_mode": RUN_MODE, "next_action": "确认训练身份后改为 real_gpu"})
```

### 5.5 独立读取结果并核对恢复、质量与重载证据

```python
def decide_sft_delivery_for_run(evidence, gates):
    """依据恢复、评测、重载和产物哈希形成交付决策。"""
    checks = {
        "execution": evidence.get("status") == "ok",
        "resume": evidence.get("checkpoint", {}).get("saved_step") == evidence.get("checkpoint", {}).get("resumed_from_step"),
        "completed_budget": evidence.get("checkpoint", {}).get("completed_step") > evidence.get("checkpoint", {}).get("resumed_from_step", 0) and evidence.get("checkpoint", {}).get("training_update_count") == evidence.get("checkpoint", {}).get("completed_step"),
        "recovery_state": all(evidence.get("checkpoint", {}).get(key) is True for key in ("optimizer_state_present", "scheduler_state_present", "rng_state_present")),
        "quality_absolute": evidence["final_eval_loss"] <= gates["maximum_eval_loss"],
        "quality_relative": evidence["final_eval_loss"] - evidence["initial_eval_loss"] <= gates["maximum_eval_loss_regression"],
        "reload": evidence["reload_loss_abs_diff"] <= gates["maximum_reload_loss_abs_diff"],
        "update_dynamics": evidence.get("update_dynamics", {}).get("finite_all") is True,
        "artifact": bool(evidence.get("artifact", {}).get("sha256")),
    }
    decision = "accept" if all(checks.values()) else "reject"
    return {"decision": decision, "failed_checks": [name for name, passed in checks.items() if not passed]}

if EVIDENCE_PATH.exists():
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    decision = decide_sft_delivery_for_run(record["evidence"], record["gates"])
    print({
        "decision": decision,
        "final_eval_loss": record["evidence"]["final_eval_loss"],
        "reload_loss_abs_diff": record["evidence"]["reload_loss_abs_diff"],
        "checkpoint": record["evidence"]["checkpoint"],
        "update_dynamics": record["evidence"].get("update_dynamics"),
        "artifact": record["evidence"]["artifact"],
        "peak_allocated_bytes": record["evidence"]["peak_allocated_bytes"],
    })
```

### 5.6 形成 SFT 基线产物清单

5.5 的全部硬门槛通过后，写入包含适配器哈希、完整恢复状态、更新动力学摘要和评测结论的交付清单。84 和 85 节从该 JSON 重新加载 SFT 基线。

| 决策项 | 实测证据 | 结果 | 后续动作 |
|:---|:---|:---|:---|
| 恢复连续性 | 适配器、优化器、调度器、随机数状态和训练步连续 | 新运行读取 5.5 结果 | 任一缺失时拒绝交付 |
| 更新过程 | `finite_all`、梯度范数、参数增量和学习率摘要 | 新运行读取 5.5 结果 | 非有限时先定位，不生成基线 |
| 质量与重载 | 独立评测变化与重载差值 | 新运行读取 5.5 结果 | 固定 SFT 基线前必须通过 |
| 项目决策 | 全部硬门槛 | 新运行读取 5.5 结果 | `accept` 后才交给 84 / 85 |

`sft_delivery_v2_1` 是补充恢复状态字段之前的历史证据：它证明了旧交付链的恢复、质量和重载一致性，但不包含调度器、随机数和更新动力学摘要。保留该证据用于对照；按新契约重新运行时使用新的 `RUN_LABEL`，再登记新的结论。

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 62 节 SFT 交付；`sft_delivery_v2_1` |
| 运行条件 | Qwen2.5-0.5B-Instruct；中断恢复训练与独立重载检查 |
| 原始证据 | 训练、恢复和评测记录保存在正式证据中 |
| 正式证据 | [SFT 证据](../../benchmarks/results/62_sft_delivery/sft_delivery_evidence_sft_delivery_v2_1.json) |
| 清单与交接 | [交付清单](../../benchmarks/results/62_sft_delivery/sft_delivery_manifest_sft_delivery_v2_1.json)；本地 adapter 与 checkpoint 由清单记录 |
| 结论 | 旧契约下可复用；当前契约仍需新标签复测调度器、随机数和更新动力学 |

```python
if RUN_MODE == "real_gpu":
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    decision = decide_sft_delivery_for_run(record["evidence"], record["gates"])
    manifest = {
        "schema_version": SCHEMA_VERSION, "semantic_id": SEMANTIC_ID,
        "run_label": RUN_LABEL, "decision": decision["decision"], "decision_details": decision,
        "source_evidence": str(EVIDENCE_PATH),
        "model": record["model"], "data": record["data"],
        "selected_config": record["selected_config"],
        "checkpoint": record["evidence"]["checkpoint"],
        "artifacts": {"adapter": {"role": "training.sft_baseline", **record["evidence"]["artifact"]}},
        "quality": {"final_eval_loss": record["evidence"]["final_eval_loss"], "reload_loss_abs_diff": record["evidence"]["reload_loss_abs_diff"]},
        "next_consumer": ["PT-DPO-OFFLINE", "PT-GRPO-GROUPWISE"] if decision["decision"] == "accept" else [],
    }
    save_json_record(MANIFEST_PATH, manifest)
    print({"manifest_path": str(MANIFEST_PATH), "decision": decision["decision"], "failed_checks": decision["failed_checks"]})
```

## 相关阅读

- [Transformers PEFT 集成文档](https://huggingface.co/docs/transformers/peft)
- [Hugging Face PEFT checkpoint 格式](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- [Part 02 · 84 DPO 偏好优化项目](./84_DPO_Preference_Project.md)
- [Part 02 · 85 GRPO 在线奖励优化项目](./85_GRPO_Groupwise_Alignment_Project.md)
