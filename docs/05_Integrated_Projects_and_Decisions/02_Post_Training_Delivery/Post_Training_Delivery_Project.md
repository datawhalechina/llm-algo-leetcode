# Post. Training Delivery Project | MiniMind 后训练项目

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery/Post_Training_Delivery_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

从 MiniMind 64M 预训练权重出发，用固定的小型本地数据对照全参数 SFT 与原生 LoRA，再从全参数 SFT 分支继续 DPO 并独立评测。结果只说明这组任务和训练预算，不代表通用能力或生产发布结论。
## 项目执行
### 1. 配置本次运行

只改上方 `RUN_LABEL`、`RUN_STAGE`。依次运行 `contract_check`、`prepare`、`probe`、`train`、`close`。`probe` 用一小步全参数 SFT、LoRA 和 DPO 检查训练路径的显存峰值；探测未通过就不要开始完整训练。

```python
RUN_LABEL = "replace_with_unique_label"  # 每轮填写新的结果标签
RUN_STAGE = "contract_check"  # contract_check、prepare、probe、train、close

```

### 2. 检查环境与输入

`prepare`、`probe` 与 `train` 会核对 `PT-SFT-CUDA` Profile；预检失败时不要继续。MiniMind 的依赖使用当前 Profile，不再安装上游整份 requirements。

`prepare` 会把固定版本源码克隆到 `~/minimind-part05`，从 Hugging Face 缓存复用或下载 `pretrain_768.pth`，并在仓库内生成带哈希的练习数据；它不启动训练。已有源码目录版本不匹配时会停止，不覆盖目录内容。

```python
from pathlib import Path
import json
import sys
import subprocess
import yaml

from tools.project_runtime import resolve_project_root
from tools.minimind_post_training_runtime import (
    close_project, load_prepared, prepare_project, probe_training_memory, train_and_evaluate, validate_config, validate_training_evidence,
)

PROJECT_ROOT = resolve_project_root()
PROJECT_DIR = PROJECT_ROOT / "05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery"
config = yaml.safe_load((PROJECT_DIR / "project_config.example.yaml").read_text(encoding="utf-8"))
config["run_label"] = RUN_LABEL
if RUN_STAGE not in {"contract_check", "prepare", "probe", "train", "close"}:
    raise ValueError(f"未知 RUN_STAGE：{RUN_STAGE}")
print({"run_label": RUN_LABEL, "run_stage": RUN_STAGE, "profile": config["environment_profile"]})

config_errors = validate_config(config, allow_placeholder=RUN_STAGE == "contract_check")
if config_errors:
    raise ValueError("项目配置错误：" + "; ".join(config_errors))
if RUN_STAGE in {"prepare", "probe", "train"}:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools/environment_preflight.py"), "--profile", "PT-SFT-CUDA"],
        cwd=PROJECT_ROOT, check=False,
    )
    if completed.returncode:
        raise RuntimeError("PT-SFT-CUDA 环境预检失败")
print({"config_valid": True, "stage": RUN_STAGE})
```

### 3. 确认训练分支与比较条件

全参数 SFT 与原生 LoRA 使用同一基座和训练预算；DPO 从全参数 SFT 分支继续。训练集与评测集分开，三条路径都使用第 2 步准备的固定输入。

| 指标 | 比较目的 | 预期解释 |
|:---|:---|:---|
| 留出集损失、任务正确率、偏好胜率 | 比较各训练分支的输出质量 | 不预设 LoRA 或 DPO 必然胜出；报告样本数与差异 |
| 峰值显存、阶段耗时 | 判断训练链是否可在目标设备完成 | LoRA 预计减少可训练参数；实际显存和耗时仍以测量为准 |
| 参数量、产物大小、SHA256 | 确认分支产物及来源 | 三条训练路径分别登记，不用训练损失代替交付证据 |
| 数据摘要、切分与重复信息 | 排除评测泄漏和输入漂移 | 任一来源身份不一致时不作配对比较 |

```python
if RUN_STAGE == "contract_check":
    print({"config": "passed", "next": "将 RUN_STAGE 改为 prepare"})
elif RUN_STAGE == "prepare":
    prepared = prepare_project(config, PROJECT_ROOT)
    print(prepared)
elif RUN_STAGE in {"probe", "train", "close"}:
    prepared = load_prepared(config, PROJECT_ROOT)
    print({"prepared": True, "data_counts": prepared["data_counts"], "split_disjoint": prepared["split_disjoint"]})
```

### 4. 执行训练并评测

`train` 要求同标签的全参数 SFT、LoRA、DPO 短步显存探测通过，再检查当前可用显存并依次执行 MiniMind 原生训练脚本。全参数 SFT 与秩 16 的 LoRA 使用相同数据、轮数、批量和学习率；DPO 从全参数 SFT 权重开始。随后在独立样本上测留出损失和偏好胜率，并记录模型/adapter 哈希、参数量、产物大小及阶段耗时。出现显存不足时停止，不自动缩小配置或重试。

```python
if RUN_STAGE == "probe":
    print(probe_training_memory(config, PROJECT_ROOT, prepared))
elif RUN_STAGE == "train":
    print(train_and_evaluate(config, PROJECT_ROOT, prepared))
else:
    print("此阶段不运行训练或评测")
```

### 5. 读取并核对实验记录

训练结束后，检查原始记录中的固定源码和权重版本、训练/评测切分、三阶段日志、逐模型指标与峰值显存。缺少任一模型或 adapter 产物及其哈希时，不进入收口。

```python
run_dir = PROJECT_ROOT / config["outputs"]["directory"] / RUN_LABEL
raw_path = run_dir / "raw_result.json"
if RUN_STAGE in {"close", "train"} and raw_path.is_file():
    raw_result = json.loads(raw_path.read_text(encoding="utf-8"))
    evidence_errors = validate_training_evidence(raw_result, config, PROJECT_ROOT)
    if evidence_errors:
        raise RuntimeError("实验记录不完整：" + "; ".join(evidence_errors))
    print({"evidence_status": "passed", "raw_result": str(raw_path), "models": list(raw_result["evaluation"])})
elif RUN_STAGE == "close":
    raise FileNotFoundError(f"先运行 RUN_STAGE='train'：{raw_path}")
else:
    print({"evidence_status": "not_run", "next": "train"})
```

### 6. 形成决策并登记结果

`close` 只读取本次原始记录，核对数据隔离、模型/adapter 哈希、全参数 SFT 与 LoRA 的留出损失差、DPO 偏好胜率和实测显存，生成 `accept / tune / reject` 报告。这里的 `accept` 仅表示本项目固定小任务通过预设门槛。

```python
if RUN_STAGE == "close":
    print(close_project(config, PROJECT_ROOT))
else:
    print("将 RUN_STAGE 设为 close 后生成报告")
```

## 相关阅读

- [13 · SFT 训练循环](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md)：理解监督样本如何参与参数更新。
- [14 · LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.md)：理解本项目全参数与 adapter 分支的差异。
- [17 · DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.md)：理解偏好对如何形成离线策略更新。
