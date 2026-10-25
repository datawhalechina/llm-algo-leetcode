# Performance. Optimization Delivery Project | 端到端性能优化与回归交付项目

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery/Performance_Optimization_Delivery_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

本项目围绕固定的单卡训练负载测量基线、优化候选和组合，并检查回归风险。73–76 的结果只作背景参考，不是准入条件。
## 项目执行

### 1. 配置本次运行

本项目固定模型快照、输入形状、候选矩阵和判断门槛。四个阶段都使用 `PERF-CUDA-MECHANISM`；依次运行 `contract_check`、`prepare`、`measure`、`close`，只有 `measure` 会加载模型并训练。

```python
RUN_LABEL = "replace_with_unique_label"  # 每轮填写新的结果标签
RUN_STAGE = "contract_check"  # contract_check、prepare、measure、close

```

### 2. 检查环境与输入

`prepare` 检查本机模型快照、CUDA、BF16 与显存余量，并运行小型设备探测；模型只从仓库缓存读取，缺少文件时停止，不下载。

```python
from pathlib import Path
import subprocess
import sys
import yaml

from tools.project_runtime import resolve_project_root
from tools.performance_delivery_runtime import review_delivery, save_json_record_once
from tools.performance_delivery_benchmark import (
    build_review_config, check_model_assets, preflight, run_benchmark,
    validate_project_protocol,
)

if RUN_STAGE not in {"contract_check", "prepare", "measure", "close"}:
    raise ValueError(f"未知 RUN_STAGE：{RUN_STAGE}")
if RUN_STAGE in {"measure", "close"} and RUN_LABEL == "replace_with_unique_label":
    raise ValueError("请先设置本次唯一 RUN_LABEL。")
PROJECT_ROOT = resolve_project_root()
PROJECT_DIR = PROJECT_ROOT / "05_Integrated_Projects_and_Decisions/03_Performance_Optimization_Delivery"
PROTOCOL = yaml.safe_load((PROJECT_DIR / "project_config.example.yaml").read_text(encoding="utf-8"))
protocol_errors = validate_project_protocol(PROTOCOL)
if protocol_errors:
    raise ValueError("固定实验协议未通过检查：" + "; ".join(protocol_errors))
ASSET_CHECK = check_model_assets(PROJECT_ROOT)
print({"stage": RUN_STAGE, "model_assets": ASSET_CHECK, "gpu_measurement_started": False})

if RUN_STAGE == 'prepare':
    import torch
    subprocess.run([sys.executable, str(PROJECT_ROOT / 'tools/environment_preflight.py'), '--profile', 'PERF-CUDA-MECHANISM'], cwd=PROJECT_ROOT, check=True)
    PREFLIGHT = preflight(torch, PROJECT_ROOT)
    print(PREFLIGHT)
elif RUN_STAGE == 'contract_check':
    print({'protocol': 'passed', 'candidate_matrix': ['baseline', 'fused_adamw', 'activation_checkpoint', 'fused_checkpoint'], 'gpu_measurement_started': False})
```

### 3. 确认候选与比较条件

基线、fused AdamW、激活检查点和两者组合使用相同模型快照及配对输入。实验前固定最低完整步收益、显存增幅上限、损失差和重复数；组合条件单独测量，不预设收益可相加。

| 指标 | 通过条件 | 解释范围 |
|:---|:---|:---|
| 完整训练步耗时 | 配对收益至少 5% | 不用优化器单步耗时替代完整训练步 |
| 峰值显存 | 相对基线增幅不超过 1% | 守护资源成本 |
| 损失差 | 回归不超过 2%，且无非有限值 | 守护数值质量 |
| 重复与组合 | 每条件 3 组配对重复；组合独立实测 | 结论仅适用于固定模型、输入与设备 |
### 4. 执行测量并保存证据

`measure` 运行完整训练步并保存配对测量记录；通过 `prepare` 后再测量，结束后进入 `close`。

```python
if RUN_STAGE == 'measure':
    import torch
    from transformers import AutoModelForCausalLM
    CAPTURE_MANIFEST = run_benchmark(torch, AutoModelForCausalLM, PROJECT_ROOT, RUN_LABEL)
    print(CAPTURE_MANIFEST)
else:
    CAPTURE_MANIFEST = None
    print({'measurement_started': False, 'stage': RUN_STAGE})
```

### 5. 读取并核对结果

决策仅比较本项目配对测量。性能收益、峰值显存、损失、剖析记录和配置回退路径均须有证据；缺项保留 `tune`，工作负载或执行正确性不匹配则拒绝该候选。

```python
if RUN_STAGE == 'close':
    CONFIG = build_review_config(PROTOCOL, PROJECT_ROOT, RUN_LABEL)
    REVIEW_RECORD = review_delivery(CONFIG, PROJECT_ROOT)
    print({'decision': REVIEW_RECORD['decision'], 'decision_detail': REVIEW_RECORD['decision_detail']})
else:
    REVIEW_RECORD = None
    print('尚未进入 close 阶段。')
```

### 6. 形成决策并登记

在 `PERF-CUDA-MECHANISM` 内核中依次运行四个阶段，每次改动 `RUN_STAGE` 后从头运行单元格。测量目录不可覆盖，重测请换新标签。`close` 不执行 GPU 运算。

```python
if RUN_STAGE == 'close':
    output_dir = PROJECT_ROOT / PROTOCOL['outputs']['directory'] / RUN_LABEL
    output_path = output_dir / 'decision.json'
    saved_path = save_json_record_once(output_path, REVIEW_RECORD)
    print({'result_path': str(saved_path), 'decision': REVIEW_RECORD['decision']})
else:
    print('本阶段未生成最终决策文件。')
```

## 相关阅读

- [75 · 优化器执行路径基准](../../02_PyTorch_Algorithms/75_Optimizer_Execution_Path_Benchmark.md)：理解融合优化器候选的执行路径。
- [76 · 激活检查点与卸载基准](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)：理解激活检查点候选的显存与吞吐权衡。
