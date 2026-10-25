# Quantization. Delivery Project | 量化交付综合项目

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/05_Integrated_Projects_and_Decisions/04_Quantization_Delivery/Quantization_Delivery_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

本项目用固定校准集生成 GPTQ W4A16 模型，再与 BF16 基线使用同一 vLLM 服务配置和请求负载配对。项目检查留出集 NLL、请求正确率、服务延迟、吞吐和显存，最后判断量化产物是否适合这组部署条件。
## 项目执行

### 1. 配置本次运行

所有阶段使用同一 `RUN_LABEL`。依次运行 `contract_check`、`prepare`、`quantize`、`serve`、`close`；每个阶段都只执行本项目对应的工作。

```python
RUN_LABEL = 'replace_with_unique_label'  # 本轮项目证据的唯一标签
RUN_STAGE = 'contract_check'  # contract_check、prepare、quantize、serve 或 close
```

### 2. 检查环境与输入

`prepare`、`quantize` 使用 `Q-CUDA-MECHANISM`；`serve` 使用隔离的 `INF-VLLM-011-CU128`。环境安装见[量化环境说明](../../gpu_environment_assets.md#q-cuda-mechanism)与[推理服务环境说明](../../gpu_environment_assets.md#inf-vllm-011-cu128)。`prepare` 会下载配置中固定版本的模型与数据；其余检查不自动换用其他版本。

```python
from pathlib import Path
import json
import subprocess
import sys
import yaml

from tools.project_runtime import resolve_project_root
from tools.part05_quantization_delivery_runtime import validate_protocol

ALLOWED_STAGES = {'contract_check', 'prepare', 'quantize', 'serve', 'close'}
if RUN_STAGE not in ALLOWED_STAGES:
    raise ValueError(f'未知 RUN_STAGE：{RUN_STAGE}')
if RUN_STAGE != 'contract_check' and RUN_LABEL == 'replace_with_unique_label':
    raise ValueError('请先设置本轮唯一 RUN_LABEL。')
PROJECT_ROOT = resolve_project_root()
PROJECT_DIR = PROJECT_ROOT / '05_Integrated_Projects_and_Decisions/04_Quantization_Delivery'
PROTOCOL = yaml.safe_load((PROJECT_DIR / 'project_config.example.yaml').read_text(encoding='utf-8'))
protocol_errors = validate_protocol(PROTOCOL)
if protocol_errors:
    raise ValueError('项目协议检查失败：' + '; '.join(protocol_errors))
PROFILE = 'INF-VLLM-011-CU128' if RUN_STAGE == 'serve' else 'Q-CUDA-MECHANISM'
if RUN_STAGE in {'prepare', 'quantize', 'serve'}:
    completed = subprocess.run([sys.executable, 'tools/environment_preflight.py', '--profile', PROFILE], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    ENVIRONMENT_REPORT = json.loads(completed.stdout)
    if not ENVIRONMENT_REPORT.get('ready'):
        raise RuntimeError('环境预检未通过：' + json.dumps(ENVIRONMENT_REPORT.get('reasons', []), ensure_ascii=False))
else:
    ENVIRONMENT_REPORT = None
print({'protocol': 'passed', 'stage': RUN_STAGE, 'profile': PROFILE if ENVIRONMENT_REPORT else None})
```

### 3. 确认候选与比较条件

基线和候选使用相同模型快照、vLLM 版本、生成参数与 20 条固定请求。量化阶段的留出集与校准集分开；服务阶段会核对候选模型路径和量化后端日志。

```python
print(json.dumps({'model': PROTOCOL['model'], 'quantization': PROTOCOL['quantization'], 'serving': PROTOCOL['serving']}, ensure_ascii=False, indent=2))
```

### 4. 执行项目并保存证据

各阶段按第 2 步所列环境运行。切换环境或重启内核后保留同一个 `RUN_LABEL`；阶段产物用文件清单和 SHA256 校验。

```python
if RUN_STAGE == 'prepare':
    from datasets import load_dataset
    from tools import model_runtime
    from tools.part05_quantization_delivery_runtime import prepare_quantization_inputs
    PREPARED = prepare_quantization_inputs(protocol=PROTOCOL, project_root=PROJECT_ROOT, run_label=RUN_LABEL, model_runtime=model_runtime, load_dataset=load_dataset)
    print({'inputs_manifest': str(PREPARED)})
elif RUN_STAGE == 'quantize':
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tools.part05_quantization_delivery_runtime import quantize_and_evaluate
    ARTIFACT_RECORD = quantize_and_evaluate(torch=torch, auto_model_class=AutoModelForCausalLM, tokenizer_class=AutoTokenizer, load_dataset=load_dataset, protocol=PROTOCOL, project_root=PROJECT_ROOT, run_label=RUN_LABEL)
    print({'artifact': ARTIFACT_RECORD['artifact_path'], 'nll_ratio_vs_bf16': ARTIFACT_RECORD['quality']['nll_ratio_vs_bf16']})
elif RUN_STAGE == 'serve':
    from tools.inference_project_runtime import run_backend_benchmark, start_optional_vllm, stop_optional_vllm
    from tools.part05_quantization_delivery_runtime import run_service_pair
    SERVICE_PAIR = run_service_pair(protocol=PROTOCOL, project_root=PROJECT_ROOT, run_label=RUN_LABEL, start_server=start_optional_vllm, stop_server=stop_optional_vllm, run_benchmark=run_backend_benchmark)
    print(SERVICE_PAIR)
else:
    print('当前阶段不生成或测量量化产物。')
```

### 5. 读取并核对结果

检查输入清单、量化记录、服务配对文件是否齐全；`close` 会再次校验产物哈希、请求 ID 与服务模型路径。缺少证据时不生成交付结论。

```python
RUN_DIR = PROJECT_ROOT / PROTOCOL['outputs']['directory'] / RUN_LABEL
EVIDENCE_PATHS = [RUN_DIR / name for name in ('inputs.json', 'quantization.json', 'service_pair.json', 'service_bf16.json', 'service_gptq.json')]
EVIDENCE_STATUS = {path.name: path.is_file() for path in EVIDENCE_PATHS}
if RUN_STAGE == 'close' and not all(EVIDENCE_STATUS.values()):
    raise FileNotFoundError('阶段证据不齐：' + json.dumps(EVIDENCE_STATUS, ensure_ascii=False))
print(EVIDENCE_STATUS)
```

### 6. 形成决策并登记

只有质量、固定请求、量化执行路径与服务门槛全部通过才判 `accept`；有效证据未达到目标判 `tune`，身份或执行路径错误判 `reject`。

```python
if RUN_STAGE == 'close':
    from tools.part05_quantization_delivery_runtime import decide_quantization_delivery
    DECISION = decide_quantization_delivery(PROJECT_ROOT, RUN_LABEL, PROTOCOL)
    print(DECISION)
else:
    print('完成 prepare、quantize、serve 后，将 RUN_STAGE 设为 close。')
```

## 相关阅读

- [40 · GPTQ 与 AWQ 权重量化](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md)：了解 GPTQ 校准与留出集评测。
- [67 · 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)：了解量化产物的 vLLM 加载与服务对照。
