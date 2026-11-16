# Operator. Optimization Delivery Project | 算子优化交付综合项目

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/05_Integrated_Projects_and_Decisions/05_Operator_Optimization_Delivery/Operator_Optimization_Delivery_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

两种注意力实现的局部耗时不同，接入解码器后还会经过投影、位置旋转、残差和 MLP。本项目把 Triton FlashAttention 接入两层解码器的预填充路径，与 PyTorch 缩放点积注意力接口（SDPA）自动选择的后端比较：只替换注意力实现，观察完整模块能快多少，以及输出误差和峰值显存是否满足要求。输入与权重由本项目合成；执行路径通过性能剖析轨迹核对。
## 项目执行

### 1. 配置本次运行

本项目使用 `PERF-CUDA-MECHANISM`。依次运行 `contract_check`、`measure`、`close`；新一轮测量使用新的 `RUN_LABEL`。

```python
RUN_LABEL = 'replace_with_unique_label'  # 本轮证据的唯一标签
RUN_STAGE = 'contract_check'  # contract_check、measure 或 close
```

### 2. 检查环境与输入

协议检查不启动 GPU 测量。`measure` 阶段会检查 CUDA、BF16 与可用显存；项目不下载模型或数据集。

```python
from pathlib import Path
import json
import yaml

from tools.project_runtime import resolve_project_root
from tools.part05_operator_delivery_runtime import preflight, validate_protocol

if RUN_STAGE not in {'contract_check', 'measure', 'close'}:
    raise ValueError(f'未知 RUN_STAGE：{RUN_STAGE}')
if RUN_STAGE in {'measure', 'close'} and RUN_LABEL == 'replace_with_unique_label':
    raise ValueError('请先设置本轮唯一 RUN_LABEL。')
PROJECT_ROOT = resolve_project_root()
PROJECT_DIR = PROJECT_ROOT / '05_Integrated_Projects_and_Decisions/05_Operator_Optimization_Delivery'
PROTOCOL = yaml.safe_load((PROJECT_DIR / 'project_config.example.yaml').read_text(encoding='utf-8'))
protocol_errors = validate_protocol(PROTOCOL)
if protocol_errors:
    raise ValueError('项目协议检查失败：' + '; '.join(protocol_errors))
PREFLIGHT = None
if RUN_STAGE == 'measure':
    import torch
    PREFLIGHT = preflight(torch, PROTOCOL)
print({'protocol': 'passed', 'stage': RUN_STAGE, 'preflight': PREFLIGHT})
```

### 3. 确认候选与比较条件

基线是 PyTorch SDPA 自动后端，候选是 Triton 前向内核。每个序列长度使用同一输入和权重，轮换执行顺序，配对测量 10 次。长序列的门槛是逐轮基线耗时／候选耗时的中位数达到 1.05；短序列回退、数值容差和显存预算一并检查。基线实际分发路径写入性能剖析记录。

```python
print(json.dumps({'workload': PROTOCOL['workload'], 'candidates': PROTOCOL['candidates'], 'gates': PROTOCOL['gates']}, ensure_ascii=False, indent=2))
```

### 4. 执行项目并保存证据

`measure` 对四种序列长度执行配对测量，并保存原始指标和两条路径的 profiler 轨迹。同一标签已有结果时不会覆盖。

```python
if RUN_STAGE == 'measure':
    from tools.part05_operator_delivery_runtime import run_operator_delivery
    RAW_RECORD = run_operator_delivery(torch, PROTOCOL, PROJECT_ROOT, RUN_LABEL)
    print({'status': RAW_RECORD['status'], 'result_path': RAW_RECORD['result_path'], 'shapes': len(RAW_RECORD['measurements'])})
else:
    print('当前阶段不执行 GPU 测量。')
```

### 5. 读取并核对结果

读取已有结果，核对本轮标签、协议、固定形状和配对样本，并检查汇总指标能否由样本复算。这里展示初步判断；下一步还会核验实际轨迹文件及其 SHA256，再保存最终决策。

```python
RAW_PATH = PROJECT_ROOT / PROTOCOL['outputs']['directory'] / RUN_LABEL / 'raw_result.json'
RAW_RECORD = json.loads(RAW_PATH.read_text(encoding='utf-8')) if RAW_PATH.is_file() else None
if RUN_STAGE == 'close' and RAW_RECORD is None:
    raise FileNotFoundError(f'先完成 RUN_STAGE=measure：{RAW_PATH}')
REVIEW = None
if RAW_RECORD is not None:
    from tools.part05_operator_delivery_runtime import evaluate_measurements
    if RAW_RECORD.get('run_label') != RUN_LABEL:
        raise ValueError('原始结果的 RUN_LABEL 与本次配置不一致。')
    REVIEW = evaluate_measurements(RAW_RECORD, PROTOCOL)
print({'record': 'loaded' if RAW_RECORD else 'not_recorded', 'path': str(RAW_PATH), 'preliminary_decision': REVIEW['decision'] if REVIEW else None})
```

### 6. 形成决策并登记

`accept` 要求数值和执行路径证据完整、两个长序列均达到收益门槛，且短序列回退和显存均在预算内；证据有效但收益不足时为 `tune`。

```python
if RUN_STAGE == 'close':
    from tools.part05_operator_delivery_runtime import close_operator_delivery
    DECISION = close_operator_delivery(PROJECT_ROOT, RUN_LABEL, PROTOCOL)
    print(DECISION)
else:
    print('将 RUN_STAGE 设为 close 后生成决策记录。')
```

## 相关阅读

- [11 · Triton Flash Attention](../../03_Triton_Kernels/11_Triton_Flash_Attention.md)：查看本项目调用的前向内核。
- [20 · LLaMA 风格 Block 集成](../../03_Triton_Kernels/20_Triton_Llama3_Block_Project.md)：了解内核接入 decoder block 的路径。
