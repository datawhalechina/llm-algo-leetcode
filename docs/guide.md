# 使用指南

只回答两件事：先在哪看，Notebook 该在哪跑。

## 环境选择

| 场景 | 建议 |
|---|---|
| 先看内容 | 在线站点 |
| Part 0 / Part 1 | 在线 Notebook 或本地基础环境 |
| Part 2 | 本地 CPU-first / Colab CPU / CNB CPU |
| Part 3 / Part 4 | 本地 NVIDIA GPU / Colab GPU / CNB GPU |
| 团队统一交付 | CNB / Docker / 云端 GPU |

## 记住三条

- `Colab` 是阅读和运行入口。
- `CPU-first` / `GPU-required` 是执行能力，不是入口名称。
- `CNB` / `Docker` / 云端 GPU 是统一交付方式。

## 常用命令

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py part4 --no-build
python verify.py all --no-build
```

定点排查时用：

```bash
python tools/test_chapter0_1_notebooks.py
python tools/test_notebook_answers.py path/to/your.ipynb --mode both
```

## Part 02 项目验证

训练、推理和显存项目除了 Notebook 答案区测试，还可能需要真实 GPU、模型下载或本地 backend。
推理项目 `66–70` 的完整检查顺序、结果文件和 JSON schema 见
[66–70 推理项目验证清单](./verification/inference_projects.md)。

建议先完成 CPU-first 验证，再按 Notebook 中的开关进入真实 backend；没有 GPU 时不要把
Practice-P1 的本地/模拟结果当作 Practice-P2 的真实服务结论。

### 项目环境预检与安装

项目节可以独立运行。进入某一节后，先运行该节的“环境预检”代码块；它会检查项目根目录、当前 Python / PyTorch、CUDA、GPU 显存、必需包、模型或 backend 能力，以及 `benchmarks/results/` 是否可写。预检状态含义如下：

| 状态 | 含义 | 处理方式 |
|---|---|---|
| `ok` | 当前配置满足本节实验要求 | 继续运行 |
| `warning` | 可运行降级路径，但可选能力不可用 | 按提示选择 CPU 或跳过扩展 |
| `blocked` | 当前配置不能安全运行目标实验 | 先按 `next_actions` 修复，不加载模型 |

也可以在仓库根目录直接执行：

```bash
python tools/environment_preflight.py --packages transformers
```

真实 GPU 项目使用：

```bash
python tools/environment_preflight.py \
  --gpu \
  --packages transformers \
  --output benchmarks/results/preflight.json
```

推理 backend 项目按本节要求追加 `vllm` 或 `sglang`；QLoRA 项目追加 `peft`、`bitsandbytes`；LoRA / SFT 项目追加 `accelerate`、`datasets`。普通依赖使用当前 Notebook 内核的 Python 安装：

```python
import sys
!{sys.executable} -m pip install -U transformers accelerate peft datasets
```

不要在预检失败时直接重装 PyTorch。若 `torch.version.cuda` 为 `None` 或 `cuda_available` 为 `False`，先确认 Colab 已启用 GPU，再重启 runtime；只有确认运行时仍缺少 CUDA wheel 时，才按平台说明选择对应的 PyTorch 安装命令。vLLM、SGLang 和 TensorRT-LLM 也必须先确认 CUDA、驱动和版本匹配后再安装。

预检通过后再下载模型、启动 backend 或执行训练。结果统一保存到 `benchmarks/results/`，并记录预检报告、模型、dtype、workload、设备、软件版本和实验状态。

每个需要额外环境的项目页，在实验开关附近只保留以下提示，并链接回本节：

> **运行提示：** 先查看[使用指南中的项目环境预检与安装说明](./guide.md#项目环境预检与安装)，再打开本节的真实实验开关。CPU-first 路径不要求 GPU；真实 GPU 或 backend 路径必须先通过预检。

## 最小规则

- Part 0 / Part 1：优先在线 Notebook 或本地基础环境。
- Part 2：默认 CPU-first，少数题再切 GPU。
- Part 3 / Part 4：完整验收需要 GPU；没有 GPU 时先阅读。
- CNB 的目标是统一交付，不是新增一套内容分层。
