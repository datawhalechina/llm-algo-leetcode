# 使用指南

只回答两件事：先在哪看，Notebook 该在哪跑。

## 开始前：确认 Python 环境入口

运行 Notebook 前，先判断当前平台是否已经提供可用的 Python 环境：

```bash
which python
python -c "import sys; print(sys.executable)"
command -v conda || true
```

不同平台的处理方式不同：

| 平台 | 是否需要自行安装 Conda | 建议 |
|---|---|---|
| Colab | 通常不需要 | 使用当前 runtime，安装本节缺失的普通依赖 |
| ModelScope Notebook | 通常不需要 | 使用平台提供的 Notebook Kernel |
| 已配置 Conda 的本地或云端 | 不需要重复安装 | 激活已有环境后启动 JupyterLab |
| 只有系统 Python 的云端容器 | 可选 | 可以直接使用系统 Python；若需要隔离，再安装 Miniconda 或创建 venv |
| 本地多项目、多 CUDA 版本 | 建议使用 | 用 Conda 分隔训练环境和推理 backend 环境 |

因此，Conda 不是本教程的硬性前置条件。它的作用是隔离依赖，不是提供 GPU 能力。GPU 能否使用，仍由驱动、CUDA 和 PyTorch wheel 决定。

如果本地或云端没有 Conda，但希望创建独立环境，可以先安装 Miniconda，再执行：

```bash
conda create -n llm_algo python=3.10 -y
conda activate llm_algo
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user \
  --name llm_algo \
  --display-name "Python (llm_algo)"
```

启动 JupyterLab 后，在 Notebook 中选择 `Python (llm_algo)` Kernel。若平台已经预装 CUDA PyTorch，不要在新环境中盲目重新安装 torch；先运行项目环境预检，再根据结果补齐普通依赖。

## 环境选择

| 场景 | 建议 |
|---|---|
| 先看内容 | 在线站点 |
| Part 0 / Part 1 | 在线 Notebook 或本地基础环境 |
| Part 2 | 本地 CPU-first / Colab CPU / CNB CPU |
| Part 3 / Part 4 | 本地 NVIDIA GPU / Colab GPU / CNB GPU |
| 团队统一交付 | CNB / Docker / 云端 GPU |

## Part 02 环境分层与决策树

Part 02 采用“基础包 + 平台 PyTorch + 项目能力包”的分层方式。依赖层和虚拟环境不是一一对应关系：Colab / ModelScope 可以在一个 runtime 中按需安装；本地或云端长期维护多个 CUDA / backend 版本时，再用 Conda 分开环境。

### 环境层与适用小节

| 环境层 | 主要内容 | 适合小节 | 是否需要 GPU |
|---|---|---|---|
| `base` | Jupyter、基础科学计算、`einops`、`transformers`、Notebook 测试工具 | Part 02 大多数机制节、60–64 的 CPU 路径、75 报告决策 | 否 |
| `torch-cpu` | CPU 版 PyTorch | CPU-first Notebook、答案区和正确性测试 | 否 |
| `torch-cu128` | CUDA 12.8 版 PyTorch | 60–65 真实训练、73 / 76 / 74 GPU 路径 | 是 |
| `fine-tuning` | `peft`、`accelerate`、`datasets` 等 | 60–64、LoRA / SFT 项目 | 通常不需要；真实训练需要 |
| `qlora` | `fine-tuning` + `bitsandbytes` | 65 QLoRA 选型 | 建议 |
| `reinforcement-learning` | `fine-tuning` + `trl` | 84–86、DPO / GRPO 等后训练项目 | 建议 |
| `distributed` | `accelerate`、分布式指标工具 | 79–81、多卡训练和推理项目 | 是 |
| `inference-vllm` | vLLM、backend 请求测试工具 | 66、68、69、70；71 的可选 backend 路径 | 是 |
| `inference-sglang` | SGLang、backend 请求测试工具 | 66、69、70；71 的可选 backend 对照路径 | 是 |
| `inference-quantized` | 与 GPTQ / AWQ / GGUF artifact 匹配的 backend | 67 量化推理与部署 | 是 |
| `profiling` | CUDA PyTorch 的 `torch.profiler`，可选 TensorBoard | 74 Profiling 收口 | GPU trace 需要 |

仓库已经提供以下可独立选择的依赖文件：

```text
requirements/base.txt             # Notebook 和通用 Python 包，不包含 torch
requirements/torch-cpu.txt        # CPU PyTorch
requirements/torch-cu128.txt      # CUDA 12.8 PyTorch
requirements/fine-tuning.txt      # 60–64：Transformers / PEFT / Datasets
requirements/qlora.txt            # 65：fine-tuning + bitsandbytes
requirements/reinforcement-learning.txt # 84–86：TRL 后训练环境
requirements/distributed.txt      # 79–81：分布式实验能力包
requirements/inference-vllm.txt   # 66、68–71 的 vLLM 路径
requirements/inference-sglang.txt # 66、69–71 的 SGLang 对照路径
requirements/profiling.txt        # 74：TensorBoard 等可视化工具
```

基础环境只安装通用包；CPU 和 CUDA PyTorch 必须二选一。Conda 文件负责创建 Python 和通用包环境，创建完成后再单独安装对应 PyTorch profile：

```bash
# CPU 环境
conda env create -f environment.yml
conda activate llm_algo
python -m pip install -r requirements/torch-cpu.txt

# CUDA 12.8 环境（主机必须有可用 NVIDIA 驱动）
conda env create -f environment-gpu.yml
conda activate llm_algo_gpu
python -m pip install -r requirements/torch-cu128.txt
```

已有 Conda 环境或 Colab / ModelScope runtime，则按需安装 profile：

```bash
# CPU-first Notebook
python -m pip install -r requirements/base.txt -r requirements/torch-cpu.txt

# 本机或云端没有 CUDA PyTorch 时，使用 CUDA 12.8
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/torch-cu128.txt

# 训练、QLoRA 按需追加
python -m pip install -r requirements/fine-tuning.txt
python -m pip install -r requirements/qlora.txt

# 后训练 / 强化学习项目
python -m pip install -r requirements/reinforcement-learning.txt

# 分布式项目；需要 CUDA PyTorch 和多 GPU 或分布式运行时
python -m pip install -r requirements/distributed.txt

# 推理 backend 二选一，并建议使用独立环境
python -m pip install -r requirements/inference-vllm.txt
# 或
python -m pip install -r requirements/inference-sglang.txt

# Profiling 工具按需追加
python -m pip install -r requirements/profiling.txt
```

不要同时安装 `torch-cpu.txt` 和 `torch-cu128.txt`。Colab、ModelScope 或云端容器如果已经提供 CUDA PyTorch，应先复用当前 Kernel；不要为了满足文件名而覆盖平台的 PyTorch。`requirements/torch-cu128.txt` 只提供用户态 CUDA PyTorch wheel，不能安装 NVIDIA 驱动，也不能解决驱动与 CUDA 版本不匹配。

Unsloth、LLaMA-Factory 和 TensorRT-LLM 属于额外工具或 backend，暂不放进通用 profile；使用它们时应创建独立环境或按其官方版本矩阵安装。SGLang 已提供独立 profile，但仍建议不要与 vLLM、训练框架混装。

### GPU 环境单独检查

GPU 项目至少有三层依赖：主机 NVIDIA 驱动、PyTorch CUDA wheel、项目能力包。`nvidia-smi` 显示的 CUDA 版本是驱动支持上限，`torch.version.cuda` 是 PyTorch wheel 携带的用户态 CUDA 版本，两者不是同一个字段。驱动必须不低于 wheel 的要求；GPU 架构还会影响 BF16、FP8 和特定 kernel 是否具备原生加速。

安装或复用 CUDA PyTorch 后，先运行：

```bash
nvidia-smi
python tools/environment_preflight.py --gpu --packages transformers
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('native_bf16:', torch.cuda.is_bf16_supported(including_emulation=False) if torch.cuda.is_available() else False)"
```

只有 `cuda_available=True` 且设备、显存和能力检查通过，才进入 60–71、73、74、76 的真实 GPU 实验。T4 等较旧 GPU 可能能创建 BF16 张量，但不代表 BF16 有原生 Tensor Core 加速；遇到这种设备，应把 FP16 作为性能对照，并在报告中记录“可用”与“原生加速”不是一回事。4090、5070 Ti 等设备的实际结果仍需按固定 workload 实测，不能由型号直接推断吞吐。

### Part 02 选择路径

```text
开始
  ├─ 只学习机制、运行答案区或 CPU 正确性？
  │    └─ base + torch-cpu
  ├─ 学习 LoRA / SFT 项目 60–64？
  │    └─ base + fine-tuning
  ├─ 学习 QLoRA / NF4 项目 65？
  │    └─ base + qlora + GPU（推荐）
  ├─ 学习真实推理 backend 66、68、69、70？
  │    └─ base + torch-cu128 + inference-vllm + GPU
  ├─ 学习 67 GPTQ / AWQ / GGUF 部署？
  │    └─ 先确认 artifact 和 backend，再安装匹配的量化环境
  ├─ 学习 73 / 76 训练显存项目？
  │    └─ base + torch-cu128 + transformers + GPU
  ├─ 学习 75 预算决策？
  │    └─ base，读取 73 / 76 的 JSON，不需要 GPU
  └─ 学习 74 真实 trace？
       └─ base + torch-cu128 + profiling + GPU
```

### 平台决策

- **Colab / ModelScope：** 通常不安装 Conda，使用平台 runtime；先预检，再补当前小节缺失的能力包。
- **云端容器：** 如果已有 CUDA PyTorch，优先复用；只有依赖冲突或需要长期保存环境时才安装 Miniconda。
- **本地单 GPU：** 可以使用一个 Conda 环境完成 60–65、73–76；运行 66–71 时若 vLLM 与训练依赖冲突，再增加独立推理环境。
- **本地多 CUDA / 多 backend：** 使用 `llm_algo_train`、`llm_algo_vllm`、`llm_algo_sglang` 等独立环境，不把 vLLM、SGLang 和 TensorRT-LLM 混装。

### 没有 Conda 时安装 Miniconda

Miniconda 只负责创建隔离环境，不负责安装驱动或提供 GPU。Linux 云端可按 Miniconda 官方安装包完成安装后执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n llm_algo python=3.10 -y
conda activate llm_algo
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user \
  --name llm_algo \
  --display-name "Python (llm_algo)"
```

随后再按目标小节安装 `base`、`fine-tuning` 或 `inference-vllm`。如果云端已经提供 Notebook Kernel 和 CUDA PyTorch，不必为了运行单个项目额外安装 Miniconda。

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

推理 backend 项目按本节要求追加 `vllm` 或 `sglang`；QLoRA 项目追加 `peft`、`bitsandbytes`；LoRA / SFT 项目追加 `accelerate`、`datasets`。普通依赖使用当前 Notebook 内核的 Python 安装。60 节会先检查缺失包；如果检测到云端 PEP 668 的 `EXTERNALLY-MANAGED` 标记，只对缺失的普通依赖自动追加 `--break-system-packages`，不会重装或替换 PyTorch：

```python
import sys
!{sys.executable} -m pip install -U transformers accelerate peft datasets
```

其他 Notebook 遇到 `externally-managed-environment` 时，不要切换到系统的另一个 Python；应使用当前 Kernel 的 `sys.executable`，或先创建独立虚拟环境。CUDA 相关组件仍然必须单独选择版本。

不要在预检失败时直接重装 PyTorch。若 `torch.version.cuda` 为 `None` 或 `cuda_available` 为 `False`，先确认 Colab 已启用 GPU，再重启 runtime；只有确认运行时仍缺少 CUDA wheel 时，才按平台说明选择对应的 PyTorch 安装命令。vLLM、SGLang 和 TensorRT-LLM 也必须先确认 CUDA、驱动和版本匹配后再安装。

预检通过后再下载模型、启动 backend 或执行训练。结果统一保存到 `benchmarks/results/`，并记录预检报告、模型、dtype、workload、设备、软件版本和实验状态。

每个需要额外环境的项目页，在实验开关附近只保留以下提示，并链接回本节：

> **运行提示：** 先查看[使用指南中的项目环境预检与安装说明](./guide.md#项目环境预检与安装)，再打开本节的真实实验开关。CPU-first 路径不要求 GPU；真实 GPU 或 backend 路径必须先通过预检。

## 最小规则

- Part 0 / Part 1：优先在线 Notebook 或本地基础环境。
- Part 2：默认 CPU-first，少数题再切 GPU。
- Part 3 / Part 4：完整验收需要 GPU；没有 GPU 时先阅读。
- CNB 的目标是统一交付，不是新增一套内容分层。
