# 量化与压缩实验 Quickstart

本页只回答“如何创建或重新进入可运行环境”。模型、数据、候选矩阵、结果路径、实际环境和硬件快照见[实验资产表](./experiment_assets.md)及结果 JSON；精确 Profile 定义在注册表中维护，跨专题隔离关系见[环境资产索引](../../gpu_environment_assets.md)。以下命令均在仓库根目录执行。

<a id="q-cpu-quickstart"></a>

## Q-CPU：量化 CPU 框架核对环境

该环境运行机制页中标记为“框架一致性核对”的 CPU 单元，只比较配置映射、状态转换和小模块数值语义，不形成 GPU Kernel、显存或性能结论。

```bash
python tools/install_environment_profile.py \
  --profile Q-CPU \
  --env-name llm_quant_cpu

conda activate llm_quant_cpu
python tools/test_framework_conformance.py \
  --all-part02 \
  --profile Q-CPU \
  --require-passed
```

后续运行 Notebook 时选择 `Python (llm_quant_cpu)`。QAT 等只依赖 PyTorch 官方参考算子的核对仍使用 `BASE-CPU`，不要求切换到本环境。

<a id="q-cuda-mechanism-quickstart"></a>

## Q-CUDA-MECHANISM：量化机制环境

25、40、41、53、54、Mixed-bit 和 QAT 使用这一独立环境。它不安装 vLLM 或 SGLang，也不与 Serving 环境混用。

### 一键创建（推荐）

在仓库根目录执行以下命令。安装器读取 `requirements/environment-profiles.json`，依次创建环境、写入 Profile 要求的 `PYTHONNOUSERSITE` / `PYTHONPATH` 隔离变量、安装依赖、注册仓库公共运行时、执行 `pip check` 与 Profile 预检，并注册同名 Jupyter Kernel；已有环境默认不会被覆盖。

```bash
python tools/install_environment_profile.py \
  --profile Q-CUDA-MECHANISM \
  --env-name llm_quant_cu128

conda activate llm_quant_cu128
python -m jupyter lab
```

需要先检查安装计划时添加 `--dry-run`。只有明确确认已有环境仍属于同一 Profile 时才能添加 `--reuse`；出现依赖冲突时应创建新环境，而不是在原环境逐包覆盖。

### 手动创建

```bash
conda create -n llm_quant_cu128 python=3.12 -y
conda activate llm_quant_cu128

python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cu128.txt
python -m pip install -r requirements/quantization.txt

python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile Q-CUDA-MECHANISM

python -m ipykernel install --user \
  --name llm_quant_cu128 \
  --display-name "Python (llm_quant_cu128)"
python -m jupyter lab
```

`register_notebook_runtime.py` 会在当前环境写入只包含本仓库路径的 `.pth` 文件，使 VS Code 从 Notebook 所在目录启动 Kernel 时也能导入 `tools`。执行注册后必须重启 Kernel。Notebook 中选择 `Python (llm_quant_cu128)`，并用 `sys.executable` 确认解释器位于该 Conda 环境。只有 `pip check` 无冲突且 Profile 预检返回 `status=ok`，才把 `RUN_MODE` 改为 `real_gpu`。

### 后续复测

```bash
conda activate llm_quant_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile Q-CUDA-MECHANISM
python -m jupyter lab
```

2026-10-06 之前创建的同名环境可能缺少后续补入的量化依赖或仓库运行时注册。重新执行 `python -m pip install -r requirements/quantization.txt` 和 `python tools/register_notebook_runtime.py`，重启 Kernel 后再运行 `pip check` 和 Profile 预检；不要从 PyPI 单独安装同名占位包。

<a id="q-qlora-quickstart"></a>

## Q-QLORA：低比特训练环境

26、65 使用独立的 QLoRA 环境。不要在 `Q-CUDA-MECHANISM` 中追加 bitsandbytes，也不要与 Serving 后端混装。

### 首次创建

```bash
conda create -n llm_qlora_cu128 python=3.12 -y
conda activate llm_qlora_cu128

python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cu128.txt --extra-index-url https://pypi.org/simple
python -m pip install -r requirements/qlora.txt \
  -c requirements/qlora-cu128-constraints.txt

python -m pip check
python tools/environment_preflight.py --profile Q-QLORA

python -m ipykernel install --user \
  --name llm_qlora_cu128 \
  --display-name "Python (llm_qlora_cu128)"
python -m jupyter lab
```

### 后续复测

```bash
conda activate llm_qlora_cu128
python tools/environment_preflight.py --profile Q-QLORA
python -m jupyter lab
```

<a id="q-vllm-quickstart"></a>

## INF-VLLM-011-CU128：量化 Serving 环境

67、82、83 使用这一 Profile 完成产物加载、真实请求和 Serving profile。82 使用 40 已生成的本地候选包；新环境需先在 `Q-CUDA-MECHANISM` 中运行 40，再在 82 填入该次运行标签。vLLM 自行提供匹配的 PyTorch、xFormers 与 compressed-tensors；不要先安装 `torch-cu128.txt`，也不要向该环境追加 LLM Compressor、torchao、HQQ 或 SGLang。

### 一键创建（推荐）

```bash
python tools/install_environment_profile.py \
  --profile INF-VLLM-011-CU128 \
  --env-name llm_vllm_cu128

conda activate llm_vllm_cu128
python -m jupyter lab
```

### 手动创建

```bash
conda create -n llm_vllm_cu128 python=3.12 -y
conda activate llm_vllm_cu128

python -m pip install --upgrade pip
python -m pip install -r requirements/inference-vllm.txt
python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile INF-VLLM-011-CU128

python -m ipykernel install --user \
  --name llm_vllm_cu128 \
  --display-name "Python (llm_vllm_cu128)"
python -m jupyter lab
```

首次验证先确保 40 已生成并接受候选包，再运行 82。只有 loader 结果为 `loaded`、往返输出非空且候选包哈希通过，才将稳定引用交给 67；不要因为产物能被 Transformers 读取，就直接认定 vLLM 兼容。

## 常见异常入口

| 现象 | 处理入口 |
|:---|:---|
| `pip check` 报跨环境依赖冲突 | 停止实验，按对应 requirements 和 constraints 重建独立环境 |
| Notebook 使用了错误解释器 | 重新选择本页登记的 ipykernel，并检查 `sys.executable` |
| Notebook 报 `No module named 'tools'` | 在仓库根目录执行 `python tools/register_notebook_runtime.py`，随后重启当前 Kernel |
| ROS、系统 Python 或用户目录污染 | 查看[环境隔离故障处理](../../gpu_environment_assets.md#environment-isolation-troubleshooting) |
| Hugging Face 下载在 SOCKS 代理下失败 | 重新安装对应 requirements；不要在 Notebook 中记录代理地址或凭据 |
| CUDA、计算能力或低精度路径预检失败 | 保持 `RUN_MODE=dry_run`，将失败状态写入结果，不绕过硬件 gate |
