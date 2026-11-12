# 大模型架构演进实验 Quickstart

首次运行按本页创建环境，重新进入环境或安装中断时使用对应恢复命令。模型、工作负载、结果路径、实际环境和硬件快照见[实验资产表](./experiment_assets.md)及结果 JSON；Profile 与隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)。以下命令均在仓库根目录执行。

<a id="arch-cuda-mechanism-quickstart"></a>

## ARCH-CUDA-MECHANISM：单卡架构机制环境

44、45、ARCH-MLA 与 ARCH-HYBRID-MEMORY 的机制 GPU 实验使用这一环境。需要真实模型快照、推理服务后端或多卡 MoE 时，切换到对应的环境配置（Profile）。

### 安装阶段

安装器按环境配置中登记的顺序完成以下阶段：

```text
创建 Conda 环境并启用路径隔离
→ requirements/torch-cu128.txt：PyTorch 与 CUDA 12.8 运行时
→ requirements/architecture.txt：Transformers、数据工具与专题依赖
  └─ requirements/base.txt：JupyterLab、ipykernel 与通用 Notebook 依赖
→ 注册仓库 tools 运行时
→ pip check 与 ARCH-CUDA-MECHANISM 预检
→ 注册同名 Jupyter 内核
```

日志停在某个 `pip install -r` 时仍处于依赖安装阶段；出现覆盖 `PYTHONPATH` 的警告表示安装器正在清除系统或 ROS 路径污染，不是安装失败。只有看到“环境安装与预检完成”，才表示以上阶段全部结束。

### 一键创建（推荐）

```bash
python tools/install_environment_profile.py \
  --profile ARCH-CUDA-MECHANISM \
  --env-name llm_arch_cu128

conda activate llm_arch_cu128
python -m jupyter lab
```

安装器会创建隔离环境、安装环境配置登记的依赖、注册仓库运行时、执行 `pip check` 和环境预检，并注册同名 Jupyter 内核。先检查计划时添加 `--dry-run`；依赖已经混装时创建新环境，不在旧环境逐包覆盖。

### 中断后续装

若环境已经创建，但在 PyTorch 或专题依赖安装期间被中断，先激活同一环境并从中断阶段续装，再让安装器完成注册、预检和 Jupyter 内核安装：

```bash
conda activate llm_arch_cu128

# 在 torch-cu128.txt 阶段中断时先执行这一行；已经完成可跳过。
python -m pip install -r requirements/torch-cu128.txt \
  --timeout 120 \
  --retries 8

# 在 architecture.txt 阶段中断时执行；它同时安装 base.txt 中的 Jupyter 依赖。
python -m pip install -r requirements/architecture.txt \
  --timeout 120 \
  --retries 8

python tools/install_environment_profile.py \
  --profile ARCH-CUDA-MECHANISM \
  --env-name llm_arch_cu128 \
  --reuse
```

只有确认 `llm_arch_cu128` 原本就是这一环境配置时才使用 `--reuse`。若 `pip check` 出现无法解释的跨栈冲突，改用新环境名重新创建，不在旧环境逐包覆盖。

### 手动创建

```bash
conda create -n llm_arch_cu128 python=3.12 -y
conda activate llm_arch_cu128

conda env config vars set PYTHONNOUSERSITE=1 PYTHONPATH=
conda deactivate
conda activate llm_arch_cu128

python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cu128.txt
python -m pip install -r requirements/architecture.txt

python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile ARCH-CUDA-MECHANISM

python -m ipykernel install --user \
  --name llm_arch_cu128 \
  --display-name "Python (llm_arch_cu128)"
python -m jupyter lab
```

只有 `pip check` 无冲突、环境预检返回 `status=ok`，且 Notebook 的 `sys.executable` 指向该环境，才能把页面的 `RUN_MODE` 从 `dry_run` 改为 `real_gpu`。

完成后在 Notebook 中选择 `Python (llm_arch_cu128)`，并执行以下最小核对：

```python
import sys
import torch

print(sys.executable)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

### 后续复测

```bash
conda activate llm_arch_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile ARCH-CUDA-MECHANISM
python -m jupyter lab
```

## 项目与推理服务环境

60、71、77、78、61 复用机制页的结构身份，但项目必须重新固定模型快照、分词器、输入、长度阶梯和执行栈。需要 vLLM、SGLang、专用稀疏计算内核或多卡 MoE 时使用独立环境配置；不要向 `ARCH-CUDA-MECHANISM` 追加项目后端依赖。

## 常见异常入口

| 现象 | 处理方式 |
|:---|:---|
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Jupyter 内核 |
| `pip check` 报跨环境冲突 | 停止实验，按本页环境配置创建新环境 |
| CUDA 或 PyTorch 预检失败 | 保持 `RUN_MODE=dry_run`，检查驱动与 cu128 构建兼容性 |
| 模型或数据下载失败 | 检查代理传输依赖和缓存目录，不在 Notebook 记录凭据 |
| 稀疏、MLA 或 SSM 路径不受支持 | 记录能力检查失败，不在回退后继续声明目标机制已执行 |
| 结果只有环境信息 | 该结果仍是预检，不是机制基准测试 |
