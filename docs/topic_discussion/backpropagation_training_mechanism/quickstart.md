# 反向传播与训练机制实验 Quickstart

本页维护环境创建、重新进入和共性故障处理。页面、工作负载与证据状态见[实验资产表](./experiment_assets.md)，全局 Profile 约束见[模型、执行栈与环境资产表](../../gpu_environment_assets.md)。以下命令均在仓库根目录执行。

<a id="train-cpu-quickstart"></a>

## TRAIN-CPU：CPU 框架核对环境

基础题目与答案测试使用 `BASE-CPU`；需要运行 PyTorch、Transformers 或 Accelerate 一致性核对时使用 `TRAIN-CPU`。该环境不形成 GPU 性能证据。

```bash
python tools/install_environment_profile.py \
  --profile TRAIN-CPU \
  --env-name llm_train_cpu

conda activate llm_train_cpu
python tools/register_notebook_runtime.py --check
python -m ipykernel install --user \
  --name llm_train_cpu \
  --display-name "Python (llm_train_cpu)"
python -m jupyter lab
```

批量核对 2.3 中已接入成熟框架的 10 个页面：

```bash
python tools/test_framework_conformance.py \
  --all-part02 \
  --profile TRAIN-CPU \
  --require-passed
```

<a id="train-cuda-mechanism-quickstart"></a>

## TRAIN-CUDA-MECHANISM：单卡训练机制环境

梯度累积、混合精度反向、激活检查点和激活卸载使用这一环境。分布式训练、完整 SFT/偏好优化和 Serving 使用独立 Profile。

| 项目 | 固定条件 |
|:---|:---|
| 覆盖页面 | Part 02 · 11、混合精度反向、Part 02 · 19、Part 02 · 42 |
| 系统与硬件 | Linux；1 张 CUDA GPU；Profile 最低参考显存 8 GiB；默认矩阵面向 12 GB 级 GPU；建议主机内存至少 16 GiB |
| 驱动与 CUDA | 主机 NVIDIA 驱动需支持 CUDA 12.8；驱动不由 pip 安装 |
| Python 与 PyTorch | Python 3.11 或 3.12；PyTorch `2.11.0+cu128` |
| 依赖入口 | 依次安装 `requirements/torch-cu128.txt` 与 `requirements/training-mechanism.txt` |
| 模型与输入 | 四页共用 `Qwen/Qwen2.5-0.5B`；首次运行下载模型和 tokenizer；固定文本构造训练批次，不下载外部数据集 |
| 环境隔离 | 不与 vLLM、SGLang、LLM Compressor 或 DeepSpeed 混装 |
| 运行入口 | 从仓库根目录启动 JupyterLab，并选择本环境注册的 Kernel |
| 成功门槛 | `pip check` 无冲突、Profile 返回 `status=ok`、Notebook 中 CUDA 可用 |

FP16 是默认低精度候选；只有硬件与 PyTorch 均报告支持时才选择 BF16。安装所需磁盘主要来自 PyTorch CUDA wheel、运行库与模型缓存，创建环境前应为 Conda 环境和 Hugging Face 缓存保留足够空间。模型下载完成后，可在各页将 `LOCAL_FILES_ONLY` 改为 `True` 进行离线复测。

Profile 预检通过只表示环境具备运行条件，不保证每页默认工作负载都能放入显存。8 GiB 环境应按页面 5.1 的顺序缩短序列长度或缩小批次，并使用新的运行标签；不能把缩小后的结果与默认矩阵直接合并。

### 一键创建（推荐）

```bash
python tools/install_environment_profile.py \
  --profile TRAIN-CUDA-MECHANISM \
  --env-name llm_train_mechanism_cu128

conda activate llm_train_mechanism_cu128
python tools/register_notebook_runtime.py --check
python -m ipykernel install --user \\
  --name llm_train_mechanism_cu128 \\
  --display-name "Python (llm_train_mechanism_cu128)"
python -m jupyter lab
```

安装器会创建隔离环境、安装依赖、注册仓库运行时，并执行 `pip check` 和 Profile 预检。上方的 `ipykernel install` 命令会明确登记同名 Jupyter Kernel。仅检查安装计划时添加 `--dry-run`；已有环境发生依赖混装时创建新环境，不逐包覆盖。

### 手动创建

```bash
conda create -n llm_train_mechanism_cu128 python=3.12 -y
conda activate llm_train_mechanism_cu128

conda env config vars set PYTHONNOUSERSITE=1 PYTHONPATH=
conda deactivate
conda activate llm_train_mechanism_cu128

python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cu128.txt
python -m pip install -r requirements/training-mechanism.txt

python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile TRAIN-CUDA-MECHANISM

python -m ipykernel install --user \
  --name llm_train_mechanism_cu128 \
  --display-name "Python (llm_train_mechanism_cu128)"
python -m jupyter lab
```

只有 `pip check` 无冲突、Profile 预检返回 `status=ok`，且 Notebook 的 `sys.executable` 指向当前环境，才能把页面的 `RUN_MODE` 从 `dry_run` 改为 `real_gpu`。

首次执行 `real_gpu` 需要访问 Hugging Face Hub。使用 SOCKS 代理时，专题依赖已安装 `socksio`；下载仍失败时先检查代理变量和 Hub 连通性，不要在 Notebook 中临时安装依赖。四页会把请求的 revision 解析为实际 commit，并将 commit 写入结果 JSON；同一轮比较必须使用同一 commit。

### 首次验证顺序

按“数值路径先行、状态策略随后”的顺序运行：

1. 混合精度反向：先验证 dtype、有限值、loss scale 与跳步语义；
2. Part 02 · 11 梯度累积：再验证有效批次、更新一致性和显存—时间交换；
3. Part 02 · 19 激活检查点：比较保存与重算；
4. Part 02 · 42 激活卸载：最后引入 CPU—GPU 搬运。

### 后续复测

```bash
conda activate llm_train_mechanism_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile TRAIN-CUDA-MECHANISM
python -m jupyter lab
```

## 常见异常入口

| 现象 | 处理方式 |
|:---|:---|
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Kernel |
| 已创建环境但缺少 `torch`、`jupyter` 或 `ipykernel` | 一键安装曾中断；激活环境后依次重跑 `requirements/torch-cu128.txt` 与 `requirements/training-mechanism.txt`，再执行注册和预检 |
| `pip check` 报跨环境冲突 | 停止实验，按本页 Profile 创建新环境 |
| CUDA 或 PyTorch 预检失败 | 保持 `RUN_MODE=dry_run`，检查驱动与 cu128 构建兼容性 |
| 模型或 tokenizer 下载失败 | 检查 Hub、代理和缓存目录；缓存完整后可设 `LOCAL_FILES_ONLY=True` |
| 候选使用了不同模型 commit | 停止比较，清理该运行标签并让所有候选重新加载同一解析 commit |
| FP16 出现非有限梯度 | 记录 loss scale、跳过更新和 failure；不要把跳步样本计入普通更新 |
| checkpoint 或 offload 结果不一致 | 先检查随机状态、原地修改、保存张量和梯度比较口径 |
| 只有理论显存账本 | 该结果仍是 CPU 机制证据，不是 GPU 峰值证据 |
