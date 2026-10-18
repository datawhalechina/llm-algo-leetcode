# 性能分析实验 Quickstart

本页只回答如何创建或重新进入单卡性能机制环境。工作负载、结果路径、实际版本、硬件快照和证据状态见[实验资产表](./experiment_assets.md)及结果 JSON；Profile 与隔离关系见[环境资产索引](../../gpu_environment_assets.md#perf-cuda-mechanism)。以下命令均在仓库根目录执行。

<a id="perf-cuda-mechanism-quickstart"></a>

## PERF-CUDA-MECHANISM：单卡性能机制环境

`PERF-TRACE`、`PERF-MEMORY-TIMELINE`、`PERF-ROOFLINE`、`PERF-REGRESSION` 以及性能优化交叉页 `PERF-INPUT`、`PERF-IO` 使用这一环境。它不安装 vLLM、SGLang、LLM Compressor 或 DeepSpeed，也不承担多卡通信和生产服务压测。

### 一键创建（推荐）

```bash
python tools/install_environment_profile.py \
  --profile PERF-CUDA-MECHANISM \
  --env-name llm_perf_cu128

conda activate llm_perf_cu128
python tools/register_notebook_runtime.py --check
python -m ipykernel install --user \
  --name llm_perf_cu128 \
  --display-name "Python (llm_perf_cu128)"
python -m jupyter lab
```

安装器会创建隔离环境、安装 `requirements/torch-cu128.txt` 与 `requirements/profiling.txt`、注册仓库运行时，并执行 `pip check` 和环境预检。上方的 `ipykernel install` 命令会明确登记同名 Jupyter 内核。检查安装计划时添加 `--dry-run`；已有环境依赖混装时创建新环境，不逐包覆盖。

### 手动创建

```bash
conda create -n llm_perf_cu128 python=3.12 -y
conda activate llm_perf_cu128

conda env config vars set PYTHONNOUSERSITE=1 PYTHONPATH=
conda deactivate
conda activate llm_perf_cu128

python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cu128.txt \
  --timeout 120 \
  --retries 8
python -m pip install -r requirements/profiling.txt

python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile PERF-CUDA-MECHANISM

python -m ipykernel install --user \
  --name llm_perf_cu128 \
  --display-name "Python (llm_perf_cu128)"
python -m jupyter lab
```

只有 `pip check` 无冲突、环境预检返回 `status=ok`，且 Notebook 的 `sys.executable` 指向该环境，才能把 `RUN_MODE` 从 `dry_run` 改为 `real_gpu`。

### 后续复测

```bash
conda activate llm_perf_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile PERF-CUDA-MECHANISM
python -m jupyter lab
```

## 系统级工具边界

PyTorch Profiler、CUDA Event 和显存 API 由本环境直接承载。Nsight Systems、Nsight Compute、NCCL 执行轨迹和服务侧监控属于外部工具或其他环境；它们生成的结果可以登记为证据，但不向本环境追加驱动工具、多卡框架或服务后端依赖。

## 常见异常入口

| 现象 | 处理方式 |
|:---|:---|
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，然后重启 Jupyter 内核 |
| `pip check` 报跨环境冲突 | 停止实验，使用新环境名重新执行一键创建 |
| CUDA 或 PyTorch 预检失败 | 保持 `RUN_MODE=dry_run`，检查驱动和 cu128 构建兼容性 |
| 结果只有环境信息 | 当前只完成预检，不是性能机制证据 |
| 执行轨迹为空或缺少 CUDA 事件 | 检查活动类型、预热、同步与正式采样步数 |
| Nsight 或多卡工具不可用 | 保留能力检查失败，不用框架级结果冒充系统级证据 |
