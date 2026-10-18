# 部署与异构实验 Quickstart

部署专题的单卡真实后端实验复用 `INF-VLLM-011-CU128`。该环境用于 Part 02 · 66、67、72、82、83 的加载、请求和服务基准；它不用于生成量化产物，也不承担多卡通信实验。vLLM 提供与其匹配的 PyTorch 栈，不要先安装 `requirements/torch-cu128.txt`，也不要把 SGLang、torchao 或量化训练依赖追加到同一环境。

## 一键创建

在仓库根目录执行：

```bash
PIP_INDEX_URL=https://pypi.org/simple \
PIP_DEFAULT_TIMEOUT=120 \
PIP_RETRIES=8 \
python tools/install_environment_profile.py \
  --profile INF-VLLM-011-CU128 \
  --env-name llm_deploy_vllm_cu128

conda activate llm_deploy_vllm_cu128
python -m jupyter lab
```

`--dry-run` 只显示 Profile、requirements 和 constraints，不创建或修改环境：

```bash
python tools/install_environment_profile.py \
  --profile INF-VLLM-011-CU128 \
  --env-name llm_deploy_vllm_cu128 \
  --dry-run
```

## 手动创建

需要逐阶段诊断安装时，使用与 Profile 相同的依赖顺序：

```bash
conda create -n llm_deploy_vllm_cu128 python=3.12 -y
conda env config vars set -n llm_deploy_vllm_cu128 \
  PYTHONNOUSERSITE=1 PYTHONPATH=
conda activate llm_deploy_vllm_cu128

PIP_INDEX_URL=https://pypi.org/simple \
PIP_DEFAULT_TIMEOUT=120 \
PIP_RETRIES=8 \
python -m pip install -r requirements/inference-vllm.txt \
  -c requirements/inference-vllm-cu128-constraints.txt

python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile INF-VLLM-011-CU128
python -m ipykernel install --user \
  --name llm_deploy_vllm_cu128 \
  --display-name "Python (llm_deploy_vllm_cu128)"
```

`requirements/inference-vllm.txt` 已经引入基础依赖、JupyterLab 和 ipykernel。手动安装时不需要再单独安装 Jupyter。

## 安装中断后继续

如果环境已经创建且确认属于同一 Profile，保留 pip 与 Hugging Face 缓存，使用 `--reuse` 继续：

```bash
PIP_INDEX_URL=https://pypi.org/simple \
PIP_DEFAULT_TIMEOUT=120 \
PIP_RETRIES=8 \
python tools/install_environment_profile.py \
  --profile INF-VLLM-011-CU128 \
  --env-name llm_deploy_vllm_cu128 \
  --reuse
```

安装器写入 `PYTHONNOUSERSITE=1` 并清空 `PYTHONPATH` 时出现覆盖环境变量提示，表示正在隔离用户包和外部 Python 路径；只有后续预检失败时才把它视为问题。环境身份不明、依赖已经被其他后端覆盖或 `pip check` 无法通过时，新建环境，不在旧环境逐包替换。

## 后续复测

重新进入环境后先检查解释器、仓库运行时和 Profile：

```bash
conda activate llm_deploy_vllm_cu128
which python
python --version
python -m pip check
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile INF-VLLM-011-CU128
python -m jupyter lab
```

Notebook 中选择 `Python (llm_deploy_vllm_cu128)`。如果刚执行过仓库运行时注册，需要重启 Kernel 后再导入 `tools`。

## 完成门槛

环境只有同时满足以下条件才可进入真实后端实验：

- `python -m pip check` 没有依赖冲突；
- Profile 预检返回 `status=ok` 和 `ready=true`；
- `sys.executable` 位于 `llm_deploy_vllm_cu128`；
- CUDA 可用，实际 PyTorch / CUDA / vLLM 版本符合 Profile；
- Notebook 能从子目录导入仓库 `tools`；
- GPU 没有被另一份模型服务或 Notebook 占用。

## 常见异常

| 现象 | 先检查什么 | 处理方式 |
|:---|:---|:---|
| 下载速度低或镜像返回 404 | 当前索引是否同步了目标 wheel | 停止当前安装，保留缓存，改用可访问索引后执行 `--reuse`；不要并行启动多个 pip 进程 |
| `No module named jupyter` | 是否只装了部分依赖，安装器是否完成 | 继续安装 `requirements/inference-vllm.txt`，不要在环境外补装 |
| `tools` 无法导入 | 仓库运行时注册与 Kernel 是否一致 | 执行注册脚本，重启 Kernel，再检查 `sys.executable` |
| vLLM 启动时显存不足 | 空闲显存、模型进程和 `gpu_memory_utilization` | 先停止其他 GPU 进程；工作负载参数在实验资产和项目页中调整 |
| 模型能加载但目标低精度路径未生效 | loader、实际数据类型、计算内核与回退日志 | 将结果标记为功能路径，不形成低精度性能结论 |
| 需要 SGLang、llama.cpp 或多卡环境 | 是否已有独立 Profile | 使用独立环境和结果标签；不要向本环境追加另一套后端 |

具体模型、工作负载、项目顺序和结果路径见[实验资产表](./experiment_assets.md)。
