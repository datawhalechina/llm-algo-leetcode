# 后训练优化实验 Quickstart

本页只负责创建、恢复和检查后训练实验环境。实验模型、数据、候选矩阵、结果路径和证据状态见[实验资产表](./experiment_assets.md)，Profile 与跨专题隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)，精确依赖由 Profile 注册表维护。以下命令均在仓库根目录执行。

| Profile | Conda 环境 | 适用页面 | 环境边界 |
|:---|:---|:---|:---|
| `PT-CPU` | `llm_pt_cpu` | 13–18、30–33 与 `PT-*` 机制页的框架和接口核对 | 只核对数据、目标函数与环境协议，不形成 GPU 性能结论 |
| `PT-SFT-CUDA` | `llm_pt_sft_cu128` | SFT、LoRA、长上下文和参数适配实验 | 不混装量化 Serving、vLLM 或分布式训练依赖 |
| `PT-RL-CUDA` | `llm_pt_rl_cu128` | 奖励模型、DPO、GRPO 和奖励信号实验 | 只提供单卡机制与有限更新环境，不替代多工作进程 RL 系统 |
| `PT-DELIVERY-CUDA` | `llm_pt_delivery_cu128` | Part 05 冻结评测、候选复测与交付决策 | 只消费已交接候选，不重新承担 SFT、DPO 或 GRPO 训练 |
| `Q-QLORA` | 由量化专题维护 | QLoRA 与低比特训练分支 | 不向 `PT-SFT-CUDA` 追加 bitsandbytes |

安装器会按 Profile 安装依赖、注册仓库运行时、执行依赖检查与环境预检，并注册同名 Jupyter Kernel。已有环境默认不会被覆盖；查看安装计划使用 `--dry-run`，只有确认旧环境属于同一 Profile 后才使用 `--reuse`。

<a id="pt-cpu-quickstart"></a>

## PT-CPU：机制核对

```bash
python tools/install_environment_profile.py \
  --profile PT-CPU \
  --env-name llm_pt_cpu
conda activate llm_pt_cpu
python -m jupyter lab
```

<a id="pt-sft-cuda-quickstart"></a>

## PT-SFT-CUDA：SFT 与 LoRA

```bash
python tools/install_environment_profile.py \
  --profile PT-SFT-CUDA \
  --env-name llm_pt_sft_cu128
conda activate llm_pt_sft_cu128
python -m jupyter lab
```

<a id="pt-rl-cuda-quickstart"></a>

## PT-RL-CUDA：偏好优化与 RL

```bash
python tools/install_environment_profile.py \
  --profile PT-RL-CUDA \
  --env-name llm_pt_rl_cu128
conda activate llm_pt_rl_cu128
python -m jupyter lab
```

该 Profile 的默认矩阵按 12 GiB 级单卡准备，环境预检使用 8 GiB 最低准入线。预检通过只表示软件和基础容量满足要求，不保证任意模型、长度或候选数都能运行；每页的实测峰值和结论仍以结果 JSON 与实验资产表为准。

<a id="pt-delivery-cuda-quickstart"></a>

## PT-DELIVERY-CUDA：综合交付评测

该环境属于 Part 05 后训练交付项目。它复用后训练模型加载栈，但使用独立环境身份，避免把机制实验的依赖状态和综合评测结果混为一谈。

```bash
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
python tools/install_environment_profile.py \
  --profile PT-DELIVERY-CUDA \
  --env-name llm_pt_delivery_cu128
conda activate llm_pt_delivery_cu128
python -m jupyter lab
```

项目的候选清单、冻结评测集、门槛和结果标签写入 Part 05 项目配置，不写入 requirements 或环境 Profile。

## 手动创建

安装器不可用时，按相同顺序手动创建。下表给出命令模板中的三个替换项，无需打开 Profile 注册表查找依赖：

| Profile | `<ENV_NAME>` | `<DEPENDENCY_ARGS>` |
|:---|:---|:---|
| `PT-CPU` | `llm_pt_cpu` | `-r requirements/torch-cpu.txt -r requirements/reinforcement-learning.txt -c requirements/post-training-cpu-constraints.txt` |
| `PT-SFT-CUDA` | `llm_pt_sft_cu128` | `-r requirements/torch-cu128.txt -r requirements/fine-tuning.txt` |
| `PT-RL-CUDA` | `llm_pt_rl_cu128` | `-r requirements/torch-cu128.txt -r requirements/fine-tuning.txt -c requirements/post-training-cu128-constraints.txt` |
| `PT-DELIVERY-CUDA` | `llm_pt_delivery_cu128` | `-r requirements/torch-cu128.txt -r requirements/post-training-delivery.txt -c requirements/post-training-cu128-constraints.txt` |

依次替换环境名、依赖参数和 Profile ID：

```bash
conda create -n <ENV_NAME> python=3.12 -y
conda env config vars set -n <ENV_NAME> PYTHONNOUSERSITE=1 PYTHONPATH=
conda activate <ENV_NAME>
python -m pip install --upgrade pip
python -m pip install <DEPENDENCY_ARGS>
python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile <PROFILE_ID>
python -m ipykernel install --user \
  --name <ENV_NAME> \
  --display-name "Python (<ENV_NAME>)"
```

设置环境隔离变量后应重新激活环境；命令中的覆盖提示是隔离动作，只有后续依赖检查或 Profile 预检失败时才视为安装失败。

## 下载中断或镜像 404 后续装

若第三方镜像出现 `.whl.metadata` 404、持续限速或下载中断，保留已创建环境，改用官方 PyPI 继续同一 Profile。pip 会复用已经安装的包和本地缓存：

```bash
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
python tools/install_environment_profile.py \
  --profile PT-RL-CUDA \
  --env-name llm_pt_rl_cu128 \
  --reuse
```

续装 `PT-SFT-CUDA` 时，将 Profile 和环境名替换为 `PT-SFT-CUDA`、`llm_pt_sft_cu128`。`--reuse` 不用于把量化、Serving 或其他 CUDA 栈混装进已有环境；环境身份不明或依赖已经冲突时应新建环境。

## 后续复测

CPU 机制环境：

```bash
conda activate llm_pt_cpu
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile PT-CPU
python -m jupyter lab
```

SFT / LoRA 环境：

```bash
conda activate llm_pt_sft_cu128
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile PT-SFT-CUDA
python -m jupyter lab
```

偏好优化 / RL 环境：

```bash
conda activate llm_pt_rl_cu128
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile PT-RL-CUDA
python -m jupyter lab
```

综合交付评测环境：

```bash
conda activate llm_pt_delivery_cu128
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile PT-DELIVERY-CUDA
python -m jupyter lab
```

完成门槛是：`pip check` 无冲突、Profile 预检返回 `status=ok`、`sys.executable` 指向目标 Conda 环境，并在 JupyterLab 中选择同名 Kernel。正式实验还需保持页面 5.1 的模型、数据和工作负载不变，将 `RUN_MODE` 改为 `real_gpu`，并为 `RUN_LABEL` 设置唯一标签。

## Q-QLORA：低比特训练交叉环境

QLoRA 的创建、预检与复测命令见[量化与压缩实验 Quickstart](../quantization_compression/quickstart.md#q-qlora-quickstart)。模型和适配器能否进入训练由后训练实验资产表记录，低比特依赖仍由量化专题维护。

## 常见异常

| 现象 | 处理方式 |
|:---|:---|
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Kernel |
| `jupyter` 或 `ipykernel` 缺失 | 续装当前 Profile；不要在用户级 Python 中单独安装后继续混用 Kernel |
| 数据下载受代理影响 | 保留 `httpx[socks]`，检查代理环境变量，不在 Notebook 写入凭据 |
| 显存不足 | 记录失败后只调整一个工作负载变量，例如长度、微批次或候选数 |
| 损失下降但独立评测未改善 | 检查标签、有效监督词元、数据切分和评测隔离，不直接采用候选 |
| 轨迹版本过旧 | 记录策略版本和新鲜度失败，不把旧轨迹静默混入当前更新 |
