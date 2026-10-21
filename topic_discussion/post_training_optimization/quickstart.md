# 后训练优化实验 Quickstart

本页用于选择、创建、恢复和检查后训练实验环境。模型、数据、工作负载和结果状态见[实验资产表](./experiment_assets.md)；Profile 的版本约束和隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)。以下命令均在仓库根目录执行。

## 1. 选择环境

| Profile | Conda 环境 | 使用场景 | 运行条件 |
|:---|:---|:---|:---|
| `PT-CPU` | `llm_pt_cpu` | 13–18、30–33 与 `PT-*` 页的框架一致性核对 | CPU |
| `PT-SFT-CUDA` | `llm_pt_sft_cu128` | SFT、LoRA、长上下文与参数适配 | 单张 CUDA GPU，至少 8 GiB |
| `PT-RL-CUDA` | `llm_pt_rl_cu128` | 奖励模型、DPO、GRPO 与奖励信号 | 单张 CUDA GPU，至少 8 GiB |
| `PT-VERL-CUDA` | 上游 uv 锁环境 | 轨迹生命周期与生成—更新系统机制 | 单张 CUDA GPU，至少 10 GiB；驱动支持 CUDA 13.0 |
| `PT-DISTILL-CUDA` | `llm_pt_distill_cu128` | 固定轨迹 KD 与在线蒸馏 | 单张 CUDA GPU，至少 10 GiB |
| `PT-DELIVERY-CUDA` | `llm_pt_delivery_cu128` | Part 05 冻结评测与交付决策 | 单张 CUDA GPU，至少 8 GiB |
| [`Q-QLORA`](../quantization_compression/quickstart.md#q-qlora-quickstart) | 由量化专题维护 | QLoRA 与低比特训练 | 使用独立量化环境 |

六个 `PT-*` Profile 彼此隔离。`PT-VERL-CUDA` 使用 verl 自带的 uv 锁文件；其余 Profile 使用仓库的 Conda 安装器。

## 2. 首次创建

先在当前终端指定下载源和容错参数。PyTorch 的 CPU 或 CUDA wheel 来源仍由对应 requirements 文件固定；这里的设置主要用于其余 PyPI 依赖，可避开本机已有镜像的限速或 `.whl.metadata` 404。

```bash
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=8
```

安装器会依次创建环境、安装登记依赖、注册仓库运行时、执行 `pip check` 和 Profile 预检，并注册同名 Jupyter Kernel。查看计划时在命令末尾添加 `--dry-run`；已有环境默认不会被修改。

<a id="pt-cpu-quickstart"></a>

### PT-CPU

```bash
python tools/install_environment_profile.py \
  --profile PT-CPU \
  --env-name llm_pt_cpu
```

<a id="pt-sft-cuda-quickstart"></a>

### PT-SFT-CUDA

```bash
python tools/install_environment_profile.py \
  --profile PT-SFT-CUDA \
  --env-name llm_pt_sft_cu128
```

<a id="pt-rl-cuda-quickstart"></a>

### PT-RL-CUDA

```bash
python tools/install_environment_profile.py \
  --profile PT-RL-CUDA \
  --env-name llm_pt_rl_cu128
```

<a id="pt-verl-cuda-quickstart"></a>

### PT-VERL-CUDA

verl 的 Torch、vLLM 与 Ray 版本由上游锁文件共同约束，不写入 `PT-RL-CUDA`。首次准备时执行：

```bash
git clone --branch v0.9.1 --depth 1 \
  https://github.com/verl-project/verl.git ~/verl-v0.9.1
cd ~/verl-v0.9.1
git rev-parse HEAD
uv sync --frozen --all-packages --extra vllm --extra fsdp
uv run --frozen --all-packages --extra vllm --extra fsdp \
  python3 examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

`git rev-parse HEAD` 应输出 `1876b06d0a3e4e71e06230be10af14492ca8a75b`。该环境由 Notebook 通过独立进程调用，无需注册为 Jupyter Kernel。

<a id="pt-distill-cuda-quickstart"></a>

### PT-DISTILL-CUDA

```bash
python tools/install_environment_profile.py \
  --profile PT-DISTILL-CUDA \
  --env-name llm_pt_distill_cu128
```

<a id="pt-delivery-cuda-quickstart"></a>

### PT-DELIVERY-CUDA

```bash
python tools/install_environment_profile.py \
  --profile PT-DELIVERY-CUDA \
  --env-name llm_pt_delivery_cu128
```

安装完成后激活刚创建的环境并启动 JupyterLab：

```bash
ENV_NAME=llm_pt_cpu  # 改为刚创建的环境名
conda activate "$ENV_NAME"
python -m jupyter lab
```

## 3. 中断续装与重新进入

下载或安装中断后保留已有环境，先从上表填写当前 Profile 和环境名，再续装：

```bash
PROFILE_ID=PT-CPU
ENV_NAME=llm_pt_cpu

export PIP_INDEX_URL=https://pypi.org/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=8

python tools/install_environment_profile.py \
  --profile "$PROFILE_ID" \
  --env-name "$ENV_NAME" \
  --reuse
```

`--reuse` 只用于已经确认属于同一 Profile 的环境。环境身份不明或 `pip check` 已有冲突时，改用新的环境名重新创建。

环境已经安装完成时无需重新运行安装器：

```bash
ENV_NAME=llm_pt_cpu
PROFILE_ID=PT-CPU

conda activate "$ENV_NAME"
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile "$PROFILE_ID"
python -m jupyter lab
```

## 4. 手动创建

安装器不可用时，从下表选择依赖参数，并保持与 Profile 相同的安装顺序。

| Profile | 环境名 | 依赖参数数组 |
|:---|:---|:---|
| `PT-CPU` | `llm_pt_cpu` | `(-r requirements/torch-cpu.txt -r requirements/reinforcement-learning.txt -c requirements/post-training-cpu-constraints.txt)` |
| `PT-SFT-CUDA` | `llm_pt_sft_cu128` | `(-r requirements/torch-cu128.txt -r requirements/fine-tuning.txt)` |
| `PT-RL-CUDA` | `llm_pt_rl_cu128` | `(-r requirements/torch-cu128.txt -r requirements/fine-tuning.txt -c requirements/post-training-cu128-constraints.txt)` |
| `PT-VERL-CUDA` | 上游 uv 锁环境 | 不适用；按 [PT-VERL-CUDA](#pt-verl-cuda-quickstart) 固定 tag、commit 与 extras |
| `PT-DISTILL-CUDA` | `llm_pt_distill_cu128` | `(-r requirements/torch-cu128.txt -r requirements/distillation.txt -c requirements/distillation-cu128-constraints.txt)` |
| `PT-DELIVERY-CUDA` | `llm_pt_delivery_cu128` | `(-r requirements/torch-cu128.txt -r requirements/post-training-delivery.txt -c requirements/post-training-cu128-constraints.txt)` |

```bash
PROFILE_ID=PT-CPU
ENV_NAME=llm_pt_cpu
DEPENDENCY_ARGS=(-r requirements/torch-cpu.txt -r requirements/reinforcement-learning.txt -c requirements/post-training-cpu-constraints.txt)

export PIP_INDEX_URL=https://pypi.org/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=8

conda create -n "$ENV_NAME" python=3.12 -y
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 PYTHONPATH=
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install "${DEPENDENCY_ARGS[@]}"
python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile "$PROFILE_ID"
python -m ipykernel install --user \
  --name "$ENV_NAME" \
  --display-name "Python ($ENV_NAME)"
```

设置环境隔离变量后出现覆盖 `PYTHONPATH` 的提示属于预期行为。安装结束后重新激活环境，再执行检查和 Kernel 注册。

## 5. 验证与故障处理

环境可用需要同时满足以下条件：

- `python -m pip check` 没有依赖冲突；
- Profile 预检返回 `status=ok`；
- `sys.executable` 指向所选 Conda 环境；
- JupyterLab 使用同名 Kernel。

`PT-CPU` 安装完成后可批量运行框架一致性核对：

```bash
conda activate llm_pt_cpu
python tools/test_framework_conformance.py \
  --all-part02 \
  --profile PT-CPU \
  --require-passed
```

| 现象 | 处理方式 |
|:---|:---|
| 下载速度持续受限或镜像返回 404 | 在当前终端设置官方 `PIP_INDEX_URL`、超时和重试参数，再用同一 Profile 执行 `--reuse` |
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，然后重启 Kernel |
| `jupyter`、`ipykernel` 或专题依赖缺失 | 安装尚未完成；对同一 Profile 执行 `--reuse` |
| `pip check` 报跨环境冲突 | 停止使用该环境，以新环境名重新创建 |
| CUDA 或显存预检失败 | 保持实验的安全默认阶段，检查驱动、CUDA 构建和页面 5.1 的工作负载 |
| 模型或数据下载失败 | 检查 Hub、代理和缓存；不要在 Notebook 中临时安装依赖或写入凭据 |
| 预检通过但真实阶段显存不足 | 只调整页面允许覆盖的一项工作负载，并使用新的 `RUN_LABEL` 保存结果 |
