# 后训练优化实验 Quickstart

本页用于选择、创建、恢复和检查后训练实验环境。模型、数据、工作负载和结果状态见[实验资产表](./experiment_assets.md)；Profile 的版本约束和隔离关系见[环境资产索引](../../gpu_environment_assets.md)。以下命令均在仓库根目录执行。

## 1. 选择环境

| Profile | 环境 | 使用场景 | 运行条件 |
|:---|:---|:---|:---|
| `PT-CPU` | `llm_pt_cpu` | 13–18、30–33、50–51、相关 `PT-*` 与 DISTILL-OPD 的框架一致性核对 | CPU |
| `PT-SFT-CUDA` | `llm_pt_sft_cu128` | SFT、LoRA、长上下文与参数适配 | 单张 CUDA GPU，至少 8 GiB |
| `PT-RL-CUDA` | `llm_pt_rl_cu128` | 奖励模型、DPO、GRPO 与奖励信号 | 单张 CUDA GPU，至少 8 GiB |
| `PT-VERL-CUDA` | 上游 uv 锁环境 | 轨迹生命周期与生成—更新系统机制 | 单张 CUDA GPU，至少 10 GiB；驱动支持 CUDA 13.0 |
| `PT-DISTILL-CUDA` | `llm_pt_distill_cu128` | 固定轨迹 KD 与在线蒸馏 | 单张 CUDA GPU，至少 10 GiB |
| `PT-DELIVERY-CUDA` | `llm_pt_delivery_cu128` | Part 05 冻结评测与交付决策 | 单张 CUDA GPU，至少 8 GiB |
| [`Q-QLORA`](../quantization_compression/quickstart.md#q-qlora-quickstart) | 由量化专题维护 | QLoRA 与低比特训练 | 使用独立量化环境 |

六个 `PT-*` Profile 彼此隔离。安装器根据 Profile 选择 Conda 或上游 uv 锁环境。

## 2. 首次创建

选择环境后，复制对应的一条安装器命令即可。若 pip 当前使用的镜像限速或返回 `.whl.metadata` 404，可先在本终端设置下列下载参数；切换来源不保证网络速度。PyTorch wheel 来源仍由 requirements 固定，`PT-VERL-CUDA` 的 uv 锁环境遵循上游锁文件，不由这些 pip 变量切换来源。

```bash
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=8
```

安装器会创建并预检登记环境。Conda Profile 同时注册 Jupyter Kernel；`PT-VERL-CUDA` 由 Notebook 通过独立进程调用。查看计划时在命令末尾添加 `--dry-run`；已有环境默认不会被修改。

<a id="pt-cpu-quickstart"></a>

### PT-CPU

创建用于后训练 CPU 机制练习与框架一致性核对的 Conda 环境：

```bash
python tools/install_environment_profile.py \
  --profile PT-CPU \
  --env-name llm_pt_cpu
```

<a id="pt-sft-cuda-quickstart"></a>

### PT-SFT-CUDA

创建用于单卡 SFT、LoRA 和长上下文实验的 Conda 环境：

```bash
python tools/install_environment_profile.py \
  --profile PT-SFT-CUDA \
  --env-name llm_pt_sft_cu128
```

<a id="pt-rl-cuda-quickstart"></a>

### PT-RL-CUDA

创建用于单卡奖励模型、奖励信号、DPO 和 GRPO 实验的 Conda 环境：

```bash
python tools/install_environment_profile.py \
  --profile PT-RL-CUDA \
  --env-name llm_pt_rl_cu128
```

<a id="pt-verl-cuda-quickstart"></a>

### PT-VERL-CUDA

`PT-VERL-CUDA` 使用独立的 uv 锁环境，Torch、vLLM 与 Ray 由上游锁文件共同约束。安装器核对固定提交、依赖和 GPU；Notebook 通过独立子进程调用它，无需注册 Jupyter Kernel。已登记的 NumPy 元数据例外会单独提示，依赖检查仍须报告该例外及其他冲突。

该环境的受控更新使用仓库登记的固定算术轨迹；模型缓存、输入准备与历史 GSM8K 筛查见[实验资产表](./experiment_assets.md#结果版本与复现)。使用下面的命令创建环境：

```bash
python tools/install_environment_profile.py \
  --profile PT-VERL-CUDA \
  --target ~/verl-v0.9.1
```

<a id="pt-distill-cuda-quickstart"></a>

### PT-DISTILL-CUDA

创建用于固定轨迹与学生在策略蒸馏实验的 Conda 环境：

```bash
python tools/install_environment_profile.py \
  --profile PT-DISTILL-CUDA \
  --env-name llm_pt_distill_cu128
```

<a id="pt-delivery-cuda-quickstart"></a>

### PT-DELIVERY-CUDA

创建用于 Part 05 后训练候选冻结评测与交付判断的 Conda 环境：

```bash
python tools/install_environment_profile.py \
  --profile PT-DELIVERY-CUDA \
  --env-name llm_pt_delivery_cu128
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

`PT-VERL-CUDA` 续装仍使用目录参数：`python tools/install_environment_profile.py --profile PT-VERL-CUDA --target ~/verl-v0.9.1 --reuse`。安装完成后无需重新同步依赖；需要进入该环境时运行 `source ~/verl-v0.9.1/.venv/bin/activate`。

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

## 4. 手动创建（可选）

<details>
<summary>安装器不可用时，按同一 Profile 手动创建</summary>

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

</details>

## 5. 验证与常见问题

Conda 环境可用需要同时满足以下条件；`PT-VERL-CUDA` 则由安装器核对上游提交、锁依赖及 GPU，并由 Notebook 使用独立子进程调用，无需同名 Kernel。已登记的锁文件元数据例外必须明确报告，不能称作依赖检查无冲突。

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
