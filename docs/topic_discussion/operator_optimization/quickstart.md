# 算子优化实验 Quickstart

以下命令在仓库根目录执行，用于创建、恢复和检查环境。实验输入与结果见[实验资产表](./experiment_assets.md)，隔离与跨专题入口见[环境资产索引](../../gpu_environment_assets.md)。Notebook应选择与目标环境一致的解释器。

## 选择环境

| 环境身份 | 环境名或入口 | 用途与最低条件 |
|:---|:---|:---|
| [主GPU环境](#triton-environment)，Conda YAML管理 | `llm_algo_cnb_gpu` | Part03 Triton、Part04原生CUDA与单GPU Runtime；需要兼容的NVIDIA GPU，原生CUDA另检查本机工具链 |
| [BASE-CPU](../../guide.md#base-cpu) | `llm_algo_cpu` | Part01/02 CPU参考及Part04标准库推导；无需GPU，不运行Triton/CUDA内核 |
| 按需：`INF-CUDA-MECHANISM` | [推理机制环境](../inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart) | Part02·20注意力实测；不为Part03 Triton练习额外安装 |
| 按需：`PERF-CUDA-MECHANISM` | [性能机制环境](../performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart) | PERF-ROOFLINE、Part05·05；已有该环境时也可运行兼容的Part04单GPU Runtime |
| 按需：`PAR-CPU` | [通信CPU环境](../communication_parallelism/quickstart.md#par-cpu-quickstart) | Part04·13的CPU/Gloo集合通信；主环境支持Gloo并通过该页测试时可复用，不必另建 |
| 按需：独立DeepSpeed环境 | 已有兼容环境；可复用[PAR-CUDA-MECHANISM](../communication_parallelism/quickstart.md#par-cuda-mechanism-quickstart)的依赖 | Part04·16可选配置解析；本项不要求两张GPU，也不代表双卡Profile预检通过 |

学习Part03和Part04单GPU主线，先准备主GPU环境即可；CPU练习也可在其中运行，无需再建同用途环境。原生CUDA需要的`nvcc`、主机编译器与头文件是额外工具链条件，不是另一个Python环境。Part04·10／11／12／17只调用CUDA PyTorch，不要求本机`nvcc`；16的标准库题目不要求DeepSpeed。

表中按需分支只在进入对应实验时选择，不要求全部创建，具体输入与状态见[实验资产表](./experiment_assets.md#环境与实验入口)。多GPU通信按通信专题的硬件与后端条件另行选择；需要DeepSpeed时使用独立兼容环境，不向性能、推理服务或Triton主环境临时混装。

## 首次创建

<a id="triton-environment"></a>

主GPU环境沿用 `cnb/environment-gpu.yml`，由Conda YAML管理，不是独立注册的环境Profile。它提供基础包、开发工具和CUDA PyTorch/Triton，供Part03与Part04单GPU主线共用。超时与重试设置只作用于当前命令，不能保证下载速度；PyTorch来源仍由依赖文件决定。

```bash
PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8 \
  conda env create -f cnb/environment-gpu.yml
```

创建后注册仓库导入路径和Jupyter内核。Notebook或VSCode选择 `Python (llm_algo_cnb_gpu)`；已打开的旧Kernel需重启，注册不等于预检通过。

```bash
conda activate llm_algo_cnb_gpu
python tools/register_notebook_runtime.py
python -m ipykernel install --user \
  --name llm_algo_cnb_gpu \
  --display-name "Python (llm_algo_cnb_gpu)"
```

## 中断续装与重新进入

环境尚未创建时重新执行创建命令。Conda已创建但pip中断时，确认环境名与来源后用同YAML续装；不加 `--prune`，不清理下载缓存或覆盖其他后端环境。身份不明或混装时先核对依赖，不逐包猜测修复。

```bash
PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8 \
  conda env update -n llm_algo_cnb_gpu -f cnb/environment-gpu.yml
conda activate llm_algo_cnb_gpu
python tools/register_notebook_runtime.py
python -m ipykernel install --user \
  --name llm_algo_cnb_gpu --display-name "Python (llm_algo_cnb_gpu)"
```

日常重新进入只激活、检查解释器与依赖、启动Jupyter，不重新安装整套环境。GPU与工具链检查见“验证与常见问题”。

```bash
conda activate llm_algo_cnb_gpu
python -c "import sys; print(sys.executable)"
python tools/register_notebook_runtime.py --check
python -m pip check
python -m jupyter lab
```

## 手动创建（可选）

<details>
<summary>YAML创建流程不可用时的等价操作</summary>

手动路径沿用YAML的Python版本和三层requirements，不另选PyTorch组合；下载来源由依赖文件保留。已创建环境但仅pip中断时，优先使用前面的续装路径。

```bash
conda create -n llm_algo_cnb_gpu -c conda-forge python=3.10 pip -y
conda activate llm_algo_cnb_gpu
python -m pip install -r requirements/base.txt -r requirements/dev.txt \
  -r requirements/gpu.txt --timeout 120 --retries 8
python tools/register_notebook_runtime.py
python -m pip check
python -m ipykernel install --user \
  --name llm_algo_cnb_gpu --display-name "Python (llm_algo_cnb_gpu)"
```

</details>

## 验证与常见问题

<a id="verification"></a>

先检查主GPU环境的仓库运行时、依赖与设备。`validate_cnb_gpu.sh`检查PyTorch/Triton导入、GPU可见性并运行PyTorch GPU计算；它不是注册Profile的完整预检，不编译教学Triton内核，也不检查原生CUDA C++工具链。

```bash
conda activate llm_algo_cnb_gpu
python tools/register_notebook_runtime.py --check
python -m pip check
bash tools/validate_cnb_gpu.sh
```

随后用实际Triton练习核对答案和未完成题目的预期失败。实验顺序与证据范围按[实验资产表](./experiment_assets.md)选择。

```bash
python tools/test_notebook_answers.py \
  03_Triton_Kernels/01_Triton_Vector_Addition.ipynb \
  --mode both --strict-question-failure
```

### 原生CUDA工具链

原生CUDA C++页面在同一Python环境中另检查Toolkit、编译器与设备能力。下列命令只说明工具可定位，真正编译和数值通过由目标Notebook确认；缺少工具或编译错误不能登记为通过，也不能归入“无GPU跳过”。

首次编译 CUDA 扩展可能超过默认测试时限，可在同一测试命令末尾增加 `--timeout 180`；这只延长子进程等待时间，不跳过编译、数值或题目门禁检查。

```bash
nvcc --version
c++ --version
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
```

<a id="zero-config-check"></a>

### Part04·16配置解析核对

激活已选择的独立DeepSpeed环境后，运行下面的严格核对命令。它只解析配置，不启动模型、优化器或通信进程；`--require-passed`将缺依赖跳过视为未通过。框架解析状态见[16节资产条目](./experiment_assets.md#zero-config-evidence)，不以配置通过替代训练验证。

```bash
python tools/test_framework_conformance.py \
  04_CUDA_and_System_Optimization/16_DeepSpeed_Zero_Config.ipynb \
  --require-passed
```

### 常见问题

| 现象 | 先检查什么 | 处理动作 |
|:---|:---|:---|
| 缺包或安装中断 | 环境是否已创建、失败依赖阶段 | 同YAML续装，再完成注册与预检；不默认删除环境 |
| 下载超时 | 实际下载源、代理和连通性 | 延长当前命令超时并重试；不全局换源或更换CUDA依赖组合 |
| 终端通过但Notebook失败 | `sys.executable` 与所选Kernel | 切换目标内核并重启，不在错误Kernel中继续安装 |
| 子目录无法导入 `tools` | 当前解释器的仓库运行时登记 | 根目录重新登记，再重启Kernel |
| GPU可见但张量操作失败 | PyTorch构建、GPU架构、驱动和解释器 | 保留原始错误并核对兼容性 |
| Triton或CUDA扩展失败 | 编译日志、输入与GPU能力；原生扩展另查Toolkit和编译器 | 按错误定位，首次编译与稳态执行分开记录 |
| `SKIPPED` | CUDA是否可用、运行前提是否满足 | 记录未执行，满足条件后重跑，不登记为正确性通过 |
