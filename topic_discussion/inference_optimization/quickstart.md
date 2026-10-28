# 推理优化实验 Quickstart

本页只说明环境创建和进入方式。页面、工作负载、结果路径、实际版本和硬件快照见[实验资产表](./experiment_assets.md)及结果 JSON；跨专题 Profile 与隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)。以下命令均在仓库根目录执行。

<a id="inf-cuda-mechanism-quickstart"></a>

## INF-CUDA-MECHANISM：单卡推理机制环境

这一环境用于真实模型的迭代成本、执行分派、显存和局部延迟对照，不安装 vLLM 或 SGLang，也不形成服务吞吐结论。

第一轮复测只修改各页 5.1 的 `RUN_MODE` 与 `RUN_LABEL`，保持默认模型、形状、候选矩阵、预热和重复次数不变。20、执行分派和执行重叠使用合成张量，不下载数据集；迭代预算下载 `Qwen/Qwen3-0.6B`；动态候选预算下载 `Qwen/Qwen2.5-1.5B-Instruct` 与 `Qwen/Qwen2.5-0.5B-Instruct`。模型由 Transformers 写入 Hugging Face 缓存，Notebook 会在结果中记录实际解析的模型版本；遇到 API 限流时先登录 Hugging Face 或等待限流窗口恢复，不把令牌写入 Notebook。

```bash
python tools/install_environment_profile.py \
  --profile INF-CUDA-MECHANISM \
  --env-name llm_inference_mechanism_cu128

conda activate llm_inference_mechanism_cu128
python -m jupyter lab
```

只检查安装计划时添加 `--dry-run`。进入 Notebook 后先确认 Kernel 为该环境，再运行 Profile 预检；只有预检通过后才把 `RUN_MODE` 从 `dry_run` 改为 `real_gpu`。

```bash
conda activate llm_inference_mechanism_cu128
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile INF-CUDA-MECHANISM
python -m jupyter lab
```

预检只确认环境、CUDA、依赖和仓库运行时可用，不代表某个候选一定更快。具体运行顺序、固定工作负载和证据状态见[实验资产表](./experiment_assets.md#单卡机制验证顺序)。

<a id="inf-vllm-spec025-quickstart"></a>

## INF-VLLM-SPEC025：投机解码选型 CUDA 12.9 环境

第 68 节使用独立的 vLLM 0.25.1 CUDA 12.9 环境完成 N-gram、原生 MTP、EAGLE、DFlash 与 DSpark 的候选准入和配对测量。0.25.1 是本项目覆盖 DFlash 与 DSpark 的最低推荐补丁版本；0.31 仅作为后续复测，不是第 68 节的硬依赖。该环境不能覆盖历史 `INF-VLLM-011-CU128` 基线。

```bash
PIP_INDEX_URL=https://pypi.org/simple \
python tools/install_environment_profile.py \
  --profile INF-VLLM-SPEC025 \
  --env-name llm_vllm_spec025_cu129

conda activate llm_vllm_spec025_cu129
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile INF-VLLM-SPEC025
```

正式安装前主机驱动必须满足 CUDA 12.9 Update 1 的门禁；本仓库按 Linux R575.57 或更高版本验收。先运行 target-only 启动与固定请求 smoke，再开放 G1–G5；每个候选仍需分别通过方法声明、checkpoint 资产和显存门禁。

<a id="inf-vllm-mtp-quickstart"></a>

## INF-VLLM-MTP：原生 MTP CUDA 12.9 单卡环境

第 35 节使用独立的 vLLM CUDA 12.9 环境，对同一个 `Qwen/Qwen3.5-0.8B` checkpoint 比较 target-only 与 MTP-1。该模型只带一个原生 MTP 层，因此默认候选深度固定为 1。默认 PyPI 上的 vLLM 0.31.0 是 CUDA 13.0 构建；该版本的官方 Release 另行发布 cu129 wheel，没有发布 cu128 wheel。本 Profile 固定 release commit 对应的 `wheels.vllm.ai` cu129 资产，不能复用已安装默认 wheel 的环境，也不要先安装 `torch-cu128.txt`，或复用 `INF-CUDA-MECHANISM` 与历史 vLLM 0.11 环境。

```bash
PIP_INDEX_URL=https://pypi.org/simple \
python tools/install_environment_profile.py \
  --profile INF-VLLM-MTP \
  --env-name llm_vllm_mtp_cu129

conda activate llm_vllm_mtp_cu129
python -m jupyter lab
```

首次安装不要添加 `--reuse`。日志必须出现 `vllm-0.31.0+cu129` 和 `torch-2.13.0+cu129`；如果出现 `+cu130`，立即停止，不要运行实验。该执行栈要求主机驱动满足 CUDA 12.9 的完整运行条件：Linux 至少使用 R575.51，CUDA 12.9 Update 1 至少使用 R575.57。R570 虽能通过部分 CUDA 12.x 基础检查，但可能在 Triton 动态内核加载阶段失败，不作为本 Profile 的可运行驱动。

进入 Notebook 前可独立复核环境：

```bash
conda activate llm_vllm_mtp_cu129
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile INF-VLLM-MTP
```

预检必须同时满足 `runtime.torch_cuda == "12.9"`、`cuda_available == true` 和 `ready == true`。旧的 `llm_vllm_mtp` CUDA 13 环境与未完成的 `llm_vllm_mtp_cu128` 环境只保留失败诊断，不用于复测。

首次验证只把第 35 节的 `RUN_MODE` 改为 `real_backend`，并设置唯一 `RUN_LABEL`。Notebook 会顺序启动两个隔离 worker，先运行 target-only，再运行 MTP-1；不要同时启动另一份占用 GPU 的 vLLM 服务。RTX 5070 Ti + R570 的已登记结果在 target-only 初始化 Triton RoPE 内核时失败；保持该主机驱动时不要反复安装或复测，改在满足驱动条件的独立机器上运行。

<a id="inf-vllm-quickstart"></a>

## INF-VLLM-011-CU128：vLLM 服务环境

连续批处理、分块预填充、KV 生命周期等真实后端机制实验，以及 66–72 的服务项目，使用独立 vLLM Profile。不要先安装 `torch-cu128.txt`，也不要向机制环境追加 vLLM。

```bash
PIP_INDEX_URL=https://pypi.org/simple \
PIP_DEFAULT_TIMEOUT=120 \
PIP_RETRIES=8 \
python tools/install_environment_profile.py \
  --profile INF-VLLM-011-CU128 \
  --env-name llm_vllm_cu128

conda activate llm_vllm_cu128
python -m jupyter lab
```

如果官方 PyPI 当前连接不稳定，可以把 `PIP_INDEX_URL` 替换为可访问的镜像源；不要在同一个命令中混用多个镜像来解析 vLLM、PyTorch 和约束文件。安装器已经使用 `conda run --no-capture-output`，终端会持续显示下载进度，不需要额外添加 `-v`。

安装中断后不要删除环境或清空 Hugging Face、pip 缓存。重新进入同一个环境，使用 `--reuse` 继续安装：

```bash
PIP_INDEX_URL=https://pypi.org/simple \
PIP_DEFAULT_TIMEOUT=120 \
PIP_RETRIES=8 \
python tools/install_environment_profile.py \
  --profile INF-VLLM-011-CU128 \
  --env-name llm_vllm_cu128 \
  --reuse
```

如果下载速度持续过低，可以先停止当前安装，稍后用同一命令重试；不要同时启动多个 pip 安装进程。只有安装器最后输出“环境安装与预检完成”，才继续注册或选择 Notebook Kernel。

SGLang 对照使用独立环境，不能与 vLLM 共享同一个 Python 环境。尚未登记稳定 Profile 的后端只作为可选候选，不进入默认复测链。

Part 05 推理服务项目也使用这个 Profile。服务和 Notebook 分别在两个终端运行，但都激活 `llm_vllm_cu128`；同一时刻只运行一个候选服务，不另建控制环境。连续批处理实验必须关闭前缀缓存，避免后运行的一组因缓存命中获得额外收益。环境预检通过后，在独立终端启动固定服务：

```bash
conda activate llm_vllm_cu128
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 8
```

第 37 节需要一个已开启 Prefix Cache 的真实服务。停止前一个服务后，再启动以下配置：

```bash
conda activate llm_vllm_cu128
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching
```

确认 `http://127.0.0.1:8000/v1/models` 与 `/metrics` 可访问后，再从同一环境启动 JupyterLab。连续批处理页和第 37 节均只修改运行模式与唯一结果标签；切换实验前先停止旧服务，不能把前缀缓存开关作为未记录变量。

第 38 节比较同一模型和请求清单下的整段预填充与分块预填充。两种服务不能同时占用同一张 GPU；先启动一种配置、运行 Notebook 并保存阶段结果，停止服务后再启动另一种配置。

整段预填充基线：

```bash
conda activate llm_vllm_cu128
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --no-enable-chunked-prefill
```

分块预填充候选：

```bash
conda activate llm_vllm_cu128
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 512
```

两次运行分别把第 38 节的 `SERVICE_VARIANT` 设为 `whole_prefill` 和 `chunked_prefill`，保持同一个 `PAIR_LABEL`。只有两个阶段结果都存在时，Notebook 才形成配对结论。

## 常见异常

| 现象 | 处理方式 |
|:---|:---|
| `pip check` 报依赖冲突 | 停止执行，按目标 Profile 新建环境，不在旧环境逐包覆盖 |
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Kernel |
| CUDA 或显存预检失败 | 保持 `RUN_MODE=dry_run`，记录失败，不绕过硬件门槛 |
| 模型或数据下载失败 | 检查代理与缓存；不要把代理地址、令牌写入 Notebook |
| 服务能够启动但没有稳定结果 | 只记录环境/能力预检，不升级为服务性能证据 |
