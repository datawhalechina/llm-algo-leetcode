# 通信与并行实验 Quickstart

本页用于创建、恢复和预检 CPU 多进程核对环境与真实多卡机制环境。页面、workload、结果路径、实际环境和硬件快照见[实验资产表](./experiment_assets.md)及结果 JSON；精确 Profile 定义在注册表中维护，隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)。以下命令均在仓库根目录执行。

<a id="par-cpu-quickstart"></a>

## PAR-CPU：两进程框架核对环境

基础题目与答案测试可以使用 `BASE-CPU`；需要核对 Gloo collective、DTensor、上下文切分或 Pipeline API 时使用 `PAR-CPU`。框架核对默认在本机启动两个短生命周期 CPU 进程，不形成 NCCL、GPU 带宽或扩展效率证据。

```bash
python tools/install_environment_profile.py \
  --profile PAR-CPU \
  --env-name llm_parallel_cpu

conda activate llm_parallel_cpu
python tools/environment_preflight.py --profile PAR-CPU
python tools/test_framework_conformance.py \
  --all-part02 \
  --profile PAR-CPU \
  --require-passed
```

已有同 Profile 环境可在安装命令末尾添加 `--reuse`。Notebook 中的框架核对会自行建立本地 Gloo rendezvous 并在结束后销毁进程组；若上一次运行被强制中断，重启 Kernel 后再运行该单元。

<a id="par-cuda-mechanism-quickstart"></a>

## PAR-CUDA-MECHANISM：双卡通信机制环境

27–29、46–48、PAR-CONTEXT 和 PAR-REPLICA 使用该环境。真实 collective 至少需要两张可见 CUDA GPU；单卡环境保持 `RUN_MODE=dry_run`，只运行 CPU 机制、测试和实验计划登记。

### 一键创建（推荐）

```bash
python tools/install_environment_profile.py \
  --profile PAR-CUDA-MECHANISM \
  --env-name llm_parallel_cu128

conda activate llm_parallel_cu128
python -m jupyter lab
```

安装前可添加 `--dry-run` 查看计划。已有环境只有在依赖仍属于同一 Profile 时才使用 `--reuse`；vLLM、SGLang 和其他自带 PyTorch 版本的后端应保留独立环境。

### 手动创建

```bash
conda create -n llm_parallel_cu128 python=3.12 -y
conda activate llm_parallel_cu128

python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cu128.txt \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirements/distributed.txt

python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile PAR-CUDA-MECHANISM

python -m ipykernel install --user \
  --name llm_parallel_cu128 \
  --display-name "Python (llm_parallel_cu128)"
python -m jupyter lab
```

### 后续复测

```bash
conda activate llm_parallel_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile PAR-CUDA-MECHANISM
python -m jupyter lab
```

Notebook 中使用 `sys.executable` 确认解释器属于当前环境。只有 `pip check` 无冲突、Profile 预检通过且 `torch.cuda.device_count() >= 2` 时，才启用真实多卡运行。

## 分布式推理环境

PAR-REPLICA 的机制实验仍使用 `PAR-CUDA-MECHANISM`，只验证独立 GPU 副本的负载分配。vLLM / SGLang 的真实请求、缓存和尾延迟证据使用各自独立 Profile，由分布式推理项目登记，不在本环境中追加安装。

## Ray 环境选择

当前机制环境不安装 Ray。27–29、46–48、PAR-CONTEXT 和 PAR-REPLICA 直接使用 `torchrun + torch.distributed + NCCL`，避免将 actor 启动、调度等待或对象存储时间混入 collective 机制结果。

跨节点训练编排使用规划中的 `PAR-TRAIN-CLUSTER-RAY`，集群 Serving 编排使用规划中的 `PAR-SERVING-RAY`。两者当前只登记职责，不加入 `requirements/distributed.txt`；适用项目、证据边界和建设状态统一见[实验资产表](./experiment_assets.md)。

## 常见异常入口

| 现象 | 处理动作 |
|:---|:---|
| `No module named 'tools'` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Kernel |
| `pip check` 出现 vLLM / torch / xformers 冲突 | 停止覆盖安装，重建本 Profile 的独立环境 |
| 只发现一张 GPU | 保持 `dry_run`；不要把单进程复制写成 NCCL 证据 |
| NCCL 初始化失败 | 检查可见设备、驱动、进程数和 rank 到设备映射，保留失败日志 |
| 运行挂起 | 检查各 rank 是否进入相同 collective，并设置实验超时 |
| 跨机网络或 RDMA 未配置 | 先完成单机双卡基线；跨机结果使用新的拓扑与运行标签 |
