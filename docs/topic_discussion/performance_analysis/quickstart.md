# 性能分析实验 Quickstart

以下命令在仓库根目录执行。本页用于创建、恢复和检查实验环境；固定输入、结果版本与复现入口见[实验资产表](./experiment_assets.md)，跨专题隔离关系见[环境资产索引](../../gpu_environment_assets.md#perf-cuda-mechanism)。

## 1. 选择环境

CPU 练习用于复算事件依赖、驻留量和回归规则；实际设备时间与峰值通过单卡环境采集。Nsight、服务后端和多卡通信使用各自的工具或环境，采集结果按来源登记。

| 实验 | 环境入口 | 运行条件 |
|:---|:---|:---|
| Part 00 / 01 与机制页 CPU 练习 | [BASE-CPU 安装与启动](../../guide.md#base-cpu) | 按各页运行环境说明选择依赖 |
| PERF-TRACE、PERF-MEMORY-TIMELINE、PERF-ROOFLINE、PERF-REGRESSION | `PERF-CUDA-MECHANISM`，环境名 `llm_perf_cu128` | 单张 GPU 至少 8 GiB 显存，CUDA PyTorch 与 Profile 预检通过 |
| PERF-INPUT、PERF-IO 等性能优化交叉实验 | 复用同一 `PERF-CUDA-MECHANISM` 环境 | 按各页固定输入和阶段范围运行 |
| 多卡通信或推理服务测量 | [通信与并行](../communication_parallelism/quickstart.md) / [推理优化](../inference_optimization/quickstart.md) | 使用目标后端与设备条件 |

## 2. 首次创建

<a id="perf-cuda-mechanism-quickstart"></a>

`PERF-CUDA-MECHANISM` 使用独立 Conda 环境。安装器依次安装 CUDA PyTorch 与性能分析依赖，注册仓库运行时，执行依赖与 Profile 检查，并登记同名 Jupyter 内核。命令添加 `--dry-run` 只列出安装计划，不创建环境或执行预检；下载中断按下一节续装。

```bash
python tools/install_environment_profile.py \
  --profile PERF-CUDA-MECHANISM \
  --env-name llm_perf_cu128
```

## 3. 中断续装与重新进入

环境尚未创建时重新执行上一节命令。已经创建且确认属于同一 Profile 时，使用下面的续装命令；`--reuse` 会继续安装依赖并完成后续检查，不能用于修复身份不明或已混装的环境。

```bash
python tools/install_environment_profile.py --profile PERF-CUDA-MECHANISM --env-name llm_perf_cu128 --reuse
```

已完成安装时只需激活、检查并启动 Jupyter。Notebook 选择 `Python (llm_perf_cu128)`；重新激活时出现 `PYTHONPATH` 覆盖提示属于隔离设置，是否可用以预检结果为准。

```bash
conda activate llm_perf_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile PERF-CUDA-MECHANISM
python -m jupyter lab
```

## 4. 手动创建（可选）

<details>
<summary>安装器不可用时的等价操作</summary>

手动路径保持与 Profile 相同的依赖和安装顺序；依赖检查、环境预检和内核登记均完成后才启动实验。

```bash
conda create -n llm_perf_cu128 python=3.12 -y
conda activate llm_perf_cu128
conda env config vars set PYTHONNOUSERSITE=1 PYTHONPATH=
conda deactivate
conda activate llm_perf_cu128
python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cu128.txt --timeout 120 --retries 8
python -m pip install -r requirements/profiling.txt --timeout 120 --retries 8
python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile PERF-CUDA-MECHANISM
python -m ipykernel install --user \
  --name llm_perf_cu128 --display-name "Python (llm_perf_cu128)"
```

</details>

## 5. 验证与常见问题

先激活目标环境，再执行下面的检查。在 Notebook 中也核对 `sys.executable`：应指向 `llm_perf_cu128`。依赖无冲突且 Profile 预检返回 `status=ok` 后，按[实验资产表](./experiment_assets.md)选择需要运行的机制页。

Notebook 的 `RUN_MODE="dry_run"` 用于按页面检查实验准备，与安装器的 `--dry-run` 是两个独立开关。需要重新采集设备数据时才设为 `real_gpu`，并使用新的运行标签；阅读已有 JSON 或轨迹无需重新运行实验。

```bash
conda activate llm_perf_cu128
python -m pip check
python tools/environment_preflight.py --profile PERF-CUDA-MECHANISM
python -c "import sys; print(sys.executable)"
```

| 现象 | 处理方式 |
|:---|:---|
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，然后重启 Jupyter 内核 |
| `pip check` 报跨环境冲突 | 停止实验，使用新环境名重新执行一键创建 |
| CUDA 或 PyTorch 预检失败 | 保持 `RUN_MODE=dry_run`，检查驱动和 cu128 构建兼容性 |
| 结果只有环境信息 | 当前只完成预检，继续检查页面中实际测量的运行条件 |
| 执行轨迹为空或缺少 CUDA 事件 | 检查活动类型、预热、同步与正式采样步数 |
| Nsight 或多卡工具不可用 | 保留能力检查失败，用对应工具补测后再形成系统级或多卡结论 |
