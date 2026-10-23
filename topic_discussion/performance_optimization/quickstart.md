# 性能优化实验 Quickstart

以下命令在仓库根目录执行，用于创建、恢复和检查实验环境。实验输入与结果见[实验资产表](./experiment_assets.md)，环境隔离要求见[环境资产索引](../../docs/gpu_environment_assets.md)。

## 1. 选择环境

已完成性能分析单卡实验的学习者可继续使用 `llm_perf_cu128`。读取现有结果并生成交付判断使用独立的 CPU 控制环境；重新测量候选时回到其原执行环境。

| 实验 | 环境入口 | 运行条件 |
|:---|:---|:---|
| 机制页 CPU 练习 | [BASE-CPU 安装与启动](../../docs/guide.md#base-cpu) | 按各页运行环境说明选择依赖 |
| PERF-INPUT、PERF-IO、PERF-CANDIDATE、PERF-COMPOSITION | [PERF-CUDA-MECHANISM](../performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart)，环境名 `llm_perf_cu128` | 单张 GPU 至少 8 GiB 显存，Profile 预检通过 |
| Part 05 性能优化交付 | `PERF-DELIVERY`，环境名 `llm_perf_delivery` | CPU；已有 Part 02 · 74 交接包及其引用的结果 |
| 训练、推理服务、量化或多卡候选复测 | 候选实验原有的环境 | 保持目标后端、硬件和测量条件一致 |

## 2. 首次创建

单卡机制环境的创建、续装和预检统一使用[性能分析 Quickstart](../performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart)。以下命令只创建本专题的交付控制环境。添加 `--dry-run` 可查看安装计划，不会创建环境或执行预检。

### PERF-DELIVERY：交付控制环境

<a id="perf-delivery-quickstart"></a>

`PERF-DELIVERY` 使用独立 Conda 环境，读取交接包中的候选组合、配对结果、瓶颈变化和回退条件。安装器自动注册仓库运行时、检查依赖与 Profile，并登记同名 Jupyter 内核；此环境无需 GPU，也不安装候选训练或服务后端。

```bash
python tools/install_environment_profile.py \
  --profile PERF-DELIVERY \
  --env-name llm_perf_delivery
```

## 3. 中断续装与重新进入

环境尚未创建时重新运行创建命令。已创建且确认属于 `PERF-DELIVERY` 时，使用下面的命令继续安装并完成检查；`--reuse` 会修改该环境，不能用于混装环境。单卡机制环境续装见[性能分析 Quickstart](../performance_analysis/quickstart.md#_3-中断续装与重新进入)。

```bash
python tools/install_environment_profile.py --profile PERF-DELIVERY --env-name llm_perf_delivery --reuse
```

安装完成后，激活环境并启动 Jupyter，在 Notebook 中选择 `Python (llm_perf_delivery)`。激活时出现 `PYTHONPATH` 覆盖提示属于隔离设置，是否可用以预检结果为准。

```bash
conda activate llm_perf_delivery
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile PERF-DELIVERY
python -m jupyter lab
```

## 4. 手动创建（可选）

<details>
<summary>安装器不可用时创建交付控制环境</summary>

手动路径使用 Profile 中同一份依赖文件；完成依赖检查、Profile 预检和内核登记后再启动 Notebook。单卡机制环境的手动步骤仍见[性能分析 Quickstart](../performance_analysis/quickstart.md#_4-手动创建-可选)。

```bash
conda create -n llm_perf_delivery python=3.12 -y
conda activate llm_perf_delivery
conda env config vars set PYTHONNOUSERSITE=1 PYTHONPATH=
conda deactivate
conda activate llm_perf_delivery
python -m pip install --upgrade pip
python -m pip install -r requirements/performance-delivery.txt --timeout 120 --retries 8
python tools/register_notebook_runtime.py
python -m pip check
python tools/environment_preflight.py --profile PERF-DELIVERY
python -m ipykernel install --user \
  --name llm_perf_delivery --display-name "Python (llm_perf_delivery)"
```

</details>

## 5. 验证与常见问题

先激活目标环境，再检查依赖、Profile 和解释器。下面验证交付控制环境；使用单卡机制环境时按[性能分析 Quickstart](../performance_analysis/quickstart.md#_5-验证与常见问题)验证。Notebook 的 `sys.executable` 应指向所选环境；预检通过仅说明环境就绪，交接包及其引用文件仍需按项目页核对。

```bash
conda activate llm_perf_delivery
python -m pip check
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile PERF-DELIVERY
python -c "import sys; print(sys.executable)"
```

| 现象 | 处理方式 |
|:---|:---|
| 控制环境找不到上游结果 | 核对 Part 02 · 74 交接包引用的 JSON、执行轨迹与产物路径；需要补测时回到候选原环境 |
| 依赖冲突或 Profile 检测到禁止安装的后端 | 使用新的环境名创建独立控制环境，保留原环境排查 |
| Notebook 找不到 `tools` 或解释器错误 | 在目标环境注册仓库运行时，选择对应内核并重启 |
| 只有 CPU 决策结果 | 可检查比较条件、冲突与回退规则；GPU 或服务收益必须来自实际测量 |
| 候选的原环境不可用 | 保存失败原因及缺失测量，暂停采用判断；恢复原环境后补测 |
