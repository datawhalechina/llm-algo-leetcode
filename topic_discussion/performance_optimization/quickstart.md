# 性能优化实验 Quickstart

以下命令在仓库根目录执行，用于创建、恢复和检查实验环境。实验输入与结果见[实验资产表](./experiment_assets.md)，环境隔离要求见[环境资产索引](../../docs/gpu_environment_assets.md)。

## 1. 选择环境

本专题的单卡测量与机制实验统一使用 `llm_perf_cu128`。Part 05 · 03的测量、配置核对与结果登记也可在这个环境完成。训练模型、推理服务和多卡实验仍按对应页面选择环境；只做CPU结果审查时，才按需选择独立的 `PERF-DELIVERY` 控制环境。

| 实验 | 环境入口 | 运行条件 |
|:---|:---|:---|
| 机制页 CPU 练习 | [BASE-CPU 安装与启动](../../docs/guide.md#base-cpu) | 按各页运行环境说明选择依赖 |
| PERF-TRACE、PERF-MEMORY-TIMELINE、PERF-ROOFLINE、PERF-REGRESSION；PERF-INPUT、PERF-IO、PERF-CANDIDATE、PERF-COMPOSITION | [PERF-CUDA-MECHANISM](gpu_quickstart.md#perf-cuda-mechanism-quickstart)，环境名 `llm_perf_cu128` | 单张 GPU 至少 8 GiB 显存，Profile 预检通过 |
| Part 05 性能优化交付 | [PERF-CUDA-MECHANISM](gpu_quickstart.md#perf-cuda-mechanism-quickstart)，环境名 `llm_perf_cu128` | 本项目所有阶段共用；`close` 不调用 GPU |
| 训练、推理服务、量化或多卡候选复测 | 候选实验原有的环境 | 保持目标后端、硬件和测量条件一致 |
| 可选：仅CPU结果审查 | [PERF-DELIVERY](#perf-delivery-quickstart)，环境名 `llm_perf_delivery` | 读取已有结果，不执行GPU测量 |

## 2. 首次创建

首次运行单卡机制实验，进入[性能机制GPU环境的一键创建](gpu_quickstart.md#perf-cuda-mechanism-quickstart)，创建 `PERF-CUDA-MECHANISM`；已有同一环境时直接进入下一节。下方只提供可选CPU控制环境的安装入口，Part 05 · 03使用主环境时无需另建。安装器添加 `--dry-run` 可查看计划，不会创建环境或执行预检。

### PERF-DELIVERY：交付控制环境

<a id="perf-delivery-quickstart"></a>

`PERF-DELIVERY` 使用独立Conda环境，供只读取已有配对测量、剖析和回退记录的CPU审查使用。Part 05 · 03独立产生其测量输入，不要求Part 02先通过。

```bash
python tools/install_environment_profile.py \
  --profile PERF-DELIVERY \
  --env-name llm_perf_delivery
```

## 3. 中断续装与重新进入

单卡主环境安装中断时，按[性能分析续装说明](gpu_quickstart.md#_3-中断续装与重新进入)恢复；已经安装完成时，只需激活并启动Jupyter，选择 `Python (llm_perf_cu128)`。其他训练或服务环境按其所属Quickstart恢复。

```bash
conda activate llm_perf_cu128
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile PERF-CUDA-MECHANISM
python -m jupyter lab
```

可选CPU控制环境已经创建且确认属于 `PERF-DELIVERY` 时，可使用下面的续装命令；`--reuse` 会修改该环境，不能用于身份不明或已混装的环境。安装完成后激活 `llm_perf_delivery` 并选择同名内核。激活时出现 `PYTHONPATH` 覆盖提示属于隔离设置，是否可用以预检结果为准。

```bash
python tools/install_environment_profile.py --profile PERF-DELIVERY --env-name llm_perf_delivery --reuse
```

## 4. 手动创建（可选）

<details>
<summary>安装器不可用时创建交付控制环境</summary>

手动路径使用 Profile 中同一份依赖文件；完成依赖检查、Profile 预检和内核登记后再启动 Notebook。单卡机制环境的手动步骤见[性能机制GPU环境](gpu_quickstart.md#_4-手动创建-可选)。

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

下面检查单卡主环境；Notebook的 `sys.executable` 应指向 `llm_perf_cu128`。CPU控制环境则将环境名与Profile分别换为 `llm_perf_delivery`、`PERF-DELIVERY`。通过预检后，按具体页面执行准备阶段；项目中的 `prepare` 与机制页中的 `dry_run` 以各页说明为准。

```bash
conda activate llm_perf_cu128
python -m pip check
python tools/register_notebook_runtime.py --check
python tools/environment_preflight.py --profile PERF-CUDA-MECHANISM
python -c "import sys; print(sys.executable)"
```

| 现象 | 处理方式 |
|:---|:---|
| 控制环境找不到本项目结果 | 核对本次 `RUN_LABEL` 的测量清单和路径；缺少测量时回到 `PERF-CUDA-MECHANISM` 运行 `measure` |
| 依赖冲突或Profile检测到禁止安装的后端 | 保留原环境排查；按所需实验的Profile用新环境名创建，不把主环境改装成控制或服务环境 |
| Notebook 找不到 `tools` 或解释器错误 | 在目标环境注册仓库运行时，选择对应内核并重启 |
| 只有 CPU 决策结果 | 可检查比较条件、冲突与回退规则；GPU 或服务收益必须来自实际测量 |
| 候选的原环境不可用 | 保存失败原因及缺失测量，暂停采用判断；恢复原环境后补测 |
