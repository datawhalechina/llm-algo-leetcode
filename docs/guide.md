# 使用指南

本指南面向学习者和实验运行者，依次回答四个问题：从哪里开始、使用什么环境、怎样确认环境可用、结果保存在哪里。

## 1. 选择学习与运行入口

| 目标 | 默认入口 | 环境要求 |
|:---|:---|:---|
| 阅读教程 | [教程首页](./index.md)或各 Part 导读 | 浏览器 |
| 运行 Part 00–01 或 Part 02 的 CPU 练习 | 目标 Notebook | `BASE-CPU` |
| 运行单卡训练、量化、架构或性能实验 | 对应专题 Quickstart | 专题 CUDA Profile |
| 运行推理服务、量化部署或多卡实验 | 对应专题 Quickstart | 独立后端或分布式 Profile |
| 复核一个项目的真实结果 | 项目 Notebook 的实验阶段 | 预检通过的环境和固定工作负载 |

建议从 [Part 02](./02_PyTorch_Algorithms/intro.md) 或目标专题开始，遇到概念、硬件或数学缺口时再回补 Part 00–01。标记为“CPU 主线”的页面可以先完成机制练习；标记为“CPU 主线，GPU 可选”或依赖真实后端的页面，进入实验前必须完成对应 Profile 的环境预检。

## 2. 运行基础 CPU 练习

### 使用平台已有环境

Colab、ModelScope Notebook 和云端镜像可能已经提供 Python 或 CUDA PyTorch。先确认当前 Notebook Kernel，不要先重装 PyTorch：

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

当前 Kernel 满足目标小节要求时，可以直接运行；Conda 只负责隔离依赖，不提供驱动或 GPU 能力。

<a id="base-cpu"></a>

### 安装并启动 BASE-CPU

Part 00–02 的普通机制练习、题目和测试统一使用 `BASE-CPU`。在仓库根目录执行：

```bash
python tools/install_environment_profile.py \
  --profile BASE-CPU \
  --env-name llm_algo_cpu
conda activate llm_algo_cpu
python -m jupyter lab
```

在 Notebook 中选择 `Python (llm_algo_cpu)`。安装器会完成依赖安装、Kernel 注册和环境预检；`BASE-CPU` 不用于真实 GPU、Serving、量化后端或多卡实验。

仓库仍保留 `environment.yml` 等兼容入口，但新环境优先使用 Profile 安装器。平台不能使用 Conda 时，可以复用现有 Kernel 或自行创建 `venv`，并确保 Jupyter 与安装依赖使用同一个 Python。

## 3. 运行专题实验

真实训练、GPU、Serving 和多卡实验从对应专题 Quickstart 创建环境：

| 实验方向 | 安装与预检入口 |
|:---|:---|
| 反向传播与训练机制 | [训练机制实验 Quickstart](./topic_discussion/backpropagation_training_mechanism/quickstart.md) |
| 后训练优化 | [后训练实验 Quickstart](./topic_discussion/post_training_optimization/quickstart.md) |
| 推理优化与 Serving | [推理优化实验 Quickstart](./topic_discussion/inference_optimization/quickstart.md) |
| 性能分析与性能优化 | [性能分析实验 Quickstart](./topic_discussion/performance_analysis/quickstart.md) |
| 架构机制 | [架构实验 Quickstart](./topic_discussion/llm_architecture_evolution/quickstart.md) |
| 量化与压缩 | [量化实验 Quickstart](./topic_discussion/quantization_compression/quickstart.md) |
| 通信与并行 | [通信与并行实验 Quickstart](./topic_discussion/communication_parallelism/quickstart.md) |
| 算子与部署 | [算子优化实验 Quickstart](./topic_discussion/operator_optimization/quickstart.md)、[部署与异构实验 Quickstart](./topic_discussion/deployment_heterogeneous/quickstart.md) |

不同专题 Profile 和独立 Serving 后端不要混装。Profile 用途、GPU 门槛和隔离关系见[环境资产索引](./gpu_environment_assets.md)；精确依赖由 [`requirements/environment-profiles.json`](../requirements/environment-profiles.json) 维护。

## 4. 预检、运行并查找结果

安装器会在创建环境时执行一次预检；重新使用已有环境或开始真实实验前，在仓库根目录再次运行目标 Profile 的预检：

```bash
python tools/environment_preflight.py --profile <PROFILE_ID>
```

| 状态 | 含义 | 下一步 |
|:---|:---|:---|
| `ok` | 当前配置满足目标实验要求 | 继续加载模型或启动后端 |
| `warning` | 主路径可运行，但可选能力缺失 | 按提示使用降级路径或跳过扩展 |
| `blocked` | 当前环境不能安全执行目标实验 | 按 `next_actions` 修复后重新预检 |

预检通过后按以下顺序运行：

1. 从仓库根目录启动 Jupyter，并选择目标 Profile 注册的 Kernel。
2. 在 Notebook 配置单元设置 `RUN_LABEL`、运行阶段、模型和工作负载等本次可修改参数。
3. 按 Notebook 的 Step 顺序执行，不跳过环境检查和实验准入条件。
4. 在 `benchmarks/results/` 查看结果 JSON、日志和 artifact；已有结果存在时更换唯一 `RUN_LABEL`，不要覆盖。

结果应记录运行时快照、配置、状态、失败原因和证据等级。CPU 账本、模拟结果和 smoke test 不能替代真实 GPU、后端或多卡证据；公共字段和证据等级见 [Part 02 结果资产与证据契约](./part02_result_assets.md)。推理项目 66–70 的专项核对见[推理项目验证清单](./verification/inference_projects.md)。

## 常见问题

| 现象 | 先检查 |
|:---|:---|
| 终端能导入包，Notebook 不能 | 当前 Kernel 的 `sys.executable` 与环境是否一致 |
| `nvidia-smi` 正常，但 PyTorch 看不到 GPU | PyTorch wheel 是否带 CUDA、驱动是否兼容 |
| 安装新包后环境冲突 | 是否把不同专题 Profile 或独立 Serving 后端混装 |
| 项目找不到仓库根目录 | 是否从仓库根目录启动 Jupyter，或是否按 Notebook 提示设置项目根目录 |
| 结果文件已存在 | 是否使用了本次运行的唯一 `RUN_LABEL` |

环境隔离问题继续查看[环境资产索引的故障入口](./gpu_environment_assets.md#environment-isolation-troubleshooting)。CNB、Docker 和云端 GPU 只是 Profile 的运行载体，仍遵循相同的预检和结果规则。

## 相关文档

- [环境资产索引](./gpu_environment_assets.md)：Profile 用途、隔离关系和运行资产入口。
- [贡献指南](./contributing.md)：参与内容、代码和测试贡献。
- [维护与发布手册](./maintenance.md)：源码同步、统一验证、迁移与发布。
