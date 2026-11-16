# 环境依赖文件

本目录区分安装依赖、版本约束与环境 Profile。一个 Profile 可以组合多份文件；文件数量不等于需要创建的环境数量。

## 先选 Profile

通常从[使用指南](../docs/guide.md)或目标专题 Quickstart 选择环境。安装顺序、约束、Python 范围和隔离要求以 [environment-profiles.json](./environment-profiles.json) 为准，本页不复制逐包版本和实验状态。

在仓库根目录查看已注册 Profile 和安装计划：

```bash
python tools/environment_preflight.py --list-profiles
python tools/install_environment_profile.py --profile BASE-CPU --env-name llm_algo_cpu --dry-run
```

安装器的 `--dry-run` 只展示计划；确认 Profile 与新环境名后，再按对应 Quickstart 执行安装。环境预检通过后，实验是否已完成仍以项目结果记录为准。

## 文件分工

| 类别 | 文件 | 作用 |
|:---|:---|:---|
| 公共依赖 | `base.txt`、`dev.txt` | Notebook、通用库与测试依赖，不提供完整 GPU 或服务后端环境 |
| PyTorch 基础栈 | `torch-cpu.txt`、`torch-cu128.txt` | 为使用共享 PyTorch 基线的 Profile 选择 CPU／CUDA 构建 |
| 机制与性能能力 | `architecture.txt`、`training-cpu.txt`、`training-mechanism.txt`、`profiling.txt`、`distributed.txt`、`inference-mechanism.txt` | 为架构、训练、性能或并行实验增加所需依赖 |
| 后训练能力 | `fine-tuning.txt`、`reinforcement-learning.txt`、`distillation.txt`、`qlora.txt` | SFT、强化学习、蒸馏和低比特适配训练依赖；各自按 Profile 隔离 |
| 综合项目入口 | `post-training-delivery.txt`、`performance-delivery.txt` | 项目评测或控制依赖；控制环境不替代测量环境 |
| 量化能力 | `quantization.txt`、`quantization-cpu.txt` | GPU 机制实验与 CPU 框架核对，不与独立 Serving 栈混装 |
| 独立推理后端 | `inference-vllm.txt`、`inference-vllm-spec025.txt`、`inference-vllm-mtp.txt` | 分别保留各后端实验的依赖组合，不自动互相升级或替换 |
| 版本约束 | `*-constraints.txt` | 限制已选择依赖的版本；约束文件本身不是完整安装清单 |
| 兼容或未注册入口 | `gpu.txt`、`inference-sglang.txt` | 用途见下方，不作为新实验的默认选择 |

## 组合与隔离

- `-r` 引用安装依赖，`-c` 引用版本约束；引用路径相对于该依赖文件，不是 Notebook 的工作目录。移动文件时必须同步修改这些引用。
- `base.txt` 不包含 PyTorch。手动安装时遵循对应 Profile 的顺序和约束，不能把所有 TXT 一次装入同一环境。
- vLLM 等独立后端维护其匹配的 PyTorch 与扩展栈，不再叠加 `torch-cu128.txt`。不同 CUDA 后端 Profile 使用独立环境。
- CPU 框架核对、GPU 机制测量、Serving 与交付控制承担不同任务；共享部分依赖不表示它们可以合并。
- `post-training-delivery.txt` 等薄入口用于表达 Profile 的职责；删除或合并前先核对注册表、安装器与消费者，而不只比较文件内容。
- 修改约束先核对安装计划与依赖一致性，再按专题补实际验证；修改版本文本不等于完成实验验证。

## 两个待整理入口

`gpu.txt` 是旧的 CUDA 开发组合，当前仍由 `cnb/environment-gpu.yml` 引用。CNB 调用迁移前保留，不直接删除。

`inference-sglang.txt` 当前未进入 Profile 注册表，不提供已验证环境承诺。若后续需要正式 SGLang 环境，应先确定 Profile、约束与验证入口；Mini-SGLang 教学工程的依赖另按该工程核对，不默认等同于这份安装清单。

## 保存范围

正式安装文件、版本约束与 Profile 注册表继续跟踪。个人 `pip freeze`、Conda 导出、试装草稿、安装日志和下载缓存放入被忽略的本地目录，不混入正式依赖清单。已完成实验的运行时版本保存在结果快照中。

当前不调整依赖路径，也不删除旧版本约束。后续重命名或移动时，同步核对 `-r/-c`、Profile、Conda/CNB、CI、Quickstart 与安装器测试。
