# 教程模型、执行栈与环境资产索引

> 统一审阅入口见[教程资产总登记与审阅台账](./asset_registry.md)。

本页帮助学习者选择环境 Profile、确认环境隔离关系，并找到模型、执行栈和实验快照的维护入口。它不复制安装命令、逐包版本、Notebook 配置或实验结果；这些信息分别由 Profile 注册表、专题 Quickstart、专题实验资产表和结果 JSON 维护。

## 信息所有权

| 信息 | 唯一维护入口 | 本页职责 |
|:---|:---|:---|
| Profile 的 requirements、constraints、必需模块、GPU 门槛和禁用包 | [`requirements/environment-profiles.json`](../requirements/environment-profiles.json) | 提供用途与入口索引 |
| 环境创建、恢复、Kernel 注册和常见故障 | 对应专题 `quickstart.md` | 链接到操作入口 |
| 模型、数据、工作负载、结果路径和当前证据状态 | 对应专题 `experiment_assets.md` | 说明资产归属 |
| 本次软件、驱动、硬件和运行参数 | 结果 JSON 的 runtime snapshot | 说明必须记录的字段组 |
| 项目公共结果字段和证据等级 | [Part 02 结果资产与证据契约](./part02_result_assets.md) | 不重复字段定义 |
| 托管镜像与文档交付状态 | [教程资产总登记与审阅台账](./asset_registry.md) | 只说明运行入口 |

## 环境分层

```text
基础运行时
  ├─ CPU PyTorch
  └─ CUDA PyTorch
        ↓
能力或后端 Profile
  ├─ 训练 / 后训练 / 量化 / 架构 / 性能 / 分布式
  └─ vLLM / SGLang / 其他独立执行后端
        ↓
专题 Quickstart → Profile 预检 → Notebook → 结果 JSON
```

基础运行时决定 Python、PyTorch 和 CUDA wheel；能力 Profile 增加专题依赖；独立后端 Profile 维护自己的 PyTorch 与 CUDA 组合。Profile 预检只能确认当前环境满足已登记条件，不能替代固定工作负载下的实验结果。

## 环境 Profile 索引

精确配置以 [`requirements/environment-profiles.json`](../requirements/environment-profiles.json) 为准。表中的“入口”负责安装与恢复，“证据登记”负责工作负载和结果状态。

### CPU 与交付控制环境

| Profile | 用途 | 操作与证据入口 |
|:---|:---|:---|
| <a id="base-cpu"></a>`BASE-CPU` | Part 00–02 普通 CPU 机制练习、题目与测试 | [使用指南](./guide.md) |
| <a id="train-cpu"></a>`TRAIN-CPU` | 反向传播与训练控制机制页的 PyTorch、Transformers 与 Accelerate 核对 | [训练机制 Quickstart](../topic_discussion/backpropagation_training_mechanism/quickstart.md#train-cpu-quickstart) · [实验资产](../topic_discussion/backpropagation_training_mechanism/experiment_assets.md) |
| <a id="pt-cpu"></a>`PT-CPU` | 后训练机制页的 Datasets、PEFT、TRL、TorchRL、Accelerate 与 Gymnasium 核对 | [后训练 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-cpu-quickstart) · [实验资产](../topic_discussion/post_training_optimization/experiment_assets.md) |
| `PERF-DELIVERY` | 汇总性能证据并形成交付判断，不重新执行 GPU 实验 | [性能优化 Quickstart](../topic_discussion/performance_optimization/quickstart.md#perf-delivery-quickstart) · [实验资产](../topic_discussion/performance_optimization/experiment_assets.md) |
| `INF-SERVICE-DELIVERY` | 汇总推理服务候选、SLO 与交付状态 | [推理优化 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-service-delivery) · [实验资产](../topic_discussion/inference_optimization/experiment_assets.md) |

### CUDA 机制与训练环境

| Profile | 设备范围与用途 | 操作与证据入口 |
|:---|:---|:---|
| <a id="train-cuda-mechanism"></a>`TRAIN-CUDA-MECHANISM` | 单卡；梯度累积、AMP、重算与卸载 | [训练机制 Quickstart](../topic_discussion/backpropagation_training_mechanism/quickstart.md#train-cuda-mechanism-quickstart) · [实验资产](../topic_discussion/backpropagation_training_mechanism/experiment_assets.md) |
| <a id="arch-cuda-mechanism"></a>`ARCH-CUDA-MECHANISM` | 单卡；Block、Attention、递推状态、MLA 与混合记忆 | [架构 Quickstart](../topic_discussion/llm_architecture_evolution/quickstart.md#arch-cuda-mechanism-quickstart) · [实验资产](../topic_discussion/llm_architecture_evolution/experiment_assets.md) |
| <a id="perf-cuda-mechanism"></a>`PERF-CUDA-MECHANISM` | 单卡；Trace、显存时间线、输入流水线和 I/O | [性能分析 Quickstart](../topic_discussion/performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart) · [实验资产](../topic_discussion/performance_analysis/experiment_assets.md) |
| <a id="inf-cuda-mechanism"></a>`INF-CUDA-MECHANISM` | 单卡；真实模型迭代成本、显存与局部延迟 | [推理优化 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart) · [实验资产](../topic_discussion/inference_optimization/experiment_assets.md) |
| <a id="par-cuda-mechanism"></a>`PAR-CUDA-MECHANISM` | 至少双卡；collective、切分、同步和局部扩展行为 | [通信与并行 Quickstart](../topic_discussion/communication_parallelism/quickstart.md#par-cuda-mechanism-quickstart) · [实验资产](../topic_discussion/communication_parallelism/experiment_assets.md) |
| <a id="q-cuda-mechanism"></a>`Q-CUDA-MECHANISM` | 单卡；校准、低精度执行和量化机制对照 | [量化 Quickstart](../topic_discussion/quantization_compression/quickstart.md#q-cuda-mechanism-quickstart) · [实验资产](../topic_discussion/quantization_compression/experiment_assets.md) |
| <a id="q-qlora"></a>`Q-QLORA` | 单卡；BF16 LoRA 与 NF4 QLoRA 配对训练 | [量化 Quickstart](../topic_discussion/quantization_compression/quickstart.md#q-qlora-quickstart) · [实验资产](../topic_discussion/quantization_compression/experiment_assets.md) |
| <a id="pt-sft-cuda"></a>`PT-SFT-CUDA` | 单卡；SFT、LoRA 更新、验证与训练产物 | [后训练 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart) · [实验资产](../topic_discussion/post_training_optimization/experiment_assets.md) |
| <a id="pt-rl-cuda"></a>`PT-RL-CUDA` | 单卡；奖励模型、奖励信号、DPO 与 GRPO 机制实验 | [后训练 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-rl-cuda-quickstart) · [实验资产](../topic_discussion/post_training_optimization/experiment_assets.md) |
| <a id="pt-verl-cuda"></a>`PT-VERL-CUDA` | 单卡；verl 轨迹准入、阶段耗时与缓存驻留机制实验 | [后训练 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-verl-cuda-quickstart) · [实验资产](../topic_discussion/post_training_optimization/experiment_assets.md) |
| <a id="pt-distill-cuda"></a>`PT-DISTILL-CUDA` | 单卡；学生基线、固定轨迹 KD 与 TRL on-policy distillation | [后训练 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-distill-cuda-quickstart) · [实验资产](../topic_discussion/post_training_optimization/experiment_assets.md) |
| `PT-DELIVERY-CUDA` | 单卡；后训练候选冻结复测与综合交付 | [后训练 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-delivery-cuda-quickstart) · [实验资产](../topic_discussion/post_training_optimization/experiment_assets.md) |

### 独立推理后端环境

| Profile | 用途 | 操作与证据入口 |
|:---|:---|:---|
| <a id="inf-vllm-011-cu128"></a>`INF-VLLM-011-CU128` | 复现 vLLM 0.11、PyTorch 2.8 与 CUDA 12.8 历史服务基线 | [推理优化 Quickstart](../topic_discussion/inference_optimization/quickstart.md) · [实验资产](../topic_discussion/inference_optimization/experiment_assets.md) |
| <a id="inf-vllm-spec025"></a>`INF-VLLM-SPEC025` | vLLM 0.25.1、CUDA 12.9 的 N-gram、draft、EAGLE、DFlash 与 DSpark 选型 | [投机解码 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-spec025-quickstart) · [实验资产](../topic_discussion/inference_optimization/experiment_assets.md) |
| <a id="inf-vllm-mtp"></a>`INF-VLLM-MTP` | vLLM 原生 MTP 路径；与历史基线和 0.25.1 选型环境分离 | [MTP Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-vllm-mtp-quickstart) · [实验资产](../topic_discussion/inference_optimization/experiment_assets.md) |

## 环境隔离关系

| 环境类型 | 可以组合 | 必须分离 |
|:---|:---|:---|
| CPU 机制环境 | 基础 CPU PyTorch + 对应能力包 | CUDA PyTorch、Serving 后端、量化运行时和多卡框架 |
| CUDA PyTorch 机制环境 | CUDA PyTorch + 一个目标专题能力 Profile | vLLM、SGLang、TensorRT-LLM 及其他维护独立 PyTorch 栈的后端 |
| 后训练环境 | SFT/LoRA、偏好/RL、教师—学生蒸馏和交付控制分别按 Profile 创建 | QLoRA 使用 `Q-QLORA`；多工作进程 rollout 与 Serving 采样另建项目环境 |
| 量化环境 | `Q-CUDA-MECHANISM` 或 `Q-QLORA` 各自独立 | 不覆盖 Serving 环境；量化产物由匹配的后端另行验证 |
| 推理后端环境 | 同一 Profile 内固定 backend、PyTorch、CUDA 和模型资产 | 不同 vLLM 版本、SGLang、llama.cpp 与 TensorRT-LLM 不混装 |
| 多卡训练环境 | CUDA PyTorch + `PAR-CUDA-MECHANISM` | Serving 后端和单卡机制环境不能直接充当多卡证据 |

驱动由主机提供；`nvidia-smi` 显示的是驱动支持上限，`torch.version.cuda` 表示当前 PyTorch wheel 的 CUDA 版本。安装 requirements 不会安装或回滚 NVIDIA 驱动。

## 托管运行环境

托管环境是 Profile 的运行载体，不是新的依赖层。正式结果必须把平台、镜像、Python、PyTorch、驱动和设备写入 runtime snapshot。

| 入口 | 仓库入口 | 使用边界 |
|:---|:---|:---|
| CNB CPU | `.cnb.yml`、`.ide/Dockerfile`、`cnb/environment.yml` | Part 00–02 CPU 学习与 smoke；不能形成 GPU 证据 |
| CNB GPU | `.cnb.yml`、`.ide/Dockerfile.gpu`、`cnb/environment-gpu.yml` | 通过 `tools/validate_cnb_gpu.sh` 后才能运行 GPU Profile |
| Colab / ModelScope | [使用指南](./guide.md) | 优先复用平台 Kernel；实际运行时版本写入结果，不作为固定镜像承诺 |
| 自建容器或云端机器 | [使用指南](./guide.md) | 部署方维护宿主机与镜像；仓库只记录 Profile 和运行快照 |

## 模型与工作负载角色

| 角色 | 默认资产 | 适用范围 | 使用边界 |
|:---|:---|:---|:---|
| 稳定功能基线 | `Qwen/Qwen2.5-0.5B-Instruct` | 机制、LoRA/QLoRA、量化、推理与通信 smoke | 固定 model/tokenizer revision；不据此推断生产吞吐 |
| 现代兼容性候选 | `Qwen/Qwen3-0.6B` | Qwen3 架构、Tokenizer、Chat Template 与后端兼容性 | 显式记录 Thinking 模式；不与 Qwen2.5 直接比较性能 |
| 量化性能工作负载 | `Qwen/Qwen2.5-1.5B-Instruct` 或专题登记的大模型 | 量化产物、显存、延迟、吞吐与质量复测 | 固定量化格式、评测清单、后端和 Token 规模 |
| 通信性能工作负载 | 专题登记的 1.5B–7B 模型或合成大 Payload | collective、TP/CP/EP、FSDP/DeepSpeed | 固定 world size、拓扑、并行度和每 rank 输入 |
| 生产候选 | 项目指定模型、产物和输入清单 | Part 05 与部署项目 | 结论只适用于登记的模型、后端、硬件和工作负载 |

模型、Tokenizer、Chat Template 或 Thinking 模式变化后必须新建候选记录，不能覆盖旧结果。训练侧保存 Tokenized manifest，Serving 侧保存应用模板后的输入身份。

## 执行栈角色

| 执行栈 | 主要用途 | 环境要求 |
|:---|:---|:---|
| Transformers | 模型与 Tokenizer 语义基线、训练和固定输入前向 | 记录 package version 与 model/tokenizer revision |
| vLLM | GPU Serving、KV Cache、调度和投机解码 | 使用对应独立 Profile，记录 Engine 参数与后端版本 |
| SGLang | Prefix Cache、结构化生成与调度对照 | 独立环境；不与 vLLM 混装 |
| llama.cpp | GGUF、本地 CPU、端侧与 CPU/GPU Offload | 记录 release/commit、构建选项、线程和 Offload 层数 |
| TensorRT-LLM | NVIDIA Engine、插件、量化和专用部署 | 优先使用版本化 release 容器，记录 TensorRT/CUDA/GPU 组合 |

执行栈名称不能代替实验身份。正式比较还要固定模型 revision、输入清单、并发、Token 数、dtype、构建选项和硬件。

## 预检、运行快照与结果入口

| 阶段 | 入口 | 产出 |
|:---|:---|:---|
| 列出或检查 Profile | `python tools/environment_preflight.py --list-profiles` / `--profile <ID>` | `ok`、`warning`、`blocked` 与修复动作 |
| 注册仓库运行时 | `python tools/register_notebook_runtime.py` | 当前环境可从 Notebook 子目录导入仓库工具 |
| 执行专题实验 | 对应专题 Quickstart 和 Notebook | 固定工作负载下的结果 JSON |
| 查询已验证结果 | 对应专题 `experiment_assets.md` | RUN_LABEL、结果路径、证据等级和当前决策 |

结果 JSON 至少保存所选 Profile、Python/PyTorch/CUDA、设备、关键包版本、模型与后端身份、工作负载、失败原因和证据等级。字段命名以 [Part 02 结果资产与证据契约](./part02_result_assets.md)为准；Profile 通过不能自动提升证据等级。

<a id="environment-isolation-troubleshooting"></a>

## 环境隔离故障入口

| 现象 | 处理入口 |
|:---|:---|
| 当前 Kernel 与安装环境不是同一个 Python | [使用指南](./guide.md)和对应专题 Quickstart 的 Kernel 注册步骤 |
| `PYTHONPATH` 混入 ROS、系统 Python 或其他环境 | 先检查 `sys.executable` 与 `sys.path`；只在确认污染后按专题 Quickstart 修复 |
| PyTorch、CUDA wheel 与驱动不兼容 | 停止实验，核对 Profile 的 CUDA 要求和主机驱动；不要在原环境覆盖安装 |
| vLLM、SGLang、量化或分布式依赖冲突 | 新建目标 Profile 环境，不在已有训练或 Serving 环境继续叠加依赖 |

## 维护边界

- 新增或修改 Profile 时，先更新 `requirements/environment-profiles.json`，再更新本索引和对应专题 Quickstart。
- 新的 requirements 或 constraints 只表达安装意图和兼容约束，不记录实验结论。
- 已验证版本和硬件进入结果 JSON；专题实验资产表只登记稳定摘要、结果路径和复测状态。
- 不提交账号、令牌、私有 Endpoint、模型缓存、大型 Checkpoint、编译缓存、Binary 或临时 Trace。
- 新后端先建立独立 Profile 和专题实验条目，不覆盖已有环境后沿用旧结果。
