# 教程模型、执行栈与环境资产表

> 统一审阅入口见 [教程资产总登记与审阅台账](./asset_registry.md)。

本文统一登记教程使用的模型角色、工作负载、执行栈、依赖 profile、环境组合、预检入口和真实运行快照规则。模型与执行栈说明“用什么验证什么”，环境资产说明“在哪种组合中运行”；实际版本和硬件仍以项目结果中的 runtime snapshot 为准。本文不保存账号、令牌、模型缓存或大型编译产物。

## 环境分层

```text
基础运行时层
  ├─ base.txt + torch-cpu.txt
  └─ base.txt + torch-cu128.txt
        ↓
能力 profile 层
  ├─ fine-tuning / qlora / quantization / reinforcement-learning
  ├─ distributed / profiling
  └─ inference-vllm / inference-sglang
        ↓
项目运行层
  └─ environment_preflight → runtime snapshot → result JSON
```

基础运行时决定 PyTorch 平台与 CUDA wheel；能力 profile 只添加任务依赖，不能判断主机是否有 GPU，也不能替换驱动或 PyTorch。真实运行前必须通过预检。

## 托管环境资产

托管环境是运行上述 profile 的交付入口，不是新的依赖层。平台提供的 Python、驱动、CUDA 与 GPU 型号必须通过预检写入运行时快照；课程依赖仍按“基础运行时 + 能力 profile”选择。

| 托管入口 | 仓库配置 / 文档 | 适用范围 | 当前可用状态 | 资产管理结论 |
| --- | --- | --- | --- | --- |
| CNB CPU 交互会话 | `.cnb.yml` 的 `vscode`、`.ide/Dockerfile`、`cnb/environment.yml` | Part 00–02 的 CPU-first 学习、Notebook 与基础测试 | CPU 镜像与 CPU CI smoke 已分离；实际会话镜像版本仍需实机快照确认 | 正式登记为统一交付环境；每次正式结果记录 CNB runner、镜像 / Python / PyTorch 快照与 profile |
| CNB GPU 交互会话 | `.cnb.yml` 的 `vscode-gpu`、`.ide/Dockerfile.gpu`、`cnb/environment-gpu.yml`、`tools/validate_cnb_gpu.sh` | Part 03 / 04，以及需要 GPU 的 Part 02 项目 | GPU runner、GPU 镜像与预检入口已分离；GPU 型号、驱动与可用 backend 未固定 | 只在预检通过后产生 GPU 证据；不能把 CPU 会话结果标为 GPU 验证 |
| Colab / ModelScope Notebook | `docs/guide.md` | 快速学习、单节探针或轻量复测 | 平台运行时由学习者选择 | 作为托管 runtime 记录；优先复用 Kernel PyTorch，不纳入固定镜像承诺 |
| 自建云端容器 / Docker | `docs/guide.md`、`.ide/Dockerfile` | 团队复现、专用 GPU 与 CI 扩展 | 宿主机与镜像组合由部署方维护 | 记录镜像、硬件与 profile；不把账号、端点或缓存提交入仓库 |

CNB CPU 环境统一为 `llm_algo_cnb_dev`，GPU 环境为 `llm_algo_cnb_gpu`。`main.push` 仅执行 CPU smoke；`vscode-gpu` 会话必须运行 `bash tools/validate_cnb_gpu.sh` 后才能产出 GPU 证据。固定镜像 digest、GPU 型号、驱动、CUDA 与 backend 版本仍待首次实机验证后写入运行时快照。

### CNB 变更记录（2026-09-30）

| 调整 | 目的 | 已完成验证 | 尚未验证 |
| --- | --- | --- | --- |
| `vscode` / `vscode-gpu` 改用独立 Dockerfile | 将 CPU 学习环境与 GPU 实验环境分开 | YAML 与 shell 静态语法检查 | CNB 实际构建与会话启动 |
| CPU CI 改为 `base + dev + torch-cpu` | 不在 CPU CI 中伪造 GPU / Triton 验证 | 配置审阅 | CNB push pipeline 实跑 |
| 新增 `environment-gpu.yml` 与 GPU 预检入口 | 让 GPU 证据必须经过硬件预检 | YAML 与 shell 静态语法检查 | GPU runner 的 CUDA、Triton 与实际设备 |
| 统一 Conda 环境名 | 消除 `llm_algo_cnb` / `llm_algo_cnb_dev` 漂移 | 配置与文档一致性检查 | 已有 CNB 会话迁移情况 |

## 基础运行时资产

| Profile | 文件 | 适用场景 | 组合规则 | 维护状态 |
| --- | --- | --- | --- | --- |
| 通用基础 | `requirements/base.txt` | Notebook、数据处理、Transformers 基础能力 | 所有环境的共同依赖；不包含 torch、驱动或 backend | 当前标准 |
| CPU PyTorch | `requirements/torch-cpu.txt` | CPU-first 机制、题目区、测试区与 dry-run | 与 CUDA PyTorch profile 互斥 | 当前标准 |
| CUDA PyTorch | `requirements/torch-cu128.txt` | PyTorch GPU、显存、性能与多卡实验 | 与 CPU profile 互斥；要求主机驱动兼容 | 当前标准 |
| GPU 便利 profile | `requirements/gpu.txt` | 旧 CUDA 开发环境 | 仅为兼容；新环境优先显式组合 base、CUDA PyTorch 与能力 profile | legacy convenience |

## 能力 profile 资产

| 能力 | 文件 | 主要依赖 | 适用内容 | 组合与限制 |
| --- | --- | --- | --- | --- |
| SFT / LoRA | `requirements/fine-tuning.txt` | `peft`、`accelerate`、`datasets` | 09–13、62–64 | 与一个明确的 CPU 或 CUDA PyTorch profile 组合 |
| QLoRA | `requirements/qlora.txt` | fine-tuning + `bitsandbytes` | 26、65 | 通常需要 CUDA；CPU 路径只保留机制和 dry-run |
| 量化机制与 PTQ | `requirements/quantization.txt` | `torchao`、`llmcompressor`、`compressed-tensors`、`datasets` | 25、40、41、52–54、82 | 与 CUDA PyTorch profile 组合；Serving 复测仍使用独立 vLLM/SGLang 环境 |
| RL / 对齐 | `requirements/reinforcement-learning.txt` | fine-tuning + `trl` | 14–16、84–86 | 与 CPU 或 CUDA PyTorch profile 组合；真实 rollout 另验资源 |
| 分布式 | `requirements/distributed.txt` | `accelerate`、`torchmetrics`、`deepspeed` | 27–29、46–49、79–81 | 机制页使用 `torchrun + torch.distributed`；真实训练项目使用 Transformers + Accelerate，并选择 FSDP 或 DeepSpeed；真实实验需 CUDA、多进程与通信后端 |
| vLLM backend | `requirements/inference-vllm.txt` | `vllm`、`httpx[socks]` | 66–72、83 | 单独维护；不默认与 SGLang 共装或替换现有 PyTorch |
| SGLang backend | `requirements/inference-sglang.txt` | `sglang`、`httpx[socks]` | 66、69、70、72、83 | 单独维护；不默认与 vLLM 共装 |
| Profiling | `requirements/profiling.txt` | `tensorboard` | 73–76、Part 03 / 04 | 真实 trace 仍依赖 PyTorch、GPU 和 profiler 能力 |

## 统一模型与工作负载资产

模型按实验角色登记，而不是按专题各自设置互不相干的默认值。下表维护模型 ID、用途和最低工具栈口径；正式运行仍需固定具体 revision，并在结果 JSON 中保存实际版本。

| 角色 | 默认模型或工作负载 | 主要适用范围 | 运行约束 | 证据边界 |
| --- | --- | --- | --- | --- |
| 稳定功能基线 | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 量化机制、LoRA/QLoRA、推理与通信 smoke | Transformers `>=4.37`；模型与 tokenizer 使用同一 revision | 只证明加载、前向和机制路径可运行 |
| 现代兼容性候选 | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Qwen3 架构、tokenizer、chat template、vLLM/SGLang 兼容性 | Transformers `>=4.51`；显式记录 `enable_thinking`，性能对照默认设为 `false` | 不与 Qwen2.5 功能基线直接比较性能 |
| 量化性能工作负载 | `Qwen/Qwen2.5-1.5B-Instruct` 或登记的更大模型 | 量化产物、显存、延迟、吞吐与质量复测 | 固定量化格式、校准/评测清单、backend 和输入 token | 只解释登记格式与执行路径 |
| 通信性能工作负载 | `Qwen/Qwen3-4B`、登记的 1.5B–7B 模型或合成大 Payload | Collective、FSDP/DeepSpeed、TP/CP/EP 和扩展效率 | 固定 world size、拓扑、并行度与每 rank 输入；小模型只做 smoke | 只有真实多卡或等价 Payload 才能形成扩展效率结论 |
| 生产候选 | 项目指定模型、产物和输入清单 | Part 05 与真实部署项目 | 固定 revision、tokenizer、chat template、硬件、runtime 和原始输入输出 | 结论只对登记环境与工作负载有效 |

Qwen2.5 与 Qwen3 不是同一性能基线。切换模型时必须新建候选记录，不覆盖原结果；至少登记 `model_id`、model/tokenizer revision、chat template、Thinking 模式、输入/输出 token、dtype 和目标格式。训练侧保存 tokenized manifest，Serving 侧保存应用 chat template 后的 `input_ids`。DeepSeek、GLM 等模型作为兼容性扩展，不替换默认基线。

## 执行栈版本与应用场景资产

执行栈版本分为四类信息，不能互相替代：

| 版本字段 | 含义 | 维护位置 |
| --- | --- | --- |
| 仓库声明 | requirements 中允许安装的包；未锁定时只表示依赖存在 | `requirements/*.txt` |
| 最低兼容 | 某个模型或功能首次可用的下限，不代表推荐性能版本 | 模型卡或上游官方文档链接 |
| 已验证快照 | 在指定硬件、驱动、CUDA、PyTorch 和 workload 上真实运行过的组合 | 结果 JSON、运行日志与本表 |
| 项目固定版本 | 本次实验实际使用的 package、镜像、commit 和构建选项 | 项目配置与 `runtime` 字段 |

当前五类执行栈的角色和版本状态如下。仓库目前未在 requirements 中固定它们的精确版本，因此除已登记快照外，不能把环境解析得到的任意最新版称为教程标准版本。

| 执行栈 | 主要场景 | 版本固定方式 | 仓库状态 / 已验证快照 | 使用要求 |
| --- | --- | --- | --- | --- |
| [Transformers](https://huggingface.co/docs/transformers/installation) | 模型与 tokenizer 参考实现、训练、固定输入前向 | package version + model/tokenizer revision；必要时记录 remote code revision | `base.txt` 未锁定；Qwen2.5-0.5B 要求 `>=4.37`，Qwen3-0.6B 要求 `>=4.51` | 作为语义基线；不据此宣称 Serving 调度性能 |
| [vLLM](https://docs.vllm.ai/en/stable/getting_started/installation/) | NVIDIA/AMD 等平台上的高吞吐 Serving、KV Cache 与调度实验 | package 或容器版本 + engine args + 硬件/CUDA 快照 | 独立 profile，未锁定；本机历史验证为 `0.11.0`，只代表该运行快照 | 正式比较需固定 model revision、请求清单、并发和输出 token |
| [SGLang](https://docs.sglang.ai/get_started/install.html) | Prefix Cache、RadixAttention、结构化生成与调度对照 | package/容器版本 + kernel/FlashInfer 依赖 + CUDA 快照 | 独立 profile，尚无统一已验证版本 | 不与 vLLM 混装；先验证安装矩阵和候选策略确实生效 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF、本地 CPU、端侧与 CPU/GPU offload | release/build tag + git commit + build backend/options | 尚无独立 profile 和已验证快照 | 同时记录 GGUF 元数据、线程、offload 层数及 CPU/GPU 后端；单独成组比较 |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/installation/index.html) | NVIDIA GPU 上的 Engine 构建、插件、量化与优化部署 | NGC image tag 或 wheel version + `TRT_LLM_GIT_COMMIT` + TensorRT/CUDA/GPU | 尚无独立 profile 和已验证快照 | 优先使用版本化 release 容器；先核对官方 support matrix，Engine 与构建环境一同登记 |

选择顺序是：先用 Transformers 固定模型语义和输入，再按目标选择 Serving 引擎；GPU 服务比较优先 vLLM，并用 SGLang做同口径候选；GGUF 与端侧路径使用 llama.cpp；确定 NVIDIA 专用优化目标后再引入 TensorRT-LLM。后四者都需要独立环境，不能通过覆盖同一个 PyTorch 环境完成升级。

结果 JSON 至少补充 `runtime.transformers_version`、`runtime.backend_name`、`runtime.backend_version`、`runtime.container_image`、`runtime.source_commit` 和 `runtime.build_options`；不适用字段可为 `null`，但不能省略 backend 名称和版本来源。

## 环境组合与教程入口

| 环境组合 | 主要验证范围 | 代表页面 / 项目 | 可以得到的证据 | 不能替代的证据 |
| --- | --- | --- | --- | --- |
| base + CPU PyTorch | 概念、机制、题目与测试 | Part 00–02 的 CPU 路径 | 状态转移、账本、输入契约 | GPU 显存、kernel、backend 吞吐 |
| base + CUDA PyTorch | 局部 GPU 操作与固定 PyTorch workload | 19、20、22、25、40、41、46、73–76 | 固定 workload 的时间、显存与机制执行证据 | 真实 backend 服务结论、多卡扩展性 |
| CUDA PyTorch + fine-tuning | SFT / LoRA 训练 | 62–64 | checkpoint、adapter、训练资源 | QLoRA 低比特路径、对齐 rollout |
| CUDA PyTorch + qlora | NF4 / PEFT 训练 | 26、65 | QLoRA smoke、adapter 交付 | GPTQ / AWQ / GGUF serving artifact |
| CUDA PyTorch + quantization | weight-only、FP8、GPTQ/AWQ、SmoothQuant | 25、40、41、52–54、82 | 局部低精度执行、校准和 compressed artifact | vLLM/SGLang 端到端吞吐与服务结论 |
| CUDA PyTorch + reinforcement-learning | 偏好优化与 RL 训练 | 84–86 | 训练接口、rollout / reward 记录 | 生产级多机 rollout 吞吐 |
| CUDA PyTorch + distributed | collective、多卡机制与真实训练 | 27–29、46–49、79、80 | collective、payload、rank 布局、负载、FSDP / DeepSpeed 训练与失败记录 | 真实推理服务 backend 对照 |
| CUDA PyTorch + vLLM | 推理 backend benchmark | 66–72、83 | backend 可用性、TTFT / TPOT / 吞吐 | SGLang 行为或跨 backend 优劣 |
| CUDA PyTorch + SGLang | SGLang backend benchmark | 66、69、70、72、83 | SGLang 策略与服务证据 | vLLM 行为或跨 backend 优劣 |
| llama.cpp 独立构建 / 镜像 | GGUF、本地 CPU、端侧和混合 offload | 部署与异构专题、后续 Part 05 项目 | 目标设备上的内存、延迟和可运行性 | vLLM/SGLang/TensorRT-LLM 的数据中心结论 |
| TensorRT-LLM release 容器 | NVIDIA 专用 Engine 与部署候选 | 部署与异构专题、后续 Part 05 项目 | 固定 Engine、GPU 与版本矩阵下的性能 | 其他硬件、其他 TensorRT/CUDA 组合 |
| CUDA PyTorch + profiling | trace 与性能归因 | 74、Part 03、Part 04 | profile、trace、热点与 overlap | 没有同口径 workload 的最终策略结论 |
| CUDA PyTorch + Triton / CUDA toolchain | kernel 与系统实验 | Part 03、Part 04 | 正确性、kernel / 系统性能证据 | 仅凭局部 kernel 推断端到端服务收益 |

## 预检与运行快照

| 阶段 | 现有入口 | 必须记录 | 输出状态 |
| --- | --- | --- | --- |
| 仓库与基础包检查 | `tools.project_runtime.bootstrap_project_root` | 项目根目录、Python、基础依赖 | 可继续或修复动作 |
| GPU / 能力预检 | `tools.project_runtime.environment_preflight` 或 `tools/environment_preflight.py` | CUDA 可用性、显存、必需包、模型 / backend 能力、结果目录 | `ok`、`warning`、`blocked` |
| 运行时快照 | `tools.project_runtime.runtime_snapshot` | 硬件、PyTorch、CUDA、dtype、关键包版本 | 写入结果 JSON |
| 项目结果 | 各 Notebook Step 5 | profile、runtime、workload、状态、失败原因、结果路径 | 证据等级与决策 |

示例预检：

```bash
python tools/environment_preflight.py --gpu --packages transformers --output benchmarks/results/preflight.json
```

backend、分布式和真实模型实验应在此基础上增加对应能力 profile 与项目专属检查；预检不会静默安装或替换驱动、CUDA PyTorch、vLLM 或 SGLang。

## 结果关联字段

环境信息不应在每份 JSON 中以不同字段名重复。新项目或迁移项目优先保留以下语义：

| 字段 | 含义 |
| --- | --- |
| `environment_profile` | 本次运行选择的基础 profile 与能力 profile 组合 |
| `runtime` / `environment` | 真实硬件、PyTorch、CUDA、dtype、关键包版本快照 |
| `preflight` | `ok` / `warning` / `blocked` 与修复动作 |
| `workload` | 与环境无关的输入、并发、长度、重复与随机种子条件 |
| `evidence_level` | 本次运行实际获得的证据；不由 profile 自动推断 |
| `failure` | 缺包、OOM、backend 启动、通信或配置失败原因 |

## 已验证环境与待补资产

| 类别 | 当前已知资产 | 维护动作 |
| --- | --- | --- |
| CPU-first | 题目区、测试区与 dry-run 是默认路径 | 保持不依赖 CUDA / backend |
| 单卡 PyTorch | 73–76 有历史 GPU JSON；74 有 trace | 补统一 profile / runtime 记录，不改写历史结果 |
| 单卡 backend | 66 有 vLLM 历史结果；67–72 提供默认关闭入口 | 固定 model revision、workload 与 backend version 后再采集正式数据 |
| 多卡通信 | 79–81 有预检、配置和聚合入口 | 补实际拓扑、NCCL / backend 与多卡结果快照 |
| Kernel / 系统 | Part 03 / 04 有 GPU-required 内容 | 在 Part 03 / 04 资产表逐项补 toolchain、GPU 架构与 trace 状态 |

## 维护边界

- 不提交账号、令牌、私有 endpoint、模型缓存、大型 checkpoint、编译缓存、binary 或临时 trace。
- 驱动由主机提供；`nvidia-smi` 报告的 CUDA 能力不等于 PyTorch wheel 的 CUDA 版本。
- Colab、ModelScope 和云端预装环境优先复用当前内核的 PyTorch；普通依赖可显式补装，CUDA 相关运行时由学习者选择并重启。
- 新 backend 或训练框架先增加独立 profile 或官方版本矩阵，再登记到本表；不要塞入 `requirements.txt`。
