# 量化与压缩实验资产表

本表回答“某个量化实验具体使用什么输入、比较什么候选、输出到哪里、当前证据到哪一步”。环境创建命令见[实验 Quickstart](./quickstart.md)；Profile 与隔离关系见[环境资产索引](../../gpu_environment_assets.md)，实际依赖版本和硬件快照写入结果 JSON。本页不复制安装命令和逐包版本。

## 环境角色

机制、训练和服务使用相互隔离的环境。依赖解析成功只代表环境可用，不代表量化路径、计算内核或性能已经验证。

| 环境 ID | 角色与直接入口 | 证据边界 |
|:---|:---|:---|
| `Q-CPU` | TorchAO、LLM Compressor、compressed-tensors 与 HQQ 的小规模框架核对 | 只形成 CPU 框架语义证据，不作为 Kernel、显存或性能证据 |
| `Q-CUDA-MECHANISM` | 25、40、41、53、54、Mixed-bit、QAT | 单卡真实模型机制、产物生成或局部执行证据；不等于 Serving 证据 |
| `Q-QLORA` | 26、65 | BF16 LoRA / NF4 QLoRA 配对训练和适配器交付 |
| `INF-VLLM-011-CU128`（专题角色：`Q-VLLM`） | 67、82、83 的主服务后端 | 量化产物加载、实际请求和服务性能证据；使用 vLLM 0.11 / PyTorch 2.8 / CUDA 12.8 锁定组合 |
| `Q-SGLANG` | 67、83 的可选同口径对照 | 必须固定同一模型、产物、硬件和请求清单 |
| `Q-GGUF` | 67 的 llama.cpp / 端侧扩展 | 尚未建立默认环境，不进入当前主实验链 |

## 模型与数据口径

| 角色 | 固定选择 | 使用规则 |
|:---|:---|:---|
| 机制矩阵基线 | `Qwen/Qwen2.5-0.5B-Instruct` | 用于真实层形状、多候选和质量对照；模型与分词器使用同一 commit |
| 规模交叉候选 | `Qwen/Qwen2.5-1.5B-Instruct` | 用于 QLoRA 规模复测和后续服务候选；不得与 0.5B 结果合并 |
| 新架构兼容候选 | `Qwen/Qwen3-0.6B` | 只验证新架构、分词器或后端兼容性，单独登记结果 |
| 校准与质量数据 | `Salesforce/wikitext / wikitext-2-raw-v1` | 固定 revision；校准、训练、验证和测试按页面职责隔离并保存样本 ID 哈希 |
| 无数据集工作负载 | 固定层、提示、Token 长度和随机种子 | 25、41、54 使用工作负载清单，不为执行路径验证下载无关数据集 |

模型、数据或量化格式改变后必须使用新的运行标签。离线 JSONL 只有在样本身份、划分和内容哈希可复核时，才能替代 Hugging Face 数据快照。

## 机制实验注册表

每个实验只在本表登记一次。详细候选参数留在对应 Notebook 的 5.1，完整指标保存在结果 JSON。

| 实验 ID / 页面 | 环境与输入 | 固定对照 | 结果资产与当前结论 |
|:---|:---|:---|:---|
| `QUANT-W8A16-LAYER` / 25 | `Q-CUDA-MECHANISM`；0.5B 的 `q_proj` / `gate_proj` / `down_proj`；1 / 32 / 256 Token | FP16 Linear → torchao INT8 仅权重量化 | `benchmarks/results/25_torchao_w8a16_layer_microbenchmark_default1.json`；容量和质量通过，只有 `q_proj`、32 Token 达到明显加速 |
| `QUANT-QLORA-MECHANISM` / 26 | `Q-QLORA`；0.5B 与 1.5B；固定训练 / 验证 / 测试划分 | BF16 LoRA → NF4 QLoRA | 结果：`benchmarks/results/26_lora_qlora_mechanism_matrix_default2.json`；适配器：`benchmarks/artifacts/26_qlora/default2/`；1.5B 显存收益伴随质量和单步时间代价 |
| `QUANT-WEIGHT-PTQ` / 40 | `Q-CUDA-MECHANISM`；0.5B；互斥校准 / 评测数据 | BF16 → RTN / GPTQ / AWQ W4A16 | 结果：`benchmarks/results/40_llmcompressor_gptq_awq_default11.json`；产物：`benchmarks/artifacts/40_gptq_awq/default11/`；GPTQ 入选下游检查，尚无 Serving 性能结论 |
| `QUANT-SMOOTHQUANT` / 53 | `Q-CUDA-MECHANISM`；0.5B；互斥校准 / 评测数据 | BF16 / 普通 W8A8 → `alpha=0.5/0.7/0.8/0.9` | 结果：`benchmarks/results/53_llmcompressor_smoothquant_default111.json`；产物：`02_PyTorch_Algorithms/benchmarks/artifacts/53_smoothquant/default111/`；`alpha=0.5` 入选，真实 W8A8 Kernel 仍待验证 |
| `QUANT-FP8-RUNTIME` / 41 | `Q-CUDA-MECHANISM`；0.5B；提示 × Token 长度矩阵 | BF16 → FP8 仅权重 / dynamic 逐张量 / dynamic PerRow | `benchmarks/results/41_torchao_fp8_runtime_default13.json`；路径完整，但门槛未全部通过，决策为 `tune_or_fallback` |
| `QUANT-KV-RUNTIME` / 54 | `Q-CUDA-MECHANISM`；0.5B；上下文 × 高精度残留窗口 | Transformers `DynamicCache` → HQQ `QuantizedCache` | `benchmarks/results/54_transformers_quantized_cache_matrix_default18.json`；长上下文出现容量收益，但有延迟和质量代价 |
| `QUANT-MIXED-BIT-ALLOCATION` / Mixed-bit | `Q-CUDA-MECHANISM`；0.5B；全层 Block 敏感度与独立评测 | 统一 INT8 / INT4 → 12.5% / 25% / 50% 高敏感 Block 保护 | `benchmarks/results/quant_mixed_bit_mechanism_default3.json`；接受 3/24 最小容量候选，6/24 为稳健候选；无加速结论 |
| `QUANT-QAT-CLOSED-LOOP` / QAT | `Q-CUDA-MECHANISM`；0.5B 中间 Block；训练 / 验证 / 测试隔离 | INT4 PTQ → QAT 训练 5 / 20 / 50 步后转换 | 结果：`benchmarks/results/quant_qat_closed_loop_default2.json`；产物：`benchmarks/artifacts/quant_qat/default2/block_12_50_steps.pt`；证明局部质量恢复，不证明完整模型压缩或生产 Kernel |

## 产物与执行后端

| 路径 | 交付物 | 验证入口 | 明确限制 |
|:---|:---|:---|:---|
| INT8 仅权重量化 | 当前进程内 torchao 模块 | 25 / PyTorch | 不自动形成 vLLM 可加载产物 |
| NF4 QLoRA | 适配器、基础模型与量化配置 | 26 → 65 / Transformers | 适配器不是 GPTQ、AWQ 或 GGUF 产物 |
| GPTQ / AWQ W4A16 | 带量化元数据的模型目录 | 40 → 82 → 67 / vLLM | 产物生成成功不证明加载器和 Kernel 支持 |
| SmoothQuant W8A8 | W8A8 压缩产物 | 53 → 82 → 兼容后端 | 必须单独确认真实 W8A8 计算内核 |
| FP8 路径 | 进程内模型或运行配置 | 41 / PyTorch | 存储类型、累加精度和 Kernel 分别取证 |
| KV Cache 量化 | Transformers 运行时 Cache 状态 | 54 / Transformers | 不代表 vLLM/SGLang 的量化 Cache 路径 |
| GGUF | `.gguf` 文件 | 67 扩展 → llama.cpp | 使用独立环境和端侧证据，不交给 vLLM |

vLLM 是 GPTQ/AWQ 服务执行的主验证后端；SGLang 只作同口径对照。TensorRT-LLM 只有在建立版本化容器、Engine 构建记录和插件支持矩阵后，才能加入默认候选。

## 项目出口

| 稳定项目 ID | 输入 → 输出 | 当前承载与主责 |
|:---|:---|:---|
| `PT-QLORA-SELECTION` | 固定数据、rank、steps 与预算 → 训练 JSON、适配器清单 | 65；后训练主归属，量化训练交叉入口 |
| `QUANT-ARTIFACT-GATE` | `quantization-artifact-candidate/v1` 候选包 → 通过血缘、文件哈希、带方向质量记录和真实加载器检查的产物 | 82；量化产物准入，只要求一个候选包路径 |
| `QUANT-INFERENCE-DEPLOY` | 浮点基线与合格产物 → 同模型后端部署结果 | 67；vLLM 主验证，GGUF 走独立路径 |
| `QUANT-SERVING-PROFILE` | 同一产物与请求清单 → 延迟优先、均衡、吞吐优先结果 | 83；服务配置与决策收口 |

### 项目验证顺序

量化项目按证据阶段运行，不按 Notebook 编号排序。每一阶段只消费一个必需的直接上游资产；更早阶段的模型、数据和产物血缘由该资产继续携带。

| 顺序 | 项目与输入 | 本阶段验证 | 通过后输出 |
|:---|:---|:---|:---|
| 0 | `INF-VLLM-011-CU128` 与候选生成环境 | Profile 预检、固定模型 revision、可写结果目录和候选包 schema | 可复测环境与候选包 |
| 1 | `QUANT-ARTIFACT-GATE` / 82，输入一个候选包 | 文件角色与哈希、数据隔离、质量方向、目标 loader 的真实加载 | 合格产物 companion 与稳定引用 |
| 2 | `QUANT-INFERENCE-DEPLOY` / 67，输入 82 的稳定引用 | 同硬件、同 backend、同模型 revision 和同 workload 下的浮点—量化配对执行 | 部署 companion 与稳定引用 |
| 3 | `QUANT-SERVING-PROFILE` / 83，输入 67 的稳定引用 | 延迟优先、均衡、吞吐优先三类 profile 的重复测量与分方向 SLO | Profile 结果、决策和交接引用 |
| 4 | 全链资产复核 | schema、血缘、失败记录、结果路径和唯一运行标签 | 可进入 Part 05 的项目证据 |

当前工作区已登记 40 的 GPTQ 候选产物，但 82、67、83 的项目结果 JSON 尚未找到；这三步仍需按顺序完成，不能把机制实验的局部结论当作量化服务链的验收结果。

82 的 `accept` 只表示产物可以进入部署验证；67 的 `accept` 表示固定工作负载下量化候选能够正确执行并达到本页门槛；83 的 `accept` 表示至少一个服务 profile 满足对应 SLO。驱动、backend、候选产物或固定 workload 发生变化时，从受影响的最早阶段开始复测，不覆盖历史结果。

## 量化契约增量

通用结果字段与证据等级遵循[Part 02 结果资产与证据契约](../../part02_result_assets.md)。量化实验额外记录以下内容。

| 量化特有层 | 必须固定或记录的字段 |
|:---|:---|
| 表示配置 | 量化对象、格式、bits、scheme、缩放因子、零点、粒度、分组大小 |
| 数据隔离 | 校准 / 训练 / 验证 / 测试职责、revision、样本 ID 哈希和更新统计的阶段 |
| 产物身份 | `artifact_id`、manifest、文件哈希、量化工具版本、基础模型版本 |
| 路径证据 | 实际 dtype、量化状态、loader、Kernel 或性能剖析器事件、fallback 原因 |

只有通用契约和量化增量同时完整，结果才能从“入口已设计”升级为“已验证快照”。
