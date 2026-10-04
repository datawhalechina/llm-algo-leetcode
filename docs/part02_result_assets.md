# Part 02 结果资产与证据契约

> 统一审阅入口见 [教程资产总登记与审阅台账](./asset_registry.md)。

本文是维护者用的结果资产清单：回答“某个小节产出的 JSON、artifact、指标和决策字段是否能被下游项目复用”。它不替代学习路径；项目入口与学习顺序仍以 [Part 02 项目资产表](./02_PyTorch_Algorithms/2_10.md) 为准。

## 使用方式

新增或改造 Part 02 的实验时，先在下表找到最接近的结果契约：沿用其公共字段，把本节特有指标放入专用字段；没有合适契约时，先记录为“待建设”，不要把临时 JSON 写成新的隐式标准。

所有会改变模型输出的候选方案，还应同时遵守 [维护规范中的跨专题质量门槛](./maintenance.md#模型评测与跨专题质量门槛)。性能指标不能单独替代质量、失败状态或最终决策。

## 稳定项目身份与公共外层

60–86 的数字编号只负责定位 Notebook；跨专题引用、结果记录和后续调号使用稳定语义 ID。唯一主归属、专题内顺序和目标契约统一登记在 [`benchmarks/project_registry.json`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/benchmarks/project_registry.json)，学习顺序见 [2.10 项目链](./02_PyTorch_Algorithms/2_10.md#project-chains-and-stable-ids--项目链与稳定身份)。

公共实验外层由以下模块维护：

| 模块 | 职责 | 不负责 |
| --- | --- | --- |
| `tools.project_contracts` | 构造 `ExperimentSpec`、`ArtifactManifest`、`RunResult`，读取项目注册表并保存 companion JSON | 不覆盖原始 backend / profiler 结果，不决定机制专属指标 |
| `tools.project_adapters` | 用同一接口包装 PyTorch、Transformers 或 serving runner，统一预检失败与执行失败记录 | 不安装依赖，不修改 workload，不把 smoke 自动升级为 benchmark |
| 既有专项 schema | 继续维护训练、推理、量化、profiling 等稳定专属字段 | 不再各自定义第二套项目身份和主归属 |

项目 Notebook 可以独立构造默认 `ExperimentSpec` 并运行；消费其他项目 artifact 时通过 `ArtifactManifest` 显式引用，不以固定文件路径形成强制执行链。

## 证据等级

| 等级 | 含义 | 可支持的结论 |
| --- | --- | --- |
| `theoretical` | 公式、账本或静态推导 | 机制方向与理论上限 |
| `cpu_simulation` | CPU 状态机、成本模型或合成时间轴 | 状态转移、约束与候选比较 |
| `gpu_smoke` | 可运行的 GPU 路径或局部算子探针 | 环境与接口可用性 |
| `real_benchmark` | 固定 workload 下的真实模型或后端复测 | 当前配置下的性能比较 |
| `profiler_trace` | 可回看的 trace / profiler 证据 | 瓶颈归因与优化解释 |

没有固定 workload、真实模型或 trace 时，不得把较低等级写成稳定 benchmark。

## 已有结果契约

| 契约 / 版本 | 覆盖范围 | 公共结构或字段 | 当前状态 |
| --- | --- | --- | --- |
| `mechanism-gpu-probe`（未版本化） | 非项目机制页的可选 GPU 实验，例如 19、20、22、24、25、26、34–41、46–49 | workload、baseline / candidate、环境、局部指标、证据等级、失败与结论 | **部分统一**：5.1–5.4 结构已在页面中形成，但尚无共享 schema 与统一校验器 |
| `fine-tuning-project/v1` | 62–65 | `config`、`baseline`、`candidates`、`quality`、`resources`、`artifacts`、`decision`、`environment` | **工具已版本化**：由 `tools/fine_tuning_result_schema.py` 提供；60 已由稳定项目注册表归入架构链 |
| `inference-benchmark/v1` | 66、68–70 | `project`、`strategy`、`role`、`config`、`metrics`、`quality`、`status`、`evidence_level`、`failure`、`decision`、`strategy_metrics` | **已版本化**：由 `tools/inference_result_schema.py` 生成；66 也承担浮点基线入口 |
| `training-memory-benchmark/v1` | 73、75、76 | `config`、`budget`、`quality_floor`、`candidates`、`summary`、`decision` | **已版本化**：由 `tools/training_memory_result_schema.py` 兼容归一化历史结果 |
| `profiling-optimization-benchmark/v1` | 74 | `config`、`candidates`、`profile`、`bottleneck`、`validation`、`decision` | **已版本化**：保留 profiler 专项证据；后续可评估是否与训练显存契约抽取共用外层 |
| `quantization-evidence/v1` | 82 → 66 → 67 → 83 的量化产物、浮点基线、部署与 serving 证据链 | `stage`、`artifact`、`hardware`、`backend`、`workload`、`metrics`、`status`、`evidence_level`、`failure`、`decision` | **已版本化**：由 `tools/quantization_result_schema.py` 维护；原始 backend JSON 仍应保留 |
| `distributed-benchmark/v1` | 79–81 与 `PAR-MODEL-INTERNAL` | `project`、`role`、`workload`、`hardware`、`metrics`、`artifacts`、`evidence_level`、`failure`、`decision` | **外层已版本化**：由 `tools/distributed_result_schema.py` 维护；79、80 已可通过聚合器生成 companion，81 与模型内部并行项目仍待接入真实执行 adapter |
| `architecture-evaluation/v1` | 60、61、71、77、78 | 统一身份、运行环境、workload、候选、质量、成本、失败、证据等级与决策；各项目保留专属机制字段 | **工具与 Notebook 均已接入**：由 `tools/architecture_result_schema.py` 维护；61 是聚合出口，其余四页生产局部证据；当前仍待采集固定环境下的真实结果 |
| 对齐与模型评测（待建设） | 50、51、84–86 | 建议复用 `eval_dataset`、`eval_dataset_revision`、`evaluation_protocol`、`quality_metrics`、`quality_threshold`、`failure_samples`、`decision` | **待建设**：协议已在后训练优化 Task6 与维护规范中说明，尚无 `model-evaluation/v1` |

## 现状审计快照（2026-09-30）

审计范围为 Part 02 的 60–86，共 27 个项目 Notebook。这里的“接入”指 Notebook 实际引用或调用结果契约工具，不把仅存在于 `tools/` 目录中的脚本误记为已迁移。

| 资产信号 | 数量 | 结论 |
| --- | ---: | --- |
| 已有结果路径或 JSON 写入逻辑 | 21 / 27 | 多数项目已经知道结果写到哪里，但路径与文件命名尚未形成一套声明机制 |
| 已标记 `tutorial-question` / `tutorial-test` / `tutorial-answer` | 16 / 27 | 新项目页的教学代码标签较完整；63–65、71、73–76、84–86 仍需单独核查或迁移 |
| 已标记 GPU 配置、执行或读取 cell | 16 / 27 | 新版 GPU 分层已覆盖一部分项目；73–76 有实际 GPU 代码，但尚未统一为 metadata tag |
| 工作区中已有实际 JSON / trace 结果 | 6 个项目族 | 当前可回看的实测主要集中在 60、66、73–76；其余多数是可生成路径或待学习者复测位置 |
| Notebook 已直接接入版本化 schema / 聚合工具 | 9 / 27 | 60、61、71、77、78 接入架构 schema；79、80 接入分布式聚合；82、83 接入量化 schema。其余 schema 目前主要是可复用工具，而非已经全面接入的事实 |

### 项目级资产矩阵

| 范围 | 当前资产 | 结果路径状态 | 契约接入状态 | 审计判断 |
| --- | --- | --- | --- | --- |
| 60 | CPU / GPU、checkpoint 与 adapter 历史资产 | 有 60 专属目录和已提交 adapter、JSON | 已接入 `architecture-evaluation/v1` producer companion | 原始结果保持不变；匹配质量缺失时只能形成 `tune` |
| 61 | 架构审计卡、CPU 账本 | 有按时间戳写入路径 | 已接入 `architecture-evaluation/v1` audit record | 只消费 producer 路径、语义 ID 与哈希；缺失、失败或篡改证据不能 `accept` |
| 62–64 | SFT、LoRA 变体、数据质量结果 | 有项目 JSON；62 还保留 60 的历史兼容入口 | 未直接接入 fine-tuning schema | 先盘点旧/新路径与 artifact 所有权，再迁移 |
| 65 | QLoRA 选型、GPU smoke、adapter manifest | 有 GPU JSON 与 manifest 路径 | 使用自定义 `qlora-benchmark/v1`，未接入通用 schema | 是后训练与量化的交叉资产，需先决定主契约再迁移 |
| 66–70 | 推理基线、量化部署、投机、缓存、调度 | 66 有现存实测；67–70 主要提供生成路径和 manifest | Notebook 未直接调用 `inference-benchmark/v1` | schema 与页面并存但未形成强制连接，是最高优先级的事实校验点 |
| 71–72 | MLA / KV 表示、单卡部署对照 | 72 有部署目录；71 已增加 backend 结果位置 | 71 已接架构 producer；72 仍走推理部署契约 | 两页继续分属架构评测与推理部署，不能混合处理 |
| 73–76 | 训练性能、profiling、预算与 checkpoint/offload | 有 JSON、trace、跨节读取链 | 页面使用原始结果；归一化工具存在但未在 Notebook 中显式调用 | 有最完整的历史证据，应保留原始数据并只增加 companion 记录 |
| 78 | 访问模式 CPU 账本与 SDPA mask probe | 有 GPU 结果路径 | 已接入 `architecture-evaluation/v1` producer companion | 固定为受控比较；明确 `sparse_kernel_executed=false`，不得宣称专用稀疏 backend 收益 |
| 77 | 长序列状态与混合记忆对照 | 有 GPU 结果路径 | 已接入 `architecture-evaluation/v1` producer companion | 默认使用观察性比较；单个 needle smoke 通过仍只能形成 `tune` |
| 79–81 | 并行、MoE dispatch、分布式推理 | 有项目目录与 smoke 路径 | 79、80 调用聚合器；81 尚未接入 | 共同字段已出现，适合优先版本化为分布式契约 |
| 82–83 | 量化 artifact gate 与 serving profile | 有专属目录与标准化 companion JSON 路径 | 已接入 `quantization-evidence/v1` | 当前最完整的端到端 asset / deployment 链样板 |
| 84–86 | DPO、GRPO、在线对齐 | 当前未见稳定的结果路径与教学 metadata | 无模型评测 schema | 先定义评测数据、质量阈值和回滚记录，再设计结果文件 |

### 工作区中已存在的可回看证据

| 项目族 | 已有资产 | 注意事项 |
| --- | --- | --- |
| 60 | 多份 LoRA JSON、adapter、tokenizer | 既有 `60_real_lora/` 是历史资产；62 已引用它，迁移时不能移动或覆盖 |
| 66 | 多份 vLLM 并发结果与 backend log | 可作为 67 量化部署的浮点基线，但必须记录匹配 workload 与模型 revision |
| 73 | 多份真实 GPU 训练 JSON | 存在不同 dtype / sequence length 的历史运行，比较前需核对统一条件 |
| 74 | baseline/checkpoint trace 与 profiling JSON | trace 是归因证据，不能仅以汇总耗时替代 |
| 75 | 多份显存预算决策 JSON | 依赖 76 结果，需保留来源关系 |
| 76 | 多份 checkpoint/offload GPU JSON | 依赖 73 基线，需保留 workload 一致性检查 |

其余项目目前主要提供可复现的结果路径、配置或读取逻辑；未在工作区发现相应实测文件，不应在资产表中标为“已有 benchmark”。

### 已确认与待评审决策

| 优先级 | 需要先确认的决策 | 原因 | 本轮动作 |
| --- | --- | --- | --- |
| 已确认 | 60 归入架构链；65 保持后训练主归属、量化交叉入口 | 避免训练 schema 继续按数字范围推断项目身份 | 已写入项目注册表；不迁移历史文件 |
| 已确认 | 72 是独立的单卡部署项目 | 它比较模型、artifact 与 backend 组合，不只服务量化链 | 使用推理目标契约；量化 artifact 作为可选输入 |
| 已确认 | 79–81 与 `PAR-MODEL-INTERNAL` 采用一份版本化分布式结果 envelope | 四页共享硬件、拓扑、通信与扩展证据 | `distributed-benchmark/v1` 已建立；下一步接通模型内部并行 adapter 并采集真实多卡结果 |
| P1 | 66 / 67 / 82 / 83 的关联键 | 浮点基线、artifact、量化部署与 serving profile 需要可追溯 | 明确 `model_revision`、`artifact_id`、`workload_id` 的最小约束 |
| 已确认 | 架构项目使用统一质量—成本外层和项目专属机制字段 | 60、61、71、77、78 的对象不同，不能只复用性能字段 | `architecture-evaluation/v1` 已建立并接入五页；下一步采集固定环境结果 |
| P2 | 对齐项目的评测数据与回滚记录 | 84–86 尚无稳定结果资产；先定义错误的代价较低 | 先写 `model-evaluation/v1` 草案 |

### 架构项目职责与目标契约

架构项目不按数字范围推断职责，而按稳定语义 ID 组成证据链。60、78、71、77 分别生产局部证据，61 只登记来源并完成跨机制审计。

| 顺序 | 语义 ID | 项目职责 | 专属记录 | 当前成熟度 |
| ---: | --- | --- | --- | --- |
| 1 | `ARCH-BLOCK-STABILITY` | 比较 Decoder Block 的单变量结构变体 | `changed_field`、Block 配置、梯度/稳定性、匹配 checkpoint | CPU、GPU 路径和统一外层已接；匹配质量缺失时只能形成 `tune` |
| 2 | `ARCH-ATTENTION-ACCESS` | 比较全局、局部与稀疏访问模式 | `attention_pattern`、访问密度、长程覆盖、kernel/backend 能力 | 已有 CPU mask 与 SDPA 证据；专用稀疏 backend 尚未接入 |
| 3 | `ARCH-MLA-KV` | 比较 MHA、GQA、MLA 的状态表示 | `representation`、KV bytes/token、latent/RoPE 维度、backend capability | 已有理论账本和成对 backend 执行入口；真实 backend 结果仍是最薄弱环节 |
| 4 | `ARCH-LONG-SEQUENCE-MEMORY` | 比较显式历史、递推状态与混合记忆 | `state_representation`、`layer_schedule`、长度阶梯、质量—成本曲线 | 结果结构最接近目标契约；默认属于跨模型观察 |
| 5 | `ARCH-MODEL-AUDIT` | 汇总前序证据并形成架构审计卡 | 生产项目语义 ID、结果路径/哈希、配置差异、综合判断 | 已有审计卡；不得重新生成局部 benchmark 指标 |

目标公共外层如下。字段允许为 `null`，但不得通过省略字段掩盖没有采集的证据。

| 字段组 | 最小字段 | 说明 |
| --- | --- | --- |
| 身份与来源 | `schema_version`、`semantic_id`、`source` | `source` 记录 Notebook、producer 与原始结果引用 |
| 运行条件 | `runtime`、`workload` | 固定模型 revision、tokenizer、输入、长度阶梯、dtype、hardware 与 backend |
| 对照对象 | `baseline`、`candidate` / `candidates` | 只改变已声明的架构变量；跨模型比较必须标记为 observational |
| 证据 | `quality`、`cost`、项目专属字段 | 质量与成本必须成对出现；机制字段不强塞入公共指标表 |
| 可信度 | `failure`、`evidence_level` | 记录不支持、回退、OOM、缺失质量与可复测入口 |
| 决策 | `decision` | 使用 `accept / tune / reject`，并记录原因与下一步 |

`tools/architecture_result_schema.py` 已实现 producer / audit 两类记录、受控/观察性比较标记、结果哈希引用和禁止隐式覆盖。接入顺序保持为 60 → 78 → 71 → 77 → 61：先让局部生产者生成 companion 记录，再让 61 消费稳定引用；原始 JSON 不覆盖。

## 项目注册表（维护视图）

项目只保留一个主专题和一个目标主契约；交叉专题只提供入口。稳定语义 ID 与顺序以机器可读注册表为准，下表用于人工审阅主归属和交付物。下游消费关系统一放在下一节的资产关系表，避免在此重复。

### 架构与后训练

| 项目 | 主专题 | 交叉入口 | 目标契约 | 交付资产 |
| --- | --- | --- | --- | --- |
| 60 Decoder Block 稳定性 | 模型架构演进 | 后训练、性能 | `architecture-evaluation/v1` | Block 稳定性与成本 |
| 61 架构探索 | 模型架构演进 | 推理、性能 | `architecture-evaluation/v1` | 架构审计卡 |
| 62 指令微调 | 后训练优化 | 数据工程、性能 | `fine-tuning-project/v1` | checkpoint / adapter、训练报告 |
| 63 LoRA 变体 | 后训练优化 | 性能 | `fine-tuning-project/v1` | 变体比较、adapter |
| 64 SFT 数据质量 | 后训练优化 | 数据工程、模型评测 | `fine-tuning-project/v1` | 数据质量与训练效果 |
| 65 QLoRA 选型 | 后训练优化 | 量化压缩、部署异构 | `fine-tuning-project/v1` | 训练报告、adapter manifest |

### 推理与性能

| 项目 | 主专题 | 交叉入口 | 目标契约 | 交付资产 |
| --- | --- | --- | --- | --- |
| 66 推理性能比较 | 推理优化 | 量化、性能、部署 | `inference-benchmark/v1` | 匹配浮点 baseline |
| 67 量化推理与部署 | 量化压缩 | 推理、部署、性能 | 推理原始记录 + 量化 companion | 量化部署比较 |
| 68 投机解码 | 推理优化 | 性能 | `inference-benchmark/v1` | 接受率与净收益 |
| 69 Prefix Cache | 推理优化 | 性能、部署 | `inference-benchmark/v1` | 命中、复用与收益 |
| 70 Serving 调度 | 推理优化 | 性能、部署 | `inference-benchmark/v1` | 延迟、公平性与队列证据 |
| 71 MLA / KV 表示 | 模型架构演进 | 推理、性能 | `architecture-evaluation/v1` | 表示成本与质量 |
| 72 单卡部署比较 | 推理优化 | 量化、部署异构 | `inference-benchmark/v1` 的 deployment 扩展 | 单卡 backend 对照 |
| 73 训练性能分析 | 性能优化 | 后训练、反向传播与训练 | `training-memory-benchmark/v1` | 训练性能基线 |
| 74 Profiling 驱动优化 | 性能优化 | 后训练、算子优化 | `profiling-optimization-benchmark/v1` | trace、瓶颈与调优证据 |
| 75 显存预算压缩 | 性能优化 | 后训练 | `training-memory-benchmark/v1` | 预算决策记录 |
| 76 Checkpoint / Offload | 性能优化 | 后训练、反向传播与训练 | `training-memory-benchmark/v1` | 内存策略比较 |
| 77 长序列记忆架构 | 模型架构演进 | 推理、性能 | `architecture-evaluation/v1` | 长序列质量—状态成本 |
| 78 Attention 访问模式 | 模型架构演进 | 推理、算子优化 | 架构契约 + GPU probe | 访问密度、质量与成本 |

### 分布式、量化与对齐

| 项目 | 主专题 | 交叉入口 | 目标契约 | 交付资产 |
| --- | --- | --- | --- | --- |
| 79 训练状态并行 | 通信与并行 | 性能、后训练 | `distributed-benchmark/v1` | DDP / FSDP / ZeRO 状态与扩展证据 |
| 80 MoE 专家并行 | 通信与并行 | 架构、性能 | `distributed-benchmark/v1` | dispatch / overflow / 通信证据 |
| 81 分布式推理 | 通信与并行 | 推理、部署、性能 | `distributed-benchmark/v1` | 多卡 serving 比较 |
| 82 量化产物评估 | 量化压缩 | 部署、模型评测 | `quantization-evidence/v1` | artifact gate |
| 83 量化 Serving Profile | 量化压缩 | 推理、部署、性能 | `quantization-evidence/v1` | serving profile 与决策 |
| 84 DPO 偏好项目 | 后训练优化 | 数据工程、模型评测 | `model-evaluation/v1` | 偏好优化与质量记录 |
| 85 GRPO 对齐项目 | 后训练优化 | Agent、数据工程、模型评测 | `model-evaluation/v1` | 组内奖励与质量记录 |
| 86 在线 DPO benchmark | 后训练优化 | 数据工程、模型评测 | `model-evaluation/v1` | freshness、回滚与上线判断 |

## 资产关系表草案（待评审）

这张表只记录真实的生产—消费关系。专题之间的“相关阅读”或知识前置不自动形成资产依赖；只有需要读取、验证、比较或复用某项结果时才登记关系。

| 资产 ID（建议） | 生产项目 | 资产内容 | 消费项目 | 关系 | 使用前提 |
| --- | --- | --- | --- | --- | --- |
| `architecture.block_stability` | 60 | Block 稳定性、变体成本与 checkpoint 记录 | 61、Part 05 架构选型 | `extends` / `consumes` | 相同或可声明差异的模型配置与 workload |
| `architecture.audit_card` | 61 | 架构 config、参数 / KV / 状态账本 | 77、78、Part 05 | `compares_against` | 模型 revision 与成本口径可追溯 |
| `training.sft_checkpoint` | 62 | SFT checkpoint / adapter 与训练报告 | 84、85、65 | `consumes` | 数据版本、chat template、质量门槛明确 |
| `training.lora_variant_report` | 63 | LoRA 变体效果与资源比较 | 65、Part 05 | `compares_against` | 相同模型、数据与训练预算 |
| `data.sft_quality_report` | 64 | 数据版本、质量过滤与效果记录 | 62、84、85 | `validates` | 训练集与评测集隔离 |
| `training.qlora_adapter` | 65 | QLoRA adapter、训练质量与来源 manifest | 82 | `consumes`（间接） | 必须先合并或导出为可部署 artifact；adapter 本身不是量化部署产物 |
| `inference.floating_baseline` | 66 | 匹配模型 / backend / workload 的浮点结果 | 67、72、83 | `compares_against` | revision、dtype、workload 与并发口径匹配 |
| `quantization.deployment_candidate` | 67 | 量化候选的部署结果 | 72、83 | `consumes` | artifact 已通过 82 gate；存在匹配 66 baseline |
| `inference.speculative_evidence` | 68 | proposal / verify / acceptance 与净收益 | Part 05 推理服务 | `consumes` | draft / target 与 workload 明确 |
| `inference.prefix_cache_evidence` | 69 | 命中、复用、维护代价与延迟 | Part 05 推理服务 | `consumes` | 共享前缀分布与缓存策略明确 |
| `inference.scheduler_evidence` | 70 | 排队、公平性、吞吐与策略结果 | Part 05 推理服务、81 | `extends` | 请求到达模式和调度策略可复现 |
| `architecture.mla_state_cost` | 71 | MLA / KV 表示成本与质量 | 61、77、Part 05 | `extends` | Attention 配置与序列长度明确 |
| `deployment.single_gpu_candidate` | 72 | 单卡 backend / artifact 组合的部署结果 | 83、Part 05 | `consumes` | 量化 artifact 时必须携带 82 gate 引用 |
| `performance.training_baseline` | 73 | 固定训练 workload 的时间、显存与质量基线 | 74、75、76 | `compares_against` | dtype、batch、seq_len、工作负载一致 |
| `performance.profiler_trace` | 74 | baseline / candidate trace 与瓶颈解释 | Part 05 性能项目 | `validates` | 对应候选必须有同口径运行结果 |
| `performance.memory_strategy` | 76 | checkpoint / offload 的显存—时间—质量比较 | 74、75 | `compares_against` | 使用 73 的匹配基线 |
| `performance.budget_decision` | 75 | 预算、策略筛选与质量门槛 | Part 05 性能项目 | `consumes` | 来源策略结果与预算约束完整 |
| `architecture.long_context_evidence` | 77、78 | 长序列状态与访问模式的质量—成本证据 | 61、Part 05 | `extends` | 任务质量、长度与成本均被记录 |
| `distributed.training_state_evidence` | 79 | DDP / FSDP / ZeRO 的训练状态与扩展证据 | 性能优化、后训练、Part 05 | `consumes` | 拓扑、每 rank 状态、通信、吞吐与 workload 记录 |
| `distributed.model_internal_evidence` | `PAR-MODEL-INTERNAL` | PP / TP / CP 与 Device Mesh 的模型内部并行证据 | 81、Part 05 | `consumes` | stage、Head、sequence 布局与 collective 记录；真实多卡 adapter 待接入 |
| `distributed.moe_dispatch_evidence` | 80 | capacity / overflow / dispatch / 通信证据 | 81、Part 05 | `consumes` | router、专家配置与拓扑明确 |
| `distributed.inference_candidate` | 81 | 副本 / 模型并行 / PD 分离的 serving 结果 | Part 05 推理部署项目 | `consumes` | 多卡硬件、backend 与 workload 可复现 |
| `quantization.artifact_gate` | 82 | artifact 完整性、可加载性与质量 gate | 67、72、83 | `validates` | artifact id、format、loader 与质量记录齐全 |
| `quantization.serving_profile` | 83 | latency / throughput / balanced profile | Part 05 推理部署项目 | `consumes` | 82 gate 与 67 或 72 的部署记录可关联 |
| `alignment.preference_evidence` | 84 | chosen / rejected、reference 与评测结果 | 86、Part 05 | `compares_against` | 偏好数据版本和评测协议明确 |
| `alignment.group_reward_evidence` | 85 | rollout、reward、组内统计与质量 | 86、Part 05 | `compares_against` | 奖励函数、采样与质量阈值明确 |
| `alignment.online_decision` | 86 | freshness、数据隔离、回滚与上线判断 | Part 05 对齐项目 | `consumes` | 84 / 85 与线上评测数据均可追溯 |

## 字段复用规则

| 字段组 | 最小内容 | 适用范围 |
| --- | --- | --- |
| 对照对象 | `baseline`、`candidate` / `candidates`、策略名称与角色 | 所有比较实验 |
| workload | 模型、输入长度、输出长度、batch、并发、缓存/数据策略；不足项显式为 `null` | GPU 探针与全部项目 benchmark |
| 环境与 artifact | hardware、backend / runtime、dtype、版本、artifact id / 路径 | 有真实运行或可加载产物的实验 |
| 指标 | 延迟、吞吐、显存、通信、质量等；机制特有指标置于专用字段 | 所有结果 JSON |
| 质量门槛 | 评测集及版本、协议、阈值、失败样例 | 会改变模型输出的候选方案 |
| 证据与决策 | `evidence_level`、`failure`、复测路径、`decision`、原因和下一步 | 所有可交付结论 |

## 覆盖状态与下一步

### 已完成

- 训练微调、推理 benchmark、训练显存、profiling、量化证据均已有明确的版本化 Python 契约。
- 79、80 已以共同必要字段进行证据聚合。
- 项目页和机制页已普遍使用 baseline / candidate、workload、证据等级与 accept / tune / reject 的表达。

### 优先补齐

1. **先建立字段与路径映射表，不改结果文件。** 对每个项目登记原始 JSON、trace、artifact、上游输入、下游消费者、当前 schema 与目标 schema；65、71、72 是跨专题资产，要先确定主归属。
2. **核对 schema 的实际接入状态。** 区分“工具已经存在”“Notebook 已调用”“现有结果已经归一化”三种状态；特别核对 60–76 不能只根据 `tools/` 推断完成迁移。
3. **冻结历史证据并定义 companion 策略。** 60、66、73–76 的现有 JSON / trace 保持原路径；后续只新增归一化 companion JSON 与清晰的来源字段。
4. **先设计目标契约，不写实现。** 依次输出 `distributed-benchmark/v1`、量化部署链映射、`architecture-evaluation/v1`、`model-evaluation/v1` 的字段草案与兼容表。
5. **最后再按依赖迁移。** 先 79–81，再 82–83 与 72，然后架构与对齐；非项目 `mechanism-gpu-probe/v1` 放在项目资产稳定后处理。

## 维护边界

- 此页记录“哪些资产可以复用、还缺什么”，不替代各项目的真实结果表。
- 原始 backend / profiler / notebook JSON 是可追溯证据；归一化记录不能覆盖或删除它们。
- 新增字段优先放入 `strategy_metrics`、`quality`、`artifacts` 等专用区域；只有跨至少两个项目稳定复用后，才提升为公共字段。
