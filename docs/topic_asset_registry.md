# 横向专题资产与依赖登记

> 统一审阅入口见 [教程资产总登记与审阅台账](./asset_registry.md)。

本文是横向专题的维护索引，记录专题如何复用 Part 00–04 的代码、图解、结果和项目资产。它不替代各专题的 `intro`、`walkthrough`、`casebook` 或正文，也不替代 Part 02 的原始实验结果。

## 使用规则

- **Part 00–04 是资产权威位置**：基础代码、kernel、构建配置、原始 JSON、artifact 和 trace 不复制到专题目录。
- **专题负责组织与解释**：`intro` 给路线，`walkthrough` 放概念图与链路图，`casebook` 放失败模式和决策案例，正文补专题特有机制。
- **项目只保留一个主叙事**：交叉专题通过关联入口复用资产，不重写目标、字段或结论。
- **Part 05 只消费已验证资产**：至少具备固定 workload、来源路径、证据等级和质量 / 成本约束之一。

## 专题资产总览

| 专题 | 定位 | 主要输入资产 | 专题自有资产 | 项目出口 | 主要消费者 |
| --- | --- | --- | --- | --- | --- |
| 推理优化 | 主线 | Part 01 的执行与访存；Part 02 Cache、调度、推测解码 | workload、服务策略图、benchmark 解释 | 66、68–72 | 部署异构、性能、Part 05 |
| 性能优化 | 主线 | Part 01 执行 / 访存 / I/O；Part 02 训练与推理实验 | trace、profile、回归与证据协议 | 73–76 | 后训练、部署、Part 05 |
| 算子优化 | 主线 | Part 01 硬件；Part 03 Triton；Part 04 CUDA | kernel 对照、融合与调优决策 | Part 03 / 04 项目 | 推理、性能、Part 05 |
| 后训练优化 | 主线 | Part 00 训练基础；Part 02 SFT、偏好与对齐 | 数据 / 对齐协议、质量门槛、在线反馈 | 62–65、84–86 | 数据工程、Agent、Part 05 |
| 模型架构演进 | 主线 | Part 00–02 的 Transformer、Attention、MoE 机制 | 架构图谱、审计卡、选型框架 | 60、61、71、77、78 | 推理、性能、Part 05 |
| 量化压缩 | 支撑 | Part 01 数值与规模；Part 02 量化与部署 | artifact gate、量化证据链 | 67、82、83 | 推理、部署异构、Part 05 |
| 通信与并行 | 支撑 | Part 01 1C；Part 02 并行与通信项目 | 拓扑、通信 / overlap 证据 | 79–81 | 性能、部署异构、Part 05 |
| 数据工程 | 支撑 | Part 00 数据入口；Part 02 SFT / 偏好数据 | 数据血缘、版本、去污染和质量协议 | 后续数据项目 | 后训练、推理 RAG、性能 |
| 部署异构 | 支撑 | Part 01 1F；Part 02 72、81–83；Part 04 系统证据 | 部署契约、兼容性与选型框架 | 72、81–83 | Part 05 |
| Agent | 领域 | 推理、后训练、数据与部署资产 | 轨迹 workload、沙盒、工具与评测环境 | 后续 Agent 项目 | Part 05 |
| 多模态 | 领域 | 视觉模型、推理、部署与数据资产 | 视觉 token workload、VLM 环境与评测 | 后续多模态项目 | Part 05 |

## 跨专题共享资产

| 资产 | 规范归属 | 生产者 | 消费者 | 复用规则 |
| --- | --- | --- | --- | --- |
| workload 契约 | 推理优化 / 性能优化 | 66、68–70、73–76 | 量化、部署、Part 05 | 引用 workload id；不复制隐含配置 |
| 质量门槛 | 后训练优化 / 数据工程 | 50、51、84–86 | 量化、并行、部署、性能 | 引用评测协议、数据版本与阈值 |
| artifact manifest | 量化压缩 | 65、82 | 67、72、83 | 以 `artifact_id`、revision 和 gate 记录关联 |
| trace / profile | 性能优化 | 74、Part 03 / 04 | 算子、部署、Part 05 | 保留原始路径与环境；摘要不能替代 trace |
| 拓扑与通信证据 | 通信与并行 | 79、80 | 81、性能、Part 05 | 固定 topology、workload 与通信指标；按训练状态、MoE、Serving 分流 |
| 架构审计卡 | 模型架构演进 | 60、61、71、77、78 | 推理、性能、Part 05 | 复用 config、质量 / 成本记录，不复制结论 |
| 数据版本与血缘 | 数据工程 | 数据工程、64、50 | 后训练、Agent、多模态、模型评测 | train / eval 隔离与去污染必须可追溯 |
| kernel 正确性与调优记录 | 算子优化 | Part 03、Part 04 | 性能、推理、Part 05 | 先满足输入契约与正确性，再比较性能 |

## 专题维护状态

| 专题 | 路线与正文 | walkthrough / casebook | 结果或 artifact 契约 | 当前优先缺口 |
| --- | --- | --- | --- | --- |
| 推理优化 | 已建设 | 已建设 | `inference-benchmark/v1` | 部署与 I/O 证据收口 |
| 性能优化 | 已建设 | 已建设 | 训练内存、profiling 契约 | 统一回归入口 |
| 算子优化 | 已建设 | 已建设 | Kernel / 系统资产表初建 | 正确性—benchmark—profile 的逐 kernel 审计 |
| 后训练优化 | 已建设 | 已建设 | 训练契约已存在；模型评测待建设 | 84–86 的评测结果资产 |
| 模型架构演进 | 路线与机制页已建立 | 五个项目已接入 | `architecture-evaluation/v1` 已建立并接入 | 60→78→71→77 生产局部证据，61 校验引用并汇总审计 |
| 量化压缩 | 已建设 | 已建设 | `quantization-evidence/v1` | 67、72、83 的部署关联键 |
| 通信与并行 | 已建设 | 已建设 | `distributed-benchmark/v1` 已建立 | 模型内部并行项目的真实多卡 adapter 与统一复测 |
| 数据工程 | 初建 | 初建 | 待建设 | 数据项目、版本与血缘资产 |
| 部署异构 | 初建 | 初建 | 复用推理 / 量化契约 | 设备兼容与 deployment contract |
| Agent | 初建 | 初建 | 待建设 | 轨迹、沙盒、评测项目 |
| 多模态 | 初建 | 初建 | 待建设 | VLM workload、视觉 token 与部署项目 |

## 三专题成熟度审计（2026-10-04）

本表按《维护与发布手册》的专题建设 SOP 审核量化压缩、通信与并行、大模型架构演进。评分只表示当前资产的完备程度，不表示内容难度：`1` 为规划或占位，`2` 为机制和 CPU 路径可用，`3` 为结构化 GPU 入口或结果契约已建立，`4` 为专题闭环基本完整，`5` 为固定 workload 下的真实结果已复测并可持续回归。

| 专题 | 专题成熟度 | 机制小节 | 机制 GPU 实验 | 项目小节 | 项目 GPU 实验 | 当前判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 量化压缩 | 4/5 | 4.5/5 | 4/5 | 4/5 | 3/5 | 机制链、执行栈和 `quantization-evidence/v1` 已形成；多数机制页已有成熟库驱动的 GPU 路径，但现存结果主要是环境预检，尚未形成当前版本的统一复测集 |
| 通信与并行 | 4/5 | 4/5 | 4/5 | 3.5/5 | 2.5/5 | 切分—collective—payload—overlap—扩展效率链条完整；除决策页 49 外，核心机制页均已有真实多卡入口；项目 adapter 与正式多卡结果仍未全部接通 |
| 大模型架构演进 | 4.5/5 | 4/5 | 3/5 | 4.5/5 | 3/5 | Task0–6、四层覆盖矩阵和 `architecture-evaluation/v1` 已建立；44、45、`ARCH-MLA`、`ARCH-HYBRID-MEMORY` 已覆盖访问范围、递推状态、紧凑表示与混合计划，正式质量与成本证据仍集中到项目页 |

### 机制页与 GPU 覆盖

| 专题 | 机制页范围 | 已有 GPU 入口 | 不应重复建设的部分 | 下一处必要补强 |
| --- | --- | --- | --- | --- |
| 量化压缩 | 25、26、40、41、52–54、`QUANT-MIXED-BIT`、`QUANT-QAT` | 除 52 外均有 5.x GPU 单元；使用 torchao、Transformers、bitsandbytes、PEFT 或 LLM Compressor | 52 的校准集隔离、误差口径和 outlier 协议以 CPU / 数据证据为主，不必强行加 GPU | 在同一环境完成 25、40、41、53、54 的 current-revision 复测，并把 artifact、loader、kernel/backend 命中与 fallback 连成证据链 |
| 通信与并行 | 27–29、46–49、`PAR-CONTEXT`、`PAR-REPLICA` | 27–29、46–48、`PAR-CONTEXT`、`PAR-REPLICA` 已有多卡执行入口 | 49 是组合决策页，不应复制每一种 collective benchmark | 在登记环境中运行新增入口，并统一记录 topology、world size、payload、collective、通信暴露时间和扩展效率 |
| 大模型架构演进 | 00–08、44、45、`ARCH-MLA`、`ARCH-HYBRID-MEMORY` | 44 验证访问 mask；45 验证单层显式历史与固定状态；`ARCH-MLA` 验证表示与重建；`ARCH-HYBRID-MEMORY` 验证多层混合计划 | 01/02 的 kernel 微基准归算子优化；06/07 的多卡 dispatch 由 80 承担；08 是配置审计入口 | 在登记单卡环境运行新增对照；真实 checkpoint 的质量—成本仍由 71、77 收口 |

### 项目页与真实证据

| 专题 | 稳定项目链 | 已完成 | 仍未完成 |
| --- | --- | --- | --- |
| 量化压缩 | `QUANT-ARTIFACT-GATE → INF-FLOAT-BASELINE → QUANT-INFERENCE-DEPLOY → QUANT-SERVING-PROFILE`；65 是低比特训练交叉入口 | 82、67、83 已接 `quantization-evidence/v1`，机制页已明确框架角色 | 当前 revision 的量化 artifact、真实 backend profile、质量门槛和 fallback 结果尚未统一采集；历史 preflight 不能当作 benchmark |
| 通信与并行 | `PAR-TRAINING-STATE → PAR-MODEL-INTERNAL → PAR-MOE-EXPERT → PAR-DISTRIBUTED-INFERENCE` | 四个语义 ID 和 `distributed-benchmark/v1` 已登记；79、80、81 已有分层 GPU 单元 | `PAR-MODEL-INTERNAL` 只有配置入口；真实多卡 adapter、统一 companion 结果和跨拓扑复测尚未完成 |
| 大模型架构演进 | `ARCH-BLOCK-STABILITY → ARCH-ATTENTION-ACCESS → ARCH-MLA-KV → ARCH-LONG-SEQUENCE-MEMORY → ARCH-MODEL-AUDIT` | 四个 producer 与 61 audit 已接 `architecture-evaluation/v1`；路径、哈希、角色和语义 ID 可校验 | 60/71/77/78 尚需在固定模型、tokenizer、workload 和环境下采集；78 仍是 SDPA mask 而非专用稀疏 kernel，71 的真实 MLA backend 证据最薄 |

本轮审计只确认“代码入口与契约是否存在”，不把未运行的 GPU cell 记为真实证据。下一轮采集应优先沿每个专题的一条最短闭环执行，而不是同时扩大所有页面的 GPU 覆盖。

## 维护边界

- 不再为每个专题新增独立的 `visual_assets.md`。概念图留在 `walkthrough`；历史视觉资产页仅作为迁移或审计记录。
- 专题表只登记依赖与成熟度，不重复 Part 00–04 的代码细节、Part 02 的项目字段或原始结果目录。
- 新专题先在本表登记“输入资产、计划自有资产、项目出口与缺口”，再决定是否建设正文或项目。
- 每次变更主项目归属、artifact 消费关系或结果契约时，同步更新本表和 [Part 02 结果资产与证据契约](./part02_result_assets.md)。
