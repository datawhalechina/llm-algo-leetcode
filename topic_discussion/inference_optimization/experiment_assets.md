# 推理优化实验资产表

本表登记推理机制实验与项目出口。环境创建见[实验 Quickstart](./quickstart.md)，统一字段见[Part 02 结果资产与证据契约](../../docs/part02_result_assets.md)。

## 环境角色

| 环境 ID | 直接入口 | 可形成的证据 |
|:---|:---|:---|
| `INF-CPU` | 请求、预算、缓存、调度与投机解码的 Step1–4 | 状态、公式、策略与决策逻辑 |
| `INF-CUDA-MECHANISM` | 20、迭代预算、执行分派、动态候选预算与状态重叠 | 真实模型或真实 CUDA 路径的正确性、延迟、显存、能力门槛与回退证据 |
| `INF-VLLM-011-CU128` | 连续批处理、分块预填充、KV 生命周期与 66–72 的主服务路径 | 固定请求清单下的 TTFT、TPOT、吞吐、P95/P99 与后端配置证据 |
| SGLang 独立环境 | 69、70、72 的可选同口径对照 | Prefix/Radix Cache、调度和模型兼容性证据 |
| 通信与并行 Profile | 81 与模型内部并行路径 | 多卡通信、扩展效率和分布式失败证据 |

机制环境不安装服务执行后端。服务能够启动只形成环境或能力预检；稳定性能结论必须来自固定模型版本、请求清单、并发和重复次数的配对实验。

新增机制页分为两层：请求生命周期、迭代预算、执行分派、EAGLE、动态候选预算、MLA、连续批处理和执行重叠属于核心或共享机制；DFlash / DSpark 与 AFD 属于扩展分支。扩展页不要求首次学习，也不为了页面齐全重复建设低价值 GPU 实验。

## 单卡硬件分层

| 硬件档位 | 默认可验证范围 | 不作为默认承诺的范围 | 环境与证据要求 |
|:---|:---|:---|:---|
| 单卡 12 GB | 经典投机解码、动态候选预算；Qwen3.5-0.8B 的 MTP-1；训练完成后的 Qwen3-0.6B DSpark 推理 | Qwen3-8B BF16 + EAGLE/DFlash；大型 GLM DSpark；未经配对确认的量化草稿器 | 固定 batch=1、上下文 1024–2048 和低并发；MTP 先通过 target-only；自训练草稿器必须先形成可追溯 artifact |
| 单卡 24 GB | Qwen3.5-0.8B MTP；同源量化 Qwen3-8B verifier + 配对 EAGLE-3 或 DFlash；Qwen3-0.6B DSpark 自训练配对 | BF16 8B 配对、高并发、长上下文、多卡扩展结论 | 使用独立 vLLM/Speculators 环境；先做显存预检；target-only 与 candidate 固定模型语义、量化格式、请求和采样条件 |

12 GB 是机制可行性和小模型研究档位，不自动产生生产级加速结论；24 GB 是配对草稿器真实后端验证的优先档位，但也不保证 BF16 verifier、speculator、KV Cache 和工作区能够同时驻留。当前 RTX 5070 Ti + R570 继续使用已验证的 cu128/Transformers 路径；35 的 vLLM cu129 失败结果保留为当前驱动—CUDA—Triton 组合的兼容性证据。学习者使用不同驱动或 wheel 时，应先跑 target-only，再决定是否启用 MTP 或外置草稿器。

## 单卡机制验证顺序

第一轮先验证不依赖服务后端的五项机制，再切换到独立 vLLM 环境验证连续批处理、分块预填充和 KV 生命周期。每项使用唯一结果标签；前一项失败时保留失败 JSON，修复后重新运行，不跳过失败直接汇总专题结论。

| 顺序 | Profile / 页面 | 固定输入或模型 | 本步要确认 | 第一轮只修改 |
|:---|:---|:---|:---|:---|
| 0 | `INF-CUDA-MECHANISM` 安装与预检 | PyTorch 2.11 CUDA 12.8、Transformers 与仓库运行时 | `pip check` 无冲突，Profile 返回 `status=ok` | 环境名按 Quickstart 使用 |
| 1 | Part 02 · 20 SDPA / Attention | 合成 Q/K/V、固定形状与 dtype | 输出一致性、显式 score 与 SDPA 的延迟和峰值显存差异 | `RUN_MODE`、`RUN_LABEL` |
| 2 | Part 02 · INF-EXECUTION-DISPATCH | RMSNorm–SwiGLU–Residual 片段，三档 Token 数 | eager / compile 正确性、图捕获、回退、Median/P95 与峰值显存 | `RUN_MODE`、`RUN_LABEL` |
| 3 | Part 02 · INF-SERVING-OVERLAP | 三档固定 H2D Payload 与 BF16 GEMM | 串行 / 双 Stream 的正确性、暴露传输和可隐藏时间 | `RUN_MODE`、`RUN_LABEL` |
| 4 | Part 02 · INF-ITERATION-BUDGET | `Qwen/Qwen3-0.6B`，固定 batch × context 矩阵 | Prefill、单步 Decode、KV 字节、峰值显存、P95 与 OOM 恢复 | `RUN_MODE`、`RUN_LABEL` |
| 5 | Part 02 · INF-PROPOSAL | Qwen2.5 1.5B 目标模型 + 0.5B 助手模型 | 无助手与三种候选预算的输出一致性、前向次数、净收益和峰值显存 | `RUN_MODE`、`RUN_LABEL` |
| 6 | `INF-VLLM-011-CU128` / Part 02 · INF-CONTINUOUS-BATCHING | `Qwen/Qwen3-0.6B`，同一请求清单与并发 | 批次屏障与滚动补位的槽位利用率、准入等待、吞吐和尾延迟 | `RUN_MODE=real_backend`、`RUN_LABEL`、`RUN_ORDER` |
| 7 | `INF-VLLM-011-CU128` / Part 02 · 38 | 长输入与短解码干扰工作负载 | 整段预填充 / 分块预填充的 TTFT、TPOT、吞吐和尾延迟 | `SERVICE_VARIANT`、`PAIR_LABEL` |
| 8 | `INF-VLLM-011-CU128` / Part 02 · 37 | `Qwen/Qwen3-0.6B`，唯一前缀 / 共享前缀配对请求 | 缓存指标、TTFT、容量压力和共享前缀收益 | `RUN_MODE=real_backend`、`RUN_LABEL` |

`ARCH-MLA` 已在架构专题的 `ARCH-CUDA-MECHANISM` 中完成实测，Task3 直接复用其表示与容量证据。AFD 需要固定插件、至少两个可独立调度的执行角色和真实后端，留到多 GPU 条件具备后验证。

## 机制实验注册表

| 机制与页面 | GPU 角色 | 固定对照与指标 | 当前状态 |
|:---|:---|:---|:---|
| 请求生命周期 / `INF-REQUEST-LIFECYCLE` | 不单设 GPU | 事件轨迹、TTFT、TPOT、E2E、失败状态 | CPU 机制完整；66 提供真实后端基线 |
| Attention 与执行 / 20、`INF-EXECUTION-DISPATCH` | 单卡机制 | 显式 Attention / SDPA、eager / compile 候选；输出一致性、延迟、峰值、路径与回退 | 20 已登记 `rtx5070ti_sdpa_matrix_v3`：四档正确性通过、`2.13×–6.18×`、命中 Flash Attention；执行分派已登记 `rtx5070ti_dispatch_v3`：1/32 Token 仅适合高复用稳态编译，256/1024 回退 eager |
| 迭代预算 / `INF-ITERATION-BUDGET` | 单卡机制 | 批量 × 上下文；中位数/P95、KV 字节、峰值、OOM 恢复与拐点 | 已登记 `rtx5070ti_iteration_budget_v3`：Qwen3-0.6B commit `c1899de`，12 个 shape 成功，KV 与峰值显存证据完整；后续交给连续批处理与准入项目复测 |
| 连续批处理 / `INF-CONTINUOUS-BATCHING` | 单卡真实后端机制 + 项目承接 | 批次屏障 / 滚动补位；槽位利用率、准入等待、TTFT、TPOT、E2E、吞吐与 P99 | 已登记 `rtx5070ti_continuous_slots_v2_forward/reverse`：交换顺序后槽位利用率提高 `18.1–23.9` 个百分点，准入等待 P99 减少 `44.2%–52.8%`，吞吐提高 `34.6%–40.4%`；TPOT 方向不稳定，不解释为单步执行加速 |
| 动态候选预算 / `INF-PROPOSAL` | 单卡机制 | Transformers 辅助生成的固定预算、启发式预算与置信度提前停止；输出一致性、目标前向次数、Median/P95 与净收益 | 已登记 `rtx5070ti_proposal_v2`：Qwen2.5-1.5B + 0.5B，动态置信度候选输出一致，延迟比 `0.854`；固定预算和启发式候选未接受，后续交给 68 扩大到专项后端 |
| 原生 MTP / 35 | 单卡 backend 机制 | 12 GB/24 GB 均先用 Qwen3.5-0.8B target-only / MTP-1；输出一致性、接受长度、Median/P95、吞吐与显存 | RTX 5070 Ti + R570 已确认模型契约，但当前 vLLM 0.31.0 cu129 组合在 target-only 的 Triton RoPE 内核加载阶段失败；换环境后从 target-only 重测，不能把该失败归因于显存或 MTP |
| EAGLE | 24 GB 优先，12 GB 条件运行 | 同源量化 Qwen3-8B verifier + `RedHatAI/Qwen3-8B-speculator.eagle3`；接受长度、候选树成本、质量与净收益 | 12 GB 只在量化 verifier 的隐藏层和分词契约通过后尝试；24 GB 从 batch 1、上下文 1024–2048、候选 2–4 起步；BF16 8B 配对不作为默认承诺 |
| DFlash / DSpark | 24 GB 配对后端；12 GB 小模型自训分支 | DFlash 使用同源量化 Qwen3-8B 与配对 artifact；DSpark 使用训练完成的 Qwen3-0.6B speculator | 12 GB 的 DSpark 训练优先采用离线两阶段；在线训练建议分离 verifier 与 trainer；没有真实配对权重时只登记资产缺口，不运行替代性探针 |
| KV 生命周期与复用 / 22、24、34、37 | 执行后端机制 + 项目承接 | 容量、命中、引用、驱逐、回收和首 Token 延迟 | 37 已登记 `v3_01` 与 `v3_02` 两次交换顺序复测，以及 `v4_03` 12/12 档容量压力矩阵：512/1024/1536 Token、并发 4/8/16/32 均成功，无抢占；首个 512 Token、并发 4 档包含冷启动异常，矩阵只支持压力趋势，尚未确定驱逐或容量上限；22 不设置替代性 GPU 探针，69 后续承接稳定服务证据 |
| 请求选择 / 36 | CPU 机制 + 项目承接 | 优先级、老化、期限、准入与抢占 | 单服务实例无法证明策略替换；70 负责真实调度策略比较 |
| 分块预填充 / 38 | 单卡真实后端机制 + 项目承接 | 整段 / 分块预填充；客户端流式输出间隔、短探针首 Token 延迟、TPOT 与吞吐 | 已登记 `rtx5070ti_chunked_interference_v2` 与 `v3_reverse`：交换运行顺序后，最大流式间隔减少约 `75.0%–81.4%`，短探针首 Token 延迟减少约 `21.7%–58.7%`；该间隔包含调度、传输和客户端读取时间，不等同于 GPU Kernel 时间 |
| 异构 PD 与服务分层 / 39 | 多资源后端 + 项目承接 | 资源池能力、KV 交接、回退、P95/P99 | 需要至少两个可独立观测的执行角色；81 承接真实拓扑证据 |
| 执行重叠 / `INF-SERVING-OVERLAP` | 单卡 CUDA 机制 + 项目承接 | 串行 / CUDA Stream；正确性、传输、暴露时间、中位数/P95 | 已登记 `rtx5070ti_overlap_v2`：三档通过，延迟比 `0.788/0.934/0.965`，重叠效率 `95.7%–97.6%`；队列与状态交接由 70 或分阶段项目承接 |
| AFD / `INF-AFD` | 可选多 GPU / 执行后端 | 普通 MoE / AFD；正确性、通信、暴露时间、TTFT、TPOT 与吞吐 | 5.1–5.4 已统一；待固定插件、模型、适配器和真实多 GPU 环境 |

## Task5 低精度执行证据

Task5 复用量化专题已经形成的机制结果，不在推理专题重复运行局部量化实验。量化结果只有依次通过版本化产物、目标加载器、实际计算路径和固定服务工作负载检查，才能形成单卡部署证据。

| 路径 | 已有机制证据 | 当前缺口 | 下一证据层 |
|:---|:---|:---|:---|
| GPTQ W4A16 | 40 已生成包含模型版本、量化配置、数据身份和文件哈希的 `candidate_bundle.json` | 尚未证明目标加载器兼容性和真实低精度计算内核 | 目标后端加载、路径命中与浮点基线配对 |
| SmoothQuant W8A8 | 53 已完成校准候选、质量比较和压缩产物生成 | 尚无统一候选包，也未验证真实 W8A8 计算内核 | 版本化候选包、加载器与内核验证 |
| FP8 | 41 已验证 torchao 局部执行路径、正确性和回退条件 | 不能外推为 vLLM 或其他服务后端的 FP8 路径 | 明确支持 FP8 的目标后端配对实验 |
| KV Cache 量化 | 54 已验证 Transformers `QuantizedCache` + HQQ 的容量、质量和延迟变化 | K/V 固定同轴；不能外推为 vLLM 或 SGLang 的原生量化缓存 | 固定后端缓存配置后比较 TPOT、并发容量、峰值显存、质量与失败率 |
| 混合位宽 | Mixed-bit 已形成层级敏感度和容量候选 | 没有可直接加载的统一服务产物，也没有加速结论 | 版本化产物、后端能力和端到端配对实验 |

当前 Task5 的最高证据等级是产物生成或局部执行验证，尚未形成完整的单卡部署证据。后续项目只消费版本化产物和结果 JSON，不改变这里的证据定义。

## Task6 分布式执行证据

Task6 的 CPU 页面可以验证切分、路由和恢复决策，但扩展效率、暴露通信和故障恢复必须来自真实多进程或多 GPU 环境。当前不使用单卡张量切分或模拟通信替代分布式证据。

| 路径 | 已有证据 | 当前缺口 | 验证条件 |
|:---|:---|:---|:---|
| Serving 副本 | PAR-REPLICA 已覆盖队列、局部性、会话连续性和路由决策 | 缺少两个真实实例的队列、缓存与故障转移结果 | 至少两个独立服务实例；固定请求流、路由策略和实例状态 |
| PP / TP / CP | 已有切分、通信量和容量机制 | 缺少真实集合通信、空转和扩展效率 | 至少双 GPU；固定模型、并行度、互联与请求工作负载 |
| EP | 已覆盖词元分发、专家容量和负载均衡 | 缺少真实 All-to-All、热点专家和失败证据 | MoE 模型、至少双 GPU、固定路由与容量配置 |
| 可靠性与恢复 | Part 01 · 28 和专题正文已定义故障域、恢复点与降级动作 | 缺少请求中断、状态重建和恢复时间实测 | 多进程或多实例故障注入；保留失败请求和状态一致性证据 |

多卡验证统一暂缓到具备稳定多 GPU 环境后执行。暂缓表示证据尚未形成，不表示机制已经通过，也不影响先完成 CPU 机制与实验契约审核。

## 项目出口

| 项目 ID | 页面 | 核心职责 |
|:---|:---|:---|
| `INF-FLOAT-BASELINE` | 66 | 建立浮点模型和统一请求指标基线 |
| `INF-SPECULATIVE-BENCH` | 68 | 比较草稿拓扑、验证语义、接受率与净收益 |
| `INF-PREFIX-CACHE` | 69 | 验证前缀复用、缓存命中和 TTFT 收益 |
| `INF-SERVING-SCHEDULER` | 70 | 比较请求选择、连续批处理、阶段调度与 overlap |
| `INF-SINGLE-GPU-DEPLOY` | 72 | 汇总通用基线、专项策略和量化服务证据，生成单卡部署交接清单 |
| `DIST-INFERENCE` | 81 | 单实例不足后的多卡执行与扩展效率 |
| `INF-INTEGRATED-DELIVERY` | Part 05 · 01 | 在冻结请求负载下复测候选，形成 SLO、质量、容量与回滚决策 |

项目只消费机制页定义的状态和指标，不重新定义机制。跨专题只交换版本化 JSON 或产物清单，不依赖另一 Notebook 的内存变量。

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、模型和分词器版本 |
| 工作负载 | 输入/输出长度、到达模式、并发、批量/Token 预算、随机种子 |
| 环境 | Profile ID、Python、PyTorch、CUDA、GPU、执行后端与版本 |
| 性能 | TTFT、TPOT、E2E、throughput、P50/P95/P99、queue wait、peak memory |
| 正确性 | 输出数量、终止状态、质量或与参考输出的一致性 |
| 结论 | `failure`、`evidence_level`、`decision` 与下一步 |
