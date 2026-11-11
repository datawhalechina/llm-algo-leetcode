# 推理优化实验资产表

本表登记输入、实验入口与已有结果。环境安装见[Quickstart](./quickstart.md)，字段定义见[Part02 结果资产与证据契约](../../docs/part02_result_assets.md)。结果中的模型、软件版本和工作负载决定结论适用范围。

<a id="环境角色"></a>

## 环境与实验入口

| 环境 | 实验入口 | 输入准备 |
| --- | --- | --- |
| [BASE-CPU](../../docs/guide.md#base-cpu) | 请求、缓存、路由、预算和调度的 Step1–4 | 页面内固定样例；不需要服务后端 |
| [INF-CUDA-MECHANISM](./quickstart.md#inf-cuda-mechanism-quickstart) | Attention、执行分派、MoE、迭代预算、动态候选与执行重叠 | 合成张量或页面指定模型；模型首次使用需下载或预置缓存 |
| [INF-VLLM-011-CU128](./quickstart.md#inf-vllm-quickstart) | 连续批处理、分块预填充、KV 生命周期及传统服务基线 | 页面指定模型、固定请求清单；外部服务参数见[服务准备](#外部服务准备) |
| [INF-VLLM-SPEC025](./quickstart.md#inf-vllm-spec025-quickstart) | Part02·68 的投机候选选型 | 同源目标模型与配对草稿资产，先通过 target-only |
| [INF-VLLM-MTP](./quickstart.md#inf-vllm-mtp-quickstart) | Part02·35 原生 MTP | Qwen3.5-0.8B；专用构建、模型契约与启动显存预检 |
| SGLang 独立环境 | 跨后端可选对照 | 见[环境索引](../../docs/gpu_environment_assets.md)；不混装到 vLLM 环境 |
| 通信与并行环境 | Part02·81 与多卡执行 | 见[并行实验资产](../communication_parallelism/experiment_assets.md)；需要目标拓扑 |

模型使用页面指定的仓库和 revision；预先缓存时保持同一身份，不用相近名称的模型替代。下载权限或网络不足时先完成 CPU 部分；认证令牌只放在本机环境或凭据存储中。服务端、客户端与 Notebook 的解释器分别核对，环境预检不能代替实验结果。

<details>
<summary>单卡容量规划：只作起步范围，不是通过承诺</summary>

| 硬件 | 可先尝试的输入 | 条件与限制 |
| --- | --- | --- |
| 12 GB | 经典投机、小模型 MTP、已训练的 Qwen3-0.6B DSpark | 从 batch=1、上下文 1024–2048 起步；先确认 target-only 和资产契约；不默认支持 BF16 8B verifier 与草稿器同时驻留 |
| 24 GB | 小模型 MTP、同源量化 Qwen3-8B 与配对 EAGLE-3／DFlash | 仍需计入草稿器、KV 与工作区；量化格式和隐藏层契约必须匹配；不外推长上下文或高并发 |

驱动升级不自动使旧失败变成通过。当前复现以所选 Profile 的门禁和本次预检为准；历史硬件与版本保留在原始报告中。

</details>

<a id="单卡机制验证顺序"></a>
<a id="机制实验注册表"></a>

## 机制实验

先选择对应瓶颈的机制，不必顺序运行全部实验：计算路径先看 Attention／执行分派，生成成本看候选预算，缓存压力看 KV，排队与干扰看连续批处理／分块预填充。MoE 使用固定路由单独比较专家执行方式。

| 机制与页面 | GPU 角色 | 固定对照与指标 | 当前状态 |
|:---|:---|:---|:---|
| 请求生命周期 / `INF-REQUEST-LIFECYCLE` | 不单设 GPU | 事件轨迹、TTFT、TPOT、E2E、失败状态 | CPU 机制完整；66 提供真实后端基线 |
| Attention 与执行 / 20、`INF-EXECUTION-DISPATCH` | 单卡机制 | 显式 Attention / SDPA、eager / compile 候选；输出一致性、延迟、峰值、路径与回退 | 20 的[正式结果](../../benchmarks/results/20_flashattention/inference-attention-sdpa_rtx5070ti_sdpa_matrix_v3.json)：四档正确性通过、`2.13×–6.18×`、命中 Flash Attention；执行分派的[正式结果](../../benchmarks/results/inf_execution_dispatch_compile_rtx5070ti_dispatch_v3.json)：1/32 Token 仅适合高复用稳态编译，256/1024 回退 eager |
| <a id="moe-execution"></a>MoE 专家执行 / [INF-MOE-EXECUTION](../../02_PyTorch_Algorithms/INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.ipynb) | `INF-CUDA-MECHANISM` 单卡专家层 | 固定八个 SwiGLU 专家、Top-K=2、BF16、隐藏宽度 512、中间宽度 1536；1/16/128 词元 × 均衡/热点路由；逐词元与按专家执行 | CPU 分组、加权恢复与负载账本测试通过；GPU 未实测。路由计划预先构造并复用，其 CPU 成本单列；完整 GPU 调用包含重排、专家计算与合并，记录误差、线性调用、CUDA 事件、延迟与额外峰值。结果写入根目录 `benchmarks/results/inf_moe_execution/moe_execution_<RUN_LABEL>.json`；不声明完整模型或动态路由收益 |
| 迭代预算 / `INF-ITERATION-BUDGET` | 单卡机制 | 批量 × 上下文；中位数/P95、KV 字节、峰值、OOM 恢复与拐点 | [正式结果](../../benchmarks/results/inf_iteration_budget/iteration_budget_rtx5070ti_iteration_budget_v3.json)登记为 `tune`：Qwen3-0.6B commit `c1899de`，12 个 shape 成功，KV 与峰值显存证据完整；后续交给连续批处理与准入项目复测 |
| 连续批处理 / `INF-CONTINUOUS-BATCHING` | 单卡真实后端机制 + 项目承接 | 客户端批次屏障 / 滚动补位；客户端槽位利用率、准入等待、TTFT、TPOT、E2E、吞吐与 P99 | [正序](../../benchmarks/results/inf_continuous_batching/continuous_batching_rtx5070ti_continuous_slots_v2_forward.json)与[逆序](../../benchmarks/results/inf_continuous_batching/continuous_batching_rtx5070ti_continuous_slots_v2_reverse.json)均接受滚动补位：交换顺序后槽位利用率提高 `18.1–23.9` 个百分点，准入等待 P99 减少 `44.2%–52.8%`，吞吐提高 `34.6%–40.4%`；TPOT 方向不稳定；这组结果验证客户端提交边界，不代表引擎内部调度器或单步计算已加速 |
| 动态候选预算 / `INF-PROPOSAL` | 单卡机制 | Transformers 辅助生成的固定预算、启发式预算与置信度提前停止；输出一致性、目标前向次数、Median/P95 与净收益 | [正式结果](../../benchmarks/results/inf_proposal_rtx5070ti_proposal_v2.json)：Qwen2.5-1.5B + 0.5B，动态置信度候选输出一致，延迟比 `0.854`；固定预算和启发式候选未接受，后续交给 68 扩大到专项后端 |
| <a id="mtp-experiment"></a>原生 MTP / 35 | 单卡 backend 机制 | 12 GB/24 GB 均先用 Qwen3.5-0.8B target-only / MTP-1；输出一致性、接受长度、Median/P95、吞吐与显存 | 尚无成功的 target-only / MTP 配对报告；已定位的失败分别涉及进程启动、驱动、启动显存和专用构建门禁，见下方历史记录。使用 `INF-VLLM-MTP` 先通过 target-only，再运行 MTP |
| EAGLE | 24 GB 优先，12 GB 条件运行 | 同源量化 Qwen3-8B verifier + `RedHatAI/Qwen3-8B-speculator.eagle3`；接受长度、候选树成本、质量与净收益 | [资产检查](../../benchmarks/results/inf_eagle_rtx5070ti_eagle_asset_v1.json)为 `not_ready`／`defer_to_multi_gpu`，没有真实配对性能。后续可在满足契约的量化目标模型或更大显存设备上复测；该历史决策名不表示必须多卡 |
| DFlash / DSpark | 24 GB 配对后端；12 GB 小模型自训分支 | DFlash 使用同源量化 Qwen3-8B 与配对 artifact；DSpark 使用训练完成的 Qwen3-0.6B speculator | [资产检查](../../benchmarks/results/inf_parallel_drafting_rtx5070ti_parallel_drafting_asset_v1.json)为 `not_ready`／`defer_to_project`；公开配对或本地训练产物未就绪，没有真实配对性能。12 GB 自训分支优先离线两阶段 |
| MLA 历史表示 / [ARCH-MLA](../../02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention.ipynb) | `ARCH-CUDA-MECHANISM` 单卡局部机制 | 普通 KV、潜在表示及吸收计算路径；数值、状态字节与成本 | [v3 正式报告](../../benchmarks/results/arch_mla_representation_rtx5070ti_mla_v3.json)：`accept_latent_representation_and_absorbed_path_mechanism`；仅证明局部表示与计算机制，不代表完整 MLA 模型或服务部署通过 |
| KV 生命周期与复用 / 22、24、34、37 | 执行后端机制 + 项目承接 | 容量、命中、引用、驱逐、回收和首 Token 延迟 | 37 的[第一次](../../benchmarks/results/37_kv_cache_lifecycle/inference-kv-prefix-lifecycle_rtx5070ti_kv_lifecycle_v3_01.json)、[第二次](../../benchmarks/results/37_kv_cache_lifecycle/inference-kv-prefix-lifecycle_rtx5070ti_kv_lifecycle_v3_02.json)交换顺序复测，以及[容量报告](../../benchmarks/results/37_kv_cache_lifecycle/rtx5070ti_kv_pressure_v4_03/kv_capacity_pressure_rtx5070ti_kv_pressure_v4_03.json) 12/12 档容量压力矩阵均登记为 `tune`：512/1024/1536 Token、并发 4/8/16/32 均成功，无抢占；首个 512 Token、并发 4 档包含冷启动异常，矩阵只支持压力趋势，尚未确定驱逐或容量上限；22 不设置替代性 GPU 探针，69 后续承接稳定服务证据 |
| 请求选择 / Part02·36 | CPU 机制 + 项目承接 | 优先级、老化、期限、准入与抢占 | 本页 CPU 模拟不证明真实策略已替换；Part02·70 在单服务实例上进行真实调度策略配对 |
| 分块预填充 / 38 | 单卡真实后端机制 + 项目承接 | 整段 / 分块预填充；客户端流式输出间隔、短探针首 Token 延迟、TPOT 与吞吐 | [正序配对](../../benchmarks/results/inf_chunked_prefill/chunked_prefill_rtx5070ti_chunked_interference_v2_pair.json)与[逆序配对](../../benchmarks/results/inf_chunked_prefill/chunked_prefill_rtx5070ti_chunked_interference_v3_reverse_pair.json)：交换运行顺序后，最大流式间隔减少约 `75.0%–81.4%`，短探针首 Token 延迟减少约 `21.7%–58.7%`；该间隔包含调度、传输和客户端读取时间，不等同于 GPU Kernel 时间 |
| 异构 PD 与服务分层 / 39 | 多资源后端 + 项目承接 | 资源池能力、KV 交接、回退、P95/P99 | 需要至少两个可独立观测的执行角色；81 承接真实拓扑证据 |
| 执行重叠 / `INF-SERVING-OVERLAP` | 单卡 CUDA 机制 + 项目承接 | 串行 / CUDA Stream；正确性、传输、暴露时间、中位数/P95 | [正式结果](../../benchmarks/results/inf_serving_overlap_rtx5070ti_overlap_v2.json)：三档通过，延迟比 `0.788/0.934/0.965`，重叠效率 `95.7%–97.6%`；队列与状态交接由 70 或分阶段项目承接 |
| AFD / `INF-AFD` | 可选多 GPU / 执行后端 | 普通 MoE / AFD；正确性、通信、暴露时间、TTFT、TPOT 与吞吐 | 尚缺固定插件、模型、适配器与真实多 GPU 结果 |

<a id="teaching-engine"></a>

### 教学引擎运行准备

学习入口分为 [nano-vLLM 请求跟踪](./nano_vllm_request_walkthrough.md)和 [Mini-SGLang 缓存与执行对照](./mini_sglang_cache_scheduling.md)：Task0 开始认识请求，Task3 加入前缀复用，Task4 继续调度观察，Task5 按分支能力核对低精度路径。两套引擎尚未固定验收 commit、登记独立 Profile 或完成本机安装及 GPU 验证，安装命令在实验前准备阶段补入 [Quickstart](./quickstart.md)。现有机制与服务环境保持独立，不向其中直接混装教学引擎依赖。

nano-vLLM 首轮拟使用固定版本的 `Qwen/Qwen3-0.6B`、单卡、eager 执行和两条短提示，生成上限 32 个词元。记录代码 commit、模型 revision、解释器、依赖、GPU、输入、输出词元与停止原因；模型准备和首次初始化与生成计时分开。通过条件是环境无依赖冲突、CUDA 实际计算可用、两条请求均正常完成且输出非空、进程正常退出。回答正确性另行读取，不以“有输出”替代任务质量，也不把批量生成总时间标成流式 TTFT／TPOT。

Mini-SGLang 首轮准备使用相同模型身份与短输入，在普通生成完成后增加一组公共前缀请求，检查实际匹配、状态引用与结束释放。当前仅确定学习与验收目标，固定版本的可观测字段和通过条件待实验前准备时核实；尚无可引用的运行结果。

Task5 的 Mini-SGLang 量化入口先作源码对照：2026-11-11 核对的 [INT8 输出投影 PR](https://github.com/sgl-project/mini-sglang/pull/169)仍为 Open，不能登记成主线已合并能力。实验前分别确认基础 commit、提案分支 commit、量化对象、计算类型和额外驻留副本，再决定是否准备该分支；尚未登记量化环境或本机结果。该局部提案不代表权重、激活与 KV 三条路径均已覆盖。

### 低精度执行分支

复用[量化专题资产](../quantization_compression/experiment_assets.md)，按权重、激活与计算、KV 状态选择证据。权重先核对版本化产物与加载器，运行时低精度配置检查缩放与实际内核，KV 量化检查追加、状态元数据和读取路径。局部量化通过不等于服务部署通过；下表保留已有证据及下一步条件。

| 路径 | 已有机制证据 | 当前缺口 | 下一证据层 |
|:---|:---|:---|:---|
| [W8A16 层表示与执行](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) | 已有 TorchAO 局部层状态、数值与计时证据，详见[量化机制登记](../quantization_compression/experiment_assets.md#机制实验注册表) | 进程内量化模块不自动成为可部署权重目录 | 按目标后端确认可保存格式、加载路径与独立请求收益 |
| GPTQ W4A16 | 40 已生成包含模型版本、量化配置、数据身份和文件哈希的 `candidate_bundle.json` | 尚未证明目标加载器兼容性和真实低精度计算内核 | 目标后端加载、路径命中与浮点基线配对 |
| SmoothQuant W8A8 | 53 已完成校准候选、质量比较和压缩产物生成 | 尚无统一候选包，也未验证真实 W8A8 计算内核 | 版本化候选包、加载器与内核验证 |
| FP8 | 41 已记录 torchao FP8 物理存储与缩放矩阵乘接口事件；18 个矩阵点完成，局部数值门槛未全部通过，0 个加速点，决策为 `tune_or_fallback` | 接口事件尚未关联 CUDA 子内核与输入／累加类型；不能外推为服务后端路径或整体质量通过 | 关联实际执行后，在支持 FP8 的目标后端进行独立质量与同负载配对实验 |
| KV Cache 量化 | 54 的 Transformers／HQQ 六组矩阵已记录增量峰值、整段生成时间与16词元固定轨迹检查；2048词元候选通过本地门槛，决策为 `accept_local_cache_candidate` | K/V 候选同轴，不是 KIVI 异轴比较；未逐次记录转存事件，整段平均时间不是独立 TPOT，固定轨迹不是任务集质量 | 在目标后端核对缓存格式与恢复路径，再比较独立任务质量、Decode 计时和并发容量；不能外推为 vLLM／SGLang 已通过 |
| 混合位宽 | Mixed-bit 已形成层级敏感度和容量候选 | 没有可直接加载的统一服务产物，也没有加速结论 | 版本化产物、后端能力和端到端配对实验 |

### 多实例与多卡分支

单实例不足时，再测量目标拓扑的通信、扩展效率和恢复行为。以下 CPU 机制或配置检查不算多卡实测；真实多 GPU 验证仍待执行。

| 路径 | 已有证据 | 当前缺口 | 验证条件 |
|:---|:---|:---|:---|
| Serving 副本 | PAR-REPLICA 已覆盖队列、局部性、会话连续性和路由决策 | 缺少两个真实实例的队列、缓存与故障转移结果 | 至少两个独立服务实例；固定请求流、路由策略和实例状态 |
| PP / TP / CP | 已有切分、通信量和容量机制 | 缺少真实集合通信、空转和扩展效率 | 至少双 GPU；固定模型、并行度、互联与请求工作负载 |
| EP | 已覆盖词元分发、专家容量和负载均衡 | 缺少真实 All-to-All、热点专家和失败证据 | MoE 模型、至少双 GPU、固定路由与容量配置 |
| 可靠性与恢复 | Part 01 · 28 和专题正文已定义故障域、恢复点与降级动作 | 缺少请求中断、状态重建和恢复时间实测 | 多进程或多实例故障注入；保留失败请求和状态一致性证据 |

<a id="项目出口"></a>

## 项目实验

| 项目入口 | 固定比较对象与职责 | 当前结果入口与结论 |
| --- | --- | --- |
| <a id="baseline-66"></a>[Part02·66 单卡测量基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) | 固定模型与请求协议，核验身份、请求、逐轮统计和可引用范围 | [历史 v4 G0 报告](../../benchmarks/results/66_g0_vllm_rtx5070ti_vllm_g0_v4.json)：Qwen2.5-0.5B-Instruct，每轮 30/30 请求成功，原协议登记 `ready`。新增首末内容时间、停止与原始报告核验的新协议尚未 GPU 实测；旧报告只读解释，不补写字段或改写历史结论 |
| [Part02·68 投机解码](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb) | 目标生成与草稿候选；质量、接受情况与净收益 | [v3 回执](../../benchmarks/results/68_evidence_record_rtx5070ti_speculative_ngram_smoke_v3.json)：`reject`，质量门禁未通过；重复输入的性能信号不替代质量结论 |
| [Part02·69 前缀缓存](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb) | 同请求、缓存关闭／开启及输出核对 | [v4 回执](../../benchmarks/results/69_evidence_record_rtx5070ti_prefix_cache_v4.json)：`tune`，性能通过但输出等价性尚未解决 |
| [Part02·70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb) | 固定请求流与配置下比较真实调度 | [v4 回执](../../benchmarks/results/70_serving_scheduler/70_evidence_record_rtx5070ti_scheduler_priority_v4.json)：`accept`，只适用于该优先级配对协议 |
| [Part02·82 量化产物准入](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.ipynb) | 核对模型、格式、校准来源与产物身份 | 本表尚未登记该项目的正式 GPU 结果；已有局部量化证据见上方分支 |
| [Part02·67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) | 目标加载器、实际执行路径与部署条件 | 本表尚未登记真实部署通过的报告，不以产物生成代替 |
| [Part02·83 量化服务复测](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.ipynb) | 在67已接受的同一量化部署上完成三类请求压力，各自在自身SLO内判断 | 本表尚未登记三类真实服务压力结果；浮点／量化配对在67完成 |
| [Part02·72 单卡部署](../../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.ipynb) | 核对候选、工作负载与部署条件，形成交接清单 | [v1 评估](../../benchmarks/results/72_single_gpu_deployment/72_evaluation_rtx5070ti_single_gpu_delivery_v1.json)：`tune`，仍需独立任务质量证据 |
| [Part02·81 分布式推理](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb) | 单实例不足后的多卡执行与扩展效率 | 真实多 GPU 证据待补；不以单卡切分模拟代替 |
| [Part05 综合实践](../../05_Integrated_Projects_and_Decisions/intro.md) · 01 推理服务 | 独立固定模型、请求与配置组合后重新测量；72 清单只是可选来源 | [历史 v1 报告](../../benchmarks/results/part05_inference_service/inference_service_delivery_rtx5070ti_inference_service_v1.json)自记 `accept`；按其自身协议解读，不能代替现行质量协议复核 |

Part02 汇总页按各自协议读取结果文件；Part05 独立准备输入并测量，不要求先得到 Part02 的接受结论。各项目通过文件交换证据，不依赖另一 Notebook 的内存变量。

<a id="公共结果字段"></a>

## 结果版本与复现

使用唯一 `RUN_LABEL` 保存原始报告、配置和失败记录。变更模型、后端或测量协议后使用新标签；旧报告保持原样。性能比较还需同一工作负载、采样条件和质量规则，结果字段以[统一契约](../../docs/part02_result_assets.md)为准。

66 的新测量按 `prepare → measure → close` 运行，每次修改阶段后顺序执行 5.1–5.6；`read` 只读已有标签。仅 `measure` 使用 GPU，`close` 复核原始报告哈希并写登记；只有 `ready` 生成同标签限定性引用，不改写旧的共享指针。新协议保留每轮 30 条请求与三轮原始记录，内容间隔代理和流结束口径分别解释，设备总显存不当作服务独占显存。

66 的固定参数使用[版本化合同](../../benchmarks/contracts/inference_baseline_v3.json)，项目注册表只指定新测量的默认合同。测量报告、登记与引用携带合同快照和规范化 JSON 内容 SHA256；`read`／`close` 从已保存报告加载该绑定，不读取新默认参数。调整参数时另建合同版本；变更时间口径或核验语义时，同时升级协议与对应代码。没有合同绑定的历史报告保持只读，不补写快照。

68 的已登记 N-gram 报告属于历史服务栈，不证明 SPEC025 的其他方法已经通过。MoE 尚无 GPU 正式结果；其计划路径不是可引用的实测文件。跨版本复测时保留各自环境快照，不把当前驱动信息补写进旧报告。

<a id="外部服务准备"></a>

### 外部服务准备

连续批处理、Part02·37 和 Part02·38 的外部服务使用 `llm_vllm_cu128`。以下命令在激活该环境后执行，保持服务运行，再从 Notebook 发起请求；切换配置前停止上一服务。先确认 `/v1/models` 返回预期模型，缓存实验还需检查 `/metrics`。由 Notebook 自动管理服务的项目（如 Part05·01）不另起同端口服务。

<details>
<summary>连续批处理：关闭前缀缓存，固定并发容量</summary>

该模板显式关闭前缀缓存，避免把复用收益计入滚动补位收益；历史报告仍按其中记录的服务配置解读。

```bash
vllm serve Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto --host 127.0.0.1 --port 8000 \
  --max-model-len 2048 --gpu-memory-utilization 0.8 \
  --max-num-seqs 8 --no-enable-prefix-caching
```

</details>

<details>
<summary>Part02·37：启用前缀缓存</summary>

```bash
vllm serve Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto --host 127.0.0.1 --port 8000 \
  --max-model-len 2048 --gpu-memory-utilization 0.8 --enable-prefix-caching
```

</details>

<details>
<summary>Part02·38：整段与分块预填充分别启动</summary>

两次运行使用相同 `PAIR_LABEL`，分别设置 `SERVICE_VARIANT=whole_prefill` 与 `chunked_prefill`；模型、请求和其他比较条件保持一致。

```bash
vllm serve Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.8 --no-enable-chunked-prefill
```

```bash
vllm serve Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
  --dtype auto --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill --max-num-batched-tokens 512
```

</details>

<details>
<summary>原生 MTP 的历史失败记录</summary>

这些报告没有形成成功配对，不用于判断 MTP 加速或模型质量。先按当前 Profile 通过预检与 target-only，再尝试候选。

| 记录 | 报告中的失败阶段 |
| --- | --- |
| [资产检查](../../benchmarks/results/inf_mtp_rtx5070ti_mtp_v1.json) | 资产未就绪，未形成真实后端测量 |
| [Qwen3.5 v1](../../benchmarks/results/inf_mtp_rtx5070ti_qwen35_08b_mtp1_v1.json) | fork 后 CUDA 重初始化，需要核对进程启动方式 |
| [Qwen3.5 v2](../../benchmarks/results/inf_mtp_rtx5070ti_qwen35_08b_mtp1_v2.json) | 驱动门禁失败 |
| [Qwen3.5 v3](../../benchmarks/results/inf_mtp_rtx5070ti_qwen35_08b_mtp1_v3.json) | 后端启动时可用显存低于请求额度 |
| [Qwen3.5 v4](../../benchmarks/results/inf_mtp_rtx5070ti_qwen35_08b_mtp1_v4.json) | 专用 MTP 构建门禁失败 |

</details>
