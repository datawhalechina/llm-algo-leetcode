# 06. Deployment and Benchmark Decision | 部署与 Benchmark 决策

## 页面目标

本节把前面的量化判断收束到部署和 benchmark：量化是否真的值得切换。

本节的输出是可交付决策：在质量、显存、吞吐、延迟和部署复杂度之间，明确采用、继续调优或回退的理由。

## 问题起点

量化最常见的误判是：显存下降，就默认值得切换。工程上还要问：

- 显存和带宽收益是否真实；
- 速度是否真的提升或至少没有明显变差；
- 精度损失是否在目标场景可接受；
- 后端、kernel 和部署复杂度是否也在可控范围。

## 你要先确认什么

- workload 是否固定。
- baseline 和 candidate 是否只改一个关键变量。
- 你的目标更偏精度、显存、吞吐，还是部署成本。

## 为什么需要部署收口

如果没有部署收口，量化专题容易停在“方法清单”；部署收口要回答的是：这次压缩在目标 workload 上是否带来值得保留的系统收益。

## 判定原则

- `accept`：在固定模型、硬件、backend 和 workload 下，质量与资源指标均达标，并且收益具有重复性。
- `tune`：方向对，但量化粒度、后端、cache policy 或 workload 还要继续调。
- `reject`：无法加载、质量不达标，或资源代价超过预算。

## 最小实验设计

先固定模型、采样参数、prompt / generated tokens、并发和上下文长度，再只替换量化候选。每个候选至少经历以下状态：

`artifact 可定位 → backend 可加载 → 格式 / kernel 可确认 → workload 可完成 → 质量达标 → 资源指标可比较`

每次实验同时维护一份 manifest。它把“这次测了什么”与结果文件绑定，避免只保留一个无法复现的数字：

| 字段组 | 最少字段 |
|:---|:---|
| artifact | model id / revision、文件或 adapter 路径、量化方法、dtype、校准数据版本、配置摘要 |
| backend | loader、backend 版本、kernel / fallback、硬件、驱动与 PyTorch 版本 |
| workload | prompt 集、输入 / 输出长度、batch、并发、warmup、重复次数、采样参数 |
| result | 结果 JSON 路径、运行时间、成功 / 失败状态、原始日志或 trace 位置 |
| decision | baseline / candidate、质量门槛、资源预算、证据等级、accept / tune / reject |

| 阶段 | 必须记录 | 失败时的结论 |
|:---|:---|:---|
| 加载 | 模型格式、量化方法、backend、版本、错误信息 | `reject` 或记录为兼容性问题 |
| 运行 | 显存、TTFT、TPOT、端到端延迟、吞吐、并发 | 不能只依据文件大小判断 |
| 质量 | 固定评测集或任务指标、输出差异 | 质量不达门槛只能 `tune` 或 `reject` |
| 重复与对照 | baseline、重复次数、硬件和 workload | 证据不足只能 `tune` |

基准比较使用固定矩阵：同一模型、硬件、backend 和 workload 下，只替换一个候选变量；同时报告质量、显存、TTFT、TPOT、端到端延迟、吞吐和 P95 / P99。若出现 kernel fallback、加载成功但执行路径未切换、或只改善文件大小而没有服务收益，应单独记录为证据状态，不写成优化收益。

训练适配产物和推理量化产物要分开记录。QLoRA 输出通常是 adapter 或合并后的训练模型；GPTQ / AWQ / GGUF / FP8 输出则是面向推理加载或执行的 artifact。它们可以在同一项目链路中衔接，但不能用“adapter 能加载”替代“量化 backend 能加载”。

| 产物类型 | 典型来源 | 部署前需要确认 |
|:---|:---|:---|
| 训练适配产物 | QLoRA、LoRA、QAT | adapter / merged model、revision、合并方式、任务质量 |
| 权重量化产物 | GPTQ、AWQ、W8A16 | quant config、量化粒度、loader、实际 kernel |
| 文件格式产物 | GGUF 等 | 文件版本、匹配 backend、执行路径和兼容性 |
| 运行时量化配置 | FP8、KV Cache quant | dtype、scale、硬件支持、Cache 行为和质量 |

部署验证需要把“方法、格式、backend、kernel”分成四个字段记录。它们不是同义词：方法说明量化参数如何得到，格式说明 artifact 如何保存，backend 说明由谁加载和调度，kernel 才说明实际执行路径。

| 量化路线 | 典型格式或配置 | 常见 backend 方向 | 必须确认的执行证据 |
|:---|:---|:---|:---|
| W8A16 / 权重-only | INT8 权重、scale | Transformers、vLLM 等 | loader、反量化位置、显存与吞吐 |
| GPTQ | GPTQ config 与量化权重 | vLLM、Transformers 等 | quant config、group size、实际 kernel |
| AWQ | AWQ config 与量化权重 | vLLM、TensorRT-LLM 等 | 权重格式、硬件支持、kernel 路径 |
| GGUF | GGUF 文件 | llama.cpp 等匹配 backend | 文件版本、offload、执行后端 |
| FP8 | FP8 dtype、scale 配置 | vLLM、TensorRT-LLM 等 | GPU 架构、scale、Tensor Core 路径 |
| KV Cache quant | Cache dtype 与量化参数 | 目标 serving backend | Cache 分配、TTFT、TPOT、质量 |

QLoRA 的训练适配结果需要经过单独的 artifact 转换链：先确认 adapter 或 merged model 的 revision 和质量，再选择部署前权重量化或运行时配置，最后回到 66 的浮点 baseline 与 67 的真实 backend 对照。adapter 能加载，只能证明训练产物可用；它不能证明量化部署路径可用。

### 65 → 66 → 67 的入口契约

| 输入阶段 | 必须存在 | 下一阶段如何使用 |
|:---|:---|:---|
| 65 QLoRA 选型 | `65_qlora_artifact_manifest.json`、模型 revision、adapter / merged model 路径、质量结果 | 需要训练适配时，先转换或合并为可部署模型；不能跳过 artifact 检查 |
| 66 浮点 baseline | `66_g0_vllm_baseline.json`、固定 workload、backend/runtime、`quantization_format=none` | 作为 67 的 G0 参照，只允许 67 改变量化 artifact 或明确的 backend 变量 |
| 67 量化部署 | 量化 artifact、format metadata、kernel evidence、独立结果 JSON 和失败记录 | 判断当前硬件与 workload 下的 `accept / tune / reject` |

这条契约把训练适配、浮点性能和量化部署分开：任何一个环节缺少路径、版本、workload 或失败状态，都只能标记为待复测，不能把局部结果写成端到端收益。

| 阶段 | 主要产物 | 最小检查 |
|:---|:---|:---|
| 训练适配 | adapter 或 merged model | revision、合并方式、任务质量 |
| 部署转换 | GPTQ / AWQ / GGUF 或运行时配置 | 格式、量化参数、文件完整性 |
| backend 加载 | 可运行的服务 artifact | loader、版本、kernel 或执行路径 |
| workload 对照 | baseline / candidate 结果 | 质量、显存、TTFT、TPOT、吞吐、P99 |

## 报告应该怎么写

一个合格的量化报告至少要同时说明：

- 压的是哪一种对象；
- 显存、带宽、TTFT、TPOT、throughput 分别怎么变；
- 精度损失是否可接受；
- backend 和部署复杂度有没有额外代价；
- 最终是继续保留、继续调优，还是换路线。
- artifact manifest、结果 JSON 和失败记录是否能让别人复测。

## 证据等级

- `simulation`：只验证公式、误差或决策逻辑；
- `load_smoke`：真实 artifact 能加载并完成少量请求；
- `fixed_benchmark`：baseline 与候选使用相同 workload 完成对比；
- `deployment_decision`：有 manifest、质量门槛、重复运行、适用边界和 accept / tune / reject，才可形成部署建议。

## 文献与工程入口

- `66` 负责固定 workload 的浮点 baseline 和统一指标；
- `67` 负责真实量化 artifact 的加载、backend/kernel 证据和部署对照；
- `65` 负责 QLoRA 训练适配和 artifact manifest，不直接替代推理 baseline；
- Part 03 `10` Triton Quantization

## 典型阅读入口

- [02 PTQ 与 QAT 的介入时机](./02_ptq_and_qat_timing.md)
- [04 权重量化与后训练压缩](./04_weight_only_compression.md)
- [05 FP8 与 KV Cache 量化](./05_fp8_and_kv_cache_quantization.md)

## 项目结论

量化路线最终不是靠“名词更先进”成立，而是靠 benchmark 和部署收益成立。

## 回到项目

部署结论回填到 `66 推理性能比较 -> 67 量化推理与部署`；如果 PTQ 后质量不足，再按需回到 `65 QLoRA 选择` 生成新的模型或 artifact，并重新进入 66 / 67 对照。如果结果只证明模型变小，却没有证明 workload 下的系统收益，应保留为 `tune` 或 `reject`，不要写成已完成优化。
