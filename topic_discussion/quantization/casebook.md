# 量化与压缩正文

量化不是“把模型改成更小的 dtype”这么简单。一个可用的量化方案，必须同时回答七个问题：压缩了什么对象、误差从哪里来、在什么时候介入、由什么方法或格式表达、由哪个 backend 执行、在目标 workload 下表现如何，以及是否值得采用。

本专题正文采用下面的判断链：

`压缩对象 → 误差来源 → 介入时机 → 方法 / 格式 → backend 执行路径 → workload 验证 → 部署决策`

它用于组织不同 Notebook 和项目节之间的关系，不把理论估算、CPU 模拟或单次加载结果写成完整的部署结论。

## 先定位问题

| 现象 | 首先检查 | 下一步 | 不应直接推出的结论 |
|:---|:---|:---|:---|
| 权重装不进目标显存 | 权重 dtype、量化粒度、额外 scale / zero-point 和加载峰值 | 先做 W8A16 或权重-only 方案，再检查 backend | 文件变小就一定能加载成功 |
| 量化后质量下降 | 校准数据、敏感层、误差分布和评测口径 | 比较 PTQ、GPTQ / AWQ、QAT 或低比特适配 | bit 数越低，质量一定越差或一定可接受 |
| 量化后没有变快 | 反量化位置、低比特 kernel、硬件支持和 backend 路径 | 用固定 workload 分解 prefill、decode、端到端时间 | 显存减少必然带来吞吐提升 |
| 长上下文或并发受限 | KV cache dtype、序列长度、请求数和 cache 管理 | 单独评估 KV cache quant 与调度策略 | 权重量化会自动解决 KV cache 压力 |
| 量化模型可以加载但不适合上线 | 格式、算子实现、质量、延迟和运维复杂度 | 运行同口径 baseline / candidate benchmark | 能加载等于适合生产部署 |

量化决策的起点应是约束，而不是算法名称。相同的 INT4 可能只改变权重存储，也可能在运行时触发反量化；相同的模型文件在不同 backend 上也可能走不同 kernel。

## 区分方法、格式与 backend

| 类别 | 典型对象 | 例子 | 主要回答的问题 |
|:---|:---|:---|:---|
| 量化方法 | 如何估计 scale、处理误差或恢复质量 | PTQ、QAT、GPTQ、AWQ、SmoothQuant | 如何得到量化参数，误差如何控制 |
| 数值格式 | 权重、激活或 cache 采用什么表示 | INT8、INT4、NF4、FP8、GGUF | 数据如何存储和传递 |
| 执行 backend | 如何加载并执行这些表示 | Transformers、bitsandbytes、vLLM、llama.cpp、TensorRT-LLM | 哪个 kernel 执行，实际显存和速度如何 |

GPTQ 和 AWQ 首先是后训练权重量化方法；GGUF 首先是文件格式与部署封装。不能把它们当成同一层面的“量化算法”直接比较，也不能用 GPTQ / AWQ 的启动参数推断 GGUF backend 的行为。

## 量化对象决定收益和代价

| 压缩对象 | 主要减少什么 | 主要代价 | 至少应观察的指标 |
|:---|:---|:---|:---|
| 权重 | 模型常驻容量、权重读取带宽 | scale / zero-point、反量化和 kernel 适配 | 加载峰值、常驻显存、加载是否成功、吞吐与质量 |
| 激活 | 中间状态和部分工作区 | 重算、低精度数值误差、算子支持 | forward / backward 峰值、step time、溢出或 NaN |
| KV cache | 长上下文和并发请求的 cache 容量 | cache 误差、读写带宽、backend 支持 | 单请求长度、并发容量、TTFT、TPOT、质量 |

权重量化不能代替激活或 KV cache 优化。显存账本中应分别记录参数、梯度、optimizer state、activation、KV cache、临时 workspace 和 allocator 保留量；只记录模型文件大小不足以解释运行时峰值。

## 介入时机：PTQ、QAT 与低比特训练适配

| 方案 | 适合的问题 | 训练 / 校准成本 | 主要风险 | 先验证什么 |
|:---|:---|:---|:---|:---|
| PTQ | 已有模型需要快速压缩或部署 | 需要代表性校准数据，通常不改训练流程 | 敏感层误差和任务质量下降 | 误差、任务质量、加载与推理指标 |
| GPTQ / AWQ | 权重量化后需要更好的误差控制 | 需要校准和方法专用实现 | 方法、格式和 backend 绑定 | 量化格式、kernel 路径、同口径质量与速度 |
| QAT | PTQ 无法满足质量要求 | 需要重新训练或适配 | 训练成本、数值稳定性和部署匹配 | 训练收敛、验证质量、最终格式 |
| QLoRA / 低比特适配 | 训练显存有限，基座权重以低比特加载 | 仍需训练 adapter | 训练表示与部署表示不是同一件事 | trainable 参数、显存、验证质量和合并 / 加载路径 |

选择介入时机时，先固定质量下限和部署目标，再比较成本。QLoRA 的低比特加载不能直接证明 GPTQ、AWQ 或 GGUF 的推理收益；训练侧的节省也不能直接替代 serving benchmark。

## 为什么量化不一定更快

量化减少的是某一类数据的存储或搬运量，而不是自动减少所有运行时工作。实际速度取决于：

1. 硬件是否有对应的低比特 Tensor Core 或高效指令；
2. backend 是否使用匹配的低比特 kernel；
3. 反量化、scale 读取和临时 workspace 是否抵消了带宽收益；
4. workload 是长 prefill、短 decode、单请求还是高并发。

因此，必须把“文件大小”“加载峰值”“运行时显存”和“吞吐 / 延迟”分开记录。某种量化在 24 GB GPU 上可能主要改善容量余量，在另一种硬件或短序列 workload 上可能几乎没有速度收益，甚至更慢。

## CPU 与真实 GPU 的证据边界

| 环境 | 可以验证 | 不能单独证明 |
|:---|:---|:---|
| CPU 或纯 PyTorch 模拟 | bit / byte 换算、scale / zero-point、舍入误差、误差统计、格式字段和决策逻辑 | 真实 kernel、CUDA workspace、显存峰值、GPU 吞吐、backend 兼容性 |
| GPU + 原生 PyTorch | dtype 运算、显存峰值、部分低精度算子和固定 workload 的时间 | 某个 serving backend 的完整部署行为 |
| GPU + 目标 backend | 量化模型能否加载、实际 kernel、显存、TTFT / TPOT、吞吐、并发和质量 | 不同硬件、版本或 workload 下的普遍结论 |

真实部署结论至少需要形成闭环：

`量化 artifact → backend 成功加载 → 确认格式与执行路径 → 同 workload baseline / candidate → 质量与资源指标 → accept / tune / reject`

没有完成这条链时，应把结果标记为“理论估算”“CPU 模拟”“GPU 探针”或“加载验证”，不要写成量化方案已经优于 baseline。

## 项目分工与收口

量化路线中的项目按问题分工：

- `65` 负责训练侧 QLoRA / 低比特适配的选择，不替代推理部署验证；
- `66` 负责浮点模型的推理 baseline，为量化比较提供固定 workload 和基线指标；
- `67` 负责 GPTQ / AWQ 真实部署，GGUF 作为独立格式与 backend 路径扩展；
- `40` 负责 GPTQ / AWQ 的机制模拟，解释误差补偿，不代替真实 backend；
- `41` 负责 FP8 与 KV cache 量化的机制和边界，重点区分权重、激活与 cache。

`67` 的最终报告应至少包含：模型与量化 artifact、量化方法和格式、backend 与版本、硬件与 dtype、校准数据摘要、prompt / generated tokens、显存、TTFT、TPOT、端到端延迟、吞吐、质量指标、失败信息和最终决策。

## 最小决策模板

每次比较都记录：

`对象 → 误差边界 → 介入时机 → 方法 → 格式 → backend → workload → 质量 → 显存 → 延迟 / 吞吐 → 部署复杂度 → 决策`

建议使用以下决策规则：

- 质量不达下限或 backend 无法加载：`reject`；
- 质量与资源均满足，但样本量、硬件覆盖或重复运行不足：`tune`；
- 在固定 workload、质量下限和资源预算内，候选方案有稳定收益：`accept`。

最终结论必须说明适用的模型、硬件、backend 和 workload。它是有条件的工程决策，不是对所有模型和平台的普遍承诺。
