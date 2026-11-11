# 通信与并行实验资产表

本表登记通信实验使用的 Profile、比较对象、结果路径和证据状态。环境创建命令见[实验 Quickstart](./quickstart.md)，Profile 与隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)，实际依赖和硬件写入结果 JSON。

## 环境角色

| 环境 ID | 直接入口 | 可形成的证据 |
|:---|:---|:---|
| `PAR-CPU` | Part 02 的 Step 4 与框架一致性核对 | 两进程 Gloo collective、DTensor 布局、上下文切分和可用的 Pipeline API 语义 |
| `PAR-CUDA-MECHANISM` | 27–29、46–48、PAR-CONTEXT、PAR-REPLICA | 单机多卡 collective、rank 布局、payload、局部计算与暴露通信 |
| `PAR-TRAIN-CLUSTER-RAY`（规划） | 79、PAR-MODEL-INTERNAL、80 的跨节点项目 | Ray Train 管理 worker、资源预留、失败恢复和任务编排；底层张量通信仍由 PyTorch Distributed / NCCL 完成 |
| `PAR-SERVING-RAY`（规划） | 81 的集群 Serving 路径 | Ray Serve 管理副本、placement group、跨节点 worker 和弹性伸缩；模型执行与缓存指标仍由 vLLM / SGLang 产生 |
| 独立 Serving backend | 81 的单机或不使用 Ray 的路径 | vLLM / SGLang 的请求、模型并行、副本和服务指标 |

`PAR-CUDA-MECHANISM` 不安装 Ray。Ray 只在项目需要跨节点资源编排、整组调度、故障恢复、服务副本或弹性伸缩时进入独立环境；放置组创建成功不能替代 NCCL 正确性、带宽或扩展效率证据。

## 多卡验证矩阵

RTX 3080 ×2、每卡 20 GB 可以完成本专题的双卡机制验证和三个项目的双卡基线。四卡不是所有实验的前置条件，只用于验证二维设备网格或复合并行；六卡可用于扩展曲线，但不作为首轮验收要求。

| 设备条件 | 优先验证 | 能形成的结论 | 暂不能外推 |
|:---|:---|:---|:---|
| 2 × RTX 3080 20 GB | 47 → 48 → 27 / 29 / 28 → PAR-CONTEXT / PAR-REPLICA → 46 | 单机双卡 collective、状态/张量/阶段/序列切分、专家与副本负载证据 | 跨节点效率、二维组合并行与大规模长尾 |
| 2 × RTX 3080 20 GB | 79、80、81 的双卡基线；87 的 PP2、TP2 或 CP2 单轴候选 | 真实模型下单轴候选的容量、正确性与关键路径 | `TP×PP`、`TP×CP` 等二维网格收益 |
| 4 × GPU | 87 的二维 Device Mesh；80 的更多专家/rank；81 的副本×模型并行组合 | 组合布局、进程组隔离、负载和恢复范围 | 跨机网络与生产集群弹性 |

## 机制实验注册表

| 实验 ID / 页面 | 固定对照 | 主要指标 | 当前状态 |
|:---|:---|:---|:---|
| `PAR-CANDIDATE-GATE` / Part 01 · 26 | 候选要求 → 环境约束 | 每 rank 容量、链路域、并行度乘积、拒绝原因 | CPU 机制计算器已验证 |
| `PAR-CHECKPOINT-RECOVERY` / Part 01 · 28 | 完整状态 → 缺失状态 / 分片 | manifest 完整性、保存开销、回滚成本、恢复条件 | CPU 机制计算器已验证 |
| `PAR-ZERO-STATE` / 27 | DDP → ZeRO-1/2/3 | 状态字节、峰值显存、通信量、step time | 5.1–5.4 已统一；真实多卡待验证 |
| `PAR-PIPELINE-SCHEDULE` / 28 | 单 stage → 两 stage；fill-drain → 1F1B | bubble、activation 峰值、stage time、吞吐 | 5.1–5.4 已统一；真实多卡待验证 |
| `PAR-TENSOR-LAYOUT` / 29 | 单卡 MLP → Column/Row TP | 数值等价、collective bytes、延迟、显存 | 5.1–5.4 已统一；真实多卡待验证 |
| `PAR-EXPERT-DISPATCH` / 46 | balanced → skewed routing | capacity、overflow、All-to-All bytes、max/mean load | 页面与 GPU 契约已统一；真实多卡待验证 |
| `PAR-NCCL-COLLECTIVE` / 47 | AllReduce / AllGather / ReduceScatter × payload | median/P95、有效带宽、每 rank 时间 | 公共多卡基线已统一；真实多卡待验证 |
| `PAR-OVERLAP-HOTSPOT` / 48 | serial → overlap | 通信总时长、暴露通信、overlap ratio、rank imbalance | 已复用公共身份与结果字段；真实多卡待验证 |
| `PAR-CONTEXT-RING` / PAR-CONTEXT | 复制完整 K/V → 序列切分与交换 | 正确性、状态字节、交换字节、延迟 | 机制实验已统一；Ring kernel 证据待项目验证 |
| `PAR-REPLICA-ROUTING` / PAR-REPLICA | local / balanced / locality-aware | makespan、每副本负载、队列等待 | 只形成副本机制证据 |

49 是 CPU 策略决策页，消费上述资产，不单独制造 GPU 结果。

## 项目接口

项目页复用机制字段，但必须重新固定真实模型、输入、拓扑与执行栈；机制 JSON 不能直接充当项目结论。

| 稳定项目 ID | 消费的机制资产 | 项目补充的证据 | 当前接口状态 |
|:---|:---|:---|:---|
| `PAR-TRAINING-STATE` / 79 | 27、47、48、`PAR-CHECKPOINT-RECOVERY` | DDP / FSDP / ZeRO 的真实训练显存、吞吐、通信与恢复条件 | 已接 `distributed-benchmark/v1` 聚合；真实多卡待验证 |
| `PAR-MODEL-INTERNAL` / 87 | 28、29、PAR-CONTEXT、47、48 | PP / TP / CP 与混合布局的真实模型结果 | CPU 项目已建设；真实执行 adapter 待接入 |
| `PAR-MOE-EXPERT` / 80 | 47、46、48 | 真实 MoE workload 的有效 token、overflow、All-to-All 与长尾 | 已接聚合器；真实 MoE backend 待接入 |
| `PAR-DISTRIBUTED-INFERENCE` / 81 | PAR-REPLICA、29、PAR-CONTEXT、48 | 副本、模型并行与 PD 分离的 Serving 指标 | 5.1–5.6 已分层；vLLM / SGLang adapter 待接入 |

四个项目共享结果 envelope，但不互相读取 Notebook 内部变量。79 交付训练状态并行基准，87 交付模型内部布局基准，80 交付专家路由与 All-to-All 基准，81 交付分布式 Serving 基准；49 只读取这些项目的归一化结果并形成组合建议。

| 项目 | 固定基线 | 候选范围 | 必须输出的专属证据 | 不承担的职责 |
|:---|:---|:---|:---|:---|
| 79 `PAR-TRAINING-STATE` | 相同模型、global batch 与 dtype 的 DDP | FSDP / ZeRO 状态分片 | 训练峰值、step time、吞吐、collective、checkpoint 恢复 | PP / TP / CP 布局和 Serving 指标 |
| 87 `PAR-MODEL-INTERNAL` | 可运行的单轴布局或单卡语义基线 | PP、TP、CP 及二维 Device Mesh | 数值正确性、局部状态、bubble、collective、扩展效率 | 优化器状态分片、专家路由和请求调度 |
| 80 `PAR-MOE-EXPERT` | 相同 token、专家与路由输入的本地或均衡候选 | EP 放置、capacity 与路由候选 | 有效 token、overflow、All-to-All、最慢 rank、输出正确性 | 通用 TP / PP 组合和服务副本路由 |
| 81 `PAR-DISTRIBUTED-INFERENCE` | 相同模型、请求集与服务目标的单实例路径 | Replica、TP 与 PD 分离 | TTFT、TPOT、吞吐、尾延迟、队列、KV handoff | 训练梯度同步和后训练质量提升 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、页面与策略 |
| 硬件与拓扑 | GPU、world size、rank-device 映射、单机/跨机、链路信息 |
| 通信 | backend、collective、payload bytes、调用次数、同步频率 |
| 时间 | compute、communication、exposed communication、median/P95 |
| 负载 | 每 rank 时间或工作量、max/mean、overflow 或空闲比例 |
| 恢复 | checkpoint ID、manifest、rank 分片覆盖、提交状态与恢复结果 |
| 结论 | correctness、failure、fallback、`evidence_level`、decision |

机制实验不能把环境预检写成通信证据，也不能把单机双卡结果外推为跨机扩展效率。项目页必须固定真实模型、输入和执行后端后重新测量。
