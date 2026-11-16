# 通信与并行专题术语表

全教程中英文写作遵循[术语表与写作规范](../../docs/terminology.md)。本表只补充通信与并行语境中容易混淆的概念；数学符号、单位和理想化成本模型统一查阅[通信与并行公式索引](./formula_reference.md)。

| 中文主称 | 英文 / 缩写 | 本专题含义 | 容易混淆之处 |
|:---|:---|:---|:---|
| 进程编号 | rank | 分布式进程在进程组中的编号 | 不一定等于物理 GPU 编号 |
| 进程总数 | world size | 参与指定分布式作业或进程组的 rank 数量 | 不一定等于机器数，也不自动等于全部可见 GPU 数 |
| 进程组 | process group | 参与同一组通信操作的 rank 集合 | world group 只是默认进程组 |
| 拓扑域 | topology domain | 共享某类链路与带宽条件的一组设备或 rank | 进程组描述通信成员，拓扑域描述物理连接条件，两者不一定相同 |
| 通信后端 | communication backend | 实现分布式通信语义的运行时；GPU 场景通常使用 NCCL | PyTorch Distributed 是编程接口，NCCL 是常用 GPU 通信后端 |
| 点对点通信 | P2P communication | 一个 rank 向另一个 rank 发送或接收数据 | 与要求多个 rank 共同参与的集合通信不同 |
| 集合通信 | collective | 多个 rank 以约定语义共同完成的数据交换 | 不是任意点对点消息 |
| 载荷 | payload | 一次通信传输或聚合的有效数据字节数 | 不等于总链路流量或模型大小 |
| All-Reduce | All-Reduce | 对所有 rank 的输入归约，并把完整结果返回每个 rank | Reduce-Scatter 只保留结果分片 |
| Reduce-Scatter | Reduce-Scatter | 先归约，再把结果分片留给不同 rank | 常与 All-Gather 组成分片状态流程 |
| All-Gather | All-Gather | 收集各 rank 分片并恢复完整张量 | 不执行数值归约 |
| All-to-All | All-to-All | 按目标 rank 重排并交换不同数据片段 | EP 中 payload 随路由分布变化 |
| 通信—计算重叠 | overlap | 通信与无依赖计算在时间线上并发执行 | 总通信时间没有消失，只减少暴露部分 |
| 暴露通信时间 | exposed communication time | 未被有效计算覆盖、仍落在关键路径上的通信时间 | 不等于 NCCL Kernel 总时长 |
| 数据并行 | DP | 多个 worker 持有模型副本并处理不同数据 | 训练 DP 通常同步梯度，Serving Replica 主要分发请求 |
| 分布式数据并行 | DDP | 复制模型并通过梯度同步保持参数更新一致 | 不切分参数、梯度和优化器状态的常驻归属 |
| 全分片数据并行 | FSDP | 按进程分片参数、梯度和优化器状态，并按执行需要聚合 | 与 ZeRO 思路相关，但 API、执行时机和状态字典契约不同 |
| 零冗余优化 | ZeRO | 按阶段逐步分片优化器状态、梯度和参数 | 阶段编号描述状态归属，不直接代表端到端速度 |
| 张量并行 | TP | 在层内切分矩阵、Head 或中间表示 | 与切层的 PP 不同 |
| 流水线并行 | PP | 按层或 Block 切分 stage | micro-batch 用于填充流水线，不是梯度累积本身 |
| 流水线气泡 | pipeline bubble | stage 因依赖尚未满足而处于空闲的时间或槽位 | 不能只用 micro-batch 数量推断，仍受最慢 stage 限制 |
| 上下文并行 | CP | 沿序列或上下文维切分 Attention 状态与计算 | 与 TP 的 Head/隐藏维切分不同 |
| 专家并行 | EP | 将 MoE 专家放置到不同 rank，并交换 token | 与静态 TP 不同，负载由路由动态决定 |
| 设备网格 | Device Mesh | 用多个命名维度组织 rank 与进程组，承载组合并行布局 | 只是拓扑和分组抽象，不自动给出最优切分策略 |
| 整组调度 | gang scheduling | 只有所需 worker 资源全部就绪时才启动分布式任务 | 解决资源一致启动，不证明集合通信或扩展效率 |
| 放置组 | placement group | 为一组 worker 声明资源打包与放置约束 | 属于编排层概念，不等于 PyTorch 进程组 |
| 专家容量 | expert capacity | 单个专家在一批输入中允许接收的 token 上限 | 容量过小会溢出，过大可能放大内存和负载不均 |
| 容量溢出 | overflow | 路由到专家的 token 超出其容量后的状态 | 丢弃、回退或重路由会产生不同质量和性能结果 |
| 服务副本 | serving replica | 每个实例持有可独立处理请求的模型副本 | 不同步训练梯度，不等于训练数据并行 |
| 最慢进程时间 | max-rank time | 同一步骤中所有 rank 用时的最大值 | 全局同步性能由最慢 rank 决定，rank 平均值会隐藏热点 |
| 扩展效率 | scaling efficiency | 增加设备后实际收益相对理想线性收益的比例 | “能运行”或总吞吐增加不代表效率高 |
| 检查点清单 | checkpoint manifest | 登记检查点标识、状态角色、分片、大小和提交状态的清单 | 文件存在不代表状态完整或已经原子提交 |
| 可恢复性 | recoverability | 检查点能否在声明的 rank 布局和训练状态下恢复一致执行 | 与仅加载模型权重进行推理不同 |

正文使用中文主称；代码、Profiler 事件、环境变量和 JSON 字段保留原始英文标识符。
