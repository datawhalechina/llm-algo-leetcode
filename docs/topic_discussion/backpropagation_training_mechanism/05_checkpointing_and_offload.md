# 05. 激活检查点与卸载 | Activation Checkpointing and Offload

## 学习问题

本页比较两种训练状态策略：激活检查点（activation checkpointing）通过反向重算减少激活保存，卸载（offload）将状态移出 GPU 以降低设备驻留。学习结果是一张包含作用对象、正确性条件、容量收益与时间代价的策略对照表。

## 核心机制

| 策略 | 改变的生命周期 | 主要依赖资源 | 主要代价 |
|:---|:---|:---|:---|
| 激活检查点 | 前向传播只保留边界状态，反向传播重算中间激活 | 额外计算 | 反向传播时间增加 |
| 卸载 | 状态仍需保存，但驻留位置移到 CPU 或其他存储层 | PCIe、NVLink 或内存带宽 | 搬出、搬回和等待时间 |

激活检查点回答“是否长期保存”，卸载回答“保存在哪里”。两者都可能降低 GPU 峰值，但不能互相替代，也不直接减少参数、梯度或优化器状态的逻辑规模。

### 激活检查点的正确性条件

设基线与 Checkpoint 路径保存的 activation 驻留量分别为 $M_{saved,base}$ 和 $M_{saved,ckpt}$，两者差值

$$
\Delta M_{ckpt}=M_{saved,base}-M_{saved,ckpt}
$$

表示理论节省量。对应的新增时间为 $\Delta T_{ckpt}=T_{ckpt}-T_{base}$，主要来自反向重算和调度开销。重算路径必须能够重放原前向传播；比较两条路径时，需要同时检查输出、输入梯度和参数梯度。

| 条件 | 不满足时的风险 | 验证方式 |
|:---|:---|:---|
| 参数和输入一致 | 重算得到不同激活 | 固定模型状态和输入 |
| 随机状态可恢复 | Dropout 等随机层产生不同结果 | 固定或保存 RNG 状态 |
| 无不受控副作用 | 重算改变计数器、缓存或外部状态 | 保持检查点函数可重放 |
| 分段边界完整 | 依赖跨越边界但未正确传递 | 比较分段路径与基线路径的输出和梯度 |

激活检查点粒度越细，长期保存可能越少，但重算次数、调度和内核开销也可能增加；粒度越粗，重算次数减少，却可能保留更多边界状态。

### 卸载的容量与传输账本

设实际搬出的状态量为 $bytes$，搬出和搬回的有效带宽分别为 $B_{out}$ 与 $B_{in}$。理论往返时间为：

$$
T_{roundtrip}\approx\frac{bytes}{B_{out}}+\frac{bytes}{B_{in}}.
$$

若其中 $T_{overlap}$ 能被其他计算覆盖，真正暴露在关键路径上的时间为：

$$
T_{exposed}=\max(0,T_{roundtrip}-T_{overlap}).
$$

搬回必须早于反向传播的消费点，否则 GPU 会等待数据。Checkpoint 与 Offload 的共用符号见[公式索引](./formula_reference.md)。

### 候选策略比较

| 观察到的主导压力 | 优先候选 | 需要确认的证据 |
|:---|:---|:---|
| 前向传播末尾的激活占主导 | 激活检查点 | 峰值下降、梯度一致、重算时间可接受 |
| 可搬运状态占主导且链路带宽足够 | 卸载 | 驻留下降、传输暴露时间、消费点无等待 |
| 参数、梯度或优化器状态占主导 | dtype、LoRA 或 ZeRO | 状态字节数和分片口径 |
| 激活与常驻状态共同形成峰值 | 组合策略 | 每项策略的独立贡献与组合复测 |

![激活检查点与卸载的容量—时间交换](../../public/topic_discussion/backpropagation_training_mechanism/checkpointing_offload_comparison.svg)

## 验证与下一步

完成本页后，应能先从生命周期账本选择候选策略，再用正确性、容量和时间三类字段描述预期变化。真实 GPU 对照由 Part02 · 19、42 和 76 提供。

下一页进入 [06 正确性与性能证据](./06_correctness_and_performance_evidence.md)，为策略比较建立采用门槛。

## 相关阅读

| 资料 | 用途 |
|:---|:---|
| [Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.md) | 检查保存—重算的正确性与峰值代价。 |
| [Part 02 · 42 激活卸载](../../02_PyTorch_Algorithms/42_Activation_Offload.md) | 检查搬出、搬回和暴露等待。 |
| [Part 02 · 76 激活检查点与卸载对照](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) | 在固定训练工作负载下比较策略。 |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | 理解激活检查点的经典空间—计算交换。 |
| [ZeRO-Offload](https://arxiv.org/abs/2101.06840) | 理解 CPU 卸载如何进入训练系统。 |
| [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint) | 核对 PyTorch checkpoint 接口和随机状态说明。 |
