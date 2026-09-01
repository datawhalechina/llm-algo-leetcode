# 反向传播与训练机制正文

本专题解决一个具体的工程问题：训练为什么会因为计算图、保存状态或更新节奏而变慢、占满显存，甚至产生看似正常但实际不可信的 loss。

它是基础支撑专题，不替代 SFT、LoRA、显存优化或性能分析项目。阅读顺序应保持一条因果链：

```text
计算图 → backward 依赖 → 保存状态 → 显存账本 → 重算 / 搬运 / 累积 → 实测决策
```

## 先判断问题发生在哪一层

不要从“换一个优化技巧”开始。先根据现象定位问题对象，再选择需要阅读或验证的内容。

| 现象 | 首先检查 | 需要回答的问题 | 下一步 |
|:---|:---|:---|:---|
| forward 能跑，backward 报错或梯度异常 | 计算图与 autograd | 梯度从 loss 沿哪条路径回传？参数是否真的收到梯度？ | [01 反向传播与计算图](./01_backpropagation_and_graph.md) |
| attention 能跑，但不知道 backward 保存了什么 | attention backward | 哪些中间量被保存、重算或融合？ | [02 自动微分与 Attention 反向传播](./02_autograd_and_attention_backward.md) |
| loss 有数值，但监督效果不可信 | 标签与 loss 对齐 | 哪些 token 进入了 loss？shift、mask、ignore_index 是否一致？ | [03 损失对齐与显存账本](./03_loss_alignment_memory_ledger.md) |
| 训练显存过高或长序列 OOM | 状态生命周期 | 峰值来自 activation、参数、梯度、optimizer state 还是 workspace？ | [04 Checkpointing 与 Offload](./04_checkpointing_and_offload.md) |
| 单步能跑，但有效 batch、吞吐或更新频率混乱 | 训练时序 | backward 多少次才执行一次 optimizer step？ | [05 梯度累积、决策与性能分析](./05_accumulation_decision_profiling.md) |

## 一条完整的机制链

### 1. 计算图决定 backward 的依赖

forward 不只是产生 logits，它还建立了后续反向传播需要的依赖关系。loss 是反向传播的起点，参数和中间节点是梯度经过的路径。某个中间结果是否需要保留，取决于后续梯度公式是否还会用到它。

因此，看到显存上涨时，不能只问“batch 是否太大”，还要问：

- 哪些中间结果在 backward 前仍然存活；
- 哪些分支、残差或共享参数延长了它们的生命周期；
- 哪些状态属于模型训练本身，哪些只是临时 workspace。

### 2. attention backward 把保存边界具体化

在 attention 中，反向路径可以简化为：

```text
dV → dP → dS → dQ / dK
```

这条路径把数学依赖和实现代价连接起来：softmax、缩放和 mask 会影响中间状态；完整保存 score 或 probability 矩阵会增加显存与 IO；fused backward 或局部重算则改变访问方式，但不改变梯度公式。

### 3. loss 对齐决定账本是否值得相信

显存分析之前必须确认监督口径。对 causal LM 而言，logits 与 labels 要做 next-token shift；prompt、response、padding 和 EOS 是否参与 loss，要由 mask 与 `ignore_index` 明确表达。

如果标签错位，loss 仍然可能是一个正常浮点数，但它不能作为训练质量或显存策略的可靠对照。

### 4. 状态账本决定优化策略的作用范围

训练显存至少要区分以下对象：

| 对象 | 生命周期 | checkpointing 是否直接减少 | 常见代价 |
|:---|:---|:---:|:---|
| 参数 | 整个训练过程 | 否 | 模型规模与 dtype |
| 梯度 | backward 后到清零或覆盖 | 否 | 参数量、梯度 dtype、通信 |
| optimizer state | 参数更新期间持续存在 | 否 | AdamW 的额外状态 |
| activation | forward 到对应 backward 完成 | 是 | 重算时间 |
| workspace / 通信缓冲 | 某些算子或通信阶段 | 通常不是 | kernel、allocator、通信实现 |

只有先知道峰值属于哪一类状态，才有理由选择 checkpointing、offload、梯度累积或参数分片。

### 5. 优化策略把代价转移到不同资源

| 策略 | 主要改变 | 可能换来的代价 | 适合验证的指标 |
|:---|:---|:---|:---|
| Checkpointing | 减少 activation 的驻留 | 重算和 step time 增加 | peak memory、step time、loss |
| Offload | 改变状态驻留位置 | CPU-GPU 搬运、同步和带宽压力 | transfer time、吞吐、峰值显存 |
| Gradient accumulation | 降低单次 micro-batch 压力 | update cadence 改变，训练时间增加 | effective batch、update time、质量 |
| Profiling | 暴露时间和显存热点 | 本身不是优化策略 | trace、阶段耗时、内存时间线 |

## 证据边界

| 环境 / 方法 | 可以确认 | 不能单独确认 |
|:---|:---|:---|
| CPU 小例子 | 梯度路径、loss shift、mask、step 次数和数值关系 | GPU 峰值显存、kernel 吞吐、CUDA workspace |
| GPU 固定 workload | peak allocated / reserved、step time、吞吐、OOM | 对所有模型和 workload 都成立 |
| GPU 重复 benchmark | 候选策略在指定设备和 workload 下的稳定性 | 不同设备上的同等收益 |
| profiler trace | 时间热点、重算、搬运、同步和阶段归因 | 没采集到的隐性系统开销 |

CPU 验证应当先保证机制和代码正确；真实 GPU 验证再回答容量、吞吐和代价转移。具体项目中，`73` 建立训练 baseline，`76` 比较 activation 策略，`75` 做预算敏感性，`74` 用 profiler trace 收口。没有匹配 workload 或 trace 时，结论应保留为待验证。

## 阅读与项目衔接

基础机制页负责解释为什么会出现某种状态和代价，项目页负责在固定条件下测量它们。推荐顺序为：

```text
01 计算图
→ 02 Autograd / Attention backward
→ 03 loss 对齐与显存账本
→ 04 Checkpointing / Offload
→ 05 梯度累积与训练闭环
→ 73 → 76 → 75 → 74
```

如果问题转向 SFT、LoRA 或数据处理，进入 [监督微调与训练工程](../fine_tuning_training/intro.md)；如果问题转向显存预算和策略选择，进入 [显存优化](../memory_performance_tuning/intro.md)；如果需要定位真实时间热点，进入 [性能分析](../profiling/intro.md)。
