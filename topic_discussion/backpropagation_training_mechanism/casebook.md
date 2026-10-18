# 反向传播与训练机制问题索引

先按现象定位最小机制和证据，再决定是否需要改变训练策略；不要把 loss 曲线、一次峰值或单次耗时直接当作结论。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 损失无法连接到参数，或参数 `.grad` 为 `None` | 参数是否可训练，损失是否仍在同一计算图 | `requires_grad`、`grad_fn`、参数梯度与图切断位置 | [01 训练闭环与计算图](./01_training_loop_and_computation_graph.md)、[02 自动求导与 Transformer 反向传播](./02_autograd_and_transformer_backward.md) | [Part 00 · 07 PyTorch 自动求导与反向传播](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb) |
| 残差或共享参数的梯度异常 | 分支贡献是否在汇合处相加 | 参考路径与汇合后总梯度 | [02 自动求导与 Transformer 反向传播](./02_autograd_and_transformer_backward.md) | [Part 02 · 10 Attention 反向传播与自定义 Autograd](../../02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.ipynb) |
| 损失下降但监督行为不可信 | 标签移位、掩码、`ignore_index` 和有效 token 分母 | 监督 token 计数、损失分子/分母与标签样例 | [03 监督、梯度尺度与有效更新](./03_supervision_gradient_scale_effective_update.md) | [后训练优化专题](../post_training_optimization/intro.md) |
| 梯度累积与完整批次不一致，或调度器推进过快 | 累计窗口、归约方式、优化器与调度器步数 | 参数增量、有效 token 数、更新计数与学习率历史 | [03 监督、梯度尺度与有效更新](./03_supervision_gradient_scale_effective_update.md) | [Part 02 · 11 梯度累积](../../02_PyTorch_Algorithms/11_Gradient_Accumulation.ipynb) |
| 理论账本与真实峰值对不上 | 哪些对象在峰值阶段同时驻留，`allocated` 与 `reserved` 是否分开记录 | 对象生命周期、峰值阶段与显存快照 | [04 训练状态生命周期](./04_training_state_lifecycle.md) | [性能分析 · 04 显存时间线与对象驻留](../performance_analysis/04_memory_timeline_and_residency.md) |
| checkpoint 后输出或梯度变化 | 重算是否重放相同输入、参数、随机状态和副作用 | 输出、输入梯度、参数梯度与 RNG 对照 | [05 激活检查点与卸载](./05_checkpointing_and_offload.md) | [Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.ipynb) |
| 卸载后等待时间过长 | 搬出、搬回是否覆盖在计算中，何时首次消费状态 | H2D/D2H 时间、消费点和暴露等待 | [05 激活检查点与卸载](./05_checkpointing_and_offload.md) | [Part 02 · 42 激活卸载](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb) |
| FP16 出现 NaN/Inf，或显存降低但训练更慢 | 缩放、有限值检查、裁剪、跳步与守护指标 | scale、梯度范数、跳过更新、峰值、单步时间和质量 | [06 正确性、数值稳定与性能证据](./06_correctness_and_performance_evidence.md) | [性能优化专题](../performance_optimization/intro.md) |
