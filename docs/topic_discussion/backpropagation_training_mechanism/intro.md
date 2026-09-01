# 反向传播与训练机制（Backpropagation and Training Mechanics）

> 专题类型：基础支撑　主服务目标：计算图与训练状态理解

## 专题定位与 Infra 层定位

本专题串起训练机制主线：先看梯度怎么沿计算图回传，再看 attention backward、loss 对齐、activation 保存、checkpointing、offload 和梯度累积怎样一起影响训练节奏与显存代价。它主要连接 Infra-L2–Infra-L3：Infra-L2 解释算子、kernel 和自动求导如何执行，Infra-L3 解释框架如何构建计算图、保存 activation 并调度 backward；Infra-L1 的显存容量与带宽是边界，checkpoint、offload 和梯度累积是跨层策略。

因此，学习者需要同时看计算、内存和通信代价，而不是把本专题当作独立优化方案。若问题进入 SFT、LoRA 或训练项目交付，应转到监督微调与训练工程；若重点是显存预算和策略选型，应转到显存优化。

## 推荐入口

推荐把本专题作为 [监督微调与训练工程](../fine_tuning_training/intro.md) 或 [显存优化](../memory_performance_tuning/intro.md) 的机制桥接。需要理解训练为什么变慢、爆显存或必须做 checkpointing 时，再进入对应的 Task，而不是把本专题当作独立项目线顺序完成。

## 前置阅读

建议先具备 PyTorch 张量与自动求导基础；如果要进入 attention backward、checkpointing 或 offload，可按 [Part 02 · 17 自动求导基础](../../02_PyTorch_Algorithms/17_Autograd_Basics.md) → [Part 02 · 18 激活与损失反向传播](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md) → [Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md) → [Part 02 · 42 激活 Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.md) 回看来源 Notebook。

## 主学习线

`Task1-5` 是学习路线，指向 Part 02 的具体小节和项目；最后一列对应专题正文页，图册作为补充阅读。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | backward 总览与计算图 | [Part 02 · 17 自动微分基础](../../02_PyTorch_Algorithms/17_Autograd_Basics.md) | [01 反向传播与计算图](./01_backpropagation_and_graph.md) |
| Task2 | autograd 与 attention backward | [Part 02 · 17 自动微分基础](../../02_PyTorch_Algorithms/17_Autograd_Basics.md) → [Part 02 · 18 激活与损失反向传播](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md) | [02 自动微分与 Attention 反向传播](./02_autograd_and_attention_backward.md) |
| Task3 | loss 对齐与显存账本 | [Part 02 · 18 激活与损失反向传播](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md) | [03 损失对齐与显存账本](./03_loss_alignment_memory_ledger.md) |
| Task4 | checkpointing 与 offload | [Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md) → [Part 02 · 42 激活 Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.md) | [04 Checkpointing 与 Offload](./04_checkpointing_and_offload.md) |
| Task5 | 梯度累积、训练闭环与 profiling | [Part 02 · 12 梯度累积](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.md) → [Part 02 · 73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) → [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) | [05 梯度累积、决策与性能分析](./05_accumulation_decision_profiling.md) |

## 正文与跳转

先按上面的 `Task1-5` 走 notebook 主线；遇到“梯度到底怎么回去”“为什么 activation 要保存”“checkpointing 和 offload 本质差别是什么”时，再回来看对应的专题正文。想看汇总版就进 [反向传播与训练机制正文](./casebook.md)，想按连续故事线走一遍就进 [反向传播与训练机制深入阅读](./walkthrough.md)。工具层补充放在 [训练工具桥](./training_tooling_bridge.md)，图册补充放在 [06 视觉资产](./06_visual_assets.md)。

如果问题已经跨到别的专题：
[监督微调与训练工程](../fine_tuning_training/intro.md) 负责训练闭环与项目交付，[显存优化](../memory_performance_tuning/intro.md) 负责训练侧显存 trade-off，[性能分析](../profiling/intro.md) 负责证据链与热点定位。

## 环境与验证

计算图、loss 和小规模 backward 实验通常可用 CPU；长序列、checkpoint/offload 和真实训练性能比较建议使用 GPU。验证时应区分数值正确性、峰值显存和 step time，不能把单项指标改善直接当成整体优化结论。
