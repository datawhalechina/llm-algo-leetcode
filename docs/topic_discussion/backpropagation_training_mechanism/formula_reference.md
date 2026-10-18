# 反向传播与训练机制公式索引与符号约定

本页集中说明 2.3 与本专题共用的梯度、有效更新、状态生命周期和策略代价公式。每个正文页继续解释公式来源、成立条件和机制关系，Notebook 负责把公式映射到代码、TODO、测试与实验记录。

## 使用方法

阅读公式时依次确认四件事：梯度相对于哪个目标求导；损失按样本、Token 还是累计窗口归约；参数是否真的完成一次更新；状态节省是否引入重算、搬运或等待。跨页对照以本页符号为准，局部推导与实现仍回到对应正文和机制 Notebook。

## 共用符号

| 符号 | 含义 | 常见粒度 |
|:---|:---|:---|
| $L$ | 当前训练目标或损失 | 样本、Token 或累计窗口 |
| $x,y$ | 算子输入与输出；具体页面可继续写成 Q、K、V、logits 等对象 | 张量 |
| $\bar y=\partial L/\partial y$ | 从输出接收到的上游梯度 | 张量 |
| $J_f(x)$ | 函数 $f$ 在 $x$ 处的雅可比矩阵 | 局部算子 |
| $g$ / $g_u$ | 参数梯度；$g_u$ 表示第 $u$ 次有效更新使用的梯度 | 参数或参数组 |
| $m_{k,i}$ | 第 $k$ 个微批次中第 $i$ 个位置的有效监督掩码，取值为 0 或 1 | Token |
| $N$ | 当前累计窗口内的有效监督位置总数 | 累计窗口 |
| $s$ | 混合精度损失缩放因子 | 更新步 |
| $\theta_u$ | 第 $u$ 次有效更新前的模型参数 | 参数 |
| $\eta_u$ | 第 $u$ 次有效更新使用的学习率 | 更新步 |
| $\mathcal U(g_u)$ | 优化器根据梯度和内部状态形成的更新方向 | 参数 |
| $b_i$ | 第 $i$ 类训练状态的字节数 | 状态对象 |
| $a_i(t)$ | 状态 $i$ 在阶段 $t$ 是否驻留，驻留为 1，否则为 0 | 训练阶段 |
| $B_{out},B_{in}$ | 搬出与搬回方向的有效带宽 | 传输链路 |
| $T_{overlap}$ | 可由其他计算覆盖的传输时间 | 关键路径 |

## 局部反向与梯度汇合

| 机制 | 规范表达 | 成立条件与观察量 | 正文与代码入口 |
|:---|:---|:---|:---|
| 向量—雅可比积 | $\bar x=J_f(x)^\mathsf{T}\bar y$ | $y=f(x)$；检查输出梯度、输入梯度和形状 | [02 自动求导与 Transformer 反向](./02_autograd_and_transformer_backward.md)、[Part 00 · 07](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md) |
| 分支梯度汇合 | $y=f(x)+h(x)$，$\bar x=J_f(x)^\mathsf{T}\bar y+J_h(x)^\mathsf{T}\bar y$ | 同一输入或共享参数从所有使用位置累计贡献 | [02 自动求导与 Transformer 反向](./02_autograd_and_transformer_backward.md)、[Part 02 · 09](../../02_PyTorch_Algorithms/09_Activation_and_Loss_Backward.md) |
| Attention 前向 | $S=QK^\mathsf{T}/\sqrt d$，$P=\operatorname{softmax}(S)$，$O=PV$ | $d$ 为每个 Attention Head 的维度；mask 在 Softmax 前作用于 $S$ | [Part 02 · 10 Attention 反向](../../02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.md) |
| Attention 反向 | $dV=P^\mathsf{T}dO$，$dP=dOV^\mathsf{T}$<br>$dS=P\odot[dP-\operatorname{rowsum}(dP\odot P)]$<br>$dQ=dSK/\sqrt d$，$dK=dS^\mathsf{T}Q/\sqrt d$ | 每行 Softmax 独立求导；检查 mask、缩放、`dQ/dK/dV` 与 `gradcheck` | [02 自动求导与 Transformer 反向](./02_autograd_and_transformer_backward.md)、[Part 02 · 10](../../02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.md) |

## 监督归约与有效更新

| 机制 | 规范表达 | 粒度与更新条件 | 正文与代码入口 |
|:---|:---|:---|:---|
| 有效 Token 损失 | $L=\dfrac{1}{N}\sum_{k,i}m_{k,i}\ell_{k,i}$，$N=\sum_{k,i}m_{k,i}$ | 先累计损失分子和有效 Token 数，再做一次统一归约 | [03 监督与有效更新](./03_supervision_gradient_scale_effective_update.md)、[Part 02 · 13 SFT](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md) |
| 累计窗口梯度 | $g=\nabla_\theta L=\dfrac{1}{N}\sum_{k,i}m_{k,i}\nabla_\theta\ell_{k,i}$ | 不同长度微批次使用同一有效 Token 分母；窗口结束前不写回参数 | [Part 02 · 11 梯度累积](../../02_PyTorch_Algorithms/11_Gradient_Accumulation.md) |
| 损失缩放与还原 | $\widetilde L=sL$，$\widetilde g=s g$，$g=\widetilde g/s$ | FP16 常用动态 $s$；有限值检查和裁剪作用于还原后的真实梯度 | [Part 02 · 混合精度反向](../../02_PyTorch_Algorithms/TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward.md) |
| 有效参数更新 | $\theta_{u+1}=\theta_u-\eta_u\mathcal U(g_u)$，条件为 $\operatorname{finite}(g_u)=1$ | 梯度非有限时参数和 scheduler 均不推进；$u$ 只统计成功更新 | [Part 02 · 12 优化器与调度](../../02_PyTorch_Algorithms/12_Optimizer_Updates_and_Learning_Rate_Scheduling.md)、[Part 02 · 55 更新动力学](../../02_PyTorch_Algorithms/55_Training_Update_Dynamics.md) |

## 训练状态生命周期与峰值

| 机制 | 规范表达 | 粒度与观察量 | 正文与代码入口 |
|:---|:---|:---|:---|
| 阶段驻留量 | $M_t=\sum_i a_i(t)b_i$ | $a_i(t)$ 由创建、最后消费和释放时刻决定 | [04 训练状态生命周期](./04_training_state_lifecycle.md)、[Part 02 · 43 内存账本](../../02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger.md) |
| 理论峰值 | $M_{peak}=\max_t M_t$ | 与 CUDA allocated/reserved 分开记录；理论账本用于解释主导对象 | [04 训练状态生命周期](./04_training_state_lifecycle.md)、[Part 00 · 18 显存账本](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.md) |
| Checkpoint 节省量 | $\Delta M_{ckpt}=M_{saved,base}-M_{saved,ckpt}$ | 基线和候选使用同一模型、输入、累计窗口和测量边界 | [05 激活检查点与卸载](./05_checkpointing_and_offload.md)、[Part 02 · 19](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.md) |
| Checkpoint 新增时间 | $\Delta T_{ckpt}=T_{ckpt}-T_{base}$ | 新增时间主要来自反向重算和调度开销；同时检查输出与梯度一致 | [Part 02 · 19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing.md) |

## Offload 搬运与暴露时间

| 机制 | 规范表达 | 粒度与观察量 | 正文与代码入口 |
|:---|:---|:---|:---|
| 理论往返时间 | $T_{roundtrip}\approx bytes/B_{out}+bytes/B_{in}$ | `bytes` 是实际搬出并在反向前搬回的状态量 | [05 激活检查点与卸载](./05_checkpointing_and_offload.md)、[Part 02 · 42](../../02_PyTorch_Algorithms/42_Activation_Offload.md) |
| 暴露传输时间 | $T_{exposed}=\max(0,T_{roundtrip}-T_{overlap})$ | 只把未被计算覆盖的搬运计入关键路径；搬回必须早于消费点 | [Part 02 · 42 激活卸载](../../02_PyTorch_Algorithms/42_Activation_Offload.md)、[Part 02 · 76 策略对照](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) |

## 与其他专题的公式边界

| 公式对象 | 归属位置 |
|:---|:---|
| SFT、DPO、GRPO、PPO、奖励与优势 | [后训练优化公式索引](../post_training_optimization/formula_reference.md) |
| All-Reduce、Reduce-Scatter、并行切分与扩展效率 | [通信与并行](../communication_parallelism/intro.md) |
| 吞吐、延迟、分位数、Roofline 与回归阈值 | [性能分析](../performance_analysis/intro.md) |
| 量化范围、缩放、校准误差与低比特表示 | [量化与压缩](../quantization_compression/intro.md) |
