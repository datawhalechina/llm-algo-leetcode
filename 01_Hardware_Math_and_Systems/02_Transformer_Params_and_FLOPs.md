# 讨论题 2：Transformer 参数量与算力推导 (Params & FLOPs Math)

**难度：** Medium | **标签：** `数学推导`, `Transformer` | **目标人群：**通用基础 (算法/Infra)

面试官最喜欢问：“你平时用 7B 模型，那你能手算一下一个标准 Transformer Block 的参数量是怎么分布的吗？”

---

## Q1：假设隐藏层维度为 $d$，词表大小为 $V$。请推导一个包含 $L$ 层的标准 Transformer Decoder 的总参数量。

<details>
<summary>点击展开查看解析</summary>

我们把 Transformer 拆解为三大部分（忽略极小的 bias 和 LayerNorm 的权重，它们对百亿参数的占比不到千分之一）：

**1. 嵌入层 (Embedding Layer) 与 输出层 (LM Head)**
- Token Embedding: 形状 $[V, d]$，参数量为 $V \t\times d$。
- LM Head (输出映射): 形状 $[d, V]$，参数量为 $V \t\times d$。
- *(注：很多模型如 Gemma/Qwen 会共享这两个权重，参数量减半。这里我们假设不共享)*。

**2. 注意力机制 (Multi-Head Attention, MHA)**
在每个 Decoder 块中：
- 投影 Q, K, V：三个形状为 $[d, d]$ 的矩阵。参数量 $3d^2$。
- 投影 Output (O)：一个形状为 $[d, d]$ 的矩阵。参数量 $d^2$。
- **MHA 总参数量 = $4d^2$**。
*(如果采用 GQA，K 和 V 的参数量会大幅减少，这里按最原始的 MHA 计算)*。

**3. 前馈神经网络 (Feed Forward Network, FFN / MLP)**
在标准 GPT 架构中，隐藏层会先升维到 $4d$，再降维回 $d$：
- 升维矩阵 $W_{up}$：$[d, 4d]$，参数量 $4d^2$。
- 降维矩阵 $W_{down}$：$[4d, d]$，参数量 $4d^2$。
- **FFN 总参数量 = $8d^2$**。
*(如果在 LLaMA 中使用 SwiGLU，维度会变为 $\frac{8}{3}d$，但有 3 个矩阵，总参数量依然是 $3 \t\times \frac{8}{3}d^2 = 8d^2$)*。

**综上所述：**
- 一个 Block 的参数量 = $4d^2$ (Attn) + $8d^2$ (MLP) = **$12d^2$**。
- 总参数量 $\approx 2Vd + L \t\times 12d^2$。

*带入 LLaMA-7B 感受一下：$d=4096, L=32, V=32000$*
*Block 参数 = $32 \t\times 12 \t\times 4096^2 \approx 6.4 \text{ Billion}$*
*Embedding = $2 \t\times 32000 \t\times 4096 \approx 0.26 \text{ Billion}$*
*总计约 6.7B，也就是所谓的 7B 模型！*
</details>

---

## Q2：训练大模型时，什么是算力利用率 (MFU, Model FLOPs Utilization)？

<details>
<summary>点击展开查看解析</summary>

MFU 是衡量大模型训练工程质量的最核心指标。

- **理论算力 (Peak FLOPs)**：显卡说明书上写的算力。比如 A100 BF16 理论峰值是 312 TFLOPs（每秒执行 312 万亿次浮点运算）。
- **实际算力 (Observed FLOPs)**：在一次 Forward + Backward 中，你的模型真正需要做的浮点乘加运算次数，除以跑完这一步所花的时间。
- **MFU = 实际算力 / 理论算力**。

**为什么 MFU 很难达到 100%？**
因为前面提到的 **Memory-bound (显存墙)** 和 **Communication (通信瓶颈)**。GPU 很多时间在等待数据从内存搬运过来，或者在等其他机器的 All-Reduce 数据传过来。
目前顶级的工业界预训练集群，MFU 通常在 **40% 到 60%** 之间。如果你微调时的 MFU 只有 10%，说明你的代码里存在严重的通信或 IO 阻塞（比如没开梯度累加，或者数据读取成了瓶颈）。
</details>
