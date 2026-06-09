# 02. LLM Params and FLOPs | 大模型参数量与算力推导 (LLM Params & FLOPs Math)

**难度：** Medium | **标签：** `数学推导`, `Transformer` | **目标人群：** 通用基础 (算法/Infra)

面试常考题：以经典的 LLaMA-7B 模型为例，深度剖析大模型底层参数量的具体分布，并一步步推导前向推理与完整训练所需的真实算力（FLOPs）需求。

> **相关阅读**:  
> 本章对应的练习资产：  
> [`02_LLM_Params_and_FLOPs_Practice.ipynb`](./02_LLM_Params_and_FLOPs_Practice.ipynb)  
> [`02_LLM_Params_and_FLOPs_Practice.md`](./02_LLM_Params_and_FLOPs_Practice.md)  

---

## Q1：假设隐藏层维度为 $d$，词表大小为 $V$。请推导一个包含 $L$ 层的标准 Transformer Decoder 的总参数量。

<details>
<summary>点击展开查看解析</summary>

我们把 Transformer 拆解为三大部分（忽略极小的 bias 和 LayerNorm 的权重，它们对百亿参数的占比不到千分之一）：

**1. 嵌入层 (Embedding Layer) 与 输出层 (LM Head)**
- Token Embedding: 形状 $[V, d]$，参数量为 $V \times d$。
- LM Head (输出映射): 形状 $[d, V]$，参数量为 $V \times d$。
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
*(如果在 LLaMA 中使用 SwiGLU，维度会变为 $\frac{8}{3}d$，但有 3 个矩阵，总参数量依然是 $3 \times \frac{8}{3}d^2 = 8d^2$)*。

**综上所述：**
- 一个 Block 的参数量 = $4d^2$ (Attn) + $8d^2$ (MLP) = **$12d^2$**。
- 总参数量 $\approx 2Vd + L \times 12d^2$。

*带入 LLaMA-7B 感受一下：$d=4096, L=32, V=32000$*
*Block 参数 = $32 \times 12 \times 4096^2 \approx 6.4 \text{ Billion}$*
*Embedding = $2 \times 32000 \times 4096 \approx 0.26 \text{ Billion}$*
*总计约 6.7B，也就是所谓的 7B 模型！*
</details>

---

## Q2：前向传播 (Inference / Forward Pass) 的 FLOPs 是怎么计算的？

<details>
<summary>点击展开查看解析</summary>

在了解了参数量之后，我们来看大模型在进行推理（前向传播）时需要多少算力。

**核心经验法则：1 个参数处理 1 个 Token，大约需要 2 次浮点运算（FLOPs）。**
为什么是 2 次？因为在矩阵乘法 $Y = W \times X$ 中，对于每一个权重元素，我们需要做一次**乘法**和一次**加法**（Multiply-Accumulate, MAC）。

**推理 FLOPs 公式：**
$$ C_{forward} \approx 2 \times P \times T $$
其中：
- $C_{forward}$ 是前向传播需要的计算量（FLOPs）
- $P$ 是模型的总参数量（Parameters）
- $T$ 是处理的 Token 数量（Tokens）

*(注：这里忽略了少量的 Attention 矩阵乘积算力等，因为在大模型中，线性层的矩阵乘法占了绝对大头，通常占 99% 以上)*
</details>

---

## Q3：训练 (Training) 时包含前向和反向传播，总 FLOPs 是多少？

<details>
<summary>点击展开查看解析</summary>

训练不仅包含前向传播计算损失，还包含反向传播计算梯度。

在反向传播中，我们需要：
1. 计算激活值（Activations）的梯度，以便将误差继续向后传（大约需要 $2 \times P \times T$ FLOPs）。
2. 计算权重（Weights）的梯度，用于模型参数更新（大约也需要 $2 \times P \times T$ FLOPs）。

因此，反向传播的计算量大约是前向传播的 **2 倍**。

**训练 FLOPs 公式：**
$$ C_{train} = C_{forward} + C_{backward} \approx 2PT + 4PT = 6 \times P \times T $$

**实战估算：**
假设我们要从头预训练一个 LLaMA-7B（70亿参数）模型，训练数据量是 1T（1万亿）个 Tokens。
需要的总理论算力 $C = 6 \times (7 \times 10^9) \times (1 \times 10^{12}) = 4.2 \times 10^{22}$ FLOPs。

如果你手里有 1000 张 A100 (每张卡假设实际算力能跑出 150 TFLOPs，即 $1.5 \times 10^{14}$ FLOPs/s)，那么训练耗时：
$$ \text{Time} = \frac{4.2 \times 10^{22}}{1000 \times 1.5 \times 10^{14}} = 2.8 \times 10^5 \text{ 秒} \approx 3.2 \text{ 天} $$
</details>

---

## Q4：训练大模型时，什么是算力利用率 (MFU, Model FLOPs Utilization)？

<details>
<summary>点击展开查看解析</summary>

通过前面的 Q3 我们算出了**理论所需算力**。但在实际工程中，硬件不可能 100% 把所有时间都花在矩阵乘法上。这就引入了 MFU，它是衡量大模型训练工程质量的最核心指标。

- **理论算力 (Peak FLOPs)**：显卡说明书上写的算力。比如 A100 BF16 理论峰值是 312 TFLOPs（每秒执行 312 万亿次浮点运算）。
- **实际算力 (Observed FLOPs)**：即我们用 $6PT$ 算出的整个训练所需的理论运算量，除以跑完这些步骤所花的**实际时间**。
- **MFU = 实际算力 / 理论算力**。

**为什么 MFU 很难达到 100%？**
因为在真正的训练集群中，存在 **Memory-bound (显存墙)** 和 **Communication (通信瓶颈)**。GPU 很多时间在等待数据从内存搬运过来，或者在等其他机器的 All-Reduce 数据传过来，并没有在做有效的乘加运算。

目前顶级的工业界预训练集群，MFU 通常在 **40% 到 60%** 之间。如果你微调时的 MFU 只有 10%，说明你的代码里存在严重的通信或 IO 阻塞（比如没开梯度累加，或者数据读取成了瓶颈）。
</details>
