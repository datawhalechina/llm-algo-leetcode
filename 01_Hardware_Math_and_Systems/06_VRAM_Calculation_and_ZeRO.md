# 讨论题 06：大模型显存占用 (VRAM Calculation) 计算指南

**难度：** Hard | **标签：** `算力评估`, `ZeRO` | **目标人群：** 模型微调与工程部署

> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../02_PyTorch_Algorithms/17_Gradient_Checkpointing.ipynb`](../02_PyTorch_Algorithms/17_Gradient_Checkpointing.ipynb)
> [`../02_PyTorch_Algorithms/19_ZeRO_1_Optimizer_Sim.ipynb`](../02_PyTorch_Algorithms/19_ZeRO_1_Optimizer_Sim.ipynb)

> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../02_PyTorch_Algorithms/17_Gradient_Checkpointing.ipynb`](../02_PyTorch_Algorithms/17_Gradient_Checkpointing.ipynb)
> [`../02_PyTorch_Algorithms/19_ZeRO_1_Optimizer_Sim.ipynb`](../02_PyTorch_Algorithms/19_ZeRO_1_Optimizer_Sim.ipynb)

在真实的工业界和面试中，最常被问到的问题之一就是：“给定一个 7B 的模型，我要用多少张 A100 才能把它跑起来？”
这个问题不仅考察你对混合精度训练的理解，还考察你对 DeepSpeed ZeRO 优化器各个阶段（Stage 1/2/3）通信原理的掌握。

---

## 题目描述 (The Question)

假设我们要对一个 **7B (70 亿参数)** 的 LLaMA 模型进行**全参微调 (Full Fine-Tuning)**。
环境配置：
- 数据类型：权重和激活使用 `BF16` (16-bit Float)，优化器状态使用 `FP32`。
- 优化器：AdamW。
- 暂不考虑：激活值 (Activations) 的显存占用（假设我们用了非常激进的 Gradient Checkpointing）以及 KV Cache 的占用。

**问题 1：在不使用任何显存优化（如 ZeRO）的情况下，单卡至少需要多少 GB 的显存才能跑通训练？**
**问题 2：如果使用 DeepSpeed ZeRO-1，显存占用会降到多少（假设我们有 8 张卡 DP=8）？**
**问题 3：如果使用 DeepSpeed ZeRO-3（DP=8），理论上的每张卡的显存占用下限是多少？**

---

## 核心推导公式 (The Formula)

在混合精度训练中，每个参数需要多少个字节（Bytes）？
1. **模型权重 (Model Weights)**: 2 bytes (BF16)
2. **梯度 (Gradients)**: 2 bytes (BF16)
3. **优化器状态 (Optimizer States)**:
   - AdamW 需要保存 FP32 的权重副本 (Master Weights): 4 bytes
   - 动量 (Momentum/m): 4 bytes
   - 方差 (Variance/v): 4 bytes
   - *总计优化器状态占用*: $4 + 4 + 4 = 12$ bytes/参数

**总公式：一个参数在标准混合精度训练中占用 $2 + 2 + 12 = 16$ bytes 的显存。**

---

<details>
<summary>点击展开查看详细解答 (Solutions)</summary>

### 答案 1：不使用 ZeRO (单卡理论值)
根据上述公式，每个参数占用 16 字节。
7B = $7 \times 10^9$ 个参数。
总显存 = $7 \times 10^9 \times 16 \text{ bytes}$
$$= 112 \times 10^9 \text{ bytes} \approx 112 \text{ GB}$$

*结论：单卡 80GB 的 A100 是绝对跑不起来 7B 模型的全参微调的，必然 OOM（Out of Memory）。这也是为什么必须引入分布式并行。*

### 答案 2：使用 ZeRO-1 (切分优化器状态，8卡 DP)
ZeRO-1 将占用最大头（12 bytes/参数）的**优化器状态 (Optimizer States)** 平分到 $N$ 张卡上。
权重和梯度依然在每张卡上保留全量备份。

- **每卡权重**: 2 bytes $\times 7\text{B} = 14 \text{ GB}$
- **每卡梯度**: 2 bytes $\times 7\text{B} = 14 \text{ GB}$
- **每卡优化器状态**: $\frac{12}{8} \text{ bytes} \times 7\text{B} = 1.5 \text{ bytes} \times 7\text{B} = 10.5 \text{ GB}$

**单卡显存总计 = $14 + 14 + 10.5 = 38.5 \text{ GB}$**
*结论：使用 ZeRO-1 后，8 张 40GB/80GB 的 A100 都可以轻松跑起来（如果不算激活值的话）。*

### 答案 3：使用 ZeRO-3 (切分所有状态，8卡 DP)
ZeRO-3 是终极的显存优化方案，它将**优化器状态、梯度、以及模型权重**全部平分到 $N$ 张卡上。计算前向或反向时，通过网络 (All-Gather) 临时拉取所需的参数。

- **单卡总显存 (理论下限)** = $\frac{16 \text{ bytes}}{N} \times \text{参数量}$
- 在 $N=8$ 的情况下：$\frac{16}{8} \times 7\text{B} = 2 \text{ bytes} \times 7\text{B} = 14 \text{ GB}$

*结论：理论上每张卡只需要 14GB 的显存存放切片。但在真实工程中，ZeRO-3 需要维护通信缓冲区 (Communication Buffers)，这通常会带来几十 GB 的额外开销。因此实际显存占用远大于 14GB。*
</details>

