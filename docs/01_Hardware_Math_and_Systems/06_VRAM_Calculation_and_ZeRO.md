# 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化 (VRAM Calculation & ZeRO)

**难度：** Hard | **标签：** `算力评估`, `ZeRO` | **目标人群：** 模型微调与工程部署

在工业界和算法工程师的面试中，评估大模型训练所需的显存资源是一项核心基本功。
这不仅考察对混合精度训练底层机制的理解，还深度考察对 DeepSpeed ZeRO 优化器各阶段（Stage 1/2/3）分布式切分原理的掌握。

> **相关阅读**:  
> 本章对应的练习资产：  
> [`06_VRAM_Calculation_and_ZeRO_Practice.ipynb`](./06_VRAM_Calculation_and_ZeRO_Practice.ipynb)  
> [`06_VRAM_Calculation_and_ZeRO_Practice.md`](./06_VRAM_Calculation_and_ZeRO_Practice.md)  

---

## Q1：在采用 AdamW 优化器的标准混合精度训练中，每个模型参数在静态状态下占用多少显存？

<details>
<summary>点击展开查看解析</summary>

在主流的大模型混合精度训练（如 BF16 权重 + FP32 优化器状态）中，显存占用主要由三部分构成：

1. **模型权重 (Model Weights)**: 使用 BF16 或 FP16 存储，每个参数占用 **2 bytes**。
2. **梯度 (Gradients)**: 同样使用 BF16 存储，用于参数更新前的信息累加，每个参数占用 **2 bytes**。
3. **优化器状态 (Optimizer States)**: 
   为了保证极小学习率下的参数更新不发生下溢出，AdamW 必须在 FP32 精度下维护三组数据：
   - FP32 的权重高精度副本 (Master Weights): 4 bytes
   - 第一阶动量 (Momentum / m): 4 bytes
   - 第二阶动量 (Variance / v): 4 bytes
   - *总计优化器状态占用*: 4 + 4 + 4 =  **12 bytes**。

**核心结论：在未切分状态下，一个参数的静态显存占用约为 2 + 2 + 12 = 16 bytes。**
</details>

---

## Q2：基于 Q1 的结论，为什么单张 80GB 显存的 A100 无法完成 7B 模型（70亿参数）的全参数微调？

<details>
<summary>点击展开查看解析</summary>

我们可以通过静态显存的理论计算来评估单卡的承载能力：

- 7B 模型拥有 7 	imes 10^9 个参数。
- 根据 Q1 的公式，每个参数占用 16 字节。
- **总静态显存占用** = 7 	imes 10^9 	imes 16 	ext{ bytes} pprox 112 	ext{ GB}。

**结论**：
仅仅是存放模型自身的训练状态（权重、梯度、优化器状态），就已经需要 112 GB 的显存。这还不包括前向传播中产生的激活值 (Activations) 缓存，以及深度学习框架运行时的上下文开销。因此，单张 80GB 的 A100 必然会发生 OOM (Out of Memory)，必须引入 ZeRO 等分布式并行优化策略。
</details>

---

## Q3：DeepSpeed ZeRO-1 是如何通过状态切分解决单卡显存不足问题的？（以单机 8 卡为例）

<details>
<summary>点击展开查看解析</summary>

ZeRO (Zero Redundancy Optimizer) 的核心思想是消除数据并行 (Data Parallelism) 中各节点对模型状态的冗余存储。

**ZeRO-1 的机制**：
- 它选择对显存占用最大、但在前反向计算中不需要参与全量矩阵乘法的**优化器状态 (Optimizer States)** 进行切分。
- 模型权重和梯度依然在每张卡上保留完整备份。

**理论显存计算 (假设 DP=8)**：
- **每卡权重**: 2 bytes 	imes 7	ext{B} = 14 	ext{ GB}
- **每卡梯度**: 2 bytes 	imes 7	ext{B} = 14 	ext{ GB}
- **每卡优化器状态**: rac{12}{8} 	ext{ bytes} 	imes 7	ext{B} = 1.5 	ext{ bytes} 	imes 7	ext{B} = 10.5 	ext{ GB}

**单卡静态显存总计 = 14 + 14 + 10.5 = 38.5 	ext{ GB}**。

**结论**：
通过 ZeRO-1 的优化，原本 112 GB 的占用被大幅缩减。8 张 40GB 或 80GB 显存的 A100 均可满足 7B 模型全参微调的基础参数驻留需求（配合 Gradient Checkpointing 压缩激活值后即可顺畅训练）。
</details>

---

## Q4：ZeRO-3 的极致切分策略是如何工作的？理论上单卡显存下限是多少？

<details>
<summary>点击展开查看解析</summary>

如果说 ZeRO-1/2 只是切分了优化器和梯度，那么 ZeRO-3 则是将“无冗余”做到了极致。

**ZeRO-3 的机制**：
- 它将**优化器状态、梯度、以及模型权重**全方位地切分并分布到 N 张卡上。
- **通信换显存**：在计算前向或反向传播时，当前计算层如果需要完整的权重，当前卡会通过网络 (All-Gather) 临时从其他卡拉取所需的参数切片。计算一旦完成，立即释放该高精度副本，显存回落。

**理论显存下限 (假设 DP=8)**：
- **单卡总参数显存** = rac{16 	ext{ bytes}}{N} 	imes 	ext{参数量}
- 在 N=8 的情况下：rac{16}{8} 	imes 7	ext{B} = 2 	ext{ bytes} 	imes 7	ext{B} = 14 	ext{ GB}。

**工程考量**：
虽然理论上每张卡只需要 14 GB 的显存，但在真实工程环境中，ZeRO-3 为了维持高效的网络传输，必须预留并维护庞大的通信缓冲区 (Communication Buffers / Fetch Buffers)。因此，实际的峰值显存占用会显著高于理论下限，并带来极大的机内通信带宽压力。
</details>


---

## Q5：在真实微调中，除了模型静态状态，激活值 (Activations) 也会占用海量显存。工业界是如何通过 FlashAttention-2 和 Gradient Checkpointing 解决这个问题的？

<details>
<summary>点击展开查看解析</summary>

在前面的计算中我们暂时忽略了激活值。实际上，如果使用原生的 PyTorch 实现，由于需要保存前向传播的中间结果以供反向传播计算梯度，激活值的显存占用会随着序列长度 (Sequence Length) N 的增长呈 O(N^2) 级数爆炸。

目前工业界在 A100/H100 服务器上的标准解法是“双管齐下”：

1. **FlashAttention-2 (算子层访存优化)**：
   - 原生 Attention 在计算时会在 HBM (全局显存) 中实例化一个庞大的 N * N 注意力分数矩阵，这是激活值显存溢出的罪魁祸首。
   - FlashAttention-2 充分利用了 A100 较大的 SRAM (共享内存)，通过分块计算 (Tiling) 和在线 Softmax (Online Softmax) 技术，在 SRAM 内部直接完成所有计算并输出最终结果，**避免了向 HBM 写入和读取 O(N^2) 的中间激活矩阵**。这不仅极大提升了运行速度，还从根本上削减了最大的激活值显存开销。

2. **Gradient Checkpointing (框架层重算优化)**：
   - 即“激活重算”机制。它不再于前向传播中保存所有层的激活值，而是仅保存少数几个关键层作为“检查点 (Checkpoints)”。
   - 在反向传播过程中，如果需要使用未保存的激活值，框架会从最近的检查点**重新进行一次前向计算**以实时恢复该值。这是一种经典的“以计算换显存”策略，通常能将剩余的激活值显存占用从 O(L) (L 为网络层数) 降低到 O(sqrt(L))。

**工程总结**：ZeRO 解决了**模型参数与优化器状态**的分布式存储问题，而 FlashAttention-2 配合 Gradient Checkpointing 则解决了**动态激活值**的显存爆炸问题。三者紧密结合，构成了现代大模型全参数微调和超长文本训练的底层系统基石。
</details>
