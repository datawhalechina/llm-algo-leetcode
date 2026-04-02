# 讨论题 1：大模型的数据格式与混合精度 (Data Types & Precision)

> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../02_PyTorch_Algorithms/16_Quantization_W8A16.ipynb`](../02_PyTorch_Algorithms/16_Quantization_W8A16.ipynb)
> [`../03_CUDA_and_Triton_Kernels/11_Triton_Quantization_Support.ipynb`](../03_CUDA_and_Triton_Kernels/11_Triton_Quantization_Support.ipynb)

**难度：** Easy | **标签：** `基础概念`, `混合精度` | **目标人群：**通用基础 (算法/Infra)

在计算任何大模型的显存或算力之前，我们必须彻底搞懂“数据”在 GPU 中到底长什么样。这是所有硬件推导和量化算法的基石。

---

## Q1：请写出 FP32, FP16, BF16, INT8 分别占用几个字节 (Bytes)？

<details>
<summary>点击展开查看解析</summary>

在计算机中，1 Byte (字节) = 8 bits (位)。
- **FP32 (单精度浮点数)**: 32 bits = **4 Bytes**
- **FP16 (半精度浮点数)**: 16 bits = **2 Bytes**
- **BF16 (BFloat16)**: 16 bits = **2 Bytes**
- **INT8 (8位整型)**: 8 bits = **1 Byte**
- **INT4 (4位整型)**: 4 bits = **0.5 Byte** (通常用于极度压缩的量化如 AWQ/GPTQ)

*做显存估算时，只需要把“模型参数量”乘以对应的“字节数”即可。比如 7B 参数的模型在 FP16 下纯权重占用 7B * 2 Bytes = 14 GB。*
</details>

---

## Q2：为什么大语言模型（LLM）的预训练目前几乎全部采用 BF16 而不是 FP16？

<details>
<summary>点击展开查看解析</summary>

这涉及浮点数在底层的位分布设计：一个浮点数由 符号位(Sign) + 指数位(Exponent) + 尾数/精度位(Mantissa/Fraction) 组成。

**1. FP16 的痛点：动态范围太窄 (容易溢出)**
- FP16 的结构：1位符号 + **5位指数** + 10位尾数。
- 5位指数意味着它能表示的最大数值只有 **65504**。
- 大模型在训练时，尤其是未经过归一化的梯度或者 Attention 的 logits，非常容易产生大于 65504 的数值，导致数值溢出变成 `NaN`，训练直接崩溃。为了防止溢出，FP16 训练必须引入繁琐的 Loss Scaling 技术。

**2. BF16 (Brain Float 16) 的优势：动态范围与 FP32 一致**
- BF16 的结构：1位符号 + **8位指数** + 7位尾数。
- 它实际上就是直接把 FP32 (8位指数) 砍掉了后面的 16 位尾数得到的！
- 因此，BF16 能表示的数值范围和 FP32 一模一样（最大到 $3.4 \t\times 10^{38}$），**绝对不可能发生数值溢出**。
- 代价是尾数位从 10 降到了 7，牺牲了一点点小数的“精确度”。但神经网络非常鲁棒，对微小的精度损失不敏感，反而对“数值范围溢出”非常敏感。

*结论：BF16 提供了无脑、稳定、免调参的混合精度训练体验，是现代大模型（LLaMA/Qwen等）预训练和微调的绝对标配。*
</details>
