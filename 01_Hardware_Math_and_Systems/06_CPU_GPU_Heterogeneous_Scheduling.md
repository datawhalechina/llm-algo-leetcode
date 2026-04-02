# 讨论题 06：CPU-GPU 异构协同与卸载 (Heterogeneous Co-Design & Offloading)
> **相关阅读**:
> 请前往实战篇进行相关代码练习：
> [`../03_CUDA_and_Triton_Kernels/15_PyTorch_CUDA_Streams_and_Transfer.ipynb`](../03_CUDA_and_Triton_Kernels/15_PyTorch_CUDA_Streams_and_Transfer.ipynb)

**难度：** Hard | **标签：** `系统架构`, `异构计算`, `DeepSpeed` | **目标人群：** 核心 Infra 与算子开发

当 GPU 显存（如 80GB）不仅塞不下庞大的模型参数，甚至连切分后的状态都塞不下时，我们不得不向廉价但海量的 CPU 内存借空间。

---

## Q1：在 LLM 训练和推理中，CPU Offload (CPU 卸载) 是什么技术？请举例说明。

<details>
<summary>点击展开查看解析</summary>

**CPU 只是指挥官，GPU 才是干活的。** 当 GPU 干活的空间（显存）不够时，指挥官会把数据存在自己宽敞的办公室（系统内存/Host RAM）里。

- **训练期的 ZeRO-Offload**：将占用大规模显存的**优化器状态**（如 Adam 的动量和方差）甚至梯度，踢到 CPU 内存里去。反向传播算完梯度后，传给 CPU；CPU 利用自身的算力更新参数，然后再把更新后的权重传回 GPU。虽然 PCIe 传输很慢，但这拯救了单卡 OOM 的命运。
- **推理期的 KV Cache Offload (如 vLLM)**：在处理超长文本或极高并发时，GPU 显存装不下所有的 KV Cache。推理引擎会将暂时不活跃的 Token 的 KV Cache 踢到 CPU 内存里（换出 Evict），等需要用到时再拉回 GPU 显存（换入 Swap-in）。这就是典型的操作系统**虚拟内存分页机制**在异构架构上的重演。
</details>

---

## Q2：跨设备传输（PCIe）这么慢，有什么技术可以重叠 (Overlap) 通信和计算的时间？

<details>
<summary>点击展开查看解析</summary>

大厂底层优化团队的必杀技：**Overlap Computation and Communication (计算与通信重叠)**。

如何隐藏通信时间？利用 CUDA 的 **异步流 (Asynchronous Streams)**。
我们可以把任务切分成多个微块（Micro-batches）：
- 当 **Stream 1** 在 GPU 上执行块 $A$ 的矩阵乘法计算时，
- **Stream 2** 可以同时在后台把块 $B$ 的数据通过 DMA (Direct Memory Access，不经过 CPU 控制) 从 CPU 内存拷贝到显存！
- 当块 $A$ 算完时，块 $B$ 的数据刚好传输完毕，GPU 无缝衔接继续算块 $B$。

优秀的 AI 框架（如 Megatron 或 FSDP）在底层都非常精妙地实现了通信与计算的 Overlap，从而让算力利用率逼近理论极限。
</details>
