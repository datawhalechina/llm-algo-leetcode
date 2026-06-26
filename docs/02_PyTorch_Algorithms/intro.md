# 第二部分：PyTorch 算法实战

## 🎯 本部分概览

本部分聚焦 PyTorch 级别的大模型实现，按“基础算子 -> 模型架构 -> 微调与训练技术 -> 对齐技术 -> 反向传播与显存优化 -> 核心推理优化 -> 高级推理优化 -> 分布式与扩展”组织。

### 学习组划分

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **2.1: 基础算子** | 00-04 | Transformer 组件 | Easy-Medium |
| **2.2: 模型架构** | 05-08 | 模型组装 | Medium |
| **2.3: 微调与训练技术** | 09-11 | SFT / LoRA / 调度器 | Medium |
| **2.4: 对齐技术** | 12-13 | RLHF / DPO | Medium-Hard |
| **2.5: 反向传播与显存优化** | 14 | Autograd / Backward | Hard |
| **2.6: 核心推理优化** | 15-17 | FlashAttention / Decoding / PagedAttention | Hard |
| **2.7: 高级推理优化** | 18-20 | Speculative / Radix / Quantization | Hard |
| **2.8: 分布式与扩展** | 21-25 | Checkpointing / ZeRO / Parallelism | Hard |

### 零基础入门 6 Task

| Task | 覆盖小节 | 学习重点 | 预期收获 |
|:---|:---|:---|:---|
| Task 1 | 2.1 | PyTorch 入门与基础算子 | 熟悉张量、模块、前向与 RMSNorm 的基本写法 |
| Task 2 | 2.1 | 激活函数与位置编码 | 理解 SwiGLU 与 RoPE 的作用和实现方式 |
| Task 3 | 2.1 | Attention 核心实现 | 能串起 MHA / GQA，并理解 KV cache 的动机 |
| Task 4 | 2.2 | 模型块组装 | 能把基础算子组装成 LLaMA3 Block，并认识 MoE Router |
| Task 5 | 2.2 | MoE 与结构技巧 | 理解负载均衡损失和常见架构技巧 |
| Task 6 | 2.3 | 训练、微调与学习率策略 | 看懂 SFT、LoRA 和学习率调度的完整训练闭环 |

### 环境边界（代码审计版）

- **整体定位：CPU-first**
- **大多数 notebook**：可在 CPU 环境下完成学习和 correctness 验证
- **已确认需要 GPU 的 notebook**：`21_Gradient_Checkpointing`，其测试会读取真实 CUDA 显存峰值
- **学习建议**：为了保持体验一致，建议所有学习者使用同一套 Python 环境；GPU 作为后段实验和真实性能验证的增强条件，而不是第二部分的统一门槛

### 学习建议

- 新手先看 **2.1 -> 2.2 -> 2.3**
- 关注训练与对齐的同学看 **2.3 -> 2.4 -> 2.5**
- 关注推理与规模化的同学看 **2.6 -> 2.7 -> 2.8**

### 前置页面

- [1A 基础数学](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)
- [1B 硬件架构](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [1C 系统与编译](../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md)

### 后续页面

- [2.1 基础算子](../02_PyTorch_Algorithms/2_1.md)
- [2.4 对齐技术](../02_PyTorch_Algorithms/2_4.md)
- [2.5 反向传播与显存优化](../02_PyTorch_Algorithms/2_5.md)
- [2.6 核心推理优化](../02_PyTorch_Algorithms/2_6.md)
- [2.8 分布式与扩展](../02_PyTorch_Algorithms/2_8.md)
- [3.1 Triton 基础](../03_Triton_Kernels/intro.md)

### 题目与测试

部分题目的占位初始化、答案验证和本地测试方式，统一以 [维护与发布手册](../maintenance.md) 和 [使用指南](../guide.md) 为准。
