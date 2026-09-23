# 05. Quantization as a Memory Tool | 量化作为显存手段

## 页面目标

本节对应 **Task5：量化与显存容量扩展**。这里把量化放回显存预算中理解：先判断压缩的是权重、activation 还是 KV Cache，再判断模型是否因此装得下，以及质量和容量代价是否可接受。完整的 backend 部署、服务延迟和吞吐比较进入[推理优化 Task5](../inference_optimization/intro.md)。

## 核心机制

### 量化对象与处理时机

量化降低数值表示的存储或计算成本，但不同对象的处理时机和部署约束不同。

| 量化对象 | 典型时机 | 主要收益 | 主要风险 |
|:---|:---|:---|:---|
| 权重 | 部署前或加载前 | 模型更容易装入显存 | 精度、格式和 kernel 约束 |
| activation | 运行时计算 | 中间张量和带宽压力下降 | 数值稳定性、硬件支持 |
| KV Cache | 请求执行过程中 | 长上下文或高并发更容易装下 | 误差、cache dtype 和 backend 支持 |

学习顺序是 [21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) → [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) → [41 FP8 / KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md) → [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)。其中 `41` 既覆盖运行时量化，也覆盖 KV Cache 量化；本专题提取对象、字节数、误差和容量证据，完整的 backend 选择和服务指标回到推理优化 Task5。

GGUF 是文件格式和部署封装，需要匹配 llama.cpp 等 backend；不能把 GGUF 的启动方式和 GPTQ / AWQ 的 backend 直接混写。

![量化作为显存工具：对象、时机与证据](../../public/topic_discussion/memory_performance_tuning/quantization_memory_tool.svg)

### 字节收益、质量代价与容量扩展

三类量化对象对应不同的验证入口：

| 量化对象 | 先确认什么 | 主要验证入口 | 最终证据 |
|:---|:---|:---|:---|
| 权重 | 文件格式、加载方式和底座 dtype | `21 → 25 → 40` | `67` 的真实 artifact、加载状态和显存容量 |
| activation | 计算路径、硬件支持和数值稳定性 | `21`、混合精度与算子支撑 | GPU 前向 / 训练吞吐和质量 |
| KV Cache | 请求阶段、cache dtype 和 backend 支持 | `41`，并与 Task4 的 Cache 机制衔接 | 长上下文、并发、显存和任务质量 |

在参数量、token 数量和其他运行条件不变时，可以先用每元素字节数估算容量变化。下表只表达存储账本，不代表对应硬件一定支持该计算路径，也不代表端到端延迟会按比例下降。

| 表示方式 | 每元素字节数 | 相对 FP16 的理论存储量 | 适合先观察的对象 |
|:---|---:|---:|:---|
| FP16 / BF16 | 2 | 100% | baseline 权重、activation 或 KV Cache |
| INT8 | 1 | 50% | 权重或受支持的运行时表示 |
| INT4 | 0.5 | 25% | 权重容量与加载边界 |
| FP8 | 1 | 50% | 权重、activation 或 KV Cache；需检查硬件和 backend |

GGUF、GPTQ、AWQ 和 FP8 不是同一层面的名称：前者可能描述文件格式和部署封装，后者可能描述量化算法或计算表示。实验报告应分别记录格式、算法、backend 和 kernel，不能只写“INT4 / FP8”。

## 判断与验证

先按“对象 → 时机 → backend → 指标”做判断：

1. 如果模型首先装不下，先检查权重表示和加载格式。
2. 如果长上下文或并发首先装不下，再检查 KV Cache 表示。
3. 如果理论压缩有效，仍要确认真实模型能加载、量化对象确实生效，并和 FP16 / BF16 baseline 使用同一 workload 对比容量。
4. 最终至少记录峰值显存、可接纳的上下文或并发、质量、格式和加载状态；完整延迟、吞吐和 backend 选择进入推理优化 Task5，再输出 `accept / tune / reject`。

量化实验的最小对照应包含：同一模型和 revision、同一请求或评测集、同一生成长度、同一 batch / concurrency，以及 FP16 或 BF16 baseline。只有量化模型成功加载并完成同口径对照，才能讨论部署收益。

CPU 可以验证量化误差、理论字节数、输出 shape 和预算逻辑；GPU backend 才能确认格式加载、实际显存、可接纳容量和任务质量。理论压缩率不等于端到端收益；kernel、延迟、吞吐和服务质量的完整归因转到推理优化 Task5 或量化专题。
