# 05. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化

## 页面目标

本节回答两个相互关联但不能混为一谈的问题：什么时候应该压执行路径或 KV Cache，而不是继续压权重；硬件、backend 和上下文长度如何改变这项选择。

输出是推理侧候选：FP8 或 KV Cache 量化能释放什么资源，是否会改变 kernel、缓存容量、延迟或质量边界。

## 问题起点

这两条路线容易被误解成“又一种更低比特”。但它们的意义并不在于位宽本身，而在于：

- FP8 更像硬件与执行栈驱动的低精度路径；
- KV cache quantization 更像推理预算驱动的缓存压缩路径。

它们都和传统 weight-only 路线不同。

## 你要先确认什么

- 当前瓶颈来自执行栈，还是来自 cache 预算。
- 硬件是否已经原生支持 FP8。
- 长上下文和并发是否已经把 KV cache 顶成第一约束。

## 核心矛盾

FP8 的核心矛盾是：更低精度的执行路径能带来更好的吞吐和存储效果，但要求硬件和 kernel 栈配合；KV cache quantization 的核心矛盾是：缓存压缩能扩大上下文和并发预算，但会影响表示精度和服务稳定性。

## 机制链

FP8 主要改变计算或数据搬运路径，需要 scale 管理、硬件指令和 kernel 共同支持；KV Cache 量化则改变请求生命周期中 K/V 状态的存储与读取方式。二者都可能减少字节数，但影响的时间段不同：前者贯穿算子执行，后者集中在 decode 阶段的 cache 读写。

| 路线 | 主要对象 | 关键变量 | 需要对齐的 workload |
|:---|:---|:---|:---|
| FP8 | 权重、激活或矩阵乘输入输出 | scaling、硬件能力、kernel、混合精度边界 | prefill / decode、输入长度、输出长度 |
| KV Cache 量化 | 每个请求的 K/V 状态 | cache dtype、量化粒度、更新与反量化位置 | 上下文长度、并发、prefix sharing、TPOT |

权重量化与 KV Cache 量化可以同时出现，但必须分别记录显存账本和质量影响，不能把两者的收益相加后直接作为部署结论。

## 演化路径

1. 先判断问题是在执行路径还是缓存预算。
2. 如果是执行栈和硬件支持，优先看 FP8。
3. 如果是长上下文和高并发预算，优先看 KV cache quantization。
4. 最后再回到推理和显存专题，看它们在请求链路和预算中的综合效果。

## 关键取舍

- FP8 更依赖硬件和 backend 的成熟度。
- KV cache quantization 更依赖 workload 的上下文长度和并发特征。
- 二者都不能只看理论压缩率，必须回到服务目标和 benchmark。

## 证据边界

CPU 实验适合验证 FP8 数值范围、scale 计算和 KV Cache 字节数估算；真实 GPU / serving backend 才能验证硬件执行路径、cache 分配、TTFT、TPOT、并发容量和任务质量。`torch.cuda.is_bf16_supported()` 或格式字段本身不能替代 kernel 和 workload 实测。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- FP8 相关资料：理解低精度执行路径如何和硬件协同。
- KV cache quantization 资料：理解缓存压缩为什么首先是推理预算问题。

## 对应 Part 02

- `41` FP8 与 KV Cache 量化
- `67` 量化推理与部署

## 典型阅读入口

- [04 权重量化与后训练压缩](./04_weight_only_compression.md)
- [06 部署与 Benchmark 决策](./06_deployment_and_benchmark_decision.md)

## 本节要点

FP8 更接近执行路径与硬件支持问题，KV cache quant 更接近长上下文下的缓存预算问题；二者都不能只用模型文件大小判断价值。

## 进入下一页

把权重、执行路径和 KV cache 的候选放到 [06 部署与 Benchmark 决策](./06_deployment_and_benchmark_decision.md) 中，用同一 workload 做最终比较。

FP8 和 KV cache quantization 都是低精度路线，但一个更偏执行栈，一个更偏缓存预算。
