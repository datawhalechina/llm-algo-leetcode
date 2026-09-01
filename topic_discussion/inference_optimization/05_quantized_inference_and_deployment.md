# 05. Quantized Inference and Deployment | 量化推理与部署

## 页面目标

本节回答：权重、激活和 KV Cache 量化分别改变什么成本，以及什么时候值得切换。

## 本节在路线中的位置

本节对应 **Task5：量化推理与部署**。它承接 Task4 对 KV Cache、显存边界和服务并发的判断，把量化作为部署候选方案；完成后进入 67 量化部署项目，再由 66 将量化和其他推理策略放回同一 workload 综合比较。GPTQ / AWQ 是后训练权重量化方法，GGUF 是文件格式与部署封装；GGUF 需要对应的独立 backend，不能与 GPTQ/AWQ 共用一套启动参数。

本节不把量化理解成单一的“降 bit”：权重量化主要改变模型驻留和带宽成本，激活量化影响运行态计算路径，KV Cache 量化直接影响长上下文和并发边界。三者必须分别记录适用条件和质量代价。

## 问题起点

量化常常被误写成“显存不够时的默认答案”。但推理场景下，量化真正要回答的是更具体的问题：

- 我是被权重驻留卡住，还是被带宽卡住，还是被 KV cache 顶住？
- 我更敏感的是 TTFT、TPOT，还是 throughput / cost？
- 这个 backend 和 deployment 栈对低比特支持到什么程度？

只有先把约束说清楚，量化才是选型动作，而不是默认操作。

## 你要先确认什么

- 你要优化的是显存、带宽还是部署成本。
- 量化后 TTFT、TPOT、throughput 是否可接受。
- 线上服务是否更敏感延迟还是吞吐。

还要明确校准数据、量化粒度、目标 dtype、推理 backend、硬件架构和输出质量指标；如果这些条件没有固定，量化收益不能直接迁移到另一套服务环境。

## 核心矛盾

量化的核心矛盾是：低比特表示能省显存和带宽，但会引入精度误差、kernel 兼容性和部署复杂度。它从来不是“免费更快”，只能说是在某些 workload 下值得交换。

## 演化路径

量化不是一个统一动作，而是分别作用在不同部位。

1. 权重量化降低模型驻留成本。
2. 激活量化改变中间计算和带宽压力。
3. KV cache 量化直接影响长上下文和并发边界。
4. GPTQ / AWQ 更偏权重压缩。
5. FP8 / KV cache quant 更偏部署侧平衡。

## 关键取舍

- 权重量化更像“把模型装下或提高 batch 的第一步”。
- `GPTQ / AWQ` 更偏离线权重压缩与精度权衡。
- `FP8` 更偏端到端部署栈是否愿意为低精度继续优化 kernel。
- `KV cache quantization` 更像推理侧资源手段，尤其适合长上下文和高并发。

最终判断要回到服务目标：

- 在线交互更怕 TTFT / TPOT 退化；
- 离线批处理更愿意为 throughput / cost 接受一定延迟变化。

## 学习者交付物

完成本节后，至少应形成一份可部署性判断：

| 项目 | 最小内容 |
|:---|:---|
| 量化对象 | 权重、激活、KV Cache，或它们的组合 |
| 配置 | bit 数、粒度、校准数据、dtype、backend 和硬件 |
| 性能 | TTFT、TPOT、throughput、P99、peak memory |
| 质量 | 输出误差、任务指标、perplexity 或其他质量约束 |
| 部署结论 | accept / tune / reject，以及适用 workload |

核心结论应能够说明：显存下降是否真的转化为服务收益，收益来自权重驻留、带宽、KV Cache 容量还是 batch 提升。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- GPTQ：帮助理解离线权重量化如何用近似最优重建减少精度损失。
- AWQ：帮助理解按激活感知选择量化尺度的动机。
- FP8 / KV cache quantization 相关资料：帮助理解部署侧怎样平衡吞吐、显存和质量。

## 常见误区

- 看到 peak memory 降了就认为方案一定更好。
- 不分在线交互和离线批处理，直接比较量化收益。
- 把推理量化和训练量化混在一起看。

## 对应 Part 02

- `25` Quantization W8A16
- `40` GPTQ and AWQ Weight Quantization
- `41` FP8 and KV Cache Quantization
- `67` Quantized Inference and Deployment

## 证据边界

CPU 可以验证量化误差、字节数和预算计算；真实量化格式加载、kernel 适配、吞吐、显存和任务质量需要匹配 backend 与硬件的 GPU 实验。模型能够加载不等于量化收益已经成立。

## 经典阅读入口

- [21 Quantization Theory and INT4 INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb)
- [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)
- [40 GPTQ and AWQ Weight Quantization](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)
- [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)
- [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)

## 相关跳转

- 看 `01`，先统一指标口径。
- 看 `04`，确认 cache 是否已经是硬约束。
- 看 `06`，把量化和其他候选方案一起比较。

## 对应项目

- **核心主题项目：** [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)，验证量化模型的加载、性能、显存和质量约束。
- **综合项目：** [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)，在统一 workload 下比较量化与其他策略。

实践级别：没有真实 backend 时完成 Practice-P1 的本地模型加载和指标模板；接入 vLLM / SGLang 并确认低比特 kernel 支持后，才升级为 Practice-P2 量化部署实验。

## 本节要点

量化是推理优化的候选方案之一，不是默认答案；最终仍要回到 workload 和服务目标。
