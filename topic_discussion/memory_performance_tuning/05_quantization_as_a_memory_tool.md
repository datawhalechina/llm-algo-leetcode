# 05. Quantization as a Memory Tool | 量化作为显存手段

## 页面目标

本节回答的是：当量化被放进显存专题时，应该怎么看它，而不是把它当成独立的“更快”或“更先进”方案。

## 问题起点

量化在显存专题里的角色，不是先回答“能不能更快”，而是先回答：

- 它省的是权重驻留、activation 还是 KV cache；
- 它能不能把系统压进预算；
- 压进预算以后，时间和质量代价是否还能接受。

所以这里的量化，是“资源手段”，不是纯算法标签。

## 你要先确认什么

- 预算瓶颈来自权重、activation，还是 cache。
- 当前 workload 更怕 OOM，还是更怕延迟恶化。
- backend 是否已经支持对应低比特路线。

## 核心矛盾

量化通过降低数值表示成本来省显存和带宽，但它一定会同时引入精度、kernel、兼容性或部署复杂度上的代价。因此，这条线的核心不是“能不能量化”，而是“量化是否值得作为显存优化工具”。

## 量化对象与证据顺序

先区分权重、activation 和 KV Cache 三个对象，再决定使用哪一种量化路线：[21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) 和 [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) 适合建立基础；[40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) 进入后训练权重量化扩展，[41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) 进入运行时与 KV Cache 扩展；GGUF 作为文件格式与部署路径，最后统一用 [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) 按对应 backend 验证显存、速度和质量是否同时满足约束。

## 演化路径

1. 先识别预算对象是权重还是 cache。
2. 选权重量化、activation 低精度或 KV cache quantization。
3. 再用 benchmark 确认省下来的显存是否真的换来更大的 batch、上下文或部署收益。

## 关键取舍

- 权重量化更适合“模型先装下”。
- KV cache quantization 更适合“长上下文和并发先装下”。
- 如果显存省了，但 TPOT、TTFT 或质量退化太大，量化就不应该被视作成功。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- GPTQ / AWQ：看权重量化如何在有限损失下省驻留。
- FP8 / KV cache quantization 资料：看缓存压缩怎样改变显存预算。

## 证据边界

CPU 可以验证量化误差、理论字节数和预算计算；真实格式能否加载、使用哪个 kernel、峰值显存、吞吐和任务质量，必须在对应 GPU backend 上用同一 workload 对比 baseline。理论压缩率不等于端到端收益。

## 对应 Part 02

- [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)
- [40 GPTQ and AWQ Weight Quantization](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)
- [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)
- [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)

## 典型阅读入口

- [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 本节要点

量化在本专题里是预算工具。它必须先回答“装不装得下”，再讨论“跑得快不快”。
