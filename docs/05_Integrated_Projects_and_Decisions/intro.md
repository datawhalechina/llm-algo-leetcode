# 第五部分：综合项目与架构决策

> 通过端到端项目，把模型、后端、运行负载和质量要求放在一起测量，再据此选择方案。

## 本部分定位

本部分从完整项目出发：先明确目标和约束，再固定负载与基线、定位瓶颈、比较方案，最后核对质量、性能、资源和成本。

## 推荐学习方式

项目按这条路径推进：

```text
明确目标与约束 → 固定负载和基线 → 定位瓶颈 → 比较方案 → 核对质量、性能与成本 → 做出决策
```

本部分分为“核心交付项目”和“开放实践项目”。核心项目按各自入口页准备环境和输入；开放实践用于探索其他模型、硬件或应用场景。

## 5.1 核心交付项目

按目标选择项目；各项目使用的环境、输入和运行步骤见项目入口页。

| 项目方向 | 组合范围 | 项目结论 | 环境 |
|:---|:---|:---|:---|
| [推理服务交付综合项目](./01_Inference_Service/index.md) | 推理后端、缓存与并发配置 | 固定请求负载下的质量、服务目标（SLO）、容量与回滚判断 | [`INF-VLLM-011-CU128`](../gpu_environment_assets.md#inf-vllm-011-cu128) |
| [后训练交付综合项目](./02_Post_Training_Delivery/index.md) | MiniMind 数据、SFT、LoRA、DPO 与独立评测 | 固定训练预算下的模型质量、资源与产物交付 | [`PT-SFT-CUDA`](../gpu_environment_assets.md#pt-sft-cuda) |
| [端到端性能优化与回归交付项目](./03_Performance_Optimization_Delivery/index.md) | 优化器路径、激活检查点与组合 | 固定训练负载下的完整训练步收益与回归判断 | [`PERF-CUDA-MECHANISM`](../gpu_environment_assets.md#perf-cuda-mechanism) |
| [量化交付综合项目](./04_Quantization_Delivery/index.md) | GPTQ 产物生成、质量评测与 vLLM 部署 | 量化产物在目标服务负载下的质量、性能与资源结论 | [`Q-CUDA-MECHANISM`](../gpu_environment_assets.md#q-cuda-mechanism) → [`INF-VLLM-011-CU128`](../gpu_environment_assets.md#inf-vllm-011-cu128) |
| [算子优化交付综合项目](./05_Operator_Optimization_Delivery/index.md) | Triton FlashAttention 与两层解码器预填充（prefill） | 固定形状下的输出正确性、长序列延迟与显存结论 | [`PERF-CUDA-MECHANISM`](../gpu_environment_assets.md#perf-cuda-mechanism) |

## 5.2 开放实践项目

可用于新模型或硬件适配、竞赛任务、行业数据、端侧部署、多模态服务等场景。项目入口和实验记录参见[开放实践项目模板](./open_practice/_template/README.md)。

## 综合决策检查表

项目结论应同时考虑质量、性能、资源、成本和回退条件，不要只凭理论峰值或单项基准选择方案。

| 决策维度 | 需要回答的问题 | 主要证据 |
|:---|:---|:---|
| 硬件与容量 | 模型、上下文和并发能否稳定运行，扩展受什么限制？ | 峰值显存、带宽、容量边界与扩展效率 |
| 架构与引擎 | 单机、多卡、多节点或分阶段部署中，哪种组合符合目标？ | 部署拓扑、后端能力、端到端延迟与可靠性 |
| 性能 | 延迟、吞吐和资源利用率是否达到目标？ | P50/P99、吞吐、各阶段耗时与回归结果 |
| 质量与交付 | 输出、训练稳定性和模型产物是否满足要求，是否可回退？ | 留出集、正确性/回归测试、模型文件校验值、启动与回滚记录 |
| 成本 | 设备、运行、通信、存储和维护成本如何变化？ | 单请求或单位吞吐成本、容量规划与敏感性分析 |
