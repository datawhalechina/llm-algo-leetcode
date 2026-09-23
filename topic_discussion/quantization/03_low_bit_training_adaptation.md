# 03. Low-Bit Training Adaptation | 低比特训练适配

## 页面目标

本节回答的是：当纯 PTQ 不够时，为什么会出现 QAT、LoRA / QLoRA 这类“训练适配量化”的路线。

本节的输出是训练适配边界：质量损失是否值得通过额外训练恢复，以及训练显存、时间和稳定性是否被纳入收益计算。

## 核心机制

很多量化问题并不是“压不下去”，而是“压下去后效果掉太多”。一旦进入这个阶段，量化就不再只是部署动作，而开始和训练、微调、参数高效适配绑在一起。

### 判断入口

- PTQ 失败的根因是误差太大，还是目标任务本来就需要继续适配。
- 你还有没有继续训练或微调预算。
- 你需要的是 full QAT，还是更轻的低比特适配路线。

### 机制取舍

低比特训练适配的核心矛盾是：系统希望把模型表示压低，但模型本身又需要重新吸收这部分误差。于是，量化不再只是表示问题，而会变成训练稳定性和微调效率问题。

### 训练状态

QLoRA 等路线通常把基座模型以低比特形式加载，计算时使用指定的 compute dtype，同时只更新 LoRA adapter。它减少的是可训练参数、梯度和 optimizer state 的规模；这不等于基座权重、激活和训练临时张量全部变成低比特。

![量化路线中的训练适配与部署回路](../../docs/public/topic_discussion/quantization/quantization_strategy_map.svg)

| 状态 | 典型表示 | 主要显存来源 | 需要确认的边界 |
|:---|:---|:---|:---|
| 基座权重 | INT4 / NF4 等低比特存储 | 量化权重、scale 和加载临时空间 | 是否由训练库正确加载并支持反量化 |
| adapter 参数 | FP16 / BF16 或 FP32 | adapter、梯度、optimizer state | trainable 参数量和优化器状态是否匹配 |
| 前向与反向 | 通常使用 compute dtype | activation、临时张量和重算 | 序列长度、batch 与 checkpoint 是否改变峰值 |

因此，QLoRA 的收益应拆成“训练状态减少”和“运行时计算代价”两部分，不能只用模型文件大小解释训练显存或速度。

## 判断框架

1. 先判断 PTQ 是否已经足够。
2. 如果不够，再判断是 full QAT 还是 LoRA / QLoRA 这类更轻量路线。
3. 最后再回到部署验证，看训练适配后的收益是否值得新增成本。

### 训练适配如何回到推理路线

低比特训练适配的输出不是“已经完成量化部署”，而是一个可以继续验证的训练产物。建议按下面的交接顺序处理：

| 阶段 | 需要确认什么 | 输出给下一阶段的内容 |
|:---|:---|:---|
| 适配训练 | 基座 revision、量化配置、compute dtype、可训练参数和验证集 | 训练配置、loss / quality 结果 |
| 产物整理 | adapter 是否可独立加载，是否需要 merge | adapter 或 merged model manifest |
| 浮点参照 | 合并后的模型是否能在固定 workload 下运行 | 66 的 G0 baseline |
| 推理量化 | 是否还要转换为 GPTQ、AWQ、GGUF 或运行时低精度格式 | 67 可加载的量化 artifact |
| 部署复测 | 量化后质量、显存、TTFT、TPOT 和吞吐是否达标 | accept / tune / reject |

因此，`26` 负责理解和实现 QLoRA 机制，`65` 负责训练适配项目的产物与选择，`66 → 67` 负责把训练结果放回统一推理 workload 中比较。adapter 能加载只能证明训练产物契约成立，不能替代量化 backend 的部署证据。

### 路线取舍

- full QAT 更完整，但成本最高。
- LoRA / QLoRA 更像“低比特前提下的适配折中”。
- 如果任务本身变化不大，继续训练未必比更好的 PTQ / AWQ / GPTQ 更划算。

### 训练框架与产物选择

框架选择首先取决于学习目标和产物要求，而不是工具名本身：

| 目标 | 更适合的路径 | 需要保留的证据 |
|:---|:---|:---|
| 理解训练机制、逐项控制配置 | TRL / 原生 Transformers + PEFT | 量化配置、trainable 参数、loss 和验证结果 |
| 单卡快速验证 QLoRA | Unsloth 或等价轻量封装 | 版本、显存、step time、adapter manifest |
| 配置驱动的重复实验 | LLaMA-Factory / Axolotl | YAML、数据版本、checkpoint 和复现实验命令 |
| 进入部署链路 | PEFT / Transformers 产物再转换 | adapter / merged model、转换日志和目标格式 |

这些框架可以降低训练入口成本，但不能替代量化 artifact 的格式检查和部署 benchmark。学习者应先明确输出是 adapter、merged model 还是推理量化文件，再选择框架。

### 证据等级

CPU 或小模型实验可以验证 adapter 是否只更新目标参数、量化配置字段、梯度路径和损失计算；真实 GPU 才能验证训练峰值、低比特 kernel、step time、OOM 和数值稳定性。训练适配后的 adapter 还要经过独立的合并或加载测试，不能直接当成 67 的推理量化 artifact。

## 参考入口

- QAT 经典资料：理解量化误差如何被训练过程显式吸收。
- QLoRA / PEFT 资料：理解为什么低比特和参数高效微调常一起出现。

### 对应 Part 02

- `26` QLoRA 与 4bit 量化
- `60` LoRA 微调项目

### 典型阅读入口

- [02 PTQ 与 QAT 的介入时机](./02_ptq_and_qat_timing.md)
- [06 部署与 Benchmark 决策](./06_deployment_and_benchmark_decision.md)

### 本节要点

低比特训练适配是量化路线和微调路线的交叉区，重点不是“多做训练”，而是“多做训练是否真能换回可接受收益”。

### 进入下一页

若目标是压缩模型驻留并尽量保持推理路径稳定，进入 [04 权重量化与后训练压缩](./04_weight_only_compression.md)；若目标是缓存预算，则进入 [05 FP8 与 KV Cache 量化](./05_fp8_and_kv_cache_quantization.md)。
