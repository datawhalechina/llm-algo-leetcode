# 训练工程附录：Trainer / Accelerate / DeepSpeed / Lightning

## 页面目标

这一页不重讲 `SFT / LoRA / scheduler / eval` 机制本身，而是回答：当你已经知道训练闭环是什么之后，应该用什么训练工程封装把它搭起来。

它是 `监督微调与训练工程` 的工程附录页，不替代 `01-06` 主线，只负责把“微调闭环理解”接到“最小训练工程落地”。

## 这页负责什么

- `Trainer / Lightning`：关注训练循环如何被组织起来。
- `Accelerate`：关注单卡到多卡、AMP、device placement 的最小封装。
- `DeepSpeed`：关注更大规模训练时的工程接入与状态分摊能力。
- `checkpoint / logging / tracking`：关注训练恢复、实验记录和项目交付。
- `DataLoader / Sampler / Collator`：关注训练输入管线的最小工程闭环。

## 这页不展开什么

- `DDP / FSDP / ZeRO` 的并行机制细节
- `autograd / backward / checkpointing` 的梯度机制细节
- 完整的数据系统、离线清洗和训练平台架构
- 推理部署、服务化和在线评测

这些分别放在 `通信与并行`、`反向传播与训练机制`、`Part 4` 和项目页里更合适。

## 最小训练工程闭环

如果你只是想把一个 SFT / LoRA 项目跑通，通常先把下面这条线立住：

1. 准备 `dataset / collator`
2. 组装 `model / tokenizer / LoRA adapters`
3. 配置 `optimizer / scheduler / accumulation / amp`
4. 选择训练封装
5. 补 `eval / checkpoint / logging`
6. 产出 `adapter / config / report / adopt decision`

这条线的重点是：先保证训练闭环可复现，再追求更大的系统封装。

## 几种常见训练封装

| 工具 | 更适合什么时候用 | 你主要得到什么 | 常见风险 |
|:---|:---|:---|:---|
| `Transformers Trainer` | 第一个 SFT / LoRA 闭环 | 最快搭起 `train / eval / save` | 容易只会配参数，不理解训练口径 |
| `Accelerate` | 需要从单卡平滑走到多卡或 AMP | 更轻的训练循环控制与 device 封装 | 训练脚本结构还是要自己管 |
| `DeepSpeed` | 模型更大、状态更重、要接入更强训练系统能力 | 状态分摊、吞吐优化、训练系统扩展 | 容易把工程框架问题误当成训练机制问题 |
| `Lightning` | 需要把训练流程组织得更结构化 | 统一 `train/val/test/logging` 组织方式 | 抽象过厚时会遮住真正的训练细节 |

## 一个实用判断

- 想先把 LoRA 闭环跑通：先用 `Trainer`
- 想保留更轻的自定义空间：先用 `Accelerate`
- 已经确认会碰到更重的训练系统约束：再引入 `DeepSpeed`
- 团队更在意训练代码结构和复用：可以考虑 `Lightning`

先把训练问题讲清楚，再升级封装层，不要反过来。

## 微调项目的工具分层

60–65 不绑定某一个一键式框架。学习者应先用最小工具理解数据、梯度、adapter、量化和评测，再根据环境选择封装层：

| 层级 | 工具 | 适合本教程的用法 | 不应该替代的内容 |
|:---|:---|:---|:---|
| 机制层 | `PyTorch` + `Transformers` | 60 的最小 LoRA 闭环、61 的结构读取、训练结果复现 | 不替代数据审计和质量评测 |
| 参数高效微调 | `PEFT` | LoRA、IA3、AdaLoRA、adapter 保存与加载 | 不负责完整数据管线和项目决策 |
| 训练任务封装 | `TRL` | SFT，以及后续 DPO / GRPO 等后训练任务 | 不负责 CUDA kernel 或显存结论 |
| 项目配置封装 | [`LLaMA-Factory`](https://llamafactory.readthedocs.io/en/latest/) | 62–65 的配置化 SFT、LoRA、QLoRA 对照；适合快速复现实验 | 不应隐藏模型、数据、dtype 和评测口径 |
| GPU 加速封装 | [`Unsloth`](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) | Colab 或消费级 GPU 上的 LoRA / QLoRA 加速对照 | 加速结果不能直接当作机制基线 |
| 规模化训练 | `Accelerate`、`FSDP`、`DeepSpeed`、`Megatron-LM` | 更大模型、多卡和状态分摊的扩展入口 | 不属于 60–65 的默认前置环境 |

### 60–65 的推荐分工

- **60**：`PyTorch + Transformers + PEFT` 作为可解释的核心实现；可选用 `TRL` 重写 SFT，但必须保持同一数据和评测口径。
- **61**：主要依赖 `Transformers` 的 config 和模块命名；不引入训练框架，避免把架构判断变成配置按钮练习。
- **62**：可用 `TRL` 或 `LLaMA-Factory` 扩展真实 SFT；重点检查 chat template、数据格式、验证集和 artifact。
- **63**：优先用 `PEFT` 做 LoRA target modules、rank 和变体比较；`LLaMA-Factory` 只作为配置化复现实验入口。
- **64**：以 `datasets`、自定义审计和评测脚本为主；框架只能消费数据，不能替数据质量背书。
- **65**：用 `PEFT + bitsandbytes` 解释 QLoRA 的 4-bit 基座与 adapter；`Unsloth` 可作为 GPU 加速扩展，不能替代基线。

### 为什么同时保留 LLaMA-Factory 和 Unsloth

两者解决的问题不同：[`LLaMA-Factory`](https://llamafactory.readthedocs.io/en/latest/) 更像覆盖多模型、多任务和多种微调算法的项目配置平台；[`Unsloth`](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) 更像面向本地/Colab GPU 的训练加速层。它们都可以降低启动成本，但也会增加版本、默认参数和 kernel 兼容性带来的变量。因此本教程的证据顺序是：

`最小 PyTorch 基线 → PEFT/TRL 机制实现 → LLaMA-Factory 或 Unsloth 扩展 → 固定 workload 的真实 GPU 对照`。

扩展实验至少保存：框架版本、模型版本、量化方式、dtype、LoRA target modules、数据切分、有效 batch、梯度累积、评测脚本和导出 artifact。否则“更快”可能只是默认配置或数据口径不同造成的表象。

## 强化学习与在线对齐框架

SFT 之后进入 DPO、GRPO 或在线对齐时，框架选择应服务于“采样—打分—更新—评测”的闭环，而不是只比较 API 是否方便。

| 框架或组件 | 适合承担的职责 | 需要保留的证据 |
|:---|:---|:---|
| [`TRL`](https://huggingface.co/docs/trl/index) | 教学和小规模实验；承接 SFT、DPO、GRPO 的训练循环 | 算法配置、reference/policy 来源、reward 或 preference 数据、评测结果 |
| [`verl`](https://verl.readthedocs.io/en/latest/) | 生产化 RLHF/RLVR；编排 rollout、训练、奖励和资源调度 | rollout backend、采样策略、更新频率、GPU 分配、吞吐与显存 |
| [`OpenRLHF`](https://openrlhf.readthedocs.io/en/latest/) | 分布式 PPO/GRPO 与多阶段 RLHF 对照 | actor/reference/reward/critic 角色、并行策略、通信和失败记录 |
| [`vLLM`](https://docs.vllm.ai/) / [`SGLang`](https://docs.sglang.ai/) | 作为 rollout / sampling backend 提供批量生成 | 模型版本、采样参数、queue wait、生成吞吐和 handoff cost |
| `Ray` | 管理 rollout、训练、评测之间的任务和资源编排 | placement、等待时间、重试、资源占用和回滚点 |

推荐的学习顺序是：先用 CPU 或最小 `TRL` 理解机制，再用 `TRL` 跑通小规模 DPO/GRPO，最后根据资源和并行需求迁移到 `vLLM/SGLang + verl` 或 `OpenRLHF`。框架扩展必须沿用同一 workload、评测集和决策阈值，不能把框架默认值当成算法结论。

### 框架实验放在哪些项目节

框架实验作为项目节的可选扩展，不改变 CPU 机制主线；每个扩展都要复用项目已有的模型、数据、checkpoint 和评测口径。

| 项目节 | 默认实现 | 可选框架实验 | 需要比较的结果 |
|:---|:---|:---|:---|
| `13` 端到端 SFT | `Transformers` / PyTorch | 用 `TRL` 的 SFT 流程复现最小 smoke | checkpoint、loss、更新耗时、产物字段 |
| `60` LoRA 项目 | `PEFT` + Transformers | `TRL`、`LLaMA-Factory` 或 `Unsloth` 三选一 | 同一 workload 下的显存、吞吐、验证 loss、adapter |
| `62` 指令微调项目 | CPU 报告 + 可选 GPU SFT | 选择 `TRL` 或 `LLaMA-Factory` 完成真实模型对照 | chat template、生成质量、资源和可复现配置 |
| `84` DPO 项目 | 最小 DPO 机制实现 | `TRL` DPOTrainer 对照 | policy/reference checkpoint、偏好指标、训练成本 |
| `85` GRPO 项目 | 组内采样与奖励机制 | `TRL` GRPOTrainer；需要更大 rollout 时再试 `verl` | reward 分布、group size、rollout 吞吐、显存 |
| `86` 在线 benchmark | CPU 决策与结果汇总 | `vLLM/SGLang` rollout + `verl` 编排作为扩展 | policy freshness、更新代价、安全门槛、回滚证据 |

学习者不需要同时运行所有框架；每个项目保留一个机制基线，再选择一个框架扩展即可。这样能把“框架能不能跑”与“算法或数据是否有效”分开判断。

### SFT checkpoint 如何进入对齐项目

SFT 项目结束时至少保存 `model_id / revision`、完整 checkpoint 或 LoRA adapter、tokenizer/chat template、数据版本、dtype、训练配置和验证结果。进入 DPO 时，通常用该 SFT checkpoint 初始化 policy，并用同一 SFT checkpoint 的冻结副本作为 reference；进入 GRPO 时，则以 SFT 或已通过评测的 DPO checkpoint 初始化 policy，再记录 rollout backend 和 reward 版本。后续项目必须能回答“使用了哪个 checkpoint、何时导出、与哪份评测结果对应”，不能只保存 loss 曲线。

## 输入管线该管到什么程度

训练工程最容易被低估的不是模型，而是输入管线。

最小需要确认的是：

- `DataLoader` 是否稳定产出 batch
- `Sampler` 是否和训练/验证划分一致
- `Collator` 是否正确对齐 padding、mask 和 labels
- 长样本是否有明确的 truncation / packing 策略

如果这些没立住，后面的 loss、吞吐和显存结论都可能失真。

## checkpoint / logging / tracking

训练工程闭环至少要留住下面这些东西：

- `checkpoint`：确认从哪一步恢复、保存了什么
- `adapter / tokenizer / config`：确认项目交付是否完整
- `train / val metrics`：确认不是只留了一条 loss 曲线
- `samples / report`：确认生成样例和最终判断可以复盘

这里不要求你一开始就把 `W&B / TensorBoard / MLflow` 都接满，但至少要有一条稳定的实验记录路径。

## 和主线怎么连接

| 当你在主线里卡住什么 | 回到这页看什么 |
|:---|:---|
| `01` 数据和 labels 已懂，但脚本还没搭起来 | `DataLoader / Collator / 最小训练封装` |
| `02` LoRA 已懂，但训练脚本太散 | `Trainer / Accelerate` |
| `03` scheduler / accumulation 已懂，但多卡和 AMP 很乱 | `Accelerate / DeepSpeed` |
| `04` 实验能跑，但记录和恢复不完整 | `checkpoint / logging / tracking` |
| `05` 项目要交付，但 artifact 不全 | `adapter / tokenizer / config / report` |

## 相关专题

- [反向传播与训练机制专题](../../backpropagation_training_mechanism/intro.md)：当你需要先理解 `autograd / AMP / checkpointing` 是怎么工作的，再回来看工程封装。
- [通信与并行](../../communication_parallel/intro.md)：当你开始比较 `DDP / FSDP / ZeRO / DeepSpeed` 的并行与状态分摊边界时先看这里。
- [显存优化](../../memory_performance_tuning/intro.md)：当训练脚本已经能跑，但 OOM 和显存账本还没压住时先看这里。

## 本节要点

训练工程附录的目的不是推荐单一框架，而是把 `SFT / LoRA` 主线接到可复现的训练脚本、输入管线和项目交付闭环。
