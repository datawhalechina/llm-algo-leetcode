# 监督微调与训练工程（Supervised Fine-Tuning and Training Engineering）

> 专题类型：基础支撑　主服务目标：SFT 训练与工程复现

## 页面导语

本专题围绕模型结构、SFT、LoRA、训练控制和项目交付，最终形成可复现、可评测、可采用的微调方案。

本专题对应监督微调与训练工程的 V1：负责从任务数据、SFT、LoRA 到训练项目交付；偏好数据、DPO、GRPO 和在线对齐属于后训练优化 V2。V1 的训练结果可以作为 V2 的输入，但两者不共用项目目标。

## 如何开始

推荐从 Part 02 的[2.3 训练与微调闭环](../../02_PyTorch_Algorithms/2_3.md)开始；结构基础不足时回补[2.1 基础算子](../../02_PyTorch_Algorithms/2_1.md)和[2.2 模型架构](../../02_PyTorch_Algorithms/2_2.md)。

- 必读前置：PyTorch 训练循环、decoder block、loss 和基本显存概念。
- 主线入口：先完成 Task0–3 的结构基础，再进入 Task4–6 的 SFT、LoRA 和项目交付。
- 按需回补：需要长上下文时看 `30`，需要 LoRA 方案比较时看 `31`，需要 QLoRA 或偏好优化时转到对应专题。
- 真实训练前：先阅读[使用指南](../../docs/guide.md)和具体 Notebook 的环境说明。

## 主学习线与分级

`Task0-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列的 `01-05` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 / 项目入口 | 专题正文 |
|:---|:---|:---|:---|
| Task0 | PyTorch 热身与训练对象 | [00 PyTorch Warmup](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb) | 共享前置：见[大模型架构](../model_architecture/intro.md) |
| Task1 | 模型结构入口 | [01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb) → [02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb) | 共享前置：见[大模型架构](../model_architecture/intro.md) |
| Task2 | 位置编码与注意力 | [03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb) → [04 MHA / GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) | 共享前置：见[大模型架构](../model_architecture/intro.md) |
| Task3 | Block 组装与架构变体 | [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) → [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb)；扩展：[Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb)、[Part 02 · 07 MoE 负载均衡](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb) | 共享前置：见[大模型架构](../model_architecture/intro.md) |
| Task4 | SFT 与 LoRA | [09 SFT Training Loop](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb) → [10 LoRA](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.ipynb)；扩展：[31 LoRA Variants Theory](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.ipynb) | [02 LoRA PEFT Design](./02_lora_peft_design.md) |
| Task5 | 项目准备、训练控制与端到端实验 | [Part 02 · 32 SFT 数据工程](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb) → [Part 02 · 33 微调准备度](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb) → [Part 02 · 12 梯度累积](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb) → [Part 02 · 11 学习率调度器](../../02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.ipynb) → [Part 02 · 13 端到端微调实验](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)；扩展：[Part 02 · 30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb) | [03 训练控制](./03_training_control.md) |
| Task6 | 综合项目与交付 | 核心：[Part 02 · 60 LoRA 微调项目](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb)；扩展：[Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)、[Part 02 · 62 指令微调](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb)、[Part 02 · 63 LoRA 变体对比](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb)、[Part 02 · 64 SFT 数据质量](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb)、[Part 02 · 65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb) | [05 项目交付与决策](./05_project_delivery_decision.md) |

### 核心与扩展分级

核心路径先建立结构、训练接口、数据准入和 LoRA 交付的共同口径；其中 32/33 是“项目准备核心”，只要求完成最小数据审计和 readiness 判断，不要求搭建完整数据平台。扩展路径再进入 MoE、真实结构探索、LoRA 变体、长上下文和 QLoRA。没有 GPU 的学习者可以先完成机制和 CPU-first 模板，有 GPU 的学习者再进入真实微调项目。

| Task | 核心路径 | 扩展路径 | 环境级别 |
|:---|:---|:---|:---|
| Task0 | [00 PyTorch Warmup](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb) | — | Practice-P0 |
| Task1 | [01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)、[02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb) | — | Practice-P0 |
| Task2 | [03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)、[04 Attention](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) | 更复杂的 Attention 变体 | Practice-P0/P1 |
| Task3 | [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb)、[Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb) | [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb)、[Part 02 · 07 负载均衡](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb) | 核心 P0/P1；扩展 P1 |
| Task4 | [09 SFT](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb) → [10 LoRA](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.ipynb) | [31 LoRA Variants Theory](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.ipynb) | Practice-P0/P1 |
| Task5 | [Part 02 · 32 SFT 数据工程](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb) → [Part 02 · 33 微调准备度](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb) → [Part 02 · 12 梯度累积](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb) → [Part 02 · 11 学习率调度器](../../02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.ipynb) → [Part 02 · 13 端到端微调实验](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb) | [Part 02 · 30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb) | 项目准备核心 P0/P1；扩展 P1 |
| Task6 | [Part 02 · 60 LoRA 微调项目](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb) | [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)、[Part 02 · 62 指令微调](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb)、[Part 02 · 63 LoRA 变体对比](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb)、[Part 02 · 64 SFT 数据质量](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb)、[Part 02 · 65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb) | 核心 P1；扩展 P1/P2 |

## 学习方式与项目产出

先按上面的 `Task0-6` 走 Notebook 主线；核心路径用于建立机制和完成最小训练闭环，扩展路径用于真实模型结构、复杂数据、长上下文和低显存方案。需要连续理解概念时阅读专题正文 `01-05`，需要判断表和项目分流时阅读[监督微调正文](./casebook.md)，需要完整串联路线时阅读[监督微调深入阅读](./walkthrough.md)。训练工程细节放在[训练工程附录](./training_engineering_appendix.md)，项目证据链放在[项目交付附录](./project_delivery_appendix.md)，60–65 的统一验证口径见[训练微调项目验证清单](../../docs/verification/fine_tuning_projects.md)。

如果问题已经跨到别的专题：[大模型架构](../model_architecture/intro.md)负责结构地基，[显存优化](../memory_performance_tuning/intro.md)负责 OOM 与显存账本，[量化与压缩](../quantization/intro.md)负责 QLoRA 与低比特路线，[后训练优化](../post_training_alignment/intro.md)负责 DPO / GRPO，[性能分析](../profiling/intro.md)负责训练证据链，[通信与并行](../communication_parallel/intro.md)负责多卡边界。

### 训练工具入口

工具按学习目标选择，不要求学习者一开始安装完整工具链：

| 目标 | 推荐工具 | 在本路线中的位置 |
|:---|:---|:---|
| 理解训练和 LoRA 机制 | `PyTorch` + `Transformers` + `PEFT` | 60、61、63 的核心实现 |
| 快速搭建 SFT / QLoRA 项目 | [`LLaMA-Factory`](https://llamafactory.readthedocs.io/en/latest/) | 62、65 的配置化扩展 |
| 在 Colab 或消费级 GPU 上加速 | [`Unsloth`](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) | 60、65 的可选 GPU 对照 |
| 使用标准训练任务封装 | `TRL` | SFT 以及后续 DPO / GRPO 扩展 |
| 扩展到多卡或更大模型 | `Accelerate` / `FSDP` / `DeepSpeed` | 训练系统扩展，不是 60–65 的默认前置 |

主线先用最小实现建立可解释基线，再选择 LLaMA-Factory 或 Unsloth 做扩展；完整工具边界、版本变量和实验记录要求见[训练工程附录](./training_engineering_appendix.md)。

### 项目产出

项目按“最小闭环 → 方案扩展 → 交付决策”分层：

- **核心项目：** [60 LoRA 微调](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb)，验证数据、adapter、训练控制和评测能否形成可交付闭环。
- **结构扩展：** [61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)，为真实模型结构和 LoRA target modules 提供依据。
- **项目扩展：** [62 指令微调](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb)、[63 LoRA 变体对比](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb)、[64 SFT 数据质量](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb)、[65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb)。

最终结论至少应同时说明：数据是否可信、LoRA 是否挂载合理、训练控制是否一致、质量是否满足目标、资源成本是否可接受，以及结果是否值得继续采用。

### 项目协议：统一骨架，保留专属指标

训练微调项目与推理、显存项目一样，需要统一结果协议，但不需要强行使用相同指标。`13` 提供训练基线，`60–65` 按问题分流，专题正文 `05` 负责解释交付检查和最终决策。所有项目至少记录 `config / baseline / candidates / quality / resources / artifacts / decision / environment`；LoRA 变体、数据审计、结构探索和 QLoRA 再在此基础上增加自己的字段。

因此，`64` 主要回答“数据是否准入”，`63` 主要回答“哪种 LoRA 方案更合适”，`65` 主要回答“质量与显存预算是否同时满足”，不能为了格式统一而给它们添加没有意义的吞吐或服务指标。统一的是证据链和 `accept / tune / reject` 决策，不是每个项目的实验内容。

## 环境与验证

基础结构、数据工程和多数 Notebook 可 CPU-first；真实 LoRA/SFT 项目建议使用单 GPU。运行前先查看具体 Notebook 的环境说明，固定模型、数据、dtype、batch、seq_len、seed 和评测口径，并保存训练配置、adapter、评测结果和项目报告。

没有 GPU 时先完成核心机制和 CPU-first 模板；接入 GPU 后再运行 60，并按需扩展 61–65。训练问题若主要表现为 OOM 或 step time 异常，应转到[显存优化](../memory_performance_tuning/intro.md)和[性能分析](../profiling/intro.md)。
