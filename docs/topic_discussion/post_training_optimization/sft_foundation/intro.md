# Task0：SFT / LoRA 基础与后训练接口

> 所属专题：后训练优化　模块定位：监督微调、参数高效训练与项目接口

## 模块定位

本模块建立进入后训练对齐所需的最小训练接口：数据如何进入 loss，LoRA 如何改变更新对象，训练结果如何成为后续偏好优化的起点。

这里不追求覆盖所有训练工程工具，而是先形成可解释、可复查的 SFT / LoRA 基线。偏好数据、DPO、GRPO 和在线对齐在上一级专题的 Task1–6 展开。

## 推荐顺序

```text
数据与 labels → SFT loss → LoRA / PEFT → 训练控制 → 端到端实验 → 项目交付
```

| 顺序 | 学习内容 | 正文入口 | 主要产出 |
|:---|:---|:---|:---|
| 1 | 数据、tokenization 与 loss 对齐 | [01 SFT 数据与 Loss](./01_sft_data_and_loss.md) | `input_ids / attention_mask / labels` 契约 |
| 2 | LoRA 与 PEFT 设计 | [02 LoRA / PEFT Design](./02_lora_peft_design.md) | adapter 配置、target modules 与参数预算 |
| 3 | 训练控制 | [03 Training Control](./03_training_control.md) | scheduler、梯度累积、effective batch 和 checkpoint 口径 |
| 4 | 端到端微调实验 | [04 End-to-End Experiment](./04_end_to_end_experiment.md) | 训练、评测、资源和回归证据 |
| 5 | 项目交付与决策 | [05 Project Delivery and Decision](./05_project_delivery_decision.md) | `accept / tune / reject` 与可交付 artifact |

## Part 02 学习入口

- 基础训练循环：[09 SFT Training Loop](../../../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)
- LoRA 机制：[10 LoRA Tutorial](../../../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- 数据与 readiness：[32 SFT 数据工程](../../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md)、[33 微调准备度](../../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md)
- 训练接口：[13 端到端微调实验](../../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- 项目入口：[60 LoRA 微调项目](../../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- 扩展分支：[61–65 训练与微调项目](../../../02_PyTorch_Algorithms/2_10.md)

## 前置与后续

结构基础不足时回补[大模型架构](../../model_architecture/intro.md)；训练显存和 step time 异常时进入[性能优化](../../performance_optimization/intro.md)；QLoRA 和低比特训练进入[量化与压缩](../../quantization/intro.md)。完成 Task0 后，再回到[后训练优化主入口](../intro.md)学习 PPO、DPO、GRPO 和在线对齐。

## 环境与证据

数据格式、loss、状态转移和小规模验证可以 CPU-first；真实 SFT / LoRA 项目通常需要单 GPU。实验至少固定模型、数据切分、dtype、batch、sequence length、seed 和评测口径，并保存 checkpoint、adapter、配置、指标和失败记录。

训练工程细节见[训练工程附录](./training_engineering_appendix.md)，项目证据协议见[项目交付附录](./project_delivery_appendix.md)。
