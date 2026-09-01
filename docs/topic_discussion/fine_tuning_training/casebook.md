# 监督微调（SFT）闭环正文

这页只做训练微调问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 判断表

先分清问题在数据、loss、LoRA 配置、训练控制还是项目交付，再判断是不是已经需要转去显存、量化或对齐分支。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 数据字段混乱、空回答或重复样本较多 | `data readiness` | [32 Data Engineering](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) | 先做字段清洗、长度统计、重复和空值审计，不要直接调训练超参 |
| loss 能跑，但生成质量没变好 | `data / loss mismatch` | [01](./01_sft_data_and_loss.md) | 检查 `input_ids / attention_mask / labels`、response-only loss、EOS 对齐 |
| LoRA 训练正常，但收益很弱 | `adapter config` | [02](./02_lora_peft_design.md) | 检查 target modules、`r / alpha / dropout`、可训练参数占比 |
| 数据可用，但目标、预算或评测窗口不清楚 | `training readiness` | [33 Fine-Tuning Readiness](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md) | 明确训练目标、数据规模、时间/显存预算和进入项目页的条件 |
| loss 波动怪，step 口径不一致 | `training control mismatch` | [03](./03_training_control.md) | 检查 scheduler、accumulation、optimizer step、effective batch |
| 训练跑通了，但实验结论站不住 | `evaluation gap` | [04](./04_end_to_end_experiment.md) | 补 train / val、样例评估、速度与显存记录 |
| 长样本导致 fit rate 低或成本明显上升 | `long-context budget` | [30 Long Context Fine-Tuning](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md) | 看长度分布、packing、fit rate、显存和 step time，再决定是否扩大窗口 |
| adapter 产出了，但项目不可交付 | `delivery gap` | [05](./05_project_delivery_decision.md) | 检查 adapter、tokenizer、config、artifact、采用结论 |
| 基础闭环跑通后需要扩展 | `branching` | [06](./06_visual_assets.md) | 再进入 `26 / 31 / 15 / 16` 等分支 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| `input_ids / attention_mask / labels` | 数据是否真的形成了正确 supervision | 只要能 tokenize 就算数据正确 |
| LoRA target modules | adapter 是否挂在真正有意义的层上 | 直接沿用默认层，不看结构差异 |
| scheduler / accumulation | 训练控制口径是否统一 | 按 micro-batch 计 step，把 loss 曲线看花 |
| eval sample / val loss | 训练结果是否真的变好 | 只看 train loss，不看生成样例 |
| artifact / report | 实验是否可复现、可交付 | 训练结束就算项目完成 |

`60` 的价值不在于再讲机制，而在于把这些判断收成真正的项目结论。

## 本节要点

这页的职责不是再讲一遍 SFT 流程，而是把训练微调里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`，项目收口留给 `60`。
