# 05. Project Delivery and Decision | 项目交付与决策

## 页面目标

这一页回答的是：SFT 微调完成后，哪些产物必须保留，什么样的结果才值得交付和采用。

## 你要先确认什么

- adapter 是否可保存、可复现。
- tokenizer 和 config 是否和训练时一致。
- 项目报告是否说明了数据、训练和评估口径。
- 结果是否足以支撑采用决策。

## 演化路径

项目交付不只是“训完了”，而是要把实验结果整理成可复现的资产。

1. 保存 adapter 和必要配置。
2. 保存 tokenizer 和数据处理口径。
3. 保存 train / val 结果和代表性样例。
4. 写清楚采用或不采用的理由。
5. 如果后续要扩展，再把结论接回进阶占位。

这一页是 SFT 闭环的最后一环。
没有它，前面训练得再完整，也很难形成真正的项目结论。

## 60 的实验步骤

完整代码和参数说明见 [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)。执行时按以下顺序建立证据：

1. 固定真实模型、数据集版本、`SPLIT_SEED`、验证集比例、`dtype`、`seq_len` 和训练步数。
2. 先运行同口径 full-parameter baseline，再运行 LoRA candidate；两者使用同一批数据和同一评测集。
3. 记录可训练参数量、峰值显存、step time、token 吞吐、train/validation loss 和生成样例。
4. 保存 adapter、tokenizer、训练配置、数据审计和 JSON 报告；需要多次运行时只改变 `REAL_SEED`。
5. 先判断质量是否达标，再综合资源收益输出 `accept / tune / reject`。

当前 60 的三组固定切分实验显示 LoRA 资源收益稳定，但尚未加入 task-level 生成质量指标，因此仍应标记为 `tune`，不能只凭 loss 下降宣布采用。

## 项目分流

完成 [13 End-to-End Fine-Tuning Experiment](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md) 后，不同项目应按问题分流，而不是全部同时运行：

| 项目 | 适合什么时候进入 | 主要产出 |
|:---|:---|:---|
| [60 LoRA Fine-Tuning](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md) | 第一个完整 LoRA 交付 | adapter、tokenizer、config、评测和采用决策 |
| [62 Instruction Fine-Tuning](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md) | 任务格式和 instruction 数据需要单独验证 | 指令格式、训练结果和任务样例 |
| [63 LoRA Variants Benchmark](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.md) | 已有稳定 baseline，需要比较 adapter 方案 | 统一规格、质量、参数效率和资源对比 |
| [64 SFT Data Quality](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) | loss 或样例异常，怀疑数据质量 | 数据审计、清洗规则和质量报告 |
| [65 QLoRA Selection](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md) | 单卡显存不足或需要低比特训练 | 量化配置、显存、吞吐、质量和选型决策 |

核心路径先完成 60；只有当问题与任务格式、LoRA 变体、数据质量或显存预算相关时，再进入对应扩展项目。扩展项目的结论必须回填到统一的 baseline / candidate / quality / resource / decision 报告中。

## 与 13 和其他项目的统一协议

`13 End-to-End Fine-Tuning Experiment` 是训练基线，不是项目交付的替代品：它证明训练循环、loss、验证集和最小端到端流程能够工作。`60–65` 在这个基线上按问题分流，分别比较 LoRA 交付、结构、指令格式、数据质量、LoRA 变体和 QLoRA 预算。最后回到本页检查 artifact、复现条件和采用理由。

所有 60–65 项目使用同一个外层报告结构：

```text
config / baseline / candidates
quality / resources / artifacts
decision / environment
```

公共结构由 [`fine_tuning_result_schema.py`](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/tools/fine_tuning_result_schema.py) 维护；项目可以增加自己的扩展字段，但不能省略比较基线、质量证据、资源证据和决策原因。详细执行步骤见[训练微调项目验证清单](../../verification/fine_tuning_projects.md)。

这与 66–70、73–76 的做法一致：统一的是结果接口和证据边界，不是把 TTFT、KV Cache 等推理指标套到训练微调项目上。

## 常见误区

- 只保存权重，不保存 tokenizer 和 config。
- 报告里没有说明数据和 loss 口径。
- 只给出 loss 曲线，没有样例和结论。
- 结果能复现，但无法解释为什么值得采用。

## 经典阅读入口

- [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- [26 QLoRA and 4bit Quantization](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [13 End-to-End Fine-Tuning Experiment](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [62 Instruction Fine-Tuning Project](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)
- [63 LoRA Variants Benchmark](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.md)
- [64 SFT Data Quality Project](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md)
- [65 QLoRA Selection Project](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md)

## 前置关系

- 先看 `04`，确认实验已经闭环。
- 再看 `05`，把实验变成交付决策。

## 项目结论

项目交付不是训练的附属步骤，而是闭环的一部分。
只有能交付、能复现、能解释的结果，才算真正完成 SFT。
