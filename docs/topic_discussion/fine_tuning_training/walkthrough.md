# 监督微调（SFT）闭环深入阅读

假设你接手的是一个新的监督微调任务：已经有一批 `prompt / response` 数据，目标不是证明模型“能训练”，而是把这批数据走成一个可验证、可交付的 LoRA 微调闭环。

这条线按问题暴露顺序展开：数据能不能进入训练，监督口径是否成立，LoRA 是否挂对了层，训练控制有没有把实验做歪，最后产出的结果能不能真正交付。

## 第一段：先确认数据能不能进入训练

故事通常从一条具体样本开始。先看 [32 Data Engineering for SFT](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md)，确认 `instruction / input / response` 字段、空回答、重复样本、长度分布和训练样本构造。然后回到 [01 SFT Data and Loss](./01_sft_data_and_loss.md)，把样本压成 `input_ids / attention_mask / labels` 这组三件套。

这一段的判断顺序是：

```text
原始记录 → 数据审计 → 训练样本 → response-only loss
```

如果这里没对齐，后面的 loss 曲线大概率没有解释力。

## 第二段：数据口径成立以后，再看 LoRA 挂载

数据能进训练以后，第二个问题通常不是“有没有 LoRA”，而是 LoRA 到底挂在什么地方、训练参数占比是否合理。这一段沿下面的线阅读：

- [09 SFT Training Loop](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)
- [10 LoRA Tutorial](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- [02 LoRA / PEFT Design](./02_lora_peft_design.md)

这里要确认 target modules、`r / alpha / dropout` 和可训练参数占比。需要比较不同 adapter 方案时，再进入 [31 LoRA Variants Theory](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.md) 和 `63`，不要一开始就把所有变体混在基础 LoRA 里。

## 第三段：进入训练前，先做 readiness 检查

LoRA 挂好以后，不要立即扩大训练规模。先看 [33 Fine-Tuning Readiness](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md)，确认模型、数据规模、训练目标、时间/显存预算和评测窗口是否写清楚。

这一步回答的是：

- 这项任务是否值得进入完整项目；
- 当前数据规模是否支撑目标；
- 预算是否允许完成计划；
- 评测是否能区分 baseline 和 candidate。

readiness 没通过时，应先缩小范围或修正计划，而不是继续调学习率。

## 第四段：训练能跑，不代表训练控制口径正确

readiness 通过后，再沿下面的线检查训练节奏：

- [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- [11 LR Schedulers WSD Cosine](../../02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.md)
- [03 Training Control](./03_training_control.md)

重点确认 micro-batch、effective batch、optimizer step、scheduler step 和日志步数是否一致。很多“能跑但结论不可信”的实验，问题都出在这里。

## 第五段：做端到端实验，确认结论是否成立

接着运行 [13 End-to-End Fine-Tuning Experiment](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)，并阅读 [04 End-to-End Experiment](./04_end_to_end_experiment.md)。这一段不只看 train loss，还要同时看：

- train / val 是否分开；
- val loss 和生成样例是否改善；
- 训练速度和显存代价是否可接受；
- baseline 与 candidate 是否使用相同 workload。

如果进入长上下文分支，再补 [30 Long Context Fine-Tuning](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md)，检查长度分布、fit rate、packing、峰值显存和 step time。

## 第六段：最后回到项目交付

基础实验成立后，进入 [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md) 和 [05 Project Delivery and Decision](./05_project_delivery_decision.md)。这时要回答的已经不是“能不能训”，而是：

- 数据是否可信；
- loss 口径是否正确；
- adapter、tokenizer、config 是否可交付；
- 训练结果是否值得采用；
- 别人能否按报告复现或继续使用。

完成核心项目后，再按目标进入扩展项目：`62` 指令微调、`63` LoRA 变体、`64` 数据质量和 `65` QLoRA 选型。`61` 属于 Task3 的结构验证扩展，不是综合微调项目的必经步骤。

把这条故事走完以后，一个更像真实交付的结论通常不是“我们完成了一次 SFT”，而是：数据口径正确、LoRA 挂载合理、训练控制一致、评测证据完整，最终产出的 adapter 和报告可以被别人复现和采用。
