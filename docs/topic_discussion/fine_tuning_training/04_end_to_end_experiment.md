# 04. End-to-End Experiment | 端到端实验

## 页面目标

这一页回答的是：SFT 闭环是否真的跑通，以及跑通之后能不能拿出可解释的实验结果。

它承接 [32 SFT 数据工程](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) 和 [33 Fine-Tuning Readiness](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md) 的项目准备，也承接 [13 End-to-End Fine-Tuning Experiment](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md) 的实际运行，最后把结果交给 [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md) 做项目交付。

## 你要先确认什么

- train / val 是否分开。
- 评估是否周期性触发。
- 样例生成是否真的发生变化。
- 显存和速度是否在预期范围内。

## 演化路径

端到端实验不是只看最终 loss，而是要把训练过程变成可验证的实验报告。

1. 先确认 32 已完成字段清洗、异常审计和训练样本构造。
2. 再确认 33 已明确目标、预算、数据规模和评测窗口。
3. 再确认训练是否能稳定推进，并核对 13 的训练控制口径。
4. 再看验证曲线和生成样例是否有实际变化。
5. 最后把结果整理成可复盘、可交付的实验结论。

这一页的重点是“闭环”而不是“数值本身”。

## 常见误区

- 只看 train loss，不看 val 和样例。
- 验证集样本太少，导致判断失真。
- 训练跑完了，但没有保存最小可复现实验信息。
- 结果看起来下降了，但无法说明模型真的变好了。

## 经典阅读入口

- [13 End-to-End Fine-Tuning Experiment](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)

## 前置关系

- 先看 `32 -> 33`，确认数据和项目准入条件。
- 再看 `01-03`，把监督口径、LoRA 和训练控制对齐。
- 再看 `04`，实验结论才有基础。

## 与项目路线的关系

- `13` 是端到端训练实验，重点是确认训练闭环和评测口径。
- `60` 是正式 LoRA 交付项目，重点是 artifact、报告和采用决策。
- `30` 长上下文微调属于扩展路径，需要额外检查长度分布、fit rate、显存和 step time。

## 本节要点

端到端实验的价值不是“证明模型能训”，而是把训练结果变成能复盘、能比较、能交付的证据。
