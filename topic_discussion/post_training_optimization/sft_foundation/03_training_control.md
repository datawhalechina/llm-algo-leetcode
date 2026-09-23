# 03. Training Control | 训练控制

## 页面目标

这一页回答的是：SFT 训练不是只看 loss，还要统一 scheduler、optimizer step 和 gradient accumulation 的口径。

## 你要先确认什么

- scheduler 的步数是不是按 optimizer update 计。
- gradient accumulation 是否改变了 effective batch。
- optimizer step 和日志步数是否对齐。
- 学习率曲线是否和训练长度匹配。

## 演化路径

训练控制的核心是把“每个 micro-batch 做什么”与“每次真正更新参数做什么”拆开。

1. micro-batch 负责前向和累积梯度。
2. accumulation 把多个 micro-batch 合成一个 effective batch。
3. optimizer step 只在累积完成后执行。
4. scheduler 按真正的 update step 推进。
5. 训练日志要和这套口径完全一致。

如果这里的步数口径乱了，后面的实验报告再漂亮也不可信。

## 常见误区

- 把 micro-batch 当成 effective batch。
- scheduler 每个 batch 都 step，和 accumulation 冲突。
- 训练日志和更新步数不一致。
- 只记录 loss，不记录学习率、batch 和更新节奏。

## 经典阅读入口

- [11 Optimizer Updates and Learning Rate Scheduling](../../../02_PyTorch_Algorithms/11_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb)
- [12 Gradient Accumulation](../../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
- [13 End-to-End Fine-Tuning Experiment](../../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)

## 前置关系

- 先看 `01` 和 `02`，确认数据和 LoRA 都已经对齐。
- 再看 `03`，否则 scheduler 和 accumulation 的关系很容易混掉。

## 本节要点

训练控制的作用不是“让训练跑起来”，而是让训练按正确的节奏跑起来。
这里的口径不统一，后面的实验结论就不稳。
