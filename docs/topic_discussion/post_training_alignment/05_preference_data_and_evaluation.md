# 05 偏好数据与对齐评测

## 页面目标

这一页负责把方法页里的目标拉回到数据和评测口径上，解释为什么对齐问题常常不是 loss 写错，而是数据和评测先出了偏差。

本页的输出是评测契约：数据格式、标签质量、reference / candidate 口径和离线指标必须共同支持最终的偏好结论。

## 问题起点

不管是 PPO、DPO 还是 GRPO，最后都绕不开两个问题：

- 偏好数据到底长什么样？
- 结果到底该用什么指标判断？

如果这两步口径不稳，方法结论通常也站不住。

## 数据形态

常见数据形态包括：

- `chosen / rejected`：适合 pairwise preference optimization
- `group candidates`：适合 group-wise 比较
- `judge score / rubric`：适合更细的评价口径

它们不是同义替换，而是对应不同方法和评测方式。

## 评测口径

常见评测包括：

- `win-rate`
- `pairwise accuracy`
- `judge score`
- 任务相关质量指标

关键不是指标越多越好，而是**训练目标、数据形态和评测口径是否一致**。

## 关键取舍

| 问题 | 你要得到什么 | 常见代价 |
|:---|:---|:---|
| 偏好对清洁度 | 可比较、可解释的 preference pair | 标注成本高、噪声大 |
| 候选组质量 | group-wise 排序更可信 | 候选来源复杂、稳定性差 |
| 统一评测 | 结果能支持 adopt / tune / reject | 口径统一难 |

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 常见误区

- 数据能跑就行，不检查偏好标签是否可靠。
- 训练用 pairwise，评测却只看不相关指标。
- judge score、win-rate、人工评审结果并列摆出，却没有统一解释。

## 对应 Part 02

- `50 Preference Data and Evaluation`：本页的主要 notebook 来源。
- `84 / 85`：项目页验证这些口径是否真能支撑结论。

## 文献锚点

- 偏好数据构造与清洗相关论文。
- LLM-as-a-judge / judge consistency 相关论文。
- 对齐评测基准与 win-rate 使用边界说明。

## 本节要点

对齐方法能否成立，最终常常取决于偏好数据和评测口径是否同向，而不是某个 loss 公式本身。

## 进入下一页

把方法、数据和评测证据交给 [06 项目决策与交付](./06_project_decision_and_delivery.md)，再判断是否进入更长训练或在线验证。
