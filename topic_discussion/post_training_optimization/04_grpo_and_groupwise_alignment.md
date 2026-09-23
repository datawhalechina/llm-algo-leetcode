# 04 GRPO 与 Group-wise 对齐

## 页面目标

这一页回答的是：当偏好关系不止是 chosen / rejected 二选一时，为什么需要 group-wise 候选和相对比较，以及这种表达能力换来了什么系统代价。

本页的输出是 GRPO 适用边界：候选组是否有信息量、组内比较是否稳定，以及采样和评测成本是否值得。

这一页解释 GRPO 为什么会出现、它和 pairwise preference optimization 有什么不同，以及它更适合什么场景。

## 问题起点

有些对齐场景里，问题不是“chosen 和 rejected 二选一”，而是：

- 同一 prompt 下有多个候选
- 模型质量更像相对排序，而不是简单二元偏好
- 任务更偏生成式、探索式或候选集比较

这就是 group-wise 路线出现的背景。

## 核心矛盾

GRPO 想利用组内相对信息，但代价是：

- 候选组构造更难
- 评测解释更复杂
- 训练稳定性更依赖候选质量

## 演化逻辑

可以把 GRPO 理解成从 pairwise 走向 group-wise 的延伸：

1. 不再只比较一个 chosen 和一个 rejected。
2. 对同一 prompt 采集一组候选。
3. 通过组内相对比较学习更稳定的偏好方向。

它特别适合那些“好坏是相对排序，不是单一标准答案”的任务。

## 关键取舍

| 取舍 | 收益 | 代价 |
|:---|:---|:---|
| group-wise 信息 | 利用更多候选关系 | 构造和评测更复杂 |
| 生成式比较 | 更贴近某些开放式任务 | 结果更依赖候选多样性 |

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 常见误区

- 把 GRPO 只理解成“DPO 的另一个名字”。
- 只看组内 loss，不看候选生成质量。
- 组内样本不稳定，却直接比较最终指标。

## 对应 Part 02

- `16 GRPO Loss Tutorial`：本页的主要 notebook 来源。
- `50 Preference Data and Evaluation`：需要一起看 group candidates 和评测。
- `85 GRPO Groupwise Alignment Project`：在项目页验证方法收益。

## 文献锚点

- GRPO / relative preference / groupwise optimization 相关论文。
- 候选生成与组内评测的工程资料。

## 本节要点

GRPO 的重点不只是一个新 loss，而是把“偏好优化对象”从 pair 扩展到 group；真正困难的部分，往往在候选构造和评测解释上。

## 进入下一页

将 DPO / GRPO 的数据形态统一交给 [05 偏好数据与对齐评测](./05_preference_data_and_evaluation.md)，检查训练目标和评测口径是否一致。
