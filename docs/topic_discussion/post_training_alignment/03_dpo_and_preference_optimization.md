# 03 DPO 与偏好优化

## 页面目标

这一页解释 DPO 为什么成为更轻量的偏好优化路线，以及它依赖什么样的数据和前置。

本页的输出是 DPO 适用边界：偏好 pair 是否可靠、reference model 和训练预算是否明确，以及去掉 reward model 后哪些风险转移到了数据和评测。

## 问题起点

如果 RLHF / PPO 的系统代价太高，一个自然问题就是：  
能不能直接在偏好对上做优化，而不把 reward model 和 rollout 全部显式引入？

DPO 的吸引力就在这里。

## 核心矛盾

DPO 试图降低方法链路复杂度，但代价是：

- 更依赖高质量的 chosen / rejected 对
- 更依赖 reference 口径和数据构造
- 更容易被误解成“只是另一种 SFT”

## 演化逻辑

DPO 的直觉是：

1. 已有一个 SFT / reference 基线。
2. 对同一 prompt，有一个更偏好的 answer 和一个较差的 answer。
3. 直接优化模型让更优答案相对更可能。

这样就把“偏好优化”从完整 RLHF 闭环收缩成“偏好对上的直接目标”。

## 关键取舍

| 取舍 | 收益 | 代价 |
|:---|:---|:---|
| 去掉完整 RLHF 闭环 | 训练更轻、更容易接入已有 LoRA / SFT 流程 | 偏好对质量要求更高 |
| 直接优化 preference pair | 方法更直接 | 对 reference 和数据分布更敏感 |

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 常见误区

- 把 DPO 当成“更简单的 SFT”。
- 只看 loss 形式，不看数据是否真的是可比的 preference pair。
- 忽略 reference model 在稳定优化中的作用。

## 对应 Part 02

- `15 DPO Loss Tutorial`：本页的主要 notebook 来源。
- `50 Preference Data and Evaluation`：需要配合一起看数据与评测。
- `84 DPO Preference Project`：用项目页验证是否真的值得采用。

## 文献锚点

- DPO 原始论文。
- DPO 后续变体与实践资料。
- 偏好数据构造与参考模型使用边界说明。

## 本节要点

DPO 的价值不在于“更简单”，而在于它把偏好优化压缩成一条更短的训练链路；代价则转移到了数据质量和评测解释上。

## 进入下一页

如果任务需要利用同一 prompt 下多个候选的相对关系，进入 [04 GRPO 与 Group-wise 对齐](./04_grpo_and_groupwise_alignment.md)；否则带着 DPO 候选进入数据与评测审计。
