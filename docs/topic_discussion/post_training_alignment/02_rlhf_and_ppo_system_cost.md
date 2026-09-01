# 02 RLHF 与 PPO 的系统代价

## 页面目标

这一页解释 RLHF / PPO 为什么是理解后训练全貌的重要基线，以及它为什么完整但重。

本页的输出是系统成本基线：reward model、rollout、policy 更新、reference 约束和评测分别增加了什么资源与工程复杂度。

## 问题起点

RLHF / PPO 不只是多一个 loss。它通常意味着：

- policy model
- reference model
- reward model
- rollout / sampling / update loop

也就是说，对齐训练不再是单纯的 supervised pass，而是一个更长的系统闭环。

## 核心矛盾

RLHF / PPO 试图更完整地表达“偏好优化”，但代价是：

- 训练对象变多
- rollout 成本上升
- 显存与通信压力更大
- 调参和评测链路更长

## 方法脉络

经典 RLHF / PPO 路线通常包含：

1. 基于 SFT 模型开始。
2. 构造偏好数据并训练 reward model。
3. 用 policy rollout 生成候选。
4. 用 PPO 类目标更新 policy，同时约束相对 reference 的偏离。

这条路线表达力强，但系统代价高，所以也为后续更轻的方法提供了参照系。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 关键取舍

| 问题 | 收益 | 代价 |
|:---|:---|:---|
| reward + policy 闭环 | 目标更完整 | 训练链路更长 |
| rollout | 更贴近在线行为 | 采样和评测成本高 |
| PPO update | 控制策略漂移 | 工程复杂度高 |

## 常见误区

- 只记住“PPO 很重”，但说不清重在哪里。
- 把 RLHF / PPO 看成后训练唯一正统路线。
- 只看方法名，不看 rollout / reward / policy 这几类对象如何叠加成本。

## 对应 Part 02

- `14 RLHF PPO Memory`：这一页的主要 notebook 来源。
- `17 / 19`：如果要解释 backward 和显存，回到训练机制专题。
- `84 / 85`：项目页用来对照“更轻路线”如何落地。

## 文献锚点

- InstructGPT / RLHF 基线论文。
- PPO 原始论文或高质量综述。
- 工程资料：rollout、KL penalty、policy/reference 边界说明。

## 本节要点

RLHF / PPO 是后训练“完整但重”的基线。理解它，后面才能真正看出 DPO 和 GRPO 在简化什么、替代什么。

## 进入下一页

如果不需要显式 reward / rollout 闭环，进入 [03 DPO 与偏好优化](./03_dpo_and_preference_optimization.md)；如果核心数据不是 pairwise，而是候选组，再继续比较 GRPO。
