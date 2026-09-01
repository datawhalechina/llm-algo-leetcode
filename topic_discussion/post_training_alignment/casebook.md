# 后训练优化正文

这页只做对齐问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 使用顺序

先确认问题确实超出 SFT，再比较 PPO、DPO、GRPO 的训练闭环；随后审计偏好数据和评测口径，最后回到项目交付。不要只按 loss 或方法流行度选择路线。

## 判断表

先分清问题在方法选择、偏好数据、评测口径还是项目交付，再判断它是不是已经退化成训练或系统代价问题。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| SFT 后结果仍然偏离偏好 | `alignment gap` | [01](./01_why_post_training_alignment.md) | 先确认是不是对齐问题而不是 SFT 基础问题 |
| 方法很强，但系统代价太重 | `ppo system cost` | [02](./02_rlhf_and_ppo_system_cost.md) | 看 rollout、reward、显存和多卡代价 |
| 偏好数据有了，但结论不稳 | `dpo / data mismatch` | [03](./03_dpo_and_preference_optimization.md), [05](./05_preference_data_and_evaluation.md) | 看 chosen / rejected 和评测口径 |
| group-wise 路线不稳定 | `grpo mismatch` | [04](./04_grpo_and_groupwise_alignment.md) | 看候选组构造和比较口径 |
| 项目页能跑，但采用建议站不住 | `delivery gap` | [06](./06_project_decision_and_delivery.md) | 回到方法、数据、评测一起收口 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| 方法选择 | 当前是 PPO、DPO 还是 GRPO 问题 | 只按流行度选方法 |
| 偏好数据 | chosen / rejected 或 group 数据是否可信 | 只要有 pair 就能训练 |
| 评测口径 | 结果是不是和目标行为一致 | 只看 loss，不看偏好评测 |
| 项目收口 | 是否能落成 adopt / tune / reject | 能跑就等于值得上 |

## 本节要点

这页的职责不是再讲一遍对齐方法名，而是把后训练里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。

## 最小决策模板

记录 `SFT 基线 -> 对齐缺口 -> 方法与数据形态 -> 评测指标 -> 系统成本 -> adopt / tune / reject`。训练指标、偏好指标和在线指标不一致时，必须明确说明差异。
