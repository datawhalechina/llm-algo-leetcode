# 后训练优化深入阅读

假设你已经把一条 SFT 路线跑通，但模型在真实偏好上仍然不稳：回答可能安全性不足、风格不对、或者 pairwise 偏好始终拉不开。接下来你会开始问：到底是继续 RLHF / PPO，还是转去 DPO / GRPO，更轻的路线能不能成立。

这条线最重要的是按暴露顺序判断：先确认这是不是 SFT 之后的真实对齐问题，再分清方法差异，最后回到偏好数据、评测和项目结论。

对应专题正文：[01 为什么 SFT 之后还要继续做对齐](./01_why_post_training_alignment.md)。先证明问题已经超出基础 SFT 范围。

## 第一段：先确认问题已经超过 SFT

故事通常从“模型已经会做任务，但输出仍然不够符合偏好”开始。第一步要先分清：这是数据或训练基础问题，还是已经进入对齐问题。

这一步对应 [01 为什么 SFT 之后还要继续做对齐](./01_why_post_training_alignment.md)。

## 第二段：PPO、DPO、GRPO 不是同一层方法

一旦确认需要后训练，第二步就不是直接选最流行的方法，而是分清它们改的是哪一层：PPO 带完整 RLHF 系统代价，DPO 更像偏好监督，GRPO 更适合 group-wise 比较。

这一步对应 [02 RLHF 与 PPO 的系统代价](./02_rlhf_and_ppo_system_cost.md)、[03 DPO 与偏好优化](./03_dpo_and_preference_optimization.md) 和 [04 GRPO 与 Group-wise 对齐](./04_grpo_and_groupwise_alignment.md)。

## 第三段：方法差异最后都要落到偏好数据和评测

真正决定结论能不能站住的，往往不是方法名，而是 chosen / rejected、group candidates 和评测口径是否可信。没有这一步，方法比较很容易失真。

这一步对应 [05 偏好数据与对齐评测](./05_preference_data_and_evaluation.md)。

## 第四段：最后回到项目收口

真正的闭环不在“我们实现了某个对齐算法”，而在项目页能不能把方法、数据、评测和系统代价一起收成 adopt / tune / reject。把这条故事走完以后，一个更像真实结论的说法通常不是“我们做了 DPO”，而是：当前任务为什么需要对齐、为什么选这条方法线、数据和评测为什么足以支持最终结论。

这一步对应 [06 项目决策与交付](./06_project_decision_and_delivery.md)，并回到 `84 / 85 / 86` 项目页验证。
