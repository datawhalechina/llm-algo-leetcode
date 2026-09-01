# 后训练优化（Post-Training Optimization）

> 专题类型：主学习路线　主服务目标：偏好对齐与训练交付

## 页面导语

本教程的 V1 版本是[监督微调与训练工程专题](../fine_tuning_training/intro.md)：重点建立模型结构、SFT、LoRA、数据工程和训练项目的基础。V2 版本在 V1 的训练结果之上继续处理偏好数据、奖励建模、DPO、GRPO 和在线对齐，因此主入口升级为“后训练优化”。

V1 并未被替代，而是作为 V2 的前置支撑；学习者可以先完成 V1 的 SFT / LoRA 基础，再进入本专题，也可以直接从已有的监督微调经验开始。60 是两版之间的接口项目，84–86 承担 V2 的主要项目闭环。

本专题串起 SFT 之后的后训练主线：先看为什么模型在 SFT 之后还会出现偏好错位，再看 PPO、DPO、GRPO 分别改的是哪一层问题，最后把偏好数据、评测和项目结论收成同一条对齐闭环。后训练以 Infra-L3 的损失、采样、reference model 和训练运行时为核心；在线采样、推理服务和反馈回路才延伸到 Infra-L4，Infra-L5 负责数据生命周期、实验编排、评测回归和交付治理。

Infra-L1/Infra-L2 的显存、算子和通信成本会直接影响 PPO、DPO、GRPO 的可行性，因此最终结论不能只看对齐指标，还要记录吞吐、显存、稳定性和服务代价。若还没有 SFT 基础，应先回[监督微调与训练工程](../fine_tuning_training/intro.md)；若重点转为单模型推理或显存预算，应转到对应专题。

## 如何开始

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的后训练优化路线进入；监督微调基础可由[监督微调与训练工程专题](../fine_tuning_training/intro.md)补齐，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 60、84、85、86 项目节。本专题适合按问题选择 DPO、GRPO 或在线流程，不要求先完整学习 PPO。

### 前置阅读

建议先掌握监督微调与训练工程中的数据、训练循环和 LoRA 基础，再阅读 [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) 以及相关的偏好数据与对齐内容。进入项目前至少明确 preference pair / group 数据格式、reference model、奖励或偏好指标，以及训练和评测的隔离方式。

## 主学习线与分级

`Task0-6` 是学习路线，指向 `Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。Task0 负责把监督微调接到后训练，Task1-5 建立方法与数据基础，Task6 完成项目决策。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task0 | SFT 到后训练的接口 | [Part 02 · 09 SFT 训练循环](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb) → [Part 02 · 10 LoRA 教程](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.ipynb) → [Part 02 · 60 LoRA 微调项目](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb) | [01 为什么需要后训练优化](./01_why_post_training_alignment.md) |
| Task1 | 为什么 SFT 之后还要对齐 | [Part 02 · 14 RLHF 与 PPO 显存](../../02_PyTorch_Algorithms/14_RLHF_PPO_Memory.ipynb) | [01 为什么需要后训练优化](./01_why_post_training_alignment.md) |
| Task2 | RLHF / PPO 的系统代价 | [Part 02 · 14 RLHF 与 PPO 显存](../../02_PyTorch_Algorithms/14_RLHF_PPO_Memory.ipynb) | [02 RLHF 与 PPO 的系统成本](./02_rlhf_and_ppo_system_cost.md) |
| Task3 | DPO 与偏好优化 | [Part 02 · 15 DPO 损失](../../02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.ipynb) → [Part 02 · 84 DPO 偏好项目](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb) → [Part 02 · 86 DPO 在线基准](../../02_PyTorch_Algorithms/86_DPO_Online_Benchmark.ipynb) | [03 DPO 与偏好优化](./03_dpo_and_preference_optimization.md) |
| Task4 | GRPO 与 group-wise 对齐 | [Part 02 · 16 GRPO 损失](../../02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.ipynb) → [Part 02 · 85 GRPO 组内对齐项目](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb) | [04 GRPO 与组内对齐](./04_grpo_and_groupwise_alignment.md) |
| Task5 | 偏好数据、在线优化与冲突评测 | [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) → [Part 02 · 51 在线 DPO](../../02_PyTorch_Algorithms/51_Online_DPO.ipynb) → [Part 02 · 52 对齐冲突与阈值](../../02_PyTorch_Algorithms/52_Alignment_Conflicts_and_Thresholds.ipynb) | [05 偏好数据与评测](./05_preference_data_and_evaluation.md) |
| Task6 | 项目收口与采用建议 | [Part 02 · 60 LoRA 微调项目](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb) → [Part 02 · 84 DPO 偏好项目](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb) → [Part 02 · 85 GRPO 组内对齐项目](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb) → [Part 02 · 86 DPO 在线基准](../../02_PyTorch_Algorithms/86_DPO_Online_Benchmark.ipynb) | [06 项目决策与交付](./06_project_decision_and_delivery.md) |

### 核心与扩展分级

核心路径先完成 SFT 接口、PPO / DPO / GRPO 的基本机制，以及离线偏好数据和评测；扩展路径再进入在线 DPO、冲突评测和完整的 84–86 项目闭环。扩展不改变主线目标，只增加数据规模、服务链路和决策证据。

## 与其他专题的边界

先按上面的 `Task0-6` 走 Notebook 主线；遇到“为什么 SFT 还不够”“PPO / DPO / GRPO 到底差在哪一层”时，再回来看对应的专题正文。想看汇总版就进 [后训练优化正文](./casebook.md)，想按连续故事线走一遍就进 [后训练优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[监督微调与训练工程](../fine_tuning_training/intro.md) 负责 SFT 前置，[反向传播与训练机制](../backpropagation_training_mechanism/intro.md) 负责 alignment loss 的 backward 代价，[显存优化](../memory_performance_tuning/intro.md) 负责 PPO 等系统成本。

## 学习方式与项目产出

推荐的实践闭环是 `60 SFT/LoRA 基线 -> 84 DPO 偏好项目 -> 85 GRPO 组内对齐项目 -> 86 在线 DPO benchmark`。学习者最终应比较数据质量、训练稳定性、偏好或任务评测、显存与吞吐，而不是只比较训练 loss；在线项目还要额外记录采样、打分和更新链路的成本。

## 环境与证据边界

损失函数、数据格式和小规模离线验证可先用 CPU；完整 SFT、DPO、GRPO 训练通常需要 GPU，在线 benchmark 还需要可用的推理后端或服务接口。应固定数据切分、随机种子、reference 配置和评测集，并保存训练配置、指标曲线和最终决策。
