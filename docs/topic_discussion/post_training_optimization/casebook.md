# 后训练优化判断手册 | Post-Training Casebook

先根据失败现象、训练信号和已有测量选择下一步，再由所需数据与更新方式定位算法。每次判断至少记录数据版本、模型版本、配置、工作负载、质量指标、资源消耗、失败样本、输出文件和最终决定。

| 现象或目标 | 先检查什么 | 至少记录 | 首选正文 | 继续验证 |
|:---|:---|:---|:---|:---|
| 模型格式不稳、损失与监督对象不一致 | 聊天模板、标签、掩码、切分与数据质量 | 数据版本、模板、有效监督词元、失败样本 | [01 SFT 数据与监督契约](./01_sft_data_and_loss.md) | [Part 02 · 32 数据准入](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) / [Part 02 · 64 数据质量项目](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) |
| 适配器无效、训练不可复现或有效批量不清 | 目标模块、秩、优化器步、调度器与评测集 | 可训练参数、有效批量、更新次数、随机种子、验证结果 | [02 LoRA / PEFT](./02_lora_peft_design.md)、[03 训练控制](./03_training_control.md) | [Part 02 · 62–63 SFT 项目链](../../02_PyTorch_Algorithms/2_10.md) / [Part 02 · 65 QLoRA](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md) |
| 训练完成但检查点、分词器或报告缺失 | 产物、恢复状态、样例输出与结果记录 | 模型或适配器、配置、分词器、清单、哈希、恢复记录 | [04 SFT 实验与交付](./04_end_to_end_experiment.md) | [Part 02 · 62 SFT 训练交付](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md) |
| 有偏好对，需要较轻的离线优化 | 优选 / 非优选回答质量、参考策略与独立评测 | 偏好对版本、参考策略、偏好边际、冻结评测结果 | [05 DPO](./05_dpo_and_preference_optimization.md) | [Part 02 · 17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.md) / [Part 02 · 84 DPO 项目](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.md) |
| 奖励分数提高，但隐藏质量、安全或鲁棒性下降 | 奖励数据切分、留出校准、奖励排序与隐藏评测 | 奖励模型版本、留出偏好、校准误差、代理奖励、隐藏质量 | [06 奖励信号、PPO / RLHF](./06_rlhf_and_ppo_system_cost.md)、[10 评测与决策](./10_evaluation_and_adoption_decision.md) | [Part 02 · PT-REWARD-MODEL](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md) |
| 需要轨迹采样、奖励或组内比较 | 奖励、优势、候选组、KL 与验证器 | 策略版本、轨迹身份、奖励分量、优势、更新约束 | [06 奖励信号、PPO / RLHF](./06_rlhf_and_ppo_system_cost.md)、[07 GRPO / RLVR](./07_grpo_and_groupwise_alignment.md) | [Part 02 · 16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md) / [Part 02 · 18 GRPO](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.md) |
| 在线轨迹无法进入当前更新 | 策略版本、环境版本、奖励状态和消费状态 | 版本差、轨迹身份、无法判定奖励、重复消费记录 | [08 轨迹生命周期](./08_trajectory_lifecycle_and_environment_interaction.md) | [Part 02 · PT-ROLLOUT-LIFECYCLE](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) |
| 在线更新变慢、昂贵或不可恢复 | 采样后端、工作进程拓扑、队列、资源与检查点 | 队列等待、阶段时间、同步、失败进程和恢复点 | [09 RL 执行系统](./09_rl_execution_system_and_efficiency.md) | [Part 02 · PT-RL-SYSTEM](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md) |
| 指标相互冲突 | 质量、安全、鲁棒性、成本和评审器一致性 | 固定基线与候选、各维度门槛、失败切片、回滚条件 | [10 独立评测、能力回归与采用决策](./10_evaluation_and_adoption_decision.md) | [Part 02 · 50 偏好评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) / [Part 02 · 51 在线回滚](../../02_PyTorch_Algorithms/51_Online_DPO.md) |
| 梯度为空、非有限值、更新边界或训练状态异常 | 计算图、保存状态、梯度累积、优化器步与混合精度 | 梯度状态、有效更新、缩放与跳步记录 | [反向传播与训练机制](../backpropagation_training_mechanism/intro.md) | [反向传播实验资产](../backpropagation_training_mechanism/experiment_assets.md) |
| 显存峰值、训练步或轨迹采样吞吐异常 | 状态生命周期、阶段时间线、通信与 I/O 等待 | 峰值显存、阶段耗时、吞吐、等待和瓶颈归因 | [性能优化](../performance_optimization/intro.md) | [性能分析判断手册](../performance_analysis/casebook.md) / [性能分析实验资产](../performance_analysis/experiment_assets.md) |
| 数据解析、血缘、去重或污染无法解释 | 数据来源、处理规则、切分、指纹与版本 | 来源许可、处理版本、样本身份、污染和质量记录 | [数据生命周期与契约](../data_engineering/00_data_lifecycle_and_contract.md)、[清洗与质量链](../data_engineering/02_cleaning_quality_chain.md) | [Part 02 · 32 数据准入](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) |
