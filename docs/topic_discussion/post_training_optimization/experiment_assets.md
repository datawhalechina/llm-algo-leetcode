# 后训练优化实验资产表

本表登记后训练机制与项目的环境、固定输入、结果和交接状态。安装与续装见[实验 Quickstart](./quickstart.md)，公共结果字段见[结果资产规范](../../part02_result_assets.md)。正式记录、历史记录及未完成项分开说明，原始 JSON 和日志保持不变。

## 环境与实验入口

模型缓存可以共享，各 Profile 的 Python 依赖分别隔离。最低硬件条件和环境预检以 Quickstart 及 Profile 为准；预检通过不代表实验通过。

| 环境与安装入口 | 适用入口 | 验证范围 |
|:---|:---|:---|
| [PT-CPU](./quickstart.md#pt-cpu-quickstart) | 13–18、30–33、50–51、相关 `PT-*` 与 DISTILL-OPD 的 Step 4 和框架核对 | 数据接口、标签、损失、适配器、偏好、奖励、策略新鲜度与阶段输入 / 输出字段 |
| [PT-SFT-CUDA](./quickstart.md#pt-sft-cuda-quickstart) | 14、15、30、31；62–64 | 真实 LoRA/SFT 更新、长上下文训练代价、LoRA 变体路径、验证集与训练产物 |
| [Q-QLORA](../quantization_compression/quickstart.md#q-qlora-quickstart) | 26、65 | NF4 QLoRA、适配器与低资源训练证据 |
| [PT-RL-CUDA](./quickstart.md#pt-rl-cuda-quickstart) | 17、18、PT-REWARD-MODEL、PT-REWARD-SIGNAL；84、85 的单卡阶段 | 偏好、组内奖励、奖励头与信号构造的单卡对照 |
| [PT-VERL-CUDA](./quickstart.md#pt-verl-cuda-quickstart) | PT-ROLLOUT-LIFECYCLE、PT-RL-SYSTEM | 一次受控更新、固定轨迹重评、权重同步与阶段资源观察 |
| [PT-DISTILL-CUDA](./quickstart.md#pt-distill-cuda-quickstart) | DISTILL-OPD；88 的教师—学生配对阶段 | 学生基线、固定轨迹与学生在策略蒸馏、学生产物与部署比较 |
| 后续多工作进程环境（暂未验证） | 在线系统扩展 | 轨迹队列、网络通信、跨节点策略同步和多工作进程恢复；不由 `PT-VERL-CUDA` 单卡证据替代 |

## 机制实验

先检查数据与监督，再按参数适配、离线偏好或在线奖励分支选择实验。学习型奖励需要奖励模型及信号检查，规则奖励按需跳过奖励模型；框架核对与单卡实测分别登记，系统两页只执行一份共享 GPU 结果。

| 前置 / 分支 | 机制页与实验 | Profile 与固定输入 | 比较变量或检查对象 | 正式结果与范围 |
|:---|:---|:---|:---|:---|
| 共同入口 | [Part 02 · 33 后训练准入与阶段选择](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md) | `PT-CPU`；题目、答案测试与框架核对 | 用稳定失败、非参数基线、SFT 基线状态和反馈形态选择后训练阶段 | CPU |
| SFT 主路径 | [Part 02 · 32 SFT 数据契约与泄漏审计](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) | `PT-CPU`；题目、答案测试与框架核对 | 检查必需字段、任务输入重复、完整样本冲突和跨切分泄漏 | CPU |
| 数据扩展 | [Part 02 · PT-SYNTHETIC-DATA 合成指令生成与验证准入](../../02_PyTorch_Algorithms/PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.md) | `PT-CPU`；题目、答案测试与框架核对 | 检查生成身份、去重、污染与验证器准入 | CPU 练习与 Datasets 核对；GPU 结果未在此登记 |
| SFT 主路径 | [Part 02 · 13 SFT 监督边界与损失](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md) | `PT-CPU`；题目、答案测试与框架核对 | 把 聊天模板、标签与有效监督词元进入损失 | CPU；真实训练在 Part 02 · 15 中执行 |
| SFT 主路径 | [Part 02 · 14 LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.md)<br>`pt-lora-targets` | `PT-SFT-CUDA`；Qwen2.5-0.5B；三组目标模块 | 目标模块、冻结、合并 | [结果 JSON](../../../benchmarks/results/pt_lora_targets/pt-lora-targets_rtx5070ti_lora_targets_v3.json)；`rtx5070ti_lora_targets_v3` 已完成；训练机制通过，最高峰值分配约 2.02 GiB；`attention_qv` 的 BF16 合并保持 `tune` |
| SFT 主路径 | [Part 02 · 15 SFT 有效更新与训练闭环](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.md)<br>`pt-sft-training` | `PT-SFT-CUDA`；固定训练 / 验证切分 | 有效更新与评测隔离 | [结果 JSON](../../../benchmarks/results/pt_sft_training/pt-sft-training_rtx5070ti_sft_v1.json)；`rtx5070ti_sft_v1` 已通过；峰值分配约 4.66 GiB，`accept_mechanism` |
| 低资源分支 | [Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md) | `Q-QLORA`；冻结 NF4 基座与适配器 | 解释 NF4、冻结基座与低资源适配器更新 | 项目比较见下方 QLoRA 分支 |
| Task1 长上下文扩展 | [Part 02 · 30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md)<br>`pt-long-context` | `PT-SFT-CUDA`；三档长度与同一组样本 | 填充 / 边界隔离 Packing | [结果 JSON](../../../benchmarks/results/pt_long_context/pt-long-context_rtx5070ti_length_packing_v4.json)；`rtx5070ti_length_packing_v4` 已通过；最高峰值分配约 2.90 GiB，`accept_mechanism` |
| 参数适配分支 | [Part 02 · 31 LoRA 变体](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.md)<br>`pt-lora-variants` | `PT-SFT-CUDA`；固定秩与固定参数预算 | LoRA / rsLoRA / DoRA | [结果 JSON](../../../benchmarks/results/pt_lora_variants/pt-lora-variants_rtx5070ti_lora_variants_v3.json)；`rtx5070ti_lora_variants_v3` 已完成，最高峰值分配约 1.97 GiB；前三组通过，固定预算 `dora_r7` 因合并输出等价性未达阈值而保持 `tune` |
| 偏好共同基础 | [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) | `PT-CPU`；题目、答案测试与框架核对 | 固定偏好对、数据隔离和独立评测协议 | CPU |
| 离线偏好分支 | [Part 02 · 17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.md)<br>`pt-dpo-real-model-update` | `PT-RL-CUDA`；Qwen2.5-0.5B；4 组偏好对、4 次更新 | β 为 0.05 / 0.10 / 0.30 | [结果 JSON](../../../benchmarks/results/pt_dpo/pt-dpo-real-model-update_rtx5070ti_dpo_v1.json)；`rtx5070ti_dpo_v1` 已通过；最高峰值分配约 3.19 GiB，`accept_mechanism` |
| 离线反馈分支 | [Part 02 · PT-KTO 二元反馈对齐](../../02_PyTorch_Algorithms/PT-KTO_Binary_Feedback_Alignment.md) | `PT-CPU`；题目、答案测试与框架核对 | 二元标签、独立 KL 样本与类别加权 | CPU 练习与 TRL 核对；未登记独立 GPU 结果 |
| 在线奖励分支 | [Part 02 · PT-RLOO 留一基线策略优化](../../02_PyTorch_Algorithms/PT-RLOO_Leave_One_Out_Policy_Optimization.md) | `PT-CPU`；题目、答案测试与框架核对 | 可评分组、留一优势与序列裁剪 | CPU 练习与 TRL 核对；未登记独立 GPU 结果 |
| 奖励优化共同基础 | [Part 02 · PT-REWARD-MODEL 奖励模型训练、校准与过度优化](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md)<br>`pt-reward-model-head-calibration` | `PT-RL-CUDA`；Qwen2.5-0.5B；6 组训练、4 组校准、4 组隐藏偏好对 | 正确标签 / 50% 错标负对照 | [结果 JSON](../../../benchmarks/results/pt_reward_model/pt-reward-model-head-calibration_rtx5070ti_reward_model_v1.json)；`rtx5070ti_reward_model_v1` 已通过；正确标签候选隐藏排序准确率 1.0，错标负对照为 0.5；峰值分配约 0.95 GiB，`accept_mechanism` |
| 奖励优化共同基础 | [Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md)<br>`pt-reward-signal-matrix` | `PT-RL-CUDA`；固定提示、回复与奖励 | 约束强度、终止奖励和回报；6 组候选 | [结果 JSON](../../../benchmarks/results/pt_reward_signal/pt-reward-signal-matrix_rtx5070ti_reward_signal_v3.json)；`rtx5070ti_reward_signal_v3` 已通过；峰值分配约 2.01 GiB，`accept_mechanism` |
| PPO 分支 | [Part 02 · 16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md) | `PT-CPU`；题目、答案测试与框架核对 | 解释策略、参考策略、奖励模型、价值模型与资源状态 | CPU 练习与 TRL 核对；尚无本页完整 PPO GPU 更新结果登记 |
| GRPO / RLVR 分支 | [Part 02 · 18 GRPO 与可验证奖励](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.md)<br>`pt-grpo-real-model-update_default3` | `PT-RL-CUDA`；Qwen2.5-0.5B；2 个提示、4 次更新 | 最小 / 扩展候选覆盖；integer-answer-v1 | [结果 JSON](../../../benchmarks/results/pt_grpo/pt-grpo-real-model-update_default3.json)；v3 `default3` 已通过；最高峰值分配约 3.06 GiB，`accept_mechanism`；扩展候选组暴露较高裁剪率，不用于自动选择组规模 |
| 在线系统共同基础 | [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) | `PT-VERL-CUDA`；共同算术轨迹与一次监督更新 | 管理轨迹身份、策略版本、新鲜度与准入 | 读取[共享系统记录](#verl-共享系统记录) |
| 在线系统 | [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md) | `PT-VERL-CUDA`；共同算术轨迹与一次监督更新 | 阶段处理率、共享链路、状态释放与恢复 | 读取[共享系统记录](#verl-共享系统记录) |
| 在线偏好分支 | [Part 02 · 51 在线 DPO](../../02_PyTorch_Algorithms/51_Online_DPO.md) | `PT-CPU`；题目、答案测试与框架核对 | 在线偏好反馈、候选采用与回滚 | CPU 练习与框架核对；verl 系统实验不替代在线 DPO 验证 |
| 多轮工具扩展 | [Part 02 · PT-AGENTIC-RL 环境轨迹与奖励归因](../../02_PyTorch_Algorithms/PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md) | `PT-CPU`；题目、答案测试与框架核对 | 环境动作、观察、终止和奖励归因 | CPU 练习与环境接口核对；未登记完整 Agent 策略更新 GPU 结果 |
| 蒸馏分支 | [Part 02 · DISTILL-OPD 在策略蒸馏与 GKD](../../02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.md) | `PT-CPU`；固定轨迹与学生轨迹、冻结教师 | 固定轨迹、学生轨迹、教师分布与梯度隔离 | CPU 练习与框架核对；GPU 训练由下方蒸馏项目登记 |

<a id="verl-共享系统记录"></a>

### verl 共享系统记录

[受控更新 v2](../../../benchmarks/results/pt_verl_closure/rtx5070ti_verl_mechanism_closure_v2.json) 是轨迹页和系统页共同使用的单卡机制结果。两组使用相同初始参数与冻结轨迹，分别测试引擎释放和保留；verl 策略工作进程 以监督损失执行一次 LoRA 更新，再重评旧轨迹并同步权重。该记录验证参数变化、轨迹准入与阶段状态，不代表 GRPO 质量收益、多进程原子激活或稳态加速。历史流程探测、奖励筛查和 v1 配对缺陷保留在本页末尾的折叠记录中。

| 检查对象 | v2 直接观察 | 解释范围 |
|:---|:---|:---|
| 参数与策略身份 | 两组初始哈希和更新后哈希分别一致；各有 48 个参数张量改变 | 相同起点的一次有效监督更新 |
| 概率与准入 | 更新后同版本最大对数概率误差 0.06250；两组旧轨迹均 7 条准入、1 条拒绝 | 固定轨迹、固定阈值下的重评与同步 |
| 更新阶段资源 | 整卡采样峰值：释放 6392、保留 10489 MiB；耗时 1.367、1.345 s | 单次阶段观测，不作为性能排序 |
| 整段资源 | 释放 10086、保留 10489 MiB；短阶段未采到的峰值为缺失 | 包含初始化和其他应用，差值不等于 KV Cache 大小 |

## 项目实验

监督和偏好 / 奖励项目按表中前置衔接；蒸馏从冻结教师与独立学生起点进入。64 与 86 的直接输入核对不依赖 CUDA；88 的终止判断与 Part 05 的 86 交接链分开登记。

| 顺序 | 项目入口 | 固定输入与前置 | 验证对象 | 正式结果与状态 | 输出与交接 |
|:---|:---|:---|:---|:---|:---|
| 1 | [Part 02 · 64 SFT 数据准入](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) | 原始数据版本、冻结评测切分 | 阻断项合并判断、跨切分泄漏检查、`accept / tune / reject` 状态转换 | [alpaca_qwen25_v2](../../../benchmarks/results/64_sft_data_quality/sft_data_admission_alpaca_qwen25_v2.json) 已完成：511 条有效记录、截断率 1.17%、跨切分重合为 0，结论 `accept`；CPU 输入检查 | 数据准入 清单、样本 ID 与切分哈希 |
| 2 | [Part 02 · 63 LoRA 变体选型](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.md) | 64 准入清单；Qwen2.5-0.5B；秩 4 / 8 / 16、3 个共享种子 | 单变量可比性、质量/显存硬门槛、可行候选选择 | [lora_variant_selection_v2](../../../benchmarks/results/63_lora_variants/lora_variant_sweep_lora_variant_selection_v2.json) 已完成 3 个秩 × 3 个共享种子的配对矩阵；rank 4 在质量等价带内资源最低，结论 `accept` | LoRA 选型 清单、完整候选矩阵与 rank 4 适配器 |
| 2B | [Part 02 · 65 QLoRA 低资源分支](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md) | 64 准入清单；0.5B 与 1.5B 容量实验分别记录 | NF4 低比特底座下的质量、显存与容量约束 | 0.5B [rtx5070ti_qlora_selection_v3](../../../benchmarks/results/65_qlora_selection/qlora_selection_rtx5070ti_qlora_selection_v3.json)：省 6.1% 峰值显存但无容量收益且步时回退 42.8%，结论 `tune`；1.5B [rtx5070ti_qlora_15b_capacity_v1](../../../benchmarks/results/65_qlora_selection/qlora_selection_rtx5070ti_qlora_15b_capacity_v1.json)：3 个共享 seed 下峰值低 23.4%，最大 batch `4 → 8`，质量通过，结论 `accept`，适用范围为容量受限训练 | QLoRA 适配器 哈希、配对 JSON 与低资源分支 清单 |
| 3 | [Part 02 · 62 SFT 训练交付](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md) | 63 选定配置；正式短训、保存后恢复及重载评测 | 上游身份匹配、产物完整性、恢复/加载检查与交付状态 | [sft_delivery_v2_1](../../../benchmarks/results/62_sft_delivery/sft_delivery_evidence_sft_delivery_v2_1.json) 是旧契约证据；新交付契约待以 `rtx5070ti_sft_delivery_v3` 复测 | 检查点、SFT 适配器、评测报告与交付 清单 |
| 4A | [Part 02 · 84 离线 DPO](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.md) | 62 的 SFT 适配器；24 组训练 / 16 组评测偏好对，长度 1024，12 次更新，β=0.1 | 偏好数据准入、独立评测检查、质量—资源采用判断 | [dpo_offline_len1024_v5](../../../benchmarks/results/84_dpo_preference/dpo_evidence_dpo_offline_len1024_v5.json) 已完成；任务输入截断率降至 5%，回复截断率为 0，质量与资源检查全部通过，结论 `accept` | DPO 适配器、评测报告与候选 清单 |
| 4B | [Part 02 · 85 在线奖励优化](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.md) | 62 的 SFT 适配器；GSM8K 训练 24 题 / 评测 12 题，每题 8 条轨迹，最多 384 词元 | 采样 / 验证器覆盖检查、有效更新吞吐、质量—资源采用判断 | [grpo_rlvr_len384_v7](../../../benchmarks/results/85_grpo_alignment/grpo_evidence_grpo_rlvr_len384_v7.json) 已完成；平均奖励 +0.0521、`pass@group` +0.0833，有效组比例 0.75，结论 `accept` | GRPO 适配器、轨迹摘要、配对评测报告与候选 清单 |
| 5 | [Part 02 · 86 生命周期与恢复](../../02_PyTorch_Algorithms/86_DPO_Online_Benchmark.md) | 84 / 85 候选与 62 回滚清单；CPU 生命周期与恢复检查 | 策略新鲜度、产物 / 回滚哈希、失败恢复与交接状态 | [post_training_lifecycle_v6](../../../benchmarks/results/86_post_training_lifecycle/lifecycle_handoff_post_training_lifecycle_v6.json) 已完成；DPO v5 与 GRPO / RLVR v7 的候选、产物哈希、回滚基线和新鲜度检查通过，交接状态 `ready` | 生命周期证据与 Part 05 交接清单；允许进入冻结评测，不表示已完成最终采用 |
| D1 | [Part 02 · 88 蒸馏学生交付](../../02_PyTorch_Algorithms/88_Distilled_Student_Delivery_Project.md) | 冻结教师与共同 Qwen2.5-0.5B 学生；GSM8K 512 条训练、32 条开发、64 条验证；终止方案为 64 步锚定加 32 步在线蒸馏 | 教师—学生身份、G0/G1/G2 可比性、质量保持与部署收益 | [rtx5070ti_distill_gsm8k_hybrid_terminal_v9](../../../benchmarks/results/88_distillation_delivery/distillation_terminal_report_rtx5070ti_distill_gsm8k_hybrid_terminal_v9.json) 已结束：学生基线 19/64，混合候选 22/64；质量增益、格式和延迟未同时过门，结论 `tune`，不打开 最终留出集 | 基线、路线、候选与终止决策证据；无部署 交接文件 |

## 结果版本与复现

读取结果时使用上表链接和对应标签；独立复现使用新的唯一标签，先按 Notebook 的安全预检阶段检查，再执行真实阶段。改变模型、输入或训练剂量时保留旧记录并重新说明比较范围。当前标签表示所选正式记录，不表示历史上所有方案都通过。

共享系统实验由轨迹页执行，系统页读取同一份结果。原生接口检查使用[只读工具](../../../tools/verl_mechanism_audit.py)，一次监督更新及同步使用[共同入口](../../../tools/verl_mechanism_closure.py)与[工作进程实现](../../../tools/verl_mechanism_worker.py)；释放组失败时停止，不自动追加训练。资源漏采记为缺失。历史 GSM8K 路径的数据准备与筛查入口留在下方历史记录，不属于当前算术任务复现前置。

| 复现对象 | 必须保存 | 继续或停止条件 |
|:---|:---|:---|
| CPU 练习与框架核对 | 页面身份、环境、逐项断言和跳过原因 | 导入成功不替代数值或状态断言 |
| 单卡机制 / 项目 | 模型与数据版本、固定输入、运行标签、比较条件、原始指标和失败记录 | 检查实际更新、质量与资源门槛，失败保留原结果 |
| verl 共享实验 | 初始参数、冻结轨迹、动作掩码、概率、一次更新、同步、事件与资源采样 | 同版本位置与概率不符时停止；第二页不重复训练 |
| 学生蒸馏 | 教师版本、共同学生起点、轨迹来源、训练剂量、独立评测及终止决定 | 无合格候选时不打开最终留出集、不生成部署交接 |

<details>
<summary>历史实验记录：verl 流程探测、奖励筛查与早期配对</summary>

仅复现下列早期 GSM8K 筛查时，在已经通过环境检查的 verl 锁环境中准备数据；当前固定算术任务不执行这一步：

```bash
cd ~/verl-v0.9.1
uv run --frozen --all-packages --extra vllm \
  python3 examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

`rtx5070ti_verl_system_v8` 使用 verl v0.9.1、Qwen2.5-0.5B-Instruct、GSM8K、每批 4 个问题与每题 4 条回复，回复上限 128 Token。生成接口使用 `async`，训练器按同步的生成—更新顺序执行。两页共享一份 [结果 JSON](../../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8.json)；下表峰值分别为整卡和整机采样占用，包含初始化和其他应用。

v8 登记为“单卡流程、过滤路径与资源观测通过”。原始 JSON 的两个 `accept_mechanism` 按指标完整性生成；日志中全部 step 的奖励、优势、loss 与梯度范数均为 0，该轮没有验证有效策略更新。各 step 回复达到上限的比例为 93.75%–100%，释放组首步生成耗时 32.14 s、第三步 2.11 s；因此不登记质量收益或缓存加速结论。早期 v1–v7 失败记录留在本地追溯配置和内存问题；最终参数更新与同步证据见本页正式受控更新记录，采用监督损失而非 GRPO 损失。

| 候选与原始日志 | 完成步数 | 整卡峰值显存（MiB） | 主机峰值内存（GiB） | 直接观察 |
|:---|:---|---:|---:|:---|
| [释放缓存](../../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8_async_release_metrics.log) | 3/3，返回码 0 | 7888 | 24.53 | 行为偏离、阶段耗时和资源指标齐全 |
| [释放缓存＋拒绝过滤](../../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8_async_release_reject.log) | 3/3，返回码 0 | 7878 | 24.25 | 平均屏蔽 8.64% 回复 Token；未验证有效更新收益 |
| [保留缓存](../../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8_async_retain_metrics.log) | 3/3，返回码 0 | 10137 | 22.38 | 比释放组整卡采样峰值高 2249 MiB；差值不能直接视作 KV Cache 大小 |

**两页的实验定位**

轨迹与策略新鲜度关注“哪条轨迹能交给哪个策略更新”；采样、训练与奖励系统关注“各阶段如何交接并竞争资源”。两页使用同一单卡环境，分别解释身份与概率、阶段与驻留，不比较算法质量或承担多节点部署验收。

| 页面 | CPU 练习已经检查 | verl GPU 最小证据 | 不从单卡结果外推 |
|:---|:---|:---|:---|
| PT-ROLLOUT-LIFECYCLE | 状态迁移、版本与行为偏离、竞争预留、激活失败处理 | 同版本概率核对、真实参数变化、同一冻结轨迹前后重评、同步后的身份与概率 | 并发原子消费、多工作进程原子激活、真实异步陈旧队列 |
| PT-RL-SYSTEM | 阶段倍率、统一容量、共享链路与拓扑判断 | 实际阶段事件、引擎释放/保留对照、与事件关联的显存与主机内存 | 分池吞吐、跨节点带宽、队列稳定性与容错恢复 |

**历史奖励筛查**

筛查入口保留在 `tools/verl_lifecycle_probe.py`，结果写入 `benchmarks/results/pt_verl_lifecycle/`。历史 GSM8K 筛查使用现有 `PT-VERL-CUDA` 环境、vLLM 和 verl 官方严格验证器，当前入口已切换为固定合成算术任务与普通采样。候选组只有在四条回答都正常终止、可解析且奖励非恒定时才准入，防止用截断或解析失败制造奖励差异。原始回复、Token、行为对数概率、模型快照、任务与提示哈希均保存到 JSON。

[`rtx5070ti_rollout_lifecycle_v9_2`](../../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_2.json) 已完成采样：32 条回复均正常终止，严格答案解析 4/32，正奖励 0/32，合格组 0/8，`next_stage_allowed=false`。对严格提取结果作数值等价比较，或读取整条为纯数字的回复时，26/32 可解释为数值，其中 2 条正确、24 条错误，另 6 条无法按该规则解析；2 条正确回复分别因标记缺失和 `16.0`/`16` 表示差异未得分。该诊断不修改原始奖励，不能据此放行更新。

[`rtx5070ti_rollout_lifecycle_v9_3`](../../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_3.json) 使用 `short_calculation_v2` 提示，允许一两行计算并要求末行严格答案格式。严格解析 6/32，正奖励 0/32，截断 3/32，合格组 0/8；本轮提示调整没有形成可用更新输入。两轮各 32 条回复的 Token、行为概率和动作掩码均对齐，但每个组都存在解析失败，v9_3 另有两个组包含截断。数值诊断只读严格答案或整条纯数字回复，无法识别推理段落中的正确答案，不作为数学能力下降的依据。

[`rtx5070ti_rollout_lifecycle_v9_4`](../../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_4.json) 已完成格式约束采样：严格解析与整条整数格式匹配均为 32/32，截断为 0/32，轨迹对齐为 32/32；正奖励为 0/32，8 个组均完整但全错，可用混合奖励组为 0。官方验证器复算与记录全部一致。本轮确认格式约束能在本机 GPU 运行，但没有形成更新所需的奖励差异。提示允许短计算，解码却只允许答案直出，存在设计冲突，不能据此断言模型无法完成这些题目。

[`rtx5070ti_rollout_lifecycle_v9_5`](../../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_5.json) 已完成：31/32 可解析，25/32 在答案前有文本，正奖励 1/32，截断 13/32，轨迹对齐 32/32。年龄题产生 `[0,0,1,0]`，但其中一条截断，完整合格组仍为 0/8。部分末行数字不断延长，一条正常终止回复缺少答案；字符数和格式约束不能保证合理计算与正常结束。官方评分与保存结果一致，登记为“出现奖励差异，但更新输入仍不合格”，暂停该 GSM8K 路径。

两轮语法均不包含正确答案，评分继续使用官方严格奖励。约束改变采样分布，本轮概率不能与未使用相同约束的学习器概率直接计算 PPO 比率；即使出现混合奖励，也只进入后续更新方案设计，不直接训练。

[`rtx5070ti_rollout_arithmetic_v1`](../../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_arithmetic_v1.json) 的 8 道单步题目在采样前固定：加、减、乘、整除各 2 题，每题自然采样 4 次，仍使用 Qwen2.5-0.5B-Instruct、温度 1、种子 42 和 128 Token 上限。题目来自仓库固定算术生成器，不再依赖 GSM8K Parquet 或下载数据；正确标签由整数运算产生，不放入提示。关闭语法约束，不设置强制答案前缀或停止字符串，按整条整数回复比较数值；格式失败不用于制造奖励差异。记录任务版本、题目池哈希与实际回复，不估计公开数据集准确率，也不与旧 GSM8K 分数比较。本轮已完成：解析、正常终止、轨迹对齐均为 32/32，正确 28/32；7 组全对，1 组全错，每组四次回答均相同，完整混合奖励组为 0。错误题 `81 - 47` 四次均回答 `24`。该结果验证普通采样与轨迹记录，不构成 GRPO 有效更新。

**受控更新的结果与复现顺序**

[受控更新 v1](../../../benchmarks/results/pt_verl_closure/rtx5070ti_verl_mechanism_closure_v1.json) 两组均完成一次更新，梯度非零且各有 48 个参数张量改变。释放组全部检查通过；保留组更新后同版本概率最大误差 0.43725 超过上限 0.15，其余生命周期检查通过。两组旧轨迹均为 7 条准入、1 条拒绝，对 `81−47` 的新生成均由 `24` 变为 `34`；这仅是策略变化的具体观测，不是质量提升评测。

两组阶段与资源记录完整，但初始 LoRA 哈希不同，不能将差异归因于释放/保留开关。更新阶段整卡采样峰值为 6262/10351 MiB，整段峰值为 9562/10351 MiB（释放/保留），两者分开解释。v1 结果原样保留，其结论为“阶段观测完成，严格配对验证未通过”。

</details>
