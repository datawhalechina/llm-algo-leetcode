# 后训练优化实验资产表

本表登记后训练机制页和项目页使用的环境、工作负载、产物与证据等级。环境创建见[实验 Quickstart](./quickstart.md)，项目公共结果字段见[Part 02 结果资产与证据契约](../../part02_result_assets.md)。

## 环境角色

| 环境 ID | 适用入口 | 可形成的证据 |
|:---|:---|:---|
| `PT-CPU` | 13–18、30–33、50–51 与相关 `PT-*` 机制页的 Step 1–4 | 数据接口、标签、损失、适配器、偏好、奖励、策略新鲜度与阶段输入 / 输出字段 |
| `PT-SFT-CUDA` | 14、15、30、31；62–64 | 真实 LoRA/SFT 更新、长上下文训练代价、LoRA 变体路径、验证集与训练产物 |
| `Q-QLORA` | 26、65 | NF4 QLoRA、适配器与低资源训练证据 |
| `PT-RL-CUDA` | 17、18、PT-REWARD-MODEL、PT-REWARD-SIGNAL；84、85 的单卡阶段 | DPO、GRPO、奖励模型与奖励信号均已完成 RTX 5070 Ti 实测 |
| 后续多工作进程环境（暂未验证） | PT-RL-SYSTEM 与在线系统扩展 | 轨迹队列、策略同步、通信和多工作进程恢复证据；不由当前 85、86 的单卡/产物实验替代 |

## 单卡 GPU 验证顺序

两个 Profile 分开创建和验证，不在同一环境追加依赖。每页先使用默认工作负载生成独立 JSON；前一页失败时保留失败记录并修复，不跳过后汇总为专题结论。

下表按依赖关系给出专题资产的推荐验证顺序。离线偏好路径运行 DPO；使用规则验证器的 GRPO / RLVR 路径运行奖励信号与 GRPO，并按需加入奖励模型；使用学习型奖励的 PPO / GRPO 路径先验证奖励模型与奖励信号。专题全量验收依次复测四项机制实验。

验证前只做四项准备：分别创建 `PT-SFT-CUDA` 与 `PT-RL-CUDA`；从对应环境启动 JupyterLab；为每次真实运行设置唯一 `RUN_LABEL`；保持页面 5.1 的默认模型、样本和候选矩阵。模型缓存可以共享，Python 包环境不能混用。两个 Profile 均以 8 GiB 作为环境准入线；`PT-RL-CUDA` 的默认矩阵按 12 GB 级单卡准备。准入线不是实测峰值承诺，显存不足时记录失败，不同时缩小多个工作负载变量。

| 顺序 | Profile / 页面 | 本步验证什么 | 进入下一步的条件 |
|:---|:---|:---|:---|
| 0 | `PT-SFT-CUDA` 安装与预检 | PyTorch、CUDA、Transformers、PEFT、Datasets 与仓库运行时 | `pip check` 无冲突，Profile 返回 `status=ok` |
| 1 | Part 02 · 14 LoRA | 真实模块命中、冻结范围、低秩梯度与合并等价 | 三组目标模块均命中，基座冻结，合并误差过门槛 |
| 2 | Part 02 · 31 LoRA 变体 | 固定秩与固定参数上限下的 LoRA、rsLoRA、DoRA 路径 | v3 已记录逐项检查、失败原因与候选决策；固定预算 DoRA 仍需调参 |
| 3 | Part 02 · 15 SFT 有效更新与训练闭环 | 数据提交版本、有效更新、训练/验证隔离与资源记录 | 固定切分可复现，更新前后损失和资源记录完整 |
| 4 | Part 02 · 30 长上下文微调 | 长度阶梯，以及固定样本的 padding / 边界隔离 Packing 对照 | 三档长度使用相同模型和测量方法；Packing 的监督 Token、标签边界、位置编号和跨样本 Attention 检查均通过 |
| 5 | `PT-RL-CUDA` 安装与预检 | 偏好与奖励实验所需的单卡隔离环境 | `pip check` 无冲突，Profile 返回 `status=ok` |
| 6 | Part 02 · PT-REWARD-MODEL | 冻结真实模型表示上的奖励头更新、留出校准和错标负对照 | 正确标签路径降低成对损失并保持隐藏集排序；两个候选身份与资源记录完整 |
| 7 | Part 02 · PT-REWARD-SIGNAL | 策略偏移、终止奖励、采样对数概率比约束与回报折扣矩阵 | 行为策略偏移非零，终止奖励位置和候选矩阵完整 |
| 8 | Part 02 · 17 DPO | 固定参考策略、成对偏好边际和 LoRA 有限更新 | 三档 β 使用同一偏好对，偏好方向与梯度隔离成立 |
| 9 | Part 02 · 18 GRPO | 验证器奖励、组内优势、候选等权与策略分离 | 最小候选组与扩展候选组都形成正确更新方向；两档只比较候选覆盖，不用于选择组规模 |

## 项目页验证顺序

项目页按 `64 → 63 / 65 → 62 → 84 / 85 → 86` 运行。斜线表示并列候选分支，各页通过一个直接上游资产进入；该资产使用 manifest 保存更早阶段的血缘，Notebook 之间通过持久化资产衔接。

| 顺序 | 项目与职责 | CPU 题目应挖空的项目判断 | 真实环境实验 | 直接输入 | 输出资产 |
|:---|:---|:---|:---|:---|:---|
| 1 | Part 02 · 64 SFT 数据准入 | blocker 合取门禁、切分泄漏门禁、`accept / tune / reject` 状态转换 | `alpaca_qwen25_v2` 已完成：511 条有效记录、截断率 1.17%、跨切分重合为 0，结论 `accept`；CPU 输入检查 | 原始数据版本与评测切分 | 数据准入 manifest、样本 ID 与切分哈希 |
| 2 | Part 02 · 63 LoRA 变体选型 | 单变量可比性、质量/显存硬门槛、可行候选选择 | `lora_variant_selection_v2` 已完成 3 个秩 × 3 个共享种子的配对矩阵；rank 4 在质量等价带内资源最低，结论 `accept` | 64 的准入 manifest | LoRA 选型 manifest、完整候选矩阵与 rank 4 适配器 |
| 2B | Part 02 · 65 QLoRA 低资源分支 | NF4 低比特底座下的质量、显存与容量约束 | 0.5B `rtx5070ti_qlora_selection_v3`：省 6.1% 峰值显存但无容量收益且步时回退 42.8%，结论 `tune`；1.5B `rtx5070ti_qlora_15b_capacity_v1`：3 个共享 seed 下峰值低 23.4%，最大 batch `4 → 8`，质量通过，结论 `accept`，适用范围为容量受限训练 | 64 的准入 manifest | QLoRA adapter 哈希、配对 JSON 与低资源分支 manifest |
| 3 | Part 02 · 62 SFT 训练交付 | 上游身份匹配、产物完整性、恢复/加载门禁与交付状态 | `sft_delivery_v2_1` 是旧契约证据；新交付契约待以 `rtx5070ti_sft_delivery_v3` 复测 | 63 的 LoRA 选型 manifest | checkpoint、SFT adapter、评测报告与交付 manifest |
| 4A | Part 02 · 84 离线 DPO | 偏好数据准入、独立评测门禁、质量—资源采用判断 | `dpo_offline_len768_v4` 已完成；任务输入截断率 27.5% 超过 25% 门槛，结论 `reject`，失败项为 `prompt_coverage` | 62 的交付 manifest | DPO adapter、评测报告与候选 manifest |
| 4B | Part 02 · 85 在线奖励优化 | rollout/验证器覆盖门禁、有效更新吞吐、质量—资源采用判断 | `grpo_rlvr_len256_v6` 已完成；平均奖励 +0.0104、`pass@group` 持平，长度预算未通过，结论 `tune` | 62 的交付 manifest | GRPO adapter、轨迹摘要、配对评测报告与候选 manifest |
| 5 | Part 02 · 86 生命周期与恢复 | 策略新鲜度、artifact/回滚哈希、失败恢复与交接状态 | `post_training_lifecycle_v3` 是重构前采用记录；重构后需用新标签生成 `lifecycle_handoff`，当前 DPO 为 `reject`、GRPO 为 `tune`，没有可交接候选 | 84、85 候选 manifest 与 62 回滚 manifest | 生命周期证据与 Part 05 交接清单；不承担最终采用决策 |

### 单卡 12 GB 默认工作负载

| 项目 | 默认模型与候选 | 初次验证规模 | 必须保留的项目证据 |
|:---|:---|:---|:---|
| 63 | `Qwen/Qwen2.5-0.5B`；LoRA rank `4 / 8 / 16`，固定 target modules、alpha 规则与 dropout | 固定训练/验证切分，短训练筛选；候选确定后再用 3 个 seed 复测，不在首次运行同时扩大长度和样本数 | 每候选 val loss、任务指标、峰值显存、step time、吞吐、失败与固定条件 |
| 62 | 63 选出的唯一 LoRA 配置 | 一次正式短程训练，加一次保存—恢复—继续更新和重新加载评测 | 模型与数据 revision、上游 manifest ID、checkpoint/adapter 哈希、恢复位置、固定评测与失败记录 |
| 84 | 62 的 SFT adapter；固定 `beta=0.1` | 24 组训练偏好对、16 组独立评测偏好对、最大长度 768、12 次更新 | v4 峰值分配约 5.48 GiB；质量门槛通过，但任务输入截断率 27.5%，结论 `reject` |
| 85 | 62 的 SFT adapter；每个问题生成 8 条轨迹 | GSM8K 训练题 24 道、独立评测题 12 道、最多生成 256 词元；更新前策略/候选使用相同评测种子 | v6 有效组 16/24，峰值分配约 6.49 GiB、保留约 10.10 GiB；长度预算未通过，结论 `tune` |

64 与 86 可以在上述隔离环境中运行真实输入检查，但其结论不依赖 CUDA。64 的结果是训练前数据准入，86 的结果是生命周期与恢复准入；两者都不得用“GPU 可用”提升证据等级。

## 机制页职责与学习级别

机制页按共同检查、基础训练顺序和按需扩展组织：先固定数据、损失和评测规则，再运行对应算法；只有出现长上下文、低显存、在线轨迹或多进程问题时，才进入相应扩展页。

| 页面 | 职责 | 学习级别 | GPU 状态 |
|:---|:---|:---|:---|
| [Part 02 · 33 后训练准入与阶段选择](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md) | 用稳定失败、非参数基线、SFT 基线状态和反馈形态选择后训练阶段 | 共同入口 | CPU |
| [Part 02 · 32 SFT 数据契约与泄漏审计](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) | 检查必需字段、任务输入重复、完整样本冲突和跨切分泄漏 | SFT 主路径 | CPU |
| [Part 02 · 13 SFT 监督边界与损失](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md) | 把 chat template、labels 与有效监督 Token 接入 loss | SFT 主路径 | CPU；真实训练在 Part 02 · 15 中执行 |
| [Part 02 · 14 LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.md) | 建立目标模块、秩、缩放与适配器生命周期 | SFT 主路径 | `rtx5070ti_lora_targets_v3` 已完成；训练机制通过，最高峰值分配约 2.02 GiB；`attention_qv` 的 BF16 合并保持 `tune` |
| [Part 02 · 15 SFT 有效更新与训练闭环](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.md) | 验证有效更新、评估隔离与训练状态记录 | SFT 主路径 | `rtx5070ti_sft_v1` 已通过；峰值分配约 4.66 GiB，`accept_mechanism` |
| [Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md) | 解释 NF4、冻结基座与低资源适配器更新 | 低资源分支 | 复用 `Q-QLORA` |
| [Part 02 · 30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md) | 比较窗口适配、受控截断、监督保留、边界隔离 Packing 和显存预算 | Task1 长上下文扩展 | `rtx5070ti_length_packing_v4` 已通过；最高峰值分配约 2.90 GiB，`accept_mechanism` |
| [Part 02 · 31 LoRA 变体](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.md) | 比较 rsLoRA、DoRA、秩与固定参数上限 | 参数适配分支 | `rtx5070ti_lora_variants_v3` 已完成，最高峰值分配约 1.97 GiB；前三组通过，固定预算 `dora_r7` 因合并输出等价性未达阈值而保持 `tune` |
| [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) | 固定偏好对、数据隔离和独立评测协议 | 偏好共同基础 | CPU |
| [Part 02 · 17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.md) | 解释参考策略、`beta` 与成对偏好更新 | 离线偏好分支 | `rtx5070ti_dpo_v1` 已通过；最高峰值分配约 3.19 GiB，`accept_mechanism` |
| [Part 02 · PT-REWARD-MODEL 奖励模型训练、校准与过度优化](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md) | 解释成对奖励目标、留出校准和代理奖励过度优化 | 奖励优化共同基础 | `rtx5070ti_reward_model_v1` 已通过；正确标签候选隐藏排序准确率 1.0，错标负对照为 0.5；峰值分配约 0.95 GiB，`accept_mechanism` |
| [Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) | 统一奖励、回报、价值与优势的关系 | 奖励优化共同基础 | `rtx5070ti_reward_signal_v3` 已通过；峰值分配约 2.01 GiB，`accept_mechanism` |
| [Part 02 · 16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md) | 解释策略、参考策略、奖励模型、价值模型与资源状态 | PPO 分支 | 系统级验证 |
| [Part 02 · 18 GRPO 与可验证奖励](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.md) | 解释候选组、验证器奖励、组内基线与受约束更新 | GRPO / RLVR 分支 | v3 `default3` 已通过；最高峰值分配约 3.06 GiB，`accept_mechanism`；扩展候选组暴露较高裁剪率，不用于自动选择组规模 |
| [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) | 管理轨迹身份、策略版本、新鲜度与准入 | 在线系统共同基础 | 系统级验证：原子预留与多工作进程同步 |
| [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md) / [Part 02 · 51 在线 DPO](../../02_PyTorch_Algorithms/51_Online_DPO.md) | 分别承担多阶段系统编排与在线偏好反馈 | 两条系统分支 | 留给多工作进程或项目验证 |

## 机制与项目接口

| 机制链 | 机制页产出 | 项目页补充 |
|:---|:---|:---|
| SFT 数据 → 标签 → 损失；长样本再进入窗口与 Packing | 32、13、15；长上下文扩展为 30 | 后续项目通过结果资产消费机制证据 |
| LoRA / QLoRA → 适配器 | 14、26、31 | 31 输出固定秩 / 固定预算机制证据；63/65 比较质量与资源，并保存通过检查的适配器 |
| 偏好对 → DPO | 17、50 | 84 的训练—评测—决策记录 |
| 偏好对 / 验证结果 → 奖励 → 价值 / 优势 → PPO 或 GRPO | 奖励模型页与奖励信号页为共同基础，16 与 18 为并列算法分支 | 85 的组采样、奖励与更新闭环 |
| 策略新鲜度 → RL 系统或在线偏好 | 轨迹生命周期页为共同基础，RL 系统页与 51 为并列系统分支 | 85/86 的版本、编排、回滚与在线成本证据 |

## PT-RL-CUDA 机制工作负载

四项机制实验共享模型系列和单卡环境。奖励模型页训练冻结主干上的标量奖励头，奖励信号页比较固定轨迹上的信号构造；17、18 共享 LoRA 有限更新路径，但各自保留候选变量。固定小样本用于隔离机制差异，不能替代完整奖励模型训练、独立偏好评测或在线系统验证。

| 页面 / 工作负载 | 固定条件 | 受控候选 | 结果路径 | 当前状态 |
|:---|:---|:---|:---|:---|
| Part 02 · PT-REWARD-MODEL / `pt-reward-model-head-calibration` | Qwen2.5-0.5B-Instruct、6 组训练偏好对、4 组校准对、4 组隐藏质量对、冻结主干与相同奖励头初始化 | 正确偏好标签 / 固定 50% 错标负对照 | `benchmarks/results/pt_reward_model/` | `rtx5070ti_reward_model_v1` 已通过；正确标签候选校准/隐藏准确率均为 1.0，错标候选分别为 0.25/0.5，清楚暴露代理目标与隐藏质量背离 |
| Part 02 · 17 / `pt-dpo-real-model-update` | Qwen2.5-0.5B-Instruct、4 组偏好对、LoRA、4 次更新 | `beta = 0.05 / 0.10 / 0.30` | `benchmarks/results/pt_dpo/` | `rtx5070ti_dpo_v1` 已通过；三档偏好准确率均为 1.0 |
| Part 02 · 18 / `pt-grpo-real-model-update` | Qwen2.5-0.5B-Instruct、2 个固定提示、`integer-answer-v1` 验证器、LoRA、4 次更新 | `minimal_pair` / `expanded_group` 候选覆盖条件 | `benchmarks/results/pt_grpo/` | v3 `default3` 已通过；两档均形成正确策略分离，扩展组用于暴露格式、不可判定与裁剪饱和风险 |
| Part 02 · PT-REWARD-SIGNAL / `pt-reward-signal-matrix` | Qwen2.5-0.5B-Instruct、固定提示与回复、固定奖励 | 采样对数概率比约束强度、终止奖励放置与回报构造 | `benchmarks/results/pt_reward_signal/` | `rtx5070ti_reward_signal_v3` 已通过；6 组候选完整，终局奖励位置检查通过 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、baseline/candidate |
| 模型与数据 | model/tokenizer revision、chat template、dataset revision、split 与样本 hash |
| 训练 | dtype、有效 batch、学习率、更新次数、随机种子与策略版本 |
| 质量与安全 | loss、任务指标、win-rate/judge 一致性、安全与回滚门槛 |
| 资源与产物 | 峰值显存、训练步 / 轨迹采样时间、产物 / 适配器路径与哈希 |
| 结论 | `failure`、`evidence_level`、decision 与下一步 |
