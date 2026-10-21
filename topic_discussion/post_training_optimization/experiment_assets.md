# 后训练优化实验资产表

本表登记后训练机制页和项目页使用的环境、工作负载、产物与证据等级。环境创建见[实验 Quickstart](./quickstart.md)，项目公共结果字段见[Part 02 结果资产与证据契约](../../docs/part02_result_assets.md)。

## 环境角色

| 环境 ID | 适用入口 | 可形成的证据 |
|:---|:---|:---|
| `PT-CPU` | 13–18、30–33、50–51 与相关 `PT-*` 机制页的 Step 1–4 | 数据接口、标签、损失、适配器、偏好、奖励、策略新鲜度与阶段输入 / 输出字段 |
| `PT-SFT-CUDA` | 14、15、30、31；62–64 | 真实 LoRA/SFT 更新、长上下文训练代价、LoRA 变体路径、验证集与训练产物 |
| `Q-QLORA` | 26、65 | NF4 QLoRA、适配器与低资源训练证据 |
| `PT-RL-CUDA` | 17、18、PT-REWARD-MODEL、PT-REWARD-SIGNAL；84、85 的单卡阶段 | DPO、GRPO、奖励模型与奖励信号均已完成 RTX 5070 Ti 实测 |
| `PT-VERL-CUDA` | PT-ROLLOUT-LIFECYCLE、PT-RL-SYSTEM | `rtx5070ti_verl_mechanism_closure_v2` 两组更新、同轨迹重评、同步与阶段资源检查全部通过，本机单卡机制验证完成 |
| 后续多工作进程环境（暂未验证） | 在线系统扩展 | 轨迹队列、网络通信、跨节点策略同步和多工作进程恢复；不由 `PT-VERL-CUDA` 单卡证据替代 |

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
| [Part 02 · 33 后训练准入与阶段选择](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.ipynb) | 用稳定失败、非参数基线、SFT 基线状态和反馈形态选择后训练阶段 | 共同入口 | CPU |
| [Part 02 · 32 SFT 数据契约与泄漏审计](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb) | 检查必需字段、任务输入重复、完整样本冲突和跨切分泄漏 | SFT 主路径 | CPU |
| [Part 02 · 13 SFT 监督边界与损失](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.ipynb) | 把 chat template、labels 与有效监督 Token 接入 loss | SFT 主路径 | CPU；真实训练在 Part 02 · 15 中执行 |
| [Part 02 · 14 LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.ipynb) | 建立目标模块、秩、缩放与适配器生命周期 | SFT 主路径 | `rtx5070ti_lora_targets_v3` 已完成；训练机制通过，最高峰值分配约 2.02 GiB；`attention_qv` 的 BF16 合并保持 `tune` |
| [Part 02 · 15 SFT 有效更新与训练闭环](../../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.ipynb) | 验证有效更新、评估隔离与训练状态记录 | SFT 主路径 | `rtx5070ti_sft_v1` 已通过；峰值分配约 4.66 GiB，`accept_mechanism` |
| [Part 02 · 26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) | 解释 NF4、冻结基座与低资源适配器更新 | 低资源分支 | 复用 `Q-QLORA` |
| [Part 02 · 30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb) | 比较窗口适配、受控截断、监督保留、边界隔离 Packing 和显存预算 | Task1 长上下文扩展 | `rtx5070ti_length_packing_v4` 已通过；最高峰值分配约 2.90 GiB，`accept_mechanism` |
| [Part 02 · 31 LoRA 变体](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.ipynb) | 比较 rsLoRA、DoRA、秩与固定参数上限 | 参数适配分支 | `rtx5070ti_lora_variants_v3` 已完成，最高峰值分配约 1.97 GiB；前三组通过，固定预算 `dora_r7` 因合并输出等价性未达阈值而保持 `tune` |
| [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) | 固定偏好对、数据隔离和独立评测协议 | 偏好共同基础 | CPU |
| [Part 02 · 17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb) | 解释参考策略、`beta` 与成对偏好更新 | 离线偏好分支 | `rtx5070ti_dpo_v1` 已通过；最高峰值分配约 3.19 GiB，`accept_mechanism` |
| [Part 02 · PT-REWARD-MODEL 奖励模型训练、校准与过度优化](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.ipynb) | 解释成对奖励目标、留出校准和代理奖励过度优化 | 奖励优化共同基础 | `rtx5070ti_reward_model_v1` 已通过；正确标签候选隐藏排序准确率 1.0，错标负对照为 0.5；峰值分配约 0.95 GiB，`accept_mechanism` |
| [Part 02 · PT-REWARD-SIGNAL 奖励、价值与优势信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb) | 统一奖励、回报、价值与优势的关系 | 奖励优化共同基础 | `rtx5070ti_reward_signal_v3` 已通过；峰值分配约 2.01 GiB，`accept_mechanism` |
| [Part 02 · 16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.ipynb) | 解释策略、参考策略、奖励模型、价值模型与资源状态 | PPO 分支 | 系统级验证 |
| [Part 02 · 18 GRPO 与可验证奖励](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.ipynb) | 解释候选组、验证器奖励、组内基线与受约束更新 | GRPO / RLVR 分支 | v3 `default3` 已通过；最高峰值分配约 3.06 GiB，`accept_mechanism`；扩展候选组暴露较高裁剪率，不用于自动选择组规模 |
| [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb) | 管理轨迹身份、策略版本、新鲜度与准入 | 在线系统共同基础 | 受控更新 v2 两组全部通过；初始与更新后哈希分别一致，同版本最大概率误差 0.06250，旧轨迹均 7 条准入、1 条拒绝 |
| [Part 02 · PT-RL-SYSTEM 采样、训练与奖励系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb) / [Part 02 · 51 在线 DPO](../../02_PyTorch_Algorithms/51_Online_DPO.ipynb) | 分别承担多阶段系统编排与在线偏好反馈 | 两条系统分支 | 系统页受控更新 v2 配对、阶段、恢复与资源检查通过；更新阶段整卡峰值为释放 6392 / 保留 10489 MiB，不构成在线 DPO 或稳态加速的验证 |

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

## PT-VERL-CUDA 单卡系统运行记录

`rtx5070ti_verl_system_v8` 使用 verl v0.9.1、Qwen2.5-0.5B-Instruct、GSM8K、每批 4 个问题与每题 4 条回复，回复上限 128 Token。生成接口使用 `async`，训练器按同步的生成—更新顺序执行。两页共享一份 [结果 JSON](../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8.json)；下表峰值分别为整卡和整机采样占用，包含初始化和其他应用。

v8 登记为“单卡流程、过滤路径与资源观测通过”。原始 JSON 的两个 `accept_mechanism` 按指标完整性生成；日志中全部 step 的奖励、优势、loss 与梯度范数均为 0，该轮没有验证有效策略更新。各 step 回复达到上限的比例为 93.75%–100%，释放组首步生成耗时 32.14 s、第三步 2.11 s；因此不登记质量收益或缓存加速结论。早期 v1–v7 失败记录留在本地追溯配置和内存问题；最终参数更新与同步证据见下方受控更新 v2，采用监督损失而非 GRPO 损失。

| 候选与原始日志 | 完成步数 | 整卡峰值显存（MiB） | 主机峰值内存（GiB） | 直接观察 |
|:---|:---|---:|---:|:---|
| [释放缓存](../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8_async_release_metrics.log) | 3/3，返回码 0 | 7888 | 24.53 | 行为偏离、阶段耗时和资源指标齐全 |
| [释放缓存＋拒绝过滤](../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8_async_release_reject.log) | 3/3，返回码 0 | 7878 | 24.25 | 平均屏蔽 8.64% 回复 Token；未验证有效更新收益 |
| [保留缓存](../../benchmarks/results/pt_verl_system/rtx5070ti_verl_system_v8_async_retain_metrics.log) | 3/3，返回码 0 | 10137 | 22.38 | 比释放组整卡采样峰值高 2249 MiB；差值不能直接视作 KV Cache 大小 |

### 两页的实验定位

轨迹与策略新鲜度关注“哪条轨迹能交给哪个策略更新”；采样、训练与奖励系统关注“各阶段如何交接并竞争资源”。两页使用同一单卡环境，分别解释身份与概率、阶段与驻留，不比较算法质量或承担多节点部署验收。

| 页面 | CPU 练习已经检查 | verl GPU 最小证据 | 不从单卡结果外推 |
|:---|:---|:---|:---|
| PT-ROLLOUT-LIFECYCLE | 状态迁移、版本与行为偏离、竞争预留、激活失败处理 | 同版本概率核对、真实参数变化、同一冻结轨迹前后重评、同步后的身份与概率 | 并发原子消费、多工作进程原子激活、真实异步陈旧队列 |
| PT-RL-SYSTEM | 阶段倍率、统一容量、共享链路与拓扑判断 | 实际阶段事件、引擎释放/保留对照、与事件关联的显存与主机内存 | 分池吞吐、跨节点带宽、队列稳定性与容错恢复 |

### 历史奖励筛查

筛查入口保留在 `tools/verl_lifecycle_probe.py`，结果写入 `benchmarks/results/pt_verl_lifecycle/`。历史 GSM8K 筛查使用现有 `PT-VERL-CUDA` 环境、vLLM 和 verl 官方严格验证器，当前入口已切换为固定合成算术任务与普通采样。候选组只有在四条回答都正常终止、可解析且奖励非恒定时才准入，防止用截断或解析失败制造奖励差异。原始回复、Token、行为对数概率、模型快照、任务与提示哈希均保存到 JSON。

[`rtx5070ti_rollout_lifecycle_v9_2`](../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_2.json) 已完成采样：32 条回复均正常终止，严格答案解析 4/32，正奖励 0/32，合格组 0/8，`next_stage_allowed=false`。对严格提取结果作数值等价比较，或读取整条为纯数字的回复时，26/32 可解释为数值，其中 2 条正确、24 条错误，另 6 条无法按该规则解析；2 条正确回复分别因标记缺失和 `16.0`/`16` 表示差异未得分。该诊断不修改原始奖励，不能据此放行更新。

[`rtx5070ti_rollout_lifecycle_v9_3`](../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_3.json) 使用 `short_calculation_v2` 提示，允许一两行计算并要求末行严格答案格式。严格解析 6/32，正奖励 0/32，截断 3/32，合格组 0/8；本轮提示调整没有形成可用更新输入。两轮各 32 条回复的 Token、行为概率和动作掩码均对齐，但每个组都存在解析失败，v9_3 另有两个组包含截断。数值诊断只读严格答案或整条纯数字回复，无法识别推理段落中的正确答案，不作为数学能力下降的依据。

[`rtx5070ti_rollout_lifecycle_v9_4`](../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_4.json) 已完成格式约束采样：严格解析与整条整数格式匹配均为 32/32，截断为 0/32，轨迹对齐为 32/32；正奖励为 0/32，8 个组均完整但全错，可用混合奖励组为 0。官方验证器复算与记录全部一致。本轮确认格式约束能在本机 GPU 运行，但没有形成更新所需的奖励差异。提示允许短计算，解码却只允许答案直出，存在设计冲突，不能据此断言模型无法完成这些题目。

[`rtx5070ti_rollout_lifecycle_v9_5`](../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_lifecycle_v9_5.json) 已完成：31/32 可解析，25/32 在答案前有文本，正奖励 1/32，截断 13/32，轨迹对齐 32/32。年龄题产生 `[0,0,1,0]`，但其中一条截断，完整合格组仍为 0/8。部分末行数字不断延长，一条正常终止回复缺少答案；字符数和格式约束不能保证合理计算与正常结束。官方评分与保存结果一致，登记为“出现奖励差异，但更新输入仍不合格”，暂停该 GSM8K 路径。

两轮语法均不包含正确答案，评分继续使用官方严格奖励。约束改变采样分布，本轮概率不能与未使用相同约束的学习器概率直接计算 PPO 比率；即使出现混合奖励，也只进入后续更新方案设计，不直接训练。

[`rtx5070ti_rollout_arithmetic_v1`](../../benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_arithmetic_v1.json) 的 8 道单步题目在采样前固定：加、减、乘、整除各 2 题，每题自然采样 4 次，仍使用 Qwen2.5-0.5B-Instruct、温度 1、种子 42 和 128 Token 上限。题目来自仓库固定算术生成器，不再依赖 GSM8K Parquet 或下载数据；正确标签由整数运算产生，不放入提示。关闭语法约束，不设置强制答案前缀或停止字符串，按整条整数回复比较数值；格式失败不用于制造奖励差异。记录任务版本、题目池哈希与实际回复，不估计公开数据集准确率，也不与旧 GSM8K 分数比较。本轮已完成：解析、正常终止、轨迹对齐均为 32/32，正确 28/32；7 组全对，1 组全错，每组四次回答均相同，完整混合奖励组为 0。错误题 `81 - 47` 四次均回答 `24`。该结果验证普通采样与轨迹记录，不构成 GRPO 有效更新。

### 受控更新的结果与复现顺序

[受控更新 v1](../../benchmarks/results/pt_verl_closure/rtx5070ti_verl_mechanism_closure_v1.json) 两组均完成一次更新，梯度非零且各有 48 个参数张量改变。释放组全部检查通过；保留组更新后同版本概率最大误差 0.43725 超过上限 0.15，其余生命周期检查通过。两组旧轨迹均为 7 条准入、1 条拒绝，对 `81−47` 的新生成均由 `24` 变为 `34`；这仅是策略变化的具体观测，不是质量提升评测。

两组阶段与资源记录完整，但初始 LoRA 哈希不同，不能将差异归因于释放/保留开关。更新阶段整卡采样峰值为 6262/10351 MiB，整段峰值为 9562/10351 MiB（释放/保留），两者分开解释。v1 结果原样保留，其结论为“阶段观测完成，严格配对验证未通过”。

最终确认 [v2](../../benchmarks/results/pt_verl_closure/rtx5070ti_verl_mechanism_closure_v2.json) 已完成，两页均通过本机单卡机制验证。创建模型前固定种子 42，初始 LoRA 哈希均为 `06380c94…`，一步更新后均为 `369392e9…`；梯度范数 0.03428，参数差值范数 0.61536，两组均有 48 个参数张量改变。更新后同版本对数概率最大误差均为 0.06250、平均误差 0.002653，满足原定上限 0.15/0.03；两组旧轨迹均为 7 条准入、1 条拒绝。样本、一步监督更新、全部阈值和资源设置未变，不再追加实验。

v2 系统配对、阶段、恢复与资源检查全部通过。更新阶段整卡峰值为释放 6392 / 保留 10489 MiB，阶段时间为 1.367 / 1.345 s；整段峰值为 10086 / 10489 MiB，释放组峰值来自初次同步。短阶段未采到的峰值继续记为 null。完整记录保留在两组候选 JSON、同名 `.resources.json` 与 `.log` 中；这组单次观测不作为稳态加速、纯 KV 大小或分布式原子激活的证据。

不再把 GRPO 混合奖励组作为两页的共同前置条件。轨迹页使用 verl 内置监督损失完成一次有界 actor 更新；训练样本与固定轨迹探针分开，不把奖励筛查改记为 GRPO 通过。参数变化、同轨迹重评和同步走实际 verl worker 路径，独立 PEFT 更新与 vLLM 加载不能替代这份证据。

锁定提交中的 11 个接口已由[只读工具](../../tools/verl_mechanism_audit.py)核对；该工具的 `gpu_recipe_ready=false` 仅表示源码检查不验证 GPU 执行。[共同运行入口](../../tools/verl_mechanism_closure.py)与[verl worker](../../tools/verl_mechanism_worker.py)的 v2 GPU 路径已验证。读取已有结果时设置 v2 标签，只执行 5.1、5.3；独立复现时使用新的唯一标签，先 `dry_run` 预检，再改为 `real_gpu`，结果写入 `benchmarks/results/pt_verl_closure/`，第二页直接读取，不重复训练。

释放组的参数变化与同步检查未通过时，保留组自动跳过。逐项判断、候选日志及资源采样分别保留；已有文件不覆盖，不自动重试或追加训练。两组均完成后，轨迹页读取固定 Token 的变化与准入判断，系统页读取配置配对、阶段事件、恢复状态标签与资源覆盖；单卡结果不作为分布式原子激活或稳态加速的证据。

`free_cache_engine` 通过引擎休眠与恢复管理状态，LoRA 路径还涉及权重恢复和 KV 恢复。资源对照称为“引擎释放/保留”，记录休眠调用完成的阶段时间、恢复调用顺序和状态标签；保留模式不调用恢复时按配置检查，整卡显存差不能直接解释成 KV Cache 大小。

| 顺序 | 页面与固定输入 | 实际执行及必须保存 | 通过条件与停止规则 |
|:---|:---|:---|:---|
| 0：接口与数据核对 | 两页，当前 verl 提交与已有算术记录 | 记录接口源码身份；核对监督损失、掩码移位、TensorDict 批次与评分位置；单卡直接调用 worker，不经 TransferQueue | CPU 检查与 GPU 执行分开；不能只凭接口名字宣布 GPU 已验证 |
| 1：同版本基线 | 轨迹页，按预先固定规则读取每题第一条完整轨迹，不按奖励挑选 | 固定提示、回复 Token、动作掩码和轨迹哈希；用同一策略核对行为侧与学习器侧概率 | 逐位置对齐且有限，满足预先声明的数值容差；失败时先查位置、采样分布或实现误差 |
| 2：一次受控更新 | 轨迹页，固定的独立监督样本，LoRA rank 16 | verl actor 使用监督损失，仅一次优化器步进；保存同名同 dtype 参数指纹、差值范数、监督 Token 数、loss 和梯度 | loss、梯度有限，梯度与实际参数差值非零；不要求非零 GRPO 优势或质量提升，不自动追加步数 |
| 3：旧轨迹重评与同步 | 轨迹页，继续使用第 1 项完全相同的冻结轨迹 | 保存更新前后逐 Token 概率、版本与产物身份；按正文公式计算平均绝对对数概率差，再走 verl 权重同步并核对生成侧 | 同轨迹身份不变、策略身份可追溯；同步后概率满足预设容差。未超过准入阈值时如实登记，不强求触发拒绝 |
| 4：阶段与状态对照 | 系统页，复用上述受控更新方式；释放与保留仅改变状态管理配置 | 分别记录生成、概率计算、更新、同步、休眠、权重恢复和 KV 恢复事件；用单调时钟关联显存与主机采样 | 两组有完整事件、配置和资源记录；漏采峰值记为未知。基本机制验证不要求宣布加速，若比较性能则另做统一预热与配对复测 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、baseline/candidate |
| 模型与数据 | model/tokenizer revision、chat template、dataset revision、split 与样本 hash |
| 训练 | dtype、有效 batch、学习率、更新次数、随机种子与策略版本 |
| 质量与安全 | loss、任务指标、win-rate/judge 一致性、安全与回滚门槛 |
| 资源与产物 | 峰值显存、训练步 / 轨迹采样时间、产物 / 适配器路径与哈希 |
| 结论 | `failure`、`evidence_level`、decision 与下一步 |
