# 02 · MiniMind 后训练项目

> 项目类型：端到端后训练　运行环境：`PT-SFT-CUDA`

## 项目目标

使用固定的 MiniMind 小模型和本地小规模任务数据，对照全参数 SFT 与 MiniMind 原生 LoRA，再从全参数 SFT 分支继续 DPO，并进行独立评测。结论限于本项目的数据和训练预算，不外推为通用能力或生产结论。

本项目采用 MiniMind 官方 PyTorch 训练实现，代码与模型权重都固定版本。训练数据由项目生成并保存在本地，不下载 MiniMind 的 GB 级主线数据集。基础模型从官方 `pretrain_768.pth` 起步；不在本项目内重复预训练。SFT 与 DPO 使用同一基础模型、同一任务语义和固定训练预算，评测集与训练集分离。

## 流程与产物

| 阶段 | 工作 | 产物 |
|:---|:---|:---|
| 数据准备与准入 | 生成 SFT 样本、偏好对与独立评测样本；分别进行一小步全参数 SFT、LoRA 和 DPO 显存探测 | 固定数据、摘要及 `memory_probe.json` |
| SFT 对照 | 从同一基座、同一数据和固定训练预算分别运行全参数 SFT 与原生 LoRA（秩 16）；两支使用相同学习率 | SFT 权重、LoRA adapter、可训练参数量和训练日志 |
| DPO | 以全参数 SFT 权重为起点执行 DPO；不把 DPO 接在 LoRA 分支上 | DPO 权重及训练日志 |
| 独立评测 | 比较基座、全参数 SFT、LoRA 与 DPO 的留出集损失、偏好胜率、模型/adapter 大小、显存和阶段耗时 | 留出集评测记录与产物哈希 |
| 结果登记 | 核对模型/代码版本、数据身份、训练预算、产物哈希与门槛 | `accept / tune / reject` 项目报告 |

小样本可能不足以区分候选；若证据不明确，报告应保留不确定性，不以训练损失下降判定能力提升。

## 预注册指标

运行前固定下列指标与门槛；它们用来判断这次端到端训练链是否可复现、资源是否可承受，不预设哪条训练分支胜出。

| 指标 | 比较用途 | 解释方式 |
|:---|:---|:---|
| 留出集损失 | 检查训练后质量 | 训练损失下降不替代留出集改善；固定 32 条评测样本并保留原始值 |
| 偏好胜率 | 比较 SFT、LoRA 与 DPO 输出 | 固定 24 组偏好对；样本只作流程核对，不外推为通用偏好 |
| 阶段耗时、峰值显存 | 判断完整流程能否在目标机器完成 | 分阶段记录；LoRA 的资源收益与质量变化一起看 |
| 可训练参数量、产物大小与 SHA256 | 核实训练分支和产物身份 | 对照全参权重、LoRA adapter、DPO 权重及其来源 |
| 重复运行和数据摘要 | 检查流程可复现性 | 固定 32 条评测样本；数据切分、源码或权重版本不一致时不作配对结论 |

## 运行入口

全流程只使用一个 `PT-SFT-CUDA` 环境；同一环境运行 Notebook、MiniMind 的原生训练脚本和评测。不要再安装 MiniMind 仓库的整份 `requirements.txt`，避免覆盖已锁定的 PyTorch。先按[后训练 Quickstart](../../topic_discussion/post_training_optimization/quickstart.md#pt-sft-cuda-quickstart)创建环境，再按本页配置准备固定版本的 MiniMind 源码和基础权重。

打开 [`Post_Training_Delivery_Project.ipynb`](./Post_Training_Delivery_Project.md)，只在首个配置单元设置唯一 `RUN_LABEL` 和 `RUN_STAGE`。按 `contract_check → prepare → probe → train → close` 运行：`prepare` 会克隆固定版本源码、复用或下载基座权重，并生成本地固定数据；它不启动训练。`probe` 用短步全参数 SFT、LoRA 和 DPO 核对显存；探测通过后，`train` 执行三条训练阶段并评测；`close` 登记结果与判断。若准备或探测未通过，不运行完整训练。

## 与 Part 02 的关系

本项目保留 DPO，是为了展示 MiniMind 从基座、SFT 到离线偏好训练的交付流程；DPO 使用固定偏好对，不在训练时生成轨迹。Part 02 的 [84 节](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.md)评估 DPO 算法，[85 节](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.md)验证 GRPO 组内更新，[86 节](../../02_PyTorch_Algorithms/86_DPO_Online_Benchmark.md)覆盖在线训练生命周期，[16 节](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md)介绍 PPO 的 Actor、Reference、Reward、Critic 协作。本项目独立固定数据和预算，不读取这些小节的结果；加入 PPO/GRPO 则会引入在线采样、奖励和策略更新等另一组实验目标。

本项目的 LoRA 使用 MiniMind 固定版本自带的 adapter 实现，不是 QLoRA。训练入口从普通未量化基座注入 LoRA 并保存 adapter；固定路径没有加入 4-bit 基座量化与量化反传。当前约 64M 的模型在 12GB 单卡上，全参数训练本身可作为对照；加入 QLoRA 的显存收益信号有限，却会增加量化实现与兼容性变量。QLoRA 更适合放在[量化机制页](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)及[量化选型项目](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md)中，和量化误差、显存及质量一起评估。

## 显存与结论边界

MiniMind 的显存占用由权重精度、序列长度、批量、激活、优化器状态及 DPO 的参考模型计算共同决定。公开的社区估算只能用于初步规划，不能当作本机门槛。本项目以当前 Profile 预检、短步显存探测和每阶段峰值记录决定是否继续；若 DPO 不满足余量门槛，可先降低序列长度或 micro-batch，再决定是否启用梯度累积。任何配置变化都记录在本次结果中，不覆盖既有测量。
