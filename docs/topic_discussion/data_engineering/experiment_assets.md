# 数据工程与模型评测实验资产

本表区分已有机制练习、可读取的项目结果和尚无专属练习的处理环节。示意记录与演算见正文，实验结果按[公共结果规范](../../part02_result_assets.md)解释；环境安装使用下列主归属入口，不另建一套数据工程依赖环境。

## 环境与实验入口

| 环境或任务 | 使用入口 | 实际运行范围 |
|:---|:---|:---|
| 配置、文件读写与基础字段检查 | [BASE-CPU](../../guide.md#base-cpu) | Part 00基础练习；不包含通用PDF/OCR解析质量验证 |
| 后训练CPU机制 | [PT-CPU](../post_training_optimization/quickstart.md#pt-cpu-quickstart) | 记录、指纹、标签、偏好与合成候选检查；框架核对按各页依赖执行 |
| SFT数据采集与训练分支 | [PT-SFT-CUDA](../post_training_optimization/quickstart.md#pt-sft-cuda-quickstart) | Part 02 · 64的数据检查在CPU执行；后续真实训练或Part 02 · 30的GPU比较另按页面阶段运行 |
| 输入性能GPU机制 | [PERF-CUDA-MECHANISM](../performance_optimization/gpu_quickstart.md#perf-cuda-mechanism-quickstart) | 固定输入负载下的数据加载、打包与搬运比较 |
| 检索语料与服务 | 按任务选择检索组件、查询集和模型后端 | 当前未登记本专题独立的检索环境与端到端结果 |

已有环境先核对Profile与解释器，依赖匹配时复用。环境预检说明软件与设备条件，数据质量和任务效果仍由相应检查确定。

## 机制实验

| 学习对象与入口 | 固定输入或主要变化 | 已有练习或结果 | 可以说明什么 |
|:---|:---|:---|:---|
| [Part 02 · 32 SFT数据与泄漏审计](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md) | SFT记录、任务指纹与评测任务集合 | CPU题目、测试及框架核对入口；未在此表登记独立采集结果 | 字段、重复、泄漏与来源保留逻辑；近似语义关系另需复核 |
| [Part 02 · 13 标签与损失](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md) | 模板、标签范围、有效目标与移位 | CPU机制与框架核对入口 | 输入位置怎样成为损失目标，训练效果另测 |
| [Part 02 · 56 持续预训练与能力保留](../../02_PyTorch_Algorithms/56_Continual_Pretraining_and_Retention.md) | 全序列标签、有效目标、领域与保留集 | 已有CPU练习、框架核对及可选GPU阶段；运行结果按后训练资产表登记 | 全序列监督与数据隔离；语料分片、读取游标和预取恢复另需验证 |
| [Part 02 · 30 长度与打包](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md) | 长度策略、截断、独立样本边界 | CPU练习；[历史GPU记录](../../../benchmarks/results/pt_long_context/pt-long-context_rtx5070ti_length_packing_v4.json)，状态与配置见[后训练资产表](../post_training_optimization/experiment_assets.md#机制实验) | 目标保留、边界与局部资源变化，限于登记负载 |
| [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) | 偏好对、切分、成对结果与评审记录 | CPU审计与统计练习入口 | 偏好数据及评测统计逻辑 |
| [Part 02 · PT-SYNTHETIC-DATA 合成候选检查](../../02_PyTorch_Algorithms/PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission.md) | 内存候选、生成配置、相似度与已有验证记录 | CPU机制与Datasets核对入口；实际生成及验证器执行需另行登记 | 检查与聚合候选记录；不把记录中的“通过”视为本次重新执行验证器 |
| [Part 02 · DISTILL-OPD 在策略蒸馏与GKD](../../02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.md) | 固定与学生来源标记、已对齐的教师／学生分布、完成区掩码 | 已有CPU练习与框架核对入口；条件与结果见[后训练资产表](../post_training_optimization/experiment_assets.md#机制实验) | 来源选择、位置与分布目标逻辑；真实学生生成和教师反馈采集另测 |
| [Part 02 · PT-ROLLOUT-LIFECYCLE 轨迹与策略新鲜度](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) | 轨迹状态、修订号、策略版本及给定动作概率 | 已有CPU状态迁移与预留练习、真实轨迹可选阶段；结果按后训练资产表登记 | 使用状态与版本比较；CPU顺序模拟不替代真实并发与故障恢复 |
| [Part 02 · PT-AGENTIC-RL 环境轨迹与奖励归因](../../02_PyTorch_Algorithms/PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md) | 给定动作驱动的确定性查询—提交环境 | 已有CPU轨迹、终止／截断、动作回报练习及Gymnasium核对；结果按后训练资产表登记 | 受控环境交互与记录逻辑；真实模型采样及有副作用工具恢复另测 |
| [Part 02 · 52 量化校准范围与裁剪](../../02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.md) | 校准数值、独立验证数值与裁剪候选 | 已有CPU范围估计与验证集选择测试、框架核对；模型实验另按量化资产表登记 | 校准与验证分工及局部量化误差；真实语料代表性和模型质量另测 |
| [Part 02 · INF-REQUEST-LIFECYCLE 请求生命周期](../../02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.md)／[Part 02 · 66 单卡推理测量基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) | 给定事件、请求记录与固定测量配置 | 已有CPU事件回放与记录核验，66另有真实单卡阶段；结果按推理资产表登记 | 事件及统计口径；原始请求集的内容、长度与到达分布另需检查 |
| [Part 02 · PERF-INPUT 输入性能](../../02_PyTorch_Algorithms/PERF-INPUT_Training_Input_Pipeline_Performance.md) | 固定批量与形状，改变内存、复制与预取设置 | [历史GPU记录](../../../benchmarks/results/perf_input_pipeline/perf-input_rtx5070ti_input_v1.json)，具体比较见[性能资产表](../performance_optimization/experiment_assets.md#机制实验) | 当前负载的输入性能；历史记录仅有汇总，不能回算逐次分布 |
| [07 检索切块与索引更新](./07_retrieval_chunks_and_index_updates.md) | 教学文档、相关查询及更新和删除情形 | 目前为正文示例与检查方法，尚无专属CPU练习或后端结果 | 解释片段来源、索引成员和全量／增量比较方法 |
| [08 评测集、推理负载与校准样本](./08_evaluation_workloads_and_calibration.md) | 质量评测、请求内容与到达方式、校准输入 | 目前为正文示例与检查方法，尚无专属请求集练习或校准结果 | 区分数据用途及需要保存的比较条件 |
| [09 逐样本评分](./09_model_evaluation_metrics_and_scoring.md) | 为具体问题写参考依据与评分规则，手算四题例子的覆盖、正确率与限时成功率 | 正文构造例子；逐条回答的任务评分器尚无新增可运行练习 | 区分答错、生成失败与评分缺失；50仅承接已有评分的统计 |
| [10 基线比较](./10_baseline_comparison_and_regression.md) | 按题目配对，区分胜负、平局、缺失和分类退化 | 复用Part 02 · 50的成对统计与一致性练习；未新增真实模型评测结果 | 50在零明确判决时的胜率占位值需结合判决数解释 |

## 项目实验

| 项目入口 | 输入与比较对象 | 结果或记录 | 使用范围 |
|:---|:---|:---|:---|
| [Part 02 · 64 SFT数据质量](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) | 固定原始数据、分词器及清洗规则，检查训练与评测成员 | [0.5B历史记录](../../../benchmarks/results/64_sft_data_quality/sft_data_admission_alpaca_qwen25_v2.json)、[1.5B历史记录](../../../benchmarks/results/64_sft_data_quality/sft_data_admission_alpaca_qwen25_15b_v1.json)均为 `fine-tuning-project/v1`；当前[v2实验合同](../../../benchmarks/contracts/sft_data_admission_v2.json)未登记新采集 | 历史文件的 `accept`按原协议读取；数据检查结果不替代后续模型质量比较，具体衔接见[后训练项目表](../post_training_optimization/experiment_assets.md#项目实验) |
| [Part 05 · 02 MiniMind](../../05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery/index.md) | 项目自产固定数据，比较全参数SFT、LoRA与后续DPO | 阶段与结果入口由项目页说明 | 独立训练流程参考；不作为外部数据版本对照结果 |

## 结果版本与复现

读取旧结果时，先检查记录里的 `schema_version`、输入、分词器、规则版本和成员清单，再按原协议解释结论。文件名中的版本后缀是运行标签的一部分，不等于实验协议版本；例如上表的0.5B文件名带有v2，记录格式仍为v1。新流程使用新的运行标签，保留旧文件及其含义。

现有练习已覆盖SFT与全序列标签、长度和打包、偏好与蒸馏、轨迹使用状态、受控环境交互、校准数值及请求统计。继续补强时，先复用这些机制，实际可运行范围按上表读取。

仍缺完整练习的内容包括：语料分片与读取进度恢复、真实在线采样与工具轨迹采集、检索切块与索引更新、逐条回答评分与评分失败分流、原始请求集统计和校准语料代表性。通用解析质量与跨样本依赖重算恢复也仍需验证；正文中的例子用于解释处理过程。

优先补强两类CPU机制：固定文档的切块、索引更新与删除，对照增量结果和全量重建；固定任务的确定性评分，分别保存答错、生成失败和评分缺失，再按同题比较。它们沿用BASE-CPU，待实现并验证后再登记页面与结果，真实检索后端和模型生成另行准备。

合成数据下一步沿[04的分数例子](./04_synthetic_data_and_trajectory_validation.md#把分数例子做成一次完整流程)补生成、精确验证、去重与切分的CPU闭环。现有PT-SYNTHETIC-DATA仍按给定候选和验证记录的检查练习登记，新增闭环暂未实现与实测。

复现已有实验时按上表进入对应页面；新增数据版本比较前先固定使用任务、验证/测试隔离和比较变量，按实际需要分别验证处理规则与任务效果。
