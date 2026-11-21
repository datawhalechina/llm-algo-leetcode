# 数据工程与模型评测判断手册

先找回数据版本、使用任务和异常样本，再沿表格定位需要检查的对象。比较模型时保留各侧运行配置，并按同题对应回答与评分；处理正确性、任务质量与运行成本分别记录，便于看清变化发生在哪一层。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 文件可读取但训练或检索结果异常 | 内容类型与解析状态 | 来源、解析器版本、结构化输出、失败样本 | [01 采集、解析与来源记录](./01_collection_parsing_contract.md) | 用人工抽样或下游失败文档复核解析结果 |
| 输入或日志含联系人、账号等敏感内容 | 当前用途需要哪些字段，敏感信息在哪些派生内容中保留 | 受限来源引用、处理规则、脱敏后样本和复核状态 | [02 敏感内容处理](./02_cleaning_quality_chain.md#sensitive-content) | 核对处理后语义与残留信息，日志和检索返回一并检查 |
| 某类语言或长样本在清洗后很少出现 | 按来源、语言、长度比较保留数量与排除原因 | 处理前后计数、失败类别、误删抽样和剩余不同样本数 | [02 过滤前后覆盖](./02_cleaning_quality_chain.md#filtering-coverage) | 查明解析或过滤原因，再按[03 数据配比](./03_data_mixture_and_curriculum.md)调整采样 |
| 去重后数据显著变少 | 重复组与相似阈值 | 被合并样本、组大小分布、误删抽样 | [02 清洗、去重与评测隔离](./02_cleaning_quality_chain.md) | 用验证集与人工抽样调整阈值，规则确定后再作最终测试 |
| 训练结果很好但独立评测失真 | 切分来源与跨集合相似样本 | 数据版本、评测成员、重复与来源关系 | [02 清洗、去重与评测隔离](./02_cleaning_quality_chain.md) | [Part 02 · 64 SFT 数据质量项目](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) |
| 领域能力提高、通用能力回退 | 来源、长度和难度配比 | 配方版本、各桶样本数、能力切片结果 | [03 数据配比与难度递进](./03_data_mixture_and_curriculum.md) | 固定训练配置后做配方消融 |
| 合成样本流畅但无法确认正确 | 格式、独立答案与验证是否完成 | 生成配置、规则或执行日志、错误与待复核记录 | [04 示范、合成与轨迹验证](./04_synthetic_data_and_trajectory_validation.md) | 按任务重试或复核；工具轨迹另查[智能体轨迹数据与验证](../agent/01_agent_trajectory_data.md) |
| 同一对回答被标注为相反偏好 | 同一上下文、判定规则及候选顺序 | 原始判断、规则版本、分歧原因与复核结论 | [04 人工示范与偏好标注](./04_synthetic_data_and_trajectory_validation.md#human-demonstrations-and-preferences) | [Part 02 · 50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md)检查记录及成对比较 |
| 训练损失下降但目标能力没有提升 | 标签范围与监督接口 | 输入字段、有效监督词元、失败切片 | [05 训练输入与采样记录](./05_post_training_data_interfaces.md) | [后训练优化](../post_training_optimization/intro.md)复测候选能力 |
| RAG 回答不可靠 | 文档结构、切块与知识版本 | 解析记录、片段来源、检索失败样本 | [07 检索切块与索引](./07_retrieval_chunks_and_index_updates.md) | 用固定查询集与相关文档检查召回和回答；模型请求执行参考[推理优化](../inference_optimization/intro.md)，全链路成本参考[性能优化](../performance_optimization/intro.md) |
| 查询返回了当前用户不可访问的片段 | 文档到片段的访问条件、查询筛选与缓存 | 查询身份、片段版本、访问标签与实际返回 | [07 检索访问范围](./07_retrieval_chunks_and_index_updates.md#retrieval-access) | 用可访问与不可访问两种身份复查片段、引用和缓存 |
| 训练吞吐不稳定 | 长度分布、打包和数据就绪时间 | 批次长度、等待时间、主机—设备搬运记录 | [03 数据配比与难度递进](./03_data_mixture_and_curriculum.md) | [性能优化](../performance_optimization/intro.md)定位关键路径 |
| 增量运行后结果无法复现 | 输入成员、处理依赖与复用条件 | 输入版本、配置、输出校验值、跨样本关系及重试记录 | [06 数据版本、增量处理与下游验证](./06_pipeline_lineage_and_downstream_validation.md) | 重算受影响阶段，核对成员完整性后再比较新旧版本 |
| 新数据处理中断或切换后需要恢复 | 正式/临时清单、失败阶段与使用方状态 | 旧版引用、临时输出、重试状态及模型或索引版本 | [06 数据版本、增量处理与下游验证](./06_pipeline_lineage_and_downstream_validation.md) | 数据读取恢复旧清单；训练另恢复检查点，检索核对匹配索引 |
| 正确率很高，但有不少问题没有评分 | 计划成员、评分覆盖及各类失败 | 计划数、已评分数、正确数、生成或评分失败记录 | [09 失败与分母](./09_model_evaluation_metrics_and_scoring.md#scoring-failures-and-denominators) | 补对应失败并同时报告覆盖、回答质量和任务完成 |
| 候选胜率提高，但平局或缺失也很多 | 同题配对、胜负平与完整成员 | 计划配对数、完成数、胜负平和未完成原因 | [10 成对比较](./10_baseline_comparison_and_regression.md#paired-model-comparison) | [Part 02 · 50](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md)核对覆盖与明确判决；无胜负时报告该状态 |
| 交换回答展示顺序后，评审结论明显变化 | 原模型与展示位置的对应、评分规则 | 原始与交换后的逐题判断、重映射和抽检 | [10 评分器复核](./10_baseline_comparison_and_regression.md#judge-review) | [Part 02 · 50](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md)检查一致性，再用依据复核分歧 |
| 总分上升，但某类原有任务下降 | 固定类别成员与逐题变化 | 总体及各类数量、分数、失败和重复结果 | [10 能力退化](./10_baseline_comparison_and_regression.md#capability-regression) | 按项目预定质量与资源要求比较，定位退化后再复测 |
