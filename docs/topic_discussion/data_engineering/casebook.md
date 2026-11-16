# 数据工程判断手册

先固定数据版本、消费者和异常样本，再沿表格定位首个需要检查的对象。最小证据必须来自同一次数据运行；没有下游对照时，只能判断结构或规则是否通过，不能宣称模型或检索质量提高。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 文件可读取但训练或检索结果异常 | 内容类型与解析状态 | 来源身份、解析器版本、结构化输出、失败样本 | [01 采集、解析与来源契约](./01_collection_parsing_contract.md) | 用人工抽样或下游失败文档复核解析结果 |
| 去重后数据显著变少 | 重复组与相似阈值 | 被合并样本、组大小分布、误删抽样 | [02 清洗、去重与质量证据链](./02_cleaning_quality_chain.md) | 调整阈值后重跑独立质量集 |
| 训练结果很好但独立评测失真 | 切分来源与跨集合相似样本 | 数据版本、评测快照、污染报告 | [02 清洗、去重与质量证据链](./02_cleaning_quality_chain.md) | [Part 02 · 64 SFT 数据质量项目](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.md) |
| 领域能力提高、通用能力回退 | 来源、长度和难度配比 | 配方版本、各桶样本数、能力切片结果 | [03 数据配比与难度递进](./03_data_mixture_and_curriculum.md) | 固定训练配置后做配方消融 |
| 合成样本流畅但无法确认正确 | 成功条件与验证路径 | 生成身份、规则或执行日志、拒绝原因 | [04 合成数据与轨迹验证](./04_synthetic_data_and_trajectory_validation.md) | 工具轨迹转交[智能体轨迹数据与验证](../agent/01_agent_trajectory_data.md) |
| 训练损失下降但目标能力没有提升 | 标签范围与监督接口 | 输入字段、有效监督词元、失败切片 | [05 后训练数据接口与诊断](./05_post_training_data_interfaces.md) | [后训练优化](../post_training_optimization/intro.md)复测候选能力 |
| RAG 回答不可靠 | 文档结构、切块与知识版本 | 解析记录、切块身份、检索失败样本 | [00 数据生命周期与消费者契约](./00_data_lifecycle_and_contract.md) | [推理优化](../inference_optimization/intro.md)验证召回、延迟和回答质量 |
| 训练吞吐不稳定 | 长度分布、打包和数据就绪时间 | 批次长度、等待时间、主机—设备搬运记录 | [03 数据配比与难度递进](./03_data_mixture_and_curriculum.md) | [性能优化](../performance_optimization/intro.md)定位关键路径 |
| 增量运行后结果无法复现 | 输入—规则—输出三元组 | 输入快照、规则版本、输出校验和、失败记录 | [06 流水线血缘与下游验证](./06_pipeline_lineage_and_downstream_validation.md) | 重放受影响对象，并与上一数据版本做下游对照 |
