# 数据工程问题链：从原始数据到训练与 RAG 证据

假设团队拿到一批网页、PDF、表格和历史对话：一部分要转成 SFT 数据，一部分要进入 RAG 知识库。真正的问题不是“能否读出文本”，而是每个下游结果能否追溯到来源、处理规则、数据版本和失败样本。

![一条数据生命周期分流到训练、RAG、评测与性能证据](../../public/topic_discussion/data_engineering/data_lifecycle_consumers.svg)

## 数据为什么要先确定消费者

先区分数据最终交给谁。训练需要样本、labels、template、split 与评测隔离；RAG 需要保留文档结构、chunk、元数据和索引输入；性能分析需要长度、packing 与 batch 就绪时间。它们可以共享来源和清洗链，但不能共用最后的消费 schema。

先填写来源、许可、版本和消费者字段，再按[00 数据生命周期与消费者契约](./00_data_lifecycle_and_contract.md)决定后续处理规则。

## 异构来源怎样变成可复查的结构化内容

文件扩展名不能决定解析路径。应先检测内容类型、编码、语言与可提取性，再路由到文本、表格、OCR 或媒体解析器；同时保存 parser 版本、降级路径和失败状态。否则“解析成功”可能掩盖表格、顺序或图像文字的丢失。

输出不是一段裸文本，而是一份带来源与解析状态的结构化记录，具体字段见[01 采集、解析与来源契约](./01_collection_parsing_contract.md)。

## 重复、污染与低质量风险怎样逐层缩小

清洗、去重、质量过滤和去污染分别处理不同问题。删除比例很高不代表质量更好；过滤器必须留下重复组、阈值、抽样复核和独立评测证据，才能判断长尾样本是否被误删、训练集是否泄漏到评测集。

完成[02 清洗、去重与质量证据链](./02_cleaning_quality_chain.md)后才能讨论样本配比，而不是把不稳定的数据直接混入训练。

## 数据配比和呈现顺序怎样接受消融

当数据可用后，训练侧仍要决定来源、领域、长度、难度与长尾样本的比例，以及是否随训练阶段调整。这里的输出不是唯一“最佳配方”，而是一份可消融的数据配方：每个候选配比都有固定版本、对照组、质量门槛和成本记录。

![数据配方从固定数据池进入下游消融](../../public/topic_discussion/data_engineering/data_mixture_validation.svg)

长度和样本打包会同时影响训练样本覆盖与训练吞吐。[03 数据配比与难度递进](./03_data_mixture_and_curriculum.md)把两类证据分别交给后训练与性能优化解释。

## 生成样本怎样获得可用证据

稀缺能力可以通过规则化任务、模型生成或工具执行补充，但候选样本必须经过与任务相匹配的验证。标准答案、单元测试、工具返回和人工偏好解决的是不同问题；验证失败也要保留原因，防止只观察成功样本。

![合成数据从任务蓝图进入验证分流](../../public/topic_discussion/data_engineering/synthetic_data_validation_flow.svg)

![规则、执行与人工复核形成不同验证路径](../../public/topic_discussion/data_engineering/trajectory_validation_routes.svg)

通过[04 合成数据与轨迹验证](./04_synthetic_data_and_trajectory_validation.md)的回答、偏好对和执行轨迹，还需完成去重、去污染和版本冻结，才能进入训练接口。

## 合格数据怎样转成后训练接口

SFT、偏好与推理数据的共同前提是可追溯、去污染和独立评测；它们的差异在于监督信号。SFT 需要 prompt/response 与 labels，偏好数据需要 chosen/rejected，推理数据还需说明过程与结果的验证方式。

![同一数据身份分流为三类后训练接口](../../public/topic_discussion/data_engineering/post_training_data_interfaces.svg)

[05 后训练数据接口与诊断](./05_post_training_data_interfaces.md)将合格数据转成不同监督接口。训练损失下降只能说明优化目标被拟合；是否真正获得目标能力，仍要回到独立评测和失败样本。

## 处理记录怎样进入下游结论

最后按消费者验证数据是否真的改善结果：训练看独立质量与行为回归，RAG 看检索/回答质量与失败文档，性能看长度、packing、CPU 准备和 H2D 是否拖慢关键路径。任一结论都要能回查数据版本、处理规则和输入分布。

![数据变更从输入快照流向下游采用判断](../../public/topic_discussion/data_engineering/pipeline_lineage_flow.svg)

[06 流水线血缘与下游验证](./06_pipeline_lineage_and_downstream_validation.md)把结构检查、质量检查和下游对照绑定到同一运行身份。三类证据能够共同回查时，数据版本才具备采用或回滚条件。
