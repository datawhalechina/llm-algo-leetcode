# 数据工程与模型评测术语表

本表用于回查概念和容易混淆的计数。学习入口指向正文中的具体解释；需要实现时，再沿对应正文进入Part练习。字段名和通用缩写保留原拼写。

| 中文主称 | 英文 / 缩写 | 本专题含义 | 容易混淆之处 | 学习入口 |
|:---|:---|:---|:---|:---|
| 来源与样本编号 | source ID / sample ID | 来源编号关联同一文档的不同形式，样本编号定位某条记录 | 同一来源可包含原文件和派生问答，文件名可能另有变化 | [00 从文件到记录](./00_data_lifecycle_and_contract.md#从原始文件到一条可复查记录) |
| 数据快照 | data snapshot | 固定成员与内容版本的一组输入或输出记录 | 目录继续变化时，按保存的成员清单与校验值找回旧版本 | [00 版本与处理记录](./00_data_lifecycle_and_contract.md#从原始文件到一条可复查记录) |
| 数据血缘 | data lineage | 输入、处理规则、输出和使用任务之间的关系 | 具体记录问答来自哪页、经哪条规则生成，流程图只能概括关系 | [00 来源到问答的关系](./00_data_lifecycle_and_contract.md#从原始文件到一条可复查记录) |
| 使用方的输入要求 | — | 训练、检索或评测需要的字段、格式和含义 | 同为JSONL的文件可能要求不同字段和样本组织 | [00 不同用途的输入](./00_data_lifecycle_and_contract.md#同一段说明怎样形成不同输入) |
| 行为对齐 | alignment | 通过示范或反馈等方式，让模型行为更符合任务目标、偏好和安全约束 | 标签与预测位置对齐检查的是损失中的位置对应，二者作用不同 | [00 行为目标与数据选择](./00_data_lifecycle_and_contract.md#behavior-alignment) |
| 来源使用范围 | — | 来源记录允许哪些用途、使用方与访问条件 | 获得文件、允许查询、允许训练和允许公开发布分别核对 | [00 来源与访问范围](./00_data_lifecycle_and_contract.md#source-use-and-access) |
| 脱敏 | de-identification / redaction | 移除或替换任务不需要的敏感内容，按用途检查残留信息 | 替换一个姓名后仍可能从其他内容识别个人，原始内容与映射另行限制访问 | [02 敏感内容处理](./02_cleaning_quality_chain.md#sensitive-content) |
| 过滤保留情况 | — | 按来源、语言、长度或任务比较处理前数量与保留量 | 保留记录逐条合格，也可能丢失某一类任务覆盖 | [02 过滤前后比较](./02_cleaning_quality_chain.md#filtering-coverage) |
| 文件校验值与文本指纹 | checksum / text fingerprint | 前者由文件字节计算，后者按约定的文本处理规则计算 | 扫描件和文字版文件不同，提取后的正文可能相同 | [02 文件与文本重复](./02_cleaning_quality_chain.md#duplicate-and-source-groups) |
| 光学字符识别 | OCR | 将页面图像中的文字识别为文本 | 数字、否定词和表格对应关系需要抽样核对 | [01 文字层与扫描件](./01_collection_parsing_contract.md#根据内容选择解析方式) |
| 数据污染 | evaluation contamination | 训练、合成或调参过程接触了应保持独立的评测内容 | 重复关系是否构成泄漏，要结合任务的隔离要求判断 | [02 按关联关系切分](./02_cleaning_quality_chain.md#split-related-records) |
| 重复组与来源组 | duplicate group / source group | 重复组关联内容重复的记录，来源组关联同一来源的派生记录 | 去重选择保留成员，切分按目标任务隔离关联样本 | [02 重复与来源分组](./02_cleaning_quality_chain.md#duplicate-and-source-groups) |
| 数据配方 | data mixture | 来源、领域、长度、难度、采样权重和顺序的组合 | 数据池名称不说明训练中各类样本实际出现多少次 | [03 配比与顺序](./03_data_mixture_and_curriculum.md#mixture-order) |
| 文件分片与读取进度 | shard / reading progress | 分片将文档分批存储，读取进度定位来源、文档及文档内词元位置 | 已预取的内容可能还未用于更新，恢复进度应与训练检查点对应 | [03 文档怎样组成窗口](./03_data_mixture_and_curriculum.md#pretraining-reading-progress) |
| 任务采样 | task sampling | 从任务池选择提示，并按配置采集回答或交互尝试 | 题目数、回答次数、生成词元与训练使用量分别统计 | [03 在线任务采样](./03_data_mixture_and_curriculum.md#online-task-sampling) |
| 样本比例与词元比例 | sample share / token share | 分别按抽取记录数与所选词元计数统计占比 | 样本长短不同，相同样本比例会产生不同词元比例 | [03 长短样本演算](./03_data_mixture_and_curriculum.md#一半样本为什么可能占九成词元) |
| 输入词元与有效目标 | input tokens / valid targets | 非填充输入词元计量输入位置，有效目标计量进入损失的预测 | 仅回答监督时，提示作为上下文，提示目标可不进入损失 | [05 三种位置计数](./05_post_training_data_interfaces.md#上下文可见与参与损失分别控制) |
| 标签掩码与注意力掩码 | label mask / attention mask | 前者选择损失目标，后者限制可读取的输入位置 | 忽略提示目标时仍保留问题上下文；标签移位按实际接口检查 | [05 上下文与损失](./05_post_training_data_interfaces.md#上下文可见与参与损失分别控制) |
| 分词与模板检查 | tokenization / chat template | 查看真实词元、角色边界、结束标记及截断后内容 | 字符数、词元数与反解显示各有含义，特殊标记需同时查看 | [05 分词后的输入](./05_post_training_data_interfaces.md#tokenization-checks) |
| 预训练目标与回答目标 | — | 自回归预训练预测语料后续词元，回答监督按选定范围预测示范回答 | 有效目标数量相同，也可能在学习不同位置的内容 | [05 两类目标的短序列](./05_post_training_data_interfaces.md#pretraining-and-sft-targets) |
| 样本打包 | packing | 把多条样本放入一个窗口以减少填充 | 独立样本需保留注意力与标签边界，EOS本身不建立隔离 | [05 截断与打包](./05_post_training_data_interfaces.md#截断和样本打包会改变什么) |
| 合成数据 | synthetic data | 由模型、规则或工具生成，并保留生成配置与验证记录的数据 | 格式、答案正确性和重复/隔离关系分别检查 | [04 任务与验证方法](./04_synthetic_data_and_trajectory_validation.md#生成前先确定任务与检查方法) |
| 标注规则与分歧复核 | annotation guidelines / adjudication | 约定示范与反馈怎样判断，保存不同判断及复核结论 | 偏好对先核对同一上下文，再区分优劣、相当和无法判断 | [04 人工示范与偏好](./04_synthetic_data_and_trajectory_validation.md#human-demonstrations-and-preferences) |
| 验证器 | verifier | 执行规则、测试或评审以检查候选的程序或过程 | 答案错误与验证超时分别记录，模型评审另查分歧 | [04 检查方式](./04_synthetic_data_and_trajectory_validation.md#verification-methods) |
| 执行轨迹 | execution trace / trajectory | 一道任务中生成、工具调用、环境返回和结束等过程记录 | 在线训练额外要求算法所需的策略、采样与奖励记录 | [04 工具交互与中间步骤](./04_synthetic_data_and_trajectory_validation.md#interaction-records) |
| 策略与采样版本 | policy / behavior policy version | 策略决定模型在当前上下文下怎样选择词元或动作，版本记录本次采样使用的模型 | 更新时的当前策略可能已经变化，记录是否可用按算法和采样条件判断 | [04 策略与交互记录](./04_synthetic_data_and_trajectory_validation.md#interaction-records) |
| 奖励 | reward | 按规则、奖励模型或环境反馈，对回答或动作给出的数值评价 | 总分随规则改变；原始检查结果、评分失败与已确认的零分分别保存 | [04 从检查结果到奖励](./04_synthetic_data_and_trajectory_validation.md#reward-and-feedback) |
| 回报 | return | 从当前动作开始累计的后续奖励，可按约定折扣 | 最后才给出的奖励也可能影响前面的动作；结束与截断处理会影响计算 | [05 奖励与后续回报](./05_post_training_data_interfaces.md#reward-return-advantage) |
| 优势 | advantage | 动作或回答相对所选基准的表现，用于构造更新信号 | 基准、缩放和估计方法由算法决定，计算时需保留相应配置与记录 | [05 组均值短例子](./05_post_training_data_interfaces.md#reward-return-advantage) |
| 学生在策略蒸馏 | on-policy distillation / OPD | 当前学生生成回答，教师对学生访问的上下文提供反馈，再据此训练学生 | 回答的生成者与反馈的提供者分别记录；已有教师答案属于另一种轨迹来源 | [04 学生采样与教师反馈](./04_synthetic_data_and_trajectory_validation.md#student-sampling-teacher-feedback)、[05 预测位置对应](./05_post_training_data_interfaces.md#distillation-positions) |
| 验证集与最终测试集 | validation / test split | 验证集用于选规则或配置，最终测试集在选择完成后用于比较 | 根据测试分数反复调参后，需要重新安排独立比较 | [02 验证与最终比较](./02_cleaning_quality_chain.md#split-related-records) |
| 评测快照与切片 | evaluation snapshot / slice | 快照固定评测成员与内容，切片按语言、长度或任务划分样本 | 同时检查总体分数、各类样本数量及失败，避免漏掉难样本 | [02 评测成员与失败统计](./02_cleaning_quality_chain.md#验证规则与比较模型分别使用哪些数据) |
| 下游对照 | downstream comparison | 声明变化变量后，比较数据版本对训练、检索或性能的影响 | 字段检查通过回答能否读取，实际任务比较回答效果如何变化 | [06 按用途复测](./06_pipeline_lineage_and_downstream_validation.md#version-choice) |
| 增量运行 | incremental processing | 按输入和处理依赖重算受影响结果，并保存复用依据 | 跨样本的去重、配比或索引更新可能需要重算整组或整个阶段 | [06 逐文件与跨样本依赖](./06_pipeline_lineage_and_downstream_validation.md#哪些步骤可以只处理变化部分) |
| 版本切换与恢复 | version switch / recovery | 让使用方选择已检查的成员清单及匹配文件，失败时恢复对应版本 | 模型参数需检查点才能恢复；检索还要核对文档与索引版本 | [06 版本选择与恢复](./06_pipeline_lineage_and_downstream_validation.md#version-choice) |
| 采样与训练使用状态 | — | 记录某次尝试是否生成、获得反馈、进入训练及对应更新 | 生成完成与更新已保存是不同状态；工具重试还要检查动作是否已执行 | [06 在线批次恢复](./06_pipeline_lineage_and_downstream_validation.md#online-batch-recovery) |
| 数据分布变化 | data distribution shift | 新旧数据在来源、语言、长度、任务或失败情况上的变化 | 先检查变化原因，再决定补数、调采样或重新评测 | [06 持续数据比较](./06_pipeline_lineage_and_downstream_validation.md#changing-data-distribution) |
| 检索切块 | chunking | 将文档划为可单独检索的片段，并保留标题、位置与适用条件 | 切得更短会减少单块长度，也可能拆开答案依据 | [07 片段边界](./07_retrieval_chunks_and_index_updates.md#documents-and-chunks) |
| 检索索引 | retrieval index | 为查找片段组织的词项、向量等数据结构 | 片段正文、嵌入模型与索引各有版本；删除文档还要处理可检索成员 | [07 词项与向量检索](./07_retrieval_chunks_and_index_updates.md#retrieval-index) |
| 推理测试负载 | inference workload | 一次测试的请求集合、生成设置及发送方式 | 请求总数相同，长度、前缀共享或到达安排不同，执行成本也会变化 | [08 请求内容与长度](./08_evaluation_workloads_and_calibration.md#inference-workload) |
| 固定并发与固定到达速率 | concurrency / arrival rate | 前者控制同时在途的请求数，后者按时间安排请求到达 | 固定并发下后续发送会受完成速度影响；按时间发送还要记录实际偏差 | [08 请求何时到达](./08_evaluation_workloads_and_calibration.md#arrival-and-failure) |
| 量化校准样本 | calibration data | 为需要校准的量化方法提供输入，参与数值统计或参数确定 | 校准参与方案构建，最终质量用独立评测检查；需求按方法选择 | [08 校准输入](./08_evaluation_workloads_and_calibration.md#calibration-samples) |
| 评测单位 | evaluation unit | 一条回答、一次任务或一个动作等被评分与统计的对象 | 同题多次生成与多个独立问题作用不同，采样次数与成本一并记录 | [09 目标与任务单位](./09_model_evaluation_metrics_and_scoring.md#evaluation-goals) |
| 评分准则 | scoring rubric | 规定哪些要求算满足、各维度怎样评分及如何处理分歧 | 格式、事实、任务完成与安全分别说明，多个分数合成前声明权重 | [09 逐样本评分](./09_model_evaluation_metrics_and_scoring.md#per-sample-scoring) |
| 可评分覆盖 | scoring coverage | 计划任务中已得到可用评分的比例 | 回答生成完成后也可能评分失败，未评分与答错分别保存 | [09 失败与分母](./09_model_evaluation_metrics_and_scoring.md#scoring-failures-and-denominators) |
| 成对比较与非平局胜率 | paired comparison / decided win rate | 对应同题双方分数，统计胜负平并在明确胜负中计算胜率 | 平局、覆盖与缺失单独报告，全平局时没有明确胜负可算 | [10 同题比较](./10_baseline_comparison_and_regression.md#paired-model-comparison) |
| 评分一致性 | judge consistency | 对同一批回答重复评分、交换展示顺序后，比较对应任务的判断 | 判断稳定仍可能判错，需要参考依据与人工或执行结果复核 | [10 评分器复核](./10_baseline_comparison_and_regression.md#judge-review) |
| 能力回归 | capability regression | 改动后原有任务表现退化 | 总分提高也可能掩盖工具、语言或长输入类别的下降 | [10 分类质量变化](./10_baseline_comparison_and_regression.md#capability-regression) |
