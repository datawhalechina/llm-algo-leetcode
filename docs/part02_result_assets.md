# 项目结果与证据契约

记录和读取实验结果时，先确认项目身份、比较条件、质量门槛与文件来源，再解释指标和决策。本页说明 Part02 与 Part05 共用的结果规则；具体实验状态查看各项目的结果表或专题实验资产表。

## 查找项目和格式

项目使用稳定语义 ID 标识，Notebook 数字编号用于定位页面。查找时使用以下入口：

| 要查什么 | 入口 |
| --- | --- |
| 项目身份、主归属、顺序与目标契约 | [项目注册表索引](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/benchmarks/project_registry.json)，按 Part 分文件保存 |
| 结果或产物的生产—消费关系 | [项目资产关系表](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/benchmarks/project_asset_relations.json) |
| Part02 项目的学习与实验入口 | [2.10 项目链](./02_PyTorch_Algorithms/2_10.md) |
| Part05 综合项目的输入与交付范围 | [Part05 项目入口](./05_Integrated_Projects_and_Decisions/intro.md) |

阅读链接表示知识关联；只有项目实际读取某个产物或结果时，才形成实验输入依赖。各项目在自己的负载与门槛下作出判断，可选参考报告用于说明方案来源。

## 共享实验外层

`tools.project_contracts` 提供 `project-evidence/v1` 格式，`tools.project_adapters` 将执行器的预检、运行和失败转换成共享记录。专项格式仍保留自己的指标与校验器。

| 记录 | 作用 | 主要字段 |
| --- | --- | --- |
| `experiment_spec` | 描述一次实验的对象与比较条件 | `semantic_id`、`experiment_id`、`primary_owner`、`workload`、`baseline`、`candidates`、`hardware`、`runtime` |
| `artifact_manifest` | 描述一个输入或输出产物 | `artifact_id`、`artifact_type`、`source`、`location`、`format`、`revision`、`metadata` |
| `run_result` | 保存执行状态、指标与判断 | `status`、`metrics`、`strategy_metrics`、`quality`、`artifacts`、`evidence_level`、`failure`、`decision` |

三类记录均包含 `schema_version` 和 `record_type`。共享实验使用 `experiment_id`；项目专属记录可能使用 `run_label`，应按相应契约校验。

共享 `run_result.status` 使用 `ok`、`failed`、`blocked`、`skipped`；`decision.status` 使用 `accept`、`tune`、`reject`、`pending`。执行成功说明该次调用完成，最终采用判断还取决于质量、性能、资源及所需证据。其他专项记录的状态字段以其校验器为准。

## 证据范围

`evidence_level` 说明证据来自什么验证过程。它与执行状态、最终决策分别记录。

| 取值 | 说明 |
| --- | --- |
| `theoretical` | 公式、资源账本或静态推导 |
| `cpu_simulation` | CPU 状态机、合成时间线或成本模型 |
| `cpu_mechanism_validation`、`cpu_project_model` | CPU 机制核对或项目模型验证 |
| `gpu_smoke` | GPU 接口或局部执行路径的小规模检查 |
| `local_mechanism_benchmark` | 固定局部机制负载下的测量 |
| `real_benchmark`、`real_project_benchmark` | 声明模型、后端和负载下的实际测量 |
| `profiler_trace` | 可回看的执行轨迹或性能剖析记录 |
| `integrated_system_evidence`、`delivery_decision` | 多项证据共同支持的系统判断或交付决策 |
| `not_recorded` | 尚未登记该项证据 |

记录中应同时说明测试了哪个对象、哪些条件保持一致、结果适用于什么范围。例如，局部内核的耗时用于判断该内核路径；完整训练步或服务收益由对应项目负载复测。

## 常用专项格式

下表列出常用的专项格式；实际项目的目标格式由注册表声明，字段与校验规则以对应模块为准。

| 格式 | 校验模块 | 适用对象 |
| --- | --- | --- |
| `fine-tuning-project/v1` | `tools/fine_tuning_result_schema.py` | 微调候选的质量、资源与产物 |
| `inference-benchmark/v1` | `tools/inference_result_schema.py` | 推理负载、策略、质量和服务指标 |
| `training-memory-benchmark/v1` | `tools/training_memory_result_schema.py` | 训练时间、显存、预算和策略比较 |
| `profiling-optimization-benchmark/v1` | `tools/profiling_result_schema.py` | 瓶颈、轨迹、优化与复核记录 |
| `quantization-evidence/v1` | `tools/quantization_result_schema.py` | 量化产物、部署与 Serving 证据 |
| `distributed-benchmark/v1` | `tools/distributed_result_schema.py` | 拓扑、通信、扩展效率与分布式结果 |
| `architecture-evaluation/v1` | `tools/architecture_result_schema.py` | 架构候选的质量、成本与来源引用 |

Part05 各项目另有最终交付格式。其注册表指向最终报告契约，准备清单、测量记录、服务配对和最终决策可以使用不同版本名；读取时逐阶段核对。

## 比较和复用前核对

| 内容 | 核对要点 |
| --- | --- |
| 比较对象 | 基线、候选和实际执行路径；发生回退时记录实际后端 |
| 输入与运行条件 | 模型及版本、数据或请求集、序列长度、batch／并发、精度、软件版本与设备 |
| 测量过程 | 预热、重复次数、配对关系、计时范围、指标单位与聚合方式 |
| 质量 | 评测数据与版本、训练／评测隔离、计算协议、阈值及失败样例 |
| 来源文件 | 生产项目、产物身份、格式、文件路径与 SHA256 |
| 判断 | 证据范围、失败原因、未完成项、采用条件与后续动作 |

产物复用时，将来源和校验值保存在相应格式规定的位置。模型、数据、协议或文件内容变化后，使用新标签重新执行相关阶段；沿用旧文件进行读取时，核对保存的身份与哈希。

## 保存与回看

原始后端报告、日志和轨迹保留其内容；归一化或汇总结果另存文件，并记录来源路径与 SHA256。相同路径已有不同内容时停止，保留原文件供复查。

报告中的已采集值、缺失证据和失败状态分别呈现。具体小节或专题资产表负责登记当前实验状态；历史审计清单留在本地交接记录中。

修改共享外层或项目最终格式后，可运行：

```bash
python -m unittest tools.tests.test_project_contracts tools.tests.test_part05_integrated_projects
```

环境准备见[环境资产索引](./gpu_environment_assets.md)，质量要求见[模型评测与跨专题质量门槛](./maintenance.md#模型评测与跨专题质量门槛)，源文件同步与发布流程见[维护手册](./maintenance.md)。
