# 部署问题链：跟踪一个候选直到交付

这条问题链持续跟踪同一个模型候选。起点是可定位的模型与产物，终点是在固定质量和服务目标下作出的采用、调优或拒绝决定。中途任何身份变化、静默回退或环境变化都会使前后结果失去可比性。

`候选身份 → 加载门槛 → 设备执行路径 → 服务拓扑 → 回归证据 → 成本与回滚 → 交付决定`

## 哪些对象共同定义部署候选

模型名称不足以标识一次部署。模型 revision、权重或量化产物、分词器、输入模板、运行时配置和工作负载共同决定实际执行对象；其中任一项变化都要形成新的候选身份。

候选身份与加载前检查见[模型产物与部署契约](./01_model_artifact_and_deployment_contract.md)。

## 文件存在为什么仍可能无法执行

推理后端需要识别模型结构、产物格式和配置，并为目标设备选择可用的数据类型与计算路径。声明支持、加载成功和实际命中目标计算内核是三层不同证据；静默反量化或回退只能算功能路径，不能算低精度执行收益。

![从模型框架到服务运行时的执行路径](../../public/topic_discussion/deployment_heterogeneous/framework_runtime_path.svg)

加载与后端差异见[推理后端兼容性与单机执行](./02_backend_compatibility_and_single_node_execution.md)，设备能力门槛见[加速器与运行时兼容性](./07_accelerator_runtime_compatibility.md)。

## 请求和状态实际经过哪些设备

输入准备、权重驻留、KV 状态、计算结果和输出回写可能分布在主机、GPU、其他加速器或网络节点。只有位于关键路径的搬运与同步才会改变端到端时间；仅观察设备利用率不能确定瓶颈。

![设备协同：请求、搬运、计算、状态与输出](../../public/topic_discussion/deployment_heterogeneous/device_coordination_path.svg)

需要记录对象驻留、传输字节、同步事件和阶段时间，具体方法见[设备协同与数据路径](./03_device_coordination_and_data_path.md)。

## 单实例不足时应该改变什么

容量、吞吐或尾延迟不达标时，候选可能增加服务副本、切分模型、分离预填充与解码，或改变端—边—云放置。不同方案改变的对象并不相同，因此必须分别记录状态所有权、交接字节、网络等待、故障域和降级路径。

![单实例之外的服务形态选择](../../public/topic_discussion/deployment_heterogeneous/service_tier_selection.svg)

多设备和阶段分离见[多设备与服务分层](./04_multi_device_and_service_tiers.md)，地域与设备层级变化见[端—边—云放置与状态交接](./08_edge_cloud_placement_and_handoff.md)。

## 怎样确认升级没有改变质量或执行路径

发布前后的结果只有在模型、产物、输入协议、工作负载和统计方式可对应时才可比较。运行日志需要同时保留质量失败、加载失败、能力回退、资源不足和性能回归，失败类型决定阻断、复测、降级还是回滚。

配对回归与失败处理见[可观测性、回归与发布](./05_observability_regression_and_release.md)。

## 哪个候选值得交付

先淘汰身份不完整、无法加载、质量不合格或路径不明的候选，再比较有效吞吐、尾延迟、容量、单位成本和维护代价。成本只对满足质量门槛并稳定完成请求的结果有意义；失败运行不能通过低成本获得优势。

![部署候选如何经过质量、资源、服务目标与成本过滤](../../public/topic_discussion/deployment_heterogeneous/deployment_decision_funnel.svg)

候选比较方法见[异构选型与成本决策](./06_heterogeneous_selection_and_cost_decision.md)。最终交付还要保存可执行配置、产物清单、环境快照、失败记录、降级路径和回滚版本。
