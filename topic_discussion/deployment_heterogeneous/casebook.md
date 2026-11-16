# 部署与异构决策手册

使用本表前先记录模型 revision、产物身份、分词器与模板、推理后端、设备和工作负载。先定位第一个失败阶段，再进入对应机制页；没有最小证据时不比较候选。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 模型或量化产物无法加载 | 产物格式、文件角色、哈希、模型 revision、分词器和后端版本 | 加载命令、错误阶段、产物清单与环境版本 | [01 模型产物与部署契约](./01_model_artifact_and_deployment_contract.md) | [Part 02 · 82 量化产物准入](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.ipynb) |
| 能加载但输出与参考路径不一致 | 输入模板、特殊词元、采样、停止条件和实际数据类型 | 同一输入的参考输出、候选输出与逐字段配置差异 | [02 推理后端兼容性与单机执行](./02_backend_compatibility_and_single_node_execution.md) | [Part 02 · 66 浮点服务基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) |
| 目标设备报错或发生静默回退 | 设备能力、运行时插件、数据类型和目标计算内核 | 设备身份、运行时版本、路径日志与回退原因 | [07 加速器与运行时兼容性](./07_accelerator_runtime_compatibility.md) | [Part 02 · 72 单卡部署比较](../../02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.ipynb) |
| GPU 利用率低且端到端延迟高 | 输入准备、主机—设备搬运、同步、缓存和输出回写 | 阶段时间、传输字节、同步事件与驻留对象 | [03 设备协同与数据路径](./03_device_coordination_and_data_path.md) | [性能优化 · 时间线与关键路径](../performance_optimization/methods/03_time_breakdown_and_trace_reading.md) |
| 单实例容量不足或尾延迟失控 | 权重和 KV 容量、并发、队列、通信与状态交接 | 峰值显存、队列等待、P95/P99、交接字节和失败请求 | [04 多设备与服务分层](./04_multi_device_and_service_tiers.md) | [Part 02 · 81 分布式推理](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb) |
| 端、边、云之间的网络等待影响服务 | 对象放置、往返时延、带宽、状态所有权和降级路径 | 每段网络时间、传输对象、区域容量和回退结果 | [08 端—边—云放置与状态交接](./08_edge_cloud_placement_and_handoff.md) | [通信与并行](../communication_parallelism/intro.md) |
| 更新后质量或性能回归 | 候选身份、工作负载、执行路径、失败样本和统计方式是否同时固定 | 基线—候选配对结果、质量切片、路径事件与失败分类 | [05 可观测性、回归与发布](./05_observability_regression_and_release.md) | [Part 02 · 83 量化服务配置基准](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.ipynb) |
| 多个候选都可运行但难以取舍 | 质量门槛、有效完成量、容量、尾延迟、成本与回滚代价 | 同一工作负载下的质量、请求指标、资源账本和成本假设 | [06 异构选型与成本决策](./06_heterogeneous_selection_and_cost_decision.md) | [Part 05 · 推理服务综合项目](../../05_Integrated_Projects_and_Decisions/01_Inference_Service_Integrated_Project.md) |
