# 多模态与视觉推理判断手册

先确认问题属于检测还是 VLM，并固定图像版本、预处理、模型或视觉塔、处理器和质量协议。表中的“最小证据”缺失时，不比较跨模型性能。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 分辨率变化后质量和显存同时改变 | 图像尺寸、图块大小与视觉表示长度 | 输入卡、视觉特征/词元数、峰值显存 | [00 视觉表示](./00_visual_representation_cnn_to_vit.md) | 固定质量集后做分辨率配对复测 |
| 检测输出框数量或类别异常 | 标注映射、候选筛选与匹配 | 数据版本、原始候选、后处理配置 | [01 检测范式](./01_detection_yolo_to_detr.md) | [数据工程](../data_engineering/intro.md)检查标注身份 |
| 实时检测 FPS 高但端到端延迟不达标 | 预处理、前向和后处理阶段 | 分阶段时间、P95、AP/AR 与失败图像 | [02 实时 DETR](./02_realtime_detr_evolution.md) | [05 关键路径](./05_multimodal_profiling_and_deployment.md) |
| VLM 更换处理器后回答退化 | 图像预处理、视觉词元和模板 | 模型/视觉塔/处理器版本、输入序列、失败回答 | [03 VLM 输入契约](./03_vlm_architecture_and_input_contract.md) | 回滚输入契约并重跑独立质量集 |
| 图像请求首词元明显变慢 | 媒体读取、视觉编码与预填充 | 图片规格、视觉词元、阶段时间、TTFT | [04 多模态请求状态](./04_vlm_workload_and_cache.md) | [推理优化](../inference_optimization/intro.md)检查预填充与调度 |
| 缓存命中但端到端收益很小 | 命中的对象与关键路径占比 | 缓存对象、命中率、节省时间和全链路延迟 | [04 多模态请求状态](./04_vlm_workload_and_cache.md) | [性能优化 · 瓶颈归因](../performance_optimization/methods/05_bottleneck_attribution.md)确认瓶颈是否转移 |
| 候选能加载但执行路径发生回退 | 运行时支持与实际计算内核 | 产物身份、后端版本、日志、执行轨迹 | [05 关键路径](./05_multimodal_profiling_and_deployment.md) | [部署与异构](../deployment_heterogeneous/intro.md)检查兼容门槛 |
| 需要决定候选是否上线 | 分支内基线、质量、资源和恢复路径 | 配对结果、失败切片、产物清单、回滚版本 | [06 交付决策](./06_multimodal_delivery_decision.md) | [Part 05 综合项目](../../05_Integrated_Projects_and_Decisions/intro.md) |
