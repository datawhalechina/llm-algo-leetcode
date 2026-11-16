# 02 实时检测 Transformer：RT-DETR 的演进与部署取舍

## 学习问题

实时检测 Transformer 不是按版本号堆叠功能。`RT-DETR → RT-DETRv2 → RT-DETRv3 → RT-DETRv4` 分别围绕实时 encoder/decoder、可部署的采样与训练策略、密集监督，以及视觉基础模型辅助蒸馏等问题展开。学习重点是识别每一代改变了哪条训练或部署路径，而不是把不同论文的指标直接放在一起排名。

## 各代候选分别改变哪条训练或执行路径

| 版本/阶段 | 主要关注的问题 | 需要固定的比较条件 | 不应混用的证据 |
|:---|:---|:---|:---|
| RT-DETR | 如何让 end-to-end DETR 进入实时检测区间 | backbone、输入尺寸、数据集、runtime | 与不同硬件/后处理下的 FPS 直接比较 |
| RT-DETRv2 | 多尺度采样、算子可部署性与训练策略如何改善基线 | sampling 实现、augmentation、导出路径 | 只看训练 AP 忽略部署算子 |
| RT-DETRv3 | 更密集的正样本监督如何改善训练与检测质量 | 标签分配、loss、训练 schedule | 将监督变化误写成纯推理优化 |
| RT-DETRv4 | 如何借助视觉基础模型表示增强轻量检测器 | teacher、蒸馏配置、学生规模与评测集 | 将训练阶段蒸馏成本忽略在部署结论外 |

![RT-DETR 系列在训练、表示与部署路径上的演进](../../public/topic_discussion/multimodal/rtdetr_evolution_deployment.svg)

## 实时部署证据契约

| 字段 | 必须记录 | 为什么需要它 |
|:---|:---|:---|
| 模型 | 版本、backbone、输入尺寸、权重与训练来源 | 区分架构、训练和模型规模变化 |
| 数据 | 检测集、标注版本、类别映射与 split | 让 AP/AR 可复查 |
| 执行 | PyTorch/导出格式、runtime、precision、batch、warmup | 防止把框架差异误写成模型收益 |
| 性能 | 预处理、网络前向、后处理、端到端延迟、P95、峰值显存 | 判断实时约束落在哪一段 |
| 质量 | AP、AR、类别错误、小目标与失败样本 | 防止仅以 FPS 决策 |

## 检测候选怎样进入关键路径分析

实时检测的视觉主干、输入尺度与部署路径可以为 VLM 的视觉侧提供参照，但检测模型的对象查询与 VLM 的视觉词元是不同对象。[03 VLM 架构与图像输入契约](./03_vlm_architecture_and_input_contract.md)解释图像特征怎样成为语言模型上下文，[05 视觉与多模态关键路径](./05_multimodal_profiling_and_deployment.md)再分别测量检测与 VLM 的预处理、视觉编码、后处理和服务 I/O。

相关阅读：[RT-DETR / RT-DETRv2 官方实现](https://github.com/thaianh-pham/RT-DETRv2)、[RT-DETRv3 官方实现](https://github.com/clxia12/RT-DETRv3)、[RT-DETRv4 官方实现](https://github.com/RT-DETRs/RT-DETRv4)。

## 什么证据可以支持实时检测选型

实时检测候选卡需要包含模型版本、数据与输入条件、训练来源、导出与运行时路径、质量指标、端到端延迟、峰值显存和兼容性失败。只有这些条件可对应时，版本差异才可以进入选型判断。
