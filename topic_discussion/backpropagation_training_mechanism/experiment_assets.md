# 反向传播与训练机制实验资产表

本表登记训练机制实验的环境、工作负载、指标和证据状态。环境创建见[实验 Quickstart](./quickstart.md)，Profile 与隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)，实际依赖与硬件写入结果 JSON。

## 直接运行环境

本专题的六个机制级 GPU 实验只使用一个环境。CPU 机制练习不需要单独创建 GPU Profile；后训练和多卡实验是后续跨专题入口，不属于本专题的安装要求。

| 环境 ID | 直接入口 | 可形成的证据 |
|:---|:---|:---|
| `TRAIN-CPU` | Part 02 · 09–12、混合精度反向、55、56、19 的框架一致性核对 | 原生算子、Accelerate、Transformers 与 checkpoint 行为 |
| `TRAIN-CUDA-MECHANISM` | Part 02 · 11、TRAIN-MIXED-PRECISION-BACKWARD、55、56、19、42 | 单卡梯度累积、AMP、更新幅度、持续预训练、检查点和卸载的正确性、峰值显存与单步时间 |

`TRAIN-CUDA-MECHANISM` 不安装 Serving、多卡编排或量化产物工具。六个 GPU 实验共用 `Qwen/Qwen2.5-0.5B` 的同一不可变 commit，以固定文本构造训练批次；它们验证真实模型计算图上的局部机制，完整训练和策略组合进入项目页。

### 跨专题环境入口

| 后续任务 | 独立 Profile | 进入条件 |
|:---|:---|:---|
| 完整 SFT 与 LoRA | `PT-SFT-CUDA` | 需要真实模型、数据集和训练产物时进入后训练专题 |
| 偏好优化与 RL | `PT-RL-CUDA` | 需要 rollout、奖励或策略更新时进入后训练专题 |
| 多卡同步与切分 | `PAR-CUDA-MECHANISM` | 需要 collective、状态切分或扩展效率时进入通信与并行专题 |

## 机制实验注册表

| 实验 ID / 页面 | 固定对照 | 主要指标 | 当前状态 |
|:---|:---|:---|:---|
| `TRAIN-LOCAL-BACKWARD` / Part 02 · 09 | 手写局部梯度 → Autograd | 输出、输入梯度、参数梯度、误差 | CPU 机制完整；不需要独立 GPU benchmark |
| `TRAIN-ATTENTION-BACKWARD` / Part 02 · 10 | 手写 Attention backward → Autograd | Q/K/V 梯度、保存状态、数值误差 | CPU 机制完整；真实 Kernel 归算子优化 |
| `TRAIN-GRADIENT-ACCUMULATION` / Part 02 · 11 | 完整批次 → 2 / 4 次微批次累积 | 参数/loss 差异、绝对与增量峰值、step median/P95 | RTX 5070 Ti、Qwen2.5-0.5B 的序列 128 / 256 对照已验证；短序列为 `tune`，序列 256 最多降低约 17.3% 峰值并增加约 33.5% median |
| `TRAIN-SCHEDULER-CLOCK` / Part 02 · 12 | 成功更新与跳步下的 WSD 时钟 | Warmup/Stable/Decay 边界、有效更新计数 | CPU 机制完整；无需独立 GPU benchmark |
| `TRAIN-UPDATE-DYNAMICS` / Part 02 · 55 | 基准学习率 / 高学习率无裁剪 / 高学习率裁剪 | 有限值、梯度范数、裁剪系数、参数增量、峰值、step median/P95 | GPU 实验已准备；局部稳定性不替代独立质量与保留集评测 |
| `TRAIN-CONTINUAL-PRETRAIN` / Part 02 · 56 | 0 更新基线 → 固定步数全序列领域续训 | 领域 / 保留 token loss、有限值、checkpoint 字段、峰值、step median/P95 | GPU 实验已准备；内置小文本只验证测量链，真实结论需版本化领域与保留集 |
| `TRAIN-AMP-BACKWARD` / 混合精度反向 | FP32 → AMP 与溢出样本 | 有限值、scale、跳步、峰值、单步时间 | RTX 5070 Ti、Qwen2.5-0.5B 的 `default14` 已验证；FP16 显存下降约 10.7%，时间基本持平 |
| `TRAIN-CHECKPOINT` / Part 02 · 19 | 普通保存 → 原生 gradient checkpointing | loss/梯度范数、峰值、单步时间 | RTX 5070 Ti、Qwen2.5-0.5B 的 `b2s512_default2` 已验证；allocated 峰值降低约 33.7%，median 增加约 19.2%，决策为 `accept` |
| `TRAIN-OFFLOAD` / Part 02 · 42 | GPU 驻留 → host offload | loss/梯度范数、D2H/H2D、峰值、单步时间 | RTX 5070 Ti、Qwen2.5-0.5B 的 `b2s512_default1` 已验证；allocated 峰值降低约 34.3%，median 增至 5.87 倍，决策为 `tune` |
| `TRAIN-STATE-LEDGER` / Part 02 · 43 | 基线生命周期 → 策略后生命周期 | 分阶段驻留、峰值对象、理论峰值 | CPU 账本完整；真实峰值由 Part 02 · 19、42、76 提供 |

## 项目接口

| 稳定项目 ID | 输入机制资产 | 项目补充的证据 |
|:---|:---|:---|
| `TRAIN-BASELINE` / Part 02 · 73 | 监督、有效更新与 AMP | 固定模型和数据的训练基线、吞吐与质量 |
| `TRAIN-PROFILING-DIAGNOSIS` / Part 02 · 74 | 生命周期账本与 trace 字段 | 主导状态、关键路径和候选归因 |
| `TRAIN-MEMORY-STRATEGY` / Part 02 · 76 | checkpoint、offload、AMP | 单策略与组合策略的质量—容量—时间决策 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、页面与候选 |
| 工作负载 | 模型 ID 与 commit、tokenizer、输入 shape、有效 batch、更新次数、dtype、随机种子 |
| 环境 | Profile ID、Python、PyTorch、CUDA、GPU 与显存 |
| 正确性 | 输出/梯度误差、有限值、跳过更新与更新次数 |
| 资源与性能 | 峰值 allocated/reserved、单步 median/P95、搬运量或重算次数 |
| 结论 | `failure`、`evidence_level`、decision 与下一步 |
