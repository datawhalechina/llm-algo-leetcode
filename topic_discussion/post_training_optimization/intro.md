# 后训练优化（Post-Training Optimization）

> 专题类型：主学习路线　副标题：从 SFT、偏好优化到 RL 与后训练工程

## 页面导语

预训练让模型获得通用能力，后训练决定这些能力能否变成可用、可控、可评估的产品能力。本专题把后训练组织成一条连续路线：先建立 SFT 数据和训练接口，再进入偏好优化与 RL，最后回到采样引擎、资源调度和项目评估。

学习者需要逐步回答：数据是否足以支撑训练？模型是否已经具备进入偏好优化的基础？偏好信号如何转化为训练目标？RL 的奖励和资源代价是否可控？最终的能力提升是否经过可靠评估？

## 如何开始

建议按 Task0–6 顺序学习。Task0–2 建立 SFT 和训练基础，Task3–4 进入偏好优化与 RL 机制，Task5 处理训练—推理协同和工程效率，Task6 完成端到端项目评估。

没有 GPU 时，可以先完成数据契约、loss、状态转移、奖励计算和评测逻辑；完整 SFT、DPO、GRPO 和在线 RL 实验需要根据硬件和 backend 条件复测。

## 后训练优化 Task0–6

| Task | 学习目标与核心问题 | 主要内容 | 交付物 | 现有入口 / 正文建设 |
|:---|:---|:---|:---|:---|
| Task0 | 后训练全景与三阶段认知：预训练之后，模型还需要经历什么才能变成“有用”的？ | SFT（教格式）→ 偏好优化（对齐偏好）→ RL（探索和突破数据上限）；阶段顺序、适用条件与 prompt / RAG 的边界 | 一张从 base model 到可用模型的后训练流程图，标注每阶段输入、输出和目标 | [后训练全景](./00_post_training_landscape.md)；连接 Part02 · 2.3、2.4 |
| Task1 | SFT 数据工程：数据从哪里来、怎么筛、怎么配比？ | 人工、开源和合成数据；多样性、难度、格式、推理质量；去重、泄漏、偏见、配比和数据诊断 | 一份面向具体场景的高质量 SFT 数据集，附来源、筛选标准和配比设计 | [SFT / LoRA 基础模块](./sft_foundation/intro.md)；补充 Part02 · 32、33、50 |
| Task2 | SFT 训练实践：全参、LoRA、QLoRA 如何选择和控制？ | PEFT、LoRA 配置、QLoRA、DoRA / rsLoRA、学习率、框架选择、过拟合和早停 | 一次 LoRA 或 QLoRA 训练，附配置理由、loss 曲线和验证集结果 | [LoRA / PEFT 设计](./sft_foundation/02_lora_peft_design.md)；连接 Part02 · 09、10、60、65 |
| Task3 | 偏好优化：DPO 为什么能去掉奖励模型？ | DPO 推导、手写 loss、β、偏好数据、DPO 变体及其局限 | 手写 DPO loss 并完成小规模偏好训练，比较训练前后的偏好变化 | [DPO 与偏好优化](./03_dpo_and_preference_optimization.md)；连接 Part02 · 15、50、84、86 |
| Task4 | RL 对齐：GRPO 与 PPO 如何组织奖励和训练循环？ | PPO 四模型代价、GRPO 组内优势、可验证奖励、奖励设计、RLVR 和稳定化策略 | 用 TRL 完成一个可验证任务的 GRPO 训练，分析组内奖励分布 | [PPO 系统代价](./02_rlhf_and_ppo_system_cost.md)、[GRPO 机制](./04_grpo_and_groupwise_alignment.md)；连接 Part02 · 14、16、85 |
| Task5 | RL 工程实现与效率：采样、训练和资源如何协同？ | verl / OpenRLHF / TRL；vLLM / SGLang rollout；Ray 编排；训练—推理资源调度、流水线重叠、多轮工具调用 RL | 跑通从采样到更新的 RL 流程，记录显存、吞吐、采样和更新代价 | 新建“RL 工程与效率”正文；连接 Part04 · 4.3、Part02 · 2.9 |
| Task6 | 综合项目与评估：怎样证明后训练真的有效？ | 指令遵循、安全、诚实、鲁棒性、对齐冲突、reward hacking、端到端报告 | 一份包含数据、训练、评估、失败案例和 reward hacking 分析的后训练报告 | [项目决策与交付](./06_project_decision_and_delivery.md)；连接 Part02 · 2.10、50–52、84–86 |

## 三阶段的顺序

```text
Base model
    ↓
SFT：建立任务格式、行为接口和可用基线
    ↓
偏好优化：让模型在多个可行输出中更符合目标偏好
    ↓
RL / RLVR：通过探索、环境反馈或可验证奖励继续扩大能力
    ↓
评估、部署与迭代
```

顺序不是绝对的训练配方，而是学习和诊断时的默认假设：SFT 基础不稳定时，偏好优化容易放大格式和数据问题；偏好信号、评测和奖励约束不清晰时，RL 容易把代理指标优化成 reward hacking。实际项目中，如果 prompt、RAG 或工具调用已经足够解决问题，不应为了使用训练方法而强行进入 SFT 或 RL。

## 专题正文建设计划

专题正文与 Task 不要求一一对应。正文负责解释概念、机制和决策关系，Notebook 负责实现和验证，项目页负责形成证据。

| 正文职责 | 需要承载的内容 | 当前状态 |
|:---|:---|:---|
| 全景与阶段判断 | 三阶段输入 / 输出、顺序、prompt / RAG / SFT / 对齐 / RL 的选择 | 新增 `00_post_training_landscape.md` |
| 数据工程 | 数据来源、筛选、合成、配比、泄漏和质量诊断 | 由 SFT 基础页扩展 |
| SFT 训练 | LoRA / QLoRA、训练控制、过拟合和配置决策 | 复用 `sft_foundation` 并补充正文关系 |
| 偏好优化 | DPO 数学、β、数据构造、变体和适用边界 | 已有 DPO 正文，需按新 Task3 归位 |
| RL 对齐 | PPO、GRPO、RLVR、奖励设计和稳定性 | 已有 PPO / GRPO 正文，需合并为 Task4 机制链 |
| RL 工程 | rollout、训练引擎、资源调度、通信和流水线重叠 | 新建 Task5 正文 |
| 项目评估 | 质量、安全、鲁棒性、reward hacking、报告和采用决策 | 已有项目决策页，需扩展评估维度 |

## 与其他专题的边界

- [SFT / LoRA 基础模块](./sft_foundation/intro.md)：提供 Task0–2 的训练前置和项目接口。
- [性能优化](../performance_optimization/intro.md)：解释训练吞吐、显存、通信和稳定性证据。
- [显存优化](../memory_performance_tuning/intro.md)：解释 PPO / GRPO 的模型驻留和预算约束。
- [推理优化](../inference_optimization/intro.md)：提供 rollout、Serving、KV Cache 和 backend 机制。
- [通信与并行](../communication_parallel/intro.md)：提供 RL 训练的多卡切分、通信和扩展效率方法。
- [模型架构](../model_architecture/intro.md)：解释模型结构、MoE 和 reasoning 分支的基础。

## 项目证据

后训练项目不能只比较训练 loss。至少需要同时记录数据口径、训练配置、偏好或奖励指标、任务质量、安全与鲁棒性、显存与吞吐、采样成本、失败案例和最终 `accept / tune / reject` 决策。

推荐的项目闭环是：`SFT / LoRA 基线 → 偏好优化 → RL 或 RLVR → 独立评估 → 资源与成本判断 → 迭代或交付`。
