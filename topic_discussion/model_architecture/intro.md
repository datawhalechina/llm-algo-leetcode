# 大模型架构与演进（Model Architecture and Evolution）

> 专题类型：主线专题　主服务目标：看懂模型的结构选择，解释它对质量、计算、显存、通信和服务方式的影响

## 页面导语

一个新模型的关键不只是参数量和 benchmark 分数，还包括它如何组织注意力、状态、归一化、激活、位置和专家路由。本专题从完整的 Transformer Block 出发，逐步进入 Attention、Norm、RoPE、MoE、SSM 和混合架构，最后训练一套阅读技术报告、拆解模型结构和做架构判断的方法。

这里的重点是“为什么这样设计”，而不是直接把每一种结构都当成优化技巧。架构专题解释结构选择及其代价；推理、性能、显存和算子专题分别负责验证这些结构在真实系统中的执行结果。

## 如何开始

- **第一次学习：** 从 Task0 开始，先掌握 Decoder-only Block 的完整数据流。
- **关注推理成本：** 从 Task1 进入，重点看 GQA、MLA、KV Cache 和注意力访问方式。
- **关注长上下文：** 从 Task3 进入，理解 RoPE、上下文扩展和位置泛化。
- **关注大规模模型：** 从 Task4 进入，学习 MoE 路由、负载均衡和通信代价。
- **关注前沿架构：** 从 Task5 或 Task6 进入，但先建立 Attention 的基线，再比较线性注意力、SSM 和混合架构。

## 主学习线

`Task0–6` 形成从基础 Block 到架构判断的连续路径。每个 Task 都需要同时回答四件事：结构是什么、解决了什么问题、引入了什么代价、如何用实验或代码验证。

| Task | 核心问题 | 主要内容 | 验证出口 |
|:---|:---|:---|:---|
| Task0：Transformer Block 与架构阅读 | 一个 decoder-only 模型由哪些结构组成？ | Token / Embedding、Attention、Residual、Norm、MLP、Position、Output Head | 画出完整 Block 数据流，并能根据配置估算参数和中间状态 |
| Task1：Attention 演进与记忆访问 | 不同 Attention 变体改变了什么？ | MHA、MQA、GQA、MLA、稀疏注意力、线性注意力、KV Cache | 对照注意力访问范围、KV 表示、计算量和缓存成本 |
| Task2：归一化、激活与 Block 稳定性 | Norm 和 Activation 为什么影响训练与执行？ | LayerNorm、RMSNorm、Pre-Norm / Post-Norm、SwiGLU、GeGLU、FFN | 比较输出、梯度、参数量、计算路径和训练稳定性 |
| Task3：位置编码与长上下文 | 模型如何表示位置，为什么长上下文会失效？ | Absolute / Relative Position、RoPE、RoPE scaling、YaRN / LongRoPE | 观察位置表示、长度扩展、质量变化和 KV / 计算代价 |
| Task4：MoE 与稀疏化设计 | 如何扩大总参数量而控制每 token 激活计算？ | Router、Top-k、Capacity、负载均衡、Token Dispatch、Expert Parallel、Shared Expert | 模拟路由、统计负载和估算 All-to-All 通信代价 |
| Task5：Attention、线性注意力与 SSM | 不同记忆机制如何处理长序列？ | Linear Attention、SSM / Mamba、Selective State、混合层排列、状态更新 | 对照 KV Cache、状态大小、并行方式和长序列成本 |
| Task6：架构选型与前沿跟踪 | 如何判断一个新模型的架构创新是否值得采用？ | 模型结构审计、技术报告阅读、质量—资源权衡、证据等级、前沿跟踪 | 完成多个模型的结构对照和一份架构选型报告 |

## 第一阶段的稳定主线与前沿扩展

| 层级 | 纳入内容 |
|:---|:---|
| 稳定主线 | Transformer Block、MHA / MQA / GQA、RMSNorm、SwiGLU、RoPE、基础 MoE |
| 深入机制 | MLA、MoE 负载均衡、Expert Parallel、长上下文扩展 |
| 前沿案例 | Native Sparse Attention、线性注意力、Mamba、混合 Attention–SSM 架构 |
| 不直接下结论 | 不把单篇论文的加速数字、单一模型的结构选择或行业趋势写成普遍规律 |

Attention、MLA 和 MoE 应优先使用原始论文与官方技术报告作为材料。DeepSeek-V2 将 MLA 与 DeepSeekMoE 作为核心架构设计，适合连接注意力记忆和稀疏专家两条线；Mamba 则适合作为选择性状态空间的代表案例，而不是 Transformer 的简单替代品。[DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[Mamba](https://arxiv.org/abs/2312.00752)

## 正文与现有材料

| 教程职责 | 现有内容 |
|:---|:---|
| Transformer 基础与输入 | [01 Decoder-only 结构](./01_transformer_decoder.md)、[02 Tokenization / BPE / Embedding](./02_tokenization_embedding.md) |
| Norm 与激活 | [03 归一化演化](./03_norm_evolution.md)、[07 MLP / FFN 演化](./07_mlp_ffn_evolution.md) |
| Attention 与位置 | [04 Attention 演化](./04_attention_evolution.md)、[05 RoPE / 位置编码](./05_rope_position_encoding.md) |
| Block 结构 | [06 Block / Residual 主干](./06_block_residual_path.md) |
| 模型对照 | [08 代表模型与结构对照](./08_representative_models.md) |
| MoE | [09 MoE / 稀疏化演化](./09_moe_sparsity_evolution.md) |
| 汇总与案例 | [Casebook](./casebook.md)、[Walkthrough](./walkthrough.md)、[10 视觉资产](./10_visual_assets.md) |

对应的代码入口包括：

- [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)
- [Part 02 · 02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb)
- [Part 02 · 03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)
- [Part 02 · 04 Attention（MHA / GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb)
- [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb)
- [Part 02 · 07 MoE 负载均衡损失](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb)
- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb)
- [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)

## 与其他主线的边界

| 专题 | 主要回答的问题 |
|:---|:---|
| 大模型架构与演进 | 为什么这样设计，结构选择解决了什么问题 |
| 推理优化 | 这个结构如何进入 Prefill、Decode、KV Cache 和服务调度 |
| 性能优化 | 运行时慢在哪里，计算、显存、通信和 I/O 成本是多少 |
| 算子优化 | 如何把某个结构对应的 kernel 实现得更快 |
| 显存优化 | 参数、激活、KV Cache 和中间状态如何容纳 |
| 量化部署 | 如何用低精度降低存储、计算和部署成本 |
| 后训练优化 | 如何通过数据和训练改变模型行为 |

同一个概念可以在多个专题出现，但职责不同。例如：

- 架构专题解释 MLA 为什么压缩 KV 表示；
- 显存专题计算 MLA 的缓存账本；
- 推理专题讨论 MLA 如何进入请求生命周期；
- 算子专题研究 MLA / Attention 的具体执行路径。

## 环境与验证

结构阅读、参数统计和大多数组件实验可以先使用 CPU。需要真实模型对照、长上下文、MoE 通信或吞吐测量时，再使用 GPU 和对应 backend。所有架构结论都应回到可观察证据：参数量、激活量、计算量、显存、通信、延迟、吞吐或质量指标。

## 建设状态

- **已完成：** 现有 Transformer、Norm、Attention、RoPE、Block、MLP、MoE 和代表模型材料的入口整理。
- **当前优先：** 定稿 Task0–2，补齐架构总览图、Attention 变体对照和 Norm / Activation 实验。
- **后续建设：** 完善长上下文、MoE 通信、SSM / 混合架构和架构选型项目。
- **当前不输出：** 不根据单篇论文或单个模型的指标，直接断言某种架构在所有任务和硬件上更优。
