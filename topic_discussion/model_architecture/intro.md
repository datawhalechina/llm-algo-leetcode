# 大模型架构（Model Architecture）

> 专题类型：基础支撑　主服务目标：结构理解与资源映射

## 专题定位与 Infra 层关系

本专题串起大模型结构主线：先看 token 怎样进入 block，再看 norm、attention、RoPE、MLP、MoE 和结构技巧分别放在什么位置，最后把这些结构差异接回训练、推理和显存路线。模型架构是运行在五层 Infra 之上的负载面，不等同于某一个软件层：结构决定计算图、参数规模、激活和 KV Cache 形态，Infra-L1–Infra-L3 决定它如何被执行，Infra-L4 决定它如何被服务，Infra-L5 负责评测、发布和部署治理。

本专题属于基础支撑专题，不采用主学习路线的六节正文结构；正文按 `01-09` 组织，`10_visual_assets.md` 作为图册补充。阅读时应把结构判断落回计算、内存、通信和质量指标；如果问题已经转到训练微调、推理服务或显存 trade-off，应转到对应主路线专题。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的结构与训练前置进入；如果目标是推理，可从 attention、RoPE、KV cache 相关小节切入。需要完整结构验证时，再进入 [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)。

## 前置阅读

建议先具备 [Part 00 基础](../../00_Prerequisites/intro.md) 的张量和模块基础，再按组件、Block、架构变体和 MoE 的顺序阅读 Part 02。核心入口是 [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)、[Part 02 · 02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb)、[Part 02 · 03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)、[Part 02 · 04 Attention（MHA/GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) 和 [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb)；架构变体和 MoE 作为后续扩展。

## 主学习线

`Task0-6` 是学习路线，指向 Part 02 的具体小节或项目；正文页负责解释结构关系，`10` 是图册补充。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task0 | Decoder-only、token 与 embedding 入口 | [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) | [01 Decoder-only 结构](./01_transformer_decoder.md), [02 Tokenization / BPE / Embedding](./02_tokenization_embedding.md) |
| Task1 | RMSNorm 与 SwiGLU 组件 | [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb) → [Part 02 · 02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb) | [03 归一化演化](./03_norm_evolution.md), [07 MLP / FFN 演化](./07_mlp_ffn_evolution.md) |
| Task2 | RoPE 与 Attention 结构 | [Part 02 · 03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb) → [Part 02 · 04 Attention（MHA/GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) | [04 Attention 演化](./04_attention_evolution.md), [05 RoPE / 位置编码](./05_rope_position_encoding.md) |
| Task3 | Block 组装与 residual 主干 | [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) | [06 Block / Residual 主干](./06_block_residual_path.md) |
| Task4 | 架构技巧与代表模型对照 | [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb) | [08 代表模型与结构对照](./08_representative_models.md) |
| Task5 | MoE、router 与稀疏化 | [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb) → [Part 02 · 07 MoE 负载均衡损失](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb) | [09 MoE / 稀疏化演化](./09_moe_sparsity_evolution.md) |
| Task6 | 真实模型结构审计项目 | [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb) | [08 代表模型与结构对照](./08_representative_models.md) |

## 正文与跳转

先按上面的 `Task1-6` 走结构主线；遇到“这些组件到底在 block 的哪一段”“真实模型和教科书结构差在哪”时，再回来看对应的专题正文。想看汇总版就进 [大模型结构和原理正文](./casebook.md)，想按连续故事线走一遍就进 [大模型结构和原理深入阅读](./walkthrough.md)。图册补充放在 [10 视觉资产](./10_visual_assets.md)。

如果问题已经跨到别的专题：
[监督微调与训练工程](../fine_tuning_training/intro.md) 负责 LoRA 和训练闭环，[推理优化](../inference_optimization/intro.md) 负责 attention、KV cache 和服务链路，[显存优化](../memory_performance_tuning/intro.md) 负责结构带来的资源代价。

## 环境与验证

结构阅读、参数统计和大多数组件实验可先用 CPU；若要比较真实模型的显存、吞吐或服务行为，需要 GPU 和对应推理后端。架构结论应回到参数量、计算量、显存、质量或延迟等可观测指标，不能只凭结构图判断优劣。
