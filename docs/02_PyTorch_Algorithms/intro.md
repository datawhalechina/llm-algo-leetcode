# Part 02: PyTorch Algorithm Practice | 第二部分：PyTorch 算法实战

## Part Overview | Part 概览

本部分聚焦 PyTorch 级别的大模型实现，位于 Part 0 / Part 1 之后、Part 3 之前，目标是把基础算子、模型组装、训练与对齐、显存优化、推理优化、并行策略和项目实战串成一条可运行的工程链。正文默认 notebook-first，组页负责组级资产与阅读路径，Part 级导学只管组间关系和阅读顺序，不下沉到具体节号。

Part 2 更像一张多入口学习地图：不同基础和目标的读者可以从不同组切入，最后都汇到项目实战，再按需要回补前面的训练、推理和并行内容。

## Part Asset Overview | Part 资产总览

本章内容按 9 个主题组组织，后续页面也沿该结构继续扩展。

> 导航说明：先看总览，再进入具体组页。
> 组页负责组内阅读顺序与资产收口，不需要一次性读完全部页面。
> Part 2 既是工程实战目录，也是 Part 0 / Part 1 之后、Part 3 之前的共同衔接层。

| 学习组 | 职责作用 | 当前内容映射 | 每组多少节 |
|:---|:---|:---|:---|
| [2.1](./2_1.md) | 建立基础算子和组件直觉 | [00](./00_PyTorch_Warmup.md)、[01](./01_RMSNorm_Tutorial.md)、[02](./02_SwiGLU_Activation.md)、[03](./03_RoPE_Tutorial.md)、[04](./04_Attention_MHA_GQA.md) | 5 |
| [2.2](./2_2.md) | 组装模型结构并理解 MoE 组件 | [05](./05_LLaMA3_Block_Tutorial.md)、[06](./06_MoE_Router.md)、[07](./07_MoE_Load_Balancing_Loss.md)、[08](./08_Architecture_Tricks.md) | 4 |
| [2.3](./2_3.md) | 搭起微调、调度器和训练闭环 | [09](./09_SFT_Training_Loop.md)、[10](./10_LoRA_Tutorial.md)、[11](./11_LR_Schedulers_WSD_Cosine.md)、[12](./12_Gradient_Accumulation.md)、[13](./13_End_to_End_Fine_Tuning_Experiment.md) | 5 |
| [2.4](./2_4.md) | 理解偏好优化与对齐链路 | [14](./14_RLHF_PPO_Memory.md)、[15](./15_DPO_Loss_Tutorial.md)、[16](./16_GRPO_Loss_Tutorial.md) | 3 |
| [2.5](./2_5.md) | 追踪反向传播和显存优化 | [17](./17_Autograd_Basics.md)、[18](./18_Activation_and_Loss_Backward.md)、[19](./19_Activation_Checkpointing_and_Activation_Offload.md) | 3 |
| [2.6](./2_6.md) | 建立推理加速和缓存直觉 | [20](./20_FlashAttention_Sim.md)、[21](./21_Decoding_Strategies.md)、[22](./22_vLLM_PagedAttention.md) | 3 |
| [2.7](./2_7.md) | 补充压缩与高级推理策略 | [23](./23_Speculative_Decoding.md)、[24](./24_SGLang_RadixAttention.md)、[25](./25_Quantization_W8A16.md)、[26](./26_QLoRA_and_4bit_Quantization.md) | 4 |
| [2.8](./2_8.md) | 形成并行策略和通信边界判断 | [27](./27_ZeRO_Optimizer_Sim.md)、[28](./28_Pipeline_Parallelism_MicroBatch.md)、[29](./29_Tensor_Parallelism_Sim.md) | 3 |
| [2.9](./2_9.md) | 汇总项目验证和工程闭环 | [30](./30_LoRA_Fine_Tuning_Project.md)、[31](./31_Inference_Performance_Comparison.md)、[32](./32_Training_Performance_Analysis.md) | 3 |

## Learning Path | 学习路径

Part 2 可以按多条入口理解：零基础入口先把算子、组装、训练与项目闭环串起来；训练优先、推理优先和并行优先入口则可以从不同工程目标切入，最后都回到项目实战。

### Recommended Order | 推荐顺序

- 零基础入口：先看 [2.1](./2_1.md) -> [2.2](./2_2.md) -> [2.3](./2_3.md) -> [2.5](./2_5.md) -> [2.9](./2_9.md)
- 训练优先入口：先看 [2.3](./2_3.md) -> [2.4](./2_4.md) -> [2.5](./2_5.md) -> [2.9](./2_9.md)
- 推理优先入口：先看 [2.6](./2_6.md) -> [2.7](./2_7.md) -> [2.9](./2_9.md)
- 并行优先入口：先看 [2.8](./2_8.md) -> [2.9](./2_9.md)
- 系统学习：按 [2.1](./2_1.md) -> [2.2](./2_2.md) -> [2.3](./2_3.md) -> [2.4](./2_4.md) -> [2.5](./2_5.md) -> [2.6](./2_6.md) -> [2.7](./2_7.md) -> [2.8](./2_8.md) -> [2.9](./2_9.md) 顺序推进

### Next Steps | 后续衔接

- 基础认知层：先看 [2.1](./2_1.md)、[2.2](./2_2.md)，把基础算子和模型组装先立住，再按需要进入 [2.5](./2_5.md)。
- 训练与对齐层：先看 [2.3](./2_3.md)、[2.4](./2_4.md)、[2.5](./2_5.md)，把训练、对齐和显存优化的链路理顺，主要衔接后续实现页和项目页。
- 推理与并行层：先看 [2.6](./2_6.md)、[2.7](./2_7.md)、[2.8](./2_8.md)，把推理、压缩和并行策略串起来，主要衔接项目实战与后续实现页。
- 项目收口：最后看 [2.9](./2_9.md)，把前面的知识点放回真实项目里验证和收束。

## Environment Notes | 环境说明

- 默认按 `CPU-first` 设计
- 这里只写 Part 级统一前提，不点到具体节号
- 少数 notebook 如需 `GPU optional`、`GPU required` 或多卡/完整工具链，以单页说明为准，不在导学页重复展开
