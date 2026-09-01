# 01 Decoder-only 结构（Transformer Decoder）

## 页面目标

这一页解释为什么现代大语言模型大多采用 decoder-only 架构，以及这种结构和训练、推理、上下文建模之间的关系。

本页的输出是结构总览：明确 token 如何进入自回归 decoder，以及后续 norm、attention、RoPE、MLP 和 residual 为什么都要围绕 block 组织。

## 问题起点

Transformer 不是只有 encoder 和 decoder 两条分支，但在大模型时代，decoder-only 成了最常见的主干选择。

这一选择背后的原因通常是：

- 生成任务天然需要自回归解码
- decoder-only 更直接地对齐 next-token prediction
- 结构上更容易和推理系统、KV cache 和采样循环连接

## 演化过程

### Encoder-Decoder 时代

早期 seq2seq 任务通常依赖 encoder-decoder 结构，适合翻译和条件生成。

这一阶段的核心是把“理解输入”和“生成输出”拆成两条路径：

- encoder 负责提取整段输入的上下文表示
- decoder 负责在上下文约束下逐步生成
- cross-attention 让 decoder 能回看 encoder，但也增加了连接复杂度

这在机器翻译里非常自然，但对大规模自回归语言建模来说，它还不是最短路径。

### Decoder-only 时代

大模型预训练逐渐收敛到自回归目标，decoder-only 结构更自然地对齐语言建模。

这一步的变化不只是少了 encoder，而是训练目标发生了收敛：

- 任务从“输入到输出”变成“下一个 token 预测”
- 所有 token 共享一条 causal path
- 模型不再需要显式的 cross-attention 去连接两套表示

对大规模预训练来说，这种结构更容易和数据管线、采样循环和推理缓存对齐。

### 工程化的 decoder-only

现代 decoder-only 模型通常结合：

- causal mask
- pre-norm block
- KV cache
- 位置编码

这些因素一起决定了它在训练和推理中的表现。

更重要的是，decoder-only 已经变成一个系统接口，而不只是一个网络骨架：

- causal mask 定义了单向生成的顺序约束
- pre-norm 和 residual 决定深层训练是否稳定
- KV cache 决定推理时能否复用历史上下文
- 位置编码决定模型能否把顺序信息带进 attention

所以这一页的重点不是“decoder-only 长什么样”，而是“为什么它会成为 LLM 的默认骨架，以及这个骨架怎样和训练、推理系统互相咬合”。

## 代表模型

- `GPT` 系列：decoder-only 路线的代表
- `LLaMA`：现代开源 decoder-only block 的典型样本
- `DeepSeek`：在 decoder-only 主干上继续做结构和效率优化

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer 的起点，理解 encoder-decoder 和 decoder 结构的基础。 |
| [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) | GPT 路线的重要入口，理解 decoder-only 预训练范式。 |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | 说明 decoder-only 模型如何自然延伸到指令微调和对齐。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 现代 decoder-only block 的高质量参考实现。 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 代表 decoder-only 结构与局部窗口、长上下文结合的实践。 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 代表 decoder-only 主干上更激进的结构优化。 |

## 与 Part 02 的对应关系

- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) 直接落在 decoder-only 主干上
- [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)、[Part 02 · 03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb) 和 [Part 02 · 04 Attention（MHA/GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) 的组件都服务于 decoder-only block
- [Part 02 · 22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)、[Part 02 · 24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb) 和 [Part 02 · 67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) 讨论的 KV Cache 和 decode loop 也依赖这个结构

## 可视化提示

建议画一张 `encoder-decoder` 到 `decoder-only` 的对比图，标出：

- 输入路径
- causal mask
- self-attention
- next-token generation loop

## 进入下一页

先进入 [02 Tokenization / BPE / Embedding](./02_tokenization_embedding.md)，确认 decoder block 接收的 hidden state 从哪里来。

## 阅读建议

如果你要继续扩展，建议接着看：

- `07_mlp_ffn_evolution.md`
- `06_block_residual_path.md`
- `08_representative_models.md`
