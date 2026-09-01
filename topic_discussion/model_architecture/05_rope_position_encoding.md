# 05 RoPE 与位置编码（RoPE / Position Encoding）

## 页面目标

这一页回答的是：位置信息如何进入 attention，以及 RoPE 和长上下文扩展为什么会同时影响结构表达与推理成本。

本页的输出是位置接口：明确位置编码与 Q/K 变换、上下文长度和 KV cache 之间的关系。

这一页解释位置编码为什么从绝对位置走向相对位置，再走向 RoPE，以及为什么长上下文模型又必须继续改造 RoPE。

## 问题起点

attention 本身不带顺序感，因此位置编码是“让模型知道 token 顺序”的必要补充。

位置编码解决的问题包括：

- token 顺序如何进入 attention
- 相对位置信息如何表达
- 长上下文如何扩展而不显著退化

## 演化过程

### 绝对位置编码

最早的做法是给每个位置一个明确的编码，但它的可泛化性和长上下文适配都有限。

这种方案的优点是直观，但问题也很直接：

- 模型更容易记住“第几个位置”
- 却不一定容易理解“两个 token 之间相隔多远”
- 一旦上下文长度超出训练分布，表现就容易下滑

因此，绝对位置编码更像是“先让模型知道顺序存在”，但还不足以让它稳定地理解相对关系。

### 相对位置编码

相对位置方法更关注 token 之间的相对关系，便于更自然地表达上下文结构。

这一步的意义在于把“位置”从绝对坐标改成关系坐标：

- 模型关注的是 token 间距离
- 不同长度的序列可以共享更一致的位置表达
- 对长文本和变长输入更友好

这也是为什么相对位置方法会成为后续改造的重要过渡层。

### RoPE

RoPE 把位置信息融入 query / key 的旋转关系中，成为现代 LLM 的常见默认选择。

它的优势通常体现在：

- 与 attention 结合自然
- 适合现代 LLM block
- 结构上便于扩展到更长上下文

RoPE 的强项不只是“效果不错”，而是它把位置信息嵌进了 attention 的几何结构里：

- 位置不再是额外附加的噪声，而是进入相似度计算本身
- query / key 的相对关系更容易表达顺序
- 对 decoder-only 结构来说，它的接口非常自然

这也是它在现代 LLM 中长期占优的重要原因。

### 长上下文扩展

当上下文长度上去之后，RoPE 通常需要进一步做缩放、插值或重参数化处理。

这里的核心矛盾是训练时看到的长度分布和推理时想要支持的长度分布不一致：

- 直接外推时，位置频率可能不再匹配
- 简单扩长会带来注意力退化
- 不同模型需要不同的缩放策略来保住长上下文质量

所以长上下文扩展不是“把窗口拉大”这么简单，而是要重新校准 RoPE 的位置几何。

### 工程样本：Qwen3 的分档上下文

如果要找一条现代工程样本来看 RoPE 和长上下文是怎样一起落地的，`Qwen3` 很适合。

它的价值不在于“又支持更长上下文”，而在于它把上下文长度做成了明确分档：

- 一部分较小 dense 模型维持在 `32K`
- 一部分更高档的 dense / MoE 模型直接走到 `128K`

这说明长上下文不只是“统一拉满窗口”，而是会和模型规模、部署目标、KV cache 成本一起联动。

从这个角度看，`Qwen3` 适合放在这里作为一个工程判断样本：

- RoPE 是否需要继续外推，不只取决于数学可行性
- 还取决于模型档位、cache 常驻成本和部署场景
- 同一家模型系列内部，也可能按规模和用途做不同上下文长度决策

## 代表模型

- `LLaMA`：RoPE 是其标准位置编码选择之一
- `Qwen`：会结合长上下文和工程需求看位置编码策略；`Qwen3` 是一个明确的分档样本
- `DeepSeek`：在更复杂的结构里继续使用或改造 RoPE 相关机制

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 位置编码的原始起点，理解绝对位置编码的基础。 |
| [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) | 相对位置编码的重要入口，适合和 RoPE 对照。 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | RoPE 的核心论文，是现代 LLM 位置编码的关键节点。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [Position Interpolation](https://arxiv.org/abs/2306.15595) | 代表长上下文扩展中的插值思路。 |
| [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) | 代表 RoPE 伸缩与长上下文扩展的工程路线。 |
| [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) | 代表更激进的长上下文扩展方法。 |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | 看同一家模型系列如何在 `32K / 128K` 间做上下文长度分档。 |

## 与 Part 02 的对应关系

- [Part 02 · 03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb) 直接讲 RoPE 的作用位置
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) 里可以看到 RoPE 如何嵌进 block
- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb) 和 [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb) 可以用于核对真实模型对 RoPE 的局部修正

## 可视化提示

建议画一张“位置编码演化图”：

- absolute position
- relative position
- RoPE
- RoPE scaling / interpolation / extension

最好同时标出：

- Q / K 上的位置变化
- 长上下文时为何需要重新标定

## 进入下一页

进入 [06 Block / Residual Path](./06_block_residual_path.md)，把 token、norm、attention 和 RoPE 组装回完整 block。

## 阅读建议

如果你要继续扩展，建议接着看：

- `06_block_residual_path.md`
- `08_representative_models.md`
- `09_moe_sparsity_evolution.md`
