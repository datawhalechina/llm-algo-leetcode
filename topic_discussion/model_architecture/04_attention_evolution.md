# 04 Attention 演化（Attention Evolution）

## 页面目标

这一页回答的是：attention 如何组织 token 间依赖，以及 MHA、GQA、MLA 和稀疏注意力分别在表达能力、KV cache 与计算成本上改变了什么。

本页的输出是上下文建模候选：明确 head、Q/K/V 和缓存组织的变化，不能只用“attention 更快”概括结构差异。

这一页解释 attention 为什么从标准 MHA 演化到 MQA、GQA，以及为什么后续又出现稀疏 attention、长上下文 attention 和系统级加速设计。

## 问题起点

Attention 解决的是 token 之间如何建立依赖关系的问题，但它同时也是：

- 训练里的主要算力和访存来源之一
- 推理时 KV cache 的核心来源之一
- 长上下文和并发场景里的主要瓶颈之一

所以 attention 的演化，本质上是在优化“表达能力 vs 成本”。

## 演化过程

### 标准 MHA

多头注意力把不同子空间的关系拆开建模，是最基础的现代 attention 形式。

它的意义在于：

- 让不同 head 可以学习不同的关系模式
- 在表达能力上提供足够大的自由度
- 作为后续所有优化版本的参照系

但 MHA 的代价也很明显：head 数越多，训练和推理时的算力与访存压力越大。

### MQA / GQA

为了降低推理时 KV cache 和带宽成本，多个 query head 可以共享更少的 key/value head。

这类设计的核心目标是：

- 降低推理显存
- 提升吞吐
- 尽量保留多头建模能力

MQA / GQA 的演化反映的是非常现实的工程权衡：

- query head 仍然需要多样性
- 但 key/value 并不一定要为每个 head 完全复制
- 在不显著伤害效果的前提下，cache 和带宽成本可以明显下降

这也是为什么很多现代模型会在“结构表达能力”和“推理成本”之间选择折中方案，而不是死守原始 MHA。

### MLA：低秩潜在注意力

在 MQA / GQA 之后，注意力演化开始进一步压缩 KV 表示本身。

MLA 可以理解为一种更激进的 KV 压缩路线：

- 它不直接缓存所有 head 的完整 KV
- 而是先把 KV 投到更低维的 latent 空间
- 推理时只维护压缩后的 latent cache，再在需要时恢复或重构注意力所需的信息

这条路线的核心收益有两个：

- 显著降低 KV cache 占用和带宽压力
- 在很多场景下保持接近甚至优于传统 MHA 的效果

所以，MLA 不是简单的“再少几个 head”，而是把 attention 的缓存表示重新设计了一遍。

### Linear Attention：改写 softmax 路径

除了压缩 KV 或做稀疏选择，另一条长期存在的路线是直接改写 attention 的计算形式，也就是常说的 `linear attention`。

它的核心想法不是“少看一些 token”，而是把原本显式构造 `L x L` 注意力矩阵的方式，改写成更接近线性复杂度的累计或核化计算。

这条路线通常想解决三个问题：

- 长上下文下 `O(L^2)` 的注意力成本太高
- 显式保存完整注意力矩阵会带来很高的显存压力
- 某些场景里更希望用流式、递推式或状态式方式处理上下文

因此，`linear attention` 更像是在改写“attention 是怎么被算出来的”，而不是只改 head 数或缓存布局。

但它没有像 `MQA / GQA / MLA` 那样直接成为当前主流开源 LLM 的默认答案，原因也很现实：

- softmax attention 的表达与训练行为更稳定，生态也更成熟
- linear attention 往往需要改写相似度定义或归一化方式
- 在很多真实系统里，瓶颈不只来自公式复杂度，还来自 cache、调度和硬件执行路径

所以可以把它理解成 attention 演化中的一条重要分支：

- `MQA / GQA`：主要压 KV cache 和带宽
- `MLA`：进一步压 KV 表示本身
- `linear attention`：直接改写注意力计算路径
- `sparse attention`：重写“哪些 token 需要互相看见”

### 稀疏 / 长上下文 attention

- 一部分方法通过局部窗口、分块或路由减少计算
- 一部分方法通过系统优化改善吞吐和缓存
- 一部分方法通过结构改造把 attention 的成本压低

在 DeepSeek 的演进里，这条线又继续分成了两类思路：

- `DSA` 这类 token-level sparse attention：先粗筛再精读
- `NSA` 这类硬件对齐的 sparse attention：把稀疏模式和 GPU 友好执行绑在一起

对长文本而言，这意味着模型不再“通读全文”，而是先判断哪些 token 值得看，再把算力集中到真正重要的部分。

当上下文拉长后，attention 的问题不再只是“算得慢”，而是“算得起吗”：

- 全局 attention 的二次复杂度会迅速放大
- KV cache 的常驻成本会挤压 batch 和并发
- 不同任务对局部依赖和全局依赖的需求并不相同

因此，稀疏、局部和分块方案本质上是在重新定义“哪些 token 真的需要互相看见”。

### DeepSeek 风格的索引-选择式注意力

DeepSeek 的稀疏注意力演化，尤其适合用“先翻目录、再精读”来理解。

可以把它拆成两个阶段：

- `Lightning Indexer`：快速给历史 token 打相关性分数
- `Selector`：只保留 Top-k token 进入后续的精细注意力

这样做的目标是把复杂度从 `O(L²)` 压到更接近 `O(L·k)`，尤其适合长上下文解码。

如果再细分实现，DeepSeek 系列的稀疏注意力可以理解成三条并行路径：

- 压缩分支：抓大意，处理长程概览
- 选择分支：对高相关 token 做精读
- 滑动窗口分支：保留局部细节和邻域信息

这类设计的关键不是“把 attention 变稀疏”这么简单，而是把全局、局部和选择性关注拆成不同通路，再让硬件执行尽可能顺滑。

### 系统级加速

FlashAttention 这类工作不是重新定义 attention 语义，而是重新定义执行方式。

这类工作的关键是把理论计算图翻译成更接近硬件的执行路径：

- 减少不必要的 HBM 读写
- 尽量在更合适的粒度上做分块和融合
- 让 attention 的瓶颈尽量从访存转向算力本身

所以 attention 的演化实际上分成两条线：

- 一条是结构语义上的演化，例如 MHA 到 GQA
- 一条是系统实现上的演化，例如 FlashAttention 这类 IO-aware 优化

现代 LLM 往往同时吃这两条收益。

## 代表模型

- `LLaMA`：以标准 attention 为基础，再通过 GQA 等设计优化成本
- `Mistral`：会把局部窗口和长上下文设计结合起来看
- `DeepSeek`：更激进地重构 attention 结构，适合作为现代注意力演化的代表案例，尤其适合看 MLA 和 sparse attention

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | attention 的总起点，理解所有后续变体必须先看它。 |
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | MQA 的经典入口，解释为什么推理时可以减少 KV 头。 |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | 理解 GQA 如何在表达能力和推理成本之间折中。 |
| [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) | linear attention 的经典入口，适合看“怎样把注意力改写成线性路径”。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | 代表系统层面的 attention 优化，直接影响训练和推理速度。 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 看 MLA 如何把 KV cache 压缩到更低维的 latent 表示。 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 代表将滑动窗口等机制融入主流 attention 的实践路线。 |
| [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) | 看 DeepSeek 如何把稀疏注意力推进到 DSA 路线。 |
| [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) | 看硬件对齐的稀疏注意力如何把局部、压缩和选择性关注组合起来。 |

## 与 Part 02 的对应关系

- [Part 02 · 04 Attention（MHA/GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) 直接讲 MHA / GQA / MQA 的 head 关系
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) 里可以看到 attention 如何放进 block
- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb) 和 [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb) 可以用于核对真实模型中的 attention 变体
- [Part 02 · 22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)、[Part 02 · 24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb) 和 [Part 02 · 67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) 形成系统侧衔接
- [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb) 和 [Part 02 · 07 MoE 负载均衡损失](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb) 展示稀疏化和路由在更大结构中的位置

## 一张最小对照表

| 路线 | 主要改什么 | 更偏哪类问题 |
|:---|:---|:---|
| `MHA` | 标准 full attention | 表达能力基线 |
| `MQA / GQA` | 减少 K/V 头或共享 K/V | 推理 cache、带宽、吞吐 |
| `MLA` | 压缩 KV 的表示空间 | KV cache 常驻成本 |
| `linear attention` | 改写注意力计算路径 | 长上下文复杂度、流式计算 |
| `sparse attention` | 只让部分 token 互相看见 | 长上下文选择性计算 |

## 可视化提示

建议画两张图：

- 一张 `MHA -> MQA -> GQA` 的 head 关系图
- 一张 `MHA -> GQA -> MLA / linear attention / sparse attention` 的演化图
- 一张 attention 成本图，标出训练计算、推理 KV cache 和系统吞吐之间的关系

## 进入下一页

进入 [05 RoPE / Position Encoding](./05_rope_position_encoding.md)，把位置关系放回 attention 的几何结构中。

## 阅读建议

如果你要继续扩展，建议接着看：

- `05_rope_position_encoding.md`
- `06_block_residual_path.md`
- `08_representative_models.md`
