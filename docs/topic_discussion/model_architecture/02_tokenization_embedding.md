# 02 分词、BPE 与 Embedding（Tokenization / BPE / Embedding）

## 页面目标

这一页回答的是：离散 token 如何变成可进入 decoder block 的连续表示，以及 tokenizer 选择为什么会影响上下文、词表和后续结构。

本页的输出是输入接口：token 序列、embedding 维度和上下文表示方式必须能与后续 block 对接。

这一页不只解释“词怎么切”，而是回答三件事：

- 为什么大模型必须先处理 tokenization
- BPE / SentencePiece / Embedding 分别解决什么问题
- 这些选择如何影响后续的 block、attention 和上下文建模

## 问题起点

如果没有一个稳定的 token 表示层，后面的 RMSNorm、Attention、RoPE 和 block 组装都没有输入基础。

tokenization 决定了：

- 词表大小和 OOV 行为
- 训练时序列长度和显存开销
- 推理时上下文切分和长度预算

Embedding 则把离散 token 变成可训练的连续表示，是大模型结构的第一层“数值入口”。

## 演化过程

### 早期方案

- 词级别切分容易遇到稀有词和词表爆炸问题
- 字符级别切分会让序列太长，训练和推理代价都高

这一阶段的根本问题是离散词表太脆弱：

- 词级别方案对未登录词很敏感
- 字符级别方案虽然彻底，但序列长度会迅速膨胀
- 如果输入层本身不稳定，后面的 block 和 attention 再强也难以弥补

### 经典 subword 方案

- BPE 用合并子词的方式折中词表规模和序列长度
- SentencePiece 把训练和切分流程统一起来，便于多语言和无空格语言场景
- WordPiece 也走向子词化，但实现和训练细节不同

这一步的关键不是“切得更细”，而是让 token 单位变得更可控：

- 高频片段保留为更稳定的词片段
- 低频词拆分成可复用的子词单元
- 词表规模、序列长度和 OOV 风险第一次得到工程化平衡

因此，subword tokenizer 不是单纯的数据预处理，而是整个模型的输入契约。

### 现代模型的落点

- 现代 LLM 通常使用 subword tokenizer 作为默认入口
- Embedding 往往与输出层共享或近似共享权重
- 一些系统开始探索 tokenizer-free 或更细粒度表示，但代价和收益仍在权衡

现代模型更关注的是“token 表示是否足够适配下游结构”：

- tokenizer 影响上下文切分、长文本覆盖率和显存开销
- embedding 决定离散 token 能否在连续空间里形成可学习的语义几何
- 权重共享让输入和输出空间形成闭环，减少参数冗余

前沿路线虽然在探索 byte-level、character-level 或动态 tokenization，但主流 LLM 仍然更依赖成熟的 subword 入口，因为它在可训练性和工程稳定性之间更平衡。

### 多语言工程样本：Qwen 路线

如果要找一条最适合放在这里的现代工程样本，`Qwen` 系列很合适。

它的价值不只在于“也是 subword tokenizer”，而在于它把 tokenizer 的几个现实问题放得更明显：

- 多语言覆盖面是否稳定
- 词表设计会不会直接影响不同语言的切分质量
- 长上下文和部署场景下，token 长度分布是否仍然可控

这也是为什么 `Qwen` 适合作为 tokenization 页里的代表：

- 它提醒读者 tokenizer 不是前处理小问题，而是多语言模型体验的一部分
- `Qwen3` 这类新代际继续保留了工程可用性导向，而不是只追求结构炫技
- 后面的长上下文、推理和部署判断，都会反过来受 tokenizer 选择影响

## 代表模型

- `LLaMA`：使用统一的 subword tokenization 作为模型入口
- `Qwen`：强调多语言和工程可用性，tokenizer 设计会直接影响输入覆盖面；`Qwen3` 继续延续这条路线
- `DeepSeek`：更关注整体结构效率，但同样依赖稳定的 token 表示入口

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | 解释 BPE 为什么能缓解稀有词问题，是 subword 路线的经典入口。 |
| [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) | 解释统一训练和切分流程的意义，适合多语言与无空格语言场景。 |
| [Fast WordPiece Tokenization](https://arxiv.org/abs/2012.15524) | WordPiece 的直接实现入口，便于和 BPE / SentencePiece 做对照。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [ByT5: Towards a token-free future with byte-level models](https://arxiv.org/abs/2105.13626) | 代表 tokenizer-free 思路，帮助理解“是否一定要 subword”这个前沿问题。 |
| [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874) | 从字符级别重新思考表示层，适合作为 tokenization 方向的前沿对照。 |
| [Charformer: Fast Character Transformers via Gradient-based Subword Tokenization](https://arxiv.org/abs/2106.12672) | 代表近年的动态 tokenization 思考，适合作为进一步扩展阅读入口。 |

## 与 Part 02 的对应关系

`Part 02 01-08` 没有直接讲 tokenization，但它们都默认输入已经完成 tokenization 并进入 embedding 层。

- `01-04` 的 hidden state 来自 token embedding
- `05` 的 block 输入依赖 token 表示维度
- `06-08` 的结构判断也建立在统一 token 表示之上

## 可视化提示

建议画一张“tokenization 到 embedding”的路径图：

- 原始文本
- tokenizer 切分
- token id
- embedding lookup
- hidden state

最好同时标出：

- 词表大小如何影响序列长度
- 稀有词如何被子词拆分
- Embedding 如何进入后续 block

## 进入下一页

进入 [03 Norm Evolution](./03_norm_evolution.md)，观察 hidden state 进入深层 block 后如何保持数值和优化稳定。

## 阅读建议

如果你要继续往下学，建议接着看：

- `03_norm_evolution.md`
- `04_attention_evolution.md`
- `05_rope_position_encoding.md`
