# 08 代表模型与结构对照（Representative Models / Cross-Module Comparison）

## 页面目标

本页的输出是结构对照和项目假设：把模块差异映射到真实模型，区分结构代际、dense/MoE 分支与后训练 reasoning 分支。

这一页不重复讲模块本身，而是回答两个更实际的问题：

- 当这些模块组合起来时，不同流行大模型到底选了什么结构
- 它们在 norm、attention、RoPE、MLP、decoder 结构上有什么差异
- 为什么这些差异值得单独学
- 横向对照时，应该从哪些维度把模型和模块放回同一张图里

## 问题起点

模块演进最终会落到具体模型上。

如果只学模块，不看模型，容易把知识停留在“概念正确”；
如果只看模型，不拆模块，又容易把结构选择混成一个黑箱。

这一页的作用，就是把两者接起来。

## 演化过程

### LLaMA：现代开源 block 的参照系

LLaMA 通常可以看成现代 dense decoder-only block 的参照样本：

- RMSNorm
- RoPE
- SwiGLU
- decoder-only 主干

它的价值不只是“很流行”，而是它把现代 LLM block 的常见选择收敛成了一条清晰路径。

从演化视角看，LLaMA 之所以重要，是因为它把前面几页讨论的模块组合成了一个足够稳定的基准：

- norm 选择上偏向简洁和稳定
- attention 选择上保持主流且可复现
- MLP 选择上采用更高效的门控形式
- 位置编码上采用现代常用的 RoPE

它不是结构实验室，但它把“现代该怎么拼”这件事讲得非常清楚。

### Mistral：面向推理和长上下文的效率优化

Mistral 更适合用来观察：

- 局部窗口 attention
- 长上下文效率
- decoder-only 结构与系统成本的关系

它说明当模型进入更实际的部署场景后，结构不再只追求统一的全局建模能力，而是开始强调：

- 哪些 token 真值得全局看见
- 哪些依赖可以局部解决
- 推理成本是否能被压到可接受范围

所以 Mistral 是“结构 + 系统”联动的代表样本。

### Qwen：工程可用性和多语言能力

Qwen 更适合用来观察：

- tokenizer 和 embedding 的覆盖面
- 长上下文和工程部署策略
- 结构设计如何服务多语言任务
- dense 和 MoE 双路线如何并存

Qwen 的意义在于，它提醒我们模型结构不是孤立的：

- tokenizer 会影响多语言覆盖和输入稳定性
- embedding 会影响词表和语义空间的形状
- 长上下文与工程部署策略会一起塑造最终体验

所以看 Qwen 时，不只是看参数量，而是看它如何把结构设计和实际使用场景绑在一起。

#### Qwen2.5 -> Qwen3：从工程稳态到双路线展开

如果把 `Qwen` 当成国产基础模型代表，当前最值得补的不是把所有新型号堆进来，而是先看 `Qwen2.5 -> Qwen3` 这条最稳定的代际线。

这条线的价值在于：

- `Qwen2.5` 更像工程稳态样本，强调多语言、长上下文和部署可用性
- `Qwen3` 则把路线进一步拆成 `dense` 和 `MoE` 两条可并行观察的结构代际

从结构角度看，`Qwen3` 最值得注意的不是“参数更大”，而是它给出了更清晰的对照面：

- dense 模型和 MoE 模型同时公开，便于横向比较
- 小模型和中大模型在 `tie embedding`、`Q/KV heads`、上下文长度上有明确分档
- `Qwen3-30B-A3B`、`Qwen3-235B-A22B` 这类模型让 `Qwen` 不再只是 dense 样本

因此，在这个专题里，`Qwen3` 应该被当成：

- 一条 `Qwen` 基础模型代际升级线
- 一组 `dense vs MoE` 的结构对照样本
- 一个把多语言、长上下文和部署需求继续往前推的工程代表

### Gemma：紧凑而稳定的现代结构样本

Gemma 常被用来观察：

- norm 和 block 的稳定性
- 结构简洁性
- 现代模型如何在效果和成本之间折中

Gemma 更像是一个“收敛后的工程样本”：

- 它不追求把每个模块都做得激进
- 但会在 block 稳定性和部署友好性上保持克制
- 适合拿来观察现代 LLM 如何在简洁和效果之间做平衡

这类模型的价值在于，它给出的是一种可持续维护的设计范式。

### DeepSeek：结构重构更激进的代表

DeepSeek 适合作为更前沿的对照样本，尤其适合看它如何把 attention 继续往前推：

#### 1. 基础阶段：从标准注意力到 MLA

DeepSeek-V2 的关键看点之一是 `MLA`（Multi-head Latent Attention）。

它可以看成对 `MHA / GQA` 的进一步升级：

- 传统 MHA / GQA 仍然要维护较完整的 head 级 KV 表示
- MLA 则把 KV 压到更低维的 latent 空间
- 推理时缓存的是压缩后的 latent 表示，而不是每个 head 的完整 KV

这带来的结果是：

- KV cache 占用明显下降
- 带宽压力更低
- 在很多场景下，效果还能保持得和 MHA 相当甚至更好

所以，DeepSeek-V2 不只是“更省显存”，而是把 attention 的缓存表示重写了。

#### 2. 进阶阶段：稀疏注意力和索引-选择式执行

DeepSeek 的后续 attention 演化继续往“先筛再算”推进。

可以把它理解成一个两级管线：

- `Lightning Indexer` 先给历史 token 打相关性分数
- `Selector` 再保留 Top-k token 进入精细注意力

如果再往实现里拆，还可以看成三条并行分支：

- 压缩注意力：抓大意
- 选择性注意力：精读高相关 token
- 滑动注意力：保留局部细节

这种设计的目的，是让长上下文不再“通读全文”，而是“先翻目录，再精读”。

#### 3. 更进一步：DeepSeek Sparse Attention

DeepSeek-V3.2 把这条线推进到 `DSA`（DeepSeek Sparse Attention）。

它的关键点在于：

- 通过动态稀疏策略降低长上下文复杂度
- 让训练和推理都能从 sparse pattern 中获益
- 和硬件友好的执行方式一起设计，而不是只在数学上做稀疏

所以，DeepSeek 的 attention 演化不是单一技巧，而是一条连续路线：

- `MHA / GQA` 解决 head 级别的成本问题
- `MLA` 解决 KV 表示和缓存问题
- `DSA / sparse attention` 解决长上下文下的选择性计算问题

这也是为什么 DeepSeek 特别适合作为 attention 前沿样本：它不是局部修补，而是沿着“缓存、路由、稀疏、硬件”四个方向连续推进。

#### 4. 版本线要分开看：V 系列 vs R 系列

这里最容易混淆的是：`DeepSeek-V2 / V3 / V4` 和 `DeepSeek-R1` 不应放在同一条“结构代际”里直接并排比较。

- `V` 系列更适合作为**基础模型 / 架构代际**来看，重点看 block、attention、MoE、长上下文和系统效率如何演化。
- `R` 系列更适合作为**后训练 / 推理强化分支**来看，重点看 reasoning 能力、思维链长度、后训练方式和评测收益。

因此，在这个专题里更合理的读法是：

- `DeepSeek-V2`：看 `MLA` 如何改写 KV 表示
- `DeepSeek-V3 / V3.2`：看 sparse attention、MoE 与长上下文如何继续推进
- `DeepSeek-V4`：看它是否已经成为新的结构代际分界点
- `DeepSeek-R1`：不要当成新的 block 架构代际，而应当放到“后训练 / 推理模式”这一层去理解

换句话说，`R1` 更像“在 DeepSeek 基座上强化 reasoning 的路线”，而不是“替代 V 系列的下一代结构总称”。

## 代表模型对照

| 模型 | 你应该关注什么 |
|:---|:---|
| LLaMA | 现代标准 block 是怎么收敛出来的 |
| Mistral | attention 和长上下文怎么结合 |
| Qwen | tokenizer、embedding、多语言覆盖，以及 `dense / MoE` 双路线怎么展开 |
| Gemma | norm、MLP、block 的稳定性 |
| DeepSeek V-series | attention / KV cache / sparse routing 还能怎么重构 |
| DeepSeek R-series | 后训练如何把 reasoning 能力继续往上推 |

## 横向对照

这一节把模型叙事再往外推一步，用来把 `01-07` 的模块变化和 `08` 里看到的真实模型连接起来。

### 对照维度

| 模块 | 关注点 |
|:---|:---|
| `02_tokenization_embedding` | 文本如何变成 token id 和 hidden state |
| `03_norm_evolution` | LayerNorm、RMSNorm、DyT 的稳定性取舍 |
| `04_attention_evolution` | MHA、GQA、MLA、稀疏 attention 的成本与表达能力 |
| `05_rope_position_encoding` | 位置关系如何进入 attention |
| `06_block_residual_path` | norm、attention、MLP、residual 如何组装成 block |
| `07_mlp_ffn_evolution` | FFN 如何从 dense 走向门控和专家化 |
| `09_moe_sparsity_evolution` | dense MLP 为什么会被 sparse router / experts 替换 |

### 一组典型对照

#### LLaMA

- `02_tokenization_embedding`：标准 subword 表示入口
- `03_norm_evolution`：RMSNorm
- `04_attention_evolution`：MHA / GQA 路线
- `05_rope_position_encoding`：RoPE
- `07_mlp_ffn_evolution`：SwiGLU
- `01_transformer_decoder`：decoder-only 主干

#### Mistral

- 更适合看 `04_attention_evolution`
- 和 `05_rope_position_encoding`、`01_transformer_decoder` 结合得更紧
- 体现长上下文和系统成本的折中

#### Qwen

- 更适合看 `02_tokenization_embedding`
- 也适合看 `05_rope_position_encoding`
- 也适合看 `08_representative_models`
- 强调工程可用性和多语言覆盖
- `Qwen3` 之后还可以进一步看 `dense` 和 `MoE` 两条结构路线

#### DeepSeek

- 更适合看 `04_attention_evolution`
- 也适合看 `09_moe_sparsity_evolution`
- `V` 系列体现结构重构、稀疏化和效率优化的前沿方向
- `R` 系列体现后训练和 reasoning 强化路线，但不应和 `V` 系列混成同一条结构代际

### 读法建议

- 想看“输入怎么变成模型表示” -> `02_tokenization_embedding`
- 想看“为什么训练更稳” -> `03_norm_evolution` + `06_block_residual_path`
- 想看“长上下文为什么慢” -> `04_attention_evolution` + `05_rope_position_encoding`
- 想看“为什么 decoder-only 成主流” -> `01_transformer_decoder`
- 想看“某个大模型到底改了什么” -> `08_representative_models`
- 想看“为什么密集 FFN 会变成稀疏路由” -> `09_moe_sparsity_evolution`

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 现代 block 的基准模型。 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 长上下文和效率优化的重要样本。 |
| [Qwen Technical Report](https://arxiv.org/abs/2309.16609) | Qwen 初代路线的代表入口。 |
| [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) | 看 Qwen 在多语言、长上下文和工程可用性上的稳态收敛。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [Gemma: Open Models Based on Gemini Research and Technology](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) | 现代紧凑模型的参考样本。 |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | 看 Qwen3 如何把 dense / MoE 两条结构路线一起公开出来。 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 看 MLA 如何把 attention 缓存压缩到 latent 空间。 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 更前沿的系统化结构演化样本。 |
| [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) | 看 DSA 如何把 sparse attention 推到新的阶段。 |

当前这页的 `DeepSeek` 结构正文覆盖主要到 `V2 / V3 / V3.2`。`V4` 已经可以作为新的结构代际入口纳入，`R1` 则更适合作为后训练 / reasoning 分支补进对照表，而不是直接并入结构演化主线。

## 国产模型版本对照

如果只在文字里分别讲 `Qwen` 和 `DeepSeek`，读者还是容易把“基础模型代际”和“后训练推理分支”混在一起。更稳的做法是先把它们放到同一张版本表里看。

| 系列 | 当前更适合怎么看 | 本专题里最该关注什么 | 更适合落在哪一页 |
|:---|:---|:---|:---|
| `Qwen2.5` | 工程稳态样本 | 多语言、长上下文、工程可用性 | `02`、`05`、`08` |
| `Qwen3 Dense` | 基础模型代际升级 | tokenizer 覆盖、RoPE 长上下文分档、dense block 稳态 | `02`、`05`、`08` |
| `Qwen3 MoE` | 稀疏结构分支 | `dense vs MoE` 双路线、专家化与系统代价 | `08`、`09` |
| `DeepSeek-V2` | 结构前沿起点 | `MLA`、KV cache 压缩 | `04`、`08` |
| `DeepSeek-V3 / V3.2` | 稀疏 attention 深化 | `DSA`、长上下文、硬件友好稀疏执行 | `04`、`08`、`09` |
| `DeepSeek-V4` | 新的结构代际入口 | 是否形成新的 attention / MoE / 系统组合样式 | `08` |
| `DeepSeek-R1` | 后训练 / reasoning 分支 | reasoning 强化、后训练与推理模式，而非 block 主线 | `08`，后续更适合衔接 `post_training_alignment` |

这张表的核心作用不是追型号，而是先把问题分清楚：

- `Qwen` 更适合当“国产基础模型工程样本”来看
- `DeepSeek V-series` 更适合当“结构重构样本”来看
- `DeepSeek R-series` 更适合当“后训练 / reasoning 分支”来看

这样后续你再补型号时，不会把不同层级的变化硬塞进同一条结构主线。

## 可视化提示

建议在这一页放两张图，把“模型选择”和“横向对照”分开：

> 图册占位：代表模型结构矩阵尚未生成，当前先以本页的对照表作为依据。

> 图册占位：跨模块知识地图尚未生成，当前先按 `01-09` 页面的引用关系阅读。

第一张图负责快速看清 LLaMA、Mistral、Qwen、Gemma、DeepSeek 各自的结构选择；第二张图负责把 01-09 页串成一张知识导航图。

如果后续继续加图，最值得补的是一张 `DeepSeek V-series vs R-series` 的版本分层图：

- 一条线画 `V2 -> V3 -> V3.2 -> V4`
- 另一条线从对应基座分叉到 `R1`

这样可以避免把“结构代际”和“后训练 reasoning 分支”混成同一条线。

另一张高优先图是 `Qwen2.5 -> Qwen3 Dense / Qwen3 MoE` 的分叉图，重点标出：

- 多语言 / tokenizer 覆盖
- `32K / 128K` 长上下文分档
- `dense` 和 `MoE` 两条结构路线

## 回到项目

把结构对照转入 `61 Model Architecture Exploration`，用参数、模块路径、显存或推理行为验证“结构差异是否带来可观测影响”。

## 与 Part 02 的对应关系

- [Part 02 · 01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)、[Part 02 · 02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb)、[Part 02 · 03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb) 和 [Part 02 · 04 Attention（MHA/GQA）](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) 是这些模型结构设计的组件来源
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) 直接对应 decoder block 的核心样式
- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb) 和 [Part 02 · 61 模型架构探索](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb) 用于核对真实模型的结构组合
- [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb)、[Part 02 · 07 MoE 负载均衡损失](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb) 对应 MoE 或稀疏结构扩展

## 阅读建议

如果你已经看过：

- `02_tokenization_embedding.md`
- `03_norm_evolution.md`
- `04_attention_evolution.md`
- `05_rope_position_encoding.md`
- `01_transformer_decoder.md`
- `07_mlp_ffn_evolution.md`
- `06_block_residual_path.md`

这一页就是把它们投射到真实模型上的地方，同时也给后面的 `09_moe_sparsity_evolution.md` 留出接口。
