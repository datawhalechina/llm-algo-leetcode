# 大模型架构演进专题术语表

全教程写作遵循[术语表与中英文写作规范](../../terminology.md)。本表只解释架构语境中容易混淆的机制身份；代码标识符、配置键、模型名和结果 JSON 字段保留原始英文。

| 中文主称 | 英文 / 缩写 | 本专题含义 | 容易混淆之处 | 学习入口 |
|:---|:---|:---|:---|:---|
| 词元 | token | 分词器按固定规则产生、供模型处理的离散单位 | 不等于字符、单词或请求中的文本长度 | [02 分词与输入表示](./02_tokenization_embedding.md) |
| 分词器 | tokenizer | 固定词表、切分规则、特殊词元和输入模板的模型资产 | 相同文本在不同分词器下可能产生不同词元序列 | [02 分词与输入表示](./02_tokenization_embedding.md) |
| 仅解码器主干 | Decoder-only backbone | 由因果解码器块堆叠形成的语言模型主干 | 不等于一次生成请求的解码阶段 | [01 仅解码器主干](./01_transformer_decoder.md) |
| 主状态 | hidden state | 词元在各层之间传递并被残差更新的表示 | 不等于 KV Cache 或递推状态 | [06 Transformer 块与残差主干](./06_block_residual_path.md) |
| 前置归一化 | Pre-Norm | 在注意力或 MLP 更新前归一化主状态 | RMSNorm 是归一化形式，Pre-Norm 是放置位置 | [03 归一化演进](./03_norm_evolution.md) |
| 残差主干 | residual stream | 跨层保留并累积子层更新的主状态路径 | 不是 KV Cache 的残留窗口 | [06 Transformer 块与残差主干](./06_block_residual_path.md) |
| 多头注意力 | MHA | Q、K、V 使用相同数量的注意力头 | 与多查询注意力、分组查询注意力的 KV 共享方式不同 | [04 注意力演进](./04_attention_evolution.md) |
| 多查询注意力 | MQA | 多个 Q 注意力头共享一组 K/V 注意力头 | 不是把所有注意力头合并成一个 | [04 注意力演进](./04_attention_evolution.md) |
| 分组查询注意力 | GQA | 多组 Q 注意力头分别共享 K/V 注意力头 | 共享粒度介于 MHA 与 MQA 之间 | [04 注意力演进](./04_attention_evolution.md) |
| 多头潜在注意力 | MLA | 将历史 K/V 表示压缩为潜在状态，并按需要重建计算表示 | 不等于普通低比特 KV Cache 量化 | [ARCH-MLA](../../02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention.md) |
| 键值缓存 | KV Cache | 在自回归生成中保存历史词元的 K/V 表示 | 表示格式、存储管理和访问范围是三个不同问题 | [04 注意力演进](./04_attention_evolution.md) |
| 解耦位置分量 | decoupled positional component | 将位置相关分量与可压缩内容分量分开保存或计算 | 不等于完全移除 RoPE | [04 注意力演进](./04_attention_evolution.md) |
| 旋转位置编码 | RoPE | 通过旋转 Q/K 成对维度表达相对位置关系 | 位置缩放改变外推条件，不自动保证长上下文质量 | [05 RoPE](./05_rope_position_encoding.md) |
| 稠密前馈网络 | Dense FFN | 每个词元经过同一组前馈网络参数 | 与 MoE 的“总参数稀疏激活”不同 | [07 MLP / FFN](./07_mlp_ffn_evolution.md) |
| 门控线性单元 | GLU / SwiGLU | 用门控分支调制值分支的逐词元通道变换 | 门控激活不是 MoE 路由器 | [07 MLP / FFN](./07_mlp_ffn_evolution.md) |
| 访问范围 | attention access pattern | 每个查询位置允许读取哪些历史位置 | 与 K/V 使用何种表示、存储格式是两个问题 | [Part 02 · 44](../../02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention.md) |
| 滑动窗口注意力 | sliding-window attention | 每个位置只访问有限邻近历史窗口 | 显式布尔掩码不代表命中专用稀疏计算内核 | [Part 02 · 44](../../02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention.md) |
| 递推状态 | recurrent state | 用固定或受控大小状态逐步汇总历史 | 不支持任意历史精确回读时，不能等同完整 KV Cache | [Part 02 · 45](../../02_PyTorch_Algorithms/45_Linear_Attention_and_Recurrent_State.md) |
| 状态空间模型 | SSM | 用状态转移与输入映射处理序列的模型族 | Mamba 是选择性 SSM 路线，不是所有线性注意力的统称 | [10 混合记忆](./10_linear_attention_ssm_and_hybrid.md) |
| 混合记忆架构 | hybrid memory architecture | 在不同层组合注意力与递推/SSM 状态 | 不等于把两个模型输出简单相加 | [ARCH-HYBRID-MEMORY](../../02_PyTorch_Algorithms/ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.md) |
| 路由器 | Router | 根据词元表示产生专家选择分数并执行 Top-k 选择 | 选择专家不等于完成词元分发 | [Part 02 · 06](../../02_PyTorch_Algorithms/06_MoE_Router.md) |
| 专家 | Expert | MoE 中可被独立选择的 FFN 参数分支 | 专家总数不等于每个词元的激活专家数 | [09 MoE](./09_moe_sparsity_evolution.md) |
| 共享专家 | shared expert | 对所有或大部分词元保持激活的公共专家分支 | 不能计入稀疏 Top-k 后假装每个词元成本不变 | [09 MoE](./09_moe_sparsity_evolution.md) |
| 专家容量 | expert capacity | 单个专家在当前批次允许处理的词元上限 | 专家总参数量不等于每个词元的激活容量 | [09 MoE](./09_moe_sparsity_evolution.md) |
| 容量溢出 | overflow | 路由词元超出专家容量后的状态 | 丢弃、回退和重路由具有不同质量与通信代价 | [Part 02 · 06](../../02_PyTorch_Algorithms/06_MoE_Router.md) |
| 分发与合并 | dispatch / combine | 按专家归属搬运词元，并将专家输出恢复到原词元顺序 | 路由器只给出归属，不承担跨设备搬运 | [Part 02 · 46](../../02_PyTorch_Algorithms/46_MoE_Expert_Parallel.md) |
| 声明结构 | declared architecture | 论文、模型卡或配置文件声称采用的结构 | 不能证明当前模型快照已按该结构实例化 | [08 结构审计](./08_representative_models.md) |
| 实例化模块 | materialized module | 加载模型快照后实际存在的模块、参数和权重形状 | 不保证目标后端真正执行预期计算内核 | [08 结构审计](./08_representative_models.md) |
| 运行路径 | runtime path | 目标输入和后端下真正执行的算子、状态与回退路径 | 不能仅由配置字段推断 | [08 结构审计](./08_representative_models.md) |
| 模型快照 | model snapshot / checkpoint | 固定版本的配置、权重和相关模型资产 | 模型家族名称不能唯一确定当前快照的结构 | [08 结构审计](./08_representative_models.md) |
| 模型版本 | model revision | 用提交哈希或等价不可变标识固定模型快照 | `main` 是可移动引用，不能单独保证复测时资产不变 | [实验资产表](./experiment_assets.md) |
| 执行后端 | execution backend | 实际加载模型并调度算子或计算内核的软件路径 | 语义正确不代表命中目标优化路径 | [实验资产表](./experiment_assets.md) |
| 架构审计卡 | architecture audit card | 记录声明结构、实例化模块、运行路径、资源影响、证据等级和决策的结构化报告 | 不替代局部机制实验和真实质量基准 | [08 结构审计](./08_representative_models.md) |

## 使用规则

- 首次出现使用“中文主称（English）”，后续使用中文主称或表中通用缩写。
- `MHA`、`MQA`、`GQA`、`MLA`、`RoPE`、`MoE`、`SSM` 可直接使用；普通叙述优先使用“工作负载、基线、候选、执行后端”等中文主称。
- 注意力的表示、访问范围、存储管理和执行内核分属不同层次，不用一个“注意力优化”概括。
- 术语修改逐页判断，不对 Notebook JSON、代码变量或模型配置执行无上下文批量替换。
