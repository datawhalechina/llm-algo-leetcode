# 01 Decoder-only 主干（Decoder-only Backbone）

## 页面导语

现代大语言模型通常把 token 表示送入重复堆叠的 Decoder Block，并用因果掩码约束每个位置只能读取已出现的 token。本页先建立这条主干，再说明归一化、Attention、MLP、残差与缓存为什么都围绕 Block 组织。

## 1. 设计压力：为什么主干收敛到 Decoder-only？

Encoder–Decoder 适合“先编码输入、再条件生成”的任务；自回归语言建模则只需一条按 token 向前推进的状态路径。Decoder-only 让预训练目标、推理循环和上下文缓存共享同一结构接口。

| 结构 | 信息路径 | 典型目标 | 主要代价 |
|:---|:---|:---|:---|
| Encoder–Decoder | Encoder 表示经 Cross-Attention 进入 Decoder | 翻译、条件生成 | 两套主干与跨模块接口更复杂 |
| Decoder-only | 所有 token 沿因果路径进入同一组 Block | 下一 token 预测、生成 | 长上下文 Attention 与状态成本持续增长 |

![Decoder-only 主干：token 表示经过多个 Block 形成下一 token 分布](../../public/topic_discussion/llm_architecture_evolution/block_overview.svg)

## 2. 结构机制：一个 Block 如何更新主干状态？

每个 Block 都围绕同一条残差流完成两次更新：Attention 负责 token 间交互，MLP 负责逐 token 的通道变换；Norm 控制送入分支的尺度，残差连接保留并累积主干状态。多个 Block 重复该结构，才形成模型深度。

| 组件 | 主要职责 | 后续演进问题 |
|:---|:---|:---|
| Norm + Residual | 稳定深层状态更新 | Pre-Norm、RMSNorm、残差缩放 |
| Attention | 读取上下文并混合 token 信息 | MHA/GQA/MLA、稀疏访问、递推记忆 |
| MLP / FFN | 重组单个 token 的通道表示 | SwiGLU、MoE、专家稀疏激活 |
| 输出头 | 将 hidden state 映射到词表分布 | 权重共享、采样与生成循环 |

## 3. 取舍与证据：结构描述不等于模型结论

“采用 Decoder-only”只能确定主干范式，不能直接推出训练稳定性、长上下文质量或推理速度。结构审计还需要记录 Block 数量、hidden size、Norm 位置、Attention 表示、MLP 类型、上下文长度以及实现 backend，再用对应证据验证质量和成本。

## 4. 代码与项目入口

- [Part 02 · 00 PyTorch 热身](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.md)：建立 token、张量和模块调用的最小接口。
- [Part 02 · 05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)：组装 Norm、Attention、MLP 与残差更新。
- [Part 02 · 08 架构技巧](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md)：读取配置并检查参数归属与结构约束。
- [Part 02 · 60 Decoder Block 稳定性项目](../../02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark.md)：比较 Block 变体的稳定性证据。

## 相关阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
