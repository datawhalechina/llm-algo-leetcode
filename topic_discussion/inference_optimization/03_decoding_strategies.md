# 03. Decoding Strategies | 解码策略

## 页面目标

本节回答：token 怎么生成，如何减少 decode 循环成本。

## 本节在路线中的位置

本节对应 **Task2：Decode 与生成策略**。它承接 01 的指标口径和 02 的 Prefill 判断，关注首 token 之后的生成阶段；完成后进入 Task3，判断 KV Cache、请求组织和调度是否成为主要瓶颈。

本节先建立解码策略的共同判断框架，再把 speculative decoding、multi-token decoding 和 decode scheduling 作为不同层次的候选动作。它们不是同一种优化，也不能只用吞吐一个指标比较。

## 问题起点

很多推理系统在长 prompt 上还能接受，但一进入连续生成阶段就开始掉速。原因在于 decode 不是一次大矩阵，而是一轮一轮的小步循环：

- 每一步都要读 KV cache；
- 每一步都要做 sampling / search；
- 请求一多，还要把不同步的会话排进同一个服务系统里。

因此，decode 优化的难点不只是“生成哪个 token”，而是“每轮循环能不能更高效、请求能不能排得更顺”。

## 你要先确认什么

- `TPOT` 是否高于预期。
- `decode_share` 是否在总耗时里占主导。
- 生成阶段是不是因为循环次数太多而慢。

如果比较 speculative decoding，还必须固定 draft model、proposal length、target model、acceptance rate 和质量约束；如果比较调度，则要固定请求到达模式、并发窗口和输出长度分布。

## 核心矛盾

decode 的核心矛盾是：每一轮工作量很小，但轮数很多；每一轮都依赖上一步输出，但系统又希望把并发和硬件利用率拉起来。这导致 decode 不能只看算法，还要看接受率、调度和 cache 命中。

## 演化路径

decode 阶段的核心不是“选哪种采样”，而是“每轮生成能不能更高效”。

1. 基础 decoding 决定每一步如何选 token。
2. speculative decoding 让 draft model 先提议。
3. multi-token decoding 让一次循环生成多个 token。
4. decode scheduling 让请求按更合理的顺序排布。
5. 最终目标是降低 TPOT 并提高吞吐。

## 关键取舍

- `speculative decoding` 只有在 acceptance 足够高时才真的划算，否则只是把验证成本又加回来了。
- `multi-token decoding` 试图减少循环次数，但会引入更复杂的接受与回退逻辑。
- `decode scheduling` 不改变模型本身，却可能显著影响多请求场景下的真实吞吐。

所以 decode 优化往往不是“某一种策略一定更好”，而是要看请求分布、草稿模型质量和服务目标。

## 学习者交付物

完成本节后，至少应能说明当前 Decode 瓶颈和候选策略的代价：

| 项目 | 最小内容 |
|:---|:---|
| 症状 | TPOT、decode_share、generated tokens/s 的变化 |
| 候选策略 | sampling、speculative、multi-token 或 scheduling |
| 策略成本 | acceptance rate、draft cost、verify cost、额外调度开销 |
| 质量约束 | 输出一致性、任务质量或允许的误差范围 |
| 下一步 | 进入 04 看 Cache / 调度，或用 68 做专项 benchmark |

核心结论应能够区分：是每轮 Decode 计算太慢、循环轮数太多，还是请求组织和 KV Cache 访问拖慢了生成。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 文献锚点

- speculative decoding 代表论文：帮助理解“先提议后验证”的速度来源。
- multi-token decoding 相关工作：帮助理解为什么减少循环轮数会直接改变 TPOT。
- serving 调度相关工程资料：帮助理解 decode 阶段为什么常常是系统问题而不是单 kernel 问题。

## 常见误区

- 只改 sampling 参数，以为就解决了吞吐问题。
- speculative decoding 只看理论速度，不看 acceptance。
- multi-token decoding 和调度问题混在一起看。

## 对应 Part 02

- `21` Decoding Strategies
- `23` Speculative Decoding
- `35` Multi-Token Decoding
- `36` Decode Scheduling
- `66` Inference Performance Comparison

## 证据边界

CPU 可以验证 sampling、draft / verify 和接受率的计算逻辑；真实 acceptance rate、TPOT、吞吐和调度收益需要目标模型、draft model 与固定请求 workload。普通解码实验不能直接证明 speculative decoding 已生效。

## 经典阅读入口

- [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)
- [23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)
- [35 Multi-Token Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb)
- [36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)

## 相关跳转

- 看 `01`，确认指标口径。
- 看 `04`，确认 decode 和 cache 怎么协作。

## 对应项目

- **扩展项目：** [68 Speculative Decoding Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)，在固定 workload 下比较 acceptance rate、draft / verify 成本和吞吐。
- **核心综合项目：** [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)，把 Decode 策略与 Cache、量化和 backend 放到同一套指标中比较。

本节不直接断言 speculative decoding 或 multi-token decoding 一定更快；是否采用必须回到请求分布、质量约束和最终服务目标。

## 本节要点

解码优化的重点是减少无效循环，让生成阶段的 token 产出更快。
