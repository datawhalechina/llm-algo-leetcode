# 03. Decoding Strategies | 解码策略

## 页面目标

从首 token 之后的生成循环开始，观察单条请求每轮如何读取已有 KV 状态、产生输出，再比较生成策略对速度、质量和额外成本的影响。

## 核心机制

Decode 是一轮一轮的小步生成：每一步读取已有 KV 状态，得到 logits，再选择下一个 token，并把新的 token 和状态追加到序列中。先用贪心解码建立基线；再把投机式生成看成一条策略链：经典 draft / target 验证、多个 token 推进与更一般的多候选验证，目标都是提高每轮有效推进量。本页重点是单条请求的生成决策；多个请求如何组织执行，放到 Task4 的调度正文中。

这里的“加速”不是简单地减少模型调用次数，而是提高每次验证真正提交的 token 数。可以用 `effective_output_tokens / decode_step` 表示单轮有效推进量；它必须和 `acceptance_rate`、验证耗时、回退次数以及最终质量一起观察。这样才能区分“候选更多”与“有效生成更多”。

![Decode 策略对照](../../public/topic_discussion/inference_optimization/decode_strategies_zh.svg)

| 策略 | 主要改变什么 | 需要观察什么 |
|:---|:---|:---|
| 贪心 Decode（基线） | 每轮选择 logits 最大的一个 token | 输出长度、TPOT、KV Cache 访问、质量 |
| 随机采样 | 按 temperature、top-k / top-p 选择 token | 质量、重复率、TPOT、输出稳定性 |
| 经典投机解码 | draft 提议、target 验证与修正 | acceptance rate、验证成本、TPOT |
| 多 Token 投机式推进 | 单轮提出多个 token 或候选 | 有效推进、首次拒绝与回退 |
| Decode Scheduling | 不同请求的执行顺序，属于 Task4 的请求级调度 | 吞吐、TPOT、P99 |

单请求的 Decode 可以按“读取状态 → 计算 logits → 选择 token → 检查停止 → 追加状态”理解。策略改变的是其中的候选生成、验证或单轮产出；它不会自动消除 KV Cache 的容量约束，也不能用吞吐提升替代质量检查。比较 speculative decoding 或 multi-token decoding 时，还要记录接受情况和回退成本。

| 生成环节 | 需要回答的问题 | 证据 |
|:---|:---|:---|
| 读取状态 | 本轮使用了哪些历史 KV？ | Cache 长度、TPOT、读写量 |
| 选择 token | 是贪心、采样，还是 draft 提议后验证？ | temperature、top-k/top-p、acceptance rate |
| 停止与回退 | 何时遇到 EOS，何时拒绝候选？ | generated tokens、停止原因、回退次数 |
| 质量与成本 | 更快是否保持了目标质量？ | 质量指标、TPOT、吞吐、额外模型成本 |

生成策略的比较需要把“可重复性”和“质量门槛”一起固定。贪心 Decode 适合作为确定性基线；采样实验应固定随机种子和采样参数；speculative decoding 还要记录 draft 与 target 的版本、接受率和拒绝后的回退路径。只有在输出质量达到同一门槛后，TPOT 或吞吐的改善才可以作为性能收益。

| 对照条件 | 应固定的内容 | 额外记录 |
|:---|:---|:---|
| 贪心基线 | 模型、Prompt、最大输出长度、停止条件 | 输出 token、TPOT、E2E、质量基线 |
| 随机采样 | seed、temperature、top-k/top-p、Prompt 集 | 重复率、格式成功率、质量分布 |
| 经典投机解码 | draft/target 模型、proposal 长度、验证规则 | acceptance rate、回退次数、额外显存 |
| 多 Token 投机式推进 | 候选来源、单轮候选数量、接受与回退规则 | 有效推进、验证成本、TPOT |

参考入口：论文 [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)；开源实现 [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

## 判断框架

本节承接 `02` 的 Prefill 与 Attention，先用 [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md) 建立单步生成口径，再用 [23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md) 理解经典 draft / target 验证，用 [35 Multi-Token Speculative Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.md) 比较候选推进分支，随后由 [68 Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md) 固定质量与成本口径；最后才通过 [36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.md) 观察多请求组织。阅读下表时，先固定 `TPOT`、`decode_share`、generated tokens 和质量约束，再根据现象选择下一步动作。

| 观察到的现象 | 优先判断 | 下一步 |
|:---|:---|:---|
| 每轮生成耗时高 | Decode 计算、KV Cache 访问或 sampling | 检查基础 Decode 和 `04` |
| 循环次数过多 | 生成长度或单轮产出受限 | 检查 speculative / multi-token |
| acceptance rate 低 | draft model 或 proposal 不匹配 | 调整 draft、长度或放弃策略 |
| 多请求吞吐低、P99 高 | 请求组织或调度问题 | 进入 `07` / `06` |
