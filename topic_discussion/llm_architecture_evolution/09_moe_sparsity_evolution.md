# 09 MoE 与稀疏容量（MoE and Sparse Capacity）

## 页面导语

Mixture of Experts（MoE）把模型总容量与每 token 激活计算拆开：Router 为 token 选择少量专家，专家完成 FFN 变换，再将结果合并回主干。本页把结构选择、训练稳定性和通信代价放在同一条证据链上。

## 1. 设计压力：容量增长为什么不应等于每 token 成本增长？

Dense FFN 让所有 token 使用同一组参数，模型越宽，每个 token 的成本越高。MoE 通过稀疏激活增加总参数，但能否获得收益取决于路由质量、专家容量和系统执行是否匹配。

![MoE 稀疏容量：Router 选择专家，dispatch 后再 combine](../../docs/public/topic_discussion/llm_architecture_evolution/moe_sparsity.svg)

## 2. 结构机制：选择、容量与回收缺一不可

| 环节 | 机制问题 | 失败表现 |
|:---|:---|:---|
| Router / Top-k | 哪些专家接收当前 token | 路由塌缩、专家长期闲置 |
| Capacity / Overflow | 单个专家最多接收多少 token | token 丢弃、回退或排队 |
| Dispatch / Combine | token 如何发送并回收结果 | All-to-All 暴露、重排开销 |
| Balance objective | 如何约束专家利用率 | 辅助损失干扰主任务或约束不足 |

共享专家、细粒度专家和无辅助损失路由都在改变上述环节，但不能绕过容量与负载证据。

## 3. 取舍与证据：MoE 同时是结构和系统问题

评估 MoE 至少记录 Top-k、专家数、capacity factor、overflow、专家负载分布、dispatch bytes、通信暴露时间与任务质量。仅报告总参数量或理论激活参数，无法判断真实吞吐和稳定性。

## 4. 代码与项目入口

- [Part 01 · 22 MoE 参数与计算](../../01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute.ipynb)：区分总参数与激活计算。
- [Part 02 · 06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb)：实现 Top-k 与容量约束。
- [Part 02 · 07 MoE 负载均衡](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb)：分析专家分布与稳定性。
- [Part 02 · 47 MoE 专家并行](../../02_PyTorch_Algorithms/47_MoE_Expert_Parallel.ipynb)：连接 dispatch/combine 与通信。
- [Part 02 · 80 MoE 专家并行项目](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb)：形成真实多卡证据。

## 相关阅读

- [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
