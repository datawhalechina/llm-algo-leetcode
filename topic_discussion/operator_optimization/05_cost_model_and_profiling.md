# 05. 成本模型与 Profiling

## 页面目标

本页回答两个问题：如何从 shape、dtype 和资源约束生成候选配置，以及如何用 profiling 判断瓶颈是否真的被改善。Autotune 是搜索过程，profiling 是证据过程，二者不能互相替代。

![Autotune 与 Profiling：从候选配置到条件化结论](../../docs/public/topic_discussion/operator_optimization/operator_autotune_evidence.svg)

## 成本模型的输入

| 输入 | 影响的决策 | 需要保留的记录 |
|:---|:---|:---|
| shape / dtype | tile、向量化和 Tensor Core 路径 | 输入分布、对齐方式 |
| 数据复用 | shared memory、register 和 fusion | 读写次数、工作集 |
| GPU 资源 | block、occupancy 和并发 | GPU 型号、编译目标 |
| workload | 候选配置和最终结论 | batch、序列长度、重复次数 |

动态 Shape 需要单独记录。一个配置在固定 shape 上最优，不代表它适合其他 batch、序列长度或隐藏维度；autotune 还可能把编译时间、缓存命中和运行时间混在一起。

| 动态因素 | 需要比较 | 建议记录 |
|:---|:---|:---|
| Shape | 固定配置、shape bucket、按 shape 搜索 | shape 分布、bucket 规则 |
| 配置缓存 | 首次编译与缓存命中后运行 | compile time、cache key |
| dtype | 不同输入 / 累加路径 | 输入 dtype、累加 dtype |
| workload | 小 batch、长序列、混合请求 | 各组独立结果，不只报平均值 |

## Profiling 证据链

先提出瓶颈假设，再用相同 workload 采集 kernel 时间、访存、occupancy、编译成本和端到端指标。一次 trace 只能描述当前配置；要形成结论，需要 baseline、候选、重复运行和失败条件。

| 现象 | 可能原因 | 下一步检查 |
|:---|:---|:---|
| kernel 快，端到端不快 | launch、同步或其他阶段占主导 | trace 与调用占比 |
| memory throughput 低 | 布局、复用或 tile 不合适 | load/store、cache、stride |
| occupancy 下降 | register / shared memory 压力 | spill、block 资源 |
| 不同 shape 波动大 | 配置泛化不足 | shape 分桶、配置缓存和 autotune |

## 本页出口

你应能把一个 profiling 现象映射到可验证的优化动作，并说明为什么“某个 shape 上最快”不等于“所有 workload 上最优”。

## 从 Part 02 · 44 迁移的自动调优机制

原 [Part 02 · 44 通用预留](../../02_PyTorch_Algorithms/44_Reserved_44.ipynb) 的自动调优内容已归入本页：先用显存和延迟约束筛掉不可行配置，再对候选配置评分，最后保留候选、约束、评分和推荐理由。这里的重点是把 autotune 当作受约束的搜索与证据记录，而不是只报告一次最快运行。

迁移后的统一记录至少包括：shape、dtype、block 配置、num warps、num stages、显存上限、延迟上限、编译时间、重复运行结果和最终选择理由。原 Notebook 仅保留迁移入口；算子优化路线中的 [Task 5：成本模型与 Profiling](./05_cost_model_and_profiling.md) 负责完整机制，Task 6 负责把候选搜索接入端到端项目。
