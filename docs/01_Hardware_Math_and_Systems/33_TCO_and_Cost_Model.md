# 33. TCO and Cost Model | 算力评估与 TCO 模型

**难度：** Medium | **环境：** CPU-first | **标签：** `成本评估`, `TCO` | **目标人群：** 需要做 GPU 选型和预算判断的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页的重点不是报一个最低价格，而是把采购、能耗、运维、迁移和风险放进同一个判断框架里。

**关键词：** `TCO`, `cost`, `power`, `migration`, `risk`
## 前置

**导语：** 这一页先把编译、选型和成本判断放到同一条判断链里，再看 TCO 为什么不能只看单卡报价。

## 相关阅读

**导语：** 如果想继续把成本、动态 shape 和虚拟化带来的长期影响串起来，可以接着看这些页。

## Q1：TCO 为什么比单卡报价更重要？

<details><summary>点击展开查看解析</summary>

TCO 看的是总拥有成本，不只是买卡的钱。

它会把采购、能耗、运维、迁移、停机风险和替换成本都放进来。对于长期运行的系统来说，单卡便宜不一定代表整体便宜，因为后续的电费、维护和迁移可能更贵。

所以真正要比较的是一段时间内谁更能稳定地把业务跑起来。
</details>
### Q1小验证：为什么长期成本更关键

看总账，而不是只看首付。

```python
def tco(purchase, power_per_month, ops_per_month, migration, months=12):
    return purchase + (power_per_month + ops_per_month) * months + migration

gpu_a = tco(120, 3, 2, 10, months=24)
gpu_b = tco(90, 5, 4, 22, months=24)
print('A:', gpu_a)
print('B:', gpu_b)
print('cheaper:', 'A' if gpu_a < gpu_b else 'B')
```

## Q2：为什么迁移成本和风险要算进模型里？

<details><summary>点击展开查看解析</summary>

硬件切换不只是换设备，还会带来软件适配、调试、重构和性能回归风险。

如果迁移需要重新适配编译器、kernel、通信库或部署流程，那这些工作本身就是成本的一部分。

所以 TCO 不是财务问题和技术问题的二选一，而是把两者放进同一个判断框架。
</details>
### Q2小验证：迁移为什么会贵

适配成本本身就是成本。

```python
def migration_cost(adapt, test, rollback):
    return adapt + test + rollback

print(migration_cost(5, 3, 2))
```

## Q3：怎样用成本模型做更合理的选型？

<details><summary>点击展开查看解析</summary>

做选型时，最好把成本、风险和收益放到同一个表里。

常见比较维度包括：
- 初始采购成本；
- 运行时电力和运维成本；
- 软件栈成熟度；
- 迁移和维护风险；
- 未来扩展的稳定性。

这样得到的不是“最低价”，而是“最适合当前约束的方案”。
</details>
### Q3小验证：选型判断先看什么

先看长期约束，再看短期价格。

```python
def score_option(total_cost, risk, growth):
    # 分数越低越好：成本、风险和增长不确定性一起看。
    return total_cost + risk * 20 + growth * 10

options = {
    'stable_gpu': score_option(60, 0.2, 0.3),
    'cheap_alt': score_option(45, 0.6, 0.7),
}
for name, score in options.items():
    print(name, score)
print('choose:', min(options, key=options.get))
```

## Q4：为什么扩容节奏和折旧周期也会改变 TCO 结论？

<details><summary>点击展开查看解析</summary>

TCO 不只是“买得起还是用得起”，还取决于你要在多长时间内把这套系统跑满。

如果扩容太慢，前期买来的算力可能闲置，单位产出成本会被拉高；如果折旧周期太短，硬件还没充分发挥就要更新，账面成本也会迅速上升。

因此真正的成本模型，往往要同时考虑：
- 初始采购成本；
- 运行成本；
- 迁移和切换成本；
- 扩容与折旧节奏。

把这些变量放在一起看，才更接近真实的技术选型决策。
</details>
### Q4小验证：扩容和折旧为什么要一起看

```python
def effective_cost(purchase, ops, migration, idle, depreciation):
    return purchase + ops + migration + idle + depreciation

scenarios = {
    'fast_expand': effective_cost(100, 20, 8, 20, 10),
    'slow_expand': effective_cost(100, 20, 8, 5, 28),
}
for name, cost in scenarios.items():
    print(name, cost)
print('more cost effective:', min(scenarios, key=scenarios.get))

```

## ⚠️ 常见误区

- 只看采购价会漏掉长期成本。
- 迁移和运维不是附带项，它们会真实影响 TCO。
- 选型不是找最便宜，而是找最适合约束的方案。
- 成本模型应同时覆盖技术和业务风险。