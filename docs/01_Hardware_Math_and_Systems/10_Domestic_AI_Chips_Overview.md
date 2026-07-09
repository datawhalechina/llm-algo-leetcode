# 10. Domestic AI Chips Overview | 算力现状与替代方案

**难度：** Medium | **环境：** CPU-first | **标签：** `系统架构`, `异构算力` | **目标人群：** 芯片选型入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把硬件规格、软件栈成熟度和迁移成本放到同一个框架里看，重点不是比纸面峰值，而是判断哪条路线更适合落地。

**关键词：** `GPU ecosystem`, `alternatives`, `software stack`, `migration cost`
## 前置
**导语：** 这一页先把硬件规格、软件栈成熟度和迁移成本放到同一个框架里，再判断哪条路线更适合落地。
- [Part 1: 1B 单卡硬件与访存优化](./1B.md)
- [Part 1: 1C 多卡通信与显存共享](./1C.md)
- [Part 1: 1E 编译优化与算力生态](./1E.md)
## 相关阅读
**导语：** 如果想继续把选型、编译和通信侧的判断补完整，可以接着看这些页。
- [Part 3: 05 Triton Autotune and Profiling](../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md)
- [Part 4: 01 CUDA Custom Kernel Intro](../04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.md)
- [Part 4: 19 Distributed Communication Primitives](../04_CUDA_and_System_Optimization/19_Distributed_Communication_Primitives.md)
## Q1：选芯片时为什么不能只看算力峰值？

<details><summary>点击展开查看解析</summary>

纸面峰值只说明“理论上能跑多快”，不说明“真实工作负载里能不能持续跑快”。

实际选型还要看：
- 软件栈是否成熟；
- 编译器和 kernel 是否可用；
- 通信和调度能力是否匹配；
- 迁移和维护成本是否可控。

所以选型不是单一性能数字的比较，而是端到端可落地性的比较。
</details>
### Q1小验证：选型时先看什么

先看能不能稳定跑，再看峰值。

```python
def score(hw, stack, migration):
    return hw * 0.4 + stack * 0.4 - migration * 0.2

print(score(9, 8, 3))
```

## Q2：为什么软件栈成熟度会直接影响硬件可用性？

<details><summary>点击展开查看解析</summary>

硬件不是孤立存在的，真正决定效率的是软件栈是否把它用起来。可用性判断的关键，不是某一层名词是否存在，而是 driver / compiler / runtime / kernel 是否能闭环。

如果编译器、驱动、通信库和 kernel 支持不完整，硬件峰值往往到不了生产场景里。相反，软件栈成熟的设备即使纸面指标没那么夸张，也可能在端到端流程里更稳。

所以“可替代方案”判断里，软件生态本身就是性能的一部分。
</details>
### Q2小验证：为什么生态会影响体验

硬件能力要经过软件栈才能落地。

```python
def stack_readiness(driver, compiler, runtime, kernel):
    # 软件栈是否成熟，不是看有没有层级名，而是看关键层是否都能闭环。
    weights = {'driver': 0.3, 'compiler': 0.3, 'runtime': 0.2, 'kernel': 0.2}
    support = {'driver': driver, 'compiler': compiler, 'runtime': runtime, 'kernel': kernel}
    score = sum(weights[name] * support[name] for name in weights)
    missing = [name for name, ok in support.items() if not ok]
    return round(score, 2), missing

cases = {
    'stack_ready': {'driver': 1, 'compiler': 1, 'runtime': 1, 'kernel': 1},
    'compiler_gaps': {'driver': 1, 'compiler': 0, 'runtime': 1, 'kernel': 1},
    'runtime_gaps': {'driver': 1, 'compiler': 1, 'runtime': 0, 'kernel': 1},
}

for name, support in cases.items():
    score, missing = stack_readiness(**support)
    print(name, '->', score, 'missing:', missing)
print('readiness score is a proxy for whether hardware can be used end-to-end')

```

## Q3：什么时候替代方案比主流 GPU 更值得考虑？

<details><summary>点击展开查看解析</summary>

当预算、供应链、功耗、部署环境或合规要求发生变化时，替代方案就不只是备选，而可能成为主方案。

但替代方案是否值得选，仍然要回到三件事：
- 能不能跑；
- 跑得稳不稳；
- 长期维护成本高不高。

所以这里的核心不是“谁最强”，而是“谁更适合当前约束”。
</details>
### Q3小验证：替代方案什么时候更合理

把硬件能力、软件栈成熟度和迁移成本放在一起，看看什么时候替代方案更值得考虑。


```python
def alternative_choice(hw_score, stack_score, migration_cost, power_budget=0.0):
    # 替代方案是否更合适，取决于综合分数而不是单项峰值。
    score = hw_score * 0.4 + stack_score * 0.4 - migration_cost * 0.15 - power_budget * 0.05
    if score >= 6.5:
        decision = 'consider_alternative'
    elif score >= 5.0:
        decision = 'context_dependent'
    else:
        decision = 'stay_with_gpu'
    return {'score': round(score, 2), 'decision': decision}

cases = [
    ('stable_gpu', 9, 9, 1, 1),
    ('alt_ready', 7, 7, 2, 2),
    ('migration_heavy', 8, 5, 5, 2),
]
for name, hw, stack, migration, power in cases:
    print(name, '->', alternative_choice(hw, stack, migration, power))
print('alternative hardware becomes attractive only when the full constraint score is strong enough')

```

## ⚠️ 常见误区

- 不要把纸面峰值当成真实吞吐。
- 软硬件栈成熟度会直接影响落地效果。
- 迁移成本往往比初始采购价更重要。
- 选型本质上是在一组约束下做最优折中。